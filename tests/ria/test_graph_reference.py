#!/usr/bin/env python3
"""Independent offline source graph contracts and native image differential.

Only synthetic pixels and host metadata execute here. Pinned publisher pure
functions, Pillow's native image engine and explicit small algebraic fixtures
provide separate oracles. These are not CUDA/model numerical qualification.
"""
from __future__ import annotations

import ast
import io
import json
import math
from pathlib import Path
import struct
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "locks/metadata/deepseek-ai/DeepSeek-V4.1-Flash/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/inference"
CONFIG = json.loads((SOURCE / "config.json").read_text())
MULTIPLIERS = (
    (76632096046245, 4839876093313, 35959672319349, 73987337458391),
    (67716810739261, 51510806800915, 30921347202721, 82619226485591),
)


def publisher_geometry():
    tree = ast.parse((SOURCE / "image_processor.py").read_text())
    names = {"llm_grid", "num_image_tokens", "solve_resize_ratio", "safe_resize", "plan_image_grid"}
    module = ast.Module([node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names], [])
    namespace = {"math": math}
    exec(compile(module, str(SOURCE / "image_processor.py"), "exec"), namespace)
    return namespace["plan_image_grid"]


def widened_bf16(array):
    bits = np.asarray(array, dtype=np.float32).view(np.uint32)
    rounded = ((bits + np.uint32(32767) + ((bits >> 16) & 1)) & np.uint32(0xffff0000))
    return rounded.view(np.float32)


def source_patches(image):
    image = image.convert("RGB")
    lh, lw, ph, pw = publisher_geometry()(image.width, image.height, SimpleNamespace(**CONFIG))
    vit_h, vit_w = ph // 14, pw // 14
    canvas = np.asarray(ImageOps.pad(image, (pw, ph), color=(127, 127, 127)), dtype=np.float32)
    normalized = widened_bf16(((canvas / np.float32(255)) - np.float32(.5)) / np.float32(.5))
    patches = normalized.transpose(2, 0, 1).reshape(3, vit_h, 14, vit_w, 14).transpose(1, 3, 0, 2, 4).copy()
    return [ph, pw, vit_h, vit_w, lh, lw, lh * (lw + 1) + 2], patches.reshape(-1)


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bf16(value):
    return float(widened_bf16(np.array([value], dtype=np.float32))[0])


class GraphReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="ria-graph-fixtures-")
        cls.binary = Path(cls.directory.name) / "contracts"
        subprocess.run([
            "gcc", "-std=c11", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Wconversion", "-Werror",
            "-ffunction-sections", "-I", str(ROOT), str(ROOT / "tests/ria/test_graph_contracts.c"),
            str(ROOT / "ria/graph.c"), str(ROOT / "ria/vision.c"), str(ROOT / "ria/common.c"), str(ROOT / "ria/expert.c"),
            "-Wl,--gc-sections", "-lcrypto", "-lpng", "-ljpeg", "-lm", "-o", str(cls.binary),
        ], check=True)
        subprocess.run([cls.binary], check=True)
        cls.report = json.loads(subprocess.check_output([cls.binary, "--report"], text=True))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def check_encoded(self, image, format_, **options):
        encoded = io.BytesIO()
        image.save(encoded, format=format_, **options)
        path = Path(self.directory.name) / "image"
        output = Path(self.directory.name) / "patches"
        path.write_bytes(encoded.getvalue())
        actual_grid = json.loads(subprocess.check_output([self.binary, "--image", path, output], text=True))
        with Image.open(io.BytesIO(encoded.getvalue())) as decoded:
            expected_grid, expected = source_patches(decoded)
        self.assertEqual(actual_grid, expected_grid)
        actual = np.fromfile(output, dtype="<f4")
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

    def test_source_dependency_aliases(self):
        expected = []
        for layer in range(CONFIG["n_layers"]):
            kv = max((s for s in CONFIG["kv_source_layers"] if s <= layer), default=0xffffffff)
            index = max((s for s in CONFIG["index_source_layers"] if s <= layer), default=0xffffffff)
            expected.append([kv, index, CONFIG["compress_ratios"][layer]])
        self.assertEqual(self.report["dependencies"], expected)
        # Source Full/Reindex/Reuse ownership is distinct from whether a new
        # compressed group completes on this position. Reuse never republishes
        # keys/selection; reindex keeps the source keys and recomputes selection.
        modes = [0 if not CONFIG["compress_ratios"][layer] else
                 1 if layer in CONFIG["kv_source_layers"] else
                 2 if layer in CONFIG["index_source_layers"] else 3
                 for layer in range(CONFIG["n_layers"])]
        self.assertEqual(self.report["modes"], modes)
        self.assertEqual([modes.count(kind) for kind in range(4)], [2, 4, 4, 30])

    def test_engram_hash_dead_boundaries(self):
        histories = ((7, 6, 5, 4), (7, -1, 5, 4), (7, 6, -1, 4), (7, 6, 5, -1), (0, 0, 0, 0), (99091, 99090, 99089, 99088))
        expected = []
        for layer, multipliers in enumerate(MULTIPLIERS):
            buckets = [16000001] * 23 + [CONFIG["engram_num_embeddings"][layer] - 23 * 16000001]
            offsets = np.cumsum([0] + buckets[:-1]).tolist()
            for history in histories:
                terms, dead = [], False
                for token, multiplier in zip(history, multipliers, strict=True):
                    dead |= token < 0
                    terms.append((2 if dead else token) * multiplier)
                cumulative = [terms[0] ^ terms[1], terms[0] ^ terms[1] ^ terms[2], terms[0] ^ terms[1] ^ terms[2] ^ terms[3]]
                expected.append([offsets[i] + cumulative[i // 8] % buckets[i] for i in range(24)])
        self.assertEqual(self.report["hashes"], expected)
        self.assertNotEqual(expected[0], expected[1])

    def test_source_geometry(self):
        shapes = ((1, 1), (800, 600), (600, 800), (16384, 1), (1, 16384), (544, 544), (1024, 1024), (1001, 777))
        expected = []
        oracle = publisher_geometry()
        for width, height in shapes:
            lh, lw, ph, pw = oracle(width, height, SimpleNamespace(**CONFIG))
            expected.append([ph, pw, ph // 14, pw // 14, lh, lw, lh * (lw + 1) + 2])
        self.assertEqual(self.report["grids"], expected)

    def test_native_png_pixels_bicubic_pad_and_modes(self):
        rng = np.random.default_rng(91)
        for width, height in ((1, 1), (43, 5), (5, 43), (17, 11), (800, 600), (123, 91)):
            image = Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8), "RGB")
            self.check_encoded(image, "PNG")
        image = Image.fromarray(rng.integers(0, 256, (23, 19, 4), dtype=np.uint8), "RGBA")
        self.check_encoded(image, "PNG")
        self.check_encoded(image.convert("P"), "PNG")
        self.check_encoded(image.convert("L"), "PNG")
        self.check_encoded(image.convert("LA"), "PNG")
        self.check_encoded(image.convert("1"), "PNG")
        gray16 = Image.fromarray(np.array([[0, 1, 127, 255, 256, 65535]], dtype=np.uint16), "I;16")
        self.check_encoded(gray16, "PNG")

    def test_native_jpeg_pixels_no_exif_rotation(self):
        rng = np.random.default_rng(113)
        image = Image.fromarray(rng.integers(0, 256, (39, 61, 3), dtype=np.uint8), "RGB")
        for mode in ("RGB", "L", "CMYK"):
            self.check_encoded(image.convert(mode), "JPEG", quality=83)
        self.check_encoded(image, "JPEG", quality=93, progressive=True)
        exif = Image.Exif()
        exif[274] = 6
        self.check_encoded(image, "JPEG", quality=90, exif=exif)

    def test_mhc_flattened_normalization_and_transpose(self):
        residual = np.array([[1., 2.], [3., 5.], [7., 11.], [13., 17.]])
        comb = np.array([[.1, .2, .3, .4], [.3, .1, .4, .2], [.2, .4, .1, .3], [.4, .3, .2, .1]])
        correct = np.einsum("ij,ik->jk", comb, residual)
        np.testing.assert_allclose(correct, [[7.6, 10.7], [7.2, 10.4], [4.8, 7.1], [4.4, 6.8]])
        self.assertFalse(np.array_equal(correct, comb @ residual))
        flat_inverse = 1 / math.sqrt(np.mean(residual**2) + CONFIG["norm_eps"])
        self.assertNotAlmostEqual(flat_inverse, 1 / math.sqrt(np.mean(residual[0]**2)))
        logits = np.array([[9., -3., 2., 1.], [-2., 7., 3., 1.], [3., 4., -1., 2.], [-2., 1., 4., 6.]])
        matrix = np.exp(logits - logits.max(axis=1, keepdims=True))
        matrix = matrix / matrix.sum(axis=1, keepdims=True) + CONFIG["hc_eps"]
        matrix /= matrix.sum(axis=0, keepdims=True) + CONFIG["hc_eps"]
        first = matrix.copy()
        for _ in range(CONFIG["hc_sinkhorn_iters"] - 1):
            matrix /= matrix.sum(axis=1, keepdims=True) + CONFIG["hc_eps"]
            matrix /= matrix.sum(axis=0, keepdims=True) + CONFIG["hc_eps"]
        self.assertGreater(np.max(np.abs(matrix - first)), .01)
        self.assertLess(np.max(np.abs(matrix.sum(axis=0) - 1)), 2e-6)

    def test_candidate_partial_block_and_exact_pool_continuation(self):
        scores = np.arange(81, dtype=float)[::-1]
        blocks = [max(scores[i:i + 8]) for i in range(0, len(scores), 8)]
        blocks[-1] = math.inf
        chosen = sorted(range(len(blocks)), key=lambda i: (-blocks[i], i))[:3]
        self.assertIn(10, chosen)
        self.assertEqual(sorted(chosen), [0, 1, 10])
        values = np.array([[1., -3.], [4., 5.], [9., 2.], [7., 8.]], dtype=np.float32)
        gates = np.array([[2., -1.], [-3., 4.], [1., 0.], [2., -5.]], dtype=np.float32)
        full = []
        for first in (0, 2):
            probability = np.exp(gates[first:first + 2] - gates[first:first + 2].max(axis=0))
            probability /= probability.sum(axis=0)
            full.append((values[first:first + 2] * probability).sum(axis=0))
        # Chunk boundaries after token 0 and token 2 preserve FP32 partial
        # group values and gate scores, not only emitted compressed rows.
        cache = [None, None]
        continued = []
        for i in range(4):
            cache[i % 2] = (values[i].copy(), gates[i].copy())
            if i % 2:
                v, g = map(np.stack, zip(*cache, strict=True))
                p = np.exp(g - g.max(axis=0))
                p /= p.sum(axis=0)
                continued.append((v * p).sum(axis=0))
        np.testing.assert_array_equal(full, continued)

    def test_sink_and_bf16_probability_boundary(self):
        logits = [0., math.log(3)]
        probabilities = [math.exp(x - max(logits)) for x in logits]
        denominator = sum(probabilities) + math.exp(math.log(2) - max(logits))
        value = bf16(sum(bf16(p) * v for p, v in zip(probabilities, [2., 5.], strict=True)) / denominator)
        self.assertEqual(value, bf16((bf16(1 / 3) * 2 + 5) / 2))
        self.assertNotEqual(value, bf16(sum(p * v for p, v in zip(probabilities, [2., 5.], strict=True)) / sum(probabilities)))

    def test_vision_unfold_channel_order_and_2d_rotation(self):
        grid = np.arange(4 * 5 * 2).reshape(4, 5, 2)
        padded = np.pad(grid.transpose(2, 0, 1), ((0, 0), (0, 2), (0, 1)))
        first = padded[:, :3, :3].reshape(-1)
        self.assertEqual(first.tolist(), [0, 2, 4, 10, 12, 14, 20, 22, 24, 1, 3, 5, 11, 13, 15, 21, 23, 25])
        last = padded[:, 3:6, 3:6].reshape(-1)
        self.assertEqual(np.count_nonzero(last), 4)
        x = np.arange(1., 65.)
        frequencies = 10000 ** (-np.arange(0., 32., 2) / 32)
        angles = np.concatenate((2 * frequencies, 3 * frequencies))
        correct = np.concatenate((x[:32] * np.cos(angles) - x[32:] * np.sin(angles), x[32:] * np.cos(angles) + x[:32] * np.sin(angles)))
        self.assertAlmostEqual(np.dot(correct, correct), np.dot(x, x), places=8)
        self.assertNotAlmostEqual(correct[0], x[0] * math.cos(2) - x[1] * math.sin(2))


if __name__ == "__main__":
    unittest.main()
