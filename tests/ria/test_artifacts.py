"""Offline synthetic boundary fixtures; no target checkpoint/model qualification."""

import hashlib
import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from ria.identity import (ArtifactError, canonical, checked_product, digest, loads,
                          seal, u64, verify_identity, within)
from ria.numeric import (bf16_decode, bf16_encode, decode_matrix_rows, e2m1_decode,
                         e4m3_decode, e4m3_encode, encode_fp8_block32)
from ria.preparation import prepare, verify_metadata_graph, verify_package
from ria.safetensors import authenticated_range, chunk_index, inspect
from ria.schemas import validate


def source_file(path, tensors):
    header, payload = {}, bytearray()
    for name, (dtype, shape, data) in tensors.items():
        start = len(payload)
        payload.extend(data)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, len(payload)]}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    return path


def fixture_recipe(root, profile="bf16", *, convert=False):
    name = "layers.0.ffn.experts.0.w1.weight"
    path = source_file(root / "input.safetensors", {name: ("BF16", [2, 4], b"".join(struct.pack("<H", bf16_encode(v)) for v in range(8)))})
    operator = seal({"schema_revision": 1, "profile": profile, "source_revision": "a" * 40,
        "graph": "deepseek_v41_flash", "weight_format": "bf16" if profile == "bf16" else "fp8_block32",
        "activation_group": 0 if profile == "bf16" else 32,
        "weight_scale_block": [] if profile == "bf16" else [32, 32],
        "activation_quantizer": "bf16_rne" if profile == "bf16" else "fp8_e4m3fn_ue8m0_32",
        "clamp_f32_bits": "41200000", "gate_clamp": "upper_only", "up_clamp": "two_sided",
        "coefficient_position": "before_down_quantizer", "accumulator": "fp32",
        "reduction_order": "increasing_expert_id_then_shared", "scale_reduction_domain": "full_original_population",
        "calibration_digest": None, "rounding": "ties_to_even", "nibble_order": "low_first"})
    (root / "operator.json").write_bytes(canonical(operator))
    token_identity = hashlib.sha256(b"synthetic fixture only").hexdigest()
    return seal({"schema_revision": 1, "model_id": "deepseek-ai/DeepSeek-V4.1-Flash", "source_revision": "a" * 40,
        "profile": profile, "logical_model_digest": token_identity, "tokenizer_digest": token_identity,
        "encoding_digest": token_identity, "operator_contract": "operator.json",
        "sources": [{"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}],
        "tensors": [{"name": name, "operation": "expert_gate", "placement": "both", "format": "bf16",
                     "logical_shape": [2, 4], "scale_names": [], "convert": convert}],
        "metadata": [], "feature_exclusions": ["synthetic offline fixture; not a target model artifact"],
        "max_shard_bytes": 1 << 20, "chunk_size": 64, "scratch_bytes": 1 << 20})


def test_jcs_rfc_numbers_and_unicode():
    assert canonical([333333333.33333329, 1e30, 4.5, 2e-3, 1e-27, -0.0]) == b"[333333333.3333333,1e+30,4.5,0.002,1e-27,0]"
    value = {"\u20ac": "Euro", "\r": "Carriage Return", "\ufb33": "Hebrew", "1": "One", "\U0001f600": "Emoji", "\u0080": "Control", "\u00f6": "Latin"}
    encoded = canonical(value).decode()
    assert [key for key in json.loads(encoded)] == ["\r", "1", "\u0080", "\u00f6", "\u20ac", "\U0001f600", "\ufb33"]
    assert canonical("\u0000\b\t\n\f\r\\\"") == b'"\\u0000\\b\\t\\n\\f\\r\\\\\\\""'


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":"\\ud800"}', b'{"a":9007199254740992}', b'{"a":1e400}'])
def test_invalid_json(raw):
    with pytest.raises(ArtifactError):
        loads(raw)


def test_depth_integer_identity_and_digest_exclusion():
    with pytest.raises(ArtifactError):
        loads(b"[" * 65 + b"0" + b"]" * 65)
    assert u64("18446744073709551615") == (1 << 64) - 1
    for value in ("01", "-1", "18446744073709551616", 1):
        with pytest.raises(ArtifactError):
            u64(value)
    value = seal({"child": {"digest": "covered"}, "id": "18446744073709551615"})
    assert digest({**value, "signatures": ["excluded"]}) == value["digest"]
    assert verify_identity(value) == value["digest"]
    value["child"]["digest"] = "different"
    with pytest.raises(ArtifactError):
        verify_identity(value)
    with pytest.raises(ArtifactError):
        checked_product([1 << 40, 1 << 40])


def test_safetensors_relative_offsets_and_corruption(tmp_path):
    path = source_file(tmp_path / "x.safetensors", {"a": ("U8", [3], b"abc"), "b": ("BF16", [1], b"\x80\x3f")})
    shard = inspect(path)
    assert shard.tensors["a"].offset == 0 and shard.tensors["b"].offset == 3
    for raw in (b"", struct.pack("<Q", 1 << 63), path.read_bytes()[:-1], struct.pack("<Q", 29) + b'{"x":1,"x":2}' + b" " * 16):
        path.write_bytes(raw)
        with pytest.raises(ArtifactError):
            inspect(path)


def test_malicious_shape_overlap_and_dtype(tmp_path):
    for header in ({"a": {"dtype": "U8", "shape": [1 << 63, 8], "data_offsets": [0, 8]}},
                   {"a": {"dtype": "U8", "shape": [3], "data_offsets": [0, 3]}, "b": {"dtype": "U8", "shape": [3], "data_offsets": [2, 5]}},
                   {"a": {"dtype": "PICKLE", "shape": [5], "data_offsets": [0, 5]}}):
        raw = json.dumps(header).encode()
        path = tmp_path / "evil"
        path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"12345")
        with pytest.raises(ArtifactError):
            inspect(path)


def test_chunk_boundary_permissions_and_tampering(tmp_path):
    path = tmp_path / "chunks"
    path.write_bytes(b"0123456789ABCDEFGHIJ")
    descriptor = chunk_index(path, 8)
    assert authenticated_range(path, descriptor, 3, 7, [(0, 16)]) == b"3456789"
    assert authenticated_range(path, descriptor, 18, 2, [(16, 20)]) == b"IJ"
    with pytest.raises(ArtifactError, match="complete verification"):
        authenticated_range(path, descriptor, 3, 7, [(3, 10)])
    path.write_bytes(b"X123456789ABCDEFGHIJ")
    with pytest.raises(ArtifactError, match="hash mismatch"):
        authenticated_range(path, descriptor, 3, 2, [(0, 8)])


def test_paths_metadata_cycles(tmp_path):
    (tmp_path / "link").symlink_to(tmp_path)
    for path in ("../escape", "/absolute", "link/target"):
        with pytest.raises(ArtifactError):
            within(tmp_path, path)
    # Cycle is rejected before attempting to treat a digest as trust for itself.
    fake = seal({"metadata": []})
    (tmp_path / "a.json").write_bytes(canonical(fake))
    reference = {"path": "a.json", "digest": fake["digest"], "kind": "metadata"}
    verify_metadata_graph(tmp_path, [reference])
    with pytest.raises(ArtifactError, match="bound"):
        verify_metadata_graph(tmp_path, [reference], max_bytes=1)


def test_preparation_atomic_resume_unknown_and_pages(tmp_path):
    recipe = fixture_recipe(tmp_path)
    output = tmp_path / "prepared"
    manifest = prepare(tmp_path, recipe, output)
    assert manifest["tensor_pages"] and not manifest["tensors"]
    assert prepare(tmp_path, recipe, output)["digest"] == manifest["digest"]
    assert verify_package(output)["digest"] == manifest["digest"]
    bad = seal({**recipe, "tensors": []})
    with pytest.raises(ArtifactError):
        prepare(tmp_path, bad, tmp_path / "bad")
    assert not (tmp_path / "bad" / "manifest.json").exists()


def test_failed_shard_publication_never_commits(tmp_path, monkeypatch):
    from ria import preparation
    recipe = fixture_recipe(tmp_path)
    output = tmp_path / "partial"
    real = preparation._write_bundle
    def fail(*args, **kwargs):
        real(*args, **kwargs)
        raise OSError("injected after durable data")
    monkeypatch.setattr(preparation, "_write_bundle", fail)
    with pytest.raises(OSError):
        prepare(tmp_path, recipe, output)
    assert not (output / "manifest.json").exists()
    monkeypatch.setattr(preparation, "_write_bundle", real)
    assert prepare(tmp_path, recipe, output)["digest"]


def test_fp8_conversion_and_independent_decodes(tmp_path):
    for code in range(256):
        if code & 127 != 127:
            assert e4m3_encode(e4m3_decode(code)) == code
    assert [e2m1_decode(code) for code in range(8)] == [0, .5, 1, 1.5, 2, 3, 4, 6]
    assert bf16_decode(bf16_encode(1.00390625)) == 1.0
    assert bf16_decode(bf16_encode(1.01171875)) == 1.015625
    values, scales = encode_fp8_block32([[float(v) for v in range(35)] for _ in range(3)])
    decoded = decode_matrix_rows(values, scales, 3, 35, "fp8_block32", 1)
    assert len(scales) == 2 and decoded[0][0] == 0 and decoded[0][-1] == 32
    recipe = fixture_recipe(tmp_path, "fp8", convert=True)
    manifest = prepare(tmp_path, recipe, tmp_path / "fp8")
    assert manifest["profile"] == "fp8"
    verify_package(tmp_path / "fp8")


def test_strict_unknown_fields_and_cpu_client():
    with pytest.raises(ArtifactError):
        validate("planning-request", {"schema_revision": 1, "surprise": True})


def test_compact_client_extraction_authentication_and_cache_completeness(tmp_path, monkeypatch):
    from ria.client import client_package
    from ria import client
    recipe = fixture_recipe(tmp_path)
    names = ["layers.0.ffn.experts.0." + part + ".weight" for part in ("w1", "w2", "w3")]
    values = {name: ("BF16", [2, 4], b"\x80\x3f" * 8) for name in names}
    values["embed.weight"] = ("BF16", [2, 4], b"\x00\x40" * 8)
    values["layers.1.engram.embed.weight"] = ("BF16", [2, 4], b"\x00\x41" * 8)
    source = source_file(tmp_path / "input.safetensors", values)
    rules = []
    for name in values:
        operation = {names[0]: "expert_gate", names[1]: "expert_down", names[2]: "expert_up",
                     "embed.weight": "embedding", "layers.1.engram.embed.weight": "engram"}[name]
        placement = "cache" if name in names else ("both" if name == "embed.weight" else "server")
        rules.append({"name": name, "operation": operation, "placement": placement, "format": "bf16",
            "logical_shape": [2, 4], "scale_names": [], "convert": False, "grant_group": placement})
    recipe = seal({**recipe, "tensors": rules, "sources": [{"path": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}]})
    server = tmp_path / "server"
    manifest = prepare(tmp_path, recipe, server)
    page = json.loads((server / manifest["tensor_pages"][0]["path"]).read_text())
    assert len(page["shards"]) == 1  # One coarse mmap contains multiple populations.
    real_range, reads = client.authenticated_range, []
    def tracked(path, descriptor, start, length, grants):
        reads.append((start, length))
        return real_range(path, descriptor, start, length, grants)
    monkeypatch.setattr(client, "authenticated_range", tracked)
    result = client_package(server, manifest["digest"], tmp_path / "client", chunk_size=64)
    assert result["role"] == "client" and result["logical_model_digest"] == manifest["logical_model_digest"]
    output_page = json.loads((tmp_path / "client" / result["tensor_pages"][0]["path"]).read_text())
    assert [item["name"] for item in output_page["tensors"]] == ["embed.weight"]
    assert reads and sum(length for _, length in reads) == 16
    assert client_package(server, manifest["digest"], tmp_path / "client", chunk_size=64)["digest"] == result["digest"]
    with pytest.raises(ArtifactError, match="gate/up/down"):
        client_package(server, manifest["digest"], tmp_path / "incomplete", selected_names=names[:1], chunk_size=64)
    cached = client_package(server, manifest["digest"], tmp_path / "cached", selected_names=names, chunk_size=64)
    cached_page = json.loads((tmp_path / "cached" / cached["tensor_pages"][0]["path"]).read_text())
    assert {item["name"] for item in cached_page["tensors"] if item["operation"] != "inactive"} == set(names) | {"embed.weight"}
    with pytest.raises(ArtifactError, match="identity"):
        client_package(server, "0" * 64, tmp_path / "untrusted", chunk_size=64)
    shard = server / page["shards"][0]["path"]
    owned = next(item for item in page["tensors"] if item["name"] == "embed.weight")
    with shard.open("r+b") as stream:
        stream.seek(int(page["shards"][0]["data_start"]) + int(owned["offset"]))
        stream.write(b"X")
    with pytest.raises(ArtifactError, match="chunk hash"):
        client_package(server, manifest["digest"], tmp_path / "tampered", chunk_size=64)
    assert not (tmp_path / "tampered" / "manifest.json").exists()


def test_engram_repacking_preserves_exact_row_codes(tmp_path):
    recipe = fixture_recipe(tmp_path)
    name = "layers.1.engram.embed.weight"
    scale = "layers.1.engram.embed.scale"
    values = bytes(range(256)) * 2
    scales = bytes(range(16))
    source = source_file(tmp_path / "input.safetensors", {name: ("F8_E4M3", [2, 256], values), scale: ("U8", [2, 8], scales)})
    recipe = seal({**recipe, "tensors": [
        {"name": name, "operation": "engram", "placement": "server", "format": "fp8_block32",
         "logical_shape": [2, 256], "scale_names": [scale], "convert": False, "repack": "engram_packed"},
        {"name": scale, "operation": "inactive", "placement": "inactive", "format": "plain",
         "logical_shape": [2, 8], "scale_names": [], "convert": False, "inactive_reason": "lossless packed row scales"}],
        "sources": [{"path": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}]})
    output = tmp_path / "packed"
    manifest = prepare(tmp_path, recipe, output)
    page = json.loads((output / manifest["tensor_pages"][0]["path"]).read_text())
    item = page["tensors"][0]
    assert item["format"] == "engram_packed" and item["physical_shape"] == [2, 264]
    shard = inspect(output / page["shards"][0]["path"])
    from ria.safetensors import tensor_blocks
    packed = b"".join(tensor_blocks(shard, shard.tensors[name]))
    assert packed == values[:256] + scales[:8] + values[256:] + scales[8:]
    assert item["source_scale_sha256"] == hashlib.sha256(scales).hexdigest()


def test_recipe_unknown_graph_and_estimate_metadata(tmp_path):
    from ria.preparation import estimate, target_operation
    with pytest.raises(ArtifactError, match="unrecognized"):
        target_operation("layers.0.fake_required.weight")
    result = estimate(tmp_path, fixture_recipe(tmp_path))
    assert result["retained_metadata_bytes"] > 0
    assert result["working_memory_upper_bytes"] > result["calibration_scratch_bytes"] + (16 << 20)
    malformed = seal({**fixture_recipe(tmp_path), "chunk_size": 64.0})
    with pytest.raises(ArtifactError, match="recipe"):
        prepare(tmp_path, malformed, tmp_path / "noninteger")


def test_actual_pinned_inventory_classifier_and_engram_constants():
    from ria.preparation import target_operation
    root = Path(__file__).resolve().parents[2]
    index = json.loads((root / "locks/metadata/nvidia/DeepSeek-V4.1-Flash-NVFP4/3431dde3247c13b5957f682b1e3c6fcae2566079/model.safetensors.index.json").read_text())
    assert len(index["weight_map"]) == 188245
    operations = [target_operation(name) for name in index["weight_map"]]
    assert operations.count("expert_gate") == operations.count("expert_up") == operations.count("expert_down") == 40 * 384
    derived = json.loads((root / "locks/derived/engram-metadata.json").read_text())
    verify_identity(derived)
    assert derived["compressed_vocab"] == 99092 and derived["pad_compressed_id"] == 2
    assert [sum(row) for row in derived["primes"]] == [384006168, 384016682]
