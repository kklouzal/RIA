#!/usr/bin/env python3
"""Independent offline expert oracle: rational value palettes and explicit casts.

Synthetic fixtures do not certify the NVIDIA checkpoint or physical CUDA
executor. Run directly; the native shared object is compiled into an owned
temporary directory. No model access or CUDA device initialization occurs.
"""
from __future__ import annotations

import ctypes
from fractions import Fraction
import math
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
F4 = [Fraction(x) for x in (0, .5, 1, 1.5, 2, 3, 4, 6)]
F8 = [Fraction(i, 512) for i in range(8)] + [
    Fraction(8 + mantissa, 8) * Fraction(2) ** exponent
    for exponent in range(-6, 9)
    for mantissa in range(8)
][:-1]


def f32(value: float | Fraction) -> float:
    try:
        return struct.unpack("<f", struct.pack("<f", float(value)))[0]
    except OverflowError:
        return math.copysign(math.inf, float(value))


def bf16(value: float) -> float:
    """Pick the nearest of adjacent BF16 representable values independently."""
    value = f32(value)
    if not math.isfinite(value):
        return value
    raw = struct.unpack("<I", struct.pack("<f", abs(value)))[0]
    lower = raw >> 16
    options = (lower, lower + 1)
    winner = min(options, key=lambda code: (
        abs(struct.unpack("<f", struct.pack("<I", code << 16))[0] - abs(value)),
        code % 2,
    ))
    result = struct.unpack("<f", struct.pack("<I", winner << 16))[0]
    return math.copysign(result, value)


def encode(value: float, palette: list[Fraction], sign_bit: int) -> int:
    absolute = Fraction(abs(value))
    code = min(range(len(palette)), key=lambda i: (abs(absolute - palette[i]), i % 2))
    return code | (sign_bit if math.copysign(1, value) < 0 else 0)


def decode(code: int, palette: list[Fraction], sign_bit: int) -> float:
    return math.copysign(float(palette[code & (sign_bit - 1)]), -1 if code & sign_bit else 1)


def quantize(values: list[float], profile: int, global_scale: float) -> tuple[list[float], list[float]]:
    logical = [bf16(value) for value in values]
    if profile == 1:
        return logical, []
    group = 32 if profile == 2 else 16
    output, factors = [], []
    for base in range(0, len(values), group):
        block = logical[base:base + group]
        maximum = max(abs(value) for value in block)
        if profile == 2:
            target = f32(max(maximum, f32(1e-4)) * f32(1 / 448))
            exponent = next(e for e in range(-127, 128) if Fraction(2) ** e >= Fraction(target))
            factor = float(Fraction(2) ** exponent)
        else:
            code = encode(f32(f32(maximum / 6) / global_scale), F8, 128)
            factor = f32(decode(code, F8, 128) * global_scale)
        factors.append(factor if profile == 2 else decode(code, F8, 128))
        palette, sign = (F8, 128) if profile == 2 else (F4, 8)
        output.extend(decode(encode(f32(value / factor) if factor else 0, palette, sign), palette, sign) for value in block)
    return output, factors


class Error(ctypes.Structure):
    _fields_ = [("code", ctypes.c_int), ("message", ctypes.c_char * 256)]


class Matrix(ctypes.Structure):
    _fields_ = [
        ("profile", ctypes.c_int), ("out_features", ctypes.c_uint64), ("in_features", ctypes.c_uint64),
        ("values", ctypes.POINTER(ctypes.c_uint8)), ("values_bytes", ctypes.c_uint64), ("value_row_stride", ctypes.c_uint64),
        ("scales", ctypes.POINTER(ctypes.c_uint8)), ("scales_bytes", ctypes.c_uint64), ("scale_row_stride", ctypes.c_uint64),
        ("weight_global_scale", ctypes.c_float), ("activation_global_scale", ctypes.c_float),
    ]


class Expert(ctypes.Structure):
    _fields_ = [("gate", Matrix), ("up", Matrix), ("down", Matrix), ("clamp", ctypes.c_float)]


class Prepared:
    def __init__(self, profile: int, n: int, k: int, rng: random.Random, factor: float, activation: float):
        self.profile, self.n, self.k, self.factor, self.activation = profile, n, k, f32(factor), f32(activation)
        self.weights = [[rng.choice((-2, -1.5, -.5, 0, .5, 1, 1.5, 2)) for _ in range(k)] for _ in range(n)]
        self.scale_values = [[1 + (row + block) % 2 for block in range((k + (31 if profile == 2 else 15)) // (32 if profile == 2 else 16))]
                             for row in range((n + 31) // 32 if profile == 2 else n)]
        packed = bytearray()
        for row in self.weights:
            if profile == 1:
                packed.extend(b"".join(struct.pack("<f", value)[2:] for value in row))
            elif profile == 2:
                packed.extend(encode(value, F8, 128) for value in row)
            else:
                codes = [encode(value, F4, 8) for value in row]
                packed.extend(codes[i] | ((codes[i + 1] if i + 1 < k else 0) << 4) for i in range(0, k, 2))
        scale_bytes = bytes(value + 127 if profile == 2 else encode(value, F8, 128)
                            for row in self.scale_values for value in row) if profile != 1 else b""
        if profile == 2:
            self.scale_values = [[2.0 ** value for value in row] for row in self.scale_values]
        self.value_buffer = (ctypes.c_uint8 * len(packed)).from_buffer_copy(packed)
        self.scale_buffer = (ctypes.c_uint8 * len(scale_bytes)).from_buffer_copy(scale_bytes) if scale_bytes else None
        self.matrix = Matrix(profile, n, k, self.value_buffer, len(packed), len(packed) // n,
                             self.scale_buffer, len(scale_bytes), len(self.scale_values[0]) if scale_bytes else 0,
                             self.factor, self.activation)

    def project(self, values: list[float], round_output: bool = True, quantize_inputs: bool = True) -> list[float]:
        q, activation_factors = quantize(values, self.profile, self.activation)
        group = self.k if self.profile == 1 else 32 if self.profile == 2 else 16
        if not quantize_inputs:
            q = [bf16(value) for value in values]
            activation_factors = [1.0] * ((self.k + group - 1) // group)
        output = []
        for row in range(self.n):
            total = 0.0
            for base in range(0, self.k, group):
                partial = 0.0
                for k in range(base, min(base + group, self.k)):
                    # Fixture operands have bounded exponents; their exact
                    # rational product+FP32 accumulator fits binary64 before
                    # this independently specified single FP32 rounding.
                    partial = f32(Fraction(q[k]) * Fraction(self.weights[row][k]) + Fraction(partial))
                if self.profile != 1:
                    scale_row = row // 32 if self.profile == 2 else row
                    partial = f32(f32(partial * activation_factors[base // group]) * self.scale_values[scale_row][base // group])
                total = f32(total + partial)
            if self.profile == 3:
                total = f32(total * (f32(self.factor * self.activation) if quantize_inputs else self.factor))
            output.append(bf16(total) if round_output else total)
        return output


class OracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="ria-expert-oracle-")
        library = Path(cls.temporary.name) / "expert.so"
        fragments = Path(cls.temporary.name) / "fragments.c"
        fragments.write_text(
            '#include "numeric.h"\n'
            'unsigned fixture_a_row(unsigned l,unsigned r) { return ria_num_mma_a_row(l,r); }\n'
            'unsigned fixture_a_col(unsigned l,unsigned r,unsigned e,unsigned b) { return ria_num_mma_a_col(l,r,e,b); }\n'
            'unsigned fixture_b_row(unsigned l,unsigned r,unsigned e,unsigned b) { return ria_num_mma_b_row(l,r,e,b); }\n'
            'unsigned fixture_c_row(unsigned l,unsigned e) { return ria_num_mma_c_row(l,e); }\n'
            'unsigned fixture_c_col(unsigned l,unsigned e) { return ria_num_mma_c_col(l,e); }\n'
            'uint32_t fixture_pack_a(const uint8_t *p,uint64_t w,uint64_t r,uint64_t rb,uint64_t k,unsigned l,unsigned g,unsigned b,unsigned valid) { return ria_num_mma_activation_fragment(p,w,r,rb,k,l,g,b,valid); }\n'
            'uint32_t fixture_pack_b(const uint8_t *p,uint64_t w,uint64_t n,uint64_t nb,uint64_t k,uint64_t s,unsigned l,unsigned g,unsigned b,unsigned valid) { return ria_num_mma_weight_fragment(p,w,n,nb,k,s,l,g,b,valid); }\n'
            'uint32_t fixture_scale_a(const uint8_t *p,uint64_t g,uint64_t r,uint64_t rb,uint64_t k,unsigned l) { return ria_num_mma_nvfp4_scale_a(p,g,r,rb,k,l); }\n'
            'uint32_t fixture_scale_b(const uint8_t *p,uint64_t s,uint64_t n,uint64_t nb,uint64_t k,unsigned l) { return ria_num_mma_nvfp4_scale_b(p,s,n,nb,k,l); }\n'
        )
        subprocess.run(["cc", "-std=c11", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror", "-fPIC", "-shared",
                        "-I", str(ROOT / "ria"), str(ROOT / "ria/expert.c"), str(fragments), "-lm", "-o", str(library)], check=True)
        cls.native = ctypes.CDLL(str(library))
        for name, count in (("a_row", 2), ("a_col", 4), ("b_row", 4), ("c_row", 2), ("c_col", 2)):
            function = getattr(cls.native, "fixture_" + name)
            function.argtypes = [ctypes.c_uint] * count
            function.restype = ctypes.c_uint
        byte_pointer = ctypes.POINTER(ctypes.c_uint8)
        for name, large, small in (("pack_a", 4, 4), ("pack_b", 5, 4), ("scale_a", 4, 1), ("scale_b", 4, 1)):
            function = getattr(cls.native, "fixture_" + name)
            function.argtypes = [byte_pointer] + [ctypes.c_uint64] * large + [ctypes.c_uint] * small
            function.restype = ctypes.c_uint32
        cls.native.ria_expert_cpu_create.argtypes = [ctypes.c_uint64] * 3 + [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(Error)]
        cls.native.ria_expert_cpu_create.restype = ctypes.c_bool
        cls.native.ria_expert_cpu_destroy.argtypes = [ctypes.c_void_p]
        cls.native.ria_expert_cpu_evaluate.argtypes = [ctypes.c_void_p, ctypes.POINTER(Expert), ctypes.POINTER(ctypes.c_float), ctypes.c_uint64,
                                                      ctypes.c_uint64, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                                      ctypes.c_uint64, ctypes.POINTER(Error)]
        cls.native.ria_expert_cpu_evaluate.restype = ctypes.c_bool
        cls.native.ria_expert_e4m3_decode.argtypes = [ctypes.c_uint8]
        cls.native.ria_expert_e4m3_decode.restype = ctypes.c_float
        cls.native.ria_expert_e4m3_encode.argtypes = [ctypes.c_float]
        cls.native.ria_expert_e4m3_encode.restype = ctypes.c_uint8

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_all_fp8_codes_and_midpoints(self):
        self.assertEqual(len(F8), 127)
        for code in range(256):
            if code & 127 == 127:
                self.assertTrue(math.isnan(self.native.ria_expert_e4m3_decode(code)))
            else:
                expected = decode(code, F8, 128)
                self.assertEqual(struct.pack("<f", expected), struct.pack("<f", self.native.ria_expert_e4m3_decode(code)))
        for left in range(126):
            value = float((F8[left] + F8[left + 1]) / 2)
            for sign in (1, -1):
                self.assertEqual(encode(sign * value, F8, 128), self.native.ria_expert_e4m3_encode(sign * value))

    def test_native_mma_fragment_layout_ptx_oracle(self):
        # Separate coordinate formulas from NVIDIA PTX ISA figures 82/84,
        # 91/93/94,96/97/98. Visit every operand/accumulator coordinate once.
        for bits, width in ((16, 16), (8, 32), (4, 64)):
            packed = 32 // bits
            actual_a, actual_b, actual_c = set(), set(), set()
            for lane in range(32):
                quad, thread = divmod(lane, 4)
                for index in range(4 * packed):
                    expected_row = quad if index < packed or 2 * packed <= index < 3 * packed else quad + 8
                    expected_col = thread * packed + index % packed + (width // 2 if index >= 2 * packed else 0)
                    reg, element = divmod(index, packed)
                    actual = self.native.fixture_a_row(lane, reg), self.native.fixture_a_col(lane, reg, element, bits)
                    self.assertEqual(actual, (expected_row, expected_col))
                    self.assertNotIn(actual, actual_a)
                    actual_a.add(actual)
                for index in range(2 * packed):
                    reg, element = divmod(index, packed)
                    expected = thread * packed + index % packed + (width // 2 if index >= packed else 0), quad
                    actual = self.native.fixture_b_row(lane, reg, element, bits), quad
                    self.assertEqual(actual, expected)
                    self.assertNotIn(actual, actual_b)
                    actual_b.add(actual)
                for element in range(4):
                    actual = self.native.fixture_c_row(lane, element), self.native.fixture_c_col(lane, element)
                    self.assertEqual(actual, (quad + (8 if element >= 2 else 0), thread * 2 + element % 2))
                    self.assertNotIn(actual, actual_c)
                    actual_c.add(actual)
            self.assertEqual(actual_a, {(row, col) for row in range(16) for col in range(width)})
            self.assertEqual(actual_b, {(row, col) for row in range(width) for col in range(8)})
            self.assertEqual(actual_c, {(row, col) for row in range(16) for col in range(8)})

    def test_native_nvfp4_scaled_k16_partial_exactness(self):
        # Any raw K16 dot is an integer multiple of 1/4 in [-576,576].
        # Maximum significand sizes: 12 bits raw, four bits each scale,
        # hence <=20 bits before group accumulation. Cover all scale codes
        # and raw extrema/rounding boundaries independently with rationals.
        for raw in (Fraction(1, 4), Fraction(3, 4), Fraction(5, 4), Fraction(63, 4), Fraction(1023, 4), Fraction(2303, 4), Fraction(576)):
            for a in F8:
                for b in F8:
                    exact = raw * a * b
                    self.assertEqual(Fraction(f32(exact)), exact)
                    self.assertEqual(f32(f32(float(raw * a)) * float(b)), float(exact))

    def test_native_ragged_fragment_and_e4m3_scale_placement(self):
        for bits in (4, 8, 16):
            group = 32 if bits == 8 else 16
            for rows, count, width in ((1, 1, 3), (3, 7, 16), (17, 9, 17), (19, 33, 32), (2, 17, 33), (18, 19, 65)):
                packed, mask = 32 // bits, (1 << bits) - 1
                a = [[(r * 7 + k * 11) & mask for k in range(width)] for r in range(rows)]
                b = [[(n * 13 + k * 3) & mask for k in range(width)] for n in range(count)]
                act = b"".join(code.to_bytes(2 if bits == 16 else 1, "little") for row in a for code in row)
                stride = (width * bits + 7) // 8 + 5
                weights = bytearray([0xa5] * (count * stride))
                for n in range(count):
                    for k, code in enumerate(b[n]):
                        if bits == 4:
                            i = n * stride + k // 2
                            shift = (k % 2) * 4
                            weights[i] = (weights[i] & ~(15 << shift)) | code << shift
                        else:
                            size = bits // 8
                            weights[n * stride + k * size:n * stride + (k + 1) * size] = code.to_bytes(size, "little")
                native_a = (ctypes.c_uint8 * len(act)).from_buffer_copy(act)
                native_b = (ctypes.c_uint8 * len(weights)).from_buffer_copy(weights)
                groups = (width + group - 1) // group
                ascale = [[(r * 11 + k * 7) % 127 for k in range(groups)] for r in range(rows)]
                bscale = [[(n * 17 + k * 3) % 127 for k in range(groups)] for n in range(count)]
                native_as = (ctypes.c_uint8 * (rows * groups))(*(value for row in ascale for value in row))
                native_bs = (ctypes.c_uint8 * (count * groups))(*(value for row in bscale for value in row))
                for rb in range(0, rows, 16):
                    for nb in range(0, count, 8):
                        for base in range(0, width, group):
                            for lane in range(32):
                                quad, thread = divmod(lane, 4)
                                for reg in range(4):
                                    expected = 0
                                    r = rb + quad + (8 if reg in (1, 3) else 0)
                                    for element in range(packed):
                                        col = thread * packed + element + (128 // bits if reg >= 2 else 0)
                                        if r < rows and col < group and base + col < width:
                                            expected |= a[r][base + col] << (element * bits)
                                    actual = self.native.fixture_pack_a(native_a, width, rows, rb, base, lane, reg, bits, group)
                                    self.assertEqual(actual, expected)
                                for reg in range(2):
                                    expected = 0
                                    n = nb + quad
                                    for element in range(packed):
                                        col = thread * packed + element + reg * (4 * packed)
                                        if n < count and col < group and base + col < width:
                                            expected |= b[n][base + col] << (element * bits)
                                    actual = self.native.fixture_pack_b(native_b, width, count, nb, base, stride, lane, reg, bits, group)
                                    self.assertEqual(actual, expected)
                                if bits == 4:
                                    r = rb + quad + (8 if thread == 1 else 0)
                                    expected_a = ascale[r][base // group] if thread in (0, 1) and r < rows else 0
                                    expected_b = bscale[nb + quad][base // group] if thread == 0 and nb + quad < count else 0
                                    self.assertEqual(self.native.fixture_scale_a(native_as, groups, rows, rb, base // group, lane), expected_a)
                                    self.assertEqual(self.native.fixture_scale_b(native_bs, groups, count, nb, base // group, lane), expected_b)

    def test_complete_profiles_and_split_rows(self):
        rng = random.Random(40103)
        for profile in (1, 2, 3):
            gate = Prepared(profile, 33, 65, rng, .5, .125)
            up = Prepared(profile, 33, 65, rng, 2, .25)
            down = Prepared(profile, 5, 33, rng, .25, .015625)
            expert = Expert(gate.matrix, up.matrix, down.matrix, 10)
            rows = [[f32(rng.uniform(-1, 1)) for _ in range(65)] for _ in range(3)]
            coefficients = [f32(.35), 0.0, f32(1.5)]
            expected = []
            wrong_order, weight_only = [], []
            for row, coefficient in zip(rows, coefficients, strict=True):
                g, u = gate.project(row), up.project(row)
                hidden = []
                for gv, uv in zip(g, u, strict=True):
                    gv, uv = min(gv, 10), max(-10, min(10, uv))
                    silu = f32(gv / f32(1 + f32(math.exp(-gv))))
                    hidden.append(bf16(f32(f32(silu * uv) * coefficient)))
                expected.extend(down.project(hidden))
                wrong_hidden = [bf16(f32(f32(gv / f32(1 + f32(math.exp(-gv)))) * uv))
                                for gv, uv in zip((min(v, 10) for v in g), (max(-10, min(10, v)) for v in u), strict=True)]
                wrong_order.extend(bf16(f32(value * coefficient)) for value in down.project(wrong_hidden))
                if profile == 3:
                    weight_only.extend(down.project(hidden, quantize_inputs=False))
            context, error = ctypes.c_void_p(), Error()
            self.assertTrue(self.native.ria_expert_cpu_create(65, 33, 5, ctypes.byref(context), ctypes.byref(error)))
            try:
                native_input = (ctypes.c_float * (3 * 65))(*(value for row in rows for value in row))
                native_coefficients = (ctypes.c_float * 3)(*coefficients)
                result = (ctypes.c_float * 15)()
                self.assertTrue(self.native.ria_expert_cpu_evaluate(context, ctypes.byref(expert), native_input, 3, 65,
                                native_coefficients, result, 5, ctypes.byref(error)), error.message)
                self.assertEqual(list(result), expected, f"profile={profile}")
                if profile == 3:
                    self.assertNotEqual(expected, wrong_order, "fixture must detect coefficient after down quantization")
                    self.assertNotEqual(expected, weight_only, "fixture must detect W4A16 down projection")
                for row in range(3):
                    split = (ctypes.c_float * 5)()
                    row_input = (ctypes.c_float * 65)(*rows[row])
                    row_coefficient = (ctypes.c_float * 1)(coefficients[row])
                    self.assertTrue(self.native.ria_expert_cpu_evaluate(context, ctypes.byref(expert), row_input, 1, 65,
                                    row_coefficient, split, 5, ctypes.byref(error)), error.message)
                    self.assertEqual(list(split), expected[row * 5:row * 5 + 5])
            finally:
                self.native.ria_expert_cpu_destroy(context)


if __name__ == "__main__":
    unittest.main()
