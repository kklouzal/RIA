"""Independent offline byte conversion primitives; never a serving engine."""

import bisect
import math
import struct

from .identity import ArtifactError

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def f32(value):
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError as exc:
        raise ArtifactError("FP32 conversion overflow") from exc


def bf16_encode(value):
    bits = struct.unpack("<I", struct.pack("<f", f32(value)))[0]
    if bits & 0x7F800000 == 0x7F800000:
        raise ArtifactError("nonfinite BF16 input")
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def bf16_decode(code):
    return struct.unpack("<f", struct.pack("<I", code << 16))[0]


def e4m3_decode(code):
    negative, exponent, mantissa = code >> 7, (code >> 3) & 15, code & 7
    if exponent == 15 and mantissa == 7:
        raise ArtifactError("nonfinite E4M3FN code")
    value = math.ldexp(mantissa / 8, -6) if exponent == 0 else math.ldexp(1 + mantissa / 8, exponent - 7)
    return -value if negative else value


E4M3_POSITIVE = tuple(e4m3_decode(code) for code in range(127))


def e4m3_encode(value):
    if not math.isfinite(value):
        raise ArtifactError("nonfinite E4M3FN input")
    sign = 128 if math.copysign(1, value) < 0 else 0
    magnitude = min(abs(value), 448)
    upper = min(bisect.bisect_left(E4M3_POSITIVE, magnitude), 126)
    lower = max(upper - 1, 0)
    dl, du = magnitude - E4M3_POSITIVE[lower], E4M3_POSITIVE[upper] - magnitude
    chosen = lower if dl < du or (dl == du and lower % 2 == 0) else upper
    return sign | chosen


def e2m1_decode(code):
    return -E2M1[code & 7] if code & 8 else E2M1[code & 7]


def ue8m0_decode(code):
    if not 0 <= code <= 254:
        raise ArtifactError("nonfinite UE8M0 code")
    return math.ldexp(1.0, code - 127)


def power2_scale(values, maximum=448.0):
    if any(not math.isfinite(value) for value in values):
        raise ArtifactError("nonfinite quantizer input")
    # Native fast_round_scale is FP32 amax * FP32(1/max), then bitwise ceil.
    amax = max(max((abs(value) for value in values), default=0), f32(1e-4))
    scale = f32(f32(amax) * f32(1.0 / maximum))
    bits = struct.unpack("<I", struct.pack("<f", scale))[0]
    exponent = ((bits >> 23) & 255) - 127 + bool(bits & 0x7FFFFF)
    code = exponent + 127
    if not 0 <= code <= 254:
        raise ArtifactError("scale cannot be represented as UE8M0")
    return code, ue8m0_decode(code)


def decode_matrix_rows(values, scales, rows, columns, representation, weight_global_scale, *, first_row=0):
    """Decode at most a caller-bounded row block from packed immutable bytes."""
    if not math.isfinite(weight_global_scale) or weight_global_scale <= 0:
        raise ArtifactError("missing positive global weight multiplier")
    if representation not in ("source_mxfp4", "nvfp4", "fp8_block32", "bf16"):
        raise ArtifactError("unsupported weight representation")
    result = []
    stride = (columns + 1) // 2 if representation in ("source_mxfp4", "nvfp4") else columns * (2 if representation == "bf16" else 1)
    if len(values) != rows * stride:
        raise ArtifactError("matrix value block size mismatch")
    for row in range(rows):
        output = []
        for column in range(columns):
            if representation == "bf16":
                value = bf16_decode(struct.unpack_from("<H", values, row * stride + 2 * column)[0])
            elif representation in ("source_mxfp4", "nvfp4"):
                byte = values[row * stride + column // 2]
                value = e2m1_decode((byte >> (4 * (column % 2))) & 15)
                group = 16 if representation == "nvfp4" else 32
                scale_code = scales[row * ((columns + group - 1) // group) + column // group]
                scale = e4m3_decode(scale_code) if representation == "nvfp4" else ue8m0_decode(scale_code)
                if scale < 0:
                    raise ArtifactError("negative weight block scale")
                value = f32(f32(value * scale) * weight_global_scale)
            else:
                value = e4m3_decode(values[row * stride + column])
                scale_row = (first_row + row) // 32 - first_row // 32
                scale_code = scales[scale_row * ((columns + 31) // 32) + column // 32]
                value = f32(value * ue8m0_decode(scale_code))
            if not math.isfinite(value):
                raise ArtifactError("nonfinite decoded weight")
            output.append(value)
        result.append(output)
    return result


def encode_bf16(rows):
    return b"".join(struct.pack("<H", bf16_encode(value)) for row in rows for value in row)


def encode_fp8_block32(rows):
    """A complete aligned 32-row tile; K grouping stays exactly 32×32."""
    if not rows or len(rows) > 32 or any(len(row) != len(rows[0]) for row in rows):
        raise ArtifactError("invalid FP8 conversion tile")
    columns = len(rows[0])
    codes = [bytearray(columns) for _ in rows]
    scale_codes = bytearray()
    for lower in range(0, columns, 32):
        population = [value for row in rows for value in row[lower:lower + 32]]
        code, scale = power2_scale(population)
        scale_codes.append(code)
        for output, row in zip(codes, rows, strict=True):
            for column in range(lower, min(lower + 32, columns)):
                output[column] = e4m3_encode(f32(row[column] / scale))
    return b"".join(codes), bytes(scale_codes)
