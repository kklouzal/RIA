"""Independent CPU expert oracle for full admitted prompt microbatches.

Rational value palettes and explicit scalar casts are separate from production
quantizers/GEMMs. Only synthetic matrices execute; CUDA remains uninitialized.
"""
import ctypes
import importlib.util
import math
from pathlib import Path
import random
import re
import subprocess
import unittest


_spec = importlib.util.spec_from_file_location("ria_prefill_reference", Path(__file__).with_name("reference_expert.py"))
oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oracle)


class PrefillExpertOracle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        oracle.OracleTests.setUpClass.__func__(cls)
        # This scalar CUDA kernel has no thread synchronization. Run its exact
        # body as C after replacing only CUDA builtins, against the independent
        # rational quantizer below. No CUDA library or device is loaded.
        text = (oracle.ROOT / "ria/expert_cuda.cu").read_text()
        start = re.search(r"static __global__ void quantize_kernel\(", text).start()
        opening = text.index("{", start)
        depth, end = 1, opening + 1
        while depth:
            depth += (text[end] == "{") - (text[end] == "}")
            end += 1
        source = Path(cls.temporary.name) / "quantizer.c"
        source.write_text('''
#include "numeric.h"
#include "expert.h"
#include <stdint.h>
#define __global__
typedef struct { uint64_t x; } dimension;
static dimension blockIdx,blockDim,threadIdx;
static void atomicExch(int *destination,int value) { *destination=value; }
static float __fmul_rn(float x,float y) { return x*y; }
''' + text[start:end] + '''
int fixture_quantize(const float *input,float *output,float *factors,
                     uint8_t *codes,uint8_t *scales,uint64_t width,
                     uint64_t rows,float global,int profile) {
    uint64_t group=profile==1 ? width : profile==2 ? 32 : 16;
    uint64_t groups=(width+group-1)/group;int error=0;
    blockDim.x=1;threadIdx.x=0;
    for (blockIdx.x=0;blockIdx.x<rows*groups;blockIdx.x++)
        quantize_kernel(input,output,factors,codes,scales,width,group,groups,rows,global,profile,&error);
    return error;
}
''')
        library = Path(cls.temporary.name) / "quantizer.so"
        subprocess.run(["cc", "-std=c99", "-O2", "-fno-fast-math", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror",
                        "-fPIC", "-shared", "-I", str(oracle.ROOT / "ria"), str(source), "-lm", "-o", str(library)], check=True)
        cls.quantizer = ctypes.CDLL(str(library))
        cls.quantizer.fixture_quantize.argtypes = [ctypes.POINTER(ctypes.c_float)] * 3 + [ctypes.POINTER(ctypes.c_uint8)] * 2 + [
            ctypes.c_uint64, ctypes.c_uint64, ctypes.c_float, ctypes.c_int]
        cls.quantizer.fixture_quantize.restype = ctypes.c_int

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_all_profiles_preserve_each_row_under_64_row_permuted_microbatches(self):
        rng = random.Random(730064)
        rows = [[oracle.f32(rng.uniform(-1, 1) * (2 ** (row % 7 - 3))) for _ in range(33)] for row in range(64)]
        coefficients = [oracle.f32((row % 13) / 8) for row in range(64)]
        # Vary each row's magnitude substantially: an accidental batch-wide
        # activation scale or moving coefficients after down quantization
        # cannot reproduce the independent scalar result.
        order = list(range(64))
        rng.shuffle(order)
        for profile in (1, 2, 3):
            with self.subTest(profile=profile):
                gate = oracle.Prepared(profile, 17, 33, rng, .5, .125)
                up = oracle.Prepared(profile, 17, 33, rng, 2, .25)
                down = oracle.Prepared(profile, 3, 17, rng, .25, .015625)
                expert = oracle.Expert(gate.matrix, up.matrix, down.matrix, 10)
                expected = []
                for row, coefficient in zip(rows, coefficients, strict=True):
                    hidden = []
                    for gate_value, up_value in zip(gate.project(row), up.project(row), strict=True):
                        gate_value = min(gate_value, 10)
                        up_value = max(-10, min(10, up_value))
                        silu = oracle.f32(gate_value / oracle.f32(1 + oracle.f32(math.exp(-gate_value))))
                        hidden.append(oracle.bf16(oracle.f32(oracle.f32(silu * up_value) * coefficient)))
                    expected.append(down.project(hidden))
                context, error = ctypes.c_void_p(), oracle.Error()
                self.assertTrue(self.native.ria_expert_cpu_create(33, 17, 3, ctypes.byref(context), ctypes.byref(error)))
                try:
                    for chunk in (1, 2, 3, 8, 16, 64):
                        reconstructed = [None] * 64
                        for first in range(0, 64, chunk):
                            subgroup = order[first:first + chunk]
                            inputs = (ctypes.c_float * (len(subgroup) * 33))(
                                *(value for row_id in subgroup for value in rows[row_id]))
                            weights = (ctypes.c_float * len(subgroup))(*(coefficients[row_id] for row_id in subgroup))
                            output = (ctypes.c_float * (len(subgroup) * 3))()
                            self.assertTrue(self.native.ria_expert_cpu_evaluate(
                                context, ctypes.byref(expert), inputs, len(subgroup), 33, weights, output, 3,
                                ctypes.byref(error)), error.message)
                            values = list(output)
                            for index, row_id in enumerate(subgroup):
                                reconstructed[row_id] = values[index * 3:index * 3 + 3]
                        self.assertEqual(reconstructed, expected, f"profile={profile}, chunk={chunk}")
                finally:
                    self.native.ria_expert_cpu_destroy(context)

    def test_exact_cuda_quantizer_body_preserves_row_domains_under_microbatch_changes(self):
        rng = random.Random(7316033)
        for width in (17, 33, 65):
            rows = [[oracle.f32(rng.uniform(-1, 1) * (2 ** (row % 9 - 4))) for _ in range(width)] for row in range(64)]
            for profile in (1, 2, 3):
                group = width if profile == 1 else 32 if profile == 2 else 16
                groups = (width + group - 1) // group
                expected = [oracle.quantize(row, profile, .125) for row in rows]
                for chunk in (1, 3, 8, 17, 64):
                    for first in range(0, 64, chunk):
                        subgroup = rows[first:first + chunk]
                        values = (ctypes.c_float * (len(subgroup) * width))(*(v for row in subgroup for v in row))
                        decoded = (ctypes.c_float * len(values))()
                        factors = (ctypes.c_float * (len(subgroup) * groups))()
                        codes = (ctypes.c_uint8 * (len(values) * 2))()
                        scales = (ctypes.c_uint8 * len(factors))()
                        self.assertEqual(self.quantizer.fixture_quantize(
                            values, decoded, factors, codes, scales, width, len(subgroup), .125, profile), 0)
                        for row in range(len(subgroup)):
                            decoded_expected, factors_expected = expected[first + row]
                            self.assertEqual(list(decoded)[row * width:(row + 1) * width], decoded_expected,
                                             f"width={width}, profile={profile}, chunk={chunk}, row={first + row}")
                            self.assertEqual(list(factors)[row * groups:(row + 1) * groups],
                                             factors_expected if profile != 1 else [1])


if __name__ == "__main__":
    unittest.main()
