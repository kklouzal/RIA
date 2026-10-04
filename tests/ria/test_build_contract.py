"""Reject builds that remove finite checks or change the reviewed CUDA flags."""
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deploy"))
from check_cuda_artifacts import INSTRUCTIONS, validate_production_instructions

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("compiler", ["gcc", "clang"])
@pytest.mark.parametrize("flag", ["-ffast-math", "-ffinite-math-only"])
def test_host_numeric_contract_rejects_relaxed_semantics(compiler, flag):
    result = subprocess.run([compiler, "-std=c99", "-Iria", "-fsyntax-only", flag, "ria/common.c"],
                            cwd=ROOT, capture_output=True, timeout=10)
    assert result.returncode != 0
    assert b"RIA numerical contracts require" in result.stderr


@pytest.mark.parametrize("flags", ["", "--use_fast_math", "--fmad=true", "--ftz=true", "--prec-div=false", "--prec-sqrt=false"])
def test_cuda_flags_cannot_bypass_reviewed_contract(flags):
    result = subprocess.run(["make", "-n", "ria-cuda", "RIA_NVCCFLAGS=" + flags],
                            cwd=ROOT, capture_output=True, timeout=10)
    assert result.returncode != 0
    assert b"fixed by the reviewed numerical and AOT build contract" in result.stderr


@pytest.mark.parametrize("name,flags", [("RIA_CFLAGS", "-ffp-contract=fast"), ("RIA_CFLAGS", "-freciprocal-math"),
                                      ("RIA_CFLAGS", ""), ("RIA_DONOR_FLAGS", "-ffp-contract=fast")])
def test_host_build_flags_match_declared_provenance(name, flags):
    result = subprocess.run(["make", "-n", "ria-cpu", name + "=" + flags], cwd=ROOT, capture_output=True, timeout=10)
    assert result.returncode != 0
    assert b"fixed by the reviewed numerical build contract" in result.stderr


def test_probe_matrix_instructions_cannot_certify_production_kernels():
    assembly = "Function : native_probe_kernel\nHMMA.16816.F32.BF16\nQMMA.SF.16832.F32.E4M3.E4M3.E8\nOMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X\n"
    for bits, _ in INSTRUCTIONS.values():
        assembly += f"Function : _Z24native_projection_kernelILj{bits}EEv\nMOV R0, R1\n"
    with pytest.raises(ValueError, match="production projection"):
        validate_production_instructions(assembly)
