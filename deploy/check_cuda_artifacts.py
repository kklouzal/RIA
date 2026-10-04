#!/usr/bin/env python3
"""Inspect AOT CUDA machine code without loading a driver or launching a kernel."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from offline_checks import execute

INSTRUCTIONS = {"bf16": (16, r"\bHMMA\.16816\.F32\.BF16\b"),
                "fp8": (8, r"\bQMMA\.SF\.16832\.F32\.E4M3\.E4M3\.E8\b"),
                "nvfp4": (4, r"\bOMMA\.SF\.16864\.F32\.E2M1\.E2M1\.UE4M3\.4X\b")}


def validate_production_instructions(assembly):
    sections = re.split(r"^\s*Function : (\S+)\s*$", assembly, flags=re.MULTILINE)
    observed = {}
    for profile, (bits, expression) in INSTRUCTIONS.items():
        matches = [body for name, body in zip(sections[1::2], sections[2::2], strict=True)
                   if re.search(r"native_projection_kernelILj" + str(bits) + r"EE", name)]
        if not matches or any(re.search(expression, body) is None for body in matches):
            raise ValueError("missing production projection matrix instructions: " + profile)
        observed[profile] = len(matches)
    return observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuobjdump", default="/usr/local/cuda/bin/cuobjdump")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checks = []
    binaries = ("ds4ctl", "ds4-expert-server", "ds4-server", "ds4", "ds4-eval", "ds4-ria-qualify")
    for name in binaries:
        path = root / "bin/cuda" / name
        observed = {"file": str(path.relative_to(root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for kind, option in (("elf", "--list-elf"), ("ptx", "--list-ptx")):
            log = args.output.parent / ("cuda-" + name + "-" + kind + ".log")
            if execute([args.cuobjdump, option, str(path)], root, log, 60):
                raise ValueError("CUDA artifact inspection failed: " + str(log))
            report = log.read_text(encoding="utf-8", errors="strict")
            if kind == "elf":
                targets = re.findall(r"\.sm_([0-9]+a?)\.", report)
                if not targets or set(targets) != {"120a"}:
                    raise ValueError("CUDA executable does not contain exclusively SM120a cubins")
                observed["cubin_count"] = len(targets)
            else:
                absent = "cuobjdump info    : No PTX file found to extract from '" + str(path) + "'. You may try with -all option."
                if report.strip() not in ("", absent):
                    raise ValueError("CUDA executable contains PTX or an unexpected inspection diagnostic")
            observed[kind + "_report_sha256"] = hashlib.sha256(log.read_bytes()).hexdigest()
        sass = args.output.parent / ("cuda-" + name + "-sass.log")
        if execute([args.cuobjdump, "--dump-sass", str(path)], root, sass, 60):
            raise ValueError("CUDA executable disassembly failed")
        observed["production_projection_sections"] = validate_production_instructions(sass.read_text(encoding="utf-8", errors="strict"))
        observed["sass_report_sha256"] = hashlib.sha256(sass.read_bytes()).hexdigest()
        checks.append(observed)
    expert = root / "build/ria/ria/expert_cuda.cu.o"
    sass = args.output.parent / "cuda-expert-sass.log"
    if execute([args.cuobjdump, "--dump-sass", str(expert)], root, sass, 60):
        raise ValueError("CUDA expert disassembly failed")
    assembly = sass.read_text(encoding="utf-8", errors="strict")
    production = validate_production_instructions(assembly)
    result = {"schema_revision": 1, "classification": "offline CUDA binary inspection",
              "kernel_execution": False, "hardware_qualified": False, "passed": True,
              "code_targets": ["sm_120a"], "ptx_present": False, "binaries": checks,
              "expert_object_sha256": hashlib.sha256(expert.read_bytes()).hexdigest(),
              "expert_sass_sha256": hashlib.sha256(sass.read_bytes()).hexdigest(),
              "matrix_instruction_patterns": {key: value[1] for key, value in INSTRUCTIONS.items()},
              "production_projection_sections": production}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("CUDA AOT artifacts passed; no kernel executed:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
