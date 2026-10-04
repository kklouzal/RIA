#!/usr/bin/env python3
"""Check trusted native ELF dependency closure through loader tracing, without main/GPU execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


BINARIES = {"ds4ctl", "ds4-expert-server", "ds4-ria-qualify", "ds4", "ds4-server", "ds4-eval"}
CPU_BINARIES = {"ds4ctl", "ds4-expert-server", "ds4-ria-qualify"}


def parse_linkage(output):
    """ldd can exit zero with missing libraries: validate every loader result."""
    if not output or "not found" in output or "not a dynamic executable" in output:
        raise ValueError("native image has unresolved runtime dependencies")
    libraries = {}
    for raw in output.splitlines():
        line = raw.strip()
        if not line:
            continue
        if re.fullmatch(r"linux-vdso\.so\.[0-9]+ \(0x[0-9a-f]+\)", line):
            continue
        match = re.fullmatch(r"([^ /]+) => (/[^ ]+) \(0x[0-9a-f]+\)", line)
        if match:
            name, path = match.groups()
            if name in libraries:
                raise ValueError("duplicate loader dependency")
            libraries[name] = path
            continue
        if re.fullmatch(r"/[^ ]+/ld-(?:linux[^ ]*|[0-9.]+)\.so(?:\.[0-9]+)? \(0x[0-9a-f]+\)", line):
            continue
        raise ValueError("unexpected loader-trace diagnostic: " + line)
    if not libraries:
        raise ValueError("loader tracing produced no dependencies")
    return libraries


def verify_runtime(binary_dir, build_info):
    expected_binaries = {"cuda": BINARIES, "cpu": CPU_BINARIES}.get(build_info["image_kind"])
    if (expected_binaries is None or build_info["hardware_qualified"] is not False or
            set(build_info["binaries"]) != expected_binaries):
        raise ValueError("runtime linkage requires exact CPU/CUDA build provenance")
    checks = []
    for name, expected in sorted(build_info["binaries"].items()):
        path = binary_dir / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 << 20:
            raise ValueError("native executable is not a bounded regular build artifact")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("runtime executable differs from build provenance")
        # These are project-owned ELF executables with the standard glibc
        # interpreter. ldd invokes its dependency-tracing mode, never main or
        # CUDA initializers; no model, GPU or network is used by this check.
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith("LD_")}
        environment.update(LC_ALL="C", LANG="C")
        result = subprocess.run(["ldd", "--", str(path)], env=environment,
                                capture_output=True, check=True, timeout=30,
                                encoding="utf-8", errors="strict")
        if len(result.stdout) > 1 << 20 or result.stderr:
            raise ValueError("unexpected or excessive loader-trace output")
        checks.append({"binary": name, "sha256": actual,
                       "resolved_libraries": parse_linkage(result.stdout)})
    return {"schema_revision": 1, "classification": "native ELF loader tracing only; no main or GPU execution",
            "passed": True, "hardware_qualified": False, "kernel_execution": False,
            "build_digest": build_info["digest"], "binaries": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary-dir", required=True, type=Path)
    parser.add_argument("--build-info", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = args.build_info.read_bytes()
    if len(data) > 1 << 20:
        raise ValueError("build metadata exceeds its bound")
    build_info = json.loads(data)
    result = verify_runtime(args.binary_dir, build_info)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(f"All {len(result['binaries'])} native {build_info['image_kind']} executables resolve runtime image libraries without GPU execution")


if __name__ == "__main__":
    main()
