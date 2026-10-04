#!/usr/bin/env python3
"""Record native source/toolchain identities without an image-digest cycle."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def run(arguments):
    result = subprocess.run(arguments, check=True, capture_output=True, timeout=30,
                            encoding="utf-8", errors="strict")
    if len(result.stdout) > 1 << 20:
        raise ValueError("build tool report exceeds its bound")
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=["cpu", "cuda"])
    parser.add_argument("--compiler", required=True, help="exact compiler executable used by the native build")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    files = set()
    for pattern in ("*.c", "*.h", "*.inc", "ria/*.c", "ria/*.h", "ria/*.cu",
                    "third_party/ryu/**/*", "third_party/iris/**/*", "deploy/*", "locks/*json",
                    "tools/*.py", "tools/ria/*.py", "requirements-ria*.txt", "protocol/*.json"):
        files.update(path for path in root.glob(pattern) if path.is_file())
    files.update(root / name for name in ("Makefile", "Makefile.ria", "LICENSE", ".dockerignore"))
    sources = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(files)}
    source_lock = json.loads((root / "locks/source-lock.json").read_bytes())
    result = {"schema_revision": 1, "image_kind": args.role,
              "canonical_repository": "https://github.com/kklouzal/RIA",
              "upstream_commit": source_lock["upstream_commits"]["antirez/ds4"],
              "source_lock_digest": source_lock["digest"], "sources": sources,
              "compiler_executable": args.compiler, "compiler": run([args.compiler, "--version"]),
              "packages": run(["dpkg-query", "-W", "-f=${Package}=${Version}\n"]),
              "architecture": run(["dpkg", "--print-architecture"]),
              "numeric_flags": ["-fno-fast-math", "-ffp-contract=off"],
              "hardware_qualified": False}
    binary_dir = root / ("bin/cuda" if args.role == "cuda" else "bin")
    binary_names = ["ds4ctl", "ds4-expert-server", "ds4-ria-qualify"]
    if args.role == "cuda":
        binary_names += ["ds4", "ds4-server", "ds4-eval"]
    result["binaries"] = {name: hashlib.sha256((binary_dir / name).read_bytes()).hexdigest()
                          for name in binary_names}
    if args.role == "cuda":
        result["nvcc"] = run(["/usr/local/cuda/bin/nvcc", "--version"])
        result["cuda_host_compiler_executable"] = "g++"
        result["cuda_host_compiler"] = run(["g++", "--version"])
        result["code_targets"] = ["sm_120a"]
        result["ptx_jit"] = False
        result["numeric_flags"] += ["--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true"]
        artifact = root / "build/ria/evidence/cuda-artifacts.json"
        report = json.loads(artifact.read_bytes())
        if not report["passed"] or report["kernel_execution"] or report["ptx_present"]:
            raise ValueError("CUDA build lacks successful offline AOT inspection")
        for binary in report["binaries"]:
            if result["binaries"].get(Path(binary["file"]).name) != binary["sha256"]:
                raise ValueError("CUDA AOT report belongs to another executable")
        result["cuda_artifact_report_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    # This schema contains only ASCII keys/strings, Booleans, and safe integers;
    # sorted compact UTF8 therefore equals RFC8785 for these exact values.
    def canonical(value):
        data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
        if not data.isascii():
            raise ValueError("build-info schema requires ASCII strings")
        return data
    result["digest"] = hashlib.sha256(canonical(result)).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical(result) + b"\n")
    print(result["digest"])


if __name__ == "__main__":
    main()
