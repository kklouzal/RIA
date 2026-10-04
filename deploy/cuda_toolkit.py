#!/usr/bin/env python3
"""Validate selected native CUDA build inputs without loading CUDA or a driver."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess


def tool_report(arguments):
    result = subprocess.run(arguments, check=True, capture_output=True, timeout=30,
                            encoding="utf-8", errors="strict")
    if len(result.stdout) > 4 << 20 or len(result.stderr) > 4 << 20:
        raise ValueError("CUDA tool report exceeds its bound")
    return result.stdout.strip()


def file_identity(path, home, maximum=256 << 20):
    path = path.resolve(strict=True)
    if not path.is_relative_to(home):
        raise ValueError("CUDA input resolves outside the selected toolkit: " + str(path))
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as source:
        size = os.fstat(source.fileno())
        if not stat.S_ISREG(size.st_mode) or not 0 < size.st_size <= maximum:
            raise ValueError("CUDA input is not a bounded nonempty regular file")
        identity = hashlib.file_digest(source, "sha256").hexdigest()
    return str(path.relative_to(home)), {"sha256": identity, "bytes": size.st_size}


def sdk_version(home, release, required):
    """SDK patches differ from NVCC/cudart component patches; never infer one."""
    metadata = home / "version.json"
    if metadata.exists():
        file_identity(metadata, home, 1 << 20)
        try:
            version = json.loads(metadata.read_bytes())["cuda"]["version"]
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError("invalid CUDA SDK release metadata") from error
        source = "version.json"
    elif required:
        # The pinned NGC devel image installs this SDK meta package at its
        # actual release patch, unlike cuda-nvcc/cudart component packages.
        package = "cuda-minimal-build-" + release.replace(".", "-")
        report = tool_report(["dpkg-query", "-W", "-f=${db:Status-Status}\t${Version}", package])
        status, package_version = report.split("\t")
        if status != "installed":
            raise ValueError("selected CUDA SDK meta package is not installed")
        version = package_version.split("-", 1)[0]
        source = "dpkg:" + package
    else:
        return None, None
    if not isinstance(version, str) or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        raise ValueError("invalid CUDA SDK release metadata")
    if version.rsplit(".", 1)[0] != release:
        raise ValueError("CUDA SDK metadata and selected NVCC release differ")
    return version, source


def inspect_toolkit(cuda_home, nvcc=None, cuobjdump=None):
    home = Path(cuda_home).resolve(strict=True)
    nvcc = Path(nvcc if nvcc is not None else home / "bin/nvcc").resolve(strict=True)
    cuobjdump = Path(cuobjdump if cuobjdump is not None else home / "bin/cuobjdump").resolve(strict=True)
    if nvcc != (home / "bin/nvcc").resolve(strict=True) or cuobjdump != (home / "bin/cuobjdump").resolve(strict=True):
        raise ValueError("selected CUDA tools and toolkit home differ")
    files = {}
    inputs = (nvcc, cuobjdump, home / "include/cuda_runtime_api.h", home / "lib64/libcudart_static.a")
    for path in inputs:
        name, identity = file_identity(path, home, 8 << 20 if path.name == "cuda_runtime_api.h" else 256 << 20)
        files[name] = identity
    report = tool_report([str(nvcc), "--version"])
    match = re.search(r"\brelease ([0-9]+)\.([0-9]+), V([0-9]+\.[0-9]+\.[0-9]+)\b", report)
    if not match or match[3].rsplit(".", 1)[0] != match[1] + "." + match[2]:
        raise ValueError("NVCC lacks a coherent CUDA release report")
    release = match[1] + "." + match[2]
    header = (home / "include/cuda_runtime_api.h").read_text(encoding="utf-8", errors="strict")
    constant = re.search(r"^\s*#\s*define\s+CUDART_VERSION\s+([0-9]+)\s*$", header, re.MULTILINE)
    expected = int(match[1]) * 1000 + int(match[2]) * 10
    if not constant or int(constant[1]) != expected:
        raise ValueError("selected CUDA runtime header and NVCC releases differ")
    archive = home / "lib64/libcudart_static.a"
    with archive.open("rb") as source:
        if source.read(8) != b"!<arch>\n":
            raise ValueError("selected CUDA static runtime is not an archive")
    symbols = tool_report(["nm", "-g", "--defined-only", str(archive)])
    if not re.search(r"\b[TW]\s+cudaRuntimeGetVersion\s*$", symbols, re.MULTILINE):
        raise ValueError("selected CUDA archive lacks the native runtime API")
    container_version = os.environ.get("CUDA_VERSION")
    if "CUDA_VERSION" in os.environ:
        lock = json.loads((Path(__file__).resolve().parent / "container-lock.json").read_bytes())
        if not container_version or container_version != lock.get("cuda_version"):
            raise ValueError("container CUDA_VERSION differs from the pinned CUDA SDK")
    version, version_source = sdk_version(home, release, "CUDA_VERSION" in os.environ)
    if container_version is not None and version != container_version:
        raise ValueError("installed CUDA SDK patch differs from the pinned container release")
    return {"home": str(home), "nvcc_executable": str(nvcc), "cuobjdump_executable": str(cuobjdump),
            "nvcc": report, "release": release, "runtime_api_version": expected,
            "runtime_linkage": "static", "sdk_version": version, "sdk_version_source": version_source,
            "container_cuda_version": container_version, "files": files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-home", default="/usr/local/cuda")
    parser.add_argument("--nvcc")
    parser.add_argument("--cuobjdump")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = json.dumps(inspect_toolkit(args.cuda_home, args.nvcc, args.cuobjdump),
                        sort_keys=True, separators=(",", ":")).encode() + b"\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Unchanged inputs retain their timestamp, so only changed toolkits rebuild
    # CUDA objects. This also prevents old objects surviving a toolkit switch.
    if not args.output.exists() or args.output.read_bytes() != result:
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_bytes(result)
        temporary.replace(args.output)
    print("Selected CUDA toolkit validated; no CUDA/driver initialization")


if __name__ == "__main__":
    main()
