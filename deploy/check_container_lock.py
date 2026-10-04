#!/usr/bin/env python3
"""Check immutable container recipes against the reviewed base-image lock offline."""
import json
from pathlib import Path
import re


SHA = re.compile(r"sha256:[0-9a-f]{64}\Z")


def validate_container_lock(root):
    lock = json.loads((root / "deploy/container-lock.json").read_text())
    if lock["schema_revision"] != 1 or lock["target"] != "linux/amd64":
        raise ValueError("unsupported container-lock target/revision")
    version = lock["cuda_version"]
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        raise ValueError("CUDA SDK version must be explicit")
    metadata = lock["cuda_registry_verification"]
    if (metadata["provider"], metadata["registry"], metadata["repository"]) != (
            "NVIDIA NGC", "nvcr.io", "nvidia/cuda"):
        raise ValueError("CUDA images must come from NVIDIA NGC")
    images = metadata["images"]
    for field, allowed in (("cuda_devel", ("devel",)), ("cuda_runtime", ("base", "runtime"))):
        reference, separator, sha = lock[field].partition("@")
        flavors = [flavor for flavor in allowed
                   if reference == f"nvcr.io/nvidia/cuda:{version}-{flavor}-ubuntu24.04"]
        if not separator or not SHA.fullmatch(sha) or len(flavors) != 1:
            raise ValueError("CUDA build/final bases must match the locked SDK and OS")
        image = images["devel" if field == "cuda_devel" else "runtime"]
        if image["flavor"] != flavors[0]:
            raise ValueError("CUDA registry role has another image flavor")
        if image["index_digest"] != sha or "linux/amd64" not in image["available_platforms"]:
            raise ValueError("CUDA reference differs from verified registry index/platform")
        for name in ("linux_amd64_manifest_digest", "linux_amd64_config_digest"):
            if not SHA.fullmatch(image[name]):
                raise ValueError("missing immutable amd64 registry identity")
    if metadata["inherited_configuration"]["environment"]["CUDA_VERSION"] != version:
        raise ValueError("base configuration has another CUDA SDK version")
    for kind, expected in (("cpu", [lock["cpu_base"], lock["cpu_base"]]),
                           ("cuda", [lock["cuda_devel"], lock["cuda_runtime"]])):
        recipe = (root / f"deploy/Dockerfile.{kind}").read_text()
        stages = re.findall(r"^FROM\s+(\S+)(?:\s+AS\s+\S+)?\s*$", recipe, re.M)
        external = [stage for stage in stages if ":" in stage or "/" in stage]
        if external != expected:
            raise ValueError(f"{kind} Dockerfile bases differ from container-lock")
        if not re.fullmatch(r"ubuntu:24\.04@sha256:[0-9a-f]{64}", lock["cpu_base"]):
            raise ValueError("CPU image must retain its CUDA-free Ubuntu base")
        if "NVIDIA_DISABLE_REQUIRE" in recipe:
            raise ValueError("container recipe must not bypass NVIDIA driver constraints")
        if not re.search(r"^ENTRYPOINT \[\"/usr/local/bin/ds4-(?:server|expert-server)\"\]$", recipe, re.M):
            raise ValueError("RIA must own its native entrypoint")
    return lock


if __name__ == "__main__":
    validate_container_lock(Path(__file__).resolve().parents[1])
    print("Immutable CPU/NGC CUDA recipes, SDK versions and registry identities agree")
