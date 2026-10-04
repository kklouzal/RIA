"""Image pins and loader failures must be detected without a GPU or image execution."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deploy"))
from check_container_lock import validate_container_lock
from check_runtime_linkage import BINARIES, parse_linkage, verify_runtime


ROOT = Path(__file__).resolve().parents[2]
LOADER = """linux-vdso.so.1 (0x00007ffed50c6000)
libstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007f01ad780000)
libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007f01ad520000)
/lib64/ld-linux-x86-64.so.2 (0x00007f01add08000)
"""


def copy_recipes(tmp_path):
    directory = tmp_path / "deploy"
    directory.mkdir()
    for name in ("container-lock.json", "Dockerfile.cpu", "Dockerfile.cuda"):
        (directory / name).write_bytes((ROOT / "deploy" / name).read_bytes())
    return json.loads((directory / "container-lock.json").read_text())


def test_current_locked_container_pair():
    assert validate_container_lock(ROOT)["target"] == "linux/amd64"


@pytest.mark.parametrize("mutation", ["version", "registry", "platform", "digest", "recipe", "cpu"])
def test_divergent_container_inputs_reject(tmp_path, mutation):
    lock = copy_recipes(tmp_path)
    if mutation == "version":
        lock["cuda_version"] = "13.1.1"
    elif mutation == "registry":
        lock["cuda_registry_verification"]["registry"] = "docker.io"
    elif mutation == "platform":
        lock["cuda_registry_verification"]["images"]["devel"]["available_platforms"] = ["linux/arm64"]
    elif mutation == "digest":
        lock["cuda_registry_verification"]["images"]["devel"]["index_digest"] = "sha256:" + "0" * 64
    elif mutation == "recipe":
        path = tmp_path / "deploy/Dockerfile.cuda"
        path.write_text(path.read_text().replace(lock["cuda_devel"], lock["cuda_runtime"], 1))
    else:
        lock["cpu_base"] = lock["cuda_runtime"]
    (tmp_path / "deploy/container-lock.json").write_text(json.dumps(lock))
    with pytest.raises(ValueError):
        validate_container_lock(tmp_path)


def test_loader_trace_ignores_only_addresses_and_virtual_loader():
    assert parse_linkage(LOADER) == {"libstdc++.so.6": "/lib/x86_64-linux-gnu/libstdc++.so.6",
                                    "libgcc_s.so.1": "/lib/x86_64-linux-gnu/libgcc_s.so.1"}


@pytest.mark.parametrize("text", ["", "libstdc++.so.6 => not found\n", "not a dynamic executable\n",
                                  "unexpected loader error\n", LOADER + LOADER])
def test_loader_stdout_failures_reject_even_when_exit_is_zero(text):
    with pytest.raises(ValueError):
        parse_linkage(text)


def test_native_binary_hashes_and_loader_failure(tmp_path, monkeypatch):
    binaries = {}
    for name in BINARIES:
        (tmp_path / name).write_bytes(b"isolated loader fixture")
        binaries[name] = hashlib.sha256(b"isolated loader fixture").hexdigest()
    build = {"image_kind": "cuda", "hardware_qualified": False, "binaries": binaries, "digest": "a" * 64}
    calls = []

    def loader(arguments, **options):
        calls.append(arguments)
        assert arguments[:2] == ["ldd", "--"]
        assert Path(arguments[2]).parent == tmp_path
        assert not any(name.startswith("LD_") for name in options["env"])
        assert options["env"]["LC_ALL"] == "C"
        return subprocess.CompletedProcess(arguments, 0, LOADER, "")

    monkeypatch.setattr(subprocess, "run", loader)
    monkeypatch.setenv("LD_PRELOAD", "/untrusted-runtime.so")
    result = verify_runtime(tmp_path, build)
    assert result["passed"] and len(calls) == 6
    assert not result["kernel_execution"] and not result["hardware_qualified"]
    changed = copy.deepcopy(build)
    changed["binaries"]["ds4"] = "0" * 64
    with pytest.raises(ValueError, match="differs from build provenance"):
        verify_runtime(tmp_path, changed)
    monkeypatch.setattr(subprocess, "run", lambda arguments, **options:
                        subprocess.CompletedProcess(arguments, 0, "libstdc++.so.6 => not found\n", ""))
    with pytest.raises(ValueError, match="unresolved"):
        verify_runtime(tmp_path, build)
