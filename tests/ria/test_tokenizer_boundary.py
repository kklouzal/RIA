"""Actual tokenizer admission without model/GPU or shared build outputs."""

import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "locks/metadata/deepseek-ai/DeepSeek-V4.1-Flash"
    / "2cba9e42aa026125f3ed06c6d98c1db82f7ca027/tokenizer.json"
)


@pytest.fixture(scope="module")
def boundary(tmp_path_factory):
    configured = os.environ.get("RIA_TOKENIZER_BOUNDARY_FIXTURE")
    if configured:
        executable = Path(configured).resolve()
        assert executable.is_file(), f"missing explicit tokenizer fixture: {executable}"
        return executable
    executable = tmp_path_factory.mktemp("tokenizer-boundary") / "fixture"
    arguments = [
        "cc", "-I.", "-Iria", "-Ithird_party/ryu", "-std=c99", "-O2", "-g",
        "-Wall", "-Wextra", "-Werror", "-Wformat=2", "-Wstrict-prototypes",
        "-Wmissing-prototypes", "-fno-fast-math", "-ffp-contract=off",
        "-ffunction-sections", "-fdata-sections",
        "tests/ria/test_tokenizer_boundary.c", "ria/tokenizer.c", "ria/json.c",
        "ria/common.c", "third_party/ryu/ryu/d2s.c", "-Wl,--gc-sections",
        "-Wl,--wrap=read", "-Wl,--wrap=__read_chk", "-lonig", "-lcrypto", "-lpthread", "-lm",
        "-o", str(executable),
    ]
    result = subprocess.run(arguments, cwd=ROOT, capture_output=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr.decode()
    return executable


def run(boundary, path, digest="0" * 64, *, mutation=False, timeout=10):
    arguments = [str(boundary), str(path), digest]
    if mutation:
        arguments.append("mutate")
    return subprocess.run(arguments, cwd=ROOT, capture_output=True, timeout=timeout, check=False)


def test_writerless_fifo_is_rejected_without_waiting(boundary, tmp_path):
    path = tmp_path / "tokenizer.fifo"
    os.mkfifo(path)
    result = run(boundary, path, timeout=1)
    assert result.returncode == 1
    assert b"regular file" in result.stderr


@pytest.mark.parametrize("kind", ["directory", "empty", "oversized", "symlink", "device"])
def test_nonregular_or_unbounded_sources_are_rejected(boundary, tmp_path, kind):
    path = tmp_path / "source"
    if kind == "directory":
        path.mkdir()
    elif kind == "empty":
        path.touch()
    elif kind == "oversized":
        with path.open("wb") as stream:
            stream.truncate(16777217)
    elif kind == "symlink":
        path.symlink_to(SOURCE)
    else:
        path = Path("/dev/null")
    result = run(boundary, path)
    assert result.returncode == 1
    assert result.stderr


def test_pinned_regular_source_and_hash_are_preserved(boundary):
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result = run(boundary, SOURCE, digest)
    assert result.returncode == 0, result.stderr.decode()
    assert b"pinned tokenizer loaded" in result.stdout
    result = run(boundary, SOURCE)
    assert result.returncode == 1
    assert b"SHA256 mismatch" in result.stderr


def test_rewritten_source_is_rejected_after_bytes_and_mtime_are_restored(boundary, tmp_path):
    path = tmp_path / "source"
    shutil.copyfile(SOURCE, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result = run(boundary, path, digest, mutation=True)
    assert result.returncode == 1
    assert b"unreadable input" in result.stderr
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
