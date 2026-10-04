"""Python writer to native loader: authenticate mutations before exercising bounds."""

import copy
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from ria.identity import atomic_json, read_json, seal
from ria.preparation import prepare
from test_artifacts import fixture_recipe

ROOT = Path(__file__).resolve().parents[2]


def run_loader(root, digest, *, validate_only=True):
    executable = Path(os.environ.get("RIA_TENSOR_FIXTURE", ROOT / "build/ria/tests/test_tensor"))
    command = [str(executable), str(root / "manifest.json"), digest]
    if validate_only:
        command.append("--validate-only")
    return subprocess.run(command, capture_output=True, timeout=15, check=False)


def fixture(tmp_path):
    root = tmp_path / "prepared"
    manifest = prepare(tmp_path, fixture_recipe(tmp_path), root)
    return root, manifest, read_json(root / manifest["tensor_pages"][0]["path"])


def reanchor(root, manifest, page):
    page = seal(page)
    atomic_json(root / manifest["tensor_pages"][0]["path"], page)
    old = manifest["tensor_pages"][0]["digest"]
    manifest["tensor_pages"][0]["digest"] = page["digest"]
    for reference in manifest["metadata"]:
        if reference["digest"] == old:
            reference["digest"] = page["digest"]
    layout = read_json(root / "layout.json")
    layout["tensor_pages"] = copy.deepcopy(manifest["tensor_pages"])
    layout = seal(layout)
    atomic_json(root / "layout.json", layout)
    manifest["layout_digest"] = layout["digest"]
    for reference in manifest["metadata"]:
        if reference["kind"] == "layout":
            reference["digest"] = layout["digest"]
    manifest = seal(manifest)
    atomic_json(root / "manifest.json", manifest)
    return manifest["digest"]


def test_actual_python_package_loads_native_with_authenticated_chunks(tmp_path):
    root, manifest, _ = fixture(tmp_path)
    result = run_loader(root, manifest["digest"], validate_only=False)
    assert result.returncode == 0, result.stderr.decode()
    assert b'"tensors":1' in result.stdout


@pytest.mark.parametrize("malformation", ["header_shape_same_product", "physical_rank", "offset_overflow",
    "unknown_dtype", "invalid_stride", "self_alias", "chunk_count", "oversized_resident",
    "shard_traversal", "undeclared_field"])
def test_reanchored_malicious_descriptor_is_rejected_by_native_loader(tmp_path, malformation):
    root, manifest, page = fixture(tmp_path)
    tensor, shard = page["tensors"][0], page["shards"][0]
    if malformation == "header_shape_same_product":
        tensor.update(logical_shape=[1, 8], physical_shape=[1, 8], value_row_stride="16")
    elif malformation == "physical_rank":
        tensor["physical_shape"] = [1, 2, 4]
    elif malformation == "offset_overflow":
        tensor["offset"] = "18446744073709551615"
    elif malformation == "unknown_dtype":
        tensor["dtype"] = "EXECUTABLE"
    elif malformation == "invalid_stride":
        tensor["value_row_stride"] = "7"
    elif malformation == "self_alias":
        tensor["alias_of"] = tensor["id"]
    elif malformation == "chunk_count":
        shard["chunk_hashes"].pop()
    elif malformation == "oversized_resident":
        shard["length"] = "18446744073709551615"
    elif malformation == "shard_traversal":
        shard["path"] = "../input.safetensors"
    else:
        tensor["executable"] = True
    digest = reanchor(root, manifest, page)
    result = run_loader(root, digest)
    assert result.returncode == 1 and result.stderr


@pytest.mark.parametrize("malformation", ["symlink", "truncation", "chunk_hash", "tensor_hash", "bad_root_hash"])
def test_native_artifact_file_and_identity_failures(tmp_path, malformation):
    root, manifest, page = fixture(tmp_path)
    shard = page["shards"][0]
    path = root / shard["path"]
    digest = manifest["digest"]
    if malformation == "symlink":
        actual = path.with_suffix(".original")
        path.rename(actual)
        path.symlink_to(actual)
    elif malformation == "truncation":
        path.write_bytes(path.read_bytes()[:-1])
    elif malformation == "chunk_hash":
        shard["chunk_hashes"][0] = "0" * 64
        digest = reanchor(root, manifest, page)
    elif malformation == "tensor_hash":
        page["tensors"][0]["sha256"] = hashlib.sha256(b"wrong tensor").hexdigest()
        digest = reanchor(root, manifest, page)
    else:
        digest = "0" * 64
    result = run_loader(root, digest)
    assert result.returncode == 1 and result.stderr
