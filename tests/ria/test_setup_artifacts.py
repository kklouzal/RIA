"""Owned setup publications over real synthetic preparer/compact-extractor paths."""

from copy import deepcopy
import hashlib
import os
from pathlib import Path
import stat
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from ria import setup_artifacts as artifacts
from ria.client import _inventory
from ria.identity import ArtifactError, atomic_json, canonical, read_json, seal
from ria.preparation import prepare, verify_package
from test_artifacts import fixture_recipe, source_file


@pytest.fixture(autouse=True)
def runtime_identity(monkeypatch):
    # CI is model/GPU-free and unprivileged; the production contract remains
    # UID/GID10001. Exercise the same inode operations as the fixture's UID/GID.
    monkeypatch.setattr(artifacts, "RUNTIME_UID", os.geteuid())
    monkeypatch.setattr(artifacts, "RUNTIME_GID", os.getegid())


def model_recipe(root, *, cache=False, tokenizer=None):
    recipe = fixture_recipe(root)
    token = root / "tokenizer.json"
    token.write_bytes(canonical(tokenizer or {"model": {"vocab": {"a": 0, "Ġabc": 1}},
                                             "added_tokens": [{"id": 2, "content": "<é>"}]}))
    result = {**recipe, "tokenizer_digest": hashlib.sha256(token.read_bytes()).hexdigest(),
              "metadata": [{"path": token.name, "sha256": hashlib.sha256(token.read_bytes()).hexdigest()}]}
    if cache:
        tensors = {"layers.0.ffn.experts.0.w1.weight": ("BF16", [2, 4], b"\0" * 16)}
        for projection, operation in (("w1", "expert_gate"), ("w2", "expert_down"), ("w3", "expert_up")):
            name = "layers.0.ffn.experts.1." + projection + ".weight"
            tensors[name] = ("BF16", [2, 4], b"\0" * 16)
            result["tensors"] = [*result["tensors"], {"name": name, "operation": operation, "placement": "cache",
                "format": "bf16", "logical_shape": [2, 4], "scale_names": [], "grant_group": "offline-cache", "convert": False}]
        path = source_file(root / "input.safetensors", tensors)
        result["sources"] = [{"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]
    return seal(result)


def settings(package, manifest, *, local=False):
    return {"profile": "bf16", "server_executor": "cpu", "security": {"mode": "tls"},
            "planning": {"context_positions": 17, "prefill_rows": 3},
            "model": {"mode": "prepared", "package_dir": str(package), "trusted_manifest_digest": manifest["digest"],
                      "chunk_size": 64, "max_shard_bytes": 1 << 20},
            "client_runtime": {"runtime": {"tokenizer_memory_bytes": "8000000", "host_state_bytes": "1024",
                "device_state_bytes": "1024", "frontend_host_bytes": "2048", "projection_tile_rows": 5,
                "state_tile_rows": 7, "max_image_patches": 11}, "host_expert_cache_bytes": 4096,
                "device_expert_cache_bytes": 0, "engram_cache_bytes": 0,
                "local_experts": [{"layer": 0, "expert": 1, "tier": "host", "local_phases": ["decode"]}] if local else []}}


def prepared(tmp_path, *, cache=False, tokenizer=None):
    source = tmp_path / "source"
    source.mkdir()
    recipe = model_recipe(source, cache=cache, tokenizer=tokenizer)
    package = tmp_path / "provided-server"
    return source, recipe, package, prepare(source, recipe, package)


def native_metadata_fixture(tmp_path, monkeypatch, *, tokenizer=None):
    directory = tmp_path / "packaged-native"
    directory.mkdir()
    payloads = {name: b"synthetic locked native metadata: " + name.encode() for name in artifacts.NATIVE_METADATA}
    if tokenizer is not None:
        payloads["tokenizer.json"] = tokenizer
    entries = []
    for name, data in payloads.items():
        (directory / name).write_bytes(data)
        entries.append({"repository": artifacts.MODEL, "revision": artifacts.SOURCE_REVISION, "path": name,
            "local_path": f"locks/metadata/{artifacts.MODEL}/{artifacts.SOURCE_REVISION}/{name}",
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    lock = tmp_path / "synthetic-source-lock.json"
    atomic_json(lock, seal({"classification": "synthetic offline metadata boundary", "references": entries}))
    monkeypatch.setattr(artifacts, "SOURCE_LOCK_FILE", lock)
    monkeypatch.setattr(artifacts, "_locked_metadata", lambda repository, revision, name: directory / name)
    return payloads, entries, directory


def test_prepared_server_preserved_compact_export_and_idempotence(tmp_path):
    _, _, package, manifest = prepared(tmp_path, cache=True)
    before = {str(path.relative_to(package)): (path.stat().st_mode, path.stat().st_uid, path.stat().st_gid)
              for path in [package, *package.rglob("*")]}
    config = settings(package, manifest)
    workspace = tmp_path / "work"
    facts = artifacts.prepare_expert(config, workspace)
    assert facts == artifacts.prepare_expert(config, workspace)
    assert facts["server_root"] == str(package)
    assert facts["export_root"] == facts["client_root"] != facts["server_root"]
    assert facts["max_token_bytes"] == 4
    assert facts["selected_tensors"] == []
    tensors, _ = _inventory(Path(facts["client_root"]), facts["client_manifest"])
    assert set(tensors) == {"layers.0.ffn.experts.0.w1.weight"}
    assert before == {str(path.relative_to(package)): (path.stat().st_mode, path.stat().st_uid, path.stat().st_gid)
                      for path in [package, *package.rglob("*")]}
    assert artifacts.verify_client_import(facts["client_root"], facts) == facts["client_manifest"]
    assert stat.S_IMODE(Path(facts["client_root"]).stat().st_mode) == 0o750
    assert stat.S_IMODE((Path(facts["client_root"]) / "manifest.json").stat().st_mode) == 0o640
    assert stat.S_IMODE(workspace.stat().st_mode) == 0o700


def test_source_uses_real_preparation_and_owned_resumable_roots(tmp_path, monkeypatch):
    source, recipe, _, _ = prepared(tmp_path)
    native_metadata_fixture(tmp_path, monkeypatch, tokenizer=(source / "tokenizer.json").read_bytes())
    monkeypatch.setattr(artifacts, "create_recipe", lambda *args, **kwargs: recipe)
    config = settings(tmp_path / "unused", {"digest": "0" * 64})
    config["model"] = {"mode": "source", "source_dir": str(source)}
    facts = artifacts.prepare_expert(config, tmp_path / "work")
    assert verify_package(facts["server_root"]) == facts["server_manifest"]
    assert facts == artifacts.prepare_expert(config, tmp_path / "work")
    root = Path(facts["server_root"])
    assert stat.S_IMODE((root / ".prepare-state.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((source / ".ria-recipes").stat().st_mode) == 0o700
    marker = root.parent / (root.name + "-owner.json")
    value = read_json(marker)
    assert value["inode"] == str(root.stat().st_ino)


def test_missing_pinned_native_source_metadata_is_private_exact_and_idempotent(tmp_path, monkeypatch):
    payloads, entries, _ = native_metadata_fixture(tmp_path, monkeypatch)
    source = artifacts.private_directory(tmp_path / "source")
    unrelated = source / "operator-checkpoint.safetensors"
    unrelated.write_bytes(b"synthetic unrelated input must remain untouched")
    refs = artifacts.provision_source_metadata(source)
    assert refs == [{"path": item["path"], "sha256": item["sha256"]} for item in entries]
    states = {}
    for name, data in payloads.items():
        path = source / name
        assert path.read_bytes() == data and stat.S_IMODE(path.stat().st_mode) == 0o600
        states[name] = path.stat()
    assert artifacts.provision_source_metadata(source) == refs
    assert all((source / name).stat() == state for name, state in states.items())
    assert unrelated.read_bytes() == b"synthetic unrelated input must remain untouched"
    assert {path.name for path in source.iterdir()} == {*payloads, unrelated.name, ".ria-recipes"}
    assert not list((source / ".ria-recipes").iterdir())


def test_source_metadata_actual_reviewed_originals_without_weight_inputs(tmp_path):
    source = artifacts.private_directory(tmp_path / "source")
    references = artifacts.provision_source_metadata(source)
    assert {path.name for path in source.iterdir()} == {*artifacts.NATIVE_METADATA, ".ria-recipes"}
    for item in references:
        assert hashlib.sha256((source / item["path"]).read_bytes()).hexdigest() == item["sha256"]
    assert hashlib.sha1(b"blob 801\0" + (source / "tokenizer_config.json").read_bytes()).hexdigest() == "f3dad388a2bbfd6a8605bd02754acd86d9ca5112"


def test_native_source_existing_originals_preserve_inode_mode_and_bytes(tmp_path, monkeypatch):
    payloads, _, _ = native_metadata_fixture(tmp_path, monkeypatch)
    source = artifacts.private_directory(tmp_path / "source")
    states = {}
    for name, data in payloads.items():
        path = source / name
        path.write_bytes(data)
        path.chmod(0o640)
        states[name] = path.stat()
    artifacts.provision_source_metadata(source)
    assert all((source / name).stat() == state for name, state in states.items())


def test_conflicting_native_source_metadata_is_named_and_never_overwritten(tmp_path, monkeypatch):
    payloads, _, _ = native_metadata_fixture(tmp_path, monkeypatch)
    source = artifacts.private_directory(tmp_path / "source")
    conflict = source / artifacts.NATIVE_METADATA[-1]
    conflict.write_bytes(b"operator must explicitly preserve or replace this conflict")
    before = conflict.read_bytes(), conflict.stat()
    with pytest.raises(ArtifactError, match=conflict.name):
        artifacts.provision_source_metadata(source)
    assert (conflict.read_bytes(), conflict.stat()) == before
    assert {path.name for path in source.iterdir()} == {conflict.name}
    assert conflict.name in payloads


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo", "directory"])
def test_native_source_metadata_rejects_links_and_specials_before_publication(tmp_path, monkeypatch, kind):
    payloads, _, _ = native_metadata_fixture(tmp_path, monkeypatch)
    source = artifacts.private_directory(tmp_path / "source")
    name = artifacts.NATIVE_METADATA[-1]
    outside = tmp_path / "outside"
    outside.write_bytes(payloads[name])
    before = outside.read_bytes(), outside.stat().st_mode
    path = source / name
    if kind == "symlink":
        path.symlink_to(outside)
    elif kind == "hardlink":
        os.link(outside, path)
    elif kind == "fifo":
        os.mkfifo(path)
    else:
        path.mkdir()
    with pytest.raises(ArtifactError, match=name):
        artifacts.provision_source_metadata(source)
    assert (outside.read_bytes(), outside.stat().st_mode) == before
    assert {entry.name for entry in source.iterdir()} == {name}


def test_packaged_native_metadata_tamper_prevents_any_source_publication(tmp_path, monkeypatch):
    _, _, directory = native_metadata_fixture(tmp_path, monkeypatch)
    source = artifacts.private_directory(tmp_path / "source")
    path = directory / artifacts.NATIVE_METADATA[-1]
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ArtifactError, match="source hash mismatch"):
        artifacts.provision_source_metadata(source)
    assert not list(source.iterdir())


def test_source_metadata_publication_race_preserves_competing_input(tmp_path, monkeypatch):
    native_metadata_fixture(tmp_path, monkeypatch)
    source = artifacts.private_directory(tmp_path / "source")
    link = artifacts.os.link
    def competing_link(src, destination, **kwargs):
        if destination == artifacts.NATIVE_METADATA[0]:
            (source / destination).write_bytes(b"competing input must not be replaced")
        return link(src, destination, **kwargs)
    monkeypatch.setattr(artifacts.os, "link", competing_link)
    with pytest.raises(ArtifactError, match=artifacts.NATIVE_METADATA[0]):
        artifacts.provision_source_metadata(source)
    assert (source / artifacts.NATIVE_METADATA[0]).read_bytes() == b"competing input must not be replaced"
    assert {entry.name for entry in source.iterdir()} == {artifacts.NATIVE_METADATA[0], ".ria-recipes"}
    assert not list((source / ".ria-recipes").iterdir())


def test_native_metadata_publication_holds_authorized_source_inode(tmp_path, monkeypatch):
    payloads, _, _ = native_metadata_fixture(tmp_path, monkeypatch)
    source = artifacts.private_directory(tmp_path / "source")
    outside = artifacts.private_directory(tmp_path / "outside")
    original = tmp_path / "authorized-source-inode"
    publish = artifacts._publish_source_metadata
    changed = False
    def substitute_path(directory, name, data, **kwargs):
        nonlocal changed
        if not changed:
            changed = True
            source.rename(original)
            source.symlink_to(outside, target_is_directory=True)
        return publish(directory, name, data, **kwargs)
    monkeypatch.setattr(artifacts, "_publish_source_metadata", substitute_path)
    artifacts.provision_source_metadata(source)
    assert not list(outside.iterdir())
    assert all((original / name).read_bytes() == data for name, data in payloads.items())


def test_never_claim_preexisting_tree_or_substituted_inode(tmp_path):
    area = artifacts.private_directory(tmp_path / "area")
    target = area / "target"
    target.mkdir(mode=0o700)
    (target / "unrelated").write_bytes(b"keep")
    with pytest.raises(ArtifactError, match="owned-inode marker"):
        artifacts._marked_directory(area, "target", "a" * 64)
    assert (target / "unrelated").read_bytes() == b"keep"
    owned = artifacts._marked_directory(area, "owned", "b" * 64)
    owned.rename(area / "original")
    owned.mkdir(mode=0o700)
    with pytest.raises(ArtifactError, match="ownership/input marker"):
        artifacts._marked_directory(area, "owned", "b" * 64)


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo"])
def test_permission_preflight_rejects_special_targets_before_changes(tmp_path, kind):
    root = artifacts.private_directory(tmp_path / "model")
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")
    outside_mode = outside.stat().st_mode
    safe = root / "a-safe"
    safe.write_bytes(b"safe")
    mode = safe.stat().st_mode
    unsafe = root / "z-unsafe"
    if kind == "symlink":
        unsafe.symlink_to(outside)
    elif kind == "hardlink":
        os.link(outside, unsafe)
    else:
        os.mkfifo(unsafe)
    with pytest.raises(ArtifactError, match="unsafe permission target"):
        artifacts.runtime_readable(root)
    assert safe.stat().st_mode == mode
    assert outside.read_bytes() == b"outside" and outside.stat().st_mode == outside_mode


def test_local_complete_cache_and_placement_preserve_explicit_pools(tmp_path):
    _, _, package, manifest = prepared(tmp_path, cache=True)
    config = settings(package, manifest, local=True)
    facts = artifacts.prepare_expert(config, tmp_path / "work")
    plan = artifacts.make_placement(config, facts["client_manifest"], facts, tmp_path / "work")
    assert len(facts["selected_tensors"]) == 3
    assert plan["expert_policy"] == "explicit" and plan["local_experts"] == config["client_runtime"]["local_experts"]
    for field, value in config["client_runtime"]["runtime"].items():
        assert plan["runtime"][field] == value
    assert plan["runtime"]["prefill_rows"] == 3 and config["planning"]["context_positions"] == 17
    changed = deepcopy(config)
    changed["client_runtime"]["local_experts"] = []
    with pytest.raises(ArtifactError, match="compact selection"):
        artifacts.make_placement(changed, facts["client_manifest"], facts, tmp_path / "other")
    changed = deepcopy(config)
    changed["client_runtime"]["local_experts"] *= 2
    with pytest.raises(ArtifactError, match="sorted and unique"):
        artifacts.make_placement(changed, facts["client_manifest"], facts, tmp_path / "other")


def test_remote_placement_and_exact_transport_grants(tmp_path):
    _, _, package, manifest = prepared(tmp_path)
    config = settings(package, manifest)
    workspace = tmp_path / "work"
    facts = artifacts.prepare_expert(config, workspace)
    placement = artifacts.make_placement(config, facts["client_manifest"], facts, workspace)
    assert placement["expert_policy"] == "remote" and not placement["local_experts"]
    clientfacts = seal({**facts, "placement_plan_digest": placement["digest"]})
    class Security:
        tls_enabled = True
        def peer_name(self, role):
            assert role == "client"
            return "ria-client-job"
    security = Security()
    grants = artifacts.make_grants(config, manifest, clientfacts, security, workspace)
    grant = grants["grants"][0]
    assert grant["expected_peer_name"] == "ria-client-job"
    assert grant["server_layout_digest"] == manifest["layout_digest"]
    assert grant["client_layout_digest"] == facts["client_manifest"]["layout_digest"]
    assert grant["placement_plan_digest"] == placement["digest"]
    config["security"]["mode"] = "trusted_network"
    with pytest.raises(ArtifactError, match="selected transport"):
        artifacts.make_grants(config, manifest, clientfacts, security, workspace)
    security.tls_enabled = False
    security.peer_name = lambda role: None
    plain = artifacts.make_grants(config, manifest, clientfacts, security, tmp_path / "plain")
    assert plain["grants"][0]["expected_peer_name"] is None


@pytest.mark.parametrize("field", ["server_layout_digest", "client_layout_digest", "tokenizer_sha256", "encoding_digest"])
def test_self_sealed_inconsistent_facts_rejected(tmp_path, field):
    _, _, package, manifest = prepared(tmp_path)
    facts = artifacts.prepare_expert(settings(package, manifest), tmp_path / "work")
    bad = seal({**facts, field: "f" * 64})
    with pytest.raises(ArtifactError, match="differs"):
        artifacts.verify_client_import(facts["client_root"], bad)


@pytest.mark.parametrize("tokenizer", [{"model": None, "added_tokens": []},
    {"model": {"vocab": {"a": True}}, "added_tokens": []},
    {"model": {"vocab": {"a": 0}}, "added_tokens": [{"id": 0, "content": "different"}]},
    {"model": {"vocab": {"💥": 0}}, "added_tokens": []}])
def test_invalid_tokenizer_metadata_stops_before_facts(tmp_path, tokenizer):
    _, _, package, manifest = prepared(tmp_path, tokenizer=tokenizer)
    with pytest.raises(ArtifactError, match="token"):
        artifacts.prepare_expert(settings(package, manifest), tmp_path / "work")
    assert not list((tmp_path / "work" / "models").glob("*-facts.json"))


def test_token_generation_private_restart_and_existing_copy(tmp_path):
    workspace = tmp_path / "work"
    path = Path(artifacts.provision_api_token(workspace))
    first = path.read_bytes()
    assert len(first.strip()) >= 32 and stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o750
    assert artifacts.provision_api_token(workspace) == str(path) and path.read_bytes() == first
    supplied = tmp_path / "provided.token"
    supplied.write_bytes(b"operator-secret\n")
    supplied.chmod(0o600)
    copied = Path(artifacts.provision_api_token(tmp_path / "other", supplied))
    assert copied.read_bytes() == supplied.read_bytes() == b"operator-secret\n"
    assert stat.S_IMODE(supplied.stat().st_mode) == 0o600
    with pytest.raises(ArtifactError, match="rotation"):
        artifacts.provision_api_token(workspace, supplied)
    assert path.read_bytes() == first


@pytest.mark.parametrize("kind", ["symlink", "fifo", "directory", "public", "empty", "large", "space", "nul", "hardlink"])
def test_token_boundary_rejects_unsafe_input_without_secret_publication(tmp_path, kind):
    source = tmp_path / "token"
    if kind == "symlink":
        target = tmp_path / "outside"
        target.write_bytes(b"private")
        source.symlink_to(target)
    elif kind == "fifo":
        os.mkfifo(source, 0o600)
    elif kind == "directory":
        source.mkdir(mode=0o700)
    else:
        source.write_bytes({"empty": b"", "large": b"a" * 4097, "space": b"a b", "nul": b"a\0b"}.get(kind, b"secret"))
        source.chmod(0o644 if kind == "public" else 0o600)
        if kind == "hardlink":
            os.link(source, tmp_path / "alias")
    with pytest.raises(ArtifactError):
        artifacts.provision_api_token(tmp_path / "work", source)
    assert not (tmp_path / "work" / "secrets" / "api.token").exists()


def test_owned_runtime_reports_and_private_directory_boundaries(tmp_path):
    reports = artifacts.private_directory(tmp_path / "reports")
    assert artifacts.runtime_reports(reports) == str(reports)
    assert artifacts.runtime_reports(reports) == str(reports)
    public = tmp_path / "public"
    public.mkdir(mode=0o755)
    with pytest.raises(ArtifactError, match="mode0700"):
        artifacts.private_directory(public)
    link = tmp_path / "link"
    link.symlink_to(public, target_is_directory=True)
    with pytest.raises(ArtifactError):
        artifacts.private_directory(link)


def test_public_runtime_directories_marked_reusable_and_independent(tmp_path):
    workspace = tmp_path / "work"
    secret = Path(artifacts.runtime_secret_directory(workspace))
    reports = Path(artifacts.runtime_report_directory(workspace))
    assert secret == workspace / "secrets" and reports == workspace / "reports"
    assert stat.S_IMODE(secret.stat().st_mode) == 0o750
    assert stat.S_IMODE(reports.stat().st_mode) == 0o700
    (reports / "completed.json").write_bytes(b"{}")
    assert artifacts.runtime_report_directory(workspace) == str(reports)
    token = artifacts.provision_api_token(workspace)
    assert artifacts.runtime_secret_directory(workspace) == str(Path(token).parent)


def test_peer_selected_local_experts_override_expert_only_settings(tmp_path):
    _, _, package, manifest = prepared(tmp_path, cache=True)
    config = settings(package, manifest, local=True)
    local = config.pop("client_runtime")["local_experts"]
    facts = artifacts.prepare_expert(config, tmp_path / "work", local_experts=local)
    assert len(facts["selected_tensors"]) == 3
    assert "client_runtime" not in config
    with pytest.raises(ArtifactError, match="local-expert selection"):
        artifacts.prepare_expert(config, tmp_path / "other", local_experts=[{"layer": True}])
