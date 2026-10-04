"""Synthetic setup inputs/Engine/proc fixtures; never Docker, GPU or a model."""

import copy
import hashlib
import os
from pathlib import Path
import socket
import stat
import struct
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from ria import host, setup_config, setup_host
from ria.host import cgroup_ancestors
from ria.identity import ArtifactError, atomic_json, canonical, read_json, seal
from ria.setup_config import build_request, template_settings, validate_settings, workspace_paths
from ria.setup_host import discover_local
from ria.setup_limits import MAX_SETUP_SECONDS, MAX_TRANSFER_SECONDS, MAX_TREE
from test_deployment import deployment_request


def settings_fixture(tmp_path, role="expert", executor="cpu"):
    request = deployment_request(tmp_path, role, executor, tls_enabled=False)
    planning = {key: copy.deepcopy(value) for key, value in request["planning_request"].items()
                if key not in ("schema_revision", "role", "executor", "profile", "logical_model_digest", "operator_contract_digest")}
    settings = {"schema_revision": 1, "role": role, "executor": executor, "profile": "bf16",
        "workspace": str(tmp_path), "service_image": request["environment"]["image"], "bind_address": "192.168.10.2",
        "server_executor": executor if role == "expert" else "cpu", "planning": planning,
        "environment": {key: request["environment"][key] for key in ("cpuset", "cgroup_bytes", "memlock_bytes", "pids_limit",
            "start_period_seconds", "stop_grace_seconds", "api_port")},
        "probe": {key: request["probe_config"][key] for key in (
            "max_host_test_bytes", "max_device_test_bytes", "max_pinned_test_bytes", "deadline_ms")},
        "network": {key: value for key, value in request["network"].items()
            if key not in ("control_address", "bulk_address", "server_executor")},
        "deadline_ms": 30000, "max_transfer_bytes": 1 << 30, "transfer_deadline_ms": 30000, "setup_deadline_ms": 60000,
        "security": {"mode": "trusted_network"}}
    if executor == "cuda":
        settings["gpu_uuid"] = request["environment"]["gpu_uuid"]
    manifest = read_json(tmp_path / "model/manifest.json")
    if role == "expert":
        settings["environment"]["cpuset"] = "0"
        settings["model"] = {"mode": "prepared", "package_dir": str(tmp_path / "model"), "trusted_manifest_digest": manifest["digest"]}
        settings["expert"] = request["expert"]
    else:
        manifest = seal({**manifest, "role": "client"})
        atomic_json(tmp_path / "model/manifest.json", manifest)
        settings["client_runtime"] = {"runtime": {"tokenizer_memory_bytes": "65536", "host_state_bytes": str(8 << 20),
            "device_state_bytes": str(4 << 20), "frontend_host_bytes": str(4 << 20),
            "projection_tile_rows": 64, "state_tile_rows": 8, "max_image_patches": 16},
            "host_expert_cache_bytes": 0, "device_expert_cache_bytes": 0, "engram_cache_bytes": 65536, "local_experts": []}
        settings["api"] = {key: value for key, value in request["api"].items() if key not in setup_config.API_CONSTANTS}
    return settings, manifest, request


def host_fixture(tmp_path, monkeypatch, role="expert", executor="cpu"):
    settings, manifest, previous = settings_fixture(tmp_path, role, executor)
    project = tmp_path / "project"
    for directory in ("locks", "deploy", "bin"):
        (project / directory).mkdir(parents=True)
    lock = seal({"schema_revision": 1, "classification": "synthetic offline source lock"})
    atomic_json(project / "locks/source-lock.json", lock)
    native = b"synthetic CPU native admission executable"
    (project / "bin/ds4ctl").write_bytes(native)
    atomic_json(project / "deploy/seccomp-numa.json", {"defaultAction": "SCMP_ACT_ERRNO", "syscalls": []})
    real_lock = read_json(Path(__file__).resolve().parents[2] / "deploy/container-lock.json")
    atomic_json(project / "deploy/container-lock.json", real_lock)
    monkeypatch.setattr(setup_config, "PROJECT_ROOT", project)
    monkeypatch.setattr(setup_host, "PROJECT_ROOT", project)
    monkeypatch.setattr(setup_host.os, "geteuid", lambda: 0)
    cpu_binaries = {name: hashlib.sha256(native).hexdigest() for name in ("ds4ctl", "ds4-expert-server", "ds4-ria-qualify")}
    own = seal({"schema_revision": 1, "image_kind": "cpu", "architecture": "amd64", "source_lock_digest": lock["digest"],
        "hardware_qualified": False, "numeric_flags": ["-fno-fast-math", "-ffp-contract=off"],
        "sources": {"Makefile.ria": "1" * 64, "tools/ria/setup_config.py": "2" * 64}, "binaries": cpu_binaries})
    atomic_json(project / "build-info.json", own)
    monkeypatch.setattr(setup_host, "BUILD_INFO", project / "build-info.json")
    build = own if executor == "cpu" else seal({**own, "image_kind": "cuda", "code_targets": ["sm_120a"], "ptx_jit": False,
        "numeric_flags": own["numeric_flags"] + ["--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true"],
        "binaries": {**cpu_binaries, **{name: "3" * 64 for name in ("ds4", "ds4-server", "ds4-eval")}}})
    proc, cgroup = tmp_path / "proc", tmp_path / "cgroup"
    (proc / "self").mkdir(parents=True)
    (proc / "4242").mkdir()
    controller_id, inspection_id = "a" * 64, "b" * 64
    (proc / "self/cgroup").write_text("0::/system.slice/docker-" + controller_id + ".scope\n")
    (proc / "4242/cgroup").write_text("0::/system.slice/docker-" + inspection_id + ".scope\n")
    for directory in (cgroup, cgroup / "system.slice"):
        directory.mkdir(parents=True, exist_ok=True)
        for field, value in {"memory.max": "max", "memory.swap.max": "max", "pids.max": "max",
            "cpuset.cpus.effective": "0-3", "cpuset.mems.effective": "0"}.items():
            (directory / field).write_text(value)
    gpu = settings.get("gpu_uuid")
    report = seal({"schema_revision": 1, "kind": "host_preflight", "architecture": "x86_64", "kernel_version": "fixture",
        "docker_version": "29.8.2", "compose_version": "5.5.1", "cgroup_driver": "systemd", "rootful": True, "cgroup_version": 2,
        "parent_ancestors": cgroup_ancestors(cgroup / "system.slice", root=cgroup),
        "numa": [{"node": 0, "cpus": "0-3", "total_bytes": str(4 << 30), "distance": [10]}],
        "physical_gpus": [], "selected_gpu_uuid": gpu, "client": role == "client", "hardware_qualified": False})
    image_environment = ["PATH=/usr/local/bin:/usr/bin:/bin"]
    if executor == "cuda":
        image_environment += ["CUDA_VERSION=" + real_lock["cuda_version"], "CUDA_DISABLE_PTX_JIT=1", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
            "NVIDIA_REQUIRE_CUDA=" + real_lock["cuda_registry_verification"]["inherited_configuration"]["environment"]["NVIDIA_REQUIRE_CUDA"]]
    image = {"Architecture": "amd64", "Os": "linux", "RepoDigests": [settings["service_image"]],
        "Config": {"User": "10001:10001", "WorkingDir": "/artifacts", "Cmd": ["--config", "/etc/dwarfstar/service.json"],
        "Entrypoint": ["/usr/local/bin/ds4-expert-server" if executor == "cpu" else "/usr/local/bin/ds4-server"], "Env": image_environment}}
    mounts = [{"Type": "bind", "Source": path, "Destination": path, "RW": writable} for path, writable in (
        ("/var/run/docker.sock", True), ("/run/lock", True), (str(tmp_path), True),
        ("/sys/fs/cgroup", False), ("/sys/devices/system/node", False))]
    controller = {"Id": controller_id, "State": {"Running": True, "Pid": 100},
        "HostConfig": {"PidMode": "host", "CgroupnsMode": "host", "NetworkMode": "host", "Privileged": False, "DeviceRequests": []},
        "Config": {"User": "0:0", "Env": []}, "Mounts": mounts}
    if executor == "cuda":
        controller["HostConfig"]["DeviceRequests"] = [{"Driver": "nvidia", "Count": 0, "DeviceIDs": [gpu], "Capabilities": [["gpu"]]}]
        controller["Config"]["Env"] = ["NVIDIA_DRIVER_CAPABILITIES=utility"]
    unix = socket.socket(socket.AF_UNIX)
    unix.bind(str(tmp_path / "engine.sock"))
    socket_metadata = {"st_uid": 0, "st_gid": 999}
    class SocketNode:
        def __fspath__(self):
            return str(tmp_path / "engine.sock")
        def stat(self, *, follow_symlinks):
            observed = Path(self).stat(follow_symlinks=follow_symlinks)
            fields = ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink", "st_ctime_ns")
            return SimpleNamespace(**{**{name: getattr(observed, name) for name in fields}, **socket_metadata})
    monkeypatch.setattr(setup_host, "DOCKER_SOCKET", SocketNode())
    calls, temporary = [], {}
    def runner(arguments, deadline, **kwargs):
        calls.append(arguments)
        assert type(deadline) is int and 0 < deadline <= settings["deadline_ms"]
        if arguments[3:5] == ["image", "inspect"]:
            return canonical(image)
        if arguments[3] == "inspect":
            return canonical(controller if arguments[-1] == controller_id else temporary)
        if arguments[3] == "run":
            assert "--runtime=runc" in arguments and not any(value.startswith("--gpus") for value in arguments)
            assert "--network=none" in arguments and "--read-only" in arguments
            if "/bin/cat" in arguments:
                return canonical(build)
            name = arguments[arguments.index("--name") + 1]
            label = arguments[arguments.index("--label") + 1].split("=", 1)[1]
            temporary.update(Id=inspection_id, Name="/" + name, State={"Running": True, "Pid": 4242},
                Config={"Image": settings["service_image"], "Labels": {"io.ria.setup.inspection": label}})
            return inspection_id.encode() + b"\n"
        if arguments[3] in ("pull", "rm"):
            return b""
        raise AssertionError(arguments)
    def observer(parent, requested_gpu, *, client, discover_gpu=False):
        assert parent == str(cgroup / "system.slice") and requested_gpu == gpu and client is (role == "client")
        assert discover_gpu is (executor == "cuda")
        return report
    return {"settings": settings, "manifest": manifest, "previous": previous, "project": project, "build": build, "own": own,
        "proc": proc, "cgroup": cgroup, "report": report, "image": image, "controller": controller,
        "runner": runner, "observer": observer, "calls": calls, "temporary": temporary, "socket": unix,
        "socket_metadata": socket_metadata}


def discover(fixture, **kwargs):
    try:
        return discover_local(fixture["settings"], runner=fixture["runner"], proc_root=fixture["proc"],
            cgroup_root=fixture["cgroup"], observer=kwargs.get("observer", fixture["observer"]))
    finally:
        fixture["socket"].close()


@pytest.mark.parametrize(("role", "executor"), [("expert", "cpu"), ("expert", "cuda"), ("client", "cuda")])
def test_normalized_settings_and_owned_discovery(tmp_path, monkeypatch, role, executor):
    fixture = host_fixture(tmp_path, monkeypatch, role, executor)
    original = copy.deepcopy(fixture["settings"])
    facts = discover(fixture)
    assert fixture["settings"] == original
    assert facts["host_report"] == fixture["report"] and facts["build_info"] == fixture["build"]
    assert read_json(facts["host_report_path"]) == fixture["report"]
    assert read_json(facts["build_info_path"]) == fixture["build"]
    assert fixture["calls"][-1][3:5] == ["rm", "--force"]
    assert fixture["calls"][-1][-1] == "b" * 64
    if executor == "cpu":
        assert facts["gpu_uuid"] is None and not any("nvidia-smi" in call for call in fixture["calls"])


@pytest.mark.parametrize("mutation", [
    lambda s: s.update(extra="unreviewed"), lambda s: s.update(executor="cpu", role="client"),
    lambda s: s["planning"].update(prefill_rows=65), lambda s: s["planning"].update(prefill_rows=65, context_positions=32),
    lambda s: s["planning"]["caps"].update(device_bytes=1), lambda s: s.update(gpu_uuid="GPU-12345678-1234-1234-1234-123456789abc"),
    lambda s: s["probe"].update(max_device_test_bytes=1), lambda s: s["probe"].update(max_host_test_bytes=1 << 30),
    lambda s: s["environment"].update(cgroup_bytes=1), lambda s: s["environment"].update(cpuset="65536"),
    lambda s: s["security"].update(expected_peer_name="manual.duplicate"), lambda s: s["security"].update(mode=False),
    lambda s: s.update(bind_address="0.0.0.0"), lambda s: s.update(bind_address="127.0.0.1"),
    lambda s: s.update(bind_address="8.8.8.8"), lambda s: s.update(workspace="/tmp/../escape"),
    lambda s: s.update(workspace="/"), lambda s: s["expert"]["nodes"][0].update(cpus=[1]),
    lambda s: s["expert"].update(host_runtime_bytes="0"), lambda s: s["expert"].update(startup_host_bytes="1"),
    lambda s: s["expert"].update(projection_tile_rows=65),
    lambda s: s["expert"].update(host_runtime_bytes="1", pinned_workspace_bytes="2"),
    lambda s: s["expert"].update(startup_host_bytes=str(int(s["expert"]["nodes"][0]["local_bytes"]) - 1)),
    lambda s: s["network"].update(max_inflight_expert_requests=3), lambda s: s["network"].update(max_frame_payload_bytes=4096),
    lambda s: s.update(max_transfer_bytes=0), lambda s: s.update(max_transfer_bytes=MAX_TREE + 1),
    lambda s: s.update(transfer_deadline_ms=86400001),
    lambda s: s.update(setup_deadline_ms=604800001),
    lambda s: s.update(setup_port=7443), lambda s: s.update(setup_port=7444),
    lambda s: s.update(deadline_ms=True), lambda s: s["planning"]["caps"].update(numa=[]),
])
def test_settings_reject_unreviewed_or_invalid_contracts(tmp_path, mutation):
    settings, _, _ = settings_fixture(tmp_path)
    mutation(settings)
    with pytest.raises(ArtifactError):
        validate_settings(settings)


def test_defaults_canonical_mask_and_source_knobs_do_not_change_budgets(tmp_path):
    settings, _, _ = settings_fixture(tmp_path)
    settings.pop("security")
    settings["model"] = {"mode": "source", "source_dir": str(tmp_path / "source")}
    settings["environment"]["cpuset"] = "0,0"
    normalized = validate_settings(settings)
    assert normalized["environment"]["cpuset"] == "0"
    assert normalized["security"] == {"mode": "tls", "api_token_file": None}
    assert normalized["setup_port"] == 9010
    assert normalized["model"]["scratch_bytes"] == 64 << 20
    assert normalized["planning"] == settings["planning"]
    assert normalized["environment"]["cgroup_bytes"] == settings["environment"]["cgroup_bytes"]


def test_workspace_rejects_ancestor_symlink(tmp_path):
    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ArtifactError, match="symlink"):
        workspace_paths(link / "child")


@pytest.mark.parametrize("path", ["//", "//srv/workspace", "/opt/RIA", "/opt/RIA/workspace", "/opt",
                                  "/opt/ria-setup/work", "/usr/share/dwarfstar", "/var/lib/docker/ria",
                                  "/etc/ria", "/run/ria", "/proc/ria", "/sys/ria", "/dev/ria"])
def test_workspace_cannot_cover_or_mutate_control_paths(path):
    with pytest.raises(ArtifactError):
        workspace_paths(path)


@pytest.mark.parametrize(("role", "executor"), [("expert", "cpu"), ("expert", "cuda"), ("client", "cuda")])
def test_templates_have_authoritative_fields_without_invented_budgets(role, executor):
    template = template_settings(role, executor)
    assert template["schema_revision"] == 1 and template["role"] == role and template["executor"] == executor
    assert template["planning"]["caps"]["host_bytes"] is None
    assert template["planning"]["caps"]["device_bytes"] is None
    assert template["planning"]["context_positions"] is None
    assert template["service_image"] is None and template["workspace"] is None
    assert template["security"] == {"mode": "tls", "api_token_file": None}
    assert template["setup_port"] == 9010 and template["environment"]["api_port"] == 8000
    role_fields = {"model", "expert"} if role == "expert" else {"client_runtime", "api"}
    assert set(template) == set(setup_config.setup_schemas()["setup-settings"]["required"]) | {
        "setup_port", "gpu_uuid", "security"} | role_fields
    with pytest.raises(ArtifactError):
        validate_settings(template)


def test_templates_reject_unsupported_roles():
    with pytest.raises(ArtifactError):
        template_settings("client", "cpu")


def test_public_controller_validation_does_not_pull_or_start(tmp_path, monkeypatch):
    fixture = host_fixture(tmp_path, monkeypatch, executor="cuda")
    original = setup_host._controller_context
    monkeypatch.setattr(setup_host, "_controller_context", lambda settings, runner, **kwargs: original(settings, runner, proc_root=fixture["proc"]))
    try:
        assert setup_host.validate_controller_context(fixture["settings"], fixture["runner"]) == fixture["settings"]["gpu_uuid"]
    finally:
        fixture["socket"].close()
    assert len(fixture["calls"]) == 1 and fixture["calls"][0][3] == "inspect"


@pytest.mark.parametrize("selector", ["index", "count", "all"])
def test_utility_selector_defers_uuid_to_actual_host_discovery(tmp_path, monkeypatch, selector):
    fixture = host_fixture(tmp_path, monkeypatch, executor="cuda")
    gpu = fixture["settings"].pop("gpu_uuid")
    device = fixture["controller"]["HostConfig"]["DeviceRequests"][0]
    device.update(Count=0 if selector == "index" else 1 if selector == "count" else -1,
                  DeviceIDs=["0"] if selector == "index" else [])
    def observer(parent, requested, *, client, discover_gpu):
        assert requested is None and discover_gpu is True and client is False
        return fixture["report"]
    facts = discover(fixture, observer=observer)
    assert facts["gpu_uuid"] == gpu and facts["host_report"]["selected_gpu_uuid"] == gpu


def gpu_observation(tmp_path, monkeypatch, raw):
    node = tmp_path / "nodes" / "node0"
    node.mkdir(parents=True)
    (node / "meminfo").write_text("Node 0 MemTotal: 1048576 kB\n")
    (node / "cpulist").write_text("0-3\n")
    (node / "distance").write_text("10\n")
    monkeypatch.setattr(host, "_host_target", lambda: None)
    monkeypatch.setattr(host, "Path", lambda value: tmp_path / "nodes" if value == "/sys/devices/system/node" else Path(value))
    monkeypatch.setattr(host, "cgroup_ancestors", lambda value: [{"path": value}])
    monkeypatch.setattr(host.platform, "release", lambda: "synthetic")
    calls = []
    def command(arguments):
        calls.append(arguments)
        if arguments[0] == "nvidia-smi":
            return raw
        if "info" in arguments:
            return canonical({"CgroupVersion": "2", "CgroupDriver": "systemd", "Architecture": "x86_64",
                              "SecurityOptions": [], "ServerVersion": "29.8.2"})
        return b"5.5.1\n"
    monkeypatch.setattr(host, "_command", command)
    return calls


GPU_A = "GPU-12345678-1234-1234-1234-123456789abc"
GPU_B = "GPU-22345678-1234-1234-1234-123456789abc"


@pytest.mark.parametrize("client", [False, True])
def test_canonical_gpu_autoselection_queries_once_and_revalidation_keeps_identity(tmp_path, monkeypatch, client):
    raw = f"{GPU_A}, NVIDIA GeForce RTX 5090, 12.0, 32607, 590.10\n".encode()
    calls = gpu_observation(tmp_path, monkeypatch, raw)
    observed = host.observe_host("/synthetic-parent", client=client, discover_gpu=True)
    assert observed["selected_gpu_uuid"] == GPU_A and len(observed["physical_gpus"]) == 1
    assert sum(call[0] == "nvidia-smi" for call in calls) == 1
    calls.clear()
    replay = host.observe_host("/synthetic-parent", GPU_A, client=client)
    assert replay == observed and sum(call[0] == "nvidia-smi" for call in calls) == 1


def test_cpu_canonical_observation_never_queries_gpu(tmp_path, monkeypatch):
    calls = gpu_observation(tmp_path, monkeypatch, b"not a GPU response")
    assert host.observe_host("/synthetic-parent")["physical_gpus"] == []
    assert all(call[0] != "nvidia-smi" for call in calls)


@pytest.mark.parametrize(("raw", "client"), [
    (f"{GPU_A}, NVIDIA GeForce RTX 5090, 12.0, 32607, 590.10\n{GPU_B}, NVIDIA GeForce RTX 5090, 12.0, 32607, 590.10\n", True),
    (f"{GPU_A}, NVIDIA Other Target, 12.0, 32607, 590.10\n", True),
    (f"{GPU_A}, NVIDIA GeForce RTX 5090, 10.0, 32607, 590.10\n", False),
    ("", False), ("malformed\n", False),
    (f"{GPU_A}, NVIDIA GeForce RTX 5090, 12.0, 32607, 590.10\n{GPU_A}, NVIDIA GeForce RTX 5090, 12.0, 32607, 590.10\n", False),
])
def test_canonical_gpu_ambiguity_unsupported_or_malformed_inventory_rejected(tmp_path, monkeypatch, raw, client):
    calls = gpu_observation(tmp_path, monkeypatch, raw.encode())
    with pytest.raises(ArtifactError):
        host.observe_host("/synthetic-parent", client=client, discover_gpu=True)
    assert sum(call[0] == "nvidia-smi" for call in calls) == 1


def test_explicit_uuid_selects_one_compatible_visible_device(tmp_path, monkeypatch):
    raw = f"{GPU_A}, NVIDIA GeForce RTX 5090, 12.0, 32607, 590.10\n{GPU_B}, NVIDIA GeForce RTX 5090, 12.0, 32607, 590.10\n".encode()
    calls = gpu_observation(tmp_path, monkeypatch, raw)
    observed = host.observe_host("/synthetic-parent", GPU_B, client=True, discover_gpu=True)
    assert observed["selected_gpu_uuid"] == GPU_B
    assert sum(call[0] == "nvidia-smi" for call in calls) == 1
    with pytest.raises(ArtifactError, match="admitted UUID"):
        host.observe_host("/synthetic-parent", "GPU-32345678-1234-1234-1234-123456789abc", client=True, discover_gpu=True)


def test_ambiguous_gpu_discovery_cleans_owned_metadata_container(tmp_path, monkeypatch):
    fixture = host_fixture(tmp_path, monkeypatch, executor="cuda")
    fixture["settings"].pop("gpu_uuid")
    fixture["controller"]["HostConfig"]["DeviceRequests"][0].update(Count=-1, DeviceIDs=[])
    def observer(*args, **kwargs):
        raise ArtifactError("GPU discovery requires exactly one compatible visible device")
    with pytest.raises(ArtifactError, match="exactly one compatible"):
        discover(fixture, observer=observer)
    assert fixture["calls"][-1][3:] == ["rm", "--force", "b" * 64]
    assert not Path(workspace_paths(tmp_path)["host_report"]).exists()


def test_download_and_long_transfer_are_explicit_and_bounded(tmp_path):
    settings, _, _ = settings_fixture(tmp_path)
    settings["model"] = {"mode": "download", "max_download_bytes": 600 << 30, "download_deadline_ms": 604800000}
    settings["transfer_deadline_ms"] = 86400000
    settings["setup_deadline_ms"] = 604800000
    settings["max_transfer_bytes"] = MAX_TREE
    normalized = validate_settings(settings)
    assert normalized["model"]["hf_token_file"] is None
    assert normalized["model"]["chunk_size"] == 4 << 20
    assert normalized["deadline_ms"] == 30000
    assert normalized["transfer_deadline_ms"] == 86400000
    assert normalized["setup_deadline_ms"] == 604800000
    assert normalized["max_transfer_bytes"] == MAX_TREE
    assert normalized["transfer_deadline_ms"] == MAX_TRANSFER_SECONDS * 1000
    assert normalized["setup_deadline_ms"] == MAX_SETUP_SECONDS * 1000
    settings["model"]["download_deadline_ms"] += 1
    with pytest.raises(ArtifactError):
        validate_settings(settings)


@pytest.mark.parametrize("mutation", [
    lambda f: f["controller"]["HostConfig"].update(PidMode=""),
    lambda f: f["controller"]["HostConfig"].update(CgroupnsMode="private"),
    lambda f: f["controller"]["HostConfig"].update(NetworkMode="bridge"),
    lambda f: f["controller"]["HostConfig"].update(Privileged=True),
    lambda f: f["controller"]["Config"].update(User="10001:10001"),
    lambda f: f["controller"]["Mounts"][-1].update(RW=True),
    lambda f: f["controller"]["Mounts"][2].update(Source="/other/workspace"),
    lambda f: f["controller"]["HostConfig"].update(DeviceRequests=[{"Count": 1}]),
    lambda f: f["controller"]["Config"].update(Env=["PATH=/bin", "PATH=/other"]),
    lambda f: f["controller"]["Config"].update(Env=["CUDA_VERSION=13.4.2"]),
    lambda f: f["controller"].update(HostConfig=None),
    lambda f: f["controller"]["Mounts"].append({"Type": "bind", "Source": "/sys/devices/system/node/node0",
        "Destination": "/sys/devices/system/node/node0", "RW": True}),
    lambda f: f["controller"]["Mounts"].append({"Type": "bind", "Source": "/unrelated/topology",
        "Destination": "/sys/devices/system/node/node0", "RW": False}),
    lambda f: f["controller"]["Mounts"].append({"Type": "bind", "Source": "/unrelated/operator",
        "Destination": f["settings"]["workspace"] + "/operator", "RW": True}),
    lambda f: f["controller"]["Mounts"].append({"Type": "bind", "Source": "/unrelated/lock",
        "Destination": "/run/lock/ria-expert.setup.lock", "RW": True}),
])
def test_controller_authority_rejected_before_mutation(tmp_path, monkeypatch, mutation):
    fixture = host_fixture(tmp_path, monkeypatch)
    mutation(fixture)
    with pytest.raises(ArtifactError):
        discover(fixture)
    assert len(fixture["calls"]) == 1 and fixture["calls"][0][3] == "inspect"


@pytest.mark.parametrize("mutation", [
    lambda f: f["controller"]["HostConfig"]["DeviceRequests"][0].update(Capabilities=[["gpu", "compute"]]),
    lambda f: f["controller"]["HostConfig"]["DeviceRequests"][0].update(Options={"unreviewed": "value"}),
    lambda f: f["controller"]["Config"].update(Env=["NVIDIA_DRIVER_CAPABILITIES=utility", "NVIDIA_DRIVER_CAPABILITIES=compute"]),
    lambda f: f["controller"]["Config"].update(Env=["NVIDIA_DRIVER_CAPABILITIES=compute,utility"]),
])
def test_gpu_controller_utility_contract_rejected_before_mutation(tmp_path, monkeypatch, mutation):
    fixture = host_fixture(tmp_path, monkeypatch, executor="cuda")
    mutation(fixture)
    with pytest.raises(ArtifactError):
        discover(fixture)
    assert len(fixture["calls"]) == 1


@pytest.mark.parametrize("kind", ["source", "prepared", "api", "hf"])
@pytest.mark.parametrize("ancestry", ["parent", "direct"])
def test_external_inputs_accept_only_same_path_host_bind_ancestry(tmp_path, monkeypatch, kind, ancestry):
    fixture = host_fixture(tmp_path, monkeypatch, role="client" if kind == "api" else "expert", executor="cuda" if kind == "api" else "cpu")
    external = tmp_path.parent / (tmp_path.name + "-inputs")
    external.mkdir()
    if kind in ("source", "prepared"):
        target = external / "package"
        target.mkdir()
        if kind == "source":
            fixture["settings"]["model"] = {"mode": "source", "source_dir": str(target)}
        else:
            fixture["settings"]["model"]["package_dir"] = str(target)
    else:
        target = external / "token"
        target.write_text("synthetic private input")
        if kind == "api":
            fixture["settings"]["security"]["api_token_file"] = str(target)
        else:
            fixture["settings"]["model"] = {"mode": "download", "max_download_bytes": 600 << 30,
                "download_deadline_ms": 604800000, "hf_token_file": str(target)}
    bound = target if ancestry == "direct" else external
    fixture["controller"]["Mounts"].append({"Type": "bind", "Source": str(bound), "Destination": str(bound), "RW": kind == "source"})
    facts = discover(fixture)
    assert facts["service_image"] == fixture["settings"]["service_image"]


@pytest.mark.parametrize("metadata", [
    {"st_uid": 10001}, {"st_uid": 1000}, {"st_mode": stat.S_IFSOCK | 0o666},
    {"st_gid": 10001, "st_mode": stat.S_IFSOCK | 0o660}, {"st_nlink": 2},
    {"st_mode": stat.S_IFLNK | 0o777}, {"st_mode": stat.S_IFIFO | 0o600},
])
def test_worker_writable_aliased_or_invalid_socket_rejected_before_authority(tmp_path, monkeypatch, metadata):
    fixture = host_fixture(tmp_path, monkeypatch)
    fixture["socket_metadata"].update(metadata)
    with pytest.raises(ArtifactError, match="Docker socket"):
        discover(fixture)
    assert not fixture["calls"]


def test_socket_replacement_during_inspection_rejected_before_mutation(tmp_path, monkeypatch):
    fixture = host_fixture(tmp_path, monkeypatch)
    previous = fixture["runner"]
    def runner(*args, **kwargs):
        result = previous(*args, **kwargs)
        fixture["socket_metadata"]["st_ino"] = 123456
        return result
    fixture["runner"] = runner
    with pytest.raises(ArtifactError, match="changed during controller"):
        discover(fixture)
    assert len(fixture["calls"]) == 1


def socket_acl(*entries):
    return struct.pack("<I", 2) + b"".join(struct.pack("<HHI", *entry) for entry in entries)


@pytest.mark.parametrize(("acl", "rejected"), [
    (socket_acl((1, 6, 0xffffffff), (2, 2, 10001), (4, 0, 0xffffffff), (16, 2, 0xffffffff), (32, 0, 0xffffffff)), True),
    (socket_acl((1, 6, 0xffffffff), (4, 0, 0xffffffff), (8, 2, 10001), (16, 2, 0xffffffff), (32, 0, 0xffffffff)), True),
    (socket_acl((1, 6, 0xffffffff), (2, 4, 10001), (4, 0, 0xffffffff), (16, 6, 0xffffffff), (32, 0, 0xffffffff)), False),
    (socket_acl((1, 6, 0xffffffff), (2, 2, 10001), (4, 0, 0xffffffff), (16, 4, 0xffffffff), (32, 0, 0xffffffff)), False),
    (b"malformed ACL", True),
    (socket_acl((1, 6, 0xffffffff), (2, 2, 10001), (4, 0, 0xffffffff), (32, 0, 0xffffffff)), True),
])
def test_socket_acl_worker_grants_are_checked_before_authority(tmp_path, monkeypatch, acl, rejected):
    fixture = host_fixture(tmp_path, monkeypatch)
    previous = os.getxattr
    def getxattr(path, name, **kwargs):
        return acl if path is setup_host.DOCKER_SOCKET and name == "system.posix_acl_access" else previous(path, name, **kwargs)
    monkeypatch.setattr(setup_host.os, "getxattr", getxattr)
    if rejected:
        with pytest.raises(ArtifactError, match="Docker socket"):
            discover(fixture)
        assert not fixture["calls"]
    else:
        assert discover(fixture)["kind"] == "setup_host_facts"


@pytest.mark.parametrize("change", ["unmounted", "foreign", "source_readonly", "recipe_readonly", "nested_foreign", "nested_writable", "volume"])
def test_external_input_mount_failures_rejected_before_mutation(tmp_path, monkeypatch, change):
    fixture = host_fixture(tmp_path, monkeypatch)
    external = tmp_path.parent / (tmp_path.name + "-inputs")
    external.mkdir()
    target = external / "package"
    target.mkdir()
    source = change in ("source_readonly", "recipe_readonly")
    if source:
        fixture["settings"]["model"] = {"mode": "source", "source_dir": str(target)}
    else:
        fixture["settings"]["model"]["package_dir"] = str(target)
    mount = {"Type": "bind", "Source": str(external), "Destination": str(external), "RW": source}
    if change == "foreign":
        mount["Source"] = str(tmp_path / "foreign")
    elif change == "source_readonly":
        mount["RW"] = False
    elif change == "volume":
        mount["Type"] = "volume"
    if change != "unmounted":
        fixture["controller"]["Mounts"].append(mount)
    if change in ("nested_foreign", "nested_writable", "recipe_readonly"):
        nested = target / (".ria-recipes" if change == "recipe_readonly" else "shards")
        fixture["controller"]["Mounts"].append({"Type": "bind", "Source": str(tmp_path / "foreign") if change == "nested_foreign" else str(nested),
            "Destination": str(nested), "RW": change == "nested_writable"})
    with pytest.raises(ArtifactError, match="setup input"):
        discover(fixture)
    assert len(fixture["calls"]) == 1 and fixture["calls"][0][3] == "inspect"


def test_input_ancestor_and_recipe_symlinks_rejected_before_mutation(tmp_path, monkeypatch):
    fixture = host_fixture(tmp_path, monkeypatch)
    source = tmp_path / "source"
    source.mkdir()
    (source / ".ria-recipes").symlink_to(tmp_path, target_is_directory=True)
    fixture["settings"]["model"] = {"mode": "source", "source_dir": str(source)}
    with pytest.raises(ArtifactError, match="symlink"):
        discover(fixture)
    assert len(fixture["calls"]) == 1 and fixture["calls"][0][3] == "inspect"


def test_ambiguous_nested_controller_identity_rejected_before_inspection(tmp_path, monkeypatch):
    fixture = host_fixture(tmp_path, monkeypatch)
    (fixture["proc"] / "self/cgroup").write_text("0::/" + "a" * 64 + "/" + "b" * 64 + "\n")
    with pytest.raises(ArtifactError, match="actual Engine container"):
        discover(fixture)
    assert not fixture["calls"]


@pytest.mark.parametrize(("field", "value"), [
    ("max_body_bytes", (64 << 20) + 1), ("max_header_bytes", 127), ("max_header_bytes", 65537),
    ("max_json_depth", 65), ("max_json_nodes", 200001), ("max_messages", 10001),
    ("max_tools", 1025), ("max_images", 17), ("max_encoded_image_bytes", (64 << 20) + 1),
    ("max_http_connections", 65), ("header_timeout_ms", 2147483648),
    ("body_timeout_ms", 2147483648), ("stream_write_timeout_ms", 2147483648),
])
def test_client_api_native_domains_rejected_before_deployment(tmp_path, field, value):
    settings, _, _ = settings_fixture(tmp_path, "client", "cuda")
    settings["api"][field] = value
    with pytest.raises(ArtifactError):
        validate_settings(settings)


@pytest.mark.parametrize("change", ["source", "missing_source", "extra_source", "binary", "image_uid", "image_digest", "cuda_version"])
def test_actual_image_build_and_runtime_mismatches_fail_before_inspection(tmp_path, monkeypatch, change):
    fixture = host_fixture(tmp_path, monkeypatch, executor="cuda" if change == "cuda_version" else "cpu")
    if change in ("source", "missing_source", "extra_source", "binary"):
        build = copy.deepcopy(fixture["build"])
        if change == "source":
            build["sources"]["tools/ria/setup_config.py"] = "9" * 64
        elif change == "missing_source":
            build["sources"].pop("tools/ria/setup_config.py")
        elif change == "extra_source":
            build["sources"]["invented.py"] = "9" * 64
        else:
            (fixture["project"] / "bin/ds4ctl").write_text("changed bytes")
        fixture["build"] = seal(build)
        previous_runner = fixture["runner"]
        def runner(arguments, deadline, **kwargs):
            if "/bin/cat" in arguments:
                fixture["calls"].append(arguments)
                return canonical(fixture["build"])
            return previous_runner(arguments, deadline, **kwargs)
        fixture["runner"] = runner
    elif change == "image_uid":
        fixture["image"]["Config"]["User"] = "0:0"
    elif change == "image_digest":
        fixture["image"]["RepoDigests"] = []
    else:
        fixture["image"]["Config"]["Env"] = [value.replace("CUDA_VERSION=13.4.2", "CUDA_VERSION=13.1.1")
            for value in fixture["image"]["Config"]["Env"]]
    with pytest.raises(ArtifactError):
        discover(fixture)
    assert not any("/bin/sleep" in call for call in fixture["calls"])


def test_pid_change_stops_only_the_owned_inspection(tmp_path, monkeypatch):
    fixture = host_fixture(tmp_path, monkeypatch)
    def observer(*args, **kwargs):
        fixture["temporary"]["State"]["Pid"] = 4243
        return fixture["report"]
    with pytest.raises(ArtifactError, match="PID/cgroup changed"):
        discover(fixture, observer=observer)
    assert fixture["calls"][-1][3:] == ["rm", "--force", "b" * 64]


def test_foreign_cleanup_identity_never_mutates_target(tmp_path, monkeypatch):
    fixture = host_fixture(tmp_path, monkeypatch)
    def observer(*args, **kwargs):
        fixture["temporary"]["Config"]["Labels"]["io.ria.setup.inspection"] = "foreign"
        raise ArtifactError("primary observation failure")
    with pytest.raises(ArtifactError, match="primary observation failure.*ambiguous inspection cleanup ownership"):
        discover(fixture, observer=observer)
    assert not any(call[3] == "rm" for call in fixture["calls"])


@pytest.mark.parametrize(("role", "executor"), [("expert", "cpu"), ("expert", "cuda"), ("client", "cuda")])
def test_factory_derives_exact_paths_identities_network_and_mode(tmp_path, monkeypatch, role, executor):
    fixture = host_fixture(tmp_path, monkeypatch, role, executor)
    facts = discover(fixture)
    paths = workspace_paths(tmp_path)
    settings = fixture["settings"]
    manifest = fixture["manifest"]
    if role == "expert":
        policy = read_json(tmp_path / "grants.json")
        atomic_json(paths["grants"], policy)
        policy_path = paths["grants"]
    else:
        settings["peer_address"] = "192.168.10.3"
        runtime = settings["client_runtime"]
        policy = seal({"schema_revision": 1, "logical_model_digest": manifest["logical_model_digest"],
            "operator_contract_digest": manifest["operator_contract_digest"], "server_layout_digest": "1" * 64, "server_executor": "cpu",
            "schedule": "full_reference", "shared_placement": "client", "expert_policy": "remote", **runtime,
            "runtime": {**runtime["runtime"], "tokenizer_file": "/model/metadata/tokenizer.bin", "tokenizer_sha256": manifest["tokenizer_digest"],
                        "prefill_rows": settings["planning"]["prefill_rows"]}})
        atomic_json(paths["placement"], policy)
        policy_path = paths["placement"]
    request = build_request(settings, tmp_path, manifest, facts, {"tls": {"enabled": False}}, policy_path)
    assert request["environment"]["host_report_digest"] == facts["host_report"]["digest"]
    assert request["environment"]["build_digest"] == facts["build_info"]["digest"]
    assert request["planning_request"]["logical_model_digest"] == manifest["logical_model_digest"]
    assert request["tls"] == {"enabled": False}
    assert request["probe_config"]["device_index"] == (None if executor == "cpu" else 0)
    assert request["network"]["control_address"] == ("0.0.0.0:7443" if role == "expert" else "192.168.10.3:7443")
    assert len(request["compose_files"]) == (2 if role == "expert" and executor == "cuda" else 1)
    if role == "client":
        assert request["api"]["bearer_token_file"] == "/run/secrets/api.token"
        assert request["api"]["max_active_generations"] == 1 and request["api"]["max_queued_generations"] == 0
    with pytest.raises(ArtifactError, match="transport mode"):
        build_request(settings, tmp_path, manifest, facts, {"tls": {"enabled": True}}, policy_path)


def test_factory_uses_original_verified_package_without_copy(tmp_path, monkeypatch):
    fixture = host_fixture(tmp_path, monkeypatch)
    facts = discover(fixture)
    paths = workspace_paths(tmp_path)
    atomic_json(paths["grants"], read_json(tmp_path / "grants.json"))
    original = tmp_path / "original-package"
    original.mkdir()
    atomic_json(original / "manifest.json", fixture["manifest"])
    request = build_request(fixture["settings"], tmp_path, fixture["manifest"], facts, {"tls": {"enabled": False}}, paths["grants"], model_root=original)
    assert request["environment"]["model_dir"] == str(original)
    assert list(original.iterdir()) == [original / "manifest.json"]
