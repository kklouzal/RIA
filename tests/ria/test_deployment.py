"""Pure deployment contract tests; mocked Compose is not a Docker execution claim."""

import copy
import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from ria.deployment import (bootstrap, controlled_environment, environment_bytes,
                            finalize, frozen_environment, validate_effective_compose)
from ria.identity import ArtifactError, atomic_json, canonical, digest, read_json, seal
from ria.schemas import validate
from test_artifacts import fixture_recipe, prepare


@pytest.fixture(autouse=True)
def isolated_controller_lock_directory(tmp_path, monkeypatch):
    monkeypatch.setattr("ria.deployment.CONTROLLER_LOCK_DIRECTORY", tmp_path)
    # This suite supplies synthetic sealed host reports, never physical queries.
    monkeypatch.setattr("ria.host.revalidate_host_report", lambda report: report)


def deployment_request(tmp_path, role="expert", executor="cpu", *, tls_enabled=True):
    recipe = seal({**fixture_recipe(tmp_path), "chunk_size": 4096})
    manifest = prepare(tmp_path, recipe, tmp_path / "model")
    identity = hashlib.sha256(b"synthetic offline deployment fixture").hexdigest()
    host_report = seal({"schema_revision": 1, "kind": "host_preflight", "rootful": True, "cgroup_version": 2,
        "kernel_version": "fixture", "docker_version": "fixture", "compose_version": "fixture",
        "selected_gpu_uuid": None if executor == "cpu" else "GPU-12345678-1234-1234-1234-123456789abc", "client": role == "client",
        "parent_ancestors": [], "classification": "synthetic offline contract fixture"})
    atomic_json(tmp_path / "host.json", host_report)
    planning = {"schema_revision": 1, "role": role, "executor": executor, "profile": "bf16",
        "logical_model_digest": recipe["logical_model_digest"], "operator_contract_digest": manifest["operator_contract_digest"],
        "context_positions": 64, "prefill_rows": 8, "caps": {"host_bytes": 16 << 20, "device_bytes": 0 if executor == "cpu" else 4 << 20,
        "pinned_bytes": 0 if executor == "cpu" else 2 << 20, "numa": [{"node": 0, "bytes": 16 << 20}]}}
    gpu = None if executor == "cpu" else "GPU-12345678-1234-1234-1234-123456789abc"
    secret_dir = tmp_path / "secrets"
    secret_dir.mkdir()
    credentials = ("ca.pem", "peer.pem", "peer.key", "api.token") if tls_enabled else (("api.token",) if role == "client" else ())
    for name in credentials:
        path = secret_dir / name
        path.write_text("test-only noncredential")
        path.chmod(0o600)
    (tmp_path / "reports").mkdir()
    seccomp = tmp_path / "seccomp.json"
    atomic_json(seccomp, {"defaultAction": "SCMP_ACT_ERRNO", "syscalls": []})
    compose = tmp_path / "compose.yaml"
    compose.write_text("# pure fixture only; no Docker invocation\n")
    ctl = tmp_path / "ds4ctl"
    ctl.write_text("#!/bin/sh\nexit 2\n")
    ctl.chmod(0o755)
    tls = {"ca_file": "/run/secrets/ca.pem", "certificate_file": "/run/secrets/peer.pem", "private_key_file": "/run/secrets/peer.key",
           "expected_peer_name": "peer.test.internal", "minimum_version": "TLS1.3", "early_data": False} if tls_enabled else {"enabled": False}
    network = {"control_address": "0.0.0.0:7443", "bulk_address": "0.0.0.0:7444", "connect_timeout_ms": 1000,
        "handshake_timeout_ms": 1000, "operation_timeout_ms": 1000, "frame_io_timeout_ms": 1000, "write_timeout_ms": 1000,
        "max_row_lookup_rows": 256, "max_inflight_payload_bytes": 67108864, "max_frame_payload_bytes": 16777216,
        "max_bulk_data_bytes": 4194304, "max_inflight_expert_requests": 2, "server_executor": executor}
    request = {"schema_revision": 1, "planning_request": planning, "host_report": str(tmp_path / "host.json"), "qualification_scope": "initial_fixture",
        "probe_config": {"schema_revision": 1, "role": role, "executor": executor, "device_index": None if executor == "cpu" else 0,
            "expected_gpu_uuid": gpu, "numa_nodes": [0], "max_host_test_bytes": 4096, "max_device_test_bytes": 0 if executor == "cpu" else 4096,
            "max_pinned_test_bytes": 0 if executor == "cpu" else 4096, "deadline_ms": 1000, "disable_core_dumps": True,
            "environment_digest": identity, "build_digest": identity, "build_info_file": "/usr/share/dwarfstar/build-info.json"},
        "environment": {"image": "registry.test/ria@sha256:" + identity, "image_kind": executor,
            "build_digest": identity, "cpuset": "0-1", "cgroup_bytes": 1 << 30, "memlock_bytes": 16 << 20, "pids_limit": 64,
            "gpu_uuid": gpu, "model_dir": str(tmp_path / "model"), "secret_dir": str(secret_dir), "report_dir": str(tmp_path / "reports"),
            "seccomp_profile": str(seccomp), "bind_ip": "192.168.10.2", "api_port": 8000, "start_period_seconds": 300,
            "stop_grace_seconds": 30, "docker_version": "fixture", "compose_version": "fixture", "kernel_version": "fixture",
            "host_report_digest": host_report["digest"], "source_lock_digest": identity}, "tls": tls, "network": network,
        "native_ctl": str(ctl), "compose_files": [str(compose)], "deadline_ms": 1000}
    if role == "client":
        request["api"] = {"bind_address": "0.0.0.0:8000", "bearer_token_file": "/run/secrets/api.token", "max_body_bytes": 67108864,
            "max_header_bytes": 16384, "max_json_depth": 32, "max_json_nodes": 100000, "max_http_connections": 8,
            "max_messages": 1000, "max_tools": 100, "max_images": 4,
            "max_encoded_image_bytes": 33554432, "max_decoded_image_bytes": 25165824,
            "header_timeout_ms": 1000, "body_timeout_ms": 1000, "stream_write_timeout_ms": 1000, "max_active_generations": 1,
            "max_queued_generations": 0, "allow_remote_image_urls": False, "cors_allowed_origins": []}
    else:
        request["expert"] = {"numa_policy": "sharded", "nodes": [{"node": 0, "cpus": [0], "workers": 1, "local_bytes": str(16 << 20)}],
            "projection_tile_rows": 64, "host_runtime_bytes": str(8 << 20), "startup_host_bytes": str(16 << 20),
            "device_workspace_bytes": "0" if executor == "cpu" else str(4 << 20), "pinned_workspace_bytes": "0" if executor == "cpu" else str(2 << 20), "drain_timeout_ms": 1000}
        grants = seal({"schema_revision": 1, "grants": [{"expected_peer_name": tls.get("expected_peer_name"),
            "logical_model_digest": planning["logical_model_digest"], "operator_contract_digest": planning["operator_contract_digest"],
            "encoding_digest": manifest["encoding_digest"], "client_layout_digest": identity, "placement_plan_digest": identity,
            "profile": "bf16", "server_executor": executor, "server_layout_digest": manifest["layout_digest"]}]})
        atomic_json(tmp_path / "grants.json", grants)
        request["peer_grants"] = str(tmp_path / "grants.json")
    validate("deployment-request", request)
    return request


def record_config_only_compose_version(request):
    """Record the actual CLI version in an otherwise synthetic host fixture."""
    from ria.deployment import _run
    version = _run(["docker", "compose", "version", "--short"], request["deadline_ms"]).decode("ascii").strip().removeprefix("v")
    report = seal({**read_json(request["host_report"]), "compose_version": version})
    atomic_json(request["host_report"], report)
    request["environment"].update(compose_version=version, host_report_digest=report["digest"])
    return version


def measurement_fixture(request, directory, *, qualified=True):
    """Synthetic sealed evidence for orchestration tests, never a hardware claim."""
    from ria.qualification import freeze_policy
    exact = {"max_abs_error": 0, "max_relative_error": 0, "max_rms_error": 0, "max_loss_delta": 0, "relative_floor": 1e-12}
    policy = freeze_policy({"schema_revision": 1, "logical_model_digest": request["planning_request"]["logical_model_digest"],
        "source_lock_digest": request["environment"]["source_lock_digest"], "thresholds": {"same_realization": exact, "native_source": exact},
        "minimum_soak_seconds": 3600, "ordered_objectives": ["fixture contract only"]}, directory / "policy.json")
    base = {"environment_digest": digest(frozen_environment(request)), "build_digest": request["environment"]["build_digest"]}
    from fixture_measurements import registration, write_component_fixture
    environments = {"server": seal(frozen_environment(request))}
    reg = registration(policy, server_environment=base["environment_digest"], server_build=base["build_digest"],
                       operator=request["planning_request"]["operator_contract_digest"], environments=environments,
                       tls_enabled=request["tls"].get("enabled", True))
    components = write_component_fixture(directory, policy, reg, environments=environments)
    calibration_evidence = seal({"schema_revision": 1, "kind": "calibration_evidence", "qualification_scope": "initial_fixture", **base,
        "policy_digest": policy["digest"], "profile": "bf16", "executor": "cpu", "operator_contract_digest": request["planning_request"]["operator_contract_digest"],
        "components": components, "comparisons": [], "release_matrix_digest": None, "release_matrix_path": None, "soak_seconds": 0, "passed": qualified})
    probe_evidence = seal({"schema_revision": 1, "kind": "probe_evidence", "role": "expert", "executor": "cpu", **base,
        "architecture": "x86_64", "uid": 10001, "dumpable": False, "seccomp_mode": 2, "no_new_privileges": True,
        "cgroup_limit_bytes": str(request["environment"]["cgroup_bytes"]), "cgroup_available_bytes": str(1 << 20), "swap_limit_bytes": "0",
        "memlock_bytes": str(request["environment"]["memlock_bytes"]), "cpu_mask": request["environment"]["cpuset"], "memory_node_mask": "0",
        "host_test_bytes": "4096", "host_test_ms": "0", "numa_locality_proven": True, "driver_version": 0, "runtime_version": 0,
        "gpu_uuid": None, "compute_major": 0, "compute_minor": 0, "gpu_allocation_ms": "0", "native_kernel_ms": "0", "native_results": [0, 0, 0],
        "host_parent_preflight_required": True, "host_available_bytes": str(request["planning_request"]["caps"]["host_bytes"]), "device_available_bytes": "0", "pinned_test_bytes": "0",
        "numa_available": [{"node": 0, "bytes": str(request["planning_request"]["caps"]["host_bytes"])}]})
    probe = seal({"schema_revision": 1, "role": "expert", "executor": "cpu", "host_bytes": request["planning_request"]["caps"]["host_bytes"], "device_bytes": 0,
        "pinned_bytes": 0, "numa": [{"node": 0, "bytes": 16 << 20}], "qualified": qualified, **base, "evidence_digest": probe_evidence["digest"]})
    calibration = seal({"schema_revision": 1, "profile": "bf16", "operator_contract_digest": request["planning_request"]["operator_contract_digest"],
        "executor": "cpu", "qualified": qualified, **base, "policy_digest": policy["digest"], "evidence_digest": calibration_evidence["digest"]})
    return probe, calibration, {"probe_evidence": probe_evidence, "calibration_evidence": calibration_evidence, "calibration_evidence_dir": directory, "policy": policy}


def compose_fixture(request, directory):
    role, env = request["planning_request"]["role"], request["environment"]
    service = {"image": env["image"], "user": "10001:10001", "read_only": True, "init": True, "restart": "no",
        "entrypoint": ["/usr/local/bin/ds4-server" if role == "client" else "/usr/local/bin/ds4-expert-server"],
        "command": ["--config", "/etc/dwarfstar/service.json"], "platform": "linux/amd64",
        "cpuset": env["cpuset"], "pids_limit": env["pids_limit"], "mem_limit": env["cgroup_bytes"], "memswap_limit": env["cgroup_bytes"],
        "cap_drop": ["ALL"], "security_opt": ["no-new-privileges:true", "seccomp:" + env["seccomp_profile"]],
        "ulimits": {"memlock": {"soft": env["memlock_bytes"], "hard": env["memlock_bytes"]}, "core": {"soft": 0, "hard": 0}},
        "volumes": [{"type": "bind", "source": path, "target": target, "read_only": readonly, "bind": {"create_host_path": False}}
                    for target, path, readonly in (("/model", env["model_dir"], True), ("/etc/dwarfstar", str(directory), True),
                                                  ("/run/secrets", env["secret_dir"], True), ("/artifacts", env["report_dir"], False))],
        "runtime": "runc" if request["planning_request"]["executor"] == "cpu" else "nvidia",
        "tmpfs": ["/run/dwarfstar:rw,noexec,nosuid,nodev,size=16m,mode=0700,uid=10001,gid=10001", "/tmp:rw,noexec,nosuid,nodev,size=64m,mode=1777"],
        "healthcheck": {"test": ["CMD", "/usr/local/bin/ds4ctl", "health", "--socket", "/run/dwarfstar/admin.sock", "--timeout-ms", "5000"],
                        "interval": "10s", "timeout": "6s", "retries": 3, "start_period": str(env["start_period_seconds"]) + "s"},
        "stop_grace_period": str(env["stop_grace_seconds"]) + "s",
        "ports": [{"target": 8000, "published": str(env["api_port"]), "host_ip": "127.0.0.1", "protocol": "tcp"}] if role == "client" else
                 [{"target": port, "published": str(port), "host_ip": env["bind_ip"], "protocol": "tcp"} for port in (7443, 7444)]}
    if request["planning_request"]["executor"] == "cuda":
        service["environment"] = {"NVIDIA_VISIBLE_DEVICES": env["gpu_uuid"], "NVIDIA_DRIVER_CAPABILITIES": "compute,utility"}
        service["deploy"] = {"resources": {"reservations": {"devices": [{"driver": "nvidia", "device_ids": [env["gpu_uuid"]], "capabilities": ["gpu"]}]}}}
    return {"name": "ria-" + role, "services": {role: service}}


@pytest.mark.parametrize("role,executor", [("expert", "cpu"), ("expert", "cuda"), ("client", "cuda")])
@pytest.mark.parametrize("tls_enabled", [True, False])
def test_bootstrap_three_roles_and_immutable_publication(tmp_path, role, executor, tls_enabled):
    request = deployment_request(tmp_path, role, executor, tls_enabled=tls_enabled)
    output = tmp_path / "bootstrap"
    calls = []
    def runner(args, deadline, cwd=None):
        calls.append(args)
        return canonical(compose_fixture(request, output)) if args[-2:] == ["--format", "json"] else b""
    result = bootstrap(request, output, runner=runner)
    assert result["status"] == "not_admitted" and result["admitted"] is False
    assert not (output / "memory-plan.json").exists() and not (output / "service.json").exists()
    assert len(calls) == 2
    with pytest.raises(ArtifactError, match="immutable"):
        bootstrap(request, output, runner=runner)


@pytest.mark.parametrize("field", ["ca_file", "certificate_file", "private_key_file", "expected_peer_name", "minimum_version", "early_data"])
def test_plaintext_config_rejects_certificate_fields(tmp_path, field):
    request = deployment_request(tmp_path)
    request["tls"] = {"enabled": False, field: request["tls"][field]}
    with pytest.raises(ArtifactError):
        validate("deployment-request", request)


@pytest.mark.parametrize("mode", [None, 0, 1, "false", "true"])
def test_transport_mode_requires_a_boolean(tmp_path, mode):
    request = deployment_request(tmp_path)
    request["tls"]["enabled"] = mode
    with pytest.raises(ArtifactError):
        validate("deployment-request", request)


def test_enabled_secure_mode_still_requires_complete_credentials(tmp_path):
    request = deployment_request(tmp_path)
    request["tls"]["enabled"] = True
    validate("deployment-request", request)
    del request["tls"]["certificate_file"]
    with pytest.raises(ArtifactError):
        validate("deployment-request", request)


@pytest.mark.parametrize("tls_enabled,credential", [(True, "ca.pem"), (True, "peer.pem"), (True, "peer.key"), (False, "api.token")])
def test_bootstrap_preserves_required_credentials_in_both_modes(tmp_path, tls_enabled, credential):
    request = deployment_request(tmp_path, "client", "cuda", tls_enabled=tls_enabled)
    (Path(request["environment"]["secret_dir"]) / credential).unlink()
    with pytest.raises(ArtifactError, match="required credential"):
        bootstrap(request, tmp_path / "bootstrap")
    assert not (tmp_path / "bootstrap").exists()


def test_plaintext_api_token_remains_private(tmp_path):
    request = deployment_request(tmp_path, "client", "cuda", tls_enabled=False)
    (Path(request["environment"]["secret_dir"]) / "api.token").chmod(0o644)
    with pytest.raises(ArtifactError, match="world-readable"):
        bootstrap(request, tmp_path / "bootstrap")


def test_controls_environment_and_host_override_rejected(tmp_path, monkeypatch):
    request = deployment_request(tmp_path)
    monkeypatch.setenv("EXPERT_IMAGE", "evil")
    monkeypatch.setenv("COMPOSE_FILE", "/unreviewed")
    assert "EXPERT_IMAGE" not in controlled_environment() and "COMPOSE_FILE" not in controlled_environment()
    assert b"registry.test" in environment_bytes(request, tmp_path / "config with spaces")
    document = compose_fixture(request, tmp_path / "config")
    for change in ({"privileged": True}, {"memswap_limit": 0}, {"cap_add": ["SYS_ADMIN"]}, {"runtime": "nvidia"}, {"entrypoint": ["sh"]}):
        altered = copy.deepcopy(document)
        altered["services"]["expert"].update(change)
        with pytest.raises(ArtifactError):
            validate_effective_compose(altered, request, tmp_path / "config")


def test_audit_scope_preserves_realization_but_peer_policy_changes_do_not(tmp_path):
    request = deployment_request(tmp_path)
    initial = frozen_environment(request)
    assert "qualification_scope" not in initial and "peer_grants_digest" in initial
    request["qualification_scope"] = "final_release"
    assert frozen_environment(request) == initial
    grants = read_json(request["peer_grants"])
    grants["grants"][0]["expected_peer_name"] = "other.test.internal"
    atomic_json(request["peer_grants"], seal(grants))
    assert frozen_environment(request) != initial


def test_unqualified_measurements_fail_before_mutation(tmp_path):
    request = deployment_request(tmp_path)
    planning = request["planning_request"]
    probe, calibration, evidence = measurement_fixture(request, tmp_path, qualified=False)
    inventory = {"schema_revision": 1, "logical_model_digest": planning["logical_model_digest"],
                 "operator_contract_digest": planning["operator_contract_digest"], "semantic_max_positions": 1048576,
                 "allocations": [{"id": "1", "name": "protected-progress", "resource": "host", "base_bytes": 4096,
                                  "bytes_per_position": 0, "numa_node": 0, "pinned": False, "protected_progress": True,
                                  "phases": ["startup", "prefill", "decode", "continuation", "image", "drain"]}]}
    with pytest.raises(ArtifactError, match="unqualified"):
        finalize(request, probe, inventory, calibration, tmp_path / "final")
    assert not (tmp_path / "final").exists()


@pytest.mark.parametrize("transport_mode", ["default_tls", "explicit_tls", "trusted_network"])
@pytest.mark.parametrize("wrong_peer_grant", [False, True])
def test_native_plan_invocation_without_python_memory_equations(tmp_path, transport_mode, wrong_peer_grant):
    request = deployment_request(tmp_path, tls_enabled=transport_mode != "trusted_network")
    if transport_mode == "explicit_tls":
        request["tls"]["enabled"] = True
    if wrong_peer_grant:
        grants = read_json(request["peer_grants"])
        grants["grants"][0]["expected_peer_name"] = None if request["tls"].get("enabled", True) else "peer.test.internal"
        atomic_json(request["peer_grants"], seal(grants))
    planning = request["planning_request"]
    probe, calibration, evidence = measurement_fixture(request, tmp_path)
    inventory = fixture_inventory(request)
    output = tmp_path / "final"
    calls = []
    def runner(args, deadline, cwd=None):
        calls.append(args)
        if args[0] == request["native_ctl"]:
            if args[1] == "inventory":
                from ria.deployment import _run
                native = Path(__file__).resolve().parents[2] / "bin/ds4ctl"
                return _run([str(native), *args[1:]], deadline)
            paths = dict(zip(args[2::2], args[3::2], strict=True))
            memory = copy.deepcopy(planning["caps"])
            # Deliberately an external oracle fixture; renderer only authenticates
            # its result. Native admission tests separately prove these equations.
            plan = seal({"schema_revision": 1, "admitted": True, "role": "expert", "executor": "cpu", "profile": "bf16",
                "logical_model_digest": planning["logical_model_digest"], "operator_contract_digest": planning["operator_contract_digest"],
                "environment_digest": probe["environment_digest"], "build_digest": probe["build_digest"], "policy_digest": calibration["policy_digest"],
                **{key + "_digest": digest(read_json(paths["--" + key])) for key in ("request", "inventory", "probe", "calibration")},
                "context_positions": 64, "prefill_rows": planning["prefill_rows"], "allocation_count": 1, "caps": planning["caps"], "peak": memory,
                "phases": {phase: memory for phase in ("startup", "prefill", "decode", "continuation", "image", "drain")}})
            atomic_json(paths["--output"], plan)
            return b""
        return canonical(compose_fixture(request, output)) if args[-2:] == ["--format", "json"] else b""
    if wrong_peer_grant:
        with pytest.raises(ArtifactError, match="peer grants"):
            finalize(request, probe, inventory, calibration, output, runner=runner, **evidence)
        assert not output.exists()
        return
    lock = finalize(request, probe, inventory, calibration, output, runner=runner, **evidence)
    assert calls[0][1] == "inventory" and calls[1][1] == "plan" and len(calls) == 4
    assert lock["memory_plan_digest"] == read_json(output / "memory-plan.json")["digest"]
    assert read_json(output / "service.json")["peer_grants"] == "/etc/dwarfstar/peer-grants.json"
    assert read_json(output / "service.json")["tls"] == request["tls"]
    assert lock["expected_peer_name"] == request["tls"].get("expected_peer_name")
    assert "private_key" not in canonical(lock).decode()
    assert lock["qualification_scope"] == "initial_fixture" and not lock["final_release_qualified"]
    assert (output / "evidence" / "experts.json").is_file()


def fixture_inventory(request):
    from ria.inventory import build_inventory
    native_request = copy.deepcopy(request)
    native_request["native_ctl"] = str(Path(__file__).resolve().parents[2] / "bin/ds4ctl")
    return build_inventory(native_request, Path(request["environment"]["model_dir"]) / "manifest.json",
                           Path(request["environment"]["report_dir"]) / "fixture-inventory.json")


@pytest.mark.parametrize("mutation", ["environment", "transport_mode", "raw_capacity", "missing_proof", "scope"])
def test_finalize_rejects_resealed_mismatched_or_missing_provenance(tmp_path, mutation):
    request = deployment_request(tmp_path)
    probe, calibration, evidence = measurement_fixture(request, tmp_path)
    if mutation == "environment":
        request["environment"]["pids_limit"] += 1
    elif mutation == "transport_mode":
        request["tls"] = {"enabled": False}
        grants = read_json(request["peer_grants"])
        grants["grants"][0]["expected_peer_name"] = None
        atomic_json(request["peer_grants"], seal(grants))
    elif mutation == "raw_capacity":
        evidence["probe_evidence"] = seal({**evidence["probe_evidence"], "host_available_bytes": "1"})
        probe = seal({**probe, "evidence_digest": evidence["probe_evidence"]["digest"]})
    elif mutation == "missing_proof":
        (tmp_path / "graph_state.json").unlink()
    else:
        request["qualification_scope"] = "final_release"
    with pytest.raises((ArtifactError, OSError)):
        finalize(request, probe, fixture_inventory(request), calibration, tmp_path / "rejected", **evidence)
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("failure", ["up", "ancestor", "none", "scope", "host"])
def test_real_native_admission_and_owned_launch_rollback(tmp_path, monkeypatch, failure):
    from ria.deployment import _run, launch
    import ria.host
    native = Path(__file__).resolve().parents[2] / "bin" / "ds4ctl"
    if not native.is_file():
        pytest.skip("native ds4ctl is a required separately built integration fixture")
    request = deployment_request(tmp_path)
    request["native_ctl"] = str(native)
    probe, calibration, evidence = measurement_fixture(request, tmp_path)
    output, calls = tmp_path / "final", []
    container_id = "c" * 64
    def runner(args, deadline, cwd=None):
        calls.append(args)
        if args[0] == str(native):
            return _run(args, deadline, cwd=cwd)
        if args[-2:] == ["--format", "json"]:
            return canonical(compose_fixture(request, output))
        if "up" in args and failure == "up":
            raise ArtifactError("synthetic ambiguous up failure")
        if "ps" in args:
            return (container_id + "\n").encode()
        if "inspect" in args:
            from test_container_inspection import inspection_fixture
            observed = inspection_fixture(request, output)
            observed.update(Id=container_id, State={"Running": True, "Pid": 12345})
            observed["Config"].update(Image=request["environment"]["image"],
                Labels={"com.docker.compose.service": "expert", "com.docker.compose.project": "ria-expert"})
            return canonical(observed)
        return b""
    lock = finalize(request, probe, fixture_inventory(request), calibration, output, runner=runner, **evidence)
    assert read_json(output / "memory-plan.json")["peak"]["host_bytes"] == request["planning_request"]["caps"]["host_bytes"]
    def verify(pid, baseline):
        assert pid == 12345 and baseline["digest"] == request["environment"]["host_report_digest"]
        if failure == "ancestor":
            raise ArtifactError("synthetic parent mismatch")
        return seal({"kind": "synthetic offline ancestry contract", "pid": pid})
    monkeypatch.setattr(ria.host, "verify_container_ancestors", verify)
    if failure in ("scope", "host"):
        if failure == "scope":
            request["qualification_scope"] = "final_release"
            message = "requested admission scope differs"
        else:
            def changed_host(_):
                raise ArtifactError("current stable host realization changed")
            monkeypatch.setattr(ria.host, "revalidate_host_report", changed_host)
            message = "current stable host realization changed"
        before = list(calls)
        with pytest.raises(ArtifactError, match=message):
            launch(request, output, runner=runner)
        assert calls == before
    elif failure == "none":
        launch(request, output, runner=runner)
        assert (Path(request["environment"]["report_dir"]) / ("launch-" + lock["digest"] + ".json")).exists()
        assert not any("stop" in args for args in calls)
    else:
        with pytest.raises(ArtifactError, match="owned service was stopped"):
            launch(request, output, runner=runner)
        assert any("stop" in args for args in calls)


def test_compose_exact_resources_health_and_canonical_duration(tmp_path):
    request = deployment_request(tmp_path)
    document = compose_fixture(request, tmp_path / "config")
    document["services"]["expert"]["healthcheck"]["start_period"] = "5m0s"
    validate_effective_compose(document, request, tmp_path / "config")
    for field, value in (("tmpfs", ["/run/dwarfstar:rw"]), ("healthcheck", {"disable": True}),
                         ("stop_grace_period", "1h0m0s"), ("sysctls", {"kernel.core_pattern": "evil"}),
                         ("ulimits", {"memlock": {"soft": 1048576, "hard": 1048576}, "core": {"soft": 0, "hard": 0}, "nproc": 0})):
        altered = copy.deepcopy(document)
        altered["services"]["expert"][field] = value
        with pytest.raises(ArtifactError):
            validate_effective_compose(altered, request, tmp_path / "config")


@pytest.mark.parametrize("version", ["2.0.0", "2.20.0", "2.40.3"])
def test_compose2_omitted_false_bind_keeps_strict_mount_contract(tmp_path, version):
    request = deployment_request(tmp_path)
    request["environment"]["compose_version"] = version
    document = compose_fixture(request, tmp_path / "config")
    for volume in document["services"]["expert"]["volumes"]:
        volume["bind"] = {}
    validate_effective_compose(document, request, tmp_path / "config")
    for mutation in ("missing", "null", "true", "zero", "propagation", "source", "readonly", "readonly_type"):
        altered = copy.deepcopy(document)
        volume = altered["services"]["expert"]["volumes"][0]
        if mutation == "missing":
            volume.pop("bind")
        elif mutation == "null":
            volume["bind"] = None
        elif mutation in ("true", "zero"):
            volume["bind"] = {"create_host_path": True if mutation == "true" else 0}
        elif mutation == "propagation":
            volume["bind"] = {"propagation": "rshared"}
        elif mutation == "source":
            volume["source"] = "/unreviewed/source"
        else:
            volume["read_only"] = False if mutation == "readonly" else "true"
        with pytest.raises(ArtifactError):
            validate_effective_compose(altered, request, tmp_path / "config")


@pytest.mark.parametrize("version", ["2.40.4", "2.41.0", "5.0.0", "5.2.0", "fixture", "2.40.3-dev"])
def test_unknown_or_optout_compose_cannot_treat_empty_bind_as_false(tmp_path, version):
    request = deployment_request(tmp_path)
    request["environment"]["compose_version"] = version
    document = compose_fixture(request, tmp_path / "config")
    validate_effective_compose(document, request, tmp_path / "config")
    document["services"]["expert"]["volumes"][0]["bind"] = {}
    with pytest.raises(ArtifactError):
        validate_effective_compose(document, request, tmp_path / "config")


def test_strict_role_budget_gpu_and_identity_crossfields(tmp_path):
    request = deployment_request(tmp_path)
    for mutate in (lambda r: r["environment"].update(image_kind="cuda"),
                   lambda r: r["planning_request"]["caps"].update(device_bytes=1),
                   lambda r: r.update(unexpected=1),
                   lambda r: r["probe_config"].update(expected_gpu_uuid="GPU-12345678-1234-1234-1234-123456789abc")):
        altered = copy.deepcopy(request)
        mutate(altered)
        with pytest.raises(ArtifactError):
            validate("deployment-request", altered)


def test_subprocess_output_is_bounded_before_retention_and_deadline():
    from ria.deployment import _run
    with pytest.raises(ArtifactError, match="output exceeded"):
        _run([sys.executable, "-c", "import os; os.write(2,b'x'*100000)"], 1000)
    with pytest.raises(ArtifactError, match="deadline"):
        _run([sys.executable, "-c", "import time; time.sleep(1)"], 50)
    with pytest.raises(ArtifactError, match="exited 7"):
        _run([sys.executable, "-c", "import sys; sys.stderr.write('owned fixture failure'); sys.exit(7)"], 1000)
    assert _run([sys.executable, "-c", "print('bounded success')"], 1000) == b"bounded success\n"


def test_placement_runtime_is_explicit_and_strict(tmp_path):
    request = deployment_request(tmp_path, "client", "cuda")
    planning = request["planning_request"]
    placement = seal({"schema_revision": 1, "logical_model_digest": planning["logical_model_digest"],
        "operator_contract_digest": planning["operator_contract_digest"], "server_layout_digest": "a" * 64,
        "server_executor": "cpu", "schedule": "full_reference", "shared_placement": "client", "expert_policy": "remote",
        "host_expert_cache_bytes": 0, "device_expert_cache_bytes": 0, "engram_cache_bytes": 0, "local_experts": [],
        "runtime": {"tokenizer_file": "/model/metadata/tokenizer.bin", "tokenizer_sha256": "b" * 64,
                    "tokenizer_memory_bytes": "10485760", "host_state_bytes": "16777216", "device_state_bytes": "16777216", "frontend_host_bytes": "16777216",
                    "projection_tile_rows": 128, "state_tile_rows": 128, "max_image_patches": 16, "prefill_rows": 8}})
    validate("placement-plan", placement)
    for field, value in (("host_state_bytes", "0"), ("device_state_bytes", "18446744073709551616"), ("projection_tile_rows", 0)):
        altered = copy.deepcopy(placement)
        altered["runtime"][field] = value
        with pytest.raises(ArtifactError):
            validate("placement-plan", altered)
    for field in ("max_images", "max_http_connections", "max_decoded_image_bytes"):
        altered = copy.deepcopy(request)
        del altered["api"][field]
        with pytest.raises(ArtifactError):
            validate("deployment-request", altered)
