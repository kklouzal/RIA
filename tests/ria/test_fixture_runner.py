"""Synthetic offline measurement contracts; no GPU or model execution here."""

import copy
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from fixture_measurements import registration, raw_population, write_component_fixture
from fixture_measurements import preflight
from ria.fixture_runner import (RUNS, _source_population, derive_bootstrap, derive_request,
    execute_run, freeze_registration, produce_components, publish_requests,
    validate_component_sources, validate_native, validate_registration, validate_transport)
from ria.identity import ArtifactError, atomic_json, canonical, hash_file, read_json, seal
from ria.qualification import validate_calibration_evidence
from ria.schemas import validate
from ria.fixture_environment import validate_preflight
from test_qualification import policy


def test_registration_is_acyclic_and_requests_are_atomic_immutable(tmp_path):
    p = policy(tmp_path)
    reg = registration(p)
    assert validate_registration(reg, p) == reg
    assert all("preregistration_digest" not in item["request_body"] for item in reg["runs"].values())
    output = tmp_path / "requests with spaces"
    publish_requests(reg, p, output)
    assert len(list(output.iterdir())) == 6
    for name in RUNS:
        request = read_json(output / (name + "-request.json"))
        assert request == derive_request(reg, name)
        assert request["preregistration_digest"] == reg["digest"]
        if name.startswith("transport_"):
            validate("transport-request", request)
            bootstrap = read_json(output / (name + "-bootstrap.json"))
            assert bootstrap == derive_bootstrap(reg, name)
            validate("transport-bootstrap", bootstrap)
        else:
            assert "digest" not in request
    with pytest.raises(ArtifactError, match="immutable"):
        publish_requests(reg, p, output)


@pytest.mark.parametrize("failure", ["policy", "role", "profile", "population", "floor", "limit", "deadline", "credits", "timestamp", "bootstrap", "cycle"])
def test_resealed_registration_cannot_change_contract(tmp_path, failure):
    p = policy(tmp_path)
    reg = copy.deepcopy(registration(p))
    body = reg["runs"]["native_server"]["request_body"]
    if failure == "policy":
        reg["policy_digest"] = "0" * 64
    elif failure == "role":
        body["role"] = "client"
    elif failure == "profile":
        body["profile"] = "nvfp4"
    elif failure == "population":
        body["expert_shape"] = "ragged"
    elif failure == "floor":
        body["relative_floor"] *= 2
    elif failure == "limit":
        reg["runs"]["native_server"]["hard_limits"]["host_bytes"] = "1"
    elif failure == "deadline":
        body["deadline_ms"] = 1
    elif failure == "credits":
        for name in ("transport_server", "transport_client"):
            reg["runs"][name]["request_body"]["row_credit"] = "431"
    elif failure == "timestamp":
        reg["registered_at"] = "2000-01-01T00:00:00Z"
    elif failure == "bootstrap":
        reg["runs"]["transport_server"]["bootstrap_body"]["build_digest"] = "0" * 64
    else:
        body["preregistration_digest"] = reg["digest"]
    with pytest.raises(ArtifactError):
        validate_registration(seal(reg), p)


def frozen_inputs(tmp_path, p):
    from test_deployment import deployment_request
    from ria.deployment import frozen_environment
    reg = registration(p)
    builds, environments = {}, {}
    for role in ("server", "client"):
        build = seal({"schema_revision": 1, "source_lock_digest": p["source_lock_digest"],
            "image_kind": "cpu" if role == "server" else "cuda", "binaries": {"ds4-ria-qualify": "e" * 64, "ds4ctl": "f" * 64}})
        bootstrap = reg["runs"]["transport_" + role]["bootstrap_body"]
        directory = tmp_path / role
        directory.mkdir()
        deployment = deployment_request(directory, role="expert" if role == "server" else "client", executor="cpu" if role == "server" else "cuda")
        deployment["environment"].update(build_digest=build["digest"], source_lock_digest=p["source_lock_digest"])
        deployment["planning_request"].update(logical_model_digest=p["logical_model_digest"], operator_contract_digest="3" * 64)
        deployment["planning_request"]["caps"].update(host_bytes=268435456,
            device_bytes=134217728 if role == "client" else 0, pinned_bytes=1048576 if role == "client" else 0)
        deployment["network"].update(bootstrap["network"], server_executor="cpu")
        environment = seal(frozen_environment(deployment))
        reg["realizations"][role] = {"environment_digest": environment["digest"], "build_digest": build["digest"]}
        for prefix in ("native", "transport"):
            run = reg["runs"][prefix + "_" + role]
            run["request_body"].update(reg["realizations"][role])
            if prefix == "transport":
                run["bootstrap_body"].update(reg["realizations"][role])
            del run["binary_sha256"]
        builds[role], environments[role] = tmp_path / (role + "-build.json"), tmp_path / (role + "-environment.json")
        atomic_json(builds[role], build)
        atomic_json(environments[role], environment)
    reg["kind"] = "fixture_registration_request"
    del reg["digest"], reg["registered_at"]
    return reg, builds, environments


def test_registration_uses_actual_sealed_build_and_environment(tmp_path):
    p = policy(tmp_path)
    request, builds, environments = frozen_inputs(tmp_path, p)
    result = freeze_registration(request, p, builds, environments, tmp_path / "registration.json")
    assert result["runs"]["native_server"]["binary_sha256"] == "e" * 64
    assert result["runs"]["transport_client"]["binary_sha256"] == "f" * 64
    request["runs"]["transport_server"]["bootstrap_body"]["network"]["control_address"] = "127.0.0.2:7443"
    with pytest.raises(ArtifactError, match="network/TLS"):
        freeze_registration(request, p, builds, environments, tmp_path / "bad-registration.json")
    assert not (tmp_path / "bad-registration.json").exists()


@pytest.mark.parametrize("failure", ["missing", "duplicate", "identity", "shape", "component", "path", "exactness", "false_exact", "numerics", "samples", "repeat", "placement", "warmup", "timing_group", "allocation_peak", "case_duration", "rss", "scope"])
def test_native_reports_reject_resealed_invalid_claims(tmp_path, failure):
    p = policy(tmp_path)
    reg = registration(p)
    raw = copy.deepcopy(raw_population(reg)["native_client"][0])
    case = raw["cases"][0]
    if failure == "missing":
        raw["cases"].pop()
    elif failure == "duplicate":
        raw["cases"][-1] = copy.deepcopy(case)
    elif failure == "identity":
        raw["request_digest"] = "0" * 64
    elif failure == "shape":
        case["input_features"] -= 1
    elif failure == "component":
        case["component"] = "graph_state"
    elif failure == "path":
        case["path"] = "vram"
    elif failure == "exactness":
        case["requires_exact_match"] = True
    elif failure == "false_exact":
        case["exact_match"] = False
    elif failure == "numerics":
        case["actual_digest"] = "0" * 64
        case["exact_match"] = False
        case["max_abs_error"] = 1
    elif failure == "samples":
        case["latency_ns"].pop()
    elif failure == "repeat":
        case["repeat_identical"] = None
    elif failure == "placement":
        raw["cases"][1]["placement_identical"] = None
    elif failure == "warmup":
        case["warmup"] = 1
    elif failure == "timing_group":
        case["timing_group"] = "another"
    elif failure == "allocation_peak":
        raw["owned_host_peak_bytes"] = "1"
    elif failure == "case_duration":
        raw["elapsed_ns"] = "1"
    elif failure == "rss":
        raw["process_peak_rss_bytes"] = reg["runs"]["native_client"]["hard_limits"]["max_rss_bytes"] + "0"
    else:
        raw["model_quality"] = "qualified"
    with pytest.raises(ArtifactError):
        validate_native(seal(raw), reg, "native_client", p)


@pytest.mark.parametrize("failure", ["missing", "duplicate", "status", "opcode", "header", "total", "certificate", "timeout", "duration", "check", "rss"])
def test_paired_transport_reports_reject_invalid_claims(tmp_path, failure):
    p = policy(tmp_path)
    reg = registration(p)
    raw = copy.deepcopy(raw_population(reg)["transport_client"][0])
    if failure == "missing":
        raw["cases"].pop()
    elif failure == "duplicate":
        raw["cases"][1] = copy.deepcopy(raw["cases"][0])
    elif failure == "status":
        raw["cases"][-1]["status"] = 1
    elif failure == "opcode":
        raw["cases"][0]["kind"] = 11
    elif failure == "header":
        raw["cases"][0]["request_bytes"] = "63"
    elif failure == "total":
        raw["measured_response_bytes"] = "1"
    elif failure == "certificate":
        raw["peer_certificate_digest"] = "0" * 64
    elif failure == "timeout":
        raw["timeout_elapsed_ns"] -= 1
    elif failure == "duration":
        raw["elapsed_ns"] = raw["timeout_elapsed_ns"]
    elif failure == "check":
        raw["checks"][0]["passed"] = False
    else:
        raw["max_rss_bytes"] = "999999999999"
    with pytest.raises(ArtifactError):
        validate_transport(seal(raw), reg, "transport_client", p)


@pytest.mark.parametrize("failure", ["wall", "rss", "binary", "paired_payload"])
def test_complete_lifetime_supervision_and_pair_identity(tmp_path, failure):
    p = policy(tmp_path)
    reg = registration(p)
    sources = copy.deepcopy(raw_population(reg))
    raw, observed = sources["transport_client"]
    if failure == "wall":
        observed["wall_ns"] = "1"
    elif failure == "rss":
        observed["peak_rss_bytes"] = "1"
    elif failure == "binary":
        observed["binary_sha256"] = "0" * 64
    else:
        raw["cases"][0]["response_sha256"] = "0" * 64
        raw = seal(raw)
        observed["raw_digest"] = raw["digest"]
    sources["transport_client"] = raw, seal(observed)
    with pytest.raises(ArtifactError):
        _source_population(reg, sources, p)


@pytest.mark.parametrize("failure", ["uid", "euid", "dumpable", "nnp", "seccomp", "core", "memlock", "swap", "cgroup", "headroom", "locked", "cpus", "nodes", "probe_cap", "probe_identity", "probe_failure", "kernel"])
def test_injected_actual_runtime_preflight_rejects_changed_restrictions(tmp_path, failure):
    p = policy(tmp_path)
    reg = registration(p)
    pre = copy.deepcopy(preflight(reg, "client"))
    observed = pre["observation"]
    if failure in ("uid", "euid", "dumpable", "seccomp", "core"):
        key = {"core": "core_soft", "seccomp": "seccomp_mode"}.get(failure, failure)
        observed[key] = 1
    elif failure == "nnp":
        observed["no_new_privileges"] = 0
    elif failure == "memlock":
        observed["memlock_hard"] = "1"
    elif failure == "swap":
        observed["swap_limit_bytes"] = "1"
    elif failure == "cgroup":
        observed["host_limit_bytes"] = "1"
    elif failure == "headroom":
        observed["host_available_bytes"] = "1"
    elif failure == "locked":
        observed["locked_bytes"] = "1048576"
    elif failure == "cpus":
        observed["cpu_mask"] = "0"
    elif failure == "nodes":
        observed["memory_node_mask"] = "1"
    elif failure == "probe_cap":
        pre["probe"] = seal({**pre["probe"], "host_bytes": 1})
    elif failure == "probe_identity":
        pre["probe"] = seal({**pre["probe"], "environment_digest": "0" * 64})
    elif failure == "probe_failure":
        pre["probe"] = seal({**pre["probe"], "qualified": False})
    else:
        pre["probe_evidence"] = seal({**pre["probe_evidence"], "native_results": [0, 0, 0]})
        pre["probe"] = seal({**pre["probe"], "evidence_digest": pre["probe_evidence"]["digest"]})
    with pytest.raises(ArtifactError):
        validate_preflight(pre, reg, "native_client")


def test_derived_components_revalidate_every_raw_source(tmp_path):
    p = policy(tmp_path)
    reg = registration(p)
    references = write_component_fixture(tmp_path, p, reg)
    proof = read_json(tmp_path / "experts.json")
    assert len(validate_component_sources(proof, tmp_path, p)) == 9
    changed = copy.deepcopy(proof)
    changed["measurements"]["native_server.max_abs_error"] = 1
    with pytest.raises(ArtifactError, match="derived"):
        validate_component_sources(seal(changed), tmp_path, p)
    proof["raw_evidence"].pop()
    with pytest.raises(ArtifactError, match="exactly four"):
        validate_component_sources(seal(proof), tmp_path, p)
    sources = raw_population(reg)
    files = {}
    for name, (raw, supervision) in sources.items():
        files[name] = tmp_path / name
        atomic_json(files[name] / "measurements.json", raw)
        atomic_json(files[name] / "supervision.json", supervision)
    result = produce_components(reg, p, files, tmp_path / "package", role="client")
    assert result["passed"] and result["executor"] == "cuda" and result["soak_seconds"] == 0
    assert len(validate_calibration_evidence(result, tmp_path / "package", p)) >= len(references) + 9
    assert read_json(tmp_path / "package" / "calibration.json")["qualified"]


def test_locked_executable_supervision_and_failed_publication(tmp_path, monkeypatch):
    p = policy(tmp_path)
    executable = tmp_path / "qualifier with spaces"
    executable.write_text("#!/bin/sh\nexit 77\n")
    executable.chmod(0o700)
    build = seal({"binaries": {"ds4-ria-qualify": hash_file(executable)}})
    build_path = tmp_path / "build.json"
    atomic_json(build_path, build)
    reg = registration(p, server_build=build["digest"], binary=hash_file(executable))
    raw = raw_population(reg)["native_server"][0]
    pre = preflight(reg, "server")
    kwargs = {}
    for key in ("environment", "probe_config", "probe", "probe_evidence"):
        path = tmp_path / (key + ".json")
        atomic_json(path, pre[key])
        kwargs[key] = path
    kwargs["observer"] = lambda: pre["observation"]
    calls = []
    def runner(args, **options):
        calls.append((args, options))
        assert args[0] == str(executable) and options["max_stdout"] == 2 << 20
        assert options["max_stderr"] == 65536 and options["max_rss_bytes"] == 209715200
        assert read_json(args[1]) == derive_request(reg, "native_server")
        result = subprocess.CompletedProcess(args, 0, canonical({key: value for key, value in raw.items() if key != "digest"}), b"")
        result.peak_rss_bytes = 2097152
        result.rss_scope = "native wait4 complete child lifetime; descendants require their own cgroup cap"
        return result
    ticks = iter((0, 60000000))
    monkeypatch.setattr("ria.fixture_runner.time.monotonic_ns", lambda: next(ticks))
    result = execute_run(reg, "native_server", p, executable, build_path, tmp_path / "passed", runner=runner, **kwargs)
    assert result["raw_digest"] == raw["digest"] and len(calls) == 1
    assert (tmp_path / "passed" / "measurements.json").exists()
    def failed(args, **options):
        return subprocess.CompletedProcess(args, 9, b"", b"bounded failure diagnostic")
    ticks = iter((0, 60000000))
    with pytest.raises(ArtifactError, match="exited 9"):
        execute_run(reg, "native_server", p, executable, build_path, tmp_path / "failed", runner=failed, **kwargs)
    assert not (tmp_path / "failed").exists()
    assert (tmp_path / "failed.failed" / "native-stderr.log").read_bytes() == b"bounded failure diagnostic"
    executable.write_text("changed")
    with pytest.raises(ArtifactError, match="locked build"):
        execute_run(reg, "native_server", p, executable, build_path, tmp_path / "unexecuted", runner=runner, **kwargs)
    assert len(calls) == 1
