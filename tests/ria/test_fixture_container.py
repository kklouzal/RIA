"""Bounded bootstrap container ownership; actual Compose parsing only, no daemon."""

import copy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import os
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from ria.deployment import (bootstrap, fixture_start, fixture_stop, fixture_exec,
    compose_arguments, controller_lock, validate_effective_compose, _run)
from ria.identity import ArtifactError, atomic_bytes, canonical, read_json, seal
from ria.host import revalidate_host_report
from test_container_inspection import inspection_fixture
from test_deployment import compose_fixture, deployment_request, record_config_only_compose_version


@pytest.fixture(autouse=True)
def isolated_controller_locks(tmp_path, monkeypatch):
    monkeypatch.setattr("ria.deployment.CONTROLLER_LOCK_DIRECTORY", tmp_path)
    # Managed-container tests inject synthetic host observations explicitly.
    monkeypatch.setattr("ria.host.revalidate_host_report", lambda report: report)


def test_role_controller_deadline_and_persistent_inode(tmp_path):
    request = {"planning_request": {"role": "expert"}, "deadline_ms": 25}
    with controller_lock(request, directory=tmp_path):
        with pytest.raises(ArtifactError, match="deadline"):
            with controller_lock(request, directory=tmp_path):
                pytest.fail("a second controller must not run")
        other = {"planning_request": {"role": "client"}, "deadline_ms": 25}
        with controller_lock(other, directory=tmp_path):
            pass
    path = tmp_path / "ria-expert.controller.lock"
    inode = path.stat().st_ino
    with controller_lock(request, directory=tmp_path):
        pass
    assert path.stat().st_ino == inode and path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("kind", ["symlink", "fifo", "hardlink", "permissions"])
def test_controller_refuses_unsafe_lock_inodes(tmp_path, kind):
    path = tmp_path / "ria-expert.controller.lock"
    if kind == "fifo":
        os.mkfifo(path, 0o600)
    elif kind == "symlink":
        path.symlink_to(tmp_path / "absent")
    else:
        path.write_bytes(b"")
        path.chmod(0o600)
        if kind == "hardlink":
            os.link(path, tmp_path / "alias")
        else:
            path.chmod(0o644)
    with pytest.raises((ArtifactError, OSError)):
        with controller_lock({"planning_request": {"role": "expert"}, "deadline_ms": 25}, directory=tmp_path):
            pytest.fail("an unsafe controller lock must not authorize work")


def qualification_compose(request, directory):
    value = compose_fixture(request, directory)
    service = value["services"][request["planning_request"]["role"]]
    service.update(entrypoint=["/usr/bin/sleep"], command=["1800"], healthcheck={"disable": True})
    return value


def mock_engine(request, directory, *, bad=False, exists=False):
    state = {"running": False, "exists": exists}
    calls = []
    def runner(args, deadline, cwd=None):
        calls.append(args)
        if args[-2:] == ["--format", "json"]:
            return canonical(qualification_compose(request, directory) if "qualification.override.yml" in " ".join(args) else compose_fixture(request, directory))
        if args[-3:] == ["up", "-d", request["planning_request"]["role"]]:
            state.update(running=True, exists=True)
        if "ps" in args:
            return b"f" * 64 if state["exists"] else b""
        if "inspect" in args:
            observed = inspection_fixture(request, directory)
            observed.update(Id="f" * 64, State={"Running": state["running"], "Pid": 1234,
                "StartedAt": datetime.now(timezone.utc).isoformat()})
            observed["Config"].update(Image=request["environment"]["image"], Entrypoint=["/usr/bin/sleep"], Cmd=["1800"],
                Healthcheck={"Test": ["NONE"]}, Labels={"com.docker.compose.service": request["planning_request"]["role"],
                                                       "com.docker.compose.project": "ria-" + request["planning_request"]["role"]})
            if bad:
                observed["HostConfig"]["Privileged"] = True
            return canonical(observed)
        if "stop" in args:
            state["running"] = False
        if "rm" in args:
            state["exists"] = False
        return b""
    return runner, calls, state


def test_fixture_container_is_inspected_before_exec_and_owned_stop(tmp_path, monkeypatch):
    request = deployment_request(tmp_path)
    directory = tmp_path / "bootstrap"
    runner, calls, state = mock_engine(request, directory)
    bootstrap(request, directory, runner=runner)
    monkeypatch.setattr("ria.host.verify_container_ancestors", lambda pid, report: seal({"classification": "synthetic offline ancestor fixture", "pid": pid}))
    lock = fixture_start(request, directory, runner=runner)
    assert state["running"] and lock["status"] == "running"
    before = len(calls)
    assert fixture_start(request, directory, runner=runner) == lock
    assert not any("up" in args for args in calls[before:])
    fixture_exec(request, directory, "probe", runner=runner)
    executed = [index for index, args in enumerate(calls) if "exec" in args]
    assert len(executed) == 1 and "inspect" in calls[executed[0] - 1]
    assert calls[executed[0]][-2:] == ["--output", "/artifacts/probe.json"]
    result = fixture_stop(request, directory, runner=runner)
    assert result["status"] == "stopped" and not state["running"] and not state["exists"]
    assert fixture_stop(request, directory, runner=runner) == result
    assert read_json(Path(request["environment"]["report_dir"]) / ("fixture-" + lock["bootstrap_digest"] + ".json"))["status"] == "stopped"


@pytest.mark.parametrize("phase", ["start", "exec"])
def test_changed_host_rejected_before_container_mutation(tmp_path, monkeypatch, phase):
    request = deployment_request(tmp_path)
    directory = tmp_path / "bootstrap"
    runner, calls, state = mock_engine(request, directory)
    bootstrap(request, directory, runner=runner)
    monkeypatch.setattr("ria.host.verify_container_ancestors", lambda pid, report: seal({"classification": "synthetic offline ancestor fixture", "pid": pid}))
    if phase == "exec":
        fixture_start(request, directory, runner=runner)
    def changed_host(_):
        raise ArtifactError("current stable host realization changed")
    monkeypatch.setattr("ria.host.revalidate_host_report", changed_host)
    before = list(calls)
    with pytest.raises(ArtifactError, match="current stable host realization changed"):
        if phase == "start":
            fixture_start(request, directory, runner=runner)
        else:
            fixture_exec(request, directory, "probe", runner=runner)
    assert calls == before
    if phase == "exec":
        assert fixture_stop(request, directory, runner=runner)["status"] == "stopped"
        assert not state["running"]


@pytest.mark.parametrize("field", ["kernel", "engine", "compose", "driver", "gpu", "numa", "parent"])
def test_stable_host_revalidation_authenticates_every_observed_field(field):
    baseline = seal({"schema_revision": 1, "kind": "host_preflight", "architecture": "x86_64",
        "kernel_version": "fixture", "docker_version": "fixture", "compose_version": "fixture",
        "cgroup_driver": "systemd", "rootful": True, "cgroup_version": 2,
        "parent_ancestors": [{"path": "/sys/fs/cgroup/fixture", "memory.max": "1048576"}],
        "numa": [{"node": 0, "cpus": "0-1", "total_bytes": "1048576", "distance": [10]}],
        "physical_gpus": [{"uuid": "GPU-12345678-1234-1234-1234-123456789abc", "name": "fixture", "driver_version": "fixture"}],
        "selected_gpu_uuid": "GPU-12345678-1234-1234-1234-123456789abc", "client": False, "hardware_qualified": False})
    calls = []
    def observer(parent, gpu, *, client):
        calls.append((parent, gpu, client))
        return baseline
    assert revalidate_host_report(baseline, observer=observer) == baseline
    assert calls == [("/sys/fs/cgroup/fixture", baseline["selected_gpu_uuid"], False)]
    changed = copy.deepcopy(baseline)
    if field in ("kernel", "engine", "compose"):
        changed[{"kernel": "kernel_version", "engine": "docker_version", "compose": "compose_version"}[field]] = "changed"
    elif field in ("driver", "gpu"):
        changed["physical_gpus"][0]["driver_version" if field == "driver" else "name"] = "changed"
    elif field == "numa":
        changed["numa"][0]["distance"] = [11]
    else:
        changed["parent_ancestors"][0]["memory.max"] = "524288"
    with pytest.raises(ArtifactError, match="fresh baseline and probe"):
        revalidate_host_report(baseline, observer=lambda *args, **kwargs: seal(changed))


def test_existing_service_and_failed_inspection_cannot_execute(tmp_path, monkeypatch):
    request = deployment_request(tmp_path)
    directory = tmp_path / "bootstrap"
    runner, _, _ = mock_engine(request, directory)
    bootstrap(request, directory, runner=runner)
    existing, calls, _ = mock_engine(request, directory, exists=True)
    with pytest.raises(ArtifactError, match="replace"):
        fixture_start(request, directory, runner=existing)
    assert not any("up" in args for args in calls)
    bad, calls, state = mock_engine(request, directory, bad=True)
    monkeypatch.setattr("ria.host.verify_container_ancestors", lambda *_: pytest.fail("invalid inspection must not reach host observation"))
    with pytest.raises(ArtifactError, match="stopped"):
        fixture_start(request, directory, runner=bad)
    assert not state["running"] and any("stop" in args for args in calls)
    assert not any("exec" in args for args in calls)


@pytest.mark.parametrize("failure", ["entrypoint", "lifetime", "health", "memory", "port"])
def test_qualification_override_has_only_one_bounded_mode(tmp_path, failure):
    request = deployment_request(tmp_path)
    value = qualification_compose(request, tmp_path)
    service = value["services"]["expert"]
    if failure == "entrypoint":
        service["entrypoint"] = ["sh"]
    elif failure == "lifetime":
        service["command"] = ["1801"]
    elif failure == "health":
        service["healthcheck"] = {"disable": True, "test": ["NONE"]}
    elif failure == "memory":
        service["memswap_limit"] = 0
    else:
        service["ports"][0]["host_ip"] = "0.0.0.0"
    with pytest.raises(ArtifactError):
        validate_effective_compose(value, request, tmp_path, qualification=True)


@pytest.mark.parametrize("role,executor", [("expert", "cpu"), ("expert", "cuda"), ("client", "cuda")])
def test_actual_qualification_compose_override_config_only(tmp_path, role, executor):
    if shutil.which("docker") is None:
        pytest.skip("offline Compose CLI separately installed")
    request = deployment_request(tmp_path, role, executor)
    root = Path(__file__).resolve().parents[2]
    request["compose_files"] = [str(root / "deploy" / ("compose.client.yml" if role == "client" else "compose.expert.yml"))]
    if role == "expert" and executor == "cuda":
        request["compose_files"].append(str(root / "deploy/compose.expert-cuda.yml"))
    request["deadline_ms"] = 10000
    record_config_only_compose_version(request)
    directory = tmp_path / "actual-compose"
    bootstrap(request, directory)
    override = directory / "qualification.override.yml"
    atomic_bytes(override, f"services:\n  {role}:\n    entrypoint: [/usr/bin/sleep]\n    command: ['1800']\n    healthcheck: !override\n      disable: true\n".encode())
    from ria.identity import loads
    value = loads(_run(compose_arguments(request, directory, directory / (role + ".env"), "-f", str(override), "config", "--format", "json"), 10000, cwd=directory))
    assert validate_effective_compose(value, request, directory, qualification=True) == value
    assert value["services"][role]["healthcheck"] == {"disable": True}
    altered = copy.deepcopy(value)
    altered["services"][role]["command"] = ["3600"]
    with pytest.raises(ArtifactError):
        validate_effective_compose(altered, request, directory, qualification=True)
