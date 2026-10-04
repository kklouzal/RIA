"""Pure inspection refusal fixtures, plus actual Compose parsing when installed."""

import copy
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from ria.container_inspection import validate_container_inspection
from ria.deployment import bootstrap
from ria.identity import ArtifactError
from test_deployment import deployment_request, record_config_only_compose_version


def inspection_fixture(request, directory):
    role, source = request["planning_request"]["role"], request["environment"]
    cuda = request["planning_request"]["executor"] == "cuda"
    return {"Config": {"User": "10001:10001", "Entrypoint": ["/usr/local/bin/ds4-server" if role == "client" else "/usr/local/bin/ds4-expert-server"],
        "Cmd": ["--config", "/etc/dwarfstar/service.json"], "StopTimeout": source["stop_grace_seconds"],
        "Healthcheck": {"Test": ["CMD", "/usr/local/bin/ds4ctl", "health", "--socket", "/run/dwarfstar/admin.sock", "--timeout-ms", "5000"],
            "Interval": 10_000_000_000, "Timeout": 6_000_000_000, "StartPeriod": source["start_period_seconds"] * 1_000_000_000, "Retries": 3},
        "Env": ["PATH=/usr/local/bin:/usr/bin:/bin"] + (["NVIDIA_VISIBLE_DEVICES=" + source["gpu_uuid"], "NVIDIA_DRIVER_CAPABILITIES=compute,utility", "CUDA_DISABLE_PTX_JIT=1"] if cuda else [])},
        "HostConfig": {"Memory": source["cgroup_bytes"], "MemorySwap": source["cgroup_bytes"], "MemoryReservation": 0,
            "PidsLimit": source["pids_limit"], "CpusetCpus": source["cpuset"], "Runtime": "nvidia" if cuda else "runc",
            "ReadonlyRootfs": True, "Privileged": False, "Init": True, "NetworkMode": "ria-" + role + "_default",
            "PidMode": "", "IpcMode": "private", "CgroupnsMode": "private", "UTSMode": "", "OomScoreAdj": 0,
            "CapAdd": None, "CapDrop": ["ALL"], "Devices": [], "DeviceCgroupRules": None, "OomKillDisable": False,
            "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0}, "Ulimits": [{"Name": name, "Soft": count, "Hard": count}
                for name, count in (("core", 0), ("memlock", source["memlock_bytes"]))],
            "SecurityOpt": ["no-new-privileges:true", "seccomp:" + source["seccomp_profile"]],
            "Tmpfs": {"/run/dwarfstar": "rw,noexec,nosuid,nodev,size=16m,mode=0700,uid=10001,gid=10001", "/tmp": "rw,noexec,nosuid,nodev,size=64m,mode=1777"},
            "PublishAllPorts": False, "PortBindings": {"8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": str(source["api_port"])}]} if role == "client" else {
                f"{port}/tcp": [{"HostIp": source["bind_ip"], "HostPort": str(port)}] for port in (7443, 7444)},
            "DeviceRequests": [{"Driver": "nvidia", "Count": 0, "DeviceIDs": [source["gpu_uuid"]], "Capabilities": [["gpu"]], "Options": {}}] if cuda else None},
        "Mounts": [{"Type": "bind", "Source": path, "Destination": target, "RW": writable, "Propagation": "rprivate"}
            for target, path, writable in (("/model", source["model_dir"], False), ("/etc/dwarfstar", str(directory), False),
                ("/run/secrets", source["secret_dir"], False), ("/artifacts", source["report_dir"], True))]}


@pytest.mark.parametrize("role,executor", [("expert", "cpu"), ("expert", "cuda"), ("client", "cuda")])
def test_actual_inspection_contract_and_refusal(tmp_path, role, executor):
    request = deployment_request(tmp_path, role, executor)
    observed = inspection_fixture(request, tmp_path)
    assert validate_container_inspection(observed, request, tmp_path) == observed
    observed["HostConfig"]["Ulimits"].append({"Name": "nofile", "Soft": 1048576, "Hard": 1048576})
    assert validate_container_inspection(observed, request, tmp_path) == observed
    for field, value in (("Privileged", True), ("MemorySwap", 0), ("Memory", 1), ("CapAdd", ["SYS_ADMIN"]),
        ("ReadonlyRootfs", False), ("PidsLimit", 0), ("CpusetCpus", "0"), ("SecurityOpt", ["seccomp=unconfined"]),
        ("Ulimits", []), ("Tmpfs", {}), ("NetworkMode", "host"), ("PortBindings", {}), ("DeviceRequests", [])):
        if field == "DeviceRequests" and executor == "cpu":
            value = [{"Driver": "nvidia"}]
        bad = copy.deepcopy(observed)
        bad["HostConfig"][field] = value
        with pytest.raises(ArtifactError):
            validate_container_inspection(bad, request, tmp_path)


@pytest.mark.parametrize("role,executor", [("expert", "cpu"), ("expert", "cuda"), ("client", "cuda")])
def test_real_compose_config_only(tmp_path, role, executor):
    if shutil.which("docker") is None:
        pytest.skip("Docker CLI/Compose is a separately installed offline dependency")
    request = deployment_request(tmp_path, role, executor)
    root = Path(__file__).resolve().parents[2]
    request["compose_files"] = [str(root / "deploy" / ("compose.client.yml" if role == "client" else "compose.expert.yml"))]
    if role == "expert" and executor == "cuda":
        request["compose_files"].append(str(root / "deploy/compose.expert-cuda.yml"))
    request["deadline_ms"] = 10000
    record_config_only_compose_version(request)
    report = bootstrap(request, tmp_path / "actual-compose")
    assert report["status"] == "not_admitted" and not report["admitted"]
