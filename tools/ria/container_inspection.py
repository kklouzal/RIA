"""Compare an owned Engine container with the approved host-local policy."""

from pathlib import Path

from .identity import ArtifactError, canonical, loads, read_json


def validate_container_inspection(observed, request, config_dir, *, qualification=False):
    role = request["planning_request"]["role"]
    source = request["environment"]
    cuda = request["planning_request"]["executor"] == "cuda"
    config, host = observed.get("Config", {}), observed.get("HostConfig", {})
    expected = {"User": "10001:10001", "Entrypoint": ["/usr/local/bin/ds4-server" if role == "client" else "/usr/local/bin/ds4-expert-server"],
                "Cmd": ["--config", "/etc/dwarfstar/service.json"]}
    if qualification:
        expected.update(Entrypoint=["/usr/bin/sleep"], Cmd=["1800"])
    if any(config.get(key) != value for key, value in expected.items()):
        raise ArtifactError("actual container native entry point or UID differs from approval")
    expected = {"Memory": source["cgroup_bytes"], "MemorySwap": source["cgroup_bytes"],
        "MemoryReservation": 0, "PidsLimit": source["pids_limit"], "CpusetCpus": source["cpuset"],
        "Runtime": "nvidia" if cuda else "runc", "ReadonlyRootfs": True, "Privileged": False,
        "Init": True, "NetworkMode": "ria-" + role + "_default", "PidMode": "", "IpcMode": "private",
        "CgroupnsMode": "private", "UTSMode": "", "OomScoreAdj": 0}
    if any(host.get(key) != value for key, value in expected.items()):
        raise ArtifactError("actual container resource, privilege or namespace policy differs from approval")
    if (host.get("CapAdd") or host.get("CapDrop") != ["ALL"] or host.get("Devices") or
            host.get("DeviceCgroupRules") or host.get("OomKillDisable") or
            host.get("RestartPolicy") != {"Name": "no", "MaximumRetryCount": 0}):
        raise ArtifactError("actual container capability/device/restart policy changed")
    limits = host.get("Ulimits", [])
    names = [item.get("Name") for item in limits]
    if (len(set(names)) != len(names) or not {"core", "memlock"} <= set(names) or
            set(names) - {"core", "memlock", "nofile"} or
            any(item.get("Soft") != (0 if item["Name"] == "core" else source["memlock_bytes"]) or
                item.get("Hard") != (0 if item["Name"] == "core" else source["memlock_bytes"])
                for item in limits if item["Name"] in ("core", "memlock")) or
            any(type(item.get("Soft")) is not int or type(item.get("Hard")) is not int or
                not 0 < item["Soft"] <= item["Hard"] <= 9007199254740991
                for item in limits if item["Name"] == "nofile")):
        raise ArtifactError("actual container dump/lock limits changed")
    security = host.get("SecurityOpt", [])
    privileges = [value for value in security if value in ("no-new-privileges", "no-new-privileges:true", "no-new-privileges=true")]
    seccomp = [value for value in security if value.startswith(("seccomp:", "seccomp="))]
    if len(security) != 2 or len(privileges) != 1 or len(seccomp) != 1:
        raise ArtifactError("actual container security options changed")
    value = seccomp[0][8:]
    if value != source["seccomp_profile"]:
        # Engine inspection commonly contains the inline profile sent by the
        # Compose client rather than its host pathname. Compare its full data.
        if canonical(loads(value, max_bytes=256 << 10)) != canonical(read_json(source["seccomp_profile"])):
            raise ArtifactError("actual container seccomp profile differs from approval")
    expected_tmpfs = {"/run/dwarfstar": "rw,noexec,nosuid,nodev,size=16m,mode=0700,uid=10001,gid=10001",
                      "/tmp": "rw,noexec,nosuid,nodev,size=64m,mode=1777"}
    if host.get("Tmpfs") != expected_tmpfs:
        raise ArtifactError("actual container tmpfs policy changed")
    mounts = observed.get("Mounts", [])
    binds = [item for item in mounts if item.get("Type") == "bind"]
    allowed = {"/model": (source["model_dir"], False), "/etc/dwarfstar": (str(Path(config_dir)), False),
               "/run/secrets": (source["secret_dir"], False), "/artifacts": (source["report_dir"], True)}
    if len(binds) != 4 or {item.get("Destination") for item in binds} != set(allowed):
        raise ArtifactError("actual container bind population changed")
    for mount in binds:
        path, writable = allowed[mount["Destination"]]
        if mount.get("Source") != path or mount.get("RW") is not writable or mount.get("Propagation") != "rprivate":
            raise ArtifactError("actual container bind source/access/propagation changed")
    if any(item.get("Type") != "bind" and
           (item.get("Type") != "tmpfs" or item.get("Destination") not in expected_tmpfs) for item in mounts):
        raise ArtifactError("actual container contains an unapproved mount")
    ports = {"8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": str(source["api_port"])}]} if role == "client" else {
        f"{port}/tcp": [{"HostIp": source["bind_ip"], "HostPort": str(port)}] for port in (7443, 7444)}
    if host.get("PortBindings") != ports or host.get("PublishAllPorts"):
        raise ArtifactError("actual container published listeners changed")
    health = {"Test": ["CMD", "/usr/local/bin/ds4ctl", "health", "--socket", "/run/dwarfstar/admin.sock", "--timeout-ms", "5000"],
              "Interval": 10_000_000_000, "Timeout": 6_000_000_000,
              "StartPeriod": source["start_period_seconds"] * 1_000_000_000, "Retries": 3}
    actual_health = config.get("Healthcheck", {})
    if qualification:
        health["Test"] = ["NONE"]
    valid_health = ((qualification and actual_health == {"Test": ["NONE"]}) or
        not (any(actual_health.get(key) != value for key, value in health.items()) or
             set(actual_health) - set(health) - {"StartInterval"} or actual_health.get("StartInterval", 0) != 0))
    if not valid_health or config.get("StopTimeout") != source["stop_grace_seconds"]:
        raise ArtifactError("actual container health/startup/drain policy changed")
    values = config.get("Env", [])
    if any(not isinstance(item, str) or "=" not in item for item in values):
        raise ArtifactError("actual container environment is malformed")
    environment = dict(item.split("=", 1) for item in values)
    if len(environment) != len(values):
        raise ArtifactError("actual container has ambiguous duplicate environment keys")
    devices = host.get("DeviceRequests") or []
    if cuda:
        if (len(devices) != 1 or devices[0].get("Driver") != "nvidia" or devices[0].get("Count") != 0 or
                devices[0].get("DeviceIDs") != [source["gpu_uuid"]] or
                devices[0].get("Capabilities") != [["gpu"]] or devices[0].get("Options") not in ({}, None) or
                environment.get("NVIDIA_VISIBLE_DEVICES") != source["gpu_uuid"] or
                environment.get("NVIDIA_DRIVER_CAPABILITIES") != "compute,utility" or
                environment.get("CUDA_DISABLE_PTX_JIT") != "1"):
            raise ArtifactError("actual container GPU identity/runtime policy changed")
    elif devices or any(name.startswith("NVIDIA_") for name in environment):
        raise ArtifactError("actual CPU container contains a GPU dependency")
    return observed
