"""Two-stage deployment orchestration around the sole native admission engine."""

import ipaddress
import os
import re
import shutil
import stat
import subprocess
import tempfile
import fcntl
import time
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timezone

from .identity import (ArtifactError, atomic_bytes, atomic_json, digest,
                       hash_file, read_json, seal, sync_directory, verify_identity)
from .schemas import tls_enabled, validate

DEPLOY_VARIABLE = re.compile(r"(?:CLIENT|EXPERT|SECCOMP)_")
CONTROLLER_LOCK_DIRECTORY = Path("/run/lock")


@contextmanager
def controller_lock(request, *, directory=None):
    """One authorized host owner serializes each exact Compose project role.

    The regular0600 inode persists across sessions and is never unlinked, so a
    waiter cannot acquire a replacement inode while another owner is active.
    LOCK_NB polling is bounded by the caller's explicit management deadline.
    """
    planning = request.get("planning_request")
    role = planning.get("role") if isinstance(planning, dict) else None
    if role not in ("expert", "client"):
        raise ArtifactError("unknown controller role")
    if type(request.get("deadline_ms")) is not int or not 0 < request["deadline_ms"] <= 3600000:
        raise ArtifactError("role controller deadline must be positive and at most one hour")
    root = Path(directory or CONTROLLER_LOCK_DIRECTORY)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ArtifactError("controller lock directory must be an existing absolute real directory")
    path = root / ("ria-" + role + ".controller.lock")
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    locked = False
    try:
        observed = os.fstat(fd)
        if not stat.S_ISREG(observed.st_mode) or observed.st_uid != os.geteuid() or observed.st_nlink != 1 or stat.S_IMODE(observed.st_mode) != 0o600:
            raise ArtifactError("controller lock must be a regular one-link0600 file owned by the authorized controller")
        deadline = time.monotonic() + request["deadline_ms"] / 1000
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
                break
            except BlockingIOError as exc:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ArtifactError("another role controller operation remained active past the management deadline") from exc
                time.sleep(min(.025, remaining))
        current = os.stat(path, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (observed.st_dev, observed.st_ino):
            raise ArtifactError("controller lock inode changed during acquisition")
        yield
    finally:
        try:
            if locked:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def controlled_environment():
    # Explicit env-file remains data; neither exported deployment overrides nor a
    # process's HOME/.env/COMPOSE_FILE can alter the reviewed application model.
    return {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}


def _run(arguments, deadline_ms, *, cwd=None):
    from .process import run_bounded
    try:
        result = run_bounded(arguments, timeout=deadline_ms / 1000, cwd=cwd,
                             env=controlled_environment(), max_stdout=16 << 20)
    except subprocess.TimeoutExpired as exc:
        raise ArtifactError("command deadline exceeded") from exc
    except OSError as exc:
        raise ArtifactError(f"bounded command failed: {arguments[0]}: {exc}") from exc
    if result.returncode:
        message = result.stderr[:8192].decode("utf-8", errors="replace")
        raise ArtifactError(f"{arguments[0]} exited {result.returncode}: {message}")
    return result.stdout


def _mask(value, maximum):
    if not isinstance(value, str) or len(value) > 4096 or not re.fullmatch(r"[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*", value):
        raise ArtifactError("invalid bounded CPU/NUMA mask")
    result = set()
    for component in value.split(","):
        ends = component.split("-")
        begin, end = int(ends[0]), int(ends[-1])
        if end < begin or end > maximum:
            raise ArtifactError("CPU/NUMA mask exceeds target bounds")
        result.update(range(begin, end + 1))
    return result


def _seconds(value):
    # compose-go serializes time.Duration using Go's canonical h/m/s form.
    match = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", value if isinstance(value, str) else "")
    if match is None or not any(part is not None for part in match.groups()):
        raise ArtifactError("Compose requires a bounded whole-second duration")
    result = sum(int(part or 0) * factor for part, factor in zip(match.groups(), (3600, 60, 1), strict=True))
    if result > 9223372036:
        raise ArtifactError("Compose duration exceeds the native duration range")
    return result


def validate_paths(request):
    environment = request["environment"]
    for key in ("model_dir", "secret_dir", "report_dir"):
        path = Path(environment[key])
        if not path.is_absolute() or path.is_symlink() or not path.is_dir():
            raise ArtifactError(f"{key} must be an existing absolute host directory")
    seccomp = Path(environment["seccomp_profile"])
    if seccomp.is_symlink() or not seccomp.is_file():
        raise ArtifactError("reviewed seccomp profile is absent")
    profile = read_json(seccomp)
    if profile.get("defaultAction") not in ("SCMP_ACT_ERRNO", "SCMP_ACT_KILL", "SCMP_ACT_KILL_PROCESS"):
        raise ArtifactError("seccomp must retain a default deny policy")
    secret_root = Path(environment["secret_dir"])
    credentials = ("ca.pem", "peer.pem", "peer.key") if tls_enabled(request["tls"]) else ()
    credentials += ("api.token",) if request["planning_request"]["role"] == "client" else ()
    for name in credentials:
        path = secret_root / name
        if path.is_symlink() or not path.is_file() or not os.access(path, os.R_OK):
            raise ArtifactError(f"required credential {name} is unreadable")
        mode = stat.S_IMODE(path.stat().st_mode)
        if name in ("peer.key", "api.token") and mode & 0o007:
            raise ArtifactError("private credentials must not be world-readable")
    ip = ipaddress.ip_address(environment["bind_ip"])
    if ip.version != 4 or (request["planning_request"]["role"] == "expert" and ip.is_unspecified):
        raise ArtifactError("expert bind requires an explicit private IPv4 address")
    for path in request["compose_files"]:
        if Path(path).is_symlink() or not Path(path).is_file():
            raise ArtifactError("Compose source must be an existing reviewed regular file")
    ctl = Path(request["native_ctl"])
    if not ctl.is_file() or ctl.is_symlink() or not os.access(ctl, os.X_OK):
        raise ArtifactError("native ds4ctl must be an executable regular file")
    host_report = read_json(request["host_report"])
    verify_identity(host_report, environment["host_report_digest"])
    if host_report.get("kind") != "host_preflight" or host_report.get("rootful") is not True or host_report.get("cgroup_version") != 2:
        raise ArtifactError("deployment requires the actual rootful cgroup-v2 host baseline")
    for field in ("kernel_version", "docker_version", "compose_version"):
        if host_report.get(field) != environment[field]:
            raise ArtifactError("deployment host versions differ from the sealed preflight baseline")
    if host_report.get("selected_gpu_uuid") != environment["gpu_uuid"] or host_report.get("client") != (request["planning_request"]["role"] == "client"):
        raise ArtifactError("host baseline selects a different physical GPU/role")


def frozen_environment(request):
    """All measured realization settings, without report/config identity cycles."""
    policies = {}
    for field, kind in (("peer_grants", "peer-grants"), ("placement_plan", "placement-plan")):
        if field in request:
            document = read_json(request[field])
            validate(kind, document)
            verify_identity(document)
            policies[field + "_digest"] = document["digest"]
    return {"schema_revision": 1, "environment": request["environment"],
        "planning_request": request["planning_request"], "tls": request["tls"],
        "network": request["network"], "seccomp_sha256": hash_file(request["environment"]["seccomp_profile"]),
        **policies,
        **{field: request[field] for field in ("api", "expert") if field in request}}


def probe_configuration(request):
    return {**request["probe_config"], "environment_digest": digest(frozen_environment(request)),
            "build_digest": request["environment"]["build_digest"]}


def environment_bytes(request, config_dir):
    role = request["planning_request"]["role"].upper()
    source = request["environment"]
    values = {f"{role}_IMAGE": source["image"], f"{role}_CPUSET": source["cpuset"],
        f"{role}_CGROUP_BYTES": str(source["cgroup_bytes"]), f"{role}_MEMLOCK_BYTES": str(source["memlock_bytes"]),
        f"{role}_PIDS_LIMIT": str(source["pids_limit"]), f"{role}_MODEL_DIR": source["model_dir"],
        f"{role}_CONFIG_DIR": str(config_dir), f"{role}_SECRET_DIR": source["secret_dir"],
        f"{role}_REPORT_DIR": source["report_dir"], f"{role}_START_PERIOD": str(source["start_period_seconds"]) + "s",
        f"{role}_STOP_GRACE": str(source["stop_grace_seconds"]) + "s", "SECCOMP_PROFILE": source["seccomp_profile"]}
    if source["gpu_uuid"]:
        values[f"{role}_GPU_UUID"] = source["gpu_uuid"]
    if role == "EXPERT":
        values["EXPERT_BIND_IP"] = source["bind_ip"]
    else:
        values["CLIENT_API_PORT"] = str(source["api_port"])
    # Compose's env grammar expands $. Quotes do not provide a safe general
    # encoding for arbitrary host values; reject ambiguous control/metacharacters.
    for value in values.values():
        if any(character in value for character in "\n\r\x00$#'\"\\") or value != value.strip():
            raise ArtifactError("deployment env value cannot be represented safely")
    return "".join(f"{key}={value}\n" for key, value in sorted(values.items())).encode()


def compose_arguments(request, directory, env_file, *operation):
    args = ["docker", "--host", "unix:///var/run/docker.sock", "compose", "--project-directory", str(directory), "--env-file", str(env_file)]
    for path in request["compose_files"]:
        args.extend(("-f", path))
    args.extend(operation)
    return args


def validate_effective_compose(document, request, config_dir, *, qualification=False):
    role = request["planning_request"]["role"]
    if set(document) - {"name", "services", "networks"} or document.get("name") != "ria-" + role:
        raise ArtifactError("Compose project identity or top-level policy changed")
    if set(document.get("services", {})) != {role}:
        raise ArtifactError("Compose must select exactly one host-local service")
    service = document["services"][role]
    allowed = {"image", "platform", "entrypoint", "command", "user", "runtime", "init", "read_only", "restart",
               "cpuset", "mem_limit", "memswap_limit", "pids_limit", "cap_drop", "security_opt", "ulimits", "tmpfs",
               "volumes", "ports", "healthcheck", "stop_grace_period", "environment", "deploy", "networks"}
    if set(service) - allowed:
        raise ArtifactError("effective Compose contains an unreviewed service policy")
    source = request["environment"]
    entrypoint = ["/usr/bin/sleep"] if qualification else ["/usr/local/bin/ds4-server" if role == "client" else "/usr/local/bin/ds4-expert-server"]
    command = ["1800"] if qualification else ["--config", "/etc/dwarfstar/service.json"]
    if service.get("entrypoint") != entrypoint or service.get("command") != command or service.get("platform") != "linux/amd64":
        raise ArtifactError("native entry point or target platform changed")
    expected = {"image": source["image"], "user": "10001:10001", "read_only": True, "init": True,
                "restart": "no", "cpuset": source["cpuset"], "pids_limit": source["pids_limit"],
                "mem_limit": source["cgroup_bytes"], "memswap_limit": source["cgroup_bytes"]}
    for key, value in expected.items():
        actual = service.get(key)
        if key in ("mem_limit", "memswap_limit") and isinstance(actual, str) and re.fullmatch(r"[1-9][0-9]*", actual):
            actual = int(actual)
        if actual != value:
            raise ArtifactError(f"effective Compose changed {key}")
    if service.get("privileged") or service.get("network_mode") == "host" or service.get("pid") == "host" or service.get("ipc") == "host" or service.get("devices") or service.get("device_cgroup_rules"):
        raise ArtifactError("forbidden host privilege/namespace")
    if service.get("cap_add") or set(service.get("cap_drop", [])) != {"ALL"}:
        raise ArtifactError("capability policy changed")
    security = service.get("security_opt", [])
    if len(security) != 2 or not any(value in ("no-new-privileges:true", "no-new-privileges") for value in security) or not any(value == "seccomp:" + source["seccomp_profile"] for value in security):
        raise ArtifactError("security options changed")
    limits = service.get("ulimits", {})
    if set(limits) != {"memlock", "core"} or limits.get("memlock") != {"soft": source["memlock_bytes"], "hard": source["memlock_bytes"]} or limits.get("core") not in ({}, {"soft": 0, "hard": 0}):
        raise ArtifactError("memory lock or core-dump policy changed")
    expected_tmpfs = ["/run/dwarfstar:rw,noexec,nosuid,nodev,size=16m,mode=0700,uid=10001,gid=10001",
                      "/tmp:rw,noexec,nosuid,nodev,size=64m,mode=1777"]
    if service.get("tmpfs") != expected_tmpfs:
        raise ArtifactError("bounded runtime temporary filesystem policy changed")
    health = service.get("healthcheck", {})
    expected_health = {"test": ["CMD", "/usr/local/bin/ds4ctl", "health", "--socket", "/run/dwarfstar/admin.sock", "--timeout-ms", "5000"],
                       "interval": "10s", "timeout": "6s", "retries": 3, "start_period": str(source["start_period_seconds"]) + "s"}
    health_valid = health == {"disable": True} if qualification else set(health) == set(expected_health) and all(health.get(key) == expected_health[key] for key in ("test", "retries")) and all(_seconds(health.get(key)) == _seconds(expected_health[key]) for key in ("interval", "timeout", "start_period"))
    if not health_valid or _seconds(service.get("stop_grace_period")) != source["stop_grace_seconds"]:
        raise ArtifactError("bounded health/startup/drain policy changed")
    if service.get("networks", {"default": None}) not in ({"default": None}, {"default": {}}):
        raise ArtifactError("service must use its isolated default bridge network")
    networks = document.get("networks", {"default": {"name": "ria-" + role + "_default"}})
    if networks not in ({"default": {"name": "ria-" + role + "_default"}}, {"default": {"name": "ria-" + role + "_default", "ipam": {}}}):
        raise ArtifactError("Compose default network policy changed")
    expected_volumes = {"/model": (source["model_dir"], True), "/etc/dwarfstar": (str(config_dir), True),
                        "/run/secrets": (source["secret_dir"], True), "/artifacts": (source["report_dir"], False)}
    # Released Compose 2 (through 2.40.3) uses bool/omitempty: a present empty
    # bind object omits false. Compose 5's OptOut/omitzero omits true instead.
    # Keep nil/missing bind distinct: Compose 2 permits legacy path creation.
    version = re.fullmatch(r"2\.(0|[1-9][0-9]?)\.(0|[1-9][0-9]{0,2})", source["compose_version"])
    empty_bind_is_false = version is not None and tuple(map(int, version.groups())) <= (40, 3)
    volumes = service.get("volumes", [])
    if len(volumes) != 4 or {volume.get("target") for volume in volumes} != set(expected_volumes):
        raise ArtifactError("unexpected mount population")
    for volume in volumes:
        if volume.get("type") != "bind" or volume.get("target") not in expected_volumes:
            raise ArtifactError("unauthorized mount")
        path, readonly = expected_volumes[volume["target"]]
        bind = volume.get("bind")
        bind_valid = isinstance(bind, dict) and ((not bind and empty_bind_is_false) or (set(bind) == {"create_host_path"} and bind["create_host_path"] is False))
        read_only = volume.get("read_only", False)
        if set(volume) - {"type", "source", "target", "read_only", "bind"} or volume.get("source") != path or type(read_only) is not bool or read_only != readonly or not bind_valid:
            raise ArtifactError("mount source/access changed")
    cuda = request["planning_request"]["executor"] == "cuda"
    devices = service.get("deploy", {}).get("resources", {}).get("reservations", {}).get("devices", [])
    if cuda:
        if service.get("deploy") not in ({"resources": {"reservations": {"devices": devices}}}, {"resources": {"reservations": {"devices": devices}}, "placement": {}}):
            raise ArtifactError("unreviewed deployment resource/scheduling policy")
        if len(devices) != 1 or devices[0].get("device_ids") != [source["gpu_uuid"]] or devices[0].get("capabilities") != ["gpu"] or "count" in devices[0] or devices[0].get("driver") != "nvidia":
            raise ArtifactError("GPU device request changed")
        environment = service.get("environment", {})
        if environment != {"NVIDIA_VISIBLE_DEVICES": source["gpu_uuid"], "NVIDIA_DRIVER_CAPABILITIES": "compute,utility"}:
            raise ArtifactError("GPU runtime policy changed")
        if service.get("runtime") != "nvidia":
            raise ArtifactError("CUDA runtime mismatch")
    elif service.get("deploy") or devices or service.get("runtime") != "runc" or service.get("devices"):
        raise ArtifactError("CPU deployment must be NVIDIA-independent")
    elif service.get("environment", {}):
        raise ArtifactError("CPU deployment environment contains unreviewed overrides")
    ports = service.get("ports", [])
    expected_ports = {8000: ("127.0.0.1", source["api_port"])} if role == "client" else {7443: (source["bind_ip"], 7443), 7444: (source["bind_ip"], 7444)}
    if len(ports) != len(expected_ports) or {port.get("target") for port in ports} != set(expected_ports):
        raise ArtifactError("unexpected published listeners")
    for port in ports:
        binding = expected_ports.get(port.get("target"))
        if set(port) - {"target", "published", "host_ip", "protocol", "mode"} or port.get("mode", "ingress") != "ingress" or binding is None or (port.get("host_ip"), str(port.get("published"))) != (binding[0], str(binding[1])) or port.get("protocol", "tcp") != "tcp":
            raise ArtifactError("published listener scope changed")
    return document


def effective_compose(request, staging, final_dir, *, runner=_run):
    from .identity import loads
    role = request["planning_request"]["role"]
    env_file = staging / (role + ".env")
    atomic_bytes(env_file, environment_bytes(request, final_dir), mode=0o600)
    args = compose_arguments(request, final_dir, env_file, "config", "--format", "json")
    result = loads(runner(args, request["deadline_ms"], cwd=staging))
    validate_effective_compose(result, request, final_dir)
    runner(compose_arguments(request, final_dir, env_file, "config", "--quiet"), request["deadline_ms"], cwd=staging)
    return result


def _publish(staging, output):
    if output.exists():
        raise ArtifactError("deployment packages are immutable; choose a new output directory")
    os.rename(staging, output)
    sync_directory(output.parent)


def bootstrap(request, output, *, runner=_run):
    validate("deployment-request", request)
    validate_paths(request)
    output = Path(output).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".ria-deploy-", dir=output.parent))
    try:
        atomic_json(staging / "probe.json", probe_configuration(request))
        atomic_json(staging / "environment.json", seal(frozen_environment(request)))
        atomic_json(staging / "host-report.json", read_json(request["host_report"]))
        compose = effective_compose(request, staging, output, runner=runner)
        atomic_json(staging / "compose-effective.json", compose)
        marker = seal({"schema_revision": 1, "admitted": False, "status": "not_admitted",
                       "environment_digest": digest(frozen_environment(request)), "request_digest": digest(request),
                       "compose_digest": digest(compose)})
        atomic_json(staging / "bootstrap.json", marker)
        _publish(staging, output)
        return marker
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _fixture_paths(request, directory):
    validate("deployment-request", request)
    validate_paths(request)
    directory = Path(directory).resolve(strict=True)
    marker = read_json(directory / "bootstrap.json")
    verify_identity(marker)
    if marker.get("admitted") is not False or marker.get("environment_digest") != digest(frozen_environment(request)) or marker.get("request_digest") != digest(request):
        raise ArtifactError("fixture container requires the exact unadmitted bootstrap package")
    verify_identity(read_json(directory / "environment.json"), marker["environment_digest"])
    verify_identity(read_json(directory / "host-report.json"), request["environment"]["host_report_digest"])
    env_file = directory / (request["planning_request"]["role"] + ".env")
    if env_file.read_bytes() != environment_bytes(request, directory):
        raise ArtifactError("fixture environment file changed")
    lock_path = Path(request["environment"]["report_dir"]) / ("fixture-" + marker["digest"] + ".json")
    return directory, env_file, marker, lock_path


def _fixture_validate_lock(request, lock):
    validate("fixture-container-lock", lock)
    verify_identity(lock)
    if (lock["environment_digest"], lock["image"], lock["role"]) != (digest(frozen_environment(request)), request["environment"]["image"], request["planning_request"]["role"]):
        raise ArtifactError("fixture ownership record has another frozen realization")


def _fixture_inspect(request, directory, lock, runner, *, running=True):
    from .identity import loads
    from .container_inspection import validate_container_inspection
    from .host import verify_container_ancestors
    _fixture_validate_lock(request, lock)
    args = ["docker", "--host", "unix:///var/run/docker.sock", "inspect", "--type", "container", "--format", "{{json .}}", lock["container_id"]]
    def observe():
        value = loads(runner(args, request["deadline_ms"], cwd=directory))
        labels = value.get("Config", {}).get("Labels", {})
        if value.get("Id") != lock["container_id"] or value.get("Config", {}).get("Image") != lock["image"] or labels.get("com.docker.compose.service") != lock["role"] or labels.get("com.docker.compose.project") != "ria-" + lock["role"]:
            raise ArtifactError("actual fixture container differs from the exact owned image/service")
        validate_container_inspection(value, request, directory, qualification=True)
        if running and value.get("State", {}).get("Running") is not True:
            raise ArtifactError("owned bounded fixture container is not running")
        return value
    value = observe()
    if running:
        proof = verify_container_ancestors(value["State"]["Pid"], read_json(directory / "host-report.json"))
        if observe()["State"]["Pid"] != value["State"]["Pid"]:
            raise ArtifactError("fixture container PID changed during host verification")
        atomic_json(Path(request["environment"]["report_dir"]) / ("fixture-parent-" + lock["container_id"] + ".json"), proof)
    return value


def fixture_stop(request, directory, *, runner=_run):
    validate("deployment-request", request)
    with controller_lock(request):
        return _fixture_stop(request, directory, runner=runner)


def _fixture_stop(request, directory, *, runner=_run):
    directory, _, marker, lock_path = _fixture_paths(request, directory)
    lock = read_json(lock_path)
    _fixture_validate_lock(request, lock)
    if lock["bootstrap_digest"] != marker["digest"]:
        raise ArtifactError("fixture lock has another bootstrap identity")
    if lock["status"] == "stopped":
        return lock
    _fixture_inspect(request, directory, lock, runner, running=False)
    runner(["docker", "--host", "unix:///var/run/docker.sock", "stop", "--time", str(request["environment"]["stop_grace_seconds"]), lock["container_id"]],
           request["environment"]["stop_grace_seconds"] * 1000 + 5000, cwd=directory)
    if _fixture_inspect(request, directory, lock, runner, running=False).get("State", {}).get("Running") is not False:
        raise ArtifactError("owned fixture container did not quiesce after finite stop")
    runner(["docker", "--host", "unix:///var/run/docker.sock", "rm", lock["container_id"]], request["deadline_ms"], cwd=directory)
    lock = seal({**lock, "status": "stopped"})
    atomic_json(lock_path, lock)
    return lock


def fixture_start(request, directory, *, runner=_run):
    validate("deployment-request", request)
    with controller_lock(request):
        return _fixture_start(request, directory, runner=runner)


def _fixture_start(request, directory, *, runner=_run):
    from .identity import loads
    from .host import revalidate_host_report
    directory, env_file, marker, lock_path = _fixture_paths(request, directory)
    revalidate_host_report(read_json(directory / "host-report.json"))
    if lock_path.exists():
        lock = read_json(lock_path)
        if lock["bootstrap_digest"] != marker["digest"] or lock["status"] != "running":
            raise ArtifactError("fixture checkpoint is stopped; render a new bootstrap package")
        _fixture_inspect(request, directory, lock, runner)
        return lock
    role = request["planning_request"]["role"]
    override = directory / "qualification.override.yml"
    content = (f"services:\n  {role}:\n    entrypoint: [/usr/bin/sleep]\n    command: ['1800']\n    healthcheck: !override\n      disable: true\n").encode()
    if override.exists() and override.read_bytes() != content:
        raise ArtifactError("fixture override changed")
    atomic_bytes(override, content, mode=0o600)
    def arguments(*operation):
        return compose_arguments(request, directory, env_file, "-f", str(override), *operation)
    current = loads(runner(arguments("config", "--format", "json"), request["deadline_ms"], cwd=directory))
    validate_effective_compose(current, request, directory, qualification=True)
    runner(arguments("config", "--quiet"), request["deadline_ms"], cwd=directory)
    if runner(arguments("ps", "--all", "--quiet", role), request["deadline_ms"], cwd=directory).strip():
        raise ArtifactError("fixture startup refuses to replace an existing service container")
    lock = None
    try:
        runner(arguments("up", "-d", role), request["deadline_ms"], cwd=directory)
        container_id = runner(arguments("ps", "--quiet", role), request["deadline_ms"], cwd=directory).decode("ascii").strip()
        if not re.fullmatch(r"[0-9a-f]{64}", container_id):
            raise ArtifactError("fixture startup did not create exactly one owned container")
        lock = seal({"schema_revision": 1, "kind": "fixture_container_lock", "container_id": container_id,
            "bootstrap_digest": marker["digest"], "environment_digest": marker["environment_digest"],
            "image": request["environment"]["image"], "role": role, "status": "running"})
        atomic_json(lock_path, lock)
        _fixture_inspect(request, directory, lock, runner)
        return lock
    except (ArtifactError, OSError, ValueError, KeyError) as primary:
        # ps was empty before up: this exact project/service is task-owned even
        # if the Engine failed before returning its container identity.
        try:
            runner(arguments("stop", "--timeout", str(request["environment"]["stop_grace_seconds"]), role),
                request["environment"]["stop_grace_seconds"] * 1000 + 5000, cwd=directory)
            if lock is not None:
                atomic_json(lock_path, seal({**lock, "status": "stopped"}))
        except (ArtifactError, OSError) as cleanup:
            raise ArtifactError(f"fixture startup failed: {primary}; owned stop failed: {cleanup}") from primary
        raise ArtifactError(f"fixture startup failed and the owned container was stopped: {primary}") from primary


def fixture_exec(request, directory, operation, *, runner=_run):
    validate("deployment-request", request)
    with controller_lock(request):
        return _fixture_exec(request, directory, operation, runner=runner)


def _fixture_exec(request, directory, operation, *, runner=_run):
    from .fixture_runner import RUNS, validate_registration
    from .identity import within
    from .host import revalidate_host_report
    directory, _, marker, lock_path = _fixture_paths(request, directory)
    revalidate_host_report(read_json(directory / "host-report.json"))
    lock = read_json(lock_path)
    if lock["bootstrap_digest"] != marker["digest"] or lock["status"] != "running":
        raise ArtifactError("fixture operation requires a live owned bootstrap container")
    value = _fixture_inspect(request, directory, lock, runner)
    role = request["planning_request"]["role"]
    if operation == "probe":
        args = ["/usr/local/bin/ds4ctl", "probe", "--probe-config", "/etc/dwarfstar/probe.json", "--output", "/artifacts/probe.json"]
        deadline_ms = read_json(directory / "probe.json")["deadline_ms"]
    elif operation in RUNS and operation.endswith("server" if role == "expert" else "client"):
        report_root = Path(request["environment"]["report_dir"])
        registration = read_json(within(report_root, "inputs/registration.json"))
        policy = read_json(within(report_root, "inputs/policy.json"))
        validate_registration(registration, policy)
        if registration["realizations"]["server" if role == "expert" else "client"]["environment_digest"] != marker["environment_digest"]:
            raise ArtifactError("registered fixture selects another frozen container environment")
        deadline_ms = registration["runs"][operation]["request_body"]["deadline_ms"]
        args = ["/opt/ria-qualification/bin/python", "/opt/ria-tools/qualify_ria.py", "run-fixture",
            "--registration", "/artifacts/inputs/registration.json", "--policy", "/artifacts/inputs/policy.json", "--run", operation,
            "--executable", "/usr/local/bin/ds4-ria-qualify" if operation.startswith("native_") else "/usr/local/bin/ds4ctl",
            "--build-info", "/usr/share/dwarfstar/build-info.json", "--environment", "/etc/dwarfstar/environment.json",
            "--probe-config", "/etc/dwarfstar/probe.json", "--probe", "/artifacts/probe.json", "--probe-evidence", "/artifacts/probe.json.details.json",
            "--output-dir", "/artifacts/" + operation.replace("_", "-")]
        if operation.startswith("transport_"):
            args += ["--transport-config", "/artifacts/inputs/" + operation + "-bootstrap.json"]
    else:
        raise ArtifactError("unknown fixture operation or incorrect host role")
    started = datetime.fromisoformat(value["State"]["StartedAt"].replace("Z", "+00:00"))
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    remaining = 1800 - elapsed
    if elapsed < 0:
        raise ArtifactError("fixture container start timestamp is in the future")
    if not 0 < deadline_ms / 1000 + 10 < remaining:
        raise ArtifactError("fixture deadline exceeds the idle container's remaining bounded lifetime")
    try:
        return runner(["docker", "--host", "unix:///var/run/docker.sock", "exec", lock["container_id"], *args], deadline_ms + 5000, cwd=directory)
    except (ArtifactError, OSError, ValueError, KeyboardInterrupt, SystemExit) as primary:
        try:
            _fixture_stop(request, directory, runner=runner)
        except (ArtifactError, OSError) as cleanup:
            raise ArtifactError(f"fixture execution failed: {primary}; owned stop failed: {cleanup}") from primary
        raise ArtifactError(f"fixture execution failed and its whole container was stopped: {primary}") from primary


def finalize(request, probe, inventory, calibration, output, *, probe_evidence=None, calibration_evidence=None, calibration_evidence_dir=None, policy=None, runner=_run):
    validate("deployment-request", request)
    validate_paths(request)
    for kind, document in (("probe-report", probe), ("inventory", inventory), ("calibration", calibration)):
        validate(kind, document)
        if "digest" in document:
            verify_identity(document)
    if not probe["qualified"] or not calibration["qualified"]:
        raise ArtifactError("unqualified hardware or calibration cannot admit deployment")
    if probe_evidence is None or calibration_evidence is None or policy is None:
        raise ArtifactError("finalize requires explicit probe evidence, scoped calibration evidence and frozen policy")
    from .qualification import policy_validate, validate_calibration_evidence
    policy_validate(policy)
    current_environment = digest(frozen_environment(request))
    current_build = request["environment"]["build_digest"]
    if policy["logical_model_digest"] != request["planning_request"]["logical_model_digest"] or policy["source_lock_digest"] != request["environment"]["source_lock_digest"]:
        raise ArtifactError("qualification policy identifies a different model/source lock")
    for compact, evidence, kind in ((probe, probe_evidence, "probe_evidence"), (calibration, calibration_evidence, "calibration_evidence")):
        verify_identity(evidence, compact["evidence_digest"])
        if evidence.get("kind") != kind:
            raise ArtifactError("compact qualification requires the complete declared evidence kind")
        if any(document.get("environment_digest") != current_environment or document.get("build_digest") != current_build for document in (compact, evidence)):
            raise ArtifactError("probe/calibration evidence differs from the frozen current environment/build")
    if calibration["policy_digest"] != policy["digest"] or calibration_evidence.get("policy_digest") != policy["digest"]:
        raise ArtifactError("calibration evidence differs from preregistered qualification policy")
    validate("probe-evidence", probe_evidence)
    if (probe_evidence["role"], probe_evidence["executor"]) != (probe["role"], probe["executor"]):
        raise ArtifactError("probe raw role/executor differs from compact report")
    expected_numa = [{"node": item["node"], "bytes": int(item["bytes"])} for item in probe_evidence["numa_available"]]
    if (probe["host_bytes"], probe["device_bytes"], probe["pinned_bytes"], probe["numa"]) != (int(probe_evidence["host_available_bytes"]), int(probe_evidence["device_available_bytes"]), int(probe_evidence["pinned_test_bytes"]), expected_numa):
        raise ArtifactError("compact probe capacities differ from authenticated raw observations")
    if probe_evidence["gpu_uuid"] != request["environment"]["gpu_uuid"] or (probe["executor"] == "cuda" and (probe_evidence["compute_major"], probe_evidence["compute_minor"], probe_evidence["native_results"]) != (12, 0, [64, 32, 16])):
        raise ArtifactError("raw probe selected device/capability/native operation results differ")
    if probe["executor"] == "cpu" and (probe_evidence["driver_version"] or probe_evidence["runtime_version"] or probe_evidence["compute_major"] or probe_evidence["compute_minor"] or any(probe_evidence["native_results"])):
        raise ArtifactError("CPU-only probe cannot report CUDA execution")
    if int(probe_evidence["cgroup_limit_bytes"]) != request["environment"]["cgroup_bytes"] or int(probe_evidence["memlock_bytes"]) != request["environment"]["memlock_bytes"] or _mask(probe_evidence["cpu_mask"], 65535) != _mask(request["environment"]["cpuset"], 65535) or int(probe_evidence["host_test_bytes"]) != request["probe_config"]["max_host_test_bytes"]:
        raise ArtifactError("raw probe effective restrictions/test bounds differ from frozen request")
    selected_nodes = set(request["probe_config"]["numa_nodes"])
    if len(expected_numa) != len(selected_nodes) or {item["node"] for item in expected_numa} != selected_nodes or not selected_nodes.issubset(_mask(probe_evidence["memory_node_mask"], 63)):
        raise ArtifactError("raw probe NUMA observations differ from explicitly selected allowed nodes")
    validate("calibration-evidence", calibration_evidence)
    if calibration_evidence["qualification_scope"] != request["qualification_scope"]:
        raise ArtifactError("calibration evidence scope differs from requested admission scope")
    if calibration_evidence_dir is None:
        raise ArtifactError("calibration proof references require their explicit local evidence directory")
    proof_references = validate_calibration_evidence(calibration_evidence, calibration_evidence_dir, policy)
    if len(proof_references) > 8192:
        raise ArtifactError("calibration proof graph exceeds bounded publication inventory")
    planning = request["planning_request"]
    placement = None
    if planning["role"] == "client":
        if "placement_plan" not in request:
            raise ArtifactError("client finalize requires an explicit reviewed placement plan")
        placement = read_json(request["placement_plan"])
        validate("placement-plan", placement)
        verify_identity(placement)
        if placement["runtime"]["prefill_rows"] != planning["prefill_rows"]:
            raise ArtifactError("placement prefill microbatch differs from planning workload")
        if (placement["logical_model_digest"], placement["operator_contract_digest"]) != (planning["logical_model_digest"], planning["operator_contract_digest"]):
            raise ArtifactError("placement plan identity mismatch")
        if placement["server_executor"] != request["network"]["server_executor"]:
            raise ArtifactError("client placement and network server modes differ")
    model_manifest = read_json(Path(request["environment"]["model_dir"]) / "manifest.json")
    validate("manifest", model_manifest)
    verify_identity(model_manifest)
    if (model_manifest["logical_model_digest"], model_manifest["operator_contract_digest"], model_manifest["profile"]) != (planning["logical_model_digest"], planning["operator_contract_digest"], planning["profile"]):
        raise ArtifactError("provisioned model differs from planning identity")
    if model_manifest["role"] != ("client" if planning["role"] == "client" else "server"):
        raise ArtifactError("provisioned model role differs from local service")
    # Re-derive with the serving metadata/NUMA accountant. A selfsealed caller
    # inventory cannot omit allocations, replicas or phases to gain admission.
    from .inventory import build_inventory
    with tempfile.TemporaryDirectory(prefix="ria-finalize-inventory-") as temporary:
        actual_inventory = build_inventory(request, Path(request["environment"]["model_dir"]) / "manifest.json",
            Path(temporary) / "inventory.json", runner=runner)
    if inventory != actual_inventory:
        raise ArtifactError("inventory differs from authenticated native population/runtime derivation; regenerate it")
    if placement is not None:
        if placement["runtime"]["tokenizer_sha256"] != model_manifest["tokenizer_digest"]:
            raise ArtifactError("client placement tokenizer identity differs from prepared model")
        from .identity import within
        tokenizer_file = placement["runtime"]["tokenizer_file"]
        if not tokenizer_file.startswith("/model/"):
            raise ArtifactError("tokenizer runtime path must use the read-only model mount")
        if hash_file(within(Path(request["environment"]["model_dir"]), tokenizer_file[len("/model/"):])) != model_manifest["tokenizer_digest"]:
            raise ArtifactError("provisioned tokenizer source bytes disagree with trusted placement")
    if (probe["role"], probe["executor"]) != (planning["role"], planning["executor"]) or calibration["profile"] != planning["profile"]:
        raise ArtifactError("probe/calibration realization mismatch")
    output = Path(output).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".ria-deploy-", dir=output.parent))
    try:
        for name, value in (("request", planning), ("inventory", inventory), ("probe-report", probe), ("calibration", calibration)):
            atomic_json(staging / (name + ".json"), value)
        atomic_json(staging / "probe.json", probe_configuration(request))
        for name, document in (("environment", seal(frozen_environment(request))), ("host-report", read_json(request["host_report"])),
                               ("probe-evidence", probe_evidence), ("calibration-evidence", calibration_evidence), ("qualification-policy", policy)):
            atomic_json(staging / (name + ".json"), document)
        from .identity import within, canonical
        proof_bytes = 0
        for reference in proof_references:
            proof = read_json(within(calibration_evidence_dir, reference["path"]))
            verify_identity(proof, reference["digest"])
            proof_bytes += len(canonical(proof))
            if proof_bytes > 512 << 20:
                raise ArtifactError("calibration proof graph exceeds bounded publication bytes")
            target = staging / "evidence" / reference["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(target, proof)
        arguments = [request["native_ctl"], "plan", "--request", str(staging / "request.json"), "--inventory", str(staging / "inventory.json"),
                     "--probe", str(staging / "probe-report.json"), "--calibration", str(staging / "calibration.json"), "--output", str(staging / "memory-plan.json")]
        runner(arguments, request["deadline_ms"], cwd=staging)
        plan = read_json(staging / "memory-plan.json")
        validate("memory-plan", plan)
        verify_identity(plan)
        if (plan["context_positions"], plan["prefill_rows"]) != (planning["context_positions"], planning["prefill_rows"]):
            raise ArtifactError("native plan changed admitted context/prefill workload bounds")
        for key, document in (("request_digest", planning), ("inventory_digest", inventory), ("probe_digest", probe), ("calibration_digest", calibration)):
            if plan[key] != digest(document):
                raise ArtifactError("native plan input identity mismatch")
        if (plan["environment_digest"], plan["build_digest"], plan["policy_digest"]) != (current_environment, current_build, policy["digest"]):
            raise ArtifactError("native plan evidence provenance mismatch")
        if planning["role"] == "expert":
            expert, startup = request["expert"], plan["phases"]["startup"]
            if any(int(expert[field]) > startup[resource] for field, resource in (("startup_host_bytes", "host_bytes"),
                   ("device_workspace_bytes", "device_bytes"), ("pinned_workspace_bytes", "pinned_bytes"))):
                raise ArtifactError("expert physical reservations exceed the native admitted startup peak")
            admitted_nodes = {node["node"]: node["bytes"] for node in startup["numa"]}
            if any(int(node["local_bytes"]) > admitted_nodes.get(node["node"], 0) for node in expert["nodes"]):
                raise ArtifactError("expert local reservations exceed the native admitted startup node peak")
        service = {"schema_revision": 1, "role": planning["role"], "executor": planning["executor"],
            "model_manifest": "/model/manifest.json", "memory_plan": "/etc/dwarfstar/memory-plan.json",
            "deployment_lock": "/etc/dwarfstar/deployment-lock.json", "device_index": request["probe_config"]["device_index"],
            "management_socket": "/run/dwarfstar/admin.sock", "artifacts_dir": "/artifacts", "tls": request["tls"], "network": request["network"]}
        if planning["role"] == "client":
            service["api"] = request["api"]
            service["placement_plan"] = "/etc/dwarfstar/placement-plan.json"
            atomic_json(staging / "placement-plan.json", placement)
        else:
            service["expert"] = request["expert"]
        grants = None
        if "peer_grants" in request:
            grants = read_json(request["peer_grants"])
            validate("peer-grants", grants)
            verify_identity(grants)
            peer_name = request["tls"].get("expected_peer_name")
            if not any(grant["expected_peer_name"] == peer_name and
                       grant["logical_model_digest"] == planning["logical_model_digest"] and
                       grant["operator_contract_digest"] == planning["operator_contract_digest"] and
                       grant["profile"] == planning["profile"] and grant["server_executor"] == planning["executor"] and
                       grant["server_layout_digest"] == model_manifest["layout_digest"] for grant in grants["grants"]):
                raise ArtifactError("peer grants do not authorize the exact admitted realization")
            service["peer_grants"] = "/etc/dwarfstar/peer-grants.json"
            atomic_json(staging / "peer-grants.json", grants)
        validate("service", service)
        atomic_json(staging / "service.json", service)
        compose = effective_compose(request, staging, output, runner=runner)
        atomic_json(staging / "compose-effective.json", compose)
        environment = request["environment"]
        lock = seal({"schema_revision": 1, "role": planning["role"], "executor": planning["executor"], "image": environment["image"],
            "qualification_scope": request["qualification_scope"], "final_release_qualified": request["qualification_scope"] == "final_release",
            "build_digest": environment["build_digest"], "source_lock_digest": environment["source_lock_digest"],
            "logical_model_digest": planning["logical_model_digest"], "operator_contract_digest": planning["operator_contract_digest"],
            "environment_digest": current_environment, "probe_digest": probe["digest"], "calibration_digest": calibration["digest"],
            "probe_evidence_digest": probe_evidence["digest"], "calibration_evidence_digest": calibration_evidence["digest"], "policy_digest": policy["digest"],
            "memory_plan_digest": plan["digest"], "service_digest": digest(service), "compose_digest": digest(compose),
            "host_report_digest": environment["host_report_digest"], "seccomp_digest": hash_file(environment["seccomp_profile"]),
            "expected_peer_name": request["tls"].get("expected_peer_name"), "gpu_uuid": environment["gpu_uuid"],
            "model_manifest_digest": model_manifest["digest"],
            **({"placement_plan_digest": placement["digest"]} if placement is not None else {}),
            **({"peer_grants_digest": grants["digest"]} if grants is not None else {})})
        validate("deployment-lock", lock)
        atomic_json(staging / "deployment-lock.json", lock)
        # Native validates once more, using a reviewed mount map so host validation
        # sees precisely the same container paths without rewriting service bytes.
        # ds4ctl's contract is container-local; its invocation is the one-shot
        # container validation command and therefore must use the completed path.
        atomic_json(staging / "publication.json", seal({"schema_revision": 1, "admitted": True, "deployment_lock_digest": lock["digest"]}))
        _publish(staging, output)
        return lock
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def launch(request, directory, *, runner=_run):
    validate("deployment-request", request)
    with controller_lock(request):
        return _launch(request, directory, runner=runner)


def _launch(request, directory, *, runner=_run):
    """Recompute the reviewed Compose model in a controlled environment, then start."""
    validate("deployment-request", request)
    directory = Path(directory).resolve(strict=True)
    lock = read_json(directory / "deployment-lock.json")
    validate("deployment-lock", lock)
    verify_identity(lock)
    if lock["qualification_scope"] != request["qualification_scope"] or lock["final_release_qualified"] != (lock["qualification_scope"] == "final_release"):
        raise ArtifactError("requested admission scope differs from the deployment's separately authenticated qualification label")
    validate_paths(request)
    if digest(frozen_environment(request)) != lock["environment_digest"]:
        raise ArtifactError("deployment environment changed; rerun bounded probe/finalize")
    for name, key in (("service.json", "service_digest"), ("memory-plan.json", "memory_plan_digest"),
                      ("placement-plan.json", "placement_plan_digest"), ("peer-grants.json", "peer_grants_digest"),
                      ("probe-evidence.json", "probe_evidence_digest"), ("calibration-evidence.json", "calibration_evidence_digest"),
                      ("qualification-policy.json", "policy_digest"), ("host-report.json", "host_report_digest")):
        if key in lock:
            value = read_json(directory / name)
            if digest(value) != lock[key]:
                raise ArtifactError("deployment file identity changed; rerun finalize")
    from .qualification import validate_calibration_evidence
    validate_calibration_evidence(read_json(directory / "calibration-evidence.json"), directory / "evidence", read_json(directory / "qualification-policy.json"))
    manifest = read_json(Path(request["environment"]["model_dir"]) / "manifest.json", max_bytes=256 << 10)
    verify_identity(manifest, lock["model_manifest_digest"])
    if hash_file(request["environment"]["seccomp_profile"]) != lock["seccomp_digest"]:
        raise ArtifactError("reviewed seccomp policy changed")
    role = request["planning_request"]["role"]
    env_file = directory / (role + ".env")
    if env_file.read_bytes() != environment_bytes(request, directory):
        raise ArtifactError("resolved environment file changed")
    from .host import revalidate_host_report
    revalidate_host_report(read_json(directory / "host-report.json"))
    from .identity import loads
    current = loads(runner(compose_arguments(request, directory, env_file, "config", "--format", "json"), request["deadline_ms"], cwd=directory))
    validate_effective_compose(current, request, directory)
    if digest(current) != lock["compose_digest"]:
        raise ArtifactError("effective Compose configuration changed")
    runner(compose_arguments(request, directory, env_file, "config", "--quiet"), request["deadline_ms"], cwd=directory)
    runner(compose_arguments(request, directory, env_file, "run", "--rm", "--no-deps", "--entrypoint", "/usr/local/bin/ds4ctl", role,
                              "validate", "--config", "/etc/dwarfstar/service.json"), request["deadline_ms"], cwd=directory)
    try:
        runner(compose_arguments(request, directory, env_file, "up", "-d", role), request["deadline_ms"], cwd=directory)
        raw_id = runner(compose_arguments(request, directory, env_file, "ps", "--quiet", role), request["deadline_ms"], cwd=directory)
        container_id = raw_id.decode("ascii", errors="strict").strip()
        if not re.fullmatch(r"[0-9a-f]{64}", container_id):
            raise ArtifactError("launch did not produce exactly one owned running container")
        inspect_args = ["docker", "--host", "unix:///var/run/docker.sock", "inspect", "--type", "container", "--format", "{{json .}}", container_id]
        def inspect_owned():
            observed = loads(runner(inspect_args, request["deadline_ms"], cwd=directory))
            labels = observed.get("Config", {}).get("Labels", {})
            if observed.get("Id") != container_id or observed.get("Config", {}).get("Image") != lock["image"] or labels.get("com.docker.compose.service") != role or labels.get("com.docker.compose.project") != current.get("name") or observed.get("State", {}).get("Running") is not True:
                raise ArtifactError("Docker inspection differs from the exact owned reviewed service")
            from .container_inspection import validate_container_inspection
            validate_container_inspection(observed, request, directory)
            return observed
        observed = inspect_owned()
        from .host import verify_container_ancestors
        verification = verify_container_ancestors(observed["State"]["Pid"], read_json(directory / "host-report.json"))
        if inspect_owned()["State"]["Pid"] != observed["State"]["Pid"]:
            raise ArtifactError("container identity changed during host ancestry verification")
        atomic_json(Path(request["environment"]["report_dir"]) / ("launch-" + lock["digest"] + ".json"), verification)
    except (ArtifactError, OSError, ValueError, KeyError) as primary:
        try:
            runner(compose_arguments(request, directory, env_file, "stop", "--timeout", str(request["environment"]["stop_grace_seconds"]), role),
                   request["environment"]["stop_grace_seconds"] * 1000 + 5000, cwd=directory)
        except (ArtifactError, OSError) as cleanup:
            raise ArtifactError(f"launch verification failed: {primary}; owned service stop also failed: {cleanup}") from primary
        raise ArtifactError(f"launch verification failed and the owned service was stopped: {primary}") from primary
