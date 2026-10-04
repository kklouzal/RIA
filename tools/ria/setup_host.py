"""Host-authorized discovery from the distinct setup container, never serving.

This module requires the actual host PID/cgroup namespaces and canonical bind
paths. It runs only fixed Docker operations and a bounded runc metadata process;
GPU utility inventory is requested only for an explicitly GPU-enabled role.
No checkpoint, numerical fixture, CUDA kernel or model is executed here.
"""

import errno
import os
from pathlib import Path
import re
import secrets
import stat
import struct

from .deployment import _mask, _run
from .host import CGROOT, GPU_UUID, _read, observe_host
from .identity import ArtifactError, atomic_json, hash_file, loads, read_json, seal, verify_identity
from .setup_config import PROJECT_ROOT, _absolute, validate_settings, workspace_paths

DOCKER = ("docker", "--host", "unix:///var/run/docker.sock")
CONTAINER_ID = re.compile(r"[0-9a-f]{64}\Z")
BUILD_INFO = Path("/usr/share/dwarfstar/build-info.json")
DOCKER_SOCKET = Path("/var/run/docker.sock")


def _worker_acl_writable(raw, group):
    # Linux UAPI posix_acl_xattr.h: version LE32, then LE16 tag/permission and
    # LE32 identifier. The network worker clears supplementary groups.
    if len(raw) < 4 or len(raw) > 65536 or (len(raw) - 4) % 8 or struct.unpack_from("<I", raw)[0] != 2:
        raise ArtifactError("Docker socket access ACL has an unsupported encoding")
    entries = {}
    for tag, permission, identifier in struct.iter_unpack("<HHI", raw[4:]):
        if (tag not in (1, 2, 4, 8, 16, 32) or permission > 7 or (tag, identifier) in entries or
                (tag in (2, 8)) == (identifier == 0xffffffff)):
            raise ArtifactError("Docker socket access ACL is malformed")
        entries[tag, identifier] = permission
    if any((tag, 0xffffffff) not in entries for tag in (1, 4, 32)):
        raise ArtifactError("Docker socket access ACL lacks required permissions")
    mask = entries.get((16, 0xffffffff), 7)
    if any(tag in (2, 8) for tag, _ in entries) and (16, 0xffffffff) not in entries:
        raise ArtifactError("Docker socket access ACL lacks its required mask")
    if (2, 10001) in entries:
        return bool(entries[2, 10001] & mask & 2)
    matches = [permission for (tag, identifier), permission in entries.items()
               if (tag == 4 and group == 10001) or (tag == 8 and identifier == 10001)]
    return bool((mask & 2 and any(permission & 2 for permission in matches)) if matches else entries[32, 0xffffffff] & 2)


def _socket_identity():
    """Admit only one stable root-owned socket inaccessible to UID/GID10001."""
    info = DOCKER_SOCKET.stat(follow_symlinks=False)
    if (not stat.S_ISSOCK(info.st_mode) or info.st_uid != 0 or info.st_nlink != 1 or
            info.st_mode & stat.S_IWOTH or (info.st_gid == 10001 and info.st_mode & stat.S_IWGRP)):
        raise ArtifactError("Docker socket must be a root-owned single-link socket without worker write access")
    try:
        acl = os.getxattr(DOCKER_SOCKET, "system.posix_acl_access", follow_symlinks=False)
    except OSError as error:
        if error.errno not in (errno.ENODATA, errno.ENOTSUP):
            raise ArtifactError("Docker socket access ACL could not be inspected") from error
        acl = None
    if acl is not None and _worker_acl_writable(acl, info.st_gid):
        raise ArtifactError("Docker socket ACL grants the protocol worker host authority")
    def key(value):
        return (value.st_dev, value.st_ino, value.st_mode, value.st_uid, value.st_gid, value.st_nlink, value.st_ctime_ns)
    if key(info) != key(DOCKER_SOCKET.stat(follow_symlinks=False)):
        raise ArtifactError("Docker socket changed during authority validation")
    return key(info), acl


def _validated_context(settings, runner, *, proc_root=Path("/proc")):
    before = _socket_identity()
    gpu = _controller_context(settings, runner, proc_root=proc_root)
    if _socket_identity() != before:
        raise ArtifactError("Docker socket changed during controller authority inspection")
    return gpu


def _inspect(target, runner, deadline):
    return loads(runner([*DOCKER, "inspect", "--type", "container", "--format", "{{json .}}", target], deadline),
                 max_bytes=2 << 20, max_nodes=100000)


def _membership(path):
    lines = _read(path, 8192).splitlines()
    paths = [line[3:] for line in lines if line.startswith("0::/")]
    if len(paths) != 1 or any(part in (".", "..") for part in paths[0].split("/")):
        raise ArtifactError("setup requires a resolvable host unified cgroup membership")
    return paths[0]


def _input_mounts(settings, mounts):
    """Bind configured inputs to the same physical paths Compose will use.

    The most specific mount determines access. Every nested mount inside an
    input must retain canonical host ancestry; a read-only input cannot be
    made writable by a child mount. Source recipes additionally need their
    reviewed writable .ria-recipes directory.
    """
    inputs = [(Path(settings["workspace"]), True), (Path("/run/lock"), True)]
    model = settings.get("model", {})
    if model.get("mode") == "source":
        source = Path(model["source_dir"])
        inputs.extend(((source, True), (source / ".ria-recipes", True)))
    elif model.get("mode") == "prepared":
        inputs.append((Path(model["package_dir"]), False))
    for value in (settings["security"]["api_token_file"], model.get("hf_token_file")):
        if value is not None:
            inputs.append((Path(value), False))
    for path, writable in inputs:
        path = Path(_absolute(str(path), "setup input"))
        ancestors = [item for item in mounts if Path(item["Destination"]) == path or Path(item["Destination"]) in path.parents]
        if not ancestors:
            raise ArtifactError("setup input requires canonical same-path host bind ancestry")
        depth = max(len(Path(item["Destination"]).parts) for item in ancestors)
        selected = [item for item in ancestors if len(Path(item["Destination"]).parts) == depth]
        if len(selected) != 1:
            raise ArtifactError("setup input has ambiguous host bind ancestry")
        mount = selected[0]
        if (mount.get("Type") != "bind" or mount.get("Source") != mount["Destination"] or type(mount.get("RW")) is not bool or
                (writable and mount["RW"] is not True)):
            raise ArtifactError("setup input requires canonical same-path host bind ancestry and source write access")
        for item in mounts:
            destination = Path(item["Destination"])
            if path in destination.parents and (item.get("Type") != "bind" or item.get("Source") != item["Destination"] or
                    type(item.get("RW")) is not bool or (mount["RW"] is False and item["RW"] is True)):
                raise ArtifactError("setup input host ancestry is shadowed by a foreign or writable mount")


def _controller_context(settings, runner, *, proc_root=Path("/proc")):
    if os.geteuid() != 0:
        raise ArtifactError("setup discovery requires the authorized root controller")
    path = _membership(Path(proc_root) / "self/cgroup")
    ids = re.findall(r"(?:^|/)(?:docker-)?([0-9a-f]{64})(?:\.scope)?(?=/|$)", path)
    if len(ids) != 1:
        raise ArtifactError("setup requires its actual Engine container in the host cgroup namespace")
    value = _inspect(ids[0], runner, settings["deadline_ms"])
    if (not isinstance(value, dict) or any(not isinstance(value.get(name), dict) for name in ("HostConfig", "Config", "State")) or
            not isinstance(value.get("Mounts"), list) or any(not isinstance(item, dict) or not isinstance(item.get("Destination"), str)
            for item in value["Mounts"])):
        raise ArtifactError("setup controller inspection is malformed")
    host, config = value.get("HostConfig", {}), value.get("Config", {})
    if (value.get("Id") != ids[0] or value.get("State", {}).get("Running") is not True or
            host.get("PidMode") != "host" or host.get("CgroupnsMode") != "host" or host.get("NetworkMode") != "host" or
            host.get("Privileged") is not False or config.get("User") not in ("0", "0:0", "root", "root:root", "")):
        raise ArtifactError("setup controller requires explicit host PID/cgroup/network namespaces and root without privileged mode")
    mounts = [item for item in value.get("Mounts", []) if item.get("Type") == "bind"]
    required = {"/var/run/docker.sock": True, "/run/lock": True, settings["workspace"]: True}
    for target, writable in required.items():
        selected = [item for item in mounts if item.get("Destination") == target]
        if len(selected) != 1 or selected[0].get("Source") != target or selected[0].get("RW") is not writable:
            raise ArtifactError("setup controller requires canonical socket, persistent locks and workspace bind paths")
    _input_mounts(settings, value["Mounts"])
    for target in ("/sys/fs/cgroup", "/sys/devices/system/node"):
        selected = [item for item in mounts if item.get("Destination") == target]
        if not selected:
            selected = [item for item in mounts if item.get("Destination") == "/sys"]
        if len(selected) != 1 or selected[0].get("Source") != selected[0].get("Destination") or selected[0].get("RW") is not False:
            raise ArtifactError("setup controller requires read-only canonical host cgroup/NUMA sysfs mounts")
        root = Path(target)
        for item in value["Mounts"]:
            destination = Path(item["Destination"])
            if (destination == root or destination in root.parents or root in destination.parents) and (
                    item.get("Type") != "bind" or item.get("Source") != item["Destination"] or item.get("RW") is not False):
                raise ArtifactError("setup controller sysfs authority is shadowed by another mount")
    devices = host.get("DeviceRequests") or []
    values = config.get("Env", [])
    if not isinstance(values, list) or any(not isinstance(value, str) or "=" not in value for value in values):
        raise ArtifactError("setup controller environment is malformed")
    environment = dict(value.split("=", 1) for value in values)
    if len(environment) != len(values):
        raise ArtifactError("setup controller environment has duplicate keys")
    if settings["executor"] == "cpu":
        if devices or any(name.startswith(("NVIDIA_", "CUDA_", "NV_CUDA_")) for name in environment):
            raise ArtifactError("CPU setup controller must have no NVIDIA dependency")
        return None
    if len(devices) != 1 or not isinstance(devices[0], dict):
        raise ArtifactError("CUDA setup utility discovery requires one explicit NVIDIA selector")
    device = devices[0]
    ids, count = device.get("DeviceIDs"), device.get("Count")
    if (device.get("Driver") != "nvidia" or device.get("Capabilities") != [["gpu"]] or
            device.get("Options") not in (None, {}) or type(count) is not int or not isinstance(ids, list) or
            not ((count in (-1, 1) and not ids) or
                 (count == 0 and len(ids) == 1 and isinstance(ids[0], str) and
                  (GPU_UUID.fullmatch(ids[0]) or re.fullmatch(r"0|[1-9][0-9]{0,9}", ids[0]))))):
        raise ArtifactError("CUDA setup utility discovery requires a bounded UUID/index, one-device or all selector")
    if environment.get("NVIDIA_DRIVER_CAPABILITIES") != "utility":
        raise ArtifactError("setup GPU discovery must expose utility capability only")
    uuid = "GPU-" + ids[0][4:].lower() if ids and GPU_UUID.fullmatch(ids[0]) else settings["gpu_uuid"]
    if ids and GPU_UUID.fullmatch(ids[0]) and settings["gpu_uuid"] is not None and settings["gpu_uuid"] != uuid:
        raise ArtifactError("setup utility GPU differs from the operator's selected UUID")
    return uuid


def _validate_selection(settings, report, gpu):
    verify_identity(report)
    if (report.get("kind"), report.get("rootful"), report.get("cgroup_version"), report.get("selected_gpu_uuid"), report.get("client")) != (
            "host_preflight", True, 2, gpu, settings["role"] == "client"):
        raise ArtifactError("host discovery returned another role/device or unsupported daemon")
    ancestors = report.get("parent_ancestors")
    nodes = report.get("numa")
    if not isinstance(ancestors, list) or not ancestors or not isinstance(nodes, list) or not nodes:
        raise ArtifactError("host discovery lacks actual ancestor limits or NUMA topology")
    cpus = _mask(settings["environment"]["cpuset"], 65535)
    caps = settings["planning"]["caps"]
    selected_nodes = {node["node"] for node in caps["numa"]}
    by_node = {node["node"]: node for node in nodes}
    if not selected_nodes.issubset(by_node):
        raise ArtifactError("selected NUMA node does not exist on the actual host")
    allowed_cpus = set().union(*(_mask(by_node[node]["cpus"], 65535) for node in selected_nodes))
    if not cpus.issubset(allowed_cpus):
        raise ArtifactError("requested CPU affinity is outside the selected NUMA nodes")
    for ancestor in ancestors:
        if not cpus.issubset(_mask(ancestor["cpuset.cpus.effective"], 65535)) or not selected_nodes.issubset(_mask(ancestor["cpuset.mems.effective"], 63)):
            raise ArtifactError("requested CPU/NUMA membership exceeds a real host ancestor")
        for field, requested in (("memory.max", settings["environment"]["cgroup_bytes"]), ("pids.max", settings["environment"]["pids_limit"])):
            if ancestor[field] != "max" and requested > int(ancestor[field]):
                raise ArtifactError("requested container limits exceed a real host ancestor")
    if any(node["bytes"] > int(by_node[node["node"]]["total_bytes"]) for node in caps["numa"]):
        raise ArtifactError("requested NUMA cap exceeds physical node memory")
    if settings["role"] == "expert":
        for node in settings["expert"]["nodes"]:
            if not set(node["cpus"]).issubset(_mask(by_node[node["node"]]["cpus"], 65535)):
                raise ArtifactError("expert node CPU membership differs from actual host topology")


def _validate_image(observed, settings):
    config = observed.get("Config", {})
    image = settings["service_image"]
    repository, image_digest = image.split("@", 1)
    # RepoDigests contain the repository without an optional human tag.
    prefix, _, leaf = repository.rpartition("/")
    repository = (prefix + "/" if prefix else "") + leaf.split(":", 1)[0]
    if (observed.get("Architecture"), observed.get("Os"), config.get("User"), config.get("WorkingDir"), config.get("Cmd"), config.get("Entrypoint")) != (
            "amd64", "linux", "10001:10001", "/artifacts", ["--config", "/etc/dwarfstar/service.json"],
            ["/usr/local/bin/ds4-expert-server" if settings["executor"] == "cpu" else "/usr/local/bin/ds4-server"]):
        raise ArtifactError("selected service image configuration differs from the native target contract")
    if repository + "@" + image_digest not in observed.get("RepoDigests", []):
        raise ArtifactError("actual service image does not retain the requested immutable digest")
    values = config.get("Env", [])
    if any(not isinstance(value, str) or "=" not in value for value in values):
        raise ArtifactError("selected service image environment is malformed")
    environment = dict(value.split("=", 1) for value in values)
    if len(environment) != len(values):
        raise ArtifactError("selected service image environment has duplicate keys")
    if settings["executor"] == "cpu":
        if any(name.startswith(("NVIDIA_", "CUDA_", "NV_CUDA_")) for name in environment):
            raise ArtifactError("CPU service image cannot depend on NVIDIA/CUDA")
    else:
        lock = read_json(PROJECT_ROOT / "deploy/container-lock.json")
        if (environment.get("CUDA_VERSION") != lock["cuda_version"] or environment.get("CUDA_DISABLE_PTX_JIT") != "1" or
                environment.get("NVIDIA_DRIVER_CAPABILITIES") != "compute,utility" or
                environment.get("NVIDIA_REQUIRE_CUDA") != lock["cuda_registry_verification"]["inherited_configuration"]["environment"]["NVIDIA_REQUIRE_CUDA"]):
            raise ArtifactError("CUDA service image differs from the pinned SDK/no-JIT runtime policy")


def _validate_build(build, executor, lock):
    verify_identity(build)
    if (build.get("image_kind"), build.get("architecture"), build.get("source_lock_digest"), build.get("hardware_qualified")) != (
            executor, "amd64", lock["digest"], False):
        raise ArtifactError("image build metadata differs from the pinned target/source lock")
    binaries = ("ds4ctl", "ds4-expert-server", "ds4-ria-qualify") + (("ds4", "ds4-server", "ds4-eval") if executor == "cuda" else ())
    if not isinstance(build.get("binaries"), dict) or set(build["binaries"]) != set(binaries) or any(
            not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in build["binaries"].values()):
        raise ArtifactError("image lacks the exact role binary identities")
    sources = build.get("sources")
    if not isinstance(sources, dict) or not sources or any(not isinstance(key, str) or not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for key, value in sources.items()):
        raise ArtifactError("image lacks its complete reviewed source identities")
    numeric = ["-fno-fast-math", "-ffp-contract=off"]
    if executor == "cuda":
        numeric += ["--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true"]
        if build.get("code_targets") != ["sm_120a"] or build.get("ptx_jit") is not False:
            raise ArtifactError("CUDA service build lacks its exact offline AOT/no-JIT contract")
    if build.get("schema_revision") != 1 or type(build["schema_revision"]) is not int or build.get("numeric_flags") != numeric:
        raise ArtifactError("service/setup image changes the required numerical build contract")


def validate_controller_context(settings, runner=None):
    """Validate administrative authority without pulling or starting images.

    The returned UUID is the configured/direct utility UUID, or null when GPU
    selection awaits discovery (and on CPU hosts). This does not query a GPU.
    """
    return _validated_context(validate_settings(settings), runner or _run)


def discover_local(settings, workspace=None, runner=None, *, proc_root=Path("/proc"), cgroup_root=CGROOT, observer=None):
    """Publish actual host/build facts; injected readers are only test seams."""
    settings = validate_settings(settings)
    paths = workspace_paths(workspace or settings["workspace"])
    if paths["root"] != settings["workspace"]:
        raise ArtifactError("discovery workspace differs from the operator settings")
    runner = runner or _run
    gpu = _validated_context(settings, runner, proc_root=proc_root)
    # Pull only the explicitly pinned service image. The metadata commands use
    # runc even for CUDA images and never expose a GPU or model volume.
    runner([*DOCKER, "pull", "--platform", "linux/amd64", settings["service_image"]], settings["deadline_ms"])
    image = loads(runner([*DOCKER, "image", "inspect", "--format", "{{json .}}", settings["service_image"]], settings["deadline_ms"]),
                  max_bytes=2 << 20, max_nodes=100000)
    _validate_image(image, settings)
    restrictions = ["--runtime=runc", "--network=none", "--read-only", "--no-healthcheck", "--user", "10001:10001",
                    "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true", "--pids-limit", "16",
                    "--memory", "64m", "--memory-swap", "64m"]
    raw = runner([*DOCKER, "run", "--rm", "--pull=never", *restrictions, "--entrypoint", "/bin/cat",
                  settings["service_image"], "/usr/share/dwarfstar/build-info.json"], settings["deadline_ms"])
    build = loads(raw, max_bytes=2 << 20, max_nodes=100000)
    lock = read_json(PROJECT_ROOT / "locks/source-lock.json")
    verify_identity(lock)
    own_build = read_json(BUILD_INFO, max_bytes=2 << 20, max_nodes=100000)
    _validate_build(own_build, "cpu", lock)
    _validate_build(build, settings["executor"], lock)
    if build["sources"] != own_build["sources"]:
        raise ArtifactError("service/setup images contain different source contracts; choose the matching publication set")
    if hash_file(PROJECT_ROOT / "bin/ds4ctl") != own_build["binaries"]["ds4ctl"]:
        raise ArtifactError("setup native admission executable differs from its actual image build identity")
    nonce = secrets.token_hex(16)
    name = "ria-setup-inspection-" + nonce
    label = "io.ria.setup.inspection=" + nonce
    container = None
    primary = None
    try:
        raw = runner([*DOCKER, "run", "--detach", "--rm", "--pull=never", "--name", name, "--label", label,
                      *restrictions, "--entrypoint", "/bin/sleep", settings["service_image"], "60"], min(settings["deadline_ms"], 60000))
        container = raw.decode("ascii", errors="strict").strip()
        if not CONTAINER_ID.fullmatch(container):
            raise ArtifactError("metadata inspection did not return exactly one owned container ID")
        def owned():
            value = _inspect(container, runner, settings["deadline_ms"])
            if (value.get("Id") != container or value.get("Name") != "/" + name or
                    value.get("Config", {}).get("Image") != settings["service_image"] or
                    value.get("Config", {}).get("Labels", {}).get("io.ria.setup.inspection") != nonce or
                    value.get("State", {}).get("Running") is not True or
                    type(value.get("State", {}).get("Pid")) is not int or not 0 < value["State"]["Pid"] <= 2147483647):
                raise ArtifactError("temporary host inspection differs from its exact owned container")
            return value
        value = owned()
        pid = value["State"]["Pid"]
        membership = _membership(Path(proc_root) / str(pid) / "cgroup")
        parent = Path(cgroup_root) / str(Path(membership).parent).lstrip("/")
        requested_gpu = gpu
        report = (observer or observe_host)(str(parent), gpu, client=settings["role"] == "client",
                                          discover_gpu=settings["executor"] == "cuda")
        gpu = report.get("selected_gpu_uuid")
        if requested_gpu is not None and gpu != requested_gpu:
            raise ArtifactError("actual discovery changed the selected/requested physical UUID")
        if settings["executor"] == "cuda" and (not isinstance(gpu, str) or not GPU_UUID.fullmatch(gpu)):
            raise ArtifactError("CUDA host discovery did not select an actual compatible physical UUID")
        if owned()["State"]["Pid"] != pid or _membership(Path(proc_root) / str(pid) / "cgroup") != membership:
            raise ArtifactError("temporary host PID/cgroup changed during discovery")
        from .host import verify_container_ancestors
        verify_container_ancestors(pid, report, proc_root=proc_root, cgroup_root=cgroup_root)
        if owned()["State"]["Pid"] != pid:
            raise ArtifactError("temporary host PID changed during ancestor verification")
        _validate_selection(settings, report, gpu)
    except BaseException as exc:
        primary = exc
        raise
    finally:
        try:
            # An Engine call can time out after creation, or return an invalid
            # ID. Recover only the exact unpredictable name and recheck the
            # ownership label even when the original response looked valid.
            target = container if isinstance(container, str) and CONTAINER_ID.fullmatch(container) else name
            value = _inspect(target, runner, min(settings["deadline_ms"], 15000))
            candidate = value.get("Id")
            if (not isinstance(candidate, str) or not CONTAINER_ID.fullmatch(candidate) or
                    value.get("Name") != "/" + name or value.get("Config", {}).get("Image") != settings["service_image"] or
                    value.get("Config", {}).get("Labels", {}).get("io.ria.setup.inspection") != nonce):
                raise ArtifactError("ambiguous inspection cleanup ownership")
            container = candidate
            runner([*DOCKER, "rm", "--force", container], min(settings["deadline_ms"], 15000))
        except (ArtifactError, OSError, ValueError) as cleanup:
            if primary is not None:
                raise ArtifactError(f"host discovery failed: {primary}; owned inspection cleanup failed: {cleanup}") from primary
            raise
    atomic_json(paths["host_report"], report)
    atomic_json(paths["build_info"], build)
    return seal({"schema_revision": 1, "kind": "setup_host_facts", "role": settings["role"], "executor": settings["executor"],
        "service_image": settings["service_image"], "gpu_uuid": gpu, "host_report": report, "build_info": build,
        "host_report_path": paths["host_report"], "build_info_path": paths["build_info"], "paths": paths})
