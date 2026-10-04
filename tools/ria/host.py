"""Read-only target-host evidence and post-launch hidden-ancestor verification."""

import csv
import io
from pathlib import Path
import platform
import re

from .identity import ArtifactError, atomic_json, loads, read_json, seal, verify_identity
from .process import run_bounded

CGROOT = Path("/sys/fs/cgroup")
GPU_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")


def _read(path, maximum=65536):
    # procfs/sysfs virtual files report length zero; read an explicit bound.
    with Path(path).open("rb") as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise ArtifactError("host observation exceeds its byte bound")
    return raw.decode("ascii", errors="strict").strip()


def _command(arguments, timeout=15):
    environment = {"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", "LANG": "C", "LC_ALL": "C"}
    result = run_bounded(arguments, timeout=timeout, env=environment)
    if result.returncode or len(result.stdout) > 1048576 or len(result.stderr) > 65536:
        raise ArtifactError(f"host observation command failed: {arguments[0]}")
    return result.stdout


def cgroup_ancestors(parent, *, root=CGROOT):
    """Stable limits, excluding mutable occupancy and transient container IDs."""
    root = Path(root).resolve(strict=True)
    parent = Path(parent)
    if not parent.is_absolute() or parent.is_symlink():
        raise ArtifactError("cgroup parent must be an absolute real directory")
    parent = parent.resolve(strict=True)
    if not parent.is_relative_to(root):
        raise ArtifactError("cgroup parent is outside the unified hierarchy")
    result = []
    while True:
        fields = {}
        for name in ("memory.max", "memory.swap.max", "pids.max"):
            value = "max" if parent == root and not (parent / name).exists() else _read(parent / name)
            if value != "max" and not re.fullmatch(r"0|[1-9][0-9]*", value):
                raise ArtifactError("invalid cgroup scalar")
            fields[name] = value
        for name in ("cpuset.cpus.effective", "cpuset.mems.effective"):
            value = _read(parent / name)
            if not re.fullmatch(r"[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*", value):
                raise ArtifactError("invalid effective cgroup mask")
            fields[name] = value
        result.append({"path": str(parent), **fields})
        if parent == root:
            return result
        parent = parent.parent


def _host_target():
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ArtifactError("host preflight requires the declared native Linux x86-64 target")
    if not (CGROOT / "cgroup.controllers").is_file():
        raise ArtifactError("host preflight requires cgroup v2")


def _gpu_inventory(raw):
    """Decode bounded NVML utility rows without initializing CUDA."""
    if not isinstance(raw, bytes) or len(raw) > 1048576:
        raise ArtifactError("physical GPU inventory exceeds its byte bound")
    gpus = []
    seen = set()
    try:
        for row in csv.reader(io.StringIO(raw.decode("ascii")), skipinitialspace=True, strict=True):
            if (len(row) != 5 or not GPU_UUID.fullmatch(row[0]) or not row[1] or
                    not re.fullmatch(r"[0-9]+\.[0-9]+", row[2]) or not re.fullmatch(r"[0-9]+", row[3]) or
                    not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", row[4]) or row[0].lower() in seen):
                raise ArtifactError("physical GPU inventory has an unsupported response")
            seen.add(row[0].lower())
            gpus.append(dict(zip(("uuid", "name", "compute_capability", "memory_mib", "driver_version"), row, strict=True)))
    except (UnicodeError, csv.Error) as error:
        raise ArtifactError("physical GPU inventory has an unsupported response") from error
    return gpus


def _select_gpu(gpus, gpu_uuid, *, client):
    if gpu_uuid is not None and not GPU_UUID.fullmatch(gpu_uuid):
        raise ArtifactError("invalid physical GPU UUID")
    compatible = [gpu for gpu in gpus if gpu["compute_capability"] == "12.0" and
                  (not client or gpu["name"] == "NVIDIA GeForce RTX 5090")]
    if gpu_uuid is not None:
        selected = [gpu for gpu in compatible if gpu["uuid"].lower() == gpu_uuid.lower()]
        if len(selected) != 1:
            raise ArtifactError("physical device does not satisfy the admitted UUID/SM/client contract")
        return gpu_uuid
    if len(compatible) != 1:
        raise ArtifactError("GPU discovery requires exactly one compatible visible device; select an explicit gpu_uuid")
    return "GPU-" + compatible[0]["uuid"][4:].lower()


def observe_host(parent, gpu_uuid=None, *, client=False, discover_gpu=False):
    """Observe one stable host realization; CUDA discovery is explicit.

    Only discover_gpu=True permits automatic utility-only GPU selection. The
    existing CPU path performs no NVIDIA query; explicit UUID revalidation
    retains the same actual inventory and selection contract.
    """
    if type(discover_gpu) is not bool:
        raise ArtifactError("GPU discovery mode must be explicit")
    _host_target()
    info = loads(_command(["docker", "--host", "unix:///var/run/docker.sock", "info", "--format", "{{json .}}"]), project=False)
    if info.get("CgroupVersion") != "2" or info.get("CgroupDriver") not in ("systemd", "cgroupfs") or any("rootless" in value for value in info.get("SecurityOptions", [])):
        raise ArtifactError("deployment requires rootful Docker with cgroup v2")
    if info.get("Architecture") != "x86_64":
        raise ArtifactError("Docker daemon is not the native x86-64 target")
    compose = _command(["docker", "compose", "version", "--short"]).decode("ascii").strip()
    if not re.fullmatch(r"v?(?:[2-9]|[1-9][0-9]+)\.[0-9]+\.[0-9]+", compose):
        raise ArtifactError("unsupported Compose-v2 CLI implementation")
    nodes = []
    for path in sorted(Path("/sys/devices/system/node").glob("node[0-9]*"), key=lambda item: int(item.name[4:])):
        node = int(path.name[4:])
        if node >= 64:
            raise ArtifactError("NUMA node exceeds the explicit bounded policy")
        memory = _read(path / "meminfo")
        match = re.search(r"\bMemTotal:\s+([0-9]+) kB\b", memory)
        if not match:
            raise ArtifactError("NUMA total memory is unavailable")
        nodes.append({"node": node, "cpus": _read(path / "cpulist"),
                      "total_bytes": str(int(match[1]) * 1024),
                      "distance": [int(value) for value in _read(path / "distance").split()]})
    if not nodes:
        raise ArtifactError("NUMA topology is unavailable")
    gpus = []
    if gpu_uuid is not None or discover_gpu:
        if gpu_uuid is not None and not GPU_UUID.fullmatch(gpu_uuid):
            raise ArtifactError("invalid physical GPU UUID")
        raw = _command(["nvidia-smi", "--query-gpu=uuid,name,compute_cap,memory.total,driver_version", "--format=csv,noheader,nounits"])
        gpus = _gpu_inventory(raw)
        gpu_uuid = _select_gpu(gpus, gpu_uuid, client=client)
    elif client:
        raise ArtifactError("client preflight requires its exact physical GPU UUID")
    return seal({"schema_revision": 1, "kind": "host_preflight", "architecture": "x86_64",
        "kernel_version": platform.release(), "docker_version": info["ServerVersion"],
        "compose_version": compose.lstrip("v"), "cgroup_driver": info["CgroupDriver"],
        "rootful": True, "cgroup_version": 2, "parent_ancestors": cgroup_ancestors(parent),
        "numa": nodes, "physical_gpus": gpus,
        "selected_gpu_uuid": gpu_uuid, "client": client, "hardware_qualified": False})


def verify_container_ancestors(pid, baseline, *, proc_root=Path("/proc"), cgroup_root=CGROOT):
    """Verify a running, explicitly owned container's real host parent limits."""
    verify_identity(baseline)
    if baseline.get("kind") != "host_preflight" or not baseline.get("rootful") or baseline.get("cgroup_version") != 2:
        raise ArtifactError("invalid host preflight identity")
    if type(pid) is not int or not 1 <= pid <= 2147483647:
        raise ArtifactError("container PID is not a running host process")
    membership = _read(Path(proc_root) / str(pid) / "cgroup", 8192)
    lines = [line[3:] for line in membership.splitlines() if line.startswith("0::/")]
    if len(lines) != 1 or ".." in lines[0].split("/"):
        raise ArtifactError("invalid unified container cgroup path")
    root = Path(cgroup_root).resolve(strict=True)
    actual = root / lines[0].lstrip("/")
    current = cgroup_ancestors(actual.parent, root=root)
    if current != baseline["parent_ancestors"]:
        raise ArtifactError("container host ancestors differ from the preregistered baseline; re-probe required")
    return seal({"schema_revision": 1, "kind": "container_parent_verification",
        "host_report_digest": baseline["digest"], "pid": pid, "cgroup_path": str(actual),
        "verified": True})


def save_host(parent, output, gpu_uuid=None, *, client=False):
    result = observe_host(parent, gpu_uuid, client=client)
    atomic_json(output, result)
    return result


def verify_host_report(path, expected):
    report = read_json(path)
    verify_identity(report, expected)
    return report


def revalidate_host_report(frozen, *, observer=None):
    """Re-observe all stable realization facts before managed execution.

    observe_host records topology, versions and configured limits, without
    transient occupancy or free-memory samples. Its complete sealed identity
    must therefore match; no changed driver or topology receives a skip.
    Tests may inject an observer without querying physical hardware.
    """
    verify_identity(frozen)
    ancestors = frozen.get("parent_ancestors")
    if frozen.get("kind") != "host_preflight" or frozen.get("rootful") is not True or frozen.get("cgroup_version") != 2 or not isinstance(ancestors, list) or not ancestors:
        raise ArtifactError("current host verification requires the sealed actual parent baseline")
    parent = ancestors[0].get("path") if isinstance(ancestors[0], dict) else None
    if not isinstance(parent, str) or not Path(parent).is_absolute() or type(frozen.get("client")) is not bool:
        raise ArtifactError("host baseline does not declare its exact cgroup parent and role")
    current = (observer or observe_host)(parent, frozen.get("selected_gpu_uuid"), client=frozen["client"])
    verify_identity(current)
    if current["digest"] != frozen["digest"]:
        raise ArtifactError("current stable host realization changed; fresh baseline and probe required")
    return current
