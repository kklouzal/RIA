"""Actual inherited process restrictions and authenticated weight-free probes.

The container owner separately verifies image, mounts, reviewed seccomp rules
and hidden host cgroup ancestors. This boundary checks the restrictions visible
inside that exact inspected container before creating a native fixture child.
"""

import ctypes
import os
from pathlib import Path
import platform
import resource

from .identity import ArtifactError, SAFE_INTEGER, verify_identity, u64
from .qualification_schema import record

UINT = {"type": "integer", "minimum": 0, "maximum": SAFE_INTEGER}
DECIMAL = {"type": "string", "pattern": "^(0|[1-9][0-9]{0,19})$"}
MASK = {"type": "string", "minLength": 1, "maxLength": 4096}
OBSERVATION = record({"architecture": {"const": "x86_64"}, "system": {"const": "Linux"},
    "uid": {"const": 10001}, "euid": {"const": 10001}, "dumpable": {"const": 0},
    "no_new_privileges": {"const": 1}, "seccomp_mode": {"const": 2},
    "core_soft": {"const": 0}, "core_hard": {"const": 0},
    **{key: DECIMAL for key in ("memlock_soft", "memlock_hard", "host_limit_bytes", "host_available_bytes", "swap_limit_bytes", "locked_bytes")},
    "cpu_mask": MASK, "memory_node_mask": MASK})
PREFLIGHT = record({"environment": {"type": "object"}, "probe_config": {"type": "object"},
    "probe": {"type": "object"}, "probe_evidence": {"type": "object"}, "observation": OBSERVATION})


def _read(path, maximum):
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        data = bytearray()
        while len(data) <= maximum:
            chunk = os.read(fd, min(4096, maximum + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        if len(data) > maximum:
            raise ArtifactError("runtime system observation exceeds its bound")
        return data.decode("ascii").strip()
    finally:
        os.close(fd)


def observe_runtime():
    """Linux-only read of real process/cgroup state; never used by offline tests."""
    if platform.system() != "Linux" or platform.machine() != "x86_64" or os.getuid() != 10001 or os.geteuid() != 10001:
        raise ArtifactError("fixture execution requires native Linux x86-64 UID10001")
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.restype = ctypes.c_int
    prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    # Python's exec startup may have reset dumpability. The native child must
    # independently disable it again at its own entry before allocations.
    if prctl(4, 0, 0, 0, 0) != 0:
        raise ArtifactError("cannot disable fixture supervisor dumpability")
    dumpable, nnp, seccomp = (prctl(option, 0, 0, 0, 0) for option in (3, 39, 21))
    core, lock = resource.getrlimit(resource.RLIMIT_CORE), resource.getrlimit(resource.RLIMIT_MEMLOCK)
    if any(value < 0 for value in (*core, *lock)):
        raise ArtifactError("fixture execution requires finite core/memlock limits")
    status = dict(line.split(":", 1) for line in _read("/proc/self/status", 16384).splitlines() if ":" in line)
    membership = _read("/proc/self/cgroup", 8192).splitlines()
    paths = [line[3:] for line in membership if line.startswith("0::/")]
    if len(paths) != 1 or any(part in (".", "..") for part in paths[0].split("/")):
        raise ArtifactError("unresolvable unified cgroup namespace; host inspection required")
    root = Path("/sys/fs/cgroup")
    if not (root / "cgroup.controllers").is_file():
        raise ArtifactError("fixture execution requires unified cgroup v2")
    current = root / paths[0].lstrip("/")
    limits, available, swaps = [], [], []
    cpu, memory = _read(current / "cpuset.cpus.effective", 4096), _read(current / "cpuset.mems.effective", 4096)
    from .deployment import _mask
    if _mask(status["Cpus_allowed_list"].strip(), 65535) != _mask(cpu, 65535) or _mask(status["Mems_allowed_list"].strip(), 63) != _mask(memory, 63):
        raise ArtifactError("process affinity/memory masks differ from effective container cpusets")
    for _ in range(64):
        def scalar(name, directory=current):
            value = _read(directory / name, 64)
            return None if value == "max" else u64(value)
        cap, used, swap = scalar("memory.max"), scalar("memory.current"), scalar("memory.swap.max")
        if cap is not None:
            limits.append(cap)
            available.append(max(0, cap - used))
        if swap is not None:
            swaps.append(swap)
        if current == root:
            break
        current = current.parent
    else:
        raise ArtifactError("visible cgroup ancestor population exceeds64")
    if not limits or not swaps:
        raise ArtifactError("fixture execution requires finite effective cgroup memory/swap limits")
    counters = dict(line.split(":", 1) for line in _read("/proc/meminfo", 16384).splitlines())
    def kib(value):
        fields = value.split()
        if len(fields) != 2 or fields[1] != "kB":
            raise ArtifactError("invalid native memory observation")
        return u64(fields[0]) * 1024
    return {"architecture": "x86_64", "system": "Linux", "uid": os.getuid(), "euid": os.geteuid(),
        "dumpable": dumpable, "no_new_privileges": nnp, "seccomp_mode": seccomp,
        "core_soft": core[0], "core_hard": core[1], "memlock_soft": str(lock[0]), "memlock_hard": str(lock[1]),
        "host_limit_bytes": str(min(limits)), "host_available_bytes": str(min(*available, kib(counters["MemAvailable"]))),
        "swap_limit_bytes": str(min(swaps)), "locked_bytes": str(kib(status["VmLck"])),
        "cpu_mask": cpu, "memory_node_mask": memory}


def validate_preflight(document, registration, name):
    from .schemas import StrictValidator, validate
    from .deployment import _mask
    error = next(StrictValidator(PREFLIGHT).iter_errors(document), None)
    if error:
        raise ArtifactError(f"fixture process preflight: {error.message}")
    environment, config, probe, evidence, observed = (document[key] for key in ("environment", "probe_config", "probe", "probe_evidence", "observation"))
    for kind, value in (("deployment-environment", environment), ("probe", config), ("probe-report", probe), ("probe-evidence", evidence)):
        validate(kind, value)
    role = "server" if name.endswith("server") else "client"
    realization = registration["realizations"][role]
    verify_identity(environment, realization["environment_digest"])
    verify_identity(probe)
    verify_identity(evidence, probe["evidence_digest"])
    if not probe["qualified"] or any(value[key] != realization[key] for value in (config, probe, evidence) for key in ("environment_digest", "build_digest")):
        raise ArtifactError("fixture probe is unqualified or identifies another frozen realization")
    executor = registration["server_executor"] if role == "server" else "cuda"
    expected_role = "expert" if role == "server" else "client"
    if any((value["role"], value["executor"]) != (expected_role, executor) for value in (config, probe, evidence)):
        raise ArtifactError("fixture probe has another selected role/executor")
    expected_numa = [{"node": item["node"], "bytes": u64(item["bytes"])} for item in evidence["numa_available"]]
    if (probe["host_bytes"], probe["device_bytes"], probe["pinned_bytes"], probe["numa"]) != (u64(evidence["host_available_bytes"]), u64(evidence["device_available_bytes"]), u64(evidence["pinned_test_bytes"]), expected_numa):
        raise ArtifactError("fixture compact probe differs from actual raw capacities")
    target = environment["environment"]
    planning = environment["planning_request"]
    if (planning["role"], planning["executor"], planning["profile"], planning["logical_model_digest"], planning["operator_contract_digest"]) != (
            expected_role, executor, registration["profile"], registration["logical_model_digest"], registration["operator_contract_digest"]):
        raise ArtifactError("fixture frozen environment changes role/model/operator")
    if any(planning["caps"][key] > probe[key] for key in ("host_bytes", "device_bytes", "pinned_bytes")):
        raise ArtifactError("fixture frozen memory caps exceed authenticated probe headroom")
    if target["build_digest"] != realization["build_digest"] or target["source_lock_digest"] != registration["source_lock_digest"]:
        raise ArtifactError("fixture frozen image build/source identity differs")
    if (u64(evidence["cgroup_limit_bytes"]), u64(evidence["memlock_bytes"])) != (target["cgroup_bytes"], target["memlock_bytes"]) or evidence["gpu_uuid"] != target["gpu_uuid"] or config["expected_gpu_uuid"] != target["gpu_uuid"]:
        raise ArtifactError("fixture probe changed frozen cgroup/memlock/device settings")
    if executor == "cuda" and (evidence["compute_major"], evidence["compute_minor"], evidence["native_results"]) != (12, 0, [64, 32, 16]):
        raise ArtifactError("fixture CUDA probe lacks actual capability/native execution proof")
    if executor == "cpu" and (any(evidence[key] for key in ("driver_version", "runtime_version", "compute_major", "compute_minor")) or any(evidence["native_results"])):
        raise ArtifactError("CPU fixture probe reports CUDA execution")
    if u64(evidence["host_test_bytes"]) != config["max_host_test_bytes"] or u64(evidence["pinned_test_bytes"]) != config["max_pinned_test_bytes"]:
        raise ArtifactError("fixture probe changed its finite allocation tests")
    selected = set(config["numa_nodes"])
    if len(expected_numa) != len(selected) or {item["node"] for item in expected_numa} != selected or not selected.issubset(_mask(evidence["memory_node_mask"], 63)):
        raise ArtifactError("fixture probe NUMA inventory does not match selected allowed nodes")
    if any(_mask(value, 65535) != _mask(target["cpuset"], 65535) for value in (evidence["cpu_mask"], observed["cpu_mask"])) or _mask(observed["memory_node_mask"], 63) != _mask(evidence["memory_node_mask"], 63):
        raise ArtifactError("actual fixture CPU/NUMA masks differ from its qualified realization")
    if (u64(observed["host_limit_bytes"]), u64(observed["memlock_soft"]), u64(observed["memlock_hard"]), u64(observed["swap_limit_bytes"])) != (target["cgroup_bytes"], target["memlock_bytes"], target["memlock_bytes"], 0):
        raise ArtifactError("actual fixture cgroup/core/memlock restrictions changed")
    limits = registration["runs"][name]["hard_limits"]
    if u64(limits["max_rss_bytes"]) > u64(observed["host_available_bytes"]) or u64(limits["pinned_bytes"]) + u64(observed["locked_bytes"]) > target["memlock_bytes"]:
        raise ArtifactError("current cgroup or memlock headroom cannot admit the fixture child")
    return document
