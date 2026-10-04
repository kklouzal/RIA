"""Normalize explicit operator settings and derive host-local serving requests.

No memory cap, numerical tolerance, placement membership or workload bound is
chosen by discovery. Derived identities are authenticated before publication;
changing a setting produces a different frozen realization and new evidence.
"""

from copy import deepcopy
import ipaddress
from pathlib import Path

from .deployment import _mask, probe_configuration
from .identity import ArtifactError, check_json, read_json, u64, verify_identity
from .schemas import StrictValidator, validate
from .setup_schema import setup_schemas

PROJECT_ROOT = Path(__file__).resolve().parents[2]
API_CONSTANTS = {"bind_address": "0.0.0.0:8000", "bearer_token_file": "/run/secrets/api.token",
                 "max_active_generations": 1, "max_queued_generations": 0,
                 "allow_remote_image_urls": False, "cors_allowed_origins": []}


def template_settings(role, executor):
    """Return an intentionally incomplete template, never a capacity policy.

    Nulls require explicit operator choices before validation. Required fields
    come from the same schema as actual settings; only stable serving/control
    constants and the chosen role/executor are populated here.
    """
    if role not in ("expert", "client") or executor not in ("cpu", "cuda") or (role == "client" and executor != "cuda"):
        raise ArtifactError("setup template requires expert cpu/cuda or client cuda")
    def skeleton(schema):
        if schema.get("type") == "object":
            return {name: skeleton(schema["properties"][name]) for name in schema["required"]}
        return None
    schema = setup_schemas()["setup-settings"]
    value = skeleton(schema)
    value.update(role=role, executor=executor, schema_revision=1,
                 setup_port=9010, gpu_uuid=None, security={"mode": "tls", "api_token_file": None})
    value["server_executor"] = executor if role == "expert" else None
    value["environment"]["api_port"] = 8000
    value["planning"]["caps"]["numa"] = [{"node": None, "bytes": None}]
    fields = schema["properties"]
    if role == "expert":
        value["model"] = {"mode": "source", "source_dir": None}
        value["expert"] = skeleton(fields["expert"])
        value["expert"]["nodes"] = [{"node": None, "cpus": None, "workers": None, "local_bytes": None}]
    else:
        value["client_runtime"] = skeleton(fields["client_runtime"])
        value["api"] = skeleton(fields["api"])
    return value


def _absolute(value, name):
    path = Path(value)
    if not path.is_absolute() or path == Path("/") or value.startswith("//") or any(part in (".", "..") for part in value.split("/")):
        raise ArtifactError(f"{name} must be an absolute non-root path without dot components")
    if any(character in value for character in "\n\r\x00$#'\"\\") or value != value.strip():
        raise ArtifactError(f"{name} cannot be represented by the sealed deployment environment")
    # Reject existing ancestor symlinks rather than resolving a caller's target
    # into an unrelated directory. Writers also use the dirfd publication APIs.
    current = path
    while current != current.parent:
        if current.is_symlink():
            raise ArtifactError(f"{name} contains a symlink")
        current = current.parent
    return str(path)


def _workspace(value):
    path = Path(value)
    # These fixed setup-image/host control paths must retain their image or
    # namespace authority. A workspace bind must never cover or modify them.
    for root in map(Path, ("/opt/RIA", "/opt/ria-setup", "/usr", "/etc", "/run",
                           "/var/run", "/sys", "/proc", "/dev", "/var/lib/docker", "/var/lib/containerd")):
        if path == root or root in path.parents or path in root.parents:
            raise ArtifactError("workspace overlaps protected setup/host control paths")
    return _absolute(str(value), "workspace")


def _private_address(value):
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise ArtifactError("setup addresses require explicit private numeric IPv4") from exc
    if address.version != 4 or not address.is_private or address.is_loopback or address.is_link_local or address.is_unspecified or address.is_multicast or address.is_reserved:
        raise ArtifactError("setup addresses require explicit private numeric IPv4")
    return str(address)


def _canonical_mask(value):
    members = sorted(_mask(value, 65535))
    ranges = []
    first = last = members[0]
    for member in members[1:]:
        if member == last + 1:
            last = member
        else:
            ranges.append(str(first) if first == last else f"{first}-{last}")
            first = last = member
    ranges.append(str(first) if first == last else f"{first}-{last}")
    return ",".join(ranges)


def workspace_paths(workspace):
    root = Path(_workspace(workspace))
    paths = {name: str(root / name) for name in ("operator", "model", "secrets", "reports", "bootstrap", "final")}
    paths["root"] = str(root)
    paths.update({name: str(root / "operator" / filename) for name, filename in (
        ("host_report", "host-report.json"), ("build_info", "build-info.json"),
        ("placement", "placement.json"), ("grants", "grants.json"))})
    return paths


def validate_settings(document):
    """Return a private normalized copy; input has no derived identity fields."""
    check_json(document, max_depth=32, max_nodes=200000)
    error = next(StrictValidator(setup_schemas()["setup-settings"]).iter_errors(document), None)
    if error:
        raise ArtifactError(f"setup settings {list(error.absolute_path)}: {error.message}")
    settings = deepcopy(document)
    settings["workspace"] = _workspace(settings["workspace"])
    settings["bind_address"] = _private_address(settings["bind_address"])
    if "peer_address" in settings:
        settings["peer_address"] = _private_address(settings["peer_address"])
    settings.setdefault("setup_port", 9010)
    settings.setdefault("security", {})
    settings["security"].setdefault("mode", "tls")
    settings["security"].setdefault("api_token_file", None)
    if settings["security"]["api_token_file"] is not None:
        settings["security"]["api_token_file"] = _absolute(settings["security"]["api_token_file"], "API token source")
    if settings["role"] == "expert" and settings["security"]["api_token_file"] is not None:
        raise ArtifactError("API token input is client-only")
    settings.setdefault("gpu_uuid", None)
    if settings["gpu_uuid"]:
        settings["gpu_uuid"] = "GPU-" + settings["gpu_uuid"][4:].lower()
    environment, planning, caps = settings["environment"], settings["planning"], settings["planning"]["caps"]
    environment["cpuset"] = _canonical_mask(environment["cpuset"])
    request = {"schema_revision": 1, "role": settings["role"], "executor": settings["executor"],
               "profile": settings["profile"], "logical_model_digest": "0" * 64,
               "operator_contract_digest": "0" * 64, **planning}
    validate("planning-request", request)
    if not caps["numa"] or any(node["bytes"] <= 0 for node in caps["numa"]):
        raise ArtifactError("setup requires explicit positive selected NUMA caps")
    caps["numa"].sort(key=lambda node: node["node"])
    if environment["cgroup_bytes"] < caps["host_bytes"] or caps["pinned_bytes"] > min(caps["host_bytes"], environment["memlock_bytes"]):
        raise ArtifactError("setup admission caps exceed cgroup/memlock bounds")
    cpu = settings["executor"] == "cpu"
    probe = settings["probe"]
    if cpu and (settings["gpu_uuid"] is not None or probe["max_device_test_bytes"] or probe["max_pinned_test_bytes"]):
        raise ArtifactError("CPU setup forbids GPU identity, queries and allocations")
    if not cpu and (not caps["device_bytes"] or not caps["pinned_bytes"] or not probe["max_device_test_bytes"] or not probe["max_pinned_test_bytes"]):
        raise ArtifactError("CUDA setup requires explicit positive device/pinned caps and probe bounds")
    for resource in ("host", "device", "pinned"):
        if probe[f"max_{resource}_test_bytes"] > caps[f"{resource}_bytes"]:
            raise ArtifactError("probe allocation bounds exceed declared capacities")
    network = settings["network"]
    if not (network["max_row_lookup_rows"] <= (1 << 32) - 1 and
            network["max_frame_payload_bytes"] <= 16 << 20 and
            network["max_bulk_data_bytes"] <= 4 << 20 and
            network["max_bulk_data_bytes"] + 64 <= network["max_frame_payload_bytes"] <= network["max_inflight_payload_bytes"] and
            network["max_inflight_expert_requests"] <= 2):
        raise ArtifactError("setup network limits cannot guarantee bounded progress")
    if settings["role"] == "expert":
        if settings["setup_port"] in (7443, 7444):
            raise ArtifactError("expert setup listener conflicts with a required native service port")
        if settings["server_executor"] != settings["executor"]:
            raise ArtifactError("expert server executor differs from its local executor")
        model = settings["model"]
        if model["mode"] in ("source", "prepared"):
            key = "source_dir" if model["mode"] == "source" else "package_dir"
            model[key] = _absolute(model[key], key)
        else:
            model.setdefault("hf_token_file", None)
            if model["hf_token_file"] is not None:
                model["hf_token_file"] = _absolute(model["hf_token_file"], "checkpoint token source")
        if model["mode"] in ("source", "download"):
            for field, default in (("chunk_size", 4 << 20), ("max_shard_bytes", 128 << 30), ("scratch_bytes", 64 << 20)):
                model.setdefault(field, default)
            if model["max_shard_bytes"] < 2 * model["chunk_size"]:
                raise ArtifactError("source shard bound must accommodate verification chunks")
        expert = settings["expert"]
        validate("service", {"schema_revision": 1, "role": "expert", "executor": settings["executor"],
            "model_manifest": "/model/manifest.json", "memory_plan": "/etc/dwarfstar/memory-plan.json",
            "deployment_lock": "/etc/dwarfstar/deployment-lock.json", "device_index": None if cpu else 0,
            "management_socket": "/run/dwarfstar/admin.sock", "artifacts_dir": "/artifacts",
            "tls": {"enabled": False}, "network": _network(settings), "peer_grants": "/etc/dwarfstar/peer-grants.json", "expert": expert})
        nodes = {node["node"]: node["bytes"] for node in caps["numa"]}
        cpus = {cpu for node in expert["nodes"] for cpu in node["cpus"]}
        if cpus != _mask(environment["cpuset"], 65535) or set(nodes) != {node["node"] for node in expert["nodes"]}:
            raise ArtifactError("expert nodes/CPUs must match the selected NUMA caps and cpuset")
        if any(u64(node["local_bytes"]) > nodes[node["node"]] for node in expert["nodes"]) or any(u64(expert[field]) > caps[resource] for field, resource in (
            ("host_runtime_bytes", "host_bytes"), ("startup_host_bytes", "host_bytes"),
            ("device_workspace_bytes", "device_bytes"), ("pinned_workspace_bytes", "pinned_bytes"))):
            raise ArtifactError("expert reservations exceed explicit setup caps")
        if u64(expert["host_runtime_bytes"]) > u64(expert["startup_host_bytes"]):
            raise ArtifactError("expert startup reservation must cover runtime")
        if not 1 <= expert["projection_tile_rows"] <= 64:
            raise ArtifactError("expert projection tile exceeds the native supported bounds")
        if u64(expert["pinned_workspace_bytes"]) > u64(expert["host_runtime_bytes"]):
            raise ArtifactError("expert pinned workspace must fit its host runtime reservation")
        if sum(u64(node["local_bytes"]) for node in expert["nodes"]) > u64(expert["startup_host_bytes"]):
            raise ArtifactError("expert NUMA reservations must fit its startup reservation")
    else:
        runtime = settings["client_runtime"]
        placement = {"schema_revision": 1, "logical_model_digest": "0" * 64, "operator_contract_digest": "0" * 64,
            "server_layout_digest": "0" * 64, "server_executor": settings["server_executor"],
            "schedule": "full_reference", "shared_placement": "client", "expert_policy": "explicit" if runtime["local_experts"] else "remote",
            **{key: deepcopy(value) for key, value in runtime.items() if key != "runtime"},
            "runtime": {**runtime["runtime"], "tokenizer_file": "/model/metadata/tokenizer.bin", "tokenizer_sha256": "0" * 64,
                        "prefill_rows": planning["prefill_rows"]}, "digest": "0" * 64}
        validate("placement-plan", placement)
        if any(not 1 <= runtime["runtime"][key] <= limit for key, limit in (
            ("projection_tile_rows", 4096), ("state_tile_rows", 4096), ("max_image_patches", 9216))):
            raise ArtifactError("client tiles/image population exceeds native supported bounds")
        if u64(runtime["runtime"]["device_state_bytes"]) > caps["device_bytes"] or sum(u64(runtime["runtime"][key]) for key in (
            "tokenizer_memory_bytes", "host_state_bytes", "frontend_host_bytes")) + caps["pinned_bytes"] > caps["host_bytes"]:
            raise ArtifactError("client runtime reservations exceed declared setup caps")
    return settings


def _network(settings):
    address = "0.0.0.0" if settings["role"] == "expert" else settings.get("peer_address")
    if address is None:
        raise ArtifactError("client request requires the invitation's exact expert IPv4")
    return {**settings["network"], "control_address": address + ":7443", "bulk_address": address + ":7444",
            "server_executor": settings["server_executor"]}


def build_request(settings, workspace, manifest, hostfacts, security, placement_or_grants, *, model_root=None):
    """Derive a strict request from authenticated local facts and sealed policy."""
    settings = validate_settings(settings)
    paths = workspace_paths(workspace)
    if paths["root"] != settings["workspace"]:
        raise ArtifactError("setup workspace differs from the authorized settings")
    validate("manifest", manifest)
    verify_identity(manifest)
    model_path = _absolute(str(model_root or paths["model"]), "prepared model root")
    if read_json(Path(model_path) / "manifest.json") != manifest:
        raise ArtifactError("selected prepared root differs from the authenticated manifest")
    verify_identity(hostfacts)
    report, build = hostfacts["host_report"], hostfacts["build_info"]
    verify_identity(report)
    verify_identity(build)
    expected = (settings["role"], settings["executor"], settings["service_image"])
    if hostfacts.get("kind") != "setup_host_facts" or tuple(hostfacts.get(key) for key in ("role", "executor", "service_image")) != expected or hostfacts.get("paths") != paths:
        raise ArtifactError("setup facts identify a different authorized realization/workspace")
    if manifest["role"] != ("server" if settings["role"] == "expert" else "client") or manifest["profile"] != settings["profile"]:
        raise ArtifactError("prepared manifest has another role/profile")
    if read_json(paths["host_report"]) != report or read_json(paths["build_info"]) != build:
        raise ArtifactError("published setup facts changed")
    source_lock = read_json(PROJECT_ROOT / "locks/source-lock.json")
    verify_identity(source_lock)
    if build.get("source_lock_digest") != source_lock["digest"] or build.get("image_kind") != settings["executor"]:
        raise ArtifactError("selected service image differs from the setup source lock/executor")
    gpu = hostfacts["gpu_uuid"]
    if report.get("selected_gpu_uuid") != gpu or report.get("client") is not (settings["role"] == "client"):
        raise ArtifactError("host report differs from selected physical device/role")
    from .setup_host import _validate_selection
    _validate_selection(settings, report, gpu)
    if settings["gpu_uuid"] is not None and settings["gpu_uuid"] != gpu:
        raise ArtifactError("discovered physical GPU differs from the requested UUID")
    policy_path = _absolute(str(placement_or_grants), "placement/grants")
    if policy_path != paths["grants" if settings["role"] == "expert" else "placement"]:
        raise ArtifactError("role policy must be published at its owned workspace path")
    policy = read_json(policy_path)
    validate("peer-grants" if settings["role"] == "expert" else "placement-plan", policy)
    verify_identity(policy)
    tls = deepcopy(security["tls"])
    if tls.get("enabled", True) != (settings["security"]["mode"] == "tls"):
        raise ArtifactError("provisioned transport mode differs from setup settings")
    request = {"schema_revision": 1, "qualification_scope": "initial_fixture",
        "host_report": paths["host_report"], "native_ctl": str(PROJECT_ROOT / "bin/ds4ctl"),
        "compose_files": [str(PROJECT_ROOT / "deploy" / ("compose." + settings["role"] + ".yml"))],
        "deadline_ms": settings["deadline_ms"],
        "planning_request": {"schema_revision": 1, "role": settings["role"], "executor": settings["executor"],
            "profile": settings["profile"], "logical_model_digest": manifest["logical_model_digest"],
            "operator_contract_digest": manifest["operator_contract_digest"], **deepcopy(settings["planning"])},
        "environment": {"image": settings["service_image"], "image_kind": settings["executor"], "build_digest": build["digest"],
            **deepcopy(settings["environment"]), "gpu_uuid": gpu,
            "model_dir": model_path, "secret_dir": paths["secrets"], "report_dir": paths["reports"],
            "seccomp_profile": str(PROJECT_ROOT / "deploy/seccomp-numa.json"), "bind_ip": settings["bind_address"],
            **{field: report[field] for field in ("docker_version", "compose_version", "kernel_version")},
            "host_report_digest": report["digest"], "source_lock_digest": source_lock["digest"]},
        "probe_config": {"schema_revision": 1, "role": settings["role"], "executor": settings["executor"],
            "device_index": None if settings["executor"] == "cpu" else 0, "expected_gpu_uuid": gpu,
            "numa_nodes": [node["node"] for node in settings["planning"]["caps"]["numa"]],
            **deepcopy(settings["probe"]), "disable_core_dumps": True, "build_info_file": "/usr/share/dwarfstar/build-info.json"},
        "tls": tls, "network": _network(settings)}
    if settings["role"] == "expert":
        request["expert"] = deepcopy(settings["expert"])
        request["peer_grants"] = policy_path
        if settings["executor"] == "cuda":
            request["compose_files"].append(str(PROJECT_ROOT / "deploy/compose.expert-cuda.yml"))
    else:
        request["api"] = {**deepcopy(settings["api"]), **deepcopy(API_CONSTANTS)}
        request["placement_plan"] = policy_path
    request["probe_config"] = probe_configuration(request)
    validate("deployment-request", request)
    return request
