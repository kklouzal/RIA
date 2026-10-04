"""Authoritative schema source; emit checked-in JSON with ``python -m ria.schemas``."""

from pathlib import Path
import ipaddress
import math
import struct

from jsonschema import Draft202012Validator, validators

from .identity import ArtifactError, SAFE_INTEGER, atomic_json, check_json, checked_product, u64
from .qualification_schema import QUALIFICATION_SCHEMAS
from .native_fixture_schema import NATIVE_FIXTURE_SCHEMAS
from .transport_fixture_schema import TRANSPORT_FIXTURE_SCHEMAS
from .release_runner import RELEASE_RUN_SCHEMAS
from .physical_contract import PHYSICAL_CONTRACT_SCHEMAS
from .fixture_runner import registration_schemas
from .fixture_environment import OBSERVATION

INT = {"type": "integer", "minimum": 0, "maximum": SAFE_INTEGER}
POS = {**INT, "minimum": 1}
U64 = {"type": "string", "pattern": "^(0|[1-9][0-9]*)$", "maxLength": 20}
SHA = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
TEXT = {"type": "string", "minLength": 1, "maxLength": 4096}
PATH = {"type": "string", "pattern": "^/", "maxLength": 4096}
BOOL = {"type": "boolean"}
REV = {"const": 1}
ROLE = {"enum": ["client", "expert"]}
EXECUTOR = {"enum": ["cpu", "cuda"]}
PROFILE = {"enum": ["nvfp4", "fp8", "bf16"]}
PHASE = {"enum": ["startup", "prefill", "decode", "continuation", "image", "drain"]}
PREFILL_ROWS = {**POS, "maximum": 64}
DTYPE = {"enum": ["BOOL", "U8", "I8", "U16", "I16", "U32", "I32", "U64", "I64", "F16", "BF16", "F32", "F64", "F8_E4M3", "F8_E5M2", "F8_E8M0"]}
FORMAT = {"enum": ["plain", "source_mxfp4", "nvfp4", "fp8_block32", "bf16", "engram_packed"]}
OPERATION = {"enum": ["embedding", "attention", "attention_state", "index", "router", "norm", "mhc", "expert_gate", "expert_up", "expert_down", "shared_gate", "shared_up", "shared_down", "engram", "vision", "output", "scale", "inactive"]}
SHAPE = {"type": "array", "items": INT, "maxItems": 8}


def array(item, *, maximum=1000000, minimum=0, unique=False):
    return {"type": "array", "items": item, "minItems": minimum, "maxItems": maximum, "uniqueItems": unique}


def obj(properties, optional=()):
    return {"type": "object", "properties": properties, "required": [k for k in properties if k not in optional], "additionalProperties": False}


def nullable(schema):
    return {"anyOf": [schema, {"type": "null"}]}


NUMA = obj({"node": INT, "bytes": INT})
NUMA["properties"]["node"] = {**INT, "maximum": 63}
CAPS = obj({"host_bytes": POS, "device_bytes": INT, "pinned_bytes": INT, "numa": array(NUMA, maximum=64)})
PLANNING = obj({"schema_revision": REV, "role": ROLE, "executor": EXECUTOR, "profile": PROFILE,
                "logical_model_digest": SHA, "operator_contract_digest": SHA, "context_positions": POS,
                "prefill_rows": PREFILL_ROWS, "caps": CAPS,
                "digest": SHA}, ("digest",))
ALLOCATION = obj({"id": U64, "name": TEXT, "resource": {"enum": ["host", "device"]}, "base_bytes": INT,
                  "bytes_per_position": INT, "numa_node": nullable({**INT, "maximum": 63}), "pinned": BOOL, "protected_progress": BOOL,
                  "phases": array(PHASE, maximum=6, minimum=1, unique=True)})
INVENTORY = obj({"schema_revision": REV, "logical_model_digest": SHA, "operator_contract_digest": SHA,
                 "semantic_max_positions": POS, "allocations": array(ALLOCATION, minimum=1), "digest": SHA}, ("digest",))
INVENTORY["properties"]["derivation"] = obj({"manifest_digest": SHA,"runtime_policy_digest": SHA,
    "request_digest": SHA,"context_positions": POS,"prefill_rows": PREFILL_ROWS})
PROBE_REPORT = obj({"schema_revision": REV, "role": ROLE, "executor": EXECUTOR, "host_bytes": POS, "device_bytes": INT,
                    "pinned_bytes": INT, "numa": array(NUMA, maximum=64), "qualified": BOOL,
                    "environment_digest": SHA, "build_digest": SHA, "evidence_digest": SHA, "digest": SHA})
CALIBRATION = obj({"schema_revision": REV, "profile": PROFILE, "operator_contract_digest": SHA,
                   "executor": EXECUTOR, "qualified": BOOL, "environment_digest": SHA, "build_digest": SHA,
                   "evidence_digest": SHA, "policy_digest": SHA, "digest": SHA})
PROBE = obj({"schema_revision": REV, "role": ROLE, "executor": EXECUTOR, "device_index": nullable({"const": 0}),
             "expected_gpu_uuid": nullable({"type": "string", "pattern": "^GPU-[0-9a-fA-F-]{36}$"}),
             "numa_nodes": array({"type": "integer", "minimum": 0, "maximum": 63}, maximum=64, minimum=1, unique=True), "max_host_test_bytes": POS,
             "max_device_test_bytes": INT, "max_pinned_test_bytes": INT, "deadline_ms": POS,
             "disable_core_dumps": {"const": True}, "environment_digest": SHA, "build_digest": SHA, "build_info_file": PATH})
TLS = obj({"ca_file": PATH, "certificate_file": PATH, "private_key_file": PATH, "expected_peer_name": TEXT,
           "minimum_version": {"const": "TLS1.3"}, "early_data": {"const": False}})
NETWORK = obj({k: TEXT if k.endswith("address") else POS for k in (
    "control_address", "bulk_address", "connect_timeout_ms", "handshake_timeout_ms", "operation_timeout_ms",
    "frame_io_timeout_ms", "write_timeout_ms", "max_row_lookup_rows", "max_inflight_payload_bytes",
    "max_frame_payload_bytes", "max_bulk_data_bytes", "max_inflight_expert_requests")})
NETWORK["properties"]["server_executor"] = EXECUTOR
NETWORK["required"].append("server_executor")
API = obj({"bind_address": TEXT, "bearer_token_file": PATH, "max_body_bytes": POS, "header_timeout_ms": POS,
           "body_timeout_ms": POS, "stream_write_timeout_ms": POS, "max_active_generations": {"const": 1},
           "max_queued_generations": {"const": 0}, "allow_remote_image_urls": {"const": False},
           "cors_allowed_origins": array(TEXT, maximum=100)})
for name in ("max_header_bytes", "max_json_depth", "max_json_nodes", "max_messages", "max_tools",
             "max_encoded_image_bytes", "max_decoded_image_bytes"):
    API["properties"][name] = POS
    API["required"].append(name)
API["properties"]["max_images"] = INT
API["required"].append("max_images")
API["properties"]["max_http_connections"] = {**POS, "maximum": 64}
API["required"].append("max_http_connections")
SERVICE = obj({"schema_revision": REV, "role": ROLE, "executor": EXECUTOR, "model_manifest": PATH,
               "memory_plan": PATH, "deployment_lock": PATH, "device_index": nullable({"const": 0}),
               "management_socket": {"const": "/run/dwarfstar/admin.sock"}, "artifacts_dir": {"const": "/artifacts"},
               "tls": TLS, "network": NETWORK, "api": API, "peer_grants": PATH,
               "placement_plan": PATH}, ("api", "peer_grants", "placement_plan"))
EXPERT_POLICY = obj({"numa_policy": {"enum": ["sharded", "replicated_experts", "replicated_server_model"]},
    "nodes": array(obj({"node": {**INT, "maximum": 63}, "cpus": array({**INT, "maximum": 65535}, minimum=1, maximum=256, unique=True),
        "workers": {**POS, "maximum": 64}, "local_bytes": U64}), minimum=1, maximum=64),
    "projection_tile_rows": {**POS, "maximum": 64}, "host_runtime_bytes": U64, "startup_host_bytes": U64,
    "device_workspace_bytes": U64, "pinned_workspace_bytes": U64, "drain_timeout_ms": POS})
SERVICE["properties"]["expert"] = EXPERT_POLICY
TENSOR = obj({"id": U64, "name": TEXT, "logical_shape": SHAPE, "physical_shape": SHAPE, "dtype": DTYPE,
              "format": FORMAT, "layout": {"enum": ["row_major_le", "opaque_source"]}, "shard": U64,
              "offset": U64, "length": U64, "sha256": SHA, "scale_ids": array(U64, maximum=32, unique=True),
              "operation": OPERATION, "placement": {"enum": ["client", "server", "both", "cache", "inactive"]},
              "alias_of": nullable(U64), "source_sha256": SHA, "byte_order": {"const": "little"},
              "bytes_per_block": POS, "block_values": POS, "group_shape": SHAPE,
              "value_row_stride": U64, "scale_row_stride": U64,
              "weight_global_scale_bits": {"type": "string", "pattern": "^[0-9a-f]{8}$"},
              "activation_global_scale_bits": {"type": "string", "pattern": "^[0-9a-f]{8}$"},
              "source_scale_sha256": SHA},
             ("value_row_stride", "scale_row_stride", "weight_global_scale_bits", "activation_global_scale_bits", "source_scale_sha256"))
SHARD = obj({"id": U64, "path": TEXT, "length": U64, "sha256": SHA, "data_start": U64,
             "chunk_size": POS, "chunk_hashes": array(SHA)})
REFERENCE = obj({"path": TEXT, "digest": SHA, "kind": {"enum": ["operator", "index", "metadata", "root", "layout"]}})
MANIFEST = obj({"schema_revision": REV, "role": {"enum": ["client", "server"]},
                "model_id": {"const": "deepseek-ai/DeepSeek-V4.1-Flash"}, "source_revision": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
                "profile": PROFILE, "logical_model_digest": SHA, "operator_contract_digest": SHA,
                "tokenizer_digest": SHA, "encoding_digest": SHA, "layout_digest": SHA,
                "tensors": array(TENSOR), "shards": array(SHARD), "tensor_pages": array(REFERENCE, maximum=4096),
                "metadata": array(REFERENCE, maximum=4096), "feature_exclusions": array(TEXT, maximum=100),
                "unexplained_required_tensors": {"const": []}, "digest": SHA})
ROOT = obj({"schema_revision": REV, "logical_model_digest": SHA, "operator_contract_digest": SHA,
            "manifest_digest": SHA, "metadata": array(REFERENCE, maximum=4096), "digest": SHA})
OPERATOR = obj({"schema_revision": REV, "profile": PROFILE, "source_revision": TEXT, "graph": {"const": "deepseek_v41_flash"},
                "weight_format": FORMAT, "activation_group": {"enum": [16, 32, 0]},
                "weight_scale_block": array(INT, maximum=2), "activation_quantizer": {"enum": ["nvfp4_dynamic16_calibrated", "fp8_e4m3fn_ue8m0_32", "bf16_rne"]},
                "clamp_f32_bits": {"const": "41200000"}, "gate_clamp": {"const": "upper_only"},
                "up_clamp": {"const": "two_sided"}, "coefficient_position": {"const": "before_down_quantizer"},
                "accumulator": {"const": "fp32"}, "reduction_order": {"const": "increasing_expert_id_then_shared"},
                "scale_reduction_domain": {"const": "full_original_population"}, "calibration_digest": nullable(SHA),
                "rounding": {"const": "ties_to_even"}, "nibble_order": {"const": "low_first"},
                "digest": SHA})
LAYOUT = obj({"schema_revision": REV, "logical_model_digest": SHA, "operator_contract_digest": SHA,
              "backend": {"enum": ["cpu", "cuda_sm120a", "source"]}, "tensors": array(TENSOR),
              "tensor_pages": array(REFERENCE, maximum=4096), "digest": SHA})
TENSOR_PAGE = obj({"schema_revision": REV, "tensors": array(TENSOR), "shards": array(SHARD), "digest": SHA})
PEAK = obj({"host_bytes": INT, "device_bytes": INT, "pinned_bytes": INT, "numa": array(NUMA, maximum=64)})
MEMORY_PLAN = obj({"schema_revision": REV, "admitted": {"const": True}, "role": ROLE, "executor": EXECUTOR,
    "profile": PROFILE, "logical_model_digest": SHA, "operator_contract_digest": SHA,
    "request_digest": SHA, "inventory_digest": SHA, "probe_digest": SHA, "calibration_digest": SHA,
    "environment_digest": SHA, "build_digest": SHA, "policy_digest": SHA,
    "context_positions": POS, "prefill_rows": PREFILL_ROWS, "allocation_count": POS, "caps": CAPS, "peak": PEAK,
    "phases": obj({name: PEAK for name in PHASE["enum"]}), "digest": SHA})
IMAGE = {"type": "string", "pattern": "^[a-zA-Z0-9._:/-]+@sha256:[0-9a-f]{64}$"}
ENVIRONMENT = obj({"image": IMAGE, "image_kind": {"enum": ["cpu", "cuda"]}, "build_digest": SHA,
                   "cpuset": {"type": "string", "pattern": "^[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*$"},
                   "cgroup_bytes": POS, "memlock_bytes": POS, "pids_limit": POS, "gpu_uuid": nullable(PROBE["properties"]["expected_gpu_uuid"]["anyOf"][0]),
                   "model_dir": PATH, "secret_dir": PATH, "report_dir": PATH, "seccomp_profile": PATH,
                   "bind_ip": TEXT, "api_port": POS, "start_period_seconds": POS, "stop_grace_seconds": POS,
                   "docker_version": TEXT, "compose_version": TEXT, "kernel_version": TEXT, "host_report_digest": SHA,
                   "source_lock_digest": SHA})
DEPLOYMENT = obj({"schema_revision": REV, "planning_request": PLANNING, "probe_config": PROBE,
                  "environment": ENVIRONMENT, "tls": TLS, "network": NETWORK, "api": API,
                  "native_ctl": PATH, "compose_files": array(PATH, minimum=1, maximum=3), "deadline_ms": POS,
                  "peer_grants": PATH, "placement_plan": PATH, "host_report": PATH,
                  "qualification_scope": {"enum": ["initial_fixture", "final_release"]}}, ("api", "peer_grants", "placement_plan"))
DEPLOYMENT["properties"]["expert"] = EXPERT_POLICY
FROZEN_ENVIRONMENT = obj({"schema_revision": REV, "environment": ENVIRONMENT,
    "planning_request": PLANNING,
    "tls": TLS, "network": NETWORK, "seccomp_sha256": SHA, "api": API, "expert": EXPERT_POLICY,
    "peer_grants_digest": SHA, "placement_plan_digest": SHA,
    "digest": SHA}, ("api", "expert", "peer_grants_digest", "placement_plan_digest"))
LOCK = obj({"schema_revision": REV, "role": ROLE, "executor": EXECUTOR, "image": IMAGE, "build_digest": SHA,
            "source_lock_digest": SHA, "logical_model_digest": SHA, "operator_contract_digest": SHA,
            "environment_digest": SHA, "probe_digest": SHA, "calibration_digest": SHA, "memory_plan_digest": SHA,
            "service_digest": SHA, "compose_digest": SHA, "host_report_digest": SHA, "seccomp_digest": SHA,
            "expected_peer_name": TEXT, "gpu_uuid": nullable(TEXT), "model_manifest_digest": SHA,
            "peer_grants_digest": SHA, "placement_plan_digest": SHA, "probe_evidence_digest": SHA,
            "calibration_evidence_digest": SHA, "policy_digest": SHA, "digest": SHA}, ("peer_grants_digest", "placement_plan_digest"))
LOCK["properties"].update(qualification_scope={"enum": ["initial_fixture", "final_release"]}, final_release_qualified=BOOL)
LOCK["required"].extend(("qualification_scope", "final_release_qualified"))
PLACEMENT_PLAN = obj({"schema_revision": REV, "logical_model_digest": SHA, "operator_contract_digest": SHA,
    "server_layout_digest": SHA, "server_executor": EXECUTOR,
    "schedule": {"const": "full_reference"}, "shared_placement": {"const": "client"}, "expert_policy": {"enum": ["remote", "explicit"]},
    "host_expert_cache_bytes": INT, "device_expert_cache_bytes": INT, "engram_cache_bytes": INT,
    "local_experts": array(obj({"layer": {"type": "integer", "minimum": 0, "maximum": 39},
        "expert": {"type": "integer", "minimum": 0, "maximum": 383}, "tier": {"enum": ["host", "vram"]},
        "local_phases": array({"enum": ["prefill", "decode", "continuation"]}, minimum=1, maximum=3, unique=True)}), maximum=15360),
    "runtime": obj({"tokenizer_file": PATH, "tokenizer_sha256": SHA, "tokenizer_memory_bytes": U64,
        "host_state_bytes": U64, "device_state_bytes": U64, "frontend_host_bytes": U64, "projection_tile_rows": POS,
        "state_tile_rows": POS, "max_image_patches": POS, "prefill_rows": PREFILL_ROWS}), "digest": SHA})
PEER_GRANTS = obj({"schema_revision": REV, "grants": array(obj({"expected_peer_name": TEXT,
    "logical_model_digest": SHA, "operator_contract_digest": SHA, "encoding_digest": SHA,
    "client_layout_digest": SHA, "placement_plan_digest": SHA, "profile": PROFILE,
    "server_executor": EXECUTOR, "server_layout_digest": SHA}), minimum=1, maximum=1024), "digest": SHA})

TRANSPORT_NETWORK = obj({name: TEXT if name.endswith("address") else {**POS, "maximum": 3600000}
    for name in ("control_address", "bulk_address", "connect_timeout_ms", "handshake_timeout_ms",
                 "frame_io_timeout_ms", "write_timeout_ms", "operation_timeout_ms")})
TRANSPORT_BOOTSTRAP = obj({"schema_revision": REV, "role": ROLE, "environment_digest": SHA,
    "build_digest": SHA, "build_info_file": PATH, "request_digest": SHA, "network": TRANSPORT_NETWORK,
    "tls": obj({"ca_file": PATH, "certificate_file": PATH, "private_key_file": PATH,
                "expected_peer_name": TEXT, "authorized_peer_sha256": SHA})})
TRANSPORT_REQUEST = obj({"schema_revision": REV, "kind": {"const": "transport_request"},
    **{name: SHA for name in ("environment_digest", "build_digest", "policy_digest", "logical_model_digest",
                             "source_lock_digest", "operator_contract_digest", "preregistration_digest")},
    "deadline_ms": {**POS, "maximum": 3600000}, "warmup": {**INT, "maximum": 100},
    "repeats": {**POS, "maximum": 1000}, "fixture_seed": INT,
    "max_frame_bytes": U64,
    **{name: U64 for name in ("control_credit", "expert_credit", "row_credit", "bulk_credit")}, "digest": SHA})
SCHEMAS = {"planning-request": PLANNING, "inventory": INVENTORY, "probe-report": PROBE_REPORT,
           "calibration": CALIBRATION, "probe": PROBE, "service": SERVICE, "manifest": MANIFEST,
           "root": ROOT, "operator-contract": OPERATOR, "physical-layout": LAYOUT,
           "deployment-request": DEPLOYMENT, "deployment-lock": LOCK}
SCHEMAS.update({"tensor-page": TENSOR_PAGE, "memory-plan": MEMORY_PLAN, "peer-grants": PEER_GRANTS, "placement-plan": PLACEMENT_PLAN,
                "transport-bootstrap": TRANSPORT_BOOTSTRAP, "transport-request": TRANSPORT_REQUEST})
SCHEMAS.update(QUALIFICATION_SCHEMAS)
SCHEMAS["deployment-environment"] = FROZEN_ENVIRONMENT
SCHEMAS["fixture-runtime-observation"] = OBSERVATION
SCHEMAS["fixture-container-lock"] = obj({"schema_revision": REV, "kind": {"const": "fixture_container_lock"},
    "container_id": SHA, "bootstrap_digest": SHA, "environment_digest": SHA, "image": IMAGE, "role": ROLE,
    "status": {"enum": ["running", "stopped"]}, "digest": SHA})
SCHEMAS.update(NATIVE_FIXTURE_SCHEMAS)
SCHEMAS.update(TRANSPORT_FIXTURE_SCHEMAS)
SCHEMAS.update(RELEASE_RUN_SCHEMAS)
SCHEMAS.update(PHYSICAL_CONTRACT_SCHEMAS)
SCHEMAS.update(registration_schemas())
StrictValidator = validators.extend(Draft202012Validator, type_checker=Draft202012Validator.TYPE_CHECKER.redefine(
    "integer", lambda checker, instance: type(instance) is int))


def validate(kind, value):
    check_json(value)
    schema = SCHEMAS.get(kind)
    if schema is None:
        raise ArtifactError(f"unknown schema kind: {kind}")
    error = next(StrictValidator(schema).iter_errors(value), None)
    if error:
        raise ArtifactError(f"{kind}: {'/'.join(map(str, error.path))}: {error.message}")
    _relations(kind, value)
    return value


def _relations(kind, value):
    if kind in ("service", "deployment-request", "transport-bootstrap"):
        for field in ("control_address", "bulk_address"):
            address = value["network"][field]
            try:
                if address.startswith("["):
                    host, port = address[1:].split("]:")
                    if ipaddress.ip_address(host).version != 6:
                        raise ValueError("brackets require IPv6")
                else:
                    host, port = address.split(":")
                    if ipaddress.ip_address(host).version != 4:
                        raise ValueError("IPv6 requires brackets")
                if not port.isascii() or not port.isdecimal() or not 1 <= int(port) <= 65535 or len(port) > 5:
                    raise ValueError("invalid port")
            except ValueError as exc:
                raise ArtifactError("native addresses require numeric IPv4/[IPv6]:port; TLS peer name is separate") from exc
    if kind == "transport-request":
        floors = {"control_credit": 131328, "expert_credit": 33160, "row_credit": 432, "bulk_credit": 1024}
        if not 4096 <= u64(value["max_frame_bytes"]) <= 16777216 or sum(u64(value[name]) for name in floors) > 67108864 or any(u64(value[name]) < minimum for name, minimum in floors.items()):
            raise ArtifactError("transport fixture frame/credits cannot provide bounded protected progress")
    if kind in ("planning-request", "probe", "service", "probe-report"):
        cpu = value["executor"] == "cpu"
        if value["role"] == "client" and cpu:
            raise ArtifactError("client neural executor must be CUDA")
        if kind == "planning-request" and cpu and (value["caps"]["device_bytes"] or value["caps"]["pinned_bytes"]):
            raise ArtifactError("CPU admission forbids GPU budgets")
        if kind == "probe" and cpu and (value["device_index"] is not None or value["expected_gpu_uuid"] is not None or value["max_device_test_bytes"] or value["max_pinned_test_bytes"]):
            raise ArtifactError("CPU probe must not initialize GPU")
        if kind in ("probe", "service") and not cpu and value["device_index"] != 0:
            raise ArtifactError("CUDA requires one device at ordinal zero")
        if kind == "service" and ((value["role"] == "client") != ("api" in value)):
            raise ArtifactError("API configuration is client-only and required")
        if kind == "service" and value["role"] == "expert" and "peer_grants" not in value:
            raise ArtifactError("expert requires immutable peer grants")
        if kind == "service" and value["role"] == "client" and "placement_plan" not in value:
            raise ArtifactError("client requires explicit placement plan")
        if kind == "service" and value["role"] == "expert" and value["network"]["server_executor"] != value["executor"]:
            raise ArtifactError("expert network executor disagrees with local executor")
    if kind in ("planning-request", "probe-report"):
        nodes = value["caps"]["numa"] if kind == "planning-request" else value["numa"]
        if len({node["node"] for node in nodes}) != len(nodes):
            raise ArtifactError("duplicate NUMA nodes")
    if kind in ("planning-request", "memory-plan") and value["prefill_rows"] > value["context_positions"]:
        raise ArtifactError("prefill microbatch exceeds semantic context capacity")
    if kind == "deployment-request":
        validate("planning-request", value["planning_request"])
    if kind == "service":
        network = value["network"]
        if network["max_bulk_data_bytes"] + 32 > network["max_frame_payload_bytes"] or network["max_frame_payload_bytes"] > network["max_inflight_payload_bytes"]:
            raise ArtifactError("bulk/frame/credit limits cannot progress")
    if kind in ("service", "deployment-request"):
        role = value["role"] if kind == "service" else value["planning_request"]["role"]
        executor = value["executor"] if kind == "service" else value["planning_request"]["executor"]
        if (role == "expert") != ("expert" in value):
            raise ArtifactError("expert service requires an explicit expert-only policy")
        if "expert" in value:
            policy = value["expert"]
            if len({node["node"] for node in policy["nodes"]}) != len(policy["nodes"]):
                raise ArtifactError("duplicate expert NUMA node")
            cpus = [cpu for node in policy["nodes"] for cpu in node["cpus"]]
            workers = sum(node["workers"] for node in policy["nodes"])
            if len(set(cpus)) != len(cpus) or workers > 128 or any(node["workers"] > len(node["cpus"]) for node in policy["nodes"]):
                raise ArtifactError("invalid bounded expert worker affinities")
            for name in ("host_runtime_bytes", "startup_host_bytes"):
                if u64(policy[name]) == 0:
                    raise ArtifactError("expert host reservations must be positive")
            for node in policy["nodes"]:
                if u64(node["local_bytes"]) == 0:
                    raise ArtifactError("expert NUMA reservation must be positive")
            if executor == "cpu" and (u64(policy["device_workspace_bytes"]) or u64(policy["pinned_workspace_bytes"])):
                raise ArtifactError("CPU expert cannot reserve CUDA workspace")
            if executor == "cuda" and (workers != 1 or u64(policy["device_workspace_bytes"]) == 0 or u64(policy["pinned_workspace_bytes"]) == 0):
                raise ArtifactError("CUDA expert requires one bounded stream owner")
    if kind == "placement-plan":
        entries = value["local_experts"]
        keys = [(entry["layer"], entry["expert"]) for entry in entries]
        if keys != sorted(set(keys)) or (value["expert_policy"] == "remote" and entries):
            raise ArtifactError("local expert membership must be explicit, sorted and unique")
        for tier, field in (("host", "host_expert_cache_bytes"), ("vram", "device_expert_cache_bytes")):
            if any(entry["tier"] == tier for entry in entries) and value[field] == 0:
                raise ArtifactError("local expert tier requires a positive admitted reservation")
        for field in ("tokenizer_memory_bytes", "host_state_bytes", "device_state_bytes", "frontend_host_bytes"):
            if u64(value["runtime"][field]) == 0:
                raise ArtifactError("client runtime requires explicit positive memory capacities")
    if kind == "inventory":
        ids = [u64(item["id"]) for item in value["allocations"]]
        if len(set(ids)) != len(ids):
            raise ArtifactError("duplicate physical allocation ID")
    if kind in ("manifest", "physical-layout", "tensor-page"):
        ids = [u64(tensor["id"]) for tensor in value["tensors"]]
        if len(set(ids)) != len(ids) or len({item["name"] for item in value["tensors"]}) != len(ids):
            raise ArtifactError("duplicate tensor identity")
        by_id = {tensor["id"]: tensor for tensor in value["tensors"]}
        complete = kind != "tensor-page" and not value.get("tensor_pages")
        from .safetensors import DTYPE_BYTES
        for tensor in value["tensors"]:
            u64(tensor["offset"])
            u64(tensor["length"])
            if kind != "tensor-page" and not value.get("tensor_pages") and any(u64(scale) not in ids for scale in tensor["scale_ids"]):
                raise ArtifactError("missing scale tensor")
            if kind != "tensor-page" and not value.get("tensor_pages") and tensor["alias_of"] is not None and u64(tensor["alias_of"]) not in ids:
                raise ArtifactError("missing tensor alias")
            shape, logical, format_name = tensor["physical_shape"], tensor["logical_shape"], tensor["format"]
            length = checked_product([checked_product(shape), DTYPE_BYTES[tensor["dtype"]]])
            if length != u64(tensor["length"]) or u64(tensor["offset"]) + length > (1 << 64) - 1:
                raise ArtifactError("tensor byte extent disagrees with dtype/shape or overflows")
            if len(shape) == 2 and "value_row_stride" in tensor and u64(tensor["value_row_stride"]) != shape[1] * DTYPE_BYTES[tensor["dtype"]]:
                raise ArtifactError("row-major value stride disagrees with physical shape")
            if format_name in ("nvfp4", "source_mxfp4", "fp8_block32"):
                group = 16 if format_name == "nvfp4" else 32
                packed = format_name != "fp8_block32"
                expected_group = [1, group] if packed else [32, 32]
                expected_dtype = "U8" if packed else "F8_E4M3"
                if len(logical) != 2 or min(logical) <= 0 or (packed and logical[1] % 2) or shape != [logical[0], logical[1] // 2 if packed else logical[1]] or tensor["dtype"] != expected_dtype or tensor["group_shape"] != expected_group or (tensor["bytes_per_block"], tensor["block_values"]) != (1, 2 if packed else 1) or len(tensor["scale_ids"]) != 1:
                    raise ArtifactError("quantized tensor descriptor violates its exact packing contract")
                if complete:
                    scale = by_id[tensor["scale_ids"][0]]
                    expected_scale = [logical[0] if packed else (logical[0] + 31) // 32, (logical[1] + group - 1) // group]
                    expected_scale_types = ("U8", "F8_E4M3") if format_name == "nvfp4" else ("U8", "F8_E8M0")
                    if scale["physical_shape"] != expected_scale or scale["dtype"] not in expected_scale_types or scale["operation"] != "scale":
                        raise ArtifactError("quantized scale descriptor violates block shape/type contract")
                if "scale_row_stride" not in tensor or u64(tensor["scale_row_stride"]) != (logical[1] + group - 1) // group:
                    raise ArtifactError("quantized scale row stride is absent or incorrect")
            elif format_name == "engram_packed":
                if len(logical) != 2 or logical[1] != 256 or shape != [logical[0], 264] or tensor["dtype"] != "U8" or tensor["scale_ids"] or "source_scale_sha256" not in tensor:
                    raise ArtifactError("Engram packed tensor requires lossless row256+8 provenance")
            elif format_name == "bf16" and (tensor["dtype"] != "BF16" or logical != shape):
                raise ArtifactError("BF16 descriptor shape/type mismatch")
            elif format_name == "plain" and logical != shape:
                raise ArtifactError("plain tensor logical and physical shapes disagree")
            for field in ("weight_global_scale_bits", "activation_global_scale_bits"):
                if field in tensor:
                    factor = struct.unpack("<f", bytes.fromhex(tensor[field])[::-1])[0]
                    if not math.isfinite(factor) or factor <= 0:
                        raise ArtifactError("global scale must be a positive finite FP32 dequant multiplier")
                elif format_name == "nvfp4":
                    raise ArtifactError("NVFP4 requires both exact global dequant multipliers")
            if complete:
                active, alias = {tensor["id"]}, tensor["alias_of"]
                while alias is not None:
                    if alias in active:
                        raise ArtifactError("tensor alias cycle")
                    active.add(alias)
                    target = by_id[alias]
                    if (target["shard"], target["offset"], target["length"], target["dtype"], target["physical_shape"]) != (tensor["shard"], tensor["offset"], tensor["length"], tensor["dtype"], tensor["physical_shape"]):
                        raise ArtifactError("alias does not reference identical physical allocation")
                    alias = target["alias_of"]
        for shard in value.get("shards", []):
            size, data_start, chunk = u64(shard["length"]), u64(shard["data_start"]), shard["chunk_size"]
            if not 8 <= data_start <= min(size, (16 << 20) + 8) or chunk > 4 << 20 or len(shard["chunk_hashes"]) != (size + chunk - 1) // chunk:
                raise ArtifactError("invalid shard header extent or chunk inventory")
    if kind == "operator-contract":
        expected = {"nvfp4": (16, [1, 16], "nvfp4_dynamic16_calibrated"), "fp8": (32, [32, 32], "fp8_e4m3fn_ue8m0_32"), "bf16": (0, [], "bf16_rne")}[value["profile"]]
        if (value["activation_group"], value["weight_scale_block"], value["activation_quantizer"]) != expected:
            raise ArtifactError("quantizer does not match declared profile")
        if value["profile"] == "nvfp4" and value["calibration_digest"] is None:
            raise ArtifactError("NVFP4 requires published calibration identity")
    if kind == "deployment-request":
        request, probe, env = value["planning_request"], value["probe_config"], value["environment"]
        validate("planning-request", request)
        validate("probe", probe)
        if (request["role"], request["executor"]) != (probe["role"], probe["executor"]):
            raise ArtifactError("probe and production roles differ")
        if (env["image_kind"] == "cpu") != (request["executor"] == "cpu"):
            raise ArtifactError("CPU/CUDA image mismatch")
        if env["gpu_uuid"] != probe["expected_gpu_uuid"] or (request["executor"] == "cuda" and env["gpu_uuid"] is None):
            raise ArtifactError("GPU exposure mismatch")
        if env["cgroup_bytes"] < request["caps"]["host_bytes"] or env["api_port"] > 65535:
            raise ArtifactError("invalid deployment bounds")
        if (request["role"] == "client") != ("api" in value):
            raise ArtifactError("client-only API configuration mismatch")
        if request["role"] == "expert" and "peer_grants" not in value:
            raise ArtifactError("expert deployment requires a peer grants input")
    if kind == "deployment-lock" and value["role"] == "expert" and "peer_grants_digest" not in value:
        raise ArtifactError("expert lock must authenticate peer grants")
    if kind == "deployment-lock" and value["role"] == "client" and "placement_plan_digest" not in value:
        raise ArtifactError("client lock must authenticate its placement plan")
    if kind == "deployment-lock" and value["final_release_qualified"] != (value["qualification_scope"] == "final_release"):
        raise ArtifactError("release qualification label disagrees with explicit calibration scope")
    if kind in ("service", "deployment-request"):
        for field, path in (("ca_file", "/run/secrets/ca.pem"), ("certificate_file", "/run/secrets/peer.pem"), ("private_key_file", "/run/secrets/peer.key")):
            if value["tls"][field] != path:
                raise ArtifactError("TLS credentials must use provisioned read-only secret paths")
        if "api" in value and value["api"]["bearer_token_file"] != "/run/secrets/api.token":
            raise ArtifactError("API token must use provisioned read-only secret path")


def generate(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, schema in SCHEMAS.items():
        Draft202012Validator.check_schema(schema)
        atomic_json(directory / (name + ".json"), {"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": f"https://github.com/kklouzal/RIA/schema/{name}.json", **schema})
    from .preparation import RECIPE_SCHEMA
    Draft202012Validator.check_schema(RECIPE_SCHEMA)
    atomic_json(directory / "preparation-recipe.json", {"$schema": "https://json-schema.org/draft/2020-12/schema", **RECIPE_SCHEMA})
    atomic_json(directory / "protocol.json", {"$schema": "https://json-schema.org/draft/2020-12/schema",
        "$ref": "../protocol/schema.json"})


if __name__ == "__main__":
    generate(Path(__file__).resolve().parents[2] / "schema")
