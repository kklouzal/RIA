"""Authoritative preregistration and offline result-comparison contracts."""

NUMBER = {"type": "number", "minimum": 0}
SHA = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
TEXT = {"type": "string", "minLength": 1, "maxLength": 4096}
BOOL = {"type": "boolean"}
SCOPE = {"enum": ["initial_fixture", "final_release"]}


def record(fields, optional=()):
    return {"type": "object", "properties": fields,
            "required": [key for key in fields if key not in optional],
            "additionalProperties": False}


THRESHOLDS = record({
    "max_abs_error": NUMBER, "max_relative_error": NUMBER,
    "max_rms_error": NUMBER, "max_loss_delta": NUMBER,
    "relative_floor": {"type": "number", "exclusiveMinimum": 0},
})
POLICY = record({
    "schema_revision": {"const": 1}, "logical_model_digest": SHA,
    "source_lock_digest": SHA,
    "thresholds": record({"same_realization": THRESHOLDS, "native_source": THRESHOLDS}),
    "minimum_soak_seconds": {"type": "integer", "minimum": 3600, "maximum": 9007199254740991},
    "ordered_objectives": {"type": "array", "minItems": 1, "maxItems": 16,
        "items": {"type": "string", "minLength": 1, "maxLength": 4096}, "uniqueItems": True},
    "registered_at": {"type": "string", "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"},
    "digest": SHA,
}, ("registered_at", "digest"))

REFERENCE = record({"evidence_path": TEXT, "evidence_digest": SHA, "passed": BOOL})
COMPONENT_NAMES = ["experts", "graph_state", "transfers", "transport", "resource_bounds"]
COMPONENT_REFERENCE = record({**REFERENCE["properties"], "name": {"enum": COMPONENT_NAMES}})
COMPARISON_REFERENCE = record({**REFERENCE["properties"], "axis": {"enum": ["same_realization", "native_source"]}})
COMPONENT = record({
    "schema_revision": {"const": 1}, "kind": {"const": "qualification_component"},
    "component": {"enum": COMPONENT_NAMES}, "qualification_scope": SCOPE,
    "environment_digest": SHA, "build_digest": SHA, "policy_digest": SHA,
    "profile": {"enum": ["nvfp4", "fp8", "bf16"]}, "executor": {"enum": ["cpu", "cuda"]},
    "operator_contract_digest": SHA, "fixture_digest": SHA,
    "registration_digest": SHA, "registration_path": TEXT,
    "raw_evidence": {"type": "array", "minItems": 1, "maxItems": 8,
                     "items": record({"path": TEXT, "digest": SHA})},
    "checks": {"type": "array", "minItems": 1, "maxItems": 4096,
               "items": record({"id": TEXT, "passed": BOOL})},
    "measurements": {"type": "object", "minProperties": 1, "maxProperties": 4096,
                     "additionalProperties": {"type": "number"}},
    "qualified": BOOL, "digest": SHA,
})
CALIBRATION_EVIDENCE = record({
    "schema_revision": {"const": 1}, "kind": {"const": "calibration_evidence"},
    "qualification_scope": SCOPE, "environment_digest": SHA, "build_digest": SHA,
    "policy_digest": SHA, "profile": {"enum": ["nvfp4", "fp8", "bf16"]},
    "executor": {"enum": ["cpu", "cuda"]}, "operator_contract_digest": SHA,
    "components": {"type": "array", "minItems": 5, "maxItems": 5, "items": COMPONENT_REFERENCE},
    "comparisons": {"type": "array", "maxItems": 2, "items": COMPARISON_REFERENCE},
    "release_matrix_digest": {"anyOf": [SHA, {"type": "null"}]},
    "release_matrix_path": {"anyOf": [TEXT, {"type": "null"}]},
    "release_runs": {"type": "array", "maxItems": 4096,
                     "items": record({"path": TEXT, "digest": SHA})},
    "soak_seconds": {"type": "integer", "minimum": 0, "maximum": 9007199254740991},
    "passed": BOOL, "digest": SHA,
}, ("release_runs",))
QUALIFICATION_SCHEMAS = {"qualification-policy": POLICY,
    "qualification-component": COMPONENT, "calibration-evidence": CALIBRATION_EVIDENCE}
MATRIX_AXES = {
    "profile": ["nvfp4", "fp8", "bf16"], "server_executor": ["cpu", "cuda"],
    "client_residency": ["resident_reference", "host_backed"],
    "placement": ["all_remote", "vram_hit", "host_only_local", "host_only_remote", "mixed"],
    "phase": ["prefill", "decode", "continuation"],
    "numa_policy": ["sharded", "replicated_experts", "replicated_server_model"],
}
MATRIX_EVIDENCE = record({"path": TEXT, "digest": SHA})
MATRIX_STATUS = {"enum": ["unexecuted", "passed", "failed", "unavailable"]}
MATRIX_ROW_FIELDS = {"status": MATRIX_STATUS,
    "evidence": {"type": "array", "maxItems": 4096, "items": MATRIX_EVIDENCE}}
RELEASE_MATRIX = record({"schema_revision": {"const": 1}, "hardware_qualified": BOOL,
    "axes": {"const": MATRIX_AXES},
    "cells": {"type": "array", "minItems": 540, "maxItems": 540,
        "items": record({**{key: {"enum": values} for key, values in MATRIX_AXES.items()}, **MATRIX_ROW_FIELDS})},
    "gates": {"type": "array", "minItems": 28, "maxItems": 28,
        "items": record({"id": {"enum": [f"G{index:02}" for index in range(1, 29)]}, **MATRIX_ROW_FIELDS})},
    "required_features": {"const": ["text", "reasoning", "tools", "images", "exact_continuation"]},
    "minimum_soak_seconds": {"type": "integer", "minimum": 3600, "maximum": 9007199254740991},
    "digest": SHA})
QUALIFICATION_SCHEMAS["release-matrix"] = RELEASE_MATRIX
U64 = {"type": "string", "pattern": "^(0|[1-9][0-9]*)$", "maxLength": 20}
MASK = {"type": "string", "pattern": "^[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*$", "maxLength": 4096}
PROBE_EVIDENCE = record({
    "schema_revision": {"const": 1}, "kind": {"const": "probe_evidence"},
    "role": {"enum": ["client", "expert"]}, "executor": {"enum": ["cpu", "cuda"]},
    "environment_digest": SHA, "build_digest": SHA, "architecture": {"const": "x86_64"},
    "uid": {"const": 10001}, "dumpable": {"const": False}, "seccomp_mode": {"const": 2},
    "no_new_privileges": {"const": True}, "cgroup_limit_bytes": U64, "cgroup_available_bytes": U64,
    "swap_limit_bytes": {"const": "0"}, "memlock_bytes": U64,
    "cpu_mask": MASK, "memory_node_mask": MASK, "host_test_bytes": U64, "host_test_ms": U64,
    "numa_locality_proven": {"const": True},
    "driver_version": {"type": "integer", "minimum": 0, "maximum": 2147483647},
    "runtime_version": {"type": "integer", "minimum": 0, "maximum": 2147483647},
    "gpu_uuid": {"anyOf": [{"type": "string", "pattern": "^GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$"}, {"type": "null"}]},
    "compute_major": {"type": "integer", "minimum": 0, "maximum": 99},
    "compute_minor": {"type": "integer", "minimum": 0, "maximum": 99},
    "gpu_allocation_ms": U64, "native_kernel_ms": U64,
    "native_results": {"type": "array", "minItems": 3, "maxItems": 3, "items": NUMBER},
    "host_parent_preflight_required": {"const": True},
    "host_available_bytes": U64, "device_available_bytes": U64, "pinned_test_bytes": U64,
    "numa_available": {"type": "array", "minItems": 1, "maxItems": 64,
        "items": record({"node": {"type": "integer", "minimum": 0, "maximum": 63}, "bytes": U64})},
    "digest": SHA,
})
QUALIFICATION_SCHEMAS["probe-evidence"] = PROBE_EVIDENCE

LOGITS_COMPARISON = record({
    "schema_revision": {"const": 1}, "kind": {"const": "teacher_forced_comparison"},
    "axis": {"enum": ["same_realization", "native_source"]},
    **{key: SHA for key in ("policy_digest", "logical_model_digest", "reference_operator_digest",
                           "candidate_operator_digest", "input_digest", "reference_sha256", "candidate_sha256")},
    "positions": {"type": "integer", "minimum": 2, "maximum": 1048576},
    "labels": {"type": "integer", "minimum": 1, "maximum": 1048575},
    "metrics": record({key: NUMBER for key in ("max_abs_error", "max_relative_error", "max_rms_error", "max_loss_delta")}),
    "reference_nll": NUMBER, "candidate_nll": NUMBER, "passed": BOOL,
    "qualification_scope": {"const": "supplied teacher-forced corpus only; other release gates remain separate"},
    "digest": SHA,
})
QUALIFICATION_SCHEMAS["logits-comparison"] = LOGITS_COMPARISON
