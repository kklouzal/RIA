"""Authoritative native fixture request, raw measurements and case populations.

Native measurements remain unqualified. The supervised policy owner validates
joint client/server provenance, explicit caps, exact case population and the
accepted relative floor before producing qualification-component evidence.
"""
from .qualification_schema import SHA, record

DECIMAL_U64 = {"type": "string", "pattern": "^(0|[1-9][0-9]{0,19})$"}
U64 = {"anyOf": [DECIMAL_U64, {"type": "integer", "minimum": 0, "maximum": 9007199254740991}]}
EXECUTOR = {"enum": ["cpu", "cuda"]}
PROFILE = {"enum": ["bf16", "fp8", "nvfp4"]}
ROLE = {"enum": ["client", "server"]}
UUID = {"type": "string", "pattern": "^GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"}
NULLABLE_UUID = {"anyOf": [UUID, {"type": "null"}]}
IDENTITY_FIELDS = ("logical_model_digest", "source_lock_digest", "environment_digest", "build_digest",
                   "policy_digest", "operator_contract_digest", "preregistration_digest")
NUMBER = {"type": "number", "minimum": 0}
POSITIVE = {"type": "number", "exclusiveMinimum": 0}
NATIVE_REQUEST = record({
    "schema_revision": {"const": 1}, "kind": {"const": "native_fixture_request"},
    "profile": PROFILE, "executor": EXECUTOR, "runner_executor": EXECUTOR, "role": ROLE,
    "gpu_uuid": NULLABLE_UUID, "expert_shape": {"enum": ["ragged", "target"]},
    **{key: SHA for key in IDENTITY_FIELDS},
    "host_budget": U64, "device_budget": U64, "pinned_budget": U64,
    "deadline_ms": {"type": "integer", "minimum": 1, "maximum": 600000},
    "repeats": {"type": "integer", "minimum": 1, "maximum": 32},
    "warmup": {"type": "integer", "minimum": 0, "maximum": 8},
    "fixture_seed": U64, "relative_floor": POSITIVE,
})
NATIVE_REQUEST["allOf"] = [
    {"if": {"properties": {"role": {"const": "client"}}},
     "then": {"properties": {"runner_executor": {"const": "cuda"}}}},
    *[{"if": {"properties": {"role": {"const": "server"}, "executor": {"const": executor}}},
       "then": {"properties": {"runner_executor": {"const": executor}}}} for executor in ("cpu", "cuda")],
    {"if": {"properties": {"runner_executor": {"const": "cpu"}}},
     "then": {"properties": {"gpu_uuid": {"type": "null"},
                             "device_budget": {"enum": [0, "0"]}, "pinned_budget": {"enum": [0, "0"]}}}},
    {"if": {"properties": {"runner_executor": {"const": "cuda"}}},
     "then": {"properties": {"gpu_uuid": UUID,
                             "device_budget": {"not": {"enum": [0, "0"]}},
                             "pinned_budget": {"not": {"enum": [0, "0"]}}}}},
]
CASE = record({
    "id": {"type": "string", "minLength": 1, "maxLength": 4096},
    "timing_group": {"type": "string", "minLength": 1, "maxLength": 4096},
    "warmup": NATIVE_REQUEST["properties"]["warmup"],
    "component": {"enum": ["experts", "graph_state", "transfers"]},
    "path": {"enum": ["host", "vram", "gpu", "pinned"]},
    **{key: {"type": "integer", "minimum": 0, "maximum": 9007199254740991}
       for key in ("rows", "input_features", "intermediate_features", "output_features", "elements")},
    **{key: SHA for key in ("input_digest", "oracle_digest", "actual_digest")},
    **{key: NUMBER for key in ("max_abs_error", "max_relative_error", "max_rms_error")},
    "requires_exact_match": {"type": "boolean"}, "exact_match": {"type": "boolean"},
    "repeat_identical": {"type": ["boolean", "null"]}, "placement_identical": {"type": ["boolean", "null"]},
    **{key: DECIMAL_U64 for key in ("owned_host_bytes", "device_bytes", "pinned_bytes", "startup_ns")},
    "latency_ns": {"type": "array", "maxItems": 32, "items": DECIMAL_U64},
})
NATIVE_MEASUREMENTS = record({
    "schema_revision": {"const": 1}, "kind": {"const": "native_fixture_measurements"},
    "qualified": {"const": False}, "qualification_scope": {"const": "initial_fixture"},
    "fixture_algorithm": {"const": "ria-native-fixtures-v1"}, "request_digest": SHA,
    "profile": PROFILE, "executor": EXECUTOR, "runner_executor": EXECUTOR, "role": ROLE,
    **{key: SHA for key in IDENTITY_FIELDS}, "gpu_uuid": NULLABLE_UUID,
    "fixture_seed": DECIMAL_U64, "relative_floor": POSITIVE,
    **{key: DECIMAL_U64 for key in ("owned_host_peak_bytes", "device_peak_bytes", "pinned_peak_bytes", "elapsed_ns", "graph_startup_ns")},
    "process_peak_rss_bytes": {"anyOf": [DECIMAL_U64, {"type": "null"}]},
    "warmup": NATIVE_REQUEST["properties"]["warmup"], "repeats": NATIVE_REQUEST["properties"]["repeats"],
    "host_accounting_scope": {"const": "owned_heap_metadata_and_pinned_pools; process_RSS_also_observed"},
    "latency_scope": {"const": "synchronous_H2D_operator_D2H; selected_experts_include_all_three_projections"},
    "deadline_enforcement": {"const": "monotonic_control_points_and_required_external_process_supervisor"},
    "full_model_graph_parity": {"const": "unexecuted"}, "model_quality": {"const": "unexecuted"},
    "graph_gpu_fixtures": {"enum": ["executed", "unexecuted"]},
    "transfer_gpu_fixtures": {"enum": ["executed", "unexecuted"]},
    "cases": {"type": "array", "minItems": 2, "maxItems": 32, "items": CASE},
})
NATIVE_SEALED_MEASUREMENTS = record({**NATIVE_MEASUREMENTS["properties"], "digest": SHA})
CPU_CASES = ("expert_dense_clamp_route", "expert_unit_palette")
CUDA_SERVER_CASES = ("expert_dense_clamp_route", "expert_resident_dense_clamp_route",
                     "expert_unit_palette", "expert_resident_unit_palette", "pinned_multichunk_odd_bytes")
CUDA_CLIENT_CASES = (*CUDA_SERVER_CASES,
    "window_fp8_group32", "compressed_fp4_group16_e4m3", "index_fp4_group32_ue8m0",
    "mhc_flatten_sinkhorn20_post_orientation", "mhc_pre_post_comb_fp32",
    "compressor_featurewise_ratio2_fp32_to_bf16", "rope_window_forward", "rope_window_inverse",
    "rope_yarn_forward", "rope_yarn_inverse", "attention_zero_query_causal65_sink",
    "attention_ragged129_online_bf16_probabilities", "index_full_closed_form_scores",
    "csa2_full_newest_partial_block", "index_reindex_closed_form_scores", "csa2_reindex_reuses_source_block_mask",
    "engram_e4m3_row256_scale32_decode", "engram_per_stream_normalized_signed_sqrt_fusion", "engram_image_mask_passes_through_exact",
    "router_bias_selects_unbiased_coefficients", "router_original_selected_ids", "expert_original_id_order_shared_once_bf16",
    "sensitive_f32_projection_preserves_unrounded_input", "source_rmsnorm5120_zero_and_nonzero_bf16")
REPEATED_CASES = frozenset((*CUDA_SERVER_CASES, "window_fp8_group32", "compressed_fp4_group16_e4m3", "index_fp4_group32_ue8m0"))
EXACT_CASES = frozenset(("pinned_multichunk_odd_bytes", "window_fp8_group32", "compressed_fp4_group16_e4m3", "index_fp4_group32_ue8m0",
    "csa2_full_newest_partial_block", "csa2_reindex_reuses_source_block_mask", "engram_e4m3_row256_scale32_decode",
    "engram_image_mask_passes_through_exact", "router_original_selected_ids", "expert_original_id_order_shared_once_bf16",
    "sensitive_f32_projection_preserves_unrounded_input"))
TIMING_GROUPS = {"mhc_pre_post_comb_fp32": "mhc_flatten_sinkhorn20_post_orientation",
                 "router_original_selected_ids": "router_bias_selects_unbiased_coefficients"}
CASE_DIMENSIONS = {
    "window_fp8_group32": (1, 512, 0, 512, 512), "compressed_fp4_group16_e4m3": (1, 512, 0, 512, 512),
    "index_fp4_group32_ue8m0": (1, 128, 0, 128, 128),
    "mhc_flatten_sinkhorn20_post_orientation": (4, 5120, 24, 5120, 20480), "mhc_pre_post_comb_fp32": (1, 20480, 0, 24, 24),
    "compressor_featurewise_ratio2_fp32_to_bf16": (2, 512, 0, 512, 512),
    **{key: (1, 512, 257, 512, 512) for key in ("rope_window_forward", "rope_window_inverse", "rope_yarn_forward", "rope_yarn_inverse")},
    "attention_zero_query_causal65_sink": (64, 512, 65, 512, 32768), "attention_ragged129_online_bf16_probabilities": (64, 512, 129, 512, 32768),
    **{key: (16409, 128, 32, 1, 16409) for key in ("index_full_closed_form_scores", "index_reindex_closed_form_scores")},
    **{key: (1, 16409, 2048, 512, 512) for key in ("csa2_full_newest_partial_block", "csa2_reindex_reuses_source_block_mask")},
    "engram_e4m3_row256_scale32_decode": (24, 256, 8, 256, 6144),
    **{key: (4, 5120, 25600, 5120, 20480) for key in ("engram_per_stream_normalized_signed_sqrt_fusion", "engram_image_mask_passes_through_exact")},
    "router_bias_selects_unbiased_coefficients": (1, 384, 6, 6, 6), "router_original_selected_ids": (1, 384, 0, 6, 6),
    "expert_original_id_order_shared_once_bf16": (6, 5120, 0, 5120, 5120),
    "sensitive_f32_projection_preserves_unrounded_input": (1, 65, 0, 19, 19),
    "source_rmsnorm5120_zero_and_nonzero_bf16": (2, 5120, 0, 5120, 10240),
}


def expected_dimensions(case_id, expert_shape, *, pinned_bytes=None):
    """Tuple order: rows,K,intermediate,N,elements; graph rows vary by op."""
    if case_id == "pinned_multichunk_odd_bytes":
        if type(pinned_bytes) is not int or pinned_bytes < 8320:
            raise ValueError("transfer dimensions require its verified page-rounded pinned pool")
        size = pinned_bytes + 137
        return 1, size, 0, size, size
    if case_id in (*CPU_CASES, "expert_resident_dense_clamp_route", "expert_resident_unit_palette"):
        if expert_shape == "ragged":
            return 17, 65, 33, 19, 323
        if expert_shape == "target":
            return 1, 5120, 2304, 5120, 5120
        raise ValueError("unknown expert fixture shape")
    return CASE_DIMENSIONS[case_id]


def expected_samples(case_id, repeats):
    return repeats if case_id in REPEATED_CASES else 1


def expected_warmup(case_id, warmup):
    return warmup if case_id in REPEATED_CASES else 0


def expected_cases(role, runner_executor):
    """Return the exact immutable population; no inferred missing proofs."""
    if role == "server" and runner_executor == "cpu":
        return CPU_CASES
    if role == "server" and runner_executor == "cuda":
        return CUDA_SERVER_CASES
    if role == "client" and runner_executor == "cuda":
        return CUDA_CLIENT_CASES
    raise ValueError("unsupported native fixture role/executor")


NATIVE_FIXTURE_SCHEMAS = {"native-fixture-request": NATIVE_REQUEST,
                        "native-fixture-measurements": NATIVE_MEASUREMENTS,
                        "native-fixture-sealed-measurements": NATIVE_SEALED_MEASUREMENTS}
