"""Fixed release obligations and the availability of their semantic producers.

G01--G28 are the plan's names for the specification's cumulative acceptance
requirements, not operator-defined numeric checks. Some complete offline
obligations have runnable tests; other obligations have partial producers or
missing instrumentation. No full semantic release adapter is registered. A
machine can have adequate hardware and still lack required test software.
"""

from .identity import seal
from .qualification_schema import MATRIX_AXES

SPECIFICATION_SHA256 = "14224cdb33476944111e14f69a5679f0597c192a44d67b48f048910f326f3f6e"

# A gate validator must derive these obligations from specific raw observations
# and independent oracles. Renaming a metric, adding pass booleans, or authenticating
# an external report does not implement the corresponding producer/validator.
GATE_OBLIGATIONS = {
    "G01": ("Source identity/inventory", "18.8;19.4;22.13", (
        "trusted_source_and_complete_target_tensor_scale_encoding_inventory",
        "modified_hash_wrong_artifact_unknown_variant_rejected",
        "approved_physical_layouts_preserve_logical_identity")),
    "G02": ("Schema/canonical identity", "18.7 additional contract tests;22.13", (
        "duplicate_unicode_nonfinite_revision_type_enum_unknown_field_rejection",
        "jcs_order_escape_number_self_digest_and_u64_above_safe_integer",
        "checked_offsets_above_4gib_and_1tib_labeled_synthetic")),
    "G03": ("Preparation integrity", "18.7 additional contract tests;19.4", (
        "bounded_truncation_overlap_layout_and_scale_validation",
        "authenticated_metadata_graph_and_chunk_permission_overfetch",
        "bounded_scratch_and_source_repack_population_peaks")),
    "G04": ("Quantization/operator parity", "18.1;18.2;18.8", (
        "all_codes_scales_rounding_nibbles_padding_odd_tile_block_boundaries",
        "strict_w4a4_w4a16_negative_w8a8_bf16_sensitive_f32_casts",
        "partition_quantizer_scope_and_both_full_model_fidelity_axes")),
    "G05": ("Expert/router parity", "18.2;18.3", (
        "target_shapes_gate_up_clamps_coefficient_before_down_quantization",
        "zero_extreme_coefficients_shared_once_original_slots_ordered_reduction",
        "raw_normalized_weights_biased_selection_text_image_close_tied_scores")),
    "G06": ("Attention/graph/state", "18.2;18.5;18.8", (
        "actual_sm120_target_causality_sparse_candidates_partial_blocks_rope_sinks",
        "source_aliases_tile_normalization_ced_mhc_engram_history_and_fusion",
        "exact_packed_history_and_state_bytes_independent_reference")),
    "G07": ("CPU mode/ISA", "18.3;18.8;22.11", (
        "gpu_free_actual_x86_isa_target_experts_all_three_profiles",
        "complete_runtime_linkage_no_cuda_driver_initialization_or_framework",
        "target_model_and_broad_bank_execution")),
    "G08": ("GPU mode/architecture", "18.3;18.8;22.11", (
        "gpu_free_aot_build_exact_uuid_sm120a_client_and_qualified_server",
        "unsupported_device_profile_library_rejected_before_bank_loading",
        "no_sm100_substitution_or_first_request_compilation")),
    "G09": ("Client tiers", "18.4;22.11", (
        "real_target_footprint_exceeds_forced_and_physical_vram",
        "nonexpert_weights_exact_state_expert_host_hits_separate_engram_caches",
        "measured_h2d_d2h_leases_zero_cpu_neural_no_remote_state_or_disk_spill")),
    "G10": ("Server bounds", "18.4;18.8;22.11", (
        "expert_only_no_duplicate_attention_kv_or_complete_transformer",
        "cache_below_selected_union_cold_double_misses_broad_prefill_progress",
        "bounded_all_pools_zero_cuda_server_cpu_expert_fallback")),
    "G11": ("NUMA", "18.3;18.4;18.8;22.11", (
        "all_required_policies_real_local_replicas_distinct_job_ownership",
        "per_node_capacity_and_local_remote_penalty_observations",
        "default_seccomp_denial_and_insufficient_node_reject_no_scatter")),
    "G12": ("Memory/capacity", "18.4;22.11", (
        "process_cgroup_startup_conversion_tmpfs_driver_unique_locked_pinned_peaks",
        "physical_pages_locality_major_faults_disk_reads_swap_observed",
        "minimum_zero_tiny_cache_union_under_budget_node_and_registration_failures",
        "actual_high_capacity_witness_separate_from_sparse_offset_fixture")),
    "G13": ("Exact continuation/features", "18.5;18.8", (
        "long_append_tool_image_chunk_source_candidate_ring_partial_alias_cases",
        "exact_retained_state_vs_independent_exact_reconstruction",
        "approximate_replay_negative_profile_invalidation_no_migration_requantization")),
    "G14": ("Framing/request association", "18.7 additional contract tests", (
        "exact_lengths_csr_rows_slots_experts_unique_ranges_dtype_layout_digest_epoch_nan",
        "duplicate_missing_out_of_order_replies_no_attacker_size_allocation",
        "no_credit_no_buffer_progress_partial_tls_operation_frame_write_deadlines")),
    "G15": ("Binding/bulk/cancellation", "18.6;18.7 additional contract tests", (
        "zero_to_bound_fresh_epoch_namespaces_one_use_grants_failed_reconnect",
        "cancel_before_dispatch_in_workers_after_result_racing_ack_terminal",
        "one_terminal_one_credit_dirty_leases_no_late_dma_or_prefetch_publication")),
    "G16": ("Two-host authentication/network", "18.8;22.9;22.11", (
        "physical_two_hosts_san_client_auth_wrong_expired_credentials",
        "bulk_binding_unauthorized_host_real_firewall_denial",
        "broken_pipe_stalled_receiver_bounded_timeouts_no_weight_miss_traffic")),
    "G17": ("Admission/rendering", "15.2;15.3;18.7 additional contract tests;22.10", (
        "bootstrap_without_final_plan_native_exact_allocation_equations",
        "incomplete_calibration_capability_secret_uuid_resource_not_admitted_rejected",
        "changed_final_resources_reprobe_atomic_publication_restart")),
    "G18": ("Effective Compose", "22.8 effective Compose configuration;22.13", (
        "cpu_expert_cuda_expert_client_exact_files_env_compose_version",
        "inherited_image_uuid_memory_override_rejected_path_spaces_dollars",
        "effective_config_equals_policy_actual_inspect_and_production_probe_settings")),
    "G19": ("Readiness/health", "15.4;22.10;22.11", (
        "hash_peer_kernel_residency_incomplete_startup_remain_unready",
        "expert_local_ready_before_client_client_waits_bound_session",
        "busy_healthy_bounded_progress_timeout_small_health_without_weight_scan")),
    "G20": ("HTTP/SSE", "18.5;22.9 user-facing API", (
        "auth_header_body_image_context_tool_depth_sampling_model_feature_bounds",
        "one_active_zero_queue_caller_prefix_isolation",
        "utf8_fragments_stops_length_usage_one_terminal_stalled_cancel_remote_failure")),
    "G21": ("Security/confidentiality", "16.4;16 crash confidentiality;22.11", (
        "authorized_uid_mount_write_private_admin_cap_seccomp_lsm_no_dangerous_mounts",
        "core_dump_attempt_no_secrets_or_state_dump",
        "initialized_network_padding_private_generation_isolation_untrusted_reports")),
    "G22": ("Fault/quiescence", "16.1;16.2;18.6", (
        "fault_before_dispatch_partial_tx_after_executor_before_reply_copy_writeback_sample",
        "worker_loss_restart_cuda_error_oom_stale_wrong_binding_cancel",
        "typed_failure_invalid_uncertain_state_bounded_quiescence_no_substitute_output")),
    "G23": ("Container pressure/shutdown", "16.5;22.10;22.11", (
        "cgroup_oom_pin_failure_numa_denial_gpu_loss_systemd_maintenance",
        "broken_stalled_peers_sigterm_sigkill_dirty_writeback_finite_stop_grace",
        "no_hung_event_unsafe_recycle_or_same_epoch_restart")),
    "G24": ("Durability/mode change", "22.7;22.10;22.11", (
        "container_recreate_remove_preserves_checkpoint_manifest_reports",
        "no_private_state_resume_or_source_bank_cleanup",
        "drain_stop_release_rebind_mode_change_failed_teardown_unavailable")),
    "G25": ("Build/reproducibility", "22.3;22.11", (
        "locked_offline_dependencies_digest_bases_gpu_free_build",
        "complete_source_library_runtime_architecture_and_linkage_metadata",
        "compared_outputs_provenance_explained_nondeterministic_metadata")),
    "G26": ("Image/distribution hygiene", "22.3;22.11", (
        "contexts_history_layers_logs_exclude_weights_secrets_prompts_build_tools",
        "authorized_acquisition_uses_buildkit_secrets",
        "sbom_notices_trusted_digests_schemas_tests_runbook_driver_free_cpu_image")),
    "G27": ("Performance/soak", "17;18.7;22.11", (
        "matched_native_container_cold_warm_startup_ttft_p95_p99_committed_rate_peaks",
        "equal_workload_locality_cache_broad_routing_context_and_mode_comparisons",
        "links_power_thermal_concurrency_uncertainty_minimum_one_hour_production_soak")),
    "G28": ("Static/test-target completeness", "22.13;22.14", (
        "actual_template_extraction_duplicate_aware_schema_yaml_json_docker_shell_markdown",
        "separate_pure_cpu_cuda_docker_physical_results_no_missing_required_tests",
        "triaged_failures_resolved_candidates_post_change_bottleneck_review")),
}

MATRIX_OBLIGATIONS = (
    "full_target_and_independent_reference_both_fidelity_axes",
    "actually_observed_profile_executor_residency_placement_phase_numa_policy",
    "original_slot_routes_ownership_expert_cache_hits_misses_and_network_traffic",
    "physical_host_device_state_migrations_and_zero_client_cpu_neural_operations",
    "real_numa_pages_replicas_and_all_resource_peaks_inside_release_images",
)

# Availability describes supplied software, not execution or qualification.
# Each position corresponds to an obligation above. "available" means there
# is a runnable producer/test for that obligation; "partial" means the cited
# path exercises only part of it. Neither makes a selfsealed gate pass valid.
OBLIGATION_SOFTWARE = {
    "G01": (("available", "tools/ria/target.py;tools/verify_ria.py"),
            ("partial", "tests/ria/test_artifacts.py;tests/ria/test_native_artifacts.py"),
            ("available", "tests/ria/test_expert.c;tests/ria/test_native_artifacts.py")),
    "G02": (("available", "tests/ria/test_artifacts.py;tests/ria/test_deployment.py;tests/ria/test_contracts.c"),
            ("available", "tests/ria/test_artifacts.py;tests/ria/test_contracts.c"),
            ("available", "tests/ria/test_expert.c")),
    "G03": (("available", "tests/ria/test_artifacts.py;tests/ria/test_native_artifacts.py"),
            ("available", "tests/ria/test_artifacts.py;tools/ria/client.py"),
            ("partial", "tools/ria/preparation.py;tests/ria/test_artifacts.py")),
    "G04": (("partial", "tests/ria/test_artifacts.py;tests/ria/test_expert.c;ria/qualify.c"),
            ("partial", "ria/qualify.c;tests/ria/test_qualify.py"),
            ("partial", "tools/qualify_ria.py compare-logits;tools/ria/fixture_runner.py")),
    "G05": (("available", "ria/qualify.c;tests/ria/reference_expert.py"),
            ("partial", "ria/qualify.c;tests/ria/test_expert.c;tests/ria/test_graph_prefill.c;tests/ria/test_prefill_expert_oracle.py"),
            ("partial", "ria/qualify.c;tests/ria/test_graph_reference.py;tests/ria/test_graph_prefill.c")),
    "G06": (("partial", "ria/qualify.c;tests/ria/test_graph_reference.py;tests/ria/test_graph_prefill.c"),
            ("partial", "ria/qualify.c;tests/ria/test_graph_contracts.c;tests/ria/test_graph_prefill.c"),
            ("partial", "tests/ria/test_state.c;tests/ria/test_graph_reference.py;tests/ria/test_graph_prefill.c")),
    "G07": (("partial", "tools/qualify_ria.py run-fixture;deploy/Dockerfile.cpu"),
            ("partial", "deploy/build_info.py;tests/ria/test_build_contract.py"),
            ("missing", "")),
    "G08": (("available", "Makefile.ria ria-cuda;deploy/check_cuda_artifacts.py;ria/probe.c"),
            ("partial", "tests/ria/test_probe_config.c;ria/runtime.c"),
            ("available", "deploy/check_cuda_artifacts.py;tests/ria/test_build_contract.py")),
    "G09": (("missing", ""), ("partial", "tools/qualify_ria.py run-fixture"), ("missing", "")),
    "G10": (("partial", "tests/ria/test_server_contracts.c;ria/inventory.c"),
            ("partial", "tests/ria/test_server_contracts.c;tests/ria/test_graph_prefill.c;tests/ria/test_server_prefill.c"),
            ("partial", "ria/qualify.c;ria/server.c;tests/ria/test_server_prefill.c")),
    "G11": (("partial", "ria/probe.c;tests/ria/test_server_contracts.c"),
            ("missing", ""), ("partial", "ria/probe.c;tests/ria/test_probe_config.c")),
    "G12": (("partial", "tools/ria/fixture_runner.py;tools/ria/host.py"),
            ("partial", "ria/probe.c;tools/ria/host.py"),
            ("partial", "tests/ria/test_contracts.c;tests/ria/test_server_contracts.c"),
            ("missing", "")),
    "G13": (("partial", "tools/run_ria_release.py;tests/ria/test_state.c;tests/ria/test_graph_prefill.c;tests/ria/test_engine_prefill.c"),
            ("partial", "tests/ria/test_graph_prefill.c"),
            ("partial", "tests/ria/test_state.c;tests/ria/test_frontend_contracts.c;tests/ria/test_engine_prefill.c")),
    "G14": (("available", "tests/ria/test_contracts.c;tests/ria/test_server_contracts.c"),
            ("partial", "tests/ria/test_contracts.c;tests/ria/test_server_contracts.c;tests/ria/test_server_lifecycle.c"),
            ("partial", "ria/qualify_transport.c;tests/ria/test_transport_qualifier.py")),
    "G15": (("partial", "tests/ria/test_contracts.c;ria/qualify_transport.c"),
            ("partial", "tests/ria/test_contracts.c;tests/ria/test_server_contracts.c;tests/ria/test_server_prefill.c;tests/ria/test_server_lifecycle.c"),
            ("missing", "")),
    "G16": (("partial", "ria/qualify_transport.c;tools/qualify_ria.py run-fixture"),
            ("partial", "ria/qualify_transport.c"), ("partial", "ria/qualify_transport.c")),
    "G17": (("available", "tests/ria/test_deployment.py;tests/ria/test_contracts.c;tools/build_inventory.py"),
            ("available", "tests/ria/test_deployment.py;tests/ria/test_probe_config.c"),
            ("available", "tests/ria/test_deployment.py;tests/ria/test_fixture_container.py")),
    "G18": (("available", "tests/ria/test_deployment.py;tests/ria/test_container_inspection.py"),
            ("partial", "tests/ria/test_deployment.py"),
            ("available", "tools/render_deployment.py;tools/ria/container_inspection.py")),
    "G19": (("partial", "tests/ria/test_server_contracts.c;tests/ria/test_admin.c"),
            ("partial", "tests/ria/test_admin.c;ria/server.c"),
            ("partial", "tests/ria/test_admin.c;tools/ria/container_inspection.py")),
    "G20": (("partial", "tests/ria/test_frontend_contracts.c;tests/ria/test_release_runner.py"),
            ("partial", "tests/ria/test_frontend_contracts.c;tests/ria/test_encoding.py"),
            ("partial", "tests/ria/test_release_runner.py;tools/run_ria_release.py")),
    "G21": (("partial", "tests/ria/test_container_inspection.py;tests/ria/test_admin.c"),
            ("missing", ""), ("partial", "tests/ria/test_contracts.c;tests/ria/test_state.c")),
    "G22": (("missing", ""), ("partial", "tests/ria/test_server_contracts.c"),
            ("partial", "tests/ria/test_server_contracts.c;tests/ria/test_state.c")),
    "G23": (("missing", ""), ("partial", "tests/ria/test_server_contracts.c"), ("missing", "")),
    "G24": (("missing", ""), ("partial", "tests/ria/test_state.c"), ("missing", "")),
    "G25": (("available", ".github/workflows/ria.yml;deploy/Dockerfile.cpu;deploy/Dockerfile.cuda;tools/verify_ria.py"),
            ("partial", "deploy/build_info.py;deploy/check_cuda_artifacts.py"), ("missing", "")),
    "G26": (("partial", ".dockerignore;deploy/Dockerfile.cpu;deploy/Dockerfile.cuda"),
            ("available", "deploy/Dockerfile.cpu;deploy/Dockerfile.cuda"),
            ("partial", ".github/workflows/ria.yml;deploy/build_info.py")),
    "G27": (("partial", "tools/run_ria_release.py"), ("missing", ""),
            ("partial", "tools/run_ria_release.py")),
    "G28": (("partial", "deploy/offline_checks.py;tests/ria/test_deployment.py"),
            ("partial", "Makefile.ria;.github/workflows/ria.yml"), ("missing", "")),
}


def _gate_readiness(row):
    software = [{"id": obligation, "status": status,
                 "supplied_paths_or_commands": paths.split(";") if paths else []}
                for obligation, (status, paths) in zip(row["obligations"], OBLIGATION_SOFTWARE[row["id"]], strict=True)]
    states = {entry["status"] for entry in software}
    return {**row, "obligation_software": software,
            "status": "available_software_unexecuted" if states == {"available"} else
                      "blocked_missing_software" if states == {"missing"} else "partial_software",
            "semantic_release_adapter": "missing",
            "blocker": "Complete authenticated gate receipt/semantic release adapter is missing; per-obligation software availability is listed separately."}


def gate_catalog():
    return seal({"schema_revision": 1, "specification_sha256": SPECIFICATION_SHA256,
        "gates": [{"id": gate, "title": data[0], "specification_sections": data[1],
                   "obligations": list(data[2])} for gate, data in GATE_OBLIGATIONS.items()],
        "matrix_axes": MATRIX_AXES, "matrix_obligations": list(MATRIX_OBLIGATIONS)})


def qualification_readiness():
    """Describe runnable partial producers and missing software, never execution."""
    catalog = gate_catalog()
    return seal({"schema_revision": 1, "kind": "qualification_software_readiness",
        "executed": False, "hardware_qualified": False, "release_ready": False,
        "gate_catalog_digest": catalog["digest"],
        "partial_producers": [
            {"command": "tools/qualify_ria.py run-fixture", "scope": "initial_fixture",
             "observes": ["target_shaped_experts", "graph_state_cases", "transfers",
                          "paired_mtls_transport_cases", "complete_child_rss_and_deadline"],
             "missing": ["full_target_model", "full_gate_cases", "all_matrix_cells"]},
            {"command": "tools/run_ria_release.py --execute", "scope": "http_replay",
             "observes": ["text_reasoning_tools_images_continuation", "native_token_timings",
                          "bounded_requests", "actual_monotonic_soak"],
             "missing": ["route_cache_residency_numa_observations", "fault_injection",
                         "independent_full_model_reference"]},
            {"command": "tools/qualify_ria.py compare-logits", "scope": "supplied_corpus",
             "observes": ["same_realization_or_native_source_saved_logit_comparison"],
             "missing": ["reference_and_candidate_logit_execution_producer",
                         "full_model_oracle_corpus_positions_profile_and_determinism_registration",
                         "all_state_operator_fault_capacity_and_container_cases"]}],
        "gates": [_gate_readiness(row) for row in catalog["gates"]],
        "matrix": {"required_cells": 540, "status": "blocked_missing_software",
                   "axes": MATRIX_AXES, "obligations": list(MATRIX_OBLIGATIONS),
                   "blocker": "No supplied full-target route/cache/residency/NUMA instrumentation and independent-reference matrix producer."},
        "required_external_inputs": [
            "authorized_complete_checkpoint_and_independent_native_source_reference",
            "reviewed_preregistered_operator_and_full_model_policies_and_corpus",
            "physical_x86_64_gpu_free_cpu_host_and_rtx5090_sm120_client",
            "separately_qualified_blackwell_server_and_sufficient_real_numa_capacity",
            "administrator_owned_fault_firewall_cgroup_gpu_access_and_container_recreation_controls"]})
