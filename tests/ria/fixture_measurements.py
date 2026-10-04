"""Synthetic offline provenance documents; these never execute or qualify hardware."""

from pathlib import Path

from ria.fixture_runner import RUNS, _component, derive_request
from ria.identity import atomic_json, digest, seal
from ria.native_fixture_schema import (EXACT_CASES, REPEATED_CASES, TIMING_GROUPS,
                                     expected_cases, expected_dimensions, expected_samples, expected_warmup)
from ria.transport_fixture_schema import CASE_KINDS, CASE_STATUSES, expected_checks


def synthetic_environment(model, source, operator, profile, executor, role, build, *, tls_enabled=True):
    gpu = "GPU-12345678-1234-1234-1234-123456789abc" if role == "client" or executor == "cuda" else None
    return seal({"schema_revision": 1, "seccomp_sha256": "8" * 64,
        "environment": {"image": "registry.test/ria@sha256:" + "9" * 64, "image_kind": "cuda" if gpu else "cpu", "build_digest": build,
            "cpuset": "0-1", "cgroup_bytes": 1073741824, "memlock_bytes": 1048576, "pids_limit": 64,
            "gpu_uuid": gpu, "model_dir": "/model", "secret_dir": "/secrets", "report_dir": "/reports", "seccomp_profile": "/seccomp.json",
            "bind_ip": "127.0.0.1", "api_port": 8000, "start_period_seconds": 300, "stop_grace_seconds": 30,
            "docker_version": "synthetic", "compose_version": "synthetic", "kernel_version": "synthetic", "host_report_digest": "7" * 64,
            "source_lock_digest": source},
        "planning_request": {"schema_revision": 1, "role": "client" if role == "client" else "expert", "executor": "cuda" if gpu else "cpu",
            "profile": profile, "logical_model_digest": model, "operator_contract_digest": operator, "context_positions": 64, "prefill_rows": 8,
            "caps": {"host_bytes": 268435456, "device_bytes": 134217728 if gpu else 0, "pinned_bytes": 1048576 if gpu else 0,
                     "numa": [{"node": 0, "bytes": 268435456}]}},
        "tls": {"ca_file": "/run/secrets/ca.pem", "certificate_file": "/run/secrets/peer.pem", "private_key_file": "/run/secrets/peer.key",
            "expected_peer_name": "peer.test.internal", "minimum_version": "TLS1.3", "early_data": False} if tls_enabled else {"enabled": False},
        "network": {"control_address": "127.0.0.1:7443", "bulk_address": "127.0.0.1:7444", "server_executor": executor,
            **{key: 10 for key in ("connect_timeout_ms", "handshake_timeout_ms", "frame_io_timeout_ms", "write_timeout_ms", "operation_timeout_ms")},
            "max_row_lookup_rows": 256, "max_inflight_payload_bytes": 67108864, "max_frame_payload_bytes": 16777216,
            "max_bulk_data_bytes": 4194304, "max_inflight_expert_requests": 2}})


def preflight(reg, role, environment=None):
    environment = environment or synthetic_environment(reg["logical_model_digest"], reg["source_lock_digest"], reg["operator_contract_digest"],
        reg["profile"], reg["server_executor"], role, reg["realizations"][role]["build_digest"],
        tls_enabled=reg["runs"]["transport_" + role]["bootstrap_body"]["tls"].get("enabled", True))
    base = reg["realizations"][role]
    planning, target = environment["planning_request"], environment["environment"]
    gpu = target["gpu_uuid"] is not None
    config = {"schema_revision": 1, "role": planning["role"], "executor": planning["executor"], "device_index": 0 if gpu else None,
        "expected_gpu_uuid": target["gpu_uuid"], "numa_nodes": [0], "max_host_test_bytes": 4096, "max_device_test_bytes": 4096 if gpu else 0,
        "max_pinned_test_bytes": planning["caps"]["pinned_bytes"], "deadline_ms": 1000, "disable_core_dumps": True,
        **base, "build_info_file": "/usr/share/dwarfstar/build-info.json"}
    evidence = seal({"schema_revision": 1, "kind": "probe_evidence", "role": planning["role"], "executor": planning["executor"], **base,
        "architecture": "x86_64", "uid": 10001, "dumpable": False, "seccomp_mode": 2, "no_new_privileges": True,
        "cgroup_limit_bytes": str(target["cgroup_bytes"]), "cgroup_available_bytes": "536870912", "swap_limit_bytes": "0",
        "memlock_bytes": str(target["memlock_bytes"]), "cpu_mask": target["cpuset"], "memory_node_mask": "0",
        "host_test_bytes": "4096", "host_test_ms": "1", "numa_locality_proven": True, "driver_version": 1 if gpu else 0, "runtime_version": 1 if gpu else 0,
        "gpu_uuid": target["gpu_uuid"], "compute_major": 12 if gpu else 0, "compute_minor": 0, "gpu_allocation_ms": "1" if gpu else "0",
        "native_kernel_ms": "1" if gpu else "0", "native_results": [64, 32, 16] if gpu else [0, 0, 0], "host_parent_preflight_required": True,
        "host_available_bytes": str(planning["caps"]["host_bytes"]), "device_available_bytes": str(planning["caps"]["device_bytes"]),
        "pinned_test_bytes": str(planning["caps"]["pinned_bytes"]), "numa_available": [{"node": 0, "bytes": str(planning["caps"]["host_bytes"])}]})
    probe = seal({"schema_revision": 1, "role": planning["role"], "executor": planning["executor"], **base,
        "host_bytes": planning["caps"]["host_bytes"], "device_bytes": planning["caps"]["device_bytes"], "pinned_bytes": planning["caps"]["pinned_bytes"],
        "numa": [{"node": 0, "bytes": planning["caps"]["host_bytes"]}], "qualified": True, "evidence_digest": evidence["digest"]})
    return {"environment": environment, "probe_config": config, "probe": probe, "probe_evidence": evidence,
        "observation": {"architecture": "x86_64", "system": "Linux", "uid": 10001, "euid": 10001, "dumpable": 0,
            "no_new_privileges": 1, "seccomp_mode": 2, "core_soft": 0, "core_hard": 0,
            "memlock_soft": str(target["memlock_bytes"]), "memlock_hard": str(target["memlock_bytes"]), "host_limit_bytes": str(target["cgroup_bytes"]),
            "host_available_bytes": "536870912", "swap_limit_bytes": "0", "locked_bytes": "0", "cpu_mask": target["cpuset"], "memory_node_mask": "0"}}


def registration(policy, *, profile="bf16", executor="cpu", server_environment=None,
                 server_build="b" * 64, client_environment="c" * 64, client_build="d" * 64,
                 operator="3" * 64, binary="e" * 64, environments=None, tls_enabled=True):
    environments = environments or {}
    for role, build in (("server", server_build), ("client", client_build)):
        if role not in environments:
            environments[role] = synthetic_environment(policy["logical_model_digest"], policy["source_lock_digest"], operator, profile, executor, role, build,
                tls_enabled=tls_enabled)
    server_environment = environments["server"]["digest"]
    client_environment = environments["client"]["digest"]
    roles = {"server": {"environment_digest": server_environment, "build_digest": server_build},
             "client": {"environment_digest": client_environment, "build_digest": client_build}}
    common = {"policy_digest": policy["digest"], "logical_model_digest": policy["logical_model_digest"],
              "source_lock_digest": policy["source_lock_digest"], "operator_contract_digest": operator}
    runs = {}
    for name in RUNS:
        role = name.rsplit("_", 1)[1]
        native = name.startswith("native_")
        gpu = native and (role == "client" or executor == "cuda")
        body = {"schema_revision": 1, **common, **roles[role], "deadline_ms": 1000,
                "warmup": 0, "repeats": 2, "fixture_seed": 123}
        hard = {"host_bytes": "104857600", "device_bytes": "104857600" if gpu else "0",
                "pinned_bytes": "1048576" if gpu else "0", "max_rss_bytes": "209715200",
                "max_elapsed_ns": "1000000000", "max_startup_ns": "500000000", "max_case_latency_ns": "100000000"}
        if native:
            body.update(kind="native_fixture_request", profile=profile, executor=executor, role=role,
                runner_executor="cuda" if role == "client" else executor, expert_shape="target",
                gpu_uuid="GPU-12345678-1234-1234-1234-123456789abc" if gpu else None,
                host_budget=hard["host_bytes"], device_budget=hard["device_bytes"], pinned_budget=hard["pinned_bytes"],
                relative_floor=policy["thresholds"]["same_realization"]["relative_floor"])
        else:
            body.update(kind="transport_request", max_frame_bytes="262144", control_credit="131328", expert_credit="33160", row_credit="432", bulk_credit="1024")
        run = {"request_body": body, "hard_limits": hard, "binary_sha256": binary}
        if not native:
            run["bootstrap_body"] = {"schema_revision": 1, "role": "expert" if role == "server" else "client",
                **roles[role], "build_info_file": "/usr/share/dwarfstar/build-info.json",
                "network": {"control_address": "127.0.0.1:7443", "bulk_address": "127.0.0.1:7444",
                    **{key: 10 for key in ("connect_timeout_ms", "handshake_timeout_ms", "frame_io_timeout_ms", "write_timeout_ms", "operation_timeout_ms")}},
                "tls": {"ca_file": "/run/secrets/ca.pem", "certificate_file": "/run/secrets/peer.pem", "private_key_file": "/run/secrets/peer.key",
                    "expected_peer_name": "peer.test.internal", "authorized_peer_sha256": "f" * 64} if tls_enabled else {"enabled": False}}
        runs[name] = run
    return seal({"schema_revision": 1, "kind": "fixture_registration", "qualification_scope": "initial_fixture", **common,
        "profile": profile, "server_executor": executor, "realizations": roles, "runs": runs, "registered_at": policy["registered_at"]})


def raw_population(reg, environments=None):
    result = {}
    for name in RUNS:
        body, request = reg["runs"][name]["request_body"], derive_request(reg, name)
        if name.startswith("native_"):
            cases = []
            for key in expected_cases(body["role"], body["runner_executor"]):
                transfer = key == "pinned_multichunk_odd_bytes"
                expert = key in ("expert_dense_clamp_route", "expert_unit_palette", "expert_resident_dense_clamp_route", "expert_resident_unit_palette")
                pin = 12288 if transfer else 0
                shape = expected_dimensions(key, "target", pinned_bytes=pin)
                identity = digest({"classification": "synthetic offline fixture", "case": key})
                cases.append({"id": key, "component": "experts" if expert else "transfers" if transfer else "graph_state",
                    "path": "vram" if key.startswith("expert_resident_") else "host" if expert else "pinned" if transfer else "gpu",
                    **dict(zip(("rows", "input_features", "intermediate_features", "output_features", "elements"), shape, strict=True)),
                    "timing_group": TIMING_GROUPS.get(key, key), "warmup": expected_warmup(key, body["warmup"]),
                    "input_digest": identity, "oracle_digest": identity, "actual_digest": identity,
                    "max_abs_error": 0, "max_relative_error": 0, "max_rms_error": 0,
                    "requires_exact_match": key in EXACT_CASES, "exact_match": True,
                    "repeat_identical": True if key in REPEATED_CASES else None,
                    "placement_identical": True if key.startswith("expert_resident_") else None,
                    "owned_host_bytes": "1000", "device_bytes": "1000" if body["runner_executor"] == "cuda" else "0",
                    "pinned_bytes": str(pin), "startup_ns": "1", "latency_ns": ["10"] * expected_samples(key, body["repeats"])})
            raw = {"schema_revision": 1, "kind": "native_fixture_measurements", "qualified": False, "qualification_scope": "initial_fixture",
                "fixture_algorithm": "ria-native-fixtures-v1", "request_digest": digest(request),
                **{key: request[key] for key in ("profile", "executor", "role", "runner_executor", "gpu_uuid", "logical_model_digest", "source_lock_digest",
                    "environment_digest", "build_digest", "policy_digest", "operator_contract_digest", "preregistration_digest", "relative_floor", "warmup", "repeats")},
                "fixture_seed": str(request["fixture_seed"]), "owned_host_peak_bytes": "10000", "device_peak_bytes": "10000" if body["runner_executor"] == "cuda" else "0",
                "pinned_peak_bytes": "12288" if body["runner_executor"] == "cuda" else "0", "process_peak_rss_bytes": "1048576",
                "elapsed_ns": "50000000", "graph_startup_ns": "1", "host_accounting_scope": "owned_heap_metadata_and_pinned_pools; process_RSS_also_observed",
                "latency_scope": "synchronous_H2D_operator_D2H; selected_experts_include_all_three_projections",
                "deadline_enforcement": "monotonic_control_points_and_required_external_process_supervisor", "full_model_graph_parity": "unexecuted",
                "model_quality": "unexecuted", "graph_gpu_fixtures": "executed" if body["role"] == "client" else "unexecuted",
                "transfer_gpu_fixtures": "executed" if body["runner_executor"] == "cuda" else "unexecuted", "cases": cases}
        else:
            security = reg["runs"][name]["bootstrap_body"]["tls"]
            enabled = security.get("enabled", True)
            cases = [{"iteration": iteration, "id": key, "kind": kind, "request_bytes": "64", "reply_bytes": "64",
                "request_sha256": "5" * 64, "response_sha256": "6" * 64, "elapsed_ns": 100,
                "status": CASE_STATUSES[key]} for iteration in range(body["repeats"]) for key, kind in CASE_KINDS.items()]
            raw = {"schema_revision": 1, "kind": "native_transport_measurements", "qualified": False, "qualification_scope": "initial_fixture",
                **{key: request[key] for key in ("logical_model_digest", "source_lock_digest", "environment_digest", "build_digest", "policy_digest", "operator_contract_digest", "preregistration_digest")},
                "request_digest": request["digest"], "tls_enabled": enabled, "peer_certificate_digest": security.get("authorized_peer_sha256"),
                "role": "expert" if name.endswith("server") else "client",
                "checks": [{"id": key, "passed": True} for key in expected_checks(enabled)], "warmup_completed": 0, "iterations_completed": 2,
                "startup_ns": 1, "elapsed_ns": 50000000, "cpu_ns": 100, "max_rss_bytes": "1048576", "owned_buffers_peak_bytes": "1000",
                "measured_request_bytes": str(len(cases) * 64), "measured_response_bytes": str(len(cases) * 64), "timeout_elapsed_ns": 10000000, "cases": cases}
        raw = seal(raw)
        supervisor = seal({"schema_revision": 1, "kind": "fixture_supervision", "run": name, "registration_digest": reg["digest"],
            "request_digest": digest(request), "raw_digest": raw["digest"], "binary_sha256": reg["runs"][name]["binary_sha256"],
            "build_digest": body["build_digest"], "wall_ns": "60000000", "peak_rss_bytes": "2097152", "stdout_bytes": "1000", "stderr_bytes": "0",
            "rss_scope": "native wait4 complete child lifetime; descendants require their own cgroup cap", "exit_code": 0,
            "preflight": preflight(reg, name.rsplit("_", 1)[1], (environments or {}).get(name.rsplit("_", 1)[1]))})
        result[name] = raw, supervisor
    return result


def write_component_fixture(directory, policy, reg, *, role="expert", environments=None):
    directory = Path(directory)
    sources = raw_population(reg, environments)
    atomic_json(directory / "registration.json", reg)
    references = []
    for name, pair in sources.items():
        for suffix, document in zip(("measurements", "supervision"), pair, strict=True):
            relative = name + "-" + suffix + ".json"
            atomic_json(directory / relative, document)
            references.append({"path": relative, "digest": document["digest"]})
    components = []
    for name in ("experts", "graph_state", "transfers", "transport", "resource_bounds"):
        proof = _component(reg, sources, policy, name, role, references, "registration.json")
        atomic_json(directory / (name + ".json"), proof)
        components.append({"name": name, "evidence_path": name + ".json", "evidence_digest": proof["digest"], "passed": True})
    return components
