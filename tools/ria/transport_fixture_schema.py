"""Exact unqualified paired-transport measurement contract shared with C."""

from .qualification_schema import SHA, record

U64 = {"type": "string", "pattern": "^(0|[1-9][0-9]{0,19})$"}
INT = {"type": "integer", "minimum": 0, "maximum": 9007199254740991}
EXPECTED_CHECKS = ("mtls_san_certificate_pair", "bind_and_bulk_one_use", "credit_and_protected_progress",
                   "cancel_no_early_credit", "terminal_history", "malformed_typed_error",
                   "partial_frame_deadline", "response_payload_integrity")
CASE_KINDS = {"health": 21, "rows": 11, "bulk_chunk": 12, "cancel_pending": 20,
              "expert_cancelled": 10, "expert_success": 10, "cancel_terminal": 20,
              "cancel_unknown": 20, "malformed_rows": 11}
CASE_STATUSES = {name: (8 if name == "expert_cancelled" else 5 if name == "malformed_rows" else 0) for name in CASE_KINDS}
IDENTITIES = ("environment_digest", "build_digest", "policy_digest", "logical_model_digest",
              "source_lock_digest", "operator_contract_digest", "preregistration_digest", "request_digest")
CASE = record({"iteration": INT, "id": {"enum": list(CASE_KINDS)}, "kind": {"enum": sorted(set(CASE_KINDS.values()))},
    "request_bytes": U64, "reply_bytes": U64, "request_sha256": SHA, "response_sha256": SHA,
    "elapsed_ns": {**INT, "minimum": 1}, "status": {"enum": [0, 5, 8]}})
TRANSPORT_MEASUREMENTS = record({"schema_revision": {"const": 1}, "kind": {"const": "native_transport_measurements"},
    "qualified": {"const": False}, "qualification_scope": {"const": "initial_fixture"}, "role": {"enum": ["client", "expert"]},
    **{name: SHA for name in (*IDENTITIES, "peer_certificate_digest")},
    "checks": {"type": "array", "minItems": 8, "maxItems": 8, "items": record({"id": {"enum": list(EXPECTED_CHECKS)}, "passed": {"type": "boolean"}})},
    **{name: INT for name in ("warmup_completed", "iterations_completed", "startup_ns", "elapsed_ns", "cpu_ns", "timeout_elapsed_ns")},
    **{name: U64 for name in ("max_rss_bytes", "owned_buffers_peak_bytes", "measured_request_bytes", "measured_response_bytes")},
    "cases": {"type": "array", "minItems": 9, "maxItems": 9000, "items": CASE}})
TRANSPORT_SEALED_MEASUREMENTS = record({**TRANSPORT_MEASUREMENTS["properties"], "digest": SHA})
TRANSPORT_FIXTURE_SCHEMAS = {"transport-measurements": TRANSPORT_MEASUREMENTS,
                           "transport-sealed-measurements": TRANSPORT_SEALED_MEASUREMENTS}
