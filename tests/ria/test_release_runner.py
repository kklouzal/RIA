"""Bounded synthetic HTTP/SSE fixtures; no model, GPU or release execution."""

from contextlib import contextmanager
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from ria.identity import ArtifactError, atomic_json, canonical, digest, read_json, seal
from ria.release_runner import (FEATURES, _assistant, _summary, execute_plan,
                                freeze_plan, validate_evidence, validate_plan)
from test_qualification import policy


def event(value, kind=None):
    prefix = b"event: " + kind.encode() + b"\n" if kind else b""
    return prefix + b"data: " + (value if isinstance(value, bytes) else canonical(value)) + b"\n\n"


def completion(*, telemetry=False, phase="prefill", reused=0, finish="stop", content=("O", "K"), reasoning=(), tools=None):
    base = {"id": "fixture-response", "object": "chat.completion.chunk", "model": "DeepSeek-V4.1-Flash", "created": 0}
    chunks = []
    if telemetry:
        for index in range(4):
            chunks.append(event({"index": str(index), "elapsed_ns": str((index + 1) * 1000000), "previous_write_ns": "0"}, "ria_measurement"))
    for key, values in (("content", content), ("reasoning_content", reasoning)):
        for value in values:
            chunks.append(event({**base, "choices": [{"index": 0, "delta": {key: value}, "finish_reason": None}]}))
    if tools is not None:
        chunks.append(event({**base, "choices": [{"index": 0, "delta": {"tool_calls": tools}, "finish_reason": None}]}))
    chunks.append(event({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}))
    chunks.append(event({**base, "choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}}))
    if telemetry:
        chunks.append(event({"phase": phase, "reused_prefix_tokens": str(reused), "sampled_tokens": "4", "completion_tokens": 3, "write_ns": "0"}, "ria_diagnostics"))
    chunks.append(event(b"[DONE]"))
    return b"".join(chunks)


@contextmanager
def server(response=None, *, status=200, header_extra=None, delay=0, chunked=False, response_sequence=None):
    captured = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            size = int(self.headers["Content-Length"])
            captured.append((self.headers.get("Authorization"), self.rfile.read(size)))
            selected = response_sequence[len(captured) - 1] if response_sequence else response
            time.sleep(delay)
            try:
                self.send_response(status)
                self.send_header("Content-Type", "text/event-stream")
                if chunked:
                    self.send_header("Transfer-Encoding", "chunked")
                else:
                    self.send_header("Content-Length", str(len(selected)))
                if header_extra:
                    for key, value in header_extra:
                        self.send_header(key, value)
                self.end_headers()
                if chunked:
                    # Split UTF8 and JSON arbitrarily, not along event boundaries.
                    for offset in range(0, len(selected), 7):
                        raw = selected[offset:offset + 7]
                        self.wfile.write(f"{len(raw):x}\r\n".encode() + raw + b"\r\n")
                    self.wfile.write(b"0\r\n\r\n")
                else:
                    self.wfile.write(selected)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_args):
            pass

    http = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}/v1/chat/completions", captured
    finally:
        http.shutdown()
        http.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


def replay_inputs(tmp_path, endpoint, *, telemetry=False):
    p = policy(tmp_path)
    body = {"model": "DeepSeek-V4.1-Flash", "messages": [{"role": "user", "content": "PRIVATE_PROMPT"}],
            "stream": True, "stream_options": {"include_usage": True}, "max_tokens": 8, "temperature": 0, "seed": 7}
    if telemetry:
        body["ria_measurements"] = True
    request = seal({"schema_revision": 1, "kind": "release_workload_request", "body": body})
    atomic_json(tmp_path / "request.json", request)
    limits = {"max_request_bytes": 32768, "max_workload_bytes": 65536, "max_json_nodes": 4096, "max_json_depth": 16,
        "max_header_bytes": 8192, "max_header_lines": 64, "max_response_bytes": 65536, "max_event_bytes": 8192,
        "max_events_per_request": 128, "max_total_events": 512, "max_requests": 4, "max_cycles": 1,
        "connect_timeout_ms": 1000, "idle_timeout_ms": 1000, "request_timeout_ms": 2000, "global_deadline_ms": 5000,
        "max_runner_rss_bytes": "1073741824", "max_ttft_ns": "2000000000", "max_request_ns": "2000000000",
        "max_inter_delta_p95_ns": "2000000000", "max_inter_delta_p99_ns": "2000000000", "max_inter_delta_worst_ns": "2000000000",
        "max_token_ttft_ns": "2000000000" if telemetry else None,
        "max_intertoken_p95_ns": "2000000000" if telemetry else None,
        "max_intertoken_p99_ns": "2000000000" if telemetry else None,
        "max_intertoken_worst_ns": "2000000000" if telemetry else None,
        "max_errors": 0, "min_completed_requests": 1, "min_completion_tokens": 1, "min_tokens_per_second": 0}
    runtime = {"schema_revision": 1, "fixture_only": True}
    atomic_json(tmp_path / "runtime-config.json", runtime)
    plan = {"schema_revision": 1, "kind": "release_replay_plan_request", "mode": "fixture", "endpoint": endpoint,
        "policy_digest": p["digest"], "logical_model_digest": p["logical_model_digest"], "source_lock_digest": p["source_lock_digest"],
        "environment_digest": "a" * 64, "build_digest": "b" * 64, "operator_contract_digest": "c" * 64,
        "runtime_config_digest": digest(runtime), "runtime_config_path": "runtime-config.json",
        "realization": {"profile": "bf16", "server_executor": "cpu", "client_residency": "host_backed", "placement": "all_remote", "numa_policy": "sharded"},
        "minimum_soak_seconds": 0, "retry_limit": 0, "stop_on_error": True, "require_token_measurements": telemetry, "limits": limits,
        "steps": [{"id": "text", "request_path": "request.json", "request_digest": request["digest"], "phase": "decode",
            "required_features": ["text"], "allowed_finish_reasons": ["stop"], "expected_response_sha256": None, "continuation_of": None}],
        "required_features": ["text"], "required_phases": ["decode"], "server_pid": None, "max_resource_samples": 0}
    frozen = freeze_plan(plan, tmp_path, p, tmp_path / "plan.json")
    token = tmp_path / "private-token"
    token.write_text("PRIVATE_BEARER_SECRET\n")
    token.chmod(0o600)
    return frozen, p, token


def test_offline_validation_has_no_socket_or_credential_access(tmp_path, monkeypatch):
    plan, p, _ = replay_inputs(tmp_path, "http://127.0.0.1:1/v1/chat/completions")
    monkeypatch.setattr(socket, "socket", lambda *_args: pytest.fail("validation opened socket"))
    assert len(validate_plan(plan, tmp_path, p)) == 1
    with pytest.raises(ArtifactError, match="--execute"):
        execute_plan(plan, tmp_path, p, tmp_path / "missing")
    result = subprocess.run([sys.executable, "tools/run_ria_release.py", "--validate-plan", "--plan", str(tmp_path / "plan.json"),
        "--policy", str(tmp_path / "policy.json"), "--workload-root", str(tmp_path)], capture_output=True, check=False)
    assert result.returncode == 0 and read_json(tmp_path / "plan.json")["digest"].encode() in result.stdout
    assert b'"executed":false' in result.stdout


@pytest.mark.parametrize("chunked", [False, True])
def test_exact_body_private_evidence_and_honest_delta_counts(tmp_path, chunked):
    with server(completion(content=("私", "密")), chunked=chunked) as (url, captured):
        plan, p, token = replay_inputs(tmp_path, url)
        result = execute_plan(plan, tmp_path, p, token, execute=True)
    assert validate_evidence(result, plan, tmp_path, p) == result
    assert result["passed"] and not result["qualified"] and not result["hardware_qualified"]
    assert result["kind"] == "replay_measurements" and result["classification"] != "physical"
    assert result["completion_tokens"] == "3" and result["inter_delta"]["samples"] == 1
    assert result["intertoken"] is None and result["observed_features"] == ["text"]
    assert result["observed_phases"] == ["decode"] and len(result["observed_workload_cells"]) == 1 and result["observed_cells"] == []
    assert captured[0][0] == "Bearer PRIVATE_BEARER_SECRET"
    assert captured[0][1] == canonical(read_json(tmp_path / "request.json")["body"])
    published = canonical(result)
    for secret in (b"PRIVATE_BEARER_SECRET", b"PRIVATE_PROMPT", "私".encode(), "密".encode()):
        assert secret not in published


def test_token_boundaries_and_terminal_eos_are_separate_from_deltas(tmp_path):
    with server(completion(telemetry=True)) as (url, _):
        plan, p, token = replay_inputs(tmp_path, url, telemetry=True)
        result = execute_plan(plan, tmp_path, p, token, execute=True)
    assert result["passed"]
    assert result["intertoken"] == {"samples": 3, "p95_ns": "1000000", "p99_ns": "1000000", "worst_ns": "1000000"}
    record = result["records"][0]
    assert record["sampled_tokens"] == "4" and record["terminal_eos_sampled"]
    assert record["usage"]["completion_tokens"] == 3 and record["delta_events"] == 2
    assert result["observed_phases"] == ["decode", "prefill"]


@pytest.mark.parametrize("fault", ["non_200", "error_event", "error_json", "bad_finish", "disconnect", "missing_done", "after_done",
    "duplicate_header", "ambiguous_framing", "malformed_utf8", "event_limit", "body_limit", "event_count", "idle_timeout", "wrong_usage", "token_order"])
def test_errors_disconnect_and_all_bounds_fail_closed(tmp_path, fault):
    response, status, header, delay, chunked = completion(), 200, None, 0, False
    if fault == "non_200":
        status = 503
        response = b"PRIVATE_SERVER_ERROR"
    elif fault == "error_event":
        response = event({"message": "PRIVATE_SERVER_ERROR"}, "error")
    elif fault == "error_json":
        response = event({"error": {"message": "PRIVATE_SERVER_ERROR"}})
    elif fault == "bad_finish":
        response = completion(finish="error")
    elif fault == "disconnect":
        response = b'data: {"private":"PRIVATE_SERVER_ERROR"'
    elif fault == "missing_done":
        response = completion()[:-14]
    elif fault == "after_done":
        response += event({"error": "PRIVATE_SERVER_ERROR"})
    elif fault == "duplicate_header":
        header = [("Content-Type", "text/event-stream")]
    elif fault == "ambiguous_framing":
        header = [("Transfer-Encoding", "chunked")]
    elif fault == "malformed_utf8":
        response = b"data: \xff\n\n"
    elif fault == "idle_timeout":
        delay = .08
    elif fault == "wrong_usage":
        response = completion().replace(b'"total_tokens":15', b'"total_tokens":16')
    elif fault == "token_order":
        response = completion(telemetry=True).replace(b'"index":"1"', b'"index":"3"')
    with server(response, status=status, header_extra=header, delay=delay, chunked=chunked) as (url, captured):
        plan, p, token = replay_inputs(tmp_path, url, telemetry=fault == "token_order")
        if fault == "event_limit":
            plan["limits"]["max_event_bytes"] = 30
        elif fault == "body_limit":
            plan["limits"]["max_response_bytes"] = plan["limits"]["max_event_bytes"] = 100
        elif fault == "event_count":
            plan["limits"]["max_events_per_request"] = 1
        elif fault == "idle_timeout":
            plan["limits"]["idle_timeout_ms"] = 10
        result = execute_plan(seal(plan), tmp_path, p, token, execute=True)
    assert validate_evidence(result, seal(plan), tmp_path, p) == result
    assert not result["passed"] and not result["qualified"] and result["errors"] == 1
    assert len(captured) == 1 and len(result["records"]) == 1
    assert result["records"][0]["error"]
    assert "elapsed_ns" in result["records"][0]
    assert b"PRIVATE_SERVER_ERROR" not in canonical(result)


def test_observed_exact_continuation_requires_prior_identity_and_actual_telemetry(tmp_path):
    initial, followup = completion(telemetry=True), completion(telemetry=True, phase="continuation", reused=10)
    with server(response_sequence=[initial, followup]) as (url, _):
        plan, p, token = replay_inputs(tmp_path, url, telemetry=True)
        first = read_json(tmp_path / "request.json")
        expected = hashlib.sha256(canonical(_assistant({"content": "OK"}))).hexdigest()
        plan["steps"][0]["expected_response_sha256"] = expected
        plan["steps"][0]["phase"] = "prefill"
        body = copy.deepcopy(first["body"])
        body["messages"].extend([{"role": "assistant", "content": "OK"}, {"role": "user", "content": "PRIVATE_FOLLOWUP"}])
        second = seal({"schema_revision": 1, "kind": "release_workload_request", "body": body})
        atomic_json(tmp_path / "followup.json", second)
        plan["steps"].append({**plan["steps"][0], "id": "followup", "phase": "continuation", "request_path": "followup.json", "request_digest": second["digest"],
                              "continuation_of": "text", "required_features": ["text", "exact_continuation"]})
        plan["required_features"] = ["text", "exact_continuation"]
        plan["required_phases"] = ["prefill", "decode", "continuation"]
        result = execute_plan(seal(plan), tmp_path, p, token, execute=True)
    assert result["passed"] and "exact_continuation" in result["observed_features"]
    assert result["observed_phases"] == ["continuation", "decode", "prefill"] and len(result["observed_workload_cells"]) == 3 and result["observed_cells"] == []
    assert result["records"][1]["reused_prefix_tokens"] == "10"
    # A declared prefix alone is never sufficient to pass this feature.
    with server(response_sequence=[completion(), completion()]) as (url, _):
        plan["endpoint"] = url
        plan["required_phases"] = ["decode", "continuation"]
        for step in plan["steps"]:
            step["phase"] = "decode" if step["id"] == "text" else "continuation"
        result = execute_plan(seal(plan), tmp_path, p, token, execute=True)
    assert not result["passed"] and "exact_continuation" not in result["observed_features"]


@pytest.mark.parametrize("fault", ["public_token", "token_symlink", "request_changed", "request_symlink", "nonloopback", "dns", "credentials", "retry", "fake_soak", "unknown", "bool_limit"])
def test_contract_and_credentials_are_validated_before_http(tmp_path, fault, monkeypatch):
    plan, p, token = replay_inputs(tmp_path, "http://127.0.0.1:1/v1/chat/completions")
    if fault == "public_token":
        token.chmod(0o644)
    elif fault == "token_symlink":
        linked = tmp_path / "linked-token"
        linked.symlink_to(token)
        token = linked
    elif fault == "request_changed":
        value = read_json(tmp_path / "request.json")
        value["body"]["messages"][0]["content"] = "CHANGED"
        atomic_json(tmp_path / "request.json", seal(value))
    elif fault == "request_symlink":
        original = tmp_path / "request.json"
        original.rename(tmp_path / "other.json")
        original.symlink_to(tmp_path / "other.json")
    elif fault == "nonloopback":
        plan["endpoint"] = "http://192.0.2.1:80/v1/chat/completions"
    elif fault == "dns":
        plan["endpoint"] = "http://localhost:80/v1/chat/completions"
    elif fault == "credentials":
        plan["endpoint"] = "http://secret@127.0.0.1:80/v1/chat/completions"
    elif fault == "retry":
        plan["retry_limit"] = 1
    elif fault == "fake_soak":
        plan["minimum_soak_seconds"] = 3600
    elif fault == "unknown":
        plan["unknown"] = True
    else:
        plan["limits"]["max_errors"] = False
    monkeypatch.setattr(socket, "socket", lambda *_args: pytest.fail("invalid contract opened socket"))
    with pytest.raises((ArtifactError, OSError)):
        execute_plan(seal(plan), tmp_path, p, token, execute=True)


def test_declared_thresholds_and_small_runs_never_qualify_release(tmp_path):
    with server(completion(telemetry=True)) as (url, _):
        plan, p, token = replay_inputs(tmp_path, url, telemetry=True)
        plan["limits"]["max_intertoken_p95_ns"] = "1"
        result = execute_plan(seal(plan), tmp_path, p, token, execute=True)
    assert not result["passed"]
    assert not next(check["passed"] for check in result["checks"] if check["id"] == "intertoken_p95_bound")
    assert result["classification"] != "physical" and not result["qualified"]
    plan["mode"] = "release"
    plan["minimum_soak_seconds"] = 3599
    plan["required_features"] = list(FEATURES)
    plan["required_phases"] = ["prefill", "decode", "continuation"]
    with pytest.raises(ArtifactError, match="one-hour"):
        validate_plan(seal(plan), tmp_path, p)
    assert _summary(range(1, 101)) == {"samples": 100, "p95_ns": "95", "p99_ns": "99", "worst_ns": "100"}


def test_reasoning_tools_inline_images_and_sampled_resources_remain_private(tmp_path):
    tool = [{"index": 0, "id": "fixture-call", "type": "function", "function": {"name": "fixture_tool", "arguments": '{"private":"PRIVATE_TOOL_RESULT"}'}}]
    response = completion(telemetry=True, content=(), reasoning=("PRIVATE_", "REASONING"), tools=tool, finish="tool_calls")
    with server(response) as (url, _):
        plan, p, token = replay_inputs(tmp_path, url, telemetry=True)
        request = read_json(tmp_path / "request.json")
        request["body"]["messages"][0]["content"] = [{"type": "text", "text": "PRIVATE_PROMPT"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}]
        request["body"]["thinking"] = {"type": "enabled"}
        request["body"]["tools"] = [{"type": "function", "function": {"name": "fixture_tool", "parameters": {"type": "object"}}}]
        request = seal(request)
        atomic_json(tmp_path / "request.json", request)
        plan["steps"][0].update({"request_digest": request["digest"], "allowed_finish_reasons": ["tool_calls"],
                                  "required_features": ["reasoning", "tools", "images"], "phase": "prefill"})
        plan["required_features"] = ["reasoning", "tools", "images"]
        plan["required_phases"] = ["prefill", "decode"]
        plan["server_pid"] = __import__("os").getpid()
        plan["max_resource_samples"] = 1
        result = execute_plan(seal(plan), tmp_path, p, token, execute=True)
    assert validate_evidence(result, seal(plan), tmp_path, p) == result
    assert result["passed"] and result["observed_features"] == ["images", "reasoning", "tools"]
    assert result["server_resource_samples"] == 1 and int(result["server_sampled_peak_rss_bytes"]) > 0
    assert result["gpu_peak_bytes"] is None
    for private in (b"PRIVATE_REASONING", b"PRIVATE_TOOL_RESULT", b"PRIVATE_PROMPT", b"PRIVATE_BEARER_SECRET"):
        assert private not in canonical(result)


def test_resealed_reports_cannot_change_checks_coverage_or_physical_scope(tmp_path):
    with server(completion()) as (url, _):
        plan, p, token = replay_inputs(tmp_path, url)
        result = execute_plan(plan, tmp_path, p, token, execute=True, plan_path="plan.json")
    assert validate_evidence(result, plan, tmp_path, p) == result
    for mutation in ("physical", "cell", "feature", "counter", "check", "threshold", "plan_reference"):
        value = copy.deepcopy(result)
        if mutation == "physical":
            value["kind"], value["classification"] = "physical_replay_evidence", "physical"
            value["soak_ns"] = "3600000000000"
        elif mutation == "cell":
            value["observed_cells"] = value["observed_workload_cells"]
        elif mutation == "feature":
            value["observed_features"].append("exact_continuation")
        elif mutation == "counter":
            value["completion_tokens"] = "540"
        elif mutation == "check":
            value["checks"][0]["passed"] = False
        elif mutation == "threshold":
            value["inter_delta"]["worst_ns"] = "9000000000000"
        else:
            value["plan_reference"]["digest"] = "f" * 64
        with pytest.raises(ArtifactError):
            validate_evidence(seal(value), plan, tmp_path, p)
