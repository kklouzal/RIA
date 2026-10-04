"""Preregister exact loopback HTTP workloads and measure bounded sequential SSE.

This runner never promotes HTTP success to source fidelity or full release
qualification. Response text, reasoning, tools and credentials remain private;
only their identities and bounded counters/timings enter published evidence.
"""

from array import array
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import ipaddress
import math
import os
import re
import resource
import socket
import stat
import sys
import time
from urllib.parse import urlsplit

from .identity import (ArtifactError, atomic_json, canonical, check_json, digest, loads,
                       open_regular, seal, u64, verify_identity, within)
from .qualification_schema import MATRIX_AXES, SHA, record

FEATURES = ("text", "reasoning", "tools", "images", "exact_continuation")
IDENTITIES = ("policy_digest", "logical_model_digest", "source_lock_digest",
              "environment_digest", "build_digest", "operator_contract_digest",
              "runtime_config_digest")
U64 = {"type": "string", "pattern": "^(0|[1-9][0-9]*)$", "maxLength": 20}
POS = {"type": "integer", "minimum": 1, "maximum": 9007199254740991}
NAME = {"type": "string", "pattern": "^[A-Za-z0-9_-]{1,64}$"}
TEXT = {"type": "string", "minLength": 1, "maxLength": 4096}
LIMITS = record({
    **{key: POS for key in ("max_request_bytes", "max_workload_bytes", "max_json_nodes",
        "max_json_depth", "max_header_bytes", "max_header_lines", "max_response_bytes",
        "max_event_bytes", "max_events_per_request", "max_total_events", "max_requests",
        "max_cycles", "connect_timeout_ms", "idle_timeout_ms", "request_timeout_ms",
        "global_deadline_ms")},
    **{key: U64 for key in ("max_runner_rss_bytes", "max_ttft_ns", "max_request_ns",
        "max_inter_delta_p95_ns", "max_inter_delta_p99_ns", "max_inter_delta_worst_ns")},
    **{key: {"anyOf": [U64, {"type": "null"}]} for key in ("max_token_ttft_ns",
        "max_intertoken_p95_ns", "max_intertoken_p99_ns", "max_intertoken_worst_ns")},
    "max_errors": {"type": "integer", "minimum": 0, "maximum": 65536},
    "min_completed_requests": POS, "min_completion_tokens": POS,
    "min_tokens_per_second": {"type": "number", "minimum": 0},
})
STEP = record({"id": NAME, "request_path": TEXT, "request_digest": SHA,
    "phase": {"enum": ["prefill", "decode", "continuation"]},
    "required_features": {"type": "array", "items": {"enum": list(FEATURES)},
        "minItems": 1, "maxItems": 5, "uniqueItems": True},
    "allowed_finish_reasons": {"type": "array", "items": {"enum": ["stop", "length", "tool_calls"]},
        "minItems": 1, "maxItems": 3, "uniqueItems": True},
    "expected_response_sha256": {"anyOf": [SHA, {"type": "null"}]},
    "continuation_of": {"anyOf": [NAME, {"type": "null"}]}})
PLAN = record({"schema_revision": {"const": 1}, "kind": {"const": "release_replay_plan"},
    "mode": {"enum": ["fixture", "release"]}, "endpoint": TEXT,
    **{key: SHA for key in IDENTITIES},
    "runtime_config_path": TEXT,
    "realization": record({key: {"enum": values} for key, values in MATRIX_AXES.items() if key != "phase"}),
    "minimum_soak_seconds": {"type": "integer", "minimum": 0, "maximum": 604800},
    "retry_limit": {"const": 0}, "stop_on_error": {"type": "boolean"},
    "require_token_measurements": {"type": "boolean"},
    "limits": LIMITS, "steps": {"type": "array", "items": STEP, "minItems": 1, "maxItems": 64},
    "required_features": {"type": "array", "items": {"enum": list(FEATURES)},
        "minItems": 1, "maxItems": 5, "uniqueItems": True},
    "required_phases": {"type": "array", "items": {"enum": ["prefill", "decode", "continuation"]},
        "minItems": 1, "maxItems": 3, "uniqueItems": True},
    "server_pid": {"anyOf": [{**POS, "maximum": 2147483647}, {"type": "null"}]},
    "max_resource_samples": {"type": "integer", "minimum": 0, "maximum": 65536},
    "registered_at": {"type": "string", "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"},
    "digest": SHA})
WORKLOAD = record({"schema_revision": {"const": 1}, "kind": {"const": "release_workload_request"},
                   "body": {"type": "object"}, "digest": SHA})
PLAN_REQUEST = deepcopy(PLAN)
PLAN_REQUEST["properties"]["kind"] = {"const": "release_replay_plan_request"}
for _field in ("registered_at", "digest"):
    del PLAN_REQUEST["properties"][_field]
    PLAN_REQUEST["required"].remove(_field)
RELEASE_RUN_SCHEMAS = {"release-replay-plan": PLAN, "release-replay-plan-request": PLAN_REQUEST,
                       "release-workload-request": WORKLOAD}


class RunFailure(ArtifactError):
    """A static public error code; never includes response or credential data."""

    def __init__(self, code):
        super().__init__(code)
        self.partial = None


def _schema(schema, value):
    from .schemas import StrictValidator
    check_json(value, max_nodes=20000, max_depth=32)
    if next(StrictValidator(schema).iter_errors(value), None):
        raise ArtifactError("invalid replay contract")


def _endpoint(value):
    if any(ord(character) < 33 or ord(character) > 126 for character in value):
        raise ArtifactError("endpoint must contain literal printable ASCII only")
    parsed = urlsplit(value)
    if parsed.scheme != "http" or parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
        raise ArtifactError("endpoint must be literal loopback HTTP")
    try:
        address = ipaddress.ip_address(parsed.hostname or "")
        port = parsed.port
    except ValueError as exc:
        raise ArtifactError("endpoint must be literal loopback HTTP") from exc
    if not address.is_loopback or getattr(address, "scope_id", None) is not None or port is None or not 1 <= port <= 65535 or parsed.path != "/v1/chat/completions":
        raise ArtifactError("endpoint must be literal loopback chat completions")
    return address, port, parsed.path


def _body(body, limits):
    if not isinstance(body, dict) or body.get("model") != "DeepSeek-V4.1-Flash" or body.get("stream") is not True:
        raise ArtifactError("workload requires the source model and streaming")
    if body.get("stream_options") != {"include_usage": True}:
        raise ArtifactError("workload requires final token usage")
    fields = {"model", "messages", "n", "stream", "stream_options", "max_tokens", "max_completion_tokens",
              "temperature", "top_p", "top_k", "min_p", "seed", "stop", "reasoning_effort", "thinking",
              "tools", "tool_choice", "ria_measurements"}
    if set(body) - fields or ("n" in body and (type(body["n"]) is not int or body["n"] != 1)):
        raise ArtifactError("unknown workload request field")
    bounds = [body[key] for key in ("max_tokens", "max_completion_tokens") if key in body]
    if len(bounds) != 1 or type(bounds[0]) is not int or not 1 <= bounds[0] <= 1048576:
        raise ArtifactError("workload needs one finite token limit")
    messages = body.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 4096:
        raise ArtifactError("workload needs bounded exact messages")
    features = set()
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in ("system", "developer", "user", "assistant", "tool"):
            raise ArtifactError("invalid workload message")
        content = message.get("content")
        if isinstance(content, str):
            if content:
                features.add("text")
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    raise ArtifactError("invalid workload content")
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    if part["text"]:
                        features.add("text")
                elif part.get("type") == "image_url":
                    image = part.get("image_url")
                    url = image.get("url") if isinstance(image, dict) else None
                    if not isinstance(url, str) or not re.fullmatch(r"data:image/(?:png|jpeg);base64,[A-Za-z0-9+/]+={0,2}", url):
                        raise ArtifactError("image workload must contain inline data only")
                    features.add("images")
                else:
                    raise ArtifactError("invalid workload content")
        elif content is not None or message["role"] not in ("assistant", "tool"):
            raise ArtifactError("invalid workload content")
    thinking = body.get("thinking", {})
    if (isinstance(thinking, dict) and thinking.get("type") == "enabled") or body.get("reasoning_effort") in ("low", "medium", "high", "max"):
        features.add("reasoning")
    if isinstance(body.get("tools"), list) and body["tools"]:
        features.add("tools")
    if "ria_measurements" in body and type(body["ria_measurements"]) is not bool:
        raise ArtifactError("invalid timing opt-in")
    raw = canonical(body)
    if len(raw) > limits["max_request_bytes"]:
        raise ArtifactError("workload request byte limit")
    return raw, features


def cell_id(realization, phase):
    cell = {**realization, "phase": phase}
    return "/".join(cell[key] for key in MATRIX_AXES)


def validate_plan(document, root, policy, *, frozen=True):
    """Read immutable workload files only; no socket or credential access."""
    from .qualification import policy_validate
    _schema(PLAN if frozen else PLAN_REQUEST, document)
    policy_validate(policy)
    if any(document[key] != policy[key] for key in ("logical_model_digest", "source_lock_digest")) or document["policy_digest"] != policy["digest"]:
        raise ArtifactError("replay differs from frozen source policy")
    if frozen:
        verify_identity(document)
        try:
            datetime.strptime(document["registered_at"], "%Y-%m-%dT%H:%M:%SZ")
        except ValueError as exc:
            raise ArtifactError("invalid replay registration time") from exc
    _endpoint(document["endpoint"])
    from .identity import read_json
    runtime_config = read_json(within(root, document["runtime_config_path"]), max_bytes=256 << 10,
                               max_nodes=20000, max_depth=32)
    if not isinstance(runtime_config, dict) or digest(runtime_config) != document["runtime_config_digest"]:
        raise ArtifactError("runtime configuration differs from the preregistered identity")
    if "digest" in runtime_config:
        verify_identity(runtime_config)
    limits = document["limits"]
    # Static parser/retention ceilings supplement, never replace, declared caps.
    ceilings = {"max_request_bytes": 64 << 20, "max_workload_bytes": 128 << 20,
        "max_json_nodes": 2000000, "max_json_depth": 64, "max_header_bytes": 1 << 20,
        "max_header_lines": 4096, "max_response_bytes": 128 << 20, "max_event_bytes": 16 << 20,
        "max_events_per_request": 1048576, "max_total_events": 4194304,
        "max_requests": 65536, "max_cycles": 65536, "global_deadline_ms": 604800000}
    if any(limits[key] > cap for key, cap in ceilings.items()) or any(u64(limits[key]) == 0 for key in LIMITS["properties"] if key.startswith("max_") and isinstance(limits[key], str)):
        raise ArtifactError("replay bounds exceed supported parser limits")
    for key in ("max_token_ttft_ns", "max_intertoken_p95_ns", "max_intertoken_p99_ns", "max_intertoken_worst_ns"):
        if document["require_token_measurements"] != (limits[key] is not None):
            raise ArtifactError("token timing limits must match the declared observer contract")
    for key in ("connect_timeout_ms", "idle_timeout_ms", "request_timeout_ms"):
        if limits[key] > limits["global_deadline_ms"]:
            raise ArtifactError("request timeout exceeds global deadline")
    if limits["max_event_bytes"] > limits["max_response_bytes"] or limits["max_events_per_request"] > limits["max_total_events"]:
        raise ArtifactError("inconsistent replay event bounds")
    if limits["min_completed_requests"] > limits["max_requests"] or limits["max_requests"] < len(document["steps"]):
        raise ArtifactError("inconsistent replay request bounds")
    if document["server_pid"] is None and document["max_resource_samples"] != 0:
        raise ArtifactError("resource sampling needs an explicit server PID")
    if document["mode"] == "release":
        if document["minimum_soak_seconds"] < policy["minimum_soak_seconds"] or document["minimum_soak_seconds"] * 1000 >= limits["global_deadline_ms"]:
            raise ArtifactError("physical release requires the declared one-hour soak and deadline")
        if set(document["required_features"]) != set(FEATURES) or set(document["required_phases"]) != {"prefill", "decode", "continuation"}:
            raise ArtifactError("release must declare every source feature and phase")
        if not document["require_token_measurements"]:
            raise ArtifactError("release workload requires exact token observation")
    elif document["minimum_soak_seconds"] != 0:
        raise ArtifactError("short fixtures must not claim a soak")
    loaded, total_bytes, node_budget, ids, possible = [], 0, 0, {}, set()
    for step in document["steps"]:
        if step["id"] in ids:
            raise ArtifactError("duplicate replay step identity")
        path = within(root, step["request_path"])
        with open_regular(path) as stream:
            size = os.fstat(stream.fileno()).st_size
            if size > limits["max_request_bytes"] + 4096 or size > limits["max_workload_bytes"] - total_bytes:
                raise ArtifactError("workload population byte limit")
            total_bytes += size
            raw = stream.read(size + 1)
            if len(raw) != size:
                raise ArtifactError("workload changed during validation")
        value = loads(raw, max_bytes=limits["max_request_bytes"] + 4096,
                      max_nodes=limits["max_json_nodes"], max_depth=limits["max_json_depth"])
        if not isinstance(value, dict):
            raise ArtifactError("invalid workload wrapper")
        _schema(WORKLOAD, {**value, "body": {}})
        verify_identity(value, step["request_digest"])
        # Bound aggregate parsing complexity across all retained request bodies.
        todo = [value]
        while todo:
            item = todo.pop()
            node_budget += 1
            if node_budget > limits["max_json_nodes"]:
                raise ArtifactError("workload population node limit")
            if isinstance(item, dict):
                todo.extend(item.keys())
                todo.extend(item.values())
            elif isinstance(item, list):
                todo.extend(item)
        body = value["body"]
        payload, features = _body(body, limits)
        if document["require_token_measurements"] and body.get("ria_measurements") is not True:
            raise ArtifactError("workload omitted the declared token observer")
        previous = step["continuation_of"]
        if previous is not None:
            if not loaded or previous != loaded[-1][0]["id"] or step["phase"] != "continuation":
                raise ArtifactError("continuation must follow its exact prior request")
            prior_step, prior_body, _, _ = loaded[-1]
            prefix = prior_body["messages"]
            if prior_step["expected_response_sha256"] is None or len(body["messages"]) <= len(prefix) or body["messages"][:len(prefix)] != prefix:
                raise ArtifactError("continuation needs an authenticated exact message prefix")
            assistant = body["messages"][len(prefix)]
            if assistant.get("role") != "assistant" or hashlib.sha256(canonical(_assistant(assistant))).hexdigest() != prior_step["expected_response_sha256"]:
                raise ArtifactError("continuation assistant differs from the frozen expected response")
            features.add("exact_continuation")
        elif "exact_continuation" in step["required_features"]:
            raise ArtifactError("continuation feature requires its prior request")
        if not set(step["required_features"]) <= features:
            raise ArtifactError("declared feature absent from the exact request")
        possible.update(step["required_features"])
        ids[step["id"]] = True
        loaded.append((step, body, payload, features))
    if not set(document["required_features"]) <= possible or not set(document["required_phases"]) <= {"decode", *(step[0]["phase"] for step in loaded)}:
        raise ArtifactError("declared replay coverage absent from workload")
    return loaded


def freeze_plan(document, root, policy, output):
    validate_plan(document, root, policy, frozen=False)
    result = seal({**document, "kind": "release_replay_plan",
                   "registered_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    atomic_json(output, result)
    return result


def _assistant(message):
    return {"role": "assistant", "content": message.get("content") or "",
            "reasoning_content": message.get("reasoning_content") or "",
            "tool_calls": message.get("tool_calls") or []}


def _credential(path):
    with open_regular(path) as stream:
        info = os.fstat(stream.fileno())
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) not in (0o400, 0o600) or info.st_nlink != 1:
            raise ArtifactError("bearer token file must be private, owned and unlinked from other names")
        value = stream.read(4097)
    if value.endswith(b"\n"):
        value = value[:-1]
    if not 1 <= len(value) <= 4096 or any(byte < 33 or byte > 126 for byte in value):
        raise ArtifactError("invalid private bearer token file")
    return value


class _Reader:
    def __init__(self, sock, limits, deadline, global_deadline):
        self.sock, self.limits = sock, limits
        self.deadline = min(deadline, global_deadline)
        self.buffer = bytearray()
        self.wire_bytes = 0
        self.max_buffer_bytes = 0

    def receive(self):
        remaining = self.deadline - time.monotonic_ns()
        if remaining <= 0:
            raise RunFailure("request_deadline")
        self.sock.settimeout(min(remaining / 1e9, self.limits["idle_timeout_ms"] / 1000))
        try:
            raw = self.sock.recv(65536)
        except socket.timeout as exc:
            raise RunFailure("io_timeout") from exc
        except OSError as exc:
            raise RunFailure("io_error") from exc
        self.wire_bytes += len(raw)
        if self.wire_bytes > self.limits["max_response_bytes"] + self.limits["max_header_bytes"] + self.limits["max_events_per_request"] * 128:
            raise RunFailure("response_wire_limit")
        return raw

    def line(self, cap):
        while True:
            end = self.buffer.find(b"\n")
            if end >= 0:
                if end + 1 > cap:
                    raise RunFailure("line_limit")
                raw = bytes(self.buffer[:end + 1])
                del self.buffer[:end + 1]
                return raw
            if len(self.buffer) >= cap:
                raise RunFailure("line_limit")
            raw = self.receive()
            if not raw:
                raise RunFailure("truncated_http")
            self.buffer.extend(raw)
            self.max_buffer_bytes = max(self.max_buffer_bytes, len(self.buffer))

    def take(self, amount):
        while amount:
            if self.buffer:
                length = min(amount, len(self.buffer), 65536)
                raw = bytes(self.buffer[:length])
                del self.buffer[:length]
            else:
                raw = self.receive()
                if not raw:
                    raise RunFailure("truncated_http")
                if len(raw) > amount:
                    self.buffer.extend(raw[amount:])
                    raw = raw[:amount]
            amount -= len(raw)
            yield raw

    def body(self):
        status = self.line(self.limits["max_header_bytes"])
        if not re.fullmatch(rb"HTTP/1\.[01] [0-9]{3} [\x20-\x7e]*\r\n", status):
            raise RunFailure("invalid_http_status")
        status_code = int(status[9:12])
        headers, count, header_bytes = {}, 0, len(status)
        while True:
            line = self.line(self.limits["max_header_bytes"])
            header_bytes += len(line)
            count += 1
            if header_bytes > self.limits["max_header_bytes"] or count > self.limits["max_header_lines"]:
                raise RunFailure("header_limit")
            if line == b"\r\n":
                break
            match = re.fullmatch(rb"([!#$%&'*+.^_`|~0-9A-Za-z-]+):[ \t]*([^\r\n]*)\r\n", line)
            if not match or any(byte < 32 and byte != 9 for byte in match[2]) or 127 in match[2]:
                raise RunFailure("invalid_http_header")
            key, value = match[1].lower(), match[2].strip()
            if key in headers:
                raise RunFailure("duplicate_http_header")
            headers[key] = value
        if status_code != 200:
            raise RunFailure("http_non_200")
        if headers.get(b"content-type", b"").split(b";", 1)[0].strip().lower() != b"text/event-stream":
            raise RunFailure("invalid_content_type")
        if b"content-encoding" in headers and headers[b"content-encoding"].lower() != b"identity":
            raise RunFailure("encoded_response")
        length, transfer = headers.get(b"content-length"), headers.get(b"transfer-encoding")
        if length is not None and transfer is not None:
            raise RunFailure("ambiguous_http_framing")
        total = 0
        if transfer is not None:
            if transfer.lower() != b"chunked":
                raise RunFailure("unsupported_http_framing")
            while True:
                line = self.line(128)
                if not re.fullmatch(rb"[0-9A-Fa-f]{1,16}\r\n", line):
                    raise RunFailure("invalid_chunk_size")
                size = int(line, 16)
                if size > self.limits["max_response_bytes"] - total:
                    raise RunFailure("response_body_limit")
                if not size:
                    if self.line(2) != b"\r\n":
                        raise RunFailure("unsupported_http_trailers")
                    break
                total += size
                yield from self.take(size)
                if b"".join(self.take(2)) != b"\r\n":
                    raise RunFailure("invalid_chunk_end")
        elif length is not None:
            if not re.fullmatch(rb"(?:0|[1-9][0-9]{0,19})", length):
                raise RunFailure("invalid_content_length")
            size = int(length)
            if size > self.limits["max_response_bytes"]:
                raise RunFailure("response_body_limit")
            yield from self.take(size)
        else:
            if self.buffer:
                total = len(self.buffer)
                if total > self.limits["max_response_bytes"]:
                    raise RunFailure("response_body_limit")
                yield bytes(self.buffer)
                self.buffer.clear()
            while raw := self.receive():
                total += len(raw)
                if total > self.limits["max_response_bytes"]:
                    raise RunFailure("response_body_limit")
                yield raw
        if self.buffer:
            raise RunFailure("trailing_http_bytes")


def _sse(chunks, limits):
    line, lines, event_bytes, event_type, data, first, pending_cr = bytearray(), 0, 0, "message", [], True, False

    def consume(raw):
        nonlocal lines, event_bytes, event_type, data, first
        if first:
            raw = raw.removeprefix(b"\xef\xbb\xbf")
            first = False
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise RunFailure("invalid_sse_utf8") from exc
        if not text:
            result = (event_type, b"\n".join(data)) if data else None
            lines, event_bytes, event_type, data = 0, 0, "message", []
            return result
        lines += 1
        event_bytes += len(raw) + 1
        if event_bytes > limits["max_event_bytes"]:
            raise RunFailure("event_byte_limit")
        if text.startswith(":"):
            return None
        field, _, value = text.partition(":")
        value = value.removeprefix(" ")
        if field == "data":
            data.append(value.encode("utf-8"))
        elif field == "event":
            event_type = value
        elif field in ("id", "retry"):
            raise RunFailure("unsupported_sse_state")
        return None

    for raw in chunks:
        for byte in raw:
            if pending_cr:
                pending_cr = False
                if byte == 10:
                    continue
            if byte in (10, 13):
                result = consume(bytes(line))
                line.clear()
                pending_cr = byte == 13
                if result is not None:
                    yield result
            else:
                line.append(byte)
                if len(line) + event_bytes > limits["max_event_bytes"]:
                    raise RunFailure("event_byte_limit")
    if line or data:
        raise RunFailure("truncated_sse")


def _summary(values):
    if not values:
        return {"samples": 0, "p95_ns": None, "p99_ns": None, "worst_ns": None}
    ordered = sorted(values)
    return {"samples": len(ordered), "p95_ns": str(ordered[math.ceil(len(ordered) * .95) - 1]),
            "p99_ns": str(ordered[math.ceil(len(ordered) * .99) - 1]), "worst_ns": str(ordered[-1])}


class _Completion:
    def __init__(self, body, step, limits, started):
        self.body, self.step, self.limits, self.started = body, step, limits, started
        self.events = 0
        self.finished = self.done = False
        self.finish = None
        self.usage = None
        self.content, self.reasoning, self.tools = [], [], {}
        self.delta_times = array("Q")
        self.event_digest = hashlib.sha256()
        self.response_id = None
        self.metadata = None
        self.token_times = array("Q")
        self.token_receipts = array("Q")
        self.previous_writes = array("Q")

    def consume(self, event, data):
        self.events += 1
        if self.events > self.limits["max_events_per_request"]:
            raise RunFailure("event_count_limit")
        if self.done:
            raise RunFailure("event_after_done")
        now = time.monotonic_ns()
        self.event_digest.update(len(event.encode()).to_bytes(8, "little") + event.encode() + len(data).to_bytes(8, "little") + data)
        if event == "error":
            raise RunFailure("server_error_event")
        if data == b"[DONE]":
            if event not in ("message", "") or not self.finished or self.usage is None:
                raise RunFailure("done_before_finish_or_usage")
            if self.body.get("ria_measurements") is True and self.metadata is None:
                raise RunFailure("missing_token_diagnostics")
            self.done = True
            return
        try:
            value = loads(data, max_bytes=self.limits["max_event_bytes"], max_nodes=self.limits["max_json_nodes"], max_depth=self.limits["max_json_depth"])
        except ArtifactError as exc:
            raise RunFailure("invalid_sse_json") from exc
        if not isinstance(value, dict) or "error" in value:
            raise RunFailure("server_error_payload")
        if event == "ria_measurement":
            if self.body.get("ria_measurements") is not True or self.finished or set(value) != {"index", "elapsed_ns", "previous_write_ns"}:
                raise RunFailure("invalid_token_measurement")
            try:
                index, elapsed, previous_write = (u64(value[key]) for key in ("index", "elapsed_ns", "previous_write_ns"))
            except ArtifactError as exc:
                raise RunFailure("invalid_token_measurement") from exc
            if index != len(self.token_times) or (self.token_times and elapsed < self.token_times[-1]) or previous_write > elapsed:
                raise RunFailure("invalid_token_measurement_order")
            self.token_times.append(elapsed)
            self.token_receipts.append(now)
            self.previous_writes.append(previous_write)
            return
        if event == "ria_diagnostics":
            if self.body.get("ria_measurements") is not True or not self.finished or self.usage is None or self.metadata is not None:
                raise RunFailure("invalid_token_diagnostics")
            if set(value) != {"phase", "reused_prefix_tokens", "sampled_tokens", "completion_tokens", "write_ns"} or value["phase"] not in ("prefill", "continuation"):
                raise RunFailure("invalid_token_diagnostics")
            try:
                reused, sampled, writes = (u64(value[key]) for key in ("reused_prefix_tokens", "sampled_tokens", "write_ns"))
            except ArtifactError as exc:
                raise RunFailure("invalid_token_diagnostics") from exc
            completion = value["completion_tokens"]
            if type(completion) is not int or completion != self.usage["completion_tokens"] or sampled != len(self.token_times) or sampled not in (completion, completion + 1):
                raise RunFailure("invalid_token_diagnostics_count")
            if (value["phase"] == "prefill" and reused != 0) or (value["phase"] == "continuation" and reused == 0) or reused > self.usage["prompt_tokens"]:
                raise RunFailure("invalid_prefix_diagnostics")
            if writes < sum(self.previous_writes) or (self.token_times and writes > self.token_times[-1] + self.limits["request_timeout_ms"] * 1000000):
                raise RunFailure("invalid_writer_diagnostics")
            self.metadata = {"phase": value["phase"], "reused_prefix_tokens": str(reused),
                "sampled_tokens": str(sampled), "terminal_eos_sampled": sampled == completion + 1, "write_ns": str(writes)}
            return
        if event not in ("message", ""):
            raise RunFailure("unsupported_sse_event")
        if value.get("object") != "chat.completion.chunk" or value.get("model") != self.body["model"] or not isinstance(value.get("id"), str):
            raise RunFailure("invalid_completion_envelope")
        if self.response_id is None:
            self.response_id = value["id"]
        elif self.response_id != value["id"]:
            raise RunFailure("changed_completion_identity")
        choices = value.get("choices")
        if not isinstance(choices, list) or len(choices) > 1:
            raise RunFailure("invalid_completion_choices")
        usage = value.get("usage")
        if usage is not None:
            if not self.finished or choices or self.usage is not None or not isinstance(usage, dict):
                raise RunFailure("invalid_completion_usage")
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if type(usage.get(key)) is not int or not 0 <= usage[key] <= 9007199254740991:
                    raise RunFailure("invalid_completion_usage")
            if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
                raise RunFailure("invalid_completion_usage")
            maximum = self.body.get("max_tokens", self.body.get("max_completion_tokens"))
            if usage["completion_tokens"] > maximum:
                raise RunFailure("completion_token_limit")
            self.usage = {key: usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
        if not choices:
            if usage is None:
                raise RunFailure("empty_completion_event")
            return
        choice = choices[0]
        if not isinstance(choice, dict) or type(choice.get("index")) is not int or choice["index"] != 0 or self.finished:
            raise RunFailure("invalid_completion_choice")
        delta, finish = choice.get("delta"), choice.get("finish_reason")
        if not isinstance(delta, dict) or set(delta) - {"role", "content", "reasoning_content", "tool_calls"}:
            raise RunFailure("invalid_completion_delta")
        if "role" in delta and delta["role"] != "assistant":
            raise RunFailure("invalid_completion_role")
        observed = False
        for key, target in (("content", self.content), ("reasoning_content", self.reasoning)):
            if key in delta:
                if not isinstance(delta[key], str):
                    raise RunFailure("invalid_completion_delta")
                target.append(delta[key])
                observed |= bool(delta[key])
        calls = delta.get("tool_calls", [])
        if not isinstance(calls, list) or len(calls) > 256:
            raise RunFailure("invalid_tool_delta")
        for call in calls:
            if not isinstance(call, dict) or type(call.get("index")) is not int or not 0 <= call["index"] < 256:
                raise RunFailure("invalid_tool_delta")
            if set(call) - {"index", "id", "type", "function", "namespace"}:
                raise RunFailure("invalid_tool_delta")
            saved = self.tools.setdefault(call["index"], {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
            for key in ("id", "namespace"):
                if key in call:
                    if not isinstance(call[key], str) or (saved.get(key) and saved[key] != call[key]):
                        raise RunFailure("invalid_tool_identity")
                    saved[key] = call[key]
            if "type" in call and call["type"] != "function":
                raise RunFailure("invalid_tool_type")
            function = call.get("function", {})
            if not isinstance(function, dict) or set(function) - {"name", "arguments"}:
                raise RunFailure("invalid_tool_delta")
            for key, chunk in function.items():
                if not isinstance(chunk, str):
                    raise RunFailure("invalid_tool_delta")
                saved["function"][key] += chunk
                observed |= bool(chunk)
        if observed:
            self.delta_times.append(now)
        if finish is not None:
            if finish not in self.step["allowed_finish_reasons"]:
                raise RunFailure("bad_finish_reason")
            self.finished, self.finish = True, finish

    def result(self):
        if not self.done:
            raise RunFailure("missing_done")
        for call in self.tools.values():
            if not call["id"] or not call["function"]["name"]:
                raise RunFailure("incomplete_tool_call")
            try:
                arguments = loads(call["function"]["arguments"], max_bytes=self.limits["max_response_bytes"])
            except ArtifactError as exc:
                raise RunFailure("invalid_tool_arguments") from exc
            if not isinstance(arguments, dict):
                raise RunFailure("invalid_tool_arguments")
        if bool(self.tools) != (self.finish == "tool_calls"):
            raise RunFailure("tool_finish_mismatch")
        message = _assistant({"content": "".join(self.content), "reasoning_content": "".join(self.reasoning),
                              "tool_calls": [self.tools[key] for key in sorted(self.tools)]})
        response_digest = hashlib.sha256(canonical(message)).hexdigest()
        if self.step["expected_response_sha256"] is not None and response_digest != self.step["expected_response_sha256"]:
            raise RunFailure("response_identity_mismatch")
        features = set()
        if message["content"]:
            features.add("text")
        if message["reasoning_content"]:
            features.add("reasoning")
        if message["tool_calls"]:
            features.add("tools")
        return response_digest, features


def _request(endpoint, token, body, payload, step, limits, global_deadline):
    address, port, path = endpoint
    started = time.monotonic_ns()
    if started >= global_deadline:
        raise RunFailure("global_deadline")
    deadline = min(global_deadline, started + limits["request_timeout_ms"] * 1000000)
    family = socket.AF_INET6 if address.version == 6 else socket.AF_INET
    host = f"[{address}]" if address.version == 6 else str(address)
    headers = (f"POST {path} HTTP/1.1\r\nHost: {host}:{port}\r\nContent-Type: application/json\r\n"
               f"Accept: text/event-stream\r\nConnection: close\r\nContent-Length: {len(payload)}\r\n"
               "Authorization: Bearer ").encode() + token + b"\r\n\r\n"
    completion = _Completion(body, step, limits, started)
    reader = None
    body_digest, body_bytes = hashlib.sha256(), 0
    try:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.settimeout(min((deadline - started) / 1e9, limits["connect_timeout_ms"] / 1000))
            try:
                sock.connect((str(address), port))
                remaining = deadline - time.monotonic_ns()
                if remaining <= 0:
                    raise RunFailure("request_deadline")
                sock.settimeout(min(remaining / 1e9, limits["idle_timeout_ms"] / 1000))
                sock.sendall(headers)
                sock.sendall(payload)
            except socket.timeout as exc:
                raise RunFailure("io_timeout") from exc
            except OSError as exc:
                raise RunFailure("io_error") from exc
            reader = _Reader(sock, limits, deadline, global_deadline)

            def observed():
                nonlocal body_bytes
                for raw in reader.body():
                    body_digest.update(raw)
                    body_bytes += len(raw)
                    yield raw

            for event, data in _sse(observed(), limits):
                completion.consume(event, data)
            response_digest, features = completion.result()
    except RunFailure as exc:
        exc.partial = {"step": step["id"], "request_digest": step["request_digest"],
            "elapsed_ns": str(time.monotonic_ns() - started), "events": completion.events,
            "response_bytes": body_bytes, "wire_bytes": reader.wire_bytes if reader else 0,
            "sse_sha256": body_digest.hexdigest(), "event_sha256": completion.event_digest.hexdigest(),
            "error": str(exc)}
        raise
    ended = time.monotonic_ns()
    gaps = array("Q", (b - a for a, b in zip(completion.delta_times, completion.delta_times[1:], strict=False)))
    token_gaps = array("Q", (b - a for a, b in zip(completion.token_times, completion.token_times[1:], strict=False)))
    metadata = completion.metadata
    record_value = {"step": step["id"], "request_digest": step["request_digest"],
        "request_body_sha256": hashlib.sha256(payload).hexdigest(), "response_sha256": response_digest,
        "sse_sha256": body_digest.hexdigest(), "event_sha256": completion.event_digest.hexdigest(),
        "elapsed_ns": str(ended - started), "ttft_ns": str(completion.delta_times[0] - started) if completion.delta_times else None,
        "events": completion.events, "delta_events": len(completion.delta_times), "response_bytes": body_bytes,
        "wire_bytes": reader.wire_bytes, "finish_reason": completion.finish, "usage": completion.usage,
        "inter_delta": _summary(gaps), "intertoken": _summary(token_gaps) if metadata else None,
        "token_ttft_ns": str(completion.token_receipts[0] - started) if completion.token_receipts else None,
        "worker_token_ttft_ns": str(completion.token_times[0]) if completion.token_times else None,
        "observed_features": sorted(features),
        "observed_phase": metadata["phase"] if metadata else None,
        "reused_prefix_tokens": metadata["reused_prefix_tokens"] if metadata else None,
        "sampled_tokens": metadata["sampled_tokens"] if metadata else None,
        "terminal_eos_sampled": metadata["terminal_eos_sampled"] if metadata else None,
        "writer_elapsed_ns": metadata["write_ns"] if metadata else None, "error": None}
    return record_value, gaps, token_gaps


def _rss():
    if sys.platform != "linux":
        raise ArtifactError("replay resource accounting requires Linux")
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _sample_server(pid):
    try:
        with open_regular(f"/proc/{pid}/stat") as stream:
            raw = stream.read(65537)
        if len(raw) > 65536:
            raise RunFailure("server_resource_read")
        # comm may contain spaces and ')'; the final ')' ends it.
        fields = raw[raw.rfind(b")") + 2:].split()
        return int(fields[19]), int(fields[21]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, IndexError, ValueError) as exc:
        raise RunFailure("server_resource_read") from exc


def execute_plan(document, root, policy, bearer_file, *, execute=False, plan_path=None):
    if execute is not True:
        raise ArtifactError("HTTP execution requires explicit --execute")
    started = time.monotonic_ns()
    loaded = validate_plan(document, root, policy)
    limits = document["limits"]
    if plan_path is not None:
        from .identity import read_json
        verify_identity(read_json(within(root, plan_path), max_bytes=256 << 10), document["digest"])
    elif document["mode"] == "release":
        raise ArtifactError("physical replay requires an immutable plan file reference")
    global_deadline = started + limits["global_deadline_ms"] * 1000000
    token = _credential(bearer_file)
    endpoint = _endpoint(document["endpoint"])
    startup = time.monotonic_ns() - started
    records, gaps, ttfts, request_times = [], array("Q"), array("Q"), array("Q")
    features, phases, observed_cells, errors = set(), set(), set(), 0
    token_gaps, token_ttfts = array("Q"), array("Q")
    completed = tokens = total_events = http_ns = cycles = resource_samples = 0
    server_start = server_peak = None
    http_started = time.monotonic_ns()
    aborted = None
    try:
        if _rss() > u64(limits["max_runner_rss_bytes"]):
            raise RunFailure("runner_rss_limit")
        for cycle in range(limits["max_cycles"]):
            previous_success = None
            for step, body, payload, request_features in loaded:
                if len(records) >= limits["max_requests"]:
                    raise RunFailure("request_population_limit")
                if time.monotonic_ns() >= global_deadline:
                    raise RunFailure("global_deadline")
                attempted_started = time.monotonic_ns()
                entry = None
                try:
                    if step["continuation_of"] is not None and previous_success != step["continuation_of"]:
                        raise RunFailure("continuation_dependency_failed")
                    remaining_events = limits["max_total_events"] - total_events
                    if remaining_events <= 0:
                        raise RunFailure("total_event_limit")
                    bounded_limits = {**limits, "max_events_per_request": min(limits["max_events_per_request"], remaining_events)}
                    entry, current_gaps, current_token_gaps = _request(endpoint, token, body, payload, step, bounded_limits, global_deadline)
                    if total_events + entry["events"] > limits["max_total_events"]:
                        raise RunFailure("total_event_limit")
                    total_events += entry["events"]
                    gaps.extend(current_gaps)
                    token_gaps.extend(current_token_gaps)
                    if entry["token_ttft_ns"] is not None:
                        token_ttfts.append(int(entry["token_ttft_ns"]))
                    request_times.append(int(entry["elapsed_ns"]))
                    if entry["ttft_ns"] is not None:
                        ttfts.append(int(entry["ttft_ns"]))
                    http_ns += int(entry["elapsed_ns"])
                    completed += 1
                    tokens += entry["usage"]["completion_tokens"]
                    # Image success observes an admitted image request, not vision quality.
                    if "images" in request_features:
                        entry["observed_features"].append("images")
                    # Exact private-state reuse requires server telemetry, never HTTP prefix inference.
                    if entry["observed_phase"] is not None:
                        phases.add(entry["observed_phase"])
                        observed_cells.add(cell_id(document["realization"], entry["observed_phase"]))
                    if step["continuation_of"] is not None and entry["observed_phase"] == "continuation" and int(entry["reused_prefix_tokens"]) > 0:
                        entry["observed_features"].append("exact_continuation")
                    observed = set(entry["observed_features"])
                    features.update(observed)
                    if entry["usage"]["completion_tokens"]:
                        phases.add("decode")
                        observed_cells.add(cell_id(document["realization"], "decode"))
                    previous_success = step["id"]
                    if step["phase"] != "decode" and entry["observed_phase"] != step["phase"]:
                        raise RunFailure("required_phase_unobserved")
                    if not set(step["required_features"]) <= observed:
                        raise RunFailure("required_feature_unobserved")
                except RunFailure as exc:
                    errors += 1
                    previous_success = None
                    if entry is None:
                        entry = exc.partial or {"step": step["id"], "request_digest": step["request_digest"],
                                                "elapsed_ns": str(time.monotonic_ns() - attempted_started)}
                        total_events += entry.get("events", 0)
                    entry["error"] = str(exc)
                    if document["stop_on_error"] or errors > limits["max_errors"]:
                        aborted = str(exc)
                entry["cycle"] = cycle
                records.append(entry)
                if _rss() > u64(limits["max_runner_rss_bytes"]):
                    raise RunFailure("runner_rss_limit")
                if document["server_pid"] is not None and resource_samples < document["max_resource_samples"]:
                    identity, rss = _sample_server(document["server_pid"])
                    if server_start is not None and identity != server_start:
                        raise RunFailure("server_pid_replaced")
                    server_start, server_peak = identity, max(server_peak or 0, rss)
                    resource_samples += 1
                if aborted is not None:
                    break
            cycles += 1
            if aborted is not None or document["mode"] == "fixture" or time.monotonic_ns() - http_started >= document["minimum_soak_seconds"] * 1000000000:
                break
        else:
            aborted = "cycle_population_limit"
    except RunFailure as exc:
        aborted = str(exc)
    ended = time.monotonic_ns()
    elapsed = ended - started
    soak = ended - http_started
    physical = document["mode"] == "release" and completed > 0 and soak >= document["minimum_soak_seconds"] * 1000000000
    inter_delta = _summary(gaps)
    ttft = _summary(ttfts)
    intertoken = _summary(token_gaps) if token_ttfts else None
    throughput = tokens * 1e9 / http_ns if http_ns else 0.0
    checks = {}
    result = {"schema_revision": 1, "kind": "physical_replay_evidence" if physical else "replay_measurements",
        "classification": "physical" if physical else "fixture_or_incomplete", "mode": document["mode"],
        **{key: document[key] for key in IDENTITIES}, "plan_digest": document["digest"],
        "plan_reference": {"path": plan_path, "digest": document["digest"]} if plan_path is not None else None,
        "qualification_scope": "HTTP operational workload only; source parity, model quality and other release gates require independent proofs",
        "qualified": False, "hardware_qualified": False, "passed": all(checks.values()),
        "checks": [{"id": key, "passed": value} for key, value in checks.items()],
        "startup_ns": str(startup), "elapsed_ns": str(elapsed), "soak_ns": str(soak),
        "active_http_ns": str(http_ns), "completed_requests": completed, "attempted_requests": len(records),
        "cycles": cycles, "completion_tokens": str(tokens), "errors": errors, "abort_reason": aborted,
        "ttft": ttft, "inter_delta": inter_delta, "intertoken": intertoken,
        "token_ttft": _summary(token_ttfts),
        "timing_scope": "TTFT/delta gaps use client monotonic parsed-event observation; token gaps use explicit worker sampled-token boundaries and include prior network backpressure and terminal EOS when present",
        "token_observer_scope": "ria_measurements=true is a separate observer region; no uninstrumented-performance claim",
        "throughput_tokens_per_second": throughput, "throughput_scope": "final server completion-token usage divided by sum of complete HTTP request lifetimes",
        "observed_features": sorted(features), "observed_phases": sorted(phases), "observed_cells": [],
        "observed_workload_cells": sorted(observed_cells), "realization": document["realization"],
        "cell_scope": "HTTP observes phases under configured realization; actual route/cache placement hits are uninstrumented, so no release placement cell is credited",
        "unexecuted_gates": ["all 28 release gates unless independently proven", "source/native model parity", "model quality", "actual route/cache placement and NUMA/residency execution",
                             *([] if token_ttfts else ["exact intertoken timing"]),
                             *([] if "exact_continuation" in features else ["exact private-state continuation"])],
        "runner_peak_rss_bytes": str(_rss()), "runner_rss_scope": "Linux ru_maxrss entire runner process lifetime",
        "server_sampled_peak_rss_bytes": str(server_peak) if server_peak is not None else None,
        "server_resource_samples": resource_samples, "server_resource_scope": "RSS samples only at completed request boundaries; not a peak bound",
        "gpu_peak_bytes": None, "records": records}
    checks = _bound_checks(result, document)
    result["checks"] = [{"id": key, "passed": value} for key, value in checks.items()]
    result["passed"] = all(checks.values())
    return seal(result)

# These publication schemas are authoritative for root's matrix/soak verifier.
# They intentionally cannot express a full release qualification claim.
COUNT = {"type": "integer", "minimum": 0, "maximum": 9007199254740991}
NULL_U64 = {"anyOf": [U64, {"type": "null"}]}
SUMMARY = record({"samples": COUNT, "p95_ns": NULL_U64, "p99_ns": NULL_U64, "worst_ns": NULL_U64})
CELL = {"type": "string", "minLength": 1, "maxLength": 256}
ERROR = {"anyOf": [{"type": "string", "pattern": "^[a-z][a-z0-9_]{0,127}$"}, {"type": "null"}]}
RECORD_FIELDS = {"step": NAME, "request_digest": SHA, "cycle": COUNT, "elapsed_ns": U64, "error": ERROR,
    **{key: SHA for key in ("request_body_sha256", "response_sha256", "sse_sha256", "event_sha256")},
    **{key: COUNT for key in ("events", "delta_events", "response_bytes", "wire_bytes")},
    **{key: NULL_U64 for key in ("ttft_ns", "token_ttft_ns", "worker_token_ttft_ns", "reused_prefix_tokens", "sampled_tokens", "writer_elapsed_ns")},
    "finish_reason": {"enum": ["stop", "length", "tool_calls"]},
    "usage": record({"prompt_tokens": COUNT, "completion_tokens": COUNT, "total_tokens": COUNT}),
    "inter_delta": SUMMARY, "intertoken": {"anyOf": [SUMMARY, {"type": "null"}]},
    "observed_features": {"type": "array", "items": {"enum": list(FEATURES)}, "maxItems": 5, "uniqueItems": True},
    "observed_phase": {"enum": ["prefill", "continuation", None]},
    "terminal_eos_sampled": {"type": ["boolean", "null"]}}
RECORD = record(RECORD_FIELDS, tuple(set(RECORD_FIELDS) - {"step", "request_digest", "cycle", "elapsed_ns", "error"}))
REPLAY_EVIDENCE = record({"schema_revision": {"const": 1},
    "kind": {"enum": ["physical_replay_evidence", "replay_measurements"]},
    "classification": {"enum": ["physical", "fixture_or_incomplete"]}, "mode": {"enum": ["fixture", "release"]},
    **{key: SHA for key in IDENTITIES}, "plan_digest": SHA,
    "plan_reference": {"anyOf": [record({"path": TEXT, "digest": SHA}), {"type": "null"}]},
    "qualification_scope": TEXT, "qualified": {"const": False}, "hardware_qualified": {"const": False},
    "passed": {"type": "boolean"}, "checks": {"type": "array", "minItems": 1, "maxItems": 32,
        "items": record({"id": {"type": "string", "pattern": "^[a-z0-9_]{1,128}$"}, "passed": {"type": "boolean"}})},
    **{key: U64 for key in ("startup_ns", "elapsed_ns", "soak_ns", "active_http_ns", "completion_tokens", "runner_peak_rss_bytes")},
    **{key: COUNT for key in ("completed_requests", "attempted_requests", "cycles", "errors", "server_resource_samples")},
    "abort_reason": ERROR, "ttft": SUMMARY, "inter_delta": SUMMARY,
    "intertoken": {"anyOf": [SUMMARY, {"type": "null"}]}, "token_ttft": SUMMARY,
    **{key: TEXT for key in ("timing_scope", "token_observer_scope", "throughput_scope", "cell_scope", "runner_rss_scope", "server_resource_scope")},
    "throughput_tokens_per_second": {"type": "number", "minimum": 0},
    "observed_features": {"type": "array", "items": {"enum": list(FEATURES)}, "maxItems": 5, "uniqueItems": True},
    "observed_phases": {"type": "array", "items": {"enum": ["prefill", "decode", "continuation"]}, "maxItems": 3, "uniqueItems": True},
    "observed_cells": {"type": "array", "maxItems": 0},
    "observed_workload_cells": {"type": "array", "items": CELL, "maxItems": 3, "uniqueItems": True},
    "realization": PLAN["properties"]["realization"],
    "unexecuted_gates": {"type": "array", "items": TEXT, "minItems": 4, "maxItems": 8},
    "server_sampled_peak_rss_bytes": NULL_U64, "gpu_peak_bytes": {"const": None},
    "records": {"type": "array", "items": RECORD, "maxItems": 65536}, "digest": SHA})
RELEASE_RUN_SCHEMAS["release-replay-evidence"] = REPLAY_EVIDENCE


def validate_evidence(evidence, document, root, policy):
    """Verify a sealed measured report and its frozen plan; never infer G gates."""
    from .schemas import StrictValidator
    check_json(evidence, max_nodes=16000000, max_depth=32)
    if next(StrictValidator(REPLAY_EVIDENCE).iter_errors(evidence), None):
        raise ArtifactError("invalid replay evidence contract")
    verify_identity(evidence)
    validate_plan(document, root, policy)
    if evidence["plan_digest"] != document["digest"] or any(evidence[key] != document[key] for key in IDENTITIES) or evidence["realization"] != document["realization"] or evidence["mode"] != document["mode"]:
        raise ArtifactError("replay evidence differs from the frozen realization")
    reference = evidence["plan_reference"]
    if reference is not None:
        from .identity import read_json
        if reference["digest"] != document["digest"]:
            raise ArtifactError("replay plan reference identity mismatch")
        value = read_json(within(root, reference["path"]), max_bytes=256 << 10)
        verify_identity(value, document["digest"])
    steps = {step["id"]: step for step in document["steps"]}
    records, limits = evidence["records"], document["limits"]
    if len(records) != evidence["attempted_requests"] or len(records) > limits["max_requests"]:
        raise ArtifactError("replay request population differs from declared bounds")
    for index, item in enumerate(records):
        step = document["steps"][index % len(document["steps"])]
        if item["step"] != step["id"] or item["cycle"] != index // len(document["steps"]) or item["request_digest"] != step["request_digest"]:
            raise ArtifactError("replay request sequence differs from exact preregistered workload")
    complete = [item for item in records if "usage" in item]
    if len(complete) != evidence["completed_requests"] or sum(item["usage"]["completion_tokens"] for item in complete) != u64(evidence["completion_tokens"]) or sum(item["error"] is not None for item in records) != evidence["errors"]:
        raise ArtifactError("replay counters differ from actual response records")
    features, phases = set(), set()
    for item in complete:
        features.update(item["observed_features"])
        if item["observed_phase"] is not None:
            phases.add(item["observed_phase"])
        if item["usage"]["completion_tokens"]:
            phases.add("decode")
        if "exact_continuation" in item["observed_features"] and (steps[item["step"]]["continuation_of"] is None or item["observed_phase"] != "continuation" or u64(item["reused_prefix_tokens"]) == 0):
            raise ArtifactError("continuation evidence lacks exact prefix observation")
    if features != set(evidence["observed_features"]) or phases != set(evidence["observed_phases"]) or set(evidence["observed_workload_cells"]) != {cell_id(document["realization"], phase) for phase in phases}:
        raise ArtifactError("replay coverage differs from actual observations")
    if u64(evidence["soak_ns"]) > u64(evidence["elapsed_ns"]) or u64(evidence["startup_ns"]) > u64(evidence["elapsed_ns"]) or sum(u64(item["elapsed_ns"]) for item in complete) != u64(evidence["active_http_ns"]):
        raise ArtifactError("invalid measured replay durations")
    physical = document["mode"] == "release" and bool(complete) and u64(evidence["soak_ns"]) >= document["minimum_soak_seconds"] * 1000000000
    if (evidence["kind"] == "physical_replay_evidence") != physical or (evidence["classification"] == "physical") != physical or (physical and reference is None):
        raise ArtifactError("physical replay classification lacks actual preregistered soak")
    checks = evidence["checks"]
    if len({check["id"] for check in checks}) != len(checks) or evidence["passed"] != all(check["passed"] for check in checks):
        raise ArtifactError("replay pass bit differs from declared checks")
    required = {"run_completed", "global_deadline", "error_bound", "completed_request_floor", "completion_token_floor", "throughput_floor", "runner_rss_bound", "ttft_bound", "request_bound", "required_features", "required_phases", "soak_floor", "inter_delta_p95_bound", "inter_delta_p99_bound", "inter_delta_worst_bound"}
    if document["require_token_measurements"]:
        required.update({"token_ttft_bound", "intertoken_p95_bound", "intertoken_p99_bound", "intertoken_worst_bound"})
    if {check["id"] for check in checks} != required:
        raise ArtifactError("replay evidence omits an explicit policy bound")
    if {check["id"]: check["passed"] for check in checks} != _bound_checks(evidence, document):
        raise ArtifactError("replay checks disagree with measured values and frozen limits")
    return evidence


def _bound_checks(evidence, document):
    limits = document["limits"]
    elapsed = u64(evidence["elapsed_ns"])
    complete = [item for item in evidence["records"] if "usage" in item]
    http_ns = u64(evidence["active_http_ns"])
    tokens = u64(evidence["completion_tokens"])
    throughput = tokens * 1e9 / http_ns if http_ns else 0.0
    if evidence["throughput_tokens_per_second"] != throughput:
        raise ArtifactError("throughput differs from actual token usage and request duration")
    ttft = evidence["ttft"]
    request_ns = [u64(item["elapsed_ns"]) for item in evidence["records"]]
    checks = {
        "run_completed": evidence["abort_reason"] is None,
        "global_deadline": elapsed <= limits["global_deadline_ms"] * 1000000,
        "error_bound": evidence["errors"] <= limits["max_errors"],
        "completed_request_floor": len(complete) >= limits["min_completed_requests"],
        "completion_token_floor": tokens >= limits["min_completion_tokens"],
        "throughput_floor": throughput >= limits["min_tokens_per_second"],
        "runner_rss_bound": u64(evidence["runner_peak_rss_bytes"]) <= u64(limits["max_runner_rss_bytes"]),
        "ttft_bound": ttft["worst_ns"] is not None and u64(ttft["worst_ns"]) <= u64(limits["max_ttft_ns"]),
        "request_bound": bool(request_ns) and max(request_ns) <= u64(limits["max_request_ns"]),
        "required_features": set(document["required_features"]) <= set(evidence["observed_features"]),
        "required_phases": set(document["required_phases"]) <= set(evidence["observed_phases"]),
        "soak_floor": document["mode"] == "fixture" or u64(evidence["soak_ns"]) >= document["minimum_soak_seconds"] * 1000000000,
    }
    for metric in ("p95", "p99", "worst"):
        value = evidence["inter_delta"][metric + "_ns"]
        checks[f"inter_delta_{metric}_bound"] = value is not None and u64(value) <= u64(limits[f"max_inter_delta_{metric}_ns"])
    if document["require_token_measurements"]:
        token_ttft = evidence["token_ttft"]["worst_ns"]
        checks["token_ttft_bound"] = token_ttft is not None and u64(token_ttft) <= u64(limits["max_token_ttft_ns"])
        for metric in ("p95", "p99", "worst"):
            value = evidence["intertoken"][metric + "_ns"] if evidence["intertoken"] else None
            checks[f"intertoken_{metric}_bound"] = value is not None and u64(value) <= u64(limits[f"max_intertoken_{metric}_ns"])
    return checks
