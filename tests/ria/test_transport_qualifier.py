"""Model-free paired production TLS/trusted-network protocol boundary checks."""

import contextlib
import hashlib
import json
import os
import socket
import struct
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXE = Path(os.environ.get("RIA_TRANSPORT_FIXTURE", str(ROOT / "bin/ds4ctl"))).resolve()


def canonical(document):
    # Fixture documents contain ASCII keys/values and safe integers exclusively.
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode()


def seal(document):
    return {**document, "digest": hashlib.sha256(canonical(document)).hexdigest()}


def invoke(config, request):
    if EXE.name == "ria-transport-fixture-main":
        return [str(EXE), str(config), str(request)]
    return [
        str(EXE),
        "qualify-transport",
        "--config",
        str(config),
        "--request",
        str(request),
    ]


def openssl(directory, *args):
    result = subprocess.run(
        ["openssl", *args], cwd=directory, capture_output=True, timeout=10, check=False
    )
    assert result.returncode == 0, result.stderr.decode()
    return result.stdout


def provision_tls(tmp_path):
    openssl(
        tmp_path,
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-days",
        "1",
        "-keyout",
        "ca.key",
        "-out",
        "ca.pem",
        "-subj",
        "/CN=fixture CA",
        "-addext",
        "basicConstraints=critical,CA:TRUE",
        "-addext",
        "keyUsage=critical,keyCertSign,cRLSign",
    )
    certificate_hash = {}
    for role in ("client", "expert"):
        openssl(
            tmp_path,
            "req",
            "-new",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            f"{role}.key",
            "-out",
            f"{role}.csr",
            "-subj",
            f"/CN={role}.test",
        )
        ext = tmp_path / f"{role}.ext"
        ext.write_text(
            f"subjectAltName=DNS:{role}.test\nextendedKeyUsage=serverAuth,clientAuth\n"
            "basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\n"
        )
        openssl(
            tmp_path,
            "x509",
            "-req",
            "-in",
            f"{role}.csr",
            "-CA",
            "ca.pem",
            "-CAkey",
            "ca.key",
            "-CAcreateserial",
            "-days",
            "1",
            "-out",
            f"{role}.pem",
            "-extfile",
            str(ext),
        )
        certificate_hash[role] = hashlib.sha256(
            openssl(tmp_path, "x509", "-in", f"{role}.pem", "-outform", "DER")
        ).hexdigest()
        (tmp_path / f"{role}.key").chmod(0o600)
    (tmp_path / "ca.key").chmod(0o600)
    return certificate_hash


@pytest.fixture()
def provision(tmp_path, request):
    tls_mode = getattr(request, "param", None)
    certificate_hash = {} if tls_mode is False else provision_tls(tmp_path)
    sockets = [socket.socket() for _ in range(2)]
    for item in sockets:
        item.bind(("127.0.0.1", 0))
    ports = [item.getsockname()[1] for item in sockets]
    for item in sockets:
        item.close()
    build = seal({"schema_revision": 1, "fixture": "actual production transport"})
    build_path = tmp_path / "build.json"
    build_path.write_bytes(canonical(build))
    configs, requests = {}, {}
    for role, peer in (("client", "expert"), ("expert", "client")):
        request = seal(
            {
                "schema_revision": 1,
                "kind": "transport_request",
                "environment_digest": hashlib.sha256(role.encode()).hexdigest(),
                "build_digest": build["digest"],
                "policy_digest": "2" * 64,
                "logical_model_digest": "3" * 64,
                "source_lock_digest": "4" * 64,
                "operator_contract_digest": "5" * 64,
                "preregistration_digest": "6" * 64,
                "deadline_ms": 15000,
                "warmup": 1,
                "repeats": 2,
                "fixture_seed": 718,
                "max_frame_bytes": "4096",
                "control_credit": "131328",
                "expert_credit": "33160",
                "row_credit": "432",
                "bulk_credit": "8192",
            }
        )
        config = {
            "schema_revision": 1,
            "role": role,
            "environment_digest": request["environment_digest"],
            "build_digest": build["digest"],
            "build_info_file": str(build_path),
            "request_digest": request["digest"],
            "network": {
                "control_address": f"127.0.0.1:{ports[0]}",
                "bulk_address": f"127.0.0.1:{ports[1]}",
                "connect_timeout_ms": 2000,
                "handshake_timeout_ms": 2000,
                "operation_timeout_ms": 3000,
                "frame_io_timeout_ms": 150,
                "write_timeout_ms": 3000,
            },
            "tls": {"enabled": False}
            if tls_mode is False
            else {
                "ca_file": str(tmp_path / "ca.pem"),
                "certificate_file": str(tmp_path / f"{role}.pem"),
                "private_key_file": str(tmp_path / f"{role}.key"),
                "expected_peer_name": f"{peer}.test",
                "authorized_peer_sha256": certificate_hash[peer],
            },
        }
        if tls_mode is True:
            config["tls"]["enabled"] = True
        requests[role], configs[role] = (
            tmp_path / f"{role}-request.json",
            tmp_path / f"{role}-config.json",
        )
        requests[role].write_bytes(canonical(request))
        configs[role].write_bytes(canonical(config))
    return configs, requests, certificate_hash


def paired(configs, requests):
    server = subprocess.Popen(
        invoke(configs["expert"], requests["expert"]),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        client = subprocess.run(
            invoke(configs["client"], requests["client"]),
            capture_output=True,
            timeout=18,
            check=False,
        )
        server_out, server_err = server.communicate(timeout=18)
        assert client.returncode == 0, client.stderr.decode()
        assert server.returncode == 0, server_err.decode()
        return json.loads(client.stdout), json.loads(server_out)
    finally:
        if server.poll() is None:
            server.kill()
            server.communicate(timeout=5)


@pytest.mark.parametrize(
    "provision",
    [None, True, False],
    indirect=True,
    ids=["tls-default", "tls-explicit", "trusted-network"],
)
def test_real_paired_transport(provision):
    configs, requests, certificates = provision
    client, server = paired(configs, requests)
    tls_enabled = bool(certificates)
    if not tls_enabled:
        assert not list(configs["client"].parent.glob("*.pem"))
        assert not list(configs["client"].parent.glob("*.key"))
    expected = {
        "health": (21, 0),
        "rows": (11, 0),
        "bulk_chunk": (12, 0),
        "cancel_pending": (20, 0),
        "expert_cancelled": (10, 8),
        "expert_success": (10, 0),
        "cancel_terminal": (20, 0),
        "cancel_unknown": (20, 0),
        "malformed_rows": (11, 5),
    }
    for document in (client, server):
        assert document["qualified"] is False
        assert document["kind"] == "native_transport_measurements"
        assert (
            document["warmup_completed"] == 1 and document["iterations_completed"] == 2
        )
        assert len(document["checks"]) == 8 and all(
            item["passed"] for item in document["checks"]
        )
        assert document["tls_enabled"] is tls_enabled
        assert document["peer_certificate_digest"] == (
            certificates["expert" if document["role"] == "client" else "client"]
            if tls_enabled
            else None
        )
        assert [item["id"] for item in document["checks"]] == [
            "mtls_san_certificate_pair" if tls_enabled else "trusted_network_peer_pair",
            "bind_and_bulk_one_use",
            "credit_and_protected_progress",
            "cancel_no_early_credit",
            "terminal_history",
            "malformed_typed_error",
            "partial_frame_deadline",
            "response_payload_integrity",
        ]
        assert document["timeout_elapsed_ns"] >= 149000000
        assert int(document["owned_buffers_peak_bytes"]) > 4096
        assert int(document["max_rss_bytes"]) > 4096
        assert (
            document["elapsed_ns"]
            >= document["startup_ns"] + document["timeout_elapsed_ns"]
        )
        assert document["cpu_ns"] > 0
        cases = document["cases"]
        assert len(cases) == 18
        assert {(item["iteration"], item["id"]) for item in cases} == {
            (iteration, case) for iteration in range(2) for case in expected
        }
        for item in cases:
            assert (item["kind"], item["status"]) == expected[item["id"]]
            assert item["elapsed_ns"] > 0
        assert int(document["measured_request_bytes"]) == sum(
            int(item["request_bytes"]) for item in cases
        )
        assert int(document["measured_response_bytes"]) == sum(
            int(item["reply_bytes"]) for item in cases
        )
    for left, right in zip(client["cases"], server["cases"], strict=True):
        for field in (
            "iteration",
            "id",
            "kind",
            "status",
            "request_bytes",
            "reply_bytes",
            "request_sha256",
            "response_sha256",
        ):
            assert left[field] == right[field]


@pytest.mark.parametrize(
    "provision", [None, False], indirect=True, ids=["tls-default", "trusted-network"]
)
def test_unregistered_request_rejected_before_network(provision):
    configs, requests, _ = provision
    request = json.loads(requests["client"].read_bytes())
    request["fixture_seed"] += 1
    requests["client"].write_bytes(canonical(request))
    result = subprocess.run(
        invoke(configs["client"], requests["client"]),
        capture_output=True,
        timeout=3,
        check=False,
    )
    assert result.returncode != 0 and not result.stdout
    assert b"authorized" in result.stderr


def test_unmatched_certificate_is_rejected(provision):
    configs, requests, _ = provision
    config = json.loads(configs["client"].read_bytes())
    config["tls"]["authorized_peer_sha256"] = "f" * 64
    configs["client"].write_bytes(canonical(config))
    server = subprocess.Popen(
        invoke(configs["expert"], requests["expert"]),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        client = subprocess.run(
            invoke(configs["client"], requests["client"]),
            capture_output=True,
            timeout=6,
            check=False,
        )
        assert client.returncode != 0 and not client.stdout
        assert b"certificate" in client.stderr
    finally:
        if server.poll() is None:
            server.kill()
        server.communicate(timeout=5)


@pytest.mark.parametrize(
    "tls",
    [
        {},
        {"enabled": True},
        {"enabled": None},
        {"enabled": 0},
        {"enabled": "false"},
        {"enabled": False, "ca_file": "/absent/ca.pem"},
        {"enabled": False, "expected_peer_name": "ignored.test"},
        {"enabled": False, "authorized_peer_sha256": "f" * 64},
        {"enabled": False, "unknown": True},
    ],
)
@pytest.mark.parametrize("provision", [False], indirect=True)
def test_invalid_mode_configuration_never_falls_back(provision, tls):
    configs, requests, _ = provision
    config = json.loads(configs["client"].read_bytes())
    config["tls"] = tls
    configs["client"].write_bytes(canonical(config))
    result = subprocess.run(
        invoke(configs["client"], requests["client"]),
        capture_output=True,
        timeout=3,
        check=False,
    )
    assert result.returncode != 0 and not result.stdout
    assert result.stderr


@pytest.mark.parametrize("plaintext_role", ["client", "expert"])
def test_mixed_modes_fail_without_fallback(provision, plaintext_role):
    configs, requests, _ = provision
    config = json.loads(configs[plaintext_role].read_bytes())
    config["tls"] = {"enabled": False}
    configs[plaintext_role].write_bytes(canonical(config))
    with server_process(configs, requests) as server:
        client = subprocess.run(
            invoke(configs["client"], requests["client"]),
            capture_output=True,
            timeout=6,
            check=False,
        )
        output, error = server.communicate(timeout=6)
        assert client.returncode != 0 and not client.stdout
        assert server.returncode != 0 and not output
        assert client.stderr and error


@contextlib.contextmanager
def server_process(configs, requests):
    server = subprocess.Popen(
        invoke(configs["expert"], requests["expert"]),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        yield server
    finally:
        if server.poll() is None:
            server.kill()
        server.communicate(timeout=5)


# Independent wire oracle: fixed little-endian fields, never a native struct.
WIRE = struct.Struct("<4sHHIIQQ16sQ8s")


def connect_plain(config_path, channel, *, source="127.0.0.1"):
    config = json.loads(config_path.read_bytes())
    host, port = config["network"][channel + "_address"].rsplit(":", 1)
    deadline = time.monotonic() + 2
    while True:
        try:
            return socket.create_connection(
                (host, int(port)), timeout=1, source_address=(source, 0)
            )
        except ConnectionRefusedError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.01)


def send_plain(sock, payload, *, kind=1, request_id=1, session=bytes(16), epoch=0):
    encoded = canonical(payload)
    sock.sendall(
        WIRE.pack(
            b"DSER", 1, kind, 0, 0, len(encoded), request_id, session, epoch, bytes(8)
        )
        + encoded
    )


def receive_plain(sock):
    def exact(length):
        data = bytearray()
        while len(data) < length:
            chunk = sock.recv(length - len(data))
            if not chunk:
                raise EOFError("incomplete qualifier frame")
            data.extend(chunk)
        return bytes(data)

    fields = WIRE.unpack(exact(WIRE.size))
    assert fields[:5] == (b"DSER", 1, 1, 1, 0)
    assert fields[5] <= 4096
    return json.loads(exact(fields[5]))


def bind_plain(sock, request_path, config_path):
    request = json.loads(request_path.read_bytes())
    network = json.loads(config_path.read_bytes())["network"]
    frame = int(request["max_frame_bytes"])
    credit = sum(
        int(request[name])
        for name in ("control_credit", "expert_credit", "row_credit", "bulk_credit")
    )
    send_plain(
        sock,
        {
            "role": "client",
            "logical_model_digest": request["logical_model_digest"],
            "operator_contract_digest": request["operator_contract_digest"],
            "encoding_digest": request["source_lock_digest"],
            "client_layout_digest": request["environment_digest"],
            "placement_plan_digest": request["preregistration_digest"],
            "profile": "bf16",
            "server_executor": "cpu",
            "limits": {
                "frame_payload_bytes": frame,
                "bulk_data_bytes": min(
                    frame - 64, 4 << 20, int(request["bulk_credit"]) - 208
                ),
                "expert_rows": 1,
                "expert_requests": 2,
                "row_lookup_rows": 1,
                "inflight_payload_bytes": credit,
                **{
                    name: network[name]
                    for name in (
                        "operation_timeout_ms",
                        "frame_io_timeout_ms",
                        "write_timeout_ms",
                    )
                },
            },
        },
    )
    return receive_plain(sock)


@pytest.mark.parametrize("provision", [False], indirect=True)
@pytest.mark.parametrize("defect", ["magic", "reserved", "oversize"])
def test_plaintext_malformed_header_rejected(provision, defect):
    configs, requests, _ = provision
    with server_process(configs, requests) as server:
        with connect_plain(configs["client"], "control") as control:
            encoded = bytearray(
                WIRE.pack(b"DSER", 1, 1, 0, 0, 0, 1, bytes(16), 0, bytes(8))
            )
            if defect == "magic":
                encoded[0] ^= 1
            elif defect == "reserved":
                encoded[-1] = 1
            else:
                struct.pack_into("<Q", encoded, 16, 4097)
            control.sendall(encoded)
            output, error = server.communicate(timeout=3)
        assert server.returncode != 0 and not output
        assert b"wire" in error


@pytest.mark.parametrize("provision", [False], indirect=True)
@pytest.mark.parametrize(
    "defect", ["peer_ip", "capability", "session", "request_order"]
)
def test_plaintext_bulk_binding_rejected(provision, defect):
    configs, requests, _ = provision
    with server_process(configs, requests) as server:
        with connect_plain(configs["client"], "control") as control:
            bound = bind_plain(control, requests["client"], configs["client"])
            with connect_plain(
                configs["client"],
                "bulk",
                source=("127.0.0.2" if defect == "peer_ip" else "127.0.0.1"),
            ) as bulk:
                if defect != "peer_ip":
                    capability = bytearray.fromhex(bound["bulk_capability"])
                    session = bytearray.fromhex(bound["session_id"])
                    if defect == "capability":
                        capability[0] ^= 1
                    if defect == "session":
                        session[0] ^= 1
                    send_plain(
                        bulk,
                        {
                            "session_id": session.hex(),
                            "epoch": "1",
                            "logical_model_digest": bound["logical_model_digest"],
                            "operator_contract_digest": bound[
                                "operator_contract_digest"
                            ],
                            "bulk_capability": capability.hex(),
                        },
                        kind=2,
                        request_id=2 if defect == "request_order" else 1,
                        session=bytes(session),
                        epoch=1,
                    )
                output, error = server.communicate(timeout=3)
        assert server.returncode != 0 and not output
        assert error
