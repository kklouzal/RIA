"""Model-free paired production TLS/protocol qualification and boundary checks."""

import hashlib
import json
import os
import socket
import subprocess
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


@pytest.fixture()
def provision(tmp_path):
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
            "tls": {
                "ca_file": str(tmp_path / "ca.pem"),
                "certificate_file": str(tmp_path / f"{role}.pem"),
                "private_key_file": str(tmp_path / f"{role}.key"),
                "expected_peer_name": f"{peer}.test",
                "authorized_peer_sha256": certificate_hash[peer],
            },
        }
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


def test_real_paired_transport(provision):
    configs, requests, certificates = provision
    client, server = paired(configs, requests)
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
        assert (
            document["peer_certificate_digest"]
            == certificates["expert" if document["role"] == "client" else "client"]
        )
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
