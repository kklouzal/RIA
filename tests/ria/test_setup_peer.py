"""Isolated model-free setup boundaries; actual UID10001 worker requires root."""

import contextlib
import base64
import hashlib
import http.client
import os
from pathlib import Path
import socket
import ssl
import struct
import threading
import time

import pytest

from ria.identity import ArtifactError, digest, loads, seal
from ria.setup_journal import SetupJournal
from ria.setup_limits import MAX_MESSAGE, MAX_CHUNK_METADATA
from ria.setup_peer import (
    BINARY_TYPE,
    BROKER_HEADER,
    CHUNK,
    JSON_TYPE,
    ExportTree,
    PeerClient,
    PeerServer,
    SetupNotReady,
    _mac,
    _receive,
    _send,
    canonical,
    import_tree,
)
from ria.setup_security import (
    SetupSecurity,
    endpoint,
    read_invitation,
    write_invitation,
    validate_invitation,
)


def address():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return "127.0.0.1:" + str(listener.getsockname()[1])


def private(path):
    path.mkdir(mode=0o700)
    return path


def expert(tmp_path, tls=False):
    return SetupSecurity.create_expert(
        tmp_path / "security", "1" * 64, address(), tls_enabled=tls
    )


@contextlib.contextmanager
def channel(tmp_path, dispatcher=None, *, tls=False, timeout=2):
    if os.geteuid() != 0:
        pytest.skip(
            "actual isolated UID10001 setup worker needs a privileged controller"
        )
    security, invitation = expert(tmp_path, tls)
    server = PeerServer(
        invitation,
        security,
        dispatcher or (lambda name, payload: {"name": name, "payload": payload}),
        timeout=timeout,
    )
    server.start()
    try:
        yield server, PeerClient(invitation, timeout=timeout), security
    finally:
        server.close()


@pytest.mark.parametrize(
    "value",
    [
        "example.org:9010",
        "0.0.0.0:9010",
        "8.8.8.8:9010",
        "127.0.0.1:80",
        "[::1]:9010",
        "127.0.0.1:09010",
    ],
)
def test_endpoint_rejects_unsupported(value):
    with pytest.raises(ArtifactError):
        endpoint(value)


def test_plaintext_creates_no_crypto_or_pem(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("plaintext must not run OpenSSL")

    monkeypatch.setattr("ria.setup_security._openssl", forbidden)
    security, invitation = expert(tmp_path)
    client = SetupSecurity.create_client(tmp_path / "client", invitation)
    assert not list(security.directory.iterdir()) and not list(
        client.directory.iterdir()
    )
    assert security.worker_credentials() is None and client.peer_name("expert") is None
    assert security.expected_client_leaf_digest() is None
    destination = tmp_path / "invitation.json"
    write_invitation(destination, invitation)
    assert destination.stat().st_mode & 0o777 == 0o600
    loaded = read_invitation(destination)
    assert loaded == invitation
    destination.chmod(0o644)
    with pytest.raises(ArtifactError):
        read_invitation(destination)


def test_tls_fixed_roles_local_keys_and_atomic_install(tmp_path):
    security, invitation = expert(tmp_path, True)
    client = SetupSecurity.create_client(tmp_path / "client", invitation)
    csr = client.client_csr()
    certificate = security.sign_client_csr(csr)
    issued = security.expected_client_leaf_digest()
    assert issued == hashlib.sha256(ssl.PEM_cert_to_DER_cert(certificate)).hexdigest()
    repeated = security.sign_client_csr(csr)
    assert repeated == certificate
    with pytest.raises((ArtifactError, ssl.SSLError)):
        client.install_client_certificate("invalid certificate")
    assert not (client.directory / "peer.pem").exists()
    client.install_client_certificate(certificate)
    client.install_client_certificate(certificate)
    assert client.peer_name("expert") == "ria-expert." + "1" * 32 + "." + "1" * 32
    assert client.peer_name("client") == "ria-client." + "1" * 32 + "." + "1" * 32
    assert not (client.directory / "ca.key").exists()
    assert (client.directory / "peer.key").read_bytes() != (
        security.directory / "peer.key"
    ).read_bytes()
    other = SetupSecurity.create_client(tmp_path / "other", invitation)
    with pytest.raises(ssl.SSLError):
        other.install_client_certificate(certificate)
    assert not (other.directory / "peer.pem").exists()
    with pytest.raises(ArtifactError, match="another client key"):
        security.sign_client_csr(other.client_csr())


def test_bad_csr_signature_is_not_issued(tmp_path):
    security, invitation = expert(tmp_path, True)
    with pytest.raises(ArtifactError, match="actually issued"):
        security.expected_client_leaf_digest()
    client = SetupSecurity.create_client(tmp_path / "client", invitation)
    lines = client.client_csr().splitlines()
    der = bytearray(base64.b64decode("".join(lines[1:-1])))
    der[-1] ^= 1
    encoded = base64.b64encode(der).decode()
    damaged = (
        lines[0]
        + "\n"
        + "\n".join(encoded[i : i + 64] for i in range(0, len(encoded), 64))
        + "\n"
        + lines[-1]
        + "\n"
    )
    with pytest.raises(ArtifactError):
        security.sign_client_csr(damaged)
    assert not list(security.directory.glob("client-*.pem"))


def test_journal_recovery_and_no_rerun(tmp_path):
    journal = SetupJournal(tmp_path / "journal", maximum=2)
    record, created = journal.accept("first", "probe", {})
    assert created and record["status"] == "accepted"
    journal.transition("first", "running")
    second, created = journal.accept("first", "probe", {})
    assert not created and second["status"] == "running"
    with pytest.raises(ArtifactError, match="different inputs"):
        journal.accept("first", "probe", {"changed": True})
    reopened = SetupJournal(journal.directory, maximum=2)
    assert reopened.interrupted == ["first"]
    assert reopened.status("first")["status"] == "failed"
    receipt, created = reopened.accept("first", "probe", {})
    assert not created and receipt["status"] == "failed"
    with pytest.raises(ArtifactError, match="transition"):
        reopened.transition("first", "running")
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in journal.directory.iterdir())


@pytest.mark.parametrize("tls", [False, True])
def test_actual_channel_and_worker_isolation(tmp_path, tls):
    with channel(tmp_path, tls=tls) as (server, client, security):
        result = client.call("health", {"test": "space Unicode λ"})
        assert result == {"name": "health", "payload": {"test": "space Unicode λ"}}
        status = Path("/proc") / str(server.process.pid) / "status"
        fields = dict(
            line.split(":", 1)
            for line in status.read_text().splitlines()
            if ":" in line
        )
        assert fields["Uid"].split() == ["10001"] * 4
        assert fields["Groups"].strip() == ""
        assert fields["NoNewPrivs"].strip() == "1"
        assert all(
            int(fields[name], 16) == 0
            for name in ("CapInh", "CapPrm", "CapEff", "CapAmb")
        )
        assert security.directory.stat().st_mode & 0o777 == 0o700
        with pytest.raises(ArtifactError):
            client.call("shell", {"command": "true"})


def test_not_ready_is_the_only_pollable_classification(tmp_path):
    def dispatcher(name, payload):
        if name == "offer":
            raise SetupNotReady("offer pending local preparation")
        raise ArtifactError("required operation failed")

    with channel(tmp_path, dispatcher) as (_, client, _):
        with pytest.raises(SetupNotReady):
            client.call("offer", {})
        with pytest.raises(ArtifactError) as failure:
            client.call("health", {})
        assert not isinstance(failure.value, SetupNotReady)


def test_mac_replay_and_mode_binding(tmp_path):
    calls = []
    with channel(tmp_path, lambda name, payload: calls.append(name) or {}) as (
        _,
        client,
        _,
    ):
        challenge = client._exchange("/challenge", {})["body"]
        request = {
            **challenge,
            "client_nonce": "2" * 64,
            "sequence": 1,
            "operation": "health",
            "payload": {},
        }
        packet = {
            "body": request,
            "mac": _mac(client.invitation, "client-request", request),
        }
        first = client._exchange("/rpc", packet)
        assert first["body"]["response"]["ok"] is True
        with pytest.raises(ArtifactError):
            client._exchange("/rpc", packet)
        for body in (
            {**request, "sequence": 2, "tls_enabled": True},
            {**request, "sequence": 2, "job_id": "3" * 64},
        ):
            with pytest.raises(ArtifactError):
                client._exchange(
                    "/rpc",
                    {
                        "body": body,
                        "mac": _mac(client.invitation, "client-request", body),
                    },
                )
        with pytest.raises(ArtifactError):
            client._exchange(
                "/rpc", {"body": {**request, "sequence": 2}, "mac": packet["mac"]}
            )
        assert calls == ["health"]


def test_async_task_same_id_never_reruns(tmp_path):
    entered, release, calls = threading.Event(), threading.Event(), []

    def dispatcher(name, payload):
        calls.append(name)
        entered.set()
        if not release.wait(3):
            raise ArtifactError("test release was absent")
        return {"proof": "receipt only"}

    with channel(tmp_path, dispatcher) as (server, client, _):
        accepted = client.submit("measured-stage", "native-fixture", {})
        assert accepted["status"] == "accepted"
        started = entered.wait(2)
        assert started
        try:
            repeated = client.submit("measured-stage", "native-fixture", {})
            assert repeated["status"] == "running"
            with pytest.raises(ArtifactError, match="another setup task"):
                client.submit("second-stage", "probe", {})
            with pytest.raises(ArtifactError):
                client.wait("measured-stage", timeout=0.1)
        finally:
            release.set()
        result = client.wait("measured-stage", timeout=2)
        assert result == {"proof": "receipt only"}
        repeated = client.submit("measured-stage", "native-fixture", {})
        assert repeated["status"] == "completed" and calls == ["native-fixture"]
        local = server.status("measured-stage")
        assert local["status"] == "completed"


def test_slow_header_deadline_preserves_following_requests(tmp_path):
    with channel(tmp_path, timeout=0.3) as (_, client, _):
        sock = socket.create_connection((client.host, client.port), timeout=1)
        try:
            sock.sendall(b"POST /challenge HTTP/1.0\r\nContent-Length: 2\r\n")
            time.sleep(0.45)
        finally:
            sock.close()
        result = client.call("health", {}, timeout=1)
        assert result["name"] == "health"


def test_late_controller_reply_retires_worker(tmp_path):
    entered = threading.Event()

    def dispatcher(name, payload):
        entered.set()
        time.sleep(0.6)
        return {"old": True}

    with pytest.raises(ArtifactError, match="in-flight"):
        with channel(tmp_path, dispatcher, timeout=0.2) as (server, client, _):
            with pytest.raises((ArtifactError, OSError, http.client.HTTPException)):
                client.call("health", {})
            started = entered.wait(1)
            assert started
            code = server.process.wait(timeout=1)
            assert code != 0


def tree(tmp_path):
    path = private(tmp_path / "source")
    (path / "nested").mkdir()
    (path / "nested" / "bytes.bin").write_bytes(b"x" * (CHUNK + 17))
    (path / "manifest.json").write_text('{"synthetic":true}')
    (path / "empty").touch()
    return path


def test_transfer_both_directions_and_exact_resume(tmp_path):
    source = tree(tmp_path)
    outputs = private(tmp_path / "outputs")
    with channel(tmp_path) as (server, client, _):
        manifest = server.export_tree("client-model", source, max_bytes=CHUNK + 100)
        fetched = client.fetch_tree(
            "client-model", outputs / "fetched", max_bytes=CHUNK + 100, timeout=3
        )
        assert fetched == manifest
        server.allow_import("client-runs", outputs / "proofs", max_bytes=CHUNK + 100)
        uploaded = client.upload_tree(
            "client-runs", source, max_bytes=CHUNK + 100, timeout=3
        )
        retried = client.upload_tree(
            "client-runs", source, max_bytes=CHUNK + 100, timeout=3
        )
        assert retried == uploaded
        assert (outputs / "proofs" / "nested" / "bytes.bin").read_bytes() == (
            source / "nested" / "bytes.bin"
        ).read_bytes()
        with pytest.raises(ArtifactError, match="authorized"):
            client.upload_tree("server-bank", source)
        (source / "empty").write_bytes(b"change")
        with pytest.raises(ArtifactError, match="immutable"):
            client.upload_tree("client-runs", source)


@pytest.mark.parametrize(
    "badpath",
    [
        "../escape",
        "/absolute",
        "a/../b",
        "a//b",
        "bad.key",
        "nested/secret.pem",
        "new\nline",
    ],
)
def test_manifest_rejects_paths_before_reader(tmp_path, badpath):
    destination = private(tmp_path / "destination")
    manifest = seal(
        {
            "schema_revision": 1,
            "name": "proofs",
            "total_bytes": 1,
            "files": [
                {
                    "path": badpath,
                    "bytes": 1,
                    "sha256": hashlib.sha256(b"x").hexdigest(),
                }
            ],
        }
    )

    def forbidden(*args):
        raise AssertionError("invalid manifest must fail before transfer")

    with pytest.raises(ArtifactError):
        import_tree(manifest, destination / "output", forbidden)
    assert not list(destination.iterdir())


def test_no_overwrite_mutation_quota_symlink_and_hash_failure(tmp_path):
    source, outputs = tree(tmp_path), private(tmp_path / "outputs")
    exported = ExportTree("model", source)
    with pytest.raises(ArtifactError, match="quota"):
        ExportTree("model", source, max_bytes=10)
    (source / "empty").write_bytes(b"changed")
    with pytest.raises(ArtifactError, match="changed"):
        exported.chunk("empty", 0, 1)
    (source / "link").symlink_to(source / "empty")
    with pytest.raises(ArtifactError, match="symlink"):
        ExportTree("model", source)
    (source / "link").unlink()
    exported = ExportTree("model", source)

    def corrupt(path, offset, length):
        return b"z" * length

    with pytest.raises(ArtifactError, match="hash"):
        import_tree(exported.manifest, outputs / "bad", corrupt)
    assert not list(outputs.iterdir())

    def conflict(path, offset, length):
        (outputs / "new").mkdir(exist_ok=True)
        return exported.chunk(path, offset, length)

    with pytest.raises(FileExistsError):
        import_tree(exported.manifest, outputs / "new", conflict)
    assert (outputs / "new").is_dir() and not list((outputs / "new").iterdir())


def test_request_bound_before_wire(tmp_path):
    with channel(tmp_path) as (_, client, _):
        with pytest.raises(ArtifactError, match="bound"):
            client.call("health", {"huge": "x" * MAX_MESSAGE})
        result = client.call("health", {})
        assert result["name"] == "health"


@pytest.mark.parametrize(
    "field,value",
    [
        ("job_id", None),
        ("pair_secret", 7),
        ("tls_enabled", 0),
        ("endpoint", []),
        ("ca_pem", "bad"),
    ],
)
def test_malformed_invitation_is_a_domain_failure(tmp_path, field, value):
    _, invitation = expert(tmp_path)
    with pytest.raises(ArtifactError):
        validate_invitation({**invitation, field: value})


@pytest.mark.parametrize("tls", [False, True])
def test_no_mode_fallback_or_pin_override(tmp_path, tls):
    with channel(tmp_path, tls=tls) as (_, client, _):
        if tls:
            wrong_pin = PeerClient(
                {**client.invitation, "server_leaf_sha256": "0" * 64}
            )
            with pytest.raises(ArtifactError, match="pin"):
                wrong_pin.call("health", {})
            invitation = {
                **client.invitation,
                "tls_enabled": False,
                "ca_pem": None,
                "server_leaf_sha256": None,
            }
        else:
            extra = private(tmp_path / "extra")
            _, tls_invitation = expert(extra, True)
            invitation = {
                **tls_invitation,
                "endpoint": client.invitation["endpoint"],
                "job_id": client.invitation["job_id"],
                "pair_secret": client.invitation["pair_secret"],
            }
        wrong_mode = PeerClient(invitation, timeout=0.5)
        with pytest.raises((ArtifactError, OSError, http.client.HTTPException)):
            wrong_mode.call("health", {})
        result = client.call("health", {})
        assert result["name"] == "health"


@pytest.mark.parametrize(
    "headers",
    [
        b"Content-Length: 2\r\nContent-Length: 2\r\n",
        b"Content-Length: 8388609\r\n",
        b"Content-Length: 2\r\nTransfer-Encoding: chunked\r\n",
        b"Content-Length: -1\r\n",
    ],
)
def test_malformed_http_framing_never_dispatches(tmp_path, headers):
    calls = []
    with channel(tmp_path, lambda name, payload: calls.append(name) or {}) as (
        _,
        client,
        _,
    ):
        with socket.create_connection((client.host, client.port), timeout=1) as sock:
            sock.sendall(b"POST /rpc HTTP/1.0\r\n" + headers + b"\r\n{}")
            response = sock.recv(4096)
        assert b" 400 " in response and not calls
        result = client.call("health", {})
        assert result == {} and calls == ["health"]


def test_response_mac_and_request_association_are_independent(tmp_path, monkeypatch):
    _, invitation = expert(tmp_path)
    client = PeerClient(invitation)
    body = {
        "job_id": invitation["job_id"],
        "tls_enabled": False,
        "server_nonce": "4" * 64,
    }

    def exchange(path, payload, **kwargs):
        if path == "/challenge":
            return {"body": body, "mac": _mac(invitation, "expert-challenge", body)}
        reply = {"request_digest": "0" * 64, "response": {"ok": True, "result": {}}}
        return {"body": reply, "mac": _mac(invitation, "expert-response", reply)}

    monkeypatch.setattr(client, "_exchange", exchange)
    with pytest.raises(ArtifactError, match="association"):
        client.call("health", {})

    def forged(path, payload, **kwargs):
        return {"body": body, "mac": "0" * 64}

    monkeypatch.setattr(client, "_exchange", forged)
    with pytest.raises(ArtifactError, match="authentication"):
        client.call("health", {})


def test_upload_lost_reply_chunks_and_crash_residue(tmp_path):
    source, outputs = tree(tmp_path), private(tmp_path / "outputs")
    security, invitation = expert(tmp_path)
    server = PeerServer(invitation, security, lambda *args: None)
    server.allow_import("client-runs", outputs / "runs")
    exported = ExportTree("client-runs", source)
    begin = {"name": "client-runs", "manifest": exported.manifest}
    first = server._operation("import-begin", begin)
    assert first["completed"] is False
    path = "nested/bytes.bin"
    raw = exported.chunk(path, 0, CHUNK)
    chunk = {
        "name": "client-runs",
        "digest": exported.manifest["digest"],
        "path": path,
        "offset": 0,
        "data": raw,
    }
    progress = server._operation("import-chunk", chunk)
    repeated = server._operation("import-chunk", chunk)
    assert repeated == progress and progress["offsets"][path] == CHUNK
    with pytest.raises(ArtifactError, match="different bytes"):
        server._operation("import-chunk", {**chunk, "data": b"y" * CHUNK})
    with pytest.raises(ArtifactError, match="incomplete"):
        server._operation(
            "import-finish",
            {"name": "client-runs", "digest": exported.manifest["digest"]},
        )
    assert not (outputs / "runs").exists()
    other = PeerServer(invitation, security, lambda *args: None)
    other.allow_import("client-runs", outputs / "runs")
    with pytest.raises(FileExistsError):
        other._operation("import-begin", begin)
    assert not (outputs / "runs").exists()


def test_close_retains_live_task_ownership(tmp_path):
    security, invitation = expert(tmp_path)
    entered, release = threading.Event(), threading.Event()

    def dispatcher(name, payload):
        entered.set()
        if not release.wait(2):
            raise ArtifactError("release missing")
        return None

    server = PeerServer(invitation, security, dispatcher)
    receipt = server.submit("owned", "probe", {})
    assert receipt["status"] == "accepted"
    started = entered.wait(1)
    assert started and server.tasks["owned"].daemon is False
    try:
        server.begin_close()
        with pytest.raises(ArtifactError, match="remains owned"):
            server.close(timeout=0.01)
        assert server.tasks["owned"].is_alive()
        with pytest.raises(ArtifactError, match="stopping"):
            server.submit("new", "probe", {})
    finally:
        release.set()
        server.close(timeout=2)
    assert not server.tasks["owned"].is_alive()
    final = server.status("owned")
    assert final["status"] == "completed" and final["result"] is None


def test_setup_client_absolute_deadline_on_slow_response(tmp_path):
    # This deliberately unauthenticated fake peer cannot make a slow stream reset
    # the total request budget by sending bytes before individual socket timeouts.
    _, invitation = expert(tmp_path)
    host, port = endpoint(invitation["endpoint"])
    ready, stop = threading.Event(), threading.Event()

    def fake():
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((host, port))
            listener.listen(1)
            listener.settimeout(1)
            ready.set()
            sock, _ = listener.accept()
            with sock:
                sock.recv(65536)
                sock.sendall(
                    (
                        "HTTP/1.0 200 OK\r\nContent-Length: 100\r\nContent-Type: "
                        + JSON_TYPE
                        + "\r\n\r\n"
                    ).encode()
                )
                try:
                    while not stop.wait(0.05):
                        sock.sendall(b" ")
                except OSError:
                    pass

    task = threading.Thread(target=fake)
    task.start()
    initialized = ready.wait(1)
    assert initialized
    try:
        client = PeerClient(invitation, timeout=0.2)
        start = time.monotonic()
        with pytest.raises((ArtifactError, OSError, http.client.HTTPException)):
            client.call("health", {})
        assert time.monotonic() - start < 0.8
    finally:
        stop.set()
        task.join(timeout=1)
    assert not task.is_alive()


def test_registration_cannot_follow_a_replaced_directory(tmp_path, monkeypatch):
    source = tree(tmp_path)
    foreign = private(tmp_path / "foreign")
    (foreign / "not-authorized").write_bytes(b"outside registered tree")
    original, changed = os.open, False

    def replacing(path, flags, *args, **kwargs):
        nonlocal changed
        if path == "nested" and flags & os.O_DIRECTORY and not changed:
            changed = True
            (source / "nested").rename(tmp_path / "old-nested")
            (source / "nested").symlink_to(foreign)
        return original(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", replacing)
    with pytest.raises(OSError):
        ExportTree("model", source)
    assert changed


@pytest.mark.parametrize("tls", [False, True])
def test_finish_response_drains_before_worker_shutdown(tmp_path, tls):
    finished = threading.Event()
    holder = {}

    def dispatcher(name, payload):
        finished.set()
        deadline = time.monotonic() + 1
        while not holder["server"].stopping and time.monotonic() < deadline:
            time.sleep(0.001)
        if not holder["server"].stopping:
            raise ArtifactError("shutdown did not race the unfinished response")
        return {"finish": "complete", "bytes": "x" * 65536}

    with channel(tmp_path, dispatcher, tls=tls) as (server, client, _):
        holder["server"] = server
        errors = []

        def closer():
            if not finished.wait(1):
                errors.append("dispatch absent")
                return
            server.begin_close()

        task = threading.Thread(target=closer)
        task.start()
        try:
            response = client.call("finish", {})
            assert response["finish"] == "complete" and len(response["bytes"]) == 65536
        finally:
            task.join(timeout=3)
        assert not errors and not task.is_alive() and server.reply_failure is None


def test_export_hash_growth_cancellation_and_hardlink_bounds(tmp_path):
    source = private(tmp_path / "source")
    leaf = source / "data"
    leaf.write_bytes(b"x" * (CHUNK + 9))
    calls = 0

    def growing():
        nonlocal calls
        calls += 1
        if calls == 3:  # directory, entry, then the first bounded hash chunk
            with leaf.open("ab") as file:
                file.write(b"growth")

    with pytest.raises(ArtifactError, match="grew"):
        ExportTree("reports", source, check=growing)
    assert calls == 5  # Read the original two chunks, never chase appended bytes.
    os.link(leaf, source / "hardlink")
    with pytest.raises(ArtifactError, match="hardlinks"):
        ExportTree("reports", source, reject_hardlinks=True)
    (source / "hardlink").unlink()
    exported = ExportTree("reports", source)
    destination = private(tmp_path / "output")
    checks = 0

    def cancelled():
        nonlocal checks
        checks += 1
        if checks == 3:
            raise ArtifactError("cancelled snapshot")

    with pytest.raises(ArtifactError, match="cancelled snapshot"):
        import_tree(
            exported.manifest, destination / "tree", exported.chunk, check=cancelled
        )
    assert not list(destination.iterdir())


def test_task_failure_preserves_primary_when_failed_receipt_cannot_commit(
    tmp_path, monkeypatch
):
    security, invitation = expert(tmp_path)

    def fail(*args):
        raise ArtifactError("primary operation")

    server = PeerServer(invitation, security, fail)
    original = server.journal.transition

    def transition(step, state, **kwargs):
        if state == "failed":
            raise OSError("receipt storage exhausted")
        return original(step, state, **kwargs)

    monkeypatch.setattr(server.journal, "transition", transition)
    receipt = server.submit("failure", "probe", {})
    assert receipt["status"] == "accepted"
    server.tasks["failure"].join(timeout=1)
    assert not server.tasks["failure"].is_alive()
    with pytest.raises(
        ArtifactError,
        match="primary operation; failed journal commit: receipt storage exhausted",
    ):
        server.status("failure")
    server.close()


@pytest.mark.parametrize("tls", [False, True])
def test_lost_final_reply_is_reported_after_broker_join(tmp_path, tls):
    if os.geteuid() != 0:
        pytest.skip("actual isolated UID10001 worker needs root")
    security, invitation = expert(tmp_path, tls)
    entered, release = threading.Event(), threading.Event()

    def finish(name, payload):
        entered.set()
        if not release.wait(2):
            raise ArtifactError("release absent")
        return {"finished": True}

    server = PeerServer(invitation, security, finish, timeout=1).start()
    client = PeerClient(invitation, timeout=1)
    errors = []

    def call():
        try:
            client.call("finish", {})
        except (ArtifactError, OSError, http.client.HTTPException) as error:
            errors.append(error)

    caller = threading.Thread(target=call)
    caller.start()
    try:
        dispatched = entered.wait(1)
        assert dispatched
        server.process.kill()  # Real worker loss before the final HTTP write/ACK.
        code = server.process.wait(timeout=1)
        assert code != 0
    finally:
        release.set()
        caller.join(timeout=2)
    assert errors and not caller.is_alive()
    with pytest.raises(ArtifactError, match="lost an in-flight authenticated response"):
        server.close(timeout=2)
    assert not server.thread.is_alive()


def test_response_drain_timeout_retains_failure_and_owned_broker(tmp_path):
    if os.geteuid() != 0:
        pytest.skip("actual isolated UID10001 worker needs root")
    security, invitation = expert(tmp_path)
    entered, release = threading.Event(), threading.Event()

    def finish(name, payload):
        entered.set()
        if not release.wait(3):
            raise ArtifactError("release absent")
        return {}

    server = PeerServer(invitation, security, finish, timeout=0.1).start()
    client = PeerClient(invitation, timeout=0.2)
    errors = []

    def call():
        try:
            client.call("finish", {})
        except (ArtifactError, OSError, http.client.HTTPException) as error:
            errors.append(error)

    caller = threading.Thread(target=call)
    caller.start()
    try:
        dispatched = entered.wait(1)
        assert dispatched
        server.begin_close()
        assert "drain deadline" in server.reply_failure and server.thread.is_alive()
        with pytest.raises(ArtifactError, match="remains owned"):
            server.close(timeout=0.01)
    finally:
        release.set()
        caller.join(timeout=1)
    with pytest.raises(ArtifactError, match="drain deadline"):
        server.close(timeout=2)
    assert errors and not caller.is_alive() and not server.thread.is_alive()


@pytest.mark.parametrize(
    "kind,length,tail,metadata",
    [
        (2, 1, 0, b"{}"),
        (0, MAX_MESSAGE + 1, 0, b""),
        (0, 2, 1, b"{}"),
        (1, MAX_CHUNK_METADATA + 1, 0, b""),
        (1, 2, CHUNK + 1, b"{}"),
        (1, 2, 1, b"{}"),
    ],
)
def test_broker_rejects_header_before_raw_allocation(kind, length, tail, metadata):
    left, right = socket.socketpair()
    with left, right:
        left.sendall(BROKER_HEADER.pack(1, kind, length, tail) + metadata)
        left.shutdown(socket.SHUT_WR)
        with pytest.raises(ArtifactError):
            _receive(right)


def test_binary_broker_exact_bytes_and_truncated_hash(tmp_path):
    request = {"operation": "import-chunk", "payload": {"data": b"abc"}}
    left, right = socket.socketpair()
    with left, right:
        _send(left, request)
        restored = _receive(right)
    assert restored == request
    descriptor = {
        "schema_revision": 1,
        "bytes": 3,
        "sha256": hashlib.sha256(b"abc").hexdigest(),
    }
    metadata = canonical({"operation": "import-chunk", "payload": {"data": descriptor}})
    for raw in (b"abx", b"ab"):
        left, right = socket.socketpair()
        with left, right:
            left.sendall(BROKER_HEADER.pack(1, 1, len(metadata), 3) + metadata + raw)
            left.shutdown(socket.SHUT_WR)
            with pytest.raises((ArtifactError, EOFError)):
                _receive(right)


@pytest.mark.parametrize("tls", [False, True])
@pytest.mark.parametrize(
    "fault",
    [
        "hash",
        "mac",
        "mode",
        "offset",
        "path",
        "metadata",
        "version",
        "replay",
        "truncated",
    ],
)
def test_raw_http_rejects_adversarial_import_before_write(tmp_path, tls, fault):
    outputs = private(tmp_path / "outputs")
    source = private(tmp_path / "source")
    (source / "data").write_bytes(b"abc")
    with channel(tmp_path, tls=tls) as (server, client, _):
        server.allow_import("client-runs", outputs / "proofs", max_bytes=3)
        manifest = ExportTree("client-runs", source).manifest
        begun = client.call(
            "import-begin", {"name": "client-runs", "manifest": manifest}
        )
        assert begun["offsets"]["data"] == 0
        body = {
            **client.challenge,
            "client_nonce": client.nonce,
            "sequence": client.sequence + 1,
            "operation": "import-chunk",
            "payload": {
                "name": "client-runs",
                "digest": manifest["digest"],
                "path": "data",
                "offset": 0,
                "data": {
                    "schema_revision": 1,
                    "bytes": 3,
                    "sha256": hashlib.sha256(b"abc").hexdigest(),
                },
            },
        }
        if fault == "mode":
            body["tls_enabled"] = not tls
        elif fault == "offset":
            body["payload"]["offset"] = 1
        elif fault == "path":
            body["payload"]["path"] = "../escape"
        elif fault == "replay":
            body["sequence"] = client.sequence
        metadata = canonical(
            {
                "body": body,
                "mac": "0" * 64
                if fault == "mac"
                else _mac(client.invitation, "client-request", body),
            }
        )
        raw = b"abx" if fault == "hash" else b"abc"
        encoded = (
            struct.pack(
                "!I", MAX_CHUNK_METADATA + 1 if fault == "metadata" else len(metadata)
            )
            + metadata
            + raw
        )
        connection = client._connection(1)
        try:
            if fault == "truncated":
                connection.connect()
                header = (
                    "POST /rpc HTTP/1.0\r\nContent-Type: "
                    + BINARY_TYPE
                    + "\r\nContent-Length: "
                    + str(len(encoded))
                    + "\r\n\r\n"
                ).encode()
                connection.sock.sendall(header + encoded[:-1])
                connection.sock.shutdown(socket.SHUT_WR)
                response = connection.getresponse()
            else:
                connection.request(
                    "POST",
                    "/rpc",
                    body=encoded,
                    headers={
                        "Content-Type": BINARY_TYPE
                        if fault != "version"
                        else "application/vnd.ria.setup.chunk;version=2",
                        "Connection": "close",
                    },
                )
                response = connection.getresponse()
            reply = response.read()
            assert response.status == 400 or b'"ok":false' in reply
        except (OSError, http.client.HTTPException):
            if fault != "truncated":
                raise
        finally:
            connection.close()
        upload = server.imports["client-runs"]["upload"]
        assert (
            upload.offsets["data"] == 0
            and (upload.staging / "data").stat().st_size == 0
        )
        assert not (outputs / "proofs").exists()


@pytest.mark.parametrize("tls", [False, True])
@pytest.mark.parametrize(
    "fault",
    ["hash", "mac", "association", "length", "metadata", "version", "truncated"],
)
def test_binary_response_authentication_and_bounds(tmp_path, tls, fault):
    security, invitation = expert(tmp_path, tls)
    host, port = endpoint(invitation["endpoint"])
    ready = threading.Event()
    failures = []

    def fake():
        try:
            with socket.socket() as listener:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind((host, port))
                listener.listen(1)
                listener.settimeout(2)
                ready.set()
                raw_socket, _ = listener.accept()
                if tls:
                    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                    context.load_cert_chain(
                        security.paths["certificate_file"],
                        security.paths["private_key_file"],
                    )
                    sock = context.wrap_socket(raw_socket, server_side=True)
                else:
                    sock = raw_socket
                with sock, sock.makefile("rb") as input_stream:
                    request_line = input_stream.readline(8193)
                    if request_line != b"POST /rpc HTTP/1.1\r\n":
                        raise AssertionError("unexpected bounded fixture endpoint")
                    headers = {}
                    while True:
                        line = input_stream.readline(8193)
                        if line == b"\r\n":
                            break
                        if not line or len(line) > 8192:
                            raise AssertionError("fixture header malformed")
                        key, value = line.decode().split(":", 1)
                        headers[key.lower()] = value.strip()
                    length = int(headers["content-length"])
                    if not 0 < length <= 8192:
                        raise AssertionError("fixture request out of bounds")
                    request = loads(input_stream.read(length))
                    descriptor = {
                        "schema_revision": 1,
                        "bytes": 3,
                        "sha256": hashlib.sha256(b"abc").hexdigest(),
                    }
                    body = {
                        "request_digest": "0" * 64
                        if fault == "association"
                        else digest(request["body"]),
                        "response": {
                            "ok": True,
                            "result": {
                                "data": descriptor,
                                "sha256": descriptor["sha256"],
                            },
                        },
                    }
                    metadata = canonical(
                        {
                            "body": body,
                            "mac": "0" * 64
                            if fault == "mac"
                            else _mac(invitation, "expert-response", body),
                        }
                    )
                    tail = b"abx" if fault == "hash" else b"abc"
                    if fault == "truncated":
                        tail = b"ab"
                    encoded = (
                        struct.pack(
                            "!I",
                            MAX_CHUNK_METADATA + 1
                            if fault == "metadata"
                            else len(metadata),
                        )
                        + metadata
                        + tail
                    )
                    declared = 4 + len(metadata) + 3 + (1 if fault == "length" else 0)
                    content_type = (
                        BINARY_TYPE
                        if fault != "version"
                        else "application/vnd.ria.setup.chunk;version=2"
                    )
                    sock.sendall(
                        (
                            "HTTP/1.0 200 OK\r\nContent-Type: "
                            + content_type
                            + "\r\nContent-Length: "
                            + str(declared)
                            + "\r\n\r\n"
                        ).encode()
                        + encoded
                    )
        except Exception as error:
            failures.append(error)

    task = threading.Thread(target=fake)
    task.start()
    initialized = ready.wait(1)
    assert initialized
    client = PeerClient(invitation, timeout=0.5)
    client.challenge = {
        "job_id": invitation["job_id"],
        "tls_enabled": tls,
        "server_nonce": "5" * 64,
    }
    try:
        with pytest.raises((ArtifactError, OSError, http.client.HTTPException)):
            client.call(
                "export-chunk",
                {"name": "client-model", "path": "data", "offset": 0, "length": 3},
            )
    finally:
        task.join(timeout=2)
    assert not task.is_alive() and not failures and client.transport is None


@pytest.mark.parametrize("tls", [False, True])
def test_connection_burst_is_fixed_four_responses(tmp_path, tls):
    with channel(tmp_path, tls=tls) as (_, client, _):
        connection = client._connection(1)
        sockets = []
        try:
            for index in range(4):
                connection.request(
                    "POST",
                    "/challenge",
                    body=b"{}",
                    headers={"Content-Type": JSON_TYPE, "Connection": "keep-alive"},
                )
                response = connection.getresponse()
                packet = loads(response.read())
                assert (
                    response.status == 200
                    and packet["body"]["job_id"] == client.invitation["job_id"]
                )
                if index < 3:
                    sockets.append(connection.sock)
                    assert not response.will_close and connection.sock is not None
                else:
                    assert response.will_close and connection.sock is None
            assert sockets[0] is sockets[1] is sockets[2]
        finally:
            connection.close()


@pytest.mark.parametrize("tls", [False, True])
def test_cached_challenge_never_refreshes_after_worker_restart(tmp_path, tls):
    if os.geteuid() != 0:
        pytest.skip("actual isolated UID10001 worker needs root")
    security, invitation = expert(tmp_path, tls)
    calls = []

    def dispatch(name, payload):
        calls.append(name)
        return {}

    first = PeerServer(invitation, security, dispatch).start()
    client = PeerClient(invitation, timeout=1)
    try:
        result = client.call("health", {})
        nonce = client.challenge["server_nonce"]
        assert result == {}
    finally:
        first.close()
    second = PeerServer(invitation, security, dispatch).start()
    try:
        with pytest.raises(ArtifactError, match="rejected"):
            client.call("health", {})
        assert client.challenge["server_nonce"] == nonce and calls == ["health"]
        fresh = PeerClient(invitation, timeout=1)
        result = fresh.call("health", {})
        assert result == {} and fresh.challenge["server_nonce"] != nonce
        assert calls == ["health", "health"]
    finally:
        second.close()


@pytest.mark.parametrize("direction", ["fetch", "upload"])
def test_local_transfer_hash_and_publication_share_absolute_deadline(
    tmp_path, monkeypatch, direction
):
    from ria import setup_peer

    source = tree(tmp_path)
    destination = private(tmp_path / "output")
    _, invitation = expert(tmp_path)
    client = PeerClient(invitation)
    manifest = ExportTree("client-model", source).manifest
    clock = [10.0]
    monkeypatch.setattr(setup_peer.time, "monotonic", lambda: clock[0])
    if direction == "upload":
        original = setup_peer.export_tree

        def expired_hash(*args, **kwargs):
            clock[0] = 12.0
            return original(*args, **kwargs)

        monkeypatch.setattr(setup_peer, "export_tree", expired_hash)

        def no_wire(*args, **kwargs):
            raise AssertionError("expired local registration must not reach network")

        monkeypatch.setattr(client, "call", no_wire)
        with pytest.raises(ArtifactError, match="transfer deadline"):
            client._upload_tree(
                "client-model", source, max_bytes=CHUNK + 100, deadline=11.0
            )
    else:
        exported = ExportTree("client-model", source)

        def reply(operation, payload, **kwargs):
            if operation == "export-manifest":
                return manifest
            data = exported.chunk(payload["path"], payload["offset"], payload["length"])
            clock[0] = 12.0
            return {"data": data}

        monkeypatch.setattr(client, "call", reply)
        with pytest.raises(ArtifactError, match="transfer deadline"):
            client._fetch_tree(
                "client-model",
                destination / "tree",
                max_bytes=CHUNK + 100,
                deadline=11.0,
            )
    assert not list(destination.iterdir())


@pytest.mark.parametrize("tls", [False, True])
def test_health_detects_worker_death_and_failed_fixed_stage(tmp_path, tls):
    if os.geteuid() != 0:
        pytest.skip("actual isolated UID10001 worker needs root")
    security, invitation = expert(tmp_path, tls)

    def failed(name, payload):
        raise ArtifactError("fixed stage cannot recover")

    server = PeerServer(invitation, security, failed).start()
    try:
        server.check_health()
        receipt = server.submit("broken", "probe", {})
        assert receipt["status"] == "accepted"
        server.tasks["broken"].join(timeout=1)
        assert not server.tasks["broken"].is_alive()
        with pytest.raises(
            ArtifactError, match="stage broken failed: fixed stage cannot recover"
        ):
            server.check_health()
        server.process.kill()
        code = server.process.wait(timeout=1)
        assert code != 0
        with pytest.raises(ArtifactError, match="worker exited|stopping"):
            server.check_health()
    finally:
        server.close()


def test_tcp_and_tls_establishment_consume_one_absolute_budget(tmp_path, monkeypatch):
    from ria import setup_peer

    _, invitation = expert(tmp_path, True)
    client = PeerClient(invitation, timeout=1)
    clock = [10.0]
    stages = []
    monkeypatch.setattr(setup_peer.time, "monotonic", lambda: clock[0])

    class SlowSocket:
        timeout = None

        def settimeout(self, timeout):
            self.timeout = timeout

        def setsockopt(self, *args):
            pass

        def connect(self, endpoint):
            stages.append(("tcp", self.timeout))
            clock[0] += 0.6

        def do_handshake(self):
            stages.append(("tls", self.timeout))
            cost = 0.6
            clock[0] += min(cost, self.timeout)
            if cost >= self.timeout:
                raise TimeoutError("controlled TLS budget exhausted")

        def shutdown(self, *args):
            pass

        def close(self):
            pass

    class SlowTLS:
        def wrap_socket(self, sock, *, server_hostname, do_handshake_on_connect=True):
            if do_handshake_on_connect:
                sock.do_handshake()
            return sock

    monkeypatch.setattr(
        setup_peer.socket, "socket", lambda *args, **kwargs: SlowSocket()
    )
    client.ssl_context = SlowTLS()
    with pytest.raises(TimeoutError, match="controlled TLS"):
        client._exchange("/challenge", {}, deadline=11.0)
    assert stages[0] == ("tcp", 1.0) and stages[1][0] == "tls"
    assert stages[1][1] == pytest.approx(0.4) and clock[0] == pytest.approx(11.0)


@pytest.mark.parametrize("failure", ["deadline", "cancelled"])
def test_upload_final_hash_observes_owned_budget_between_chunks(
    tmp_path, monkeypatch, failure
):
    from ria import setup_peer

    source = private(tmp_path / "source")
    (source / "data").write_bytes(b"x" * (CHUNK + 17))
    outputs = private(tmp_path / "outputs")
    security, invitation = expert(tmp_path)
    server = PeerServer(invitation, security, lambda *args: None)
    cancel = threading.Event()

    def check():
        if cancel.is_set():
            raise ArtifactError("owned setup cancelled")

    server.allow_import("client-runs", outputs / "published", check=check)
    exported = ExportTree("client-runs", source)
    begun = server._operation(
        "import-begin", {"name": "client-runs", "manifest": exported.manifest}
    )
    assert begun["completed"] is False
    for offset in (0, CHUNK):
        data = exported.chunk("data", offset, min(CHUNK, CHUNK + 17 - offset))
        progress = server._operation(
            "import-chunk",
            {
                "name": "client-runs",
                "digest": exported.manifest["digest"],
                "path": "data",
                "offset": offset,
                "data": data,
            },
        )
        assert progress["offsets"]["data"] == offset + len(data)
    original = setup_peer.open_regular
    reads = []

    @contextlib.contextmanager
    def interrupted(path):
        with original(path) as stream:

            class File:
                def fileno(self):
                    return stream.fileno()

                def read(self, length):
                    data = stream.read(length)
                    reads.append(len(data))
                    if failure == "deadline":
                        server._rpc_deadline = time.monotonic() - 1
                    else:
                        cancel.set()
                    return data

            yield File()

    monkeypatch.setattr(setup_peer, "open_regular", interrupted)
    with pytest.raises(ArtifactError, match="deadline|cancelled"):
        server._operation(
            "import-finish",
            {"name": "client-runs", "digest": exported.manifest["digest"]},
        )
    assert reads == [CHUNK] and not (outputs / "published").exists()
    assert server.imports["client-runs"]["upload"].completed is False
    server.close()


def test_task_inputs_are_owned_at_acceptance(tmp_path):
    security, invitation = expert(tmp_path)
    entered, release = threading.Event(), threading.Event()

    def dispatch(name, payload):
        entered.set()
        if not release.wait(1):
            raise ArtifactError("fixture release absent")
        return payload["values"]

    server = PeerServer(invitation, security, dispatch)
    payload = {"values": ["accepted"]}
    try:
        receipt = server.submit("immutable", "probe", payload)
        entered_task = entered.wait(1)
        assert entered_task
        payload["values"][0] = "caller mutation"
    finally:
        release.set()
        server.close(timeout=2)
    result = server.status("immutable")
    assert (
        result["result"] == ["accepted"]
        and result["input_digest"] == receipt["input_digest"]
    )
    assert result["input_digest"] == digest(
        {"name": "probe", "payload": {"values": ["accepted"]}}
    )


def test_export_publication_rechecks_concurrent_population(tmp_path, monkeypatch):
    from ria import setup_peer

    security, invitation = expert(tmp_path)
    source = private(tmp_path / "source")
    (source / "data").write_bytes(b"fixed")
    server = PeerServer(invitation, security, lambda *args: None)
    for i in range(15):
        server.export_tree("existing" + str(i), source)
    constructed = threading.Barrier(2)
    original = setup_peer.export_tree

    def simultaneous(*args, **kwargs):
        tree = original(*args, **kwargs)
        constructed.wait(timeout=1)
        return tree

    monkeypatch.setattr(setup_peer, "export_tree", simultaneous)
    successes, errors = [], []

    def register(name):
        try:
            successes.append(server.export_tree(name, source))
        except ArtifactError as error:
            errors.append(str(error))

    threads = [
        threading.Thread(target=register, args=(name,)) for name in ("one", "two")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
    assert all(not thread.is_alive() for thread in threads)
    assert len(successes) == 1 and errors == ["setup export population exhausted"]
    with server.mutex:
        assert len(server.exports) == 16
    with pytest.raises(ArtifactError, match="unknown locally registered") as error:
        server._operation("export-manifest", {"name": "privatecredential" * 256})
    assert "privatecredential" not in str(error.value)
    server.close()


@pytest.mark.parametrize("tls", [False, True])
def test_verified_http_tail_is_attached_once_inside_client_operation(
    tmp_path, monkeypatch, tls
):
    from ria import setup_peer

    source = private(tmp_path / "source")
    (source / "data").write_bytes(b"independent exact-byte oracle")
    with channel(tmp_path, tls=tls) as (server, client, _):
        server.export_tree("client-model", source)
        original = setup_peer._validate_raw
        validated = []

        def validate(value, raw):
            original(value, raw)
            validated.append(raw)

        monkeypatch.setattr(setup_peer, "_validate_raw", validate)
        reply = client.call(
            "export-chunk",
            {"name": "client-model", "path": "data", "offset": 0, "length": 29},
        )
        assert reply["data"] == (source / "data").read_bytes()
        assert reply["sha256"] == hashlib.sha256(reply["data"]).hexdigest()
        assert validated == [reply["data"]]


def test_close_joins_all_tasks_despite_an_unstarted_thread(tmp_path):
    security, invitation = expert(tmp_path)
    server = PeerServer(invitation, security, lambda *args: None)
    entered, release = threading.Event(), threading.Event()

    def owned():
        entered.set()
        release.wait(2)

    never_started = threading.Thread(target=lambda: None)
    active = threading.Thread(target=owned)
    active.start()
    began = entered.wait(1)
    assert began
    with server.mutex:
        server.tasks.update(unstarted=never_started, active=active)
    try:
        with pytest.raises(ArtifactError, match="remains owned.*task join"):
            server.close(timeout=0.01)
        assert active.is_alive()
    finally:
        release.set()
        with pytest.raises(ArtifactError, match="task join"):
            server.close(timeout=1)
    assert not active.is_alive()


def test_live_broker_and_contended_snapshot_do_not_hang_cleanup(tmp_path):
    security, invitation = expert(tmp_path)
    server = PeerServer(invitation, security, lambda *args: None)
    release = threading.Event()
    broker = threading.Thread(target=lambda: release.wait(2))
    broker.start()
    server.thread = broker
    server.mutex.acquire()
    try:
        before = time.monotonic()
        with pytest.raises(ArtifactError, match="broker work remains owned"):
            server.close(timeout=0.01)
        assert time.monotonic() - before < 0.3
        with pytest.raises(ArtifactError, match="snapshot unavailable"):
            server.pending_tasks(timeout=0.01)
    finally:
        server.mutex.release()
        release.set()
        broker.join(timeout=1)
        server.close()
    assert not broker.is_alive()


def test_failed_thread_start_has_failed_durable_stage(tmp_path, monkeypatch):
    security, invitation = expert(tmp_path)
    server = PeerServer(invitation, security, lambda *args: None)

    def exhausted(thread):
        raise RuntimeError("thread resource exhaustion")

    monkeypatch.setattr(threading.Thread, "start", exhausted)
    with pytest.raises(
        ArtifactError, match="could not start: thread resource exhaustion"
    ):
        server.submit("not-started", "probe", {})
    with pytest.raises(
        ArtifactError, match="durable failure.*thread resource exhaustion"
    ):
        server.status("not-started")
    with pytest.raises(ArtifactError, match="stage not-started failed"):
        server.journal.check_health()
    tasks = server.pending_tasks()
    assert len(tasks) == 1 and not tasks[0].is_alive()
    with pytest.raises(ArtifactError, match="task join"):
        server.close()
