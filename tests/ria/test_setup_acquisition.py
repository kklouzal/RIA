"""All HTTP is synthetic; no publisher/model/weight endpoint is contacted."""

from contextlib import contextmanager
from copy import deepcopy
from email.message import Message
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from ria import setup_acquisition as acquisition
from ria.identity import ArtifactError, canonical, read_json, seal


class Response:
    def __init__(self, body, *, status=200, headers=None):
        self.stream, self.status = io.BytesIO(body), status
        self.headers = Message()
        for key, value in (headers or {}).items():
            self.headers.add_header(key, value)

    def read(self, count):
        return self.stream.read(count)


@pytest.fixture
def publisher(tmp_path, monkeypatch):
    bodies = {"config.json": b'{"synthetic":true}', "model.safetensors.index.json": b'{"weight_map":{"x":"model-01.safetensors"}}',
              "model-01.safetensors": b"tiny synthetic file, not a model"}
    files = []
    siblings = []
    for name, data in bodies.items():
        lfs = hashlib.sha256(data).hexdigest() if name != "config.json" else None
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        files.append({"path": name, "bytes": len(data), "blob_id": blob, "lfs_sha256": lfs, "sha256": hashlib.sha256(data).hexdigest()})
        peer = {"rfilename": name, "size": len(data), "blobId": blob}
        if lfs:
            peer["lfs"] = {"sha256": lfs, "size": len(data)}
        siblings.append(peer)
    native = tmp_path / "native"
    native.mkdir()
    token = native / "tokenizer.json"
    token.write_bytes(b'{"synthetic":true}')
    metadata = [{"path": token.name, "bytes": token.stat().st_size, "sha256": hashlib.sha256(token.read_bytes()).hexdigest()}]
    plan = seal({"schema_revision": 1, "repository": acquisition.NVIDIA, "revision": acquisition.NVIDIA_REVISION,
                 "files": sorted(files, key=lambda item: item["path"]), "native_metadata": metadata,
                 "total_bytes": sum(map(len, bodies.values())) + token.stat().st_size})
    monkeypatch.setattr(acquisition, "acquisition_plan", lambda: plan)
    monkeypatch.setattr(acquisition, "_locked_metadata", lambda repository, revision, name: native / name)
    remote = {"id": acquisition.NVIDIA, "sha": acquisition.NVIDIA_REVISION, "siblings": siblings}
    calls = []
    @contextmanager
    def request(url, headers, deadline, clock):
        calls.append((url, dict(headers)))
        if "/api/models/" in url:
            yield Response(canonical(remote))
        else:
            name = url.rsplit("/", 1)[1]
            assert name in bodies
            offset = int(headers.get("Range", "bytes=0-")[6:-1])
            framing = {"Content-Length": str(len(bodies[name]) - offset)}
            if offset:
                framing["Content-Range"] = f"bytes {offset}-{len(bodies[name]) - 1}/{len(bodies[name])}"
            yield Response(bodies[name][offset:], status=206 if offset else 200, headers=framing)
    return plan, remote, bodies, calls, request


def test_pinned_plan_exact_complete_population_without_network():
    plan = acquisition.acquisition_plan()
    assert plan["repository"] == acquisition.NVIDIA and plan["revision"] == acquisition.NVIDIA_REVISION
    assert len([item for item in plan["files"] if item["path"].endswith(".safetensors")]) == 48
    assert sum(item["bytes"] for item in plan["files"] if item["path"].endswith(".safetensors")) == 527293384576
    assert {item["path"] for item in plan["native_metadata"]} == {"tokenizer.json", "tokenizer_config.json", "chat_template.jinja"}


def test_bounded_download_authentication_manifest_last_restart(tmp_path, publisher):
    plan, _, bodies, calls, request = publisher
    workspace = tmp_path / "work"
    facts = acquisition._acquire(workspace, plan["total_bytes"], 30000, request=request)
    root = Path(facts["source_dir"])
    assert read_json(root / "source-acquisition.json") == facts
    assert set(item["path"] for item in facts["files"]) == set(bodies)
    assert all((root / name).read_bytes() == data for name, data in bodies.items())
    assert all((root / name).stat().st_mode & 0o777 == 0o600 for name in bodies)
    count = len(calls)
    assert acquisition._acquire(workspace, plan["total_bytes"], 30000, request=request) == facts
    assert len(calls) == count + 1  # Metadata revalidated; complete files rehashed locally.
    assert not list((root / ".download").iterdir())


def test_resume_strict_range_and_complete_identity(tmp_path, publisher):
    plan, _, bodies, calls, request = publisher
    workspace = acquisition.private_directory(tmp_path / "work")
    root = acquisition._marked_directory(workspace, "source-" + acquisition.NVIDIA_REVISION, plan["digest"], modes=(0o700,))
    partial = acquisition.private_directory(root / ".download")
    name = "model-01.safetensors"
    path = partial / (hashlib.sha256(name.encode()).hexdigest() + ".partial")
    path.write_bytes(bodies[name][:7])
    path.chmod(0o600)
    acquisition._acquire(workspace, plan["total_bytes"], 30000, request=request)
    assert any(headers.get("Range") == "bytes=7-" for _, headers in calls)
    assert (root / name).read_bytes() == bodies[name] and not path.exists()


def test_reconcile_crash_between_link_and_unlink(tmp_path, publisher):
    plan, _, bodies, _, request = publisher
    workspace = acquisition.private_directory(tmp_path / "work")
    root = acquisition._marked_directory(workspace, "source-" + acquisition.NVIDIA_REVISION, plan["digest"], modes=(0o700,))
    partial = acquisition.private_directory(root / ".download")
    name = "model-01.safetensors"
    path = partial / (hashlib.sha256(name.encode()).hexdigest() + ".partial")
    path.write_bytes(bodies[name])
    path.chmod(0o600)
    os.link(path, root / name)
    acquisition._acquire(workspace, plan["total_bytes"], 30000, request=request)
    assert not path.exists() and (root / name).stat().st_nlink == 1


@pytest.mark.parametrize("mutation", ["revision", "size", "blob", "sha", "duplicate", "invalid_lfs"])
def test_remote_metadata_discrepancy_before_owned_output(tmp_path, publisher, mutation):
    plan, remote, _, _, request = publisher
    if mutation == "revision":
        remote["sha"] = "a" * 40
    elif mutation == "duplicate":
        remote["siblings"].append(deepcopy(remote["siblings"][0]))
    elif mutation == "size":
        remote["siblings"][0]["size"] += 1
    elif mutation == "blob":
        remote["siblings"][0]["blobId"] = "a" * 40
    elif mutation == "sha":
        remote["siblings"][1]["lfs"]["sha256"] = "f" * 64
    else:
        remote["siblings"][0]["lfs"] = False
    with pytest.raises(ArtifactError, match="metadata|identity"):
        acquisition._acquire(tmp_path / "work", plan["total_bytes"], 30000, request=request)
    assert not (tmp_path / "work").exists()


def test_byte_limit_before_any_http(tmp_path, publisher):
    plan, _, _, calls, request = publisher
    with pytest.raises(ArtifactError, match="byte limit"):
        acquisition._acquire(tmp_path / "work", plan["total_bytes"] - 1, 30000, request=request)
    assert not calls and not (tmp_path / "work").exists()


def test_corrupt_owned_partial_deleted_without_publishing_manifest(tmp_path, publisher):
    plan, _, bodies, _, request = publisher
    bodies["config.json"] = b"x" * len(bodies["config.json"])
    with pytest.raises(ArtifactError, match="publisher content identity"):
        acquisition._acquire(tmp_path / "work", plan["total_bytes"], 30000, request=request)
    root = tmp_path / "work" / ("source-" + acquisition.NVIDIA_REVISION)
    assert not (root / "source-acquisition.json").exists()
    assert not (root / "config.json").exists()
    assert not list((root / ".download").iterdir())


@pytest.mark.parametrize("header,value", [("Content-Length", "01"), ("Content-Encoding", "gzip"),
    ("Transfer-Encoding", "chunked"), ("Content-Range", "bytes 0-3/4")])
def test_http_framing_rejects_inexact_bodies(header, value):
    response = Response(b"abcd", headers={"Content-Length": "4", header: value})
    with pytest.raises(ArtifactError, match="HTTP"):
        acquisition._framing(response, 4, 0)


def test_duplicate_framing_header_rejected():
    response = Response(b"abcd", headers={"Content-Length": "4"})
    response.headers.add_header("Content-Length", "4")
    with pytest.raises(ArtifactError, match="repeats"):
        acquisition._framing(response, 4, 0)


@pytest.mark.parametrize("url", ["http://huggingface.co/a", "https://127.0.0.1/a", "https://evil.example/a",
    "https://huggingface.co@evil.example/a", "https://huggingface.co:444/a", "https://huggingface.co/a#fragment"])
def test_redirect_target_fail_closed(url):
    with pytest.raises(ArtifactError, match="approved HTTPS"):
        acquisition._url(url)


def test_real_https_adapter_removes_credential_on_redirect_and_closes(monkeypatch):
    records, responses = [], []
    class Socket:
        def settimeout(self, value):
            assert 0 < value <= 30
    class HTTPResponse:
        def __init__(self, number):
            self.status = 302 if number == 0 else 200
            self.headers = Message()
            self.closed = False
        def getheader(self, name):
            return "https://cas-bridge.xethub.hf.co/signed?sig=private"
        def close(self):
            self.closed = True
        def read1(self, count):
            return b""
    class Connection:
        def __init__(self, host, **kwargs):
            self.host, self.sock = host, Socket()
        def connect(self):
            pass
        def request(self, method, path, headers):
            records.append((self.host, dict(headers)))
        def getresponse(self):
            response = HTTPResponse(len(responses))
            responses.append(response)
            return response
        def close(self):
            pass
    monkeypatch.setattr(acquisition.http.client, "HTTPSConnection", Connection)
    with acquisition._request("https://huggingface.co/pinned", {"Authorization": "Bearer private"}, 30, lambda: 0) as response:
        assert response.read(1) == b""
    assert records[0][1]["Authorization"] == "Bearer private"
    assert "Authorization" not in records[1][1]
    assert all(response.closed for response in responses)


def test_public_supervisor_bounds_worker_and_keeps_token_out_of_arguments(tmp_path, monkeypatch):
    captured = {}
    def run(command, **kwargs):
        captured.update(command=command, kwargs=kwargs)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])
    monkeypatch.setattr(acquisition.subprocess, "run", run)
    with pytest.raises(ArtifactError, match="reaped"):
        acquisition.acquire_source(tmp_path / "work with spaces", max_bytes=1000, deadline_ms=7, token_file=tmp_path / "private.token")
    assert captured["kwargs"]["timeout"] == 0.007
    assert str(tmp_path / "work with spaces") in captured["command"]
    assert captured["kwargs"]["stdin"] == subprocess.DEVNULL
    assert captured["kwargs"]["env"].keys() == {"PATH", "PYTHONPATH", "LANG"}


def test_actual_supervised_worker_rejects_small_budget_before_network(tmp_path):
    with pytest.raises(ArtifactError, match="byte limit"):
        acquisition.acquire_source(tmp_path / "work", max_bytes=1, deadline_ms=30000)
    assert not (tmp_path / "work").exists()


def test_interrupted_file_resumes_without_republishing_unfinished_root(tmp_path, publisher):
    plan, _, bodies, calls, request = publisher
    @contextmanager
    def interrupted(url, headers, deadline, clock):
        with request(url, headers, deadline, clock) as response:
            if url.endswith("model-01.safetensors"):
                ordinary = response.read
                used = False
                def read(count):
                    nonlocal used
                    if used:
                        raise OSError("synthetic connection loss")
                    used = True
                    return ordinary(min(count, 7))
                response.read = read
            yield response
    workspace = tmp_path / "work"
    with pytest.raises(OSError, match="connection loss"):
        acquisition._acquire(workspace, plan["total_bytes"], 30000, request=interrupted)
    root = workspace / ("source-" + acquisition.NVIDIA_REVISION)
    assert not (root / "source-acquisition.json").exists()
    assert (root / "config.json").exists()
    facts = acquisition._acquire(workspace, plan["total_bytes"], 30000, request=request)
    assert facts["source_dir"] == str(root) and (root / "model-01.safetensors").read_bytes() == bodies["model-01.safetensors"]
    assert any(headers.get("Range") == "bytes=7-" for _, headers in calls)


@pytest.mark.parametrize("size,offset,status,framing", [(4, 2, 200, {"Content-Length": "2"}),
    (4, 2, 206, {"Content-Length": "2", "Content-Range": "bytes 1-2/4"}),
    (4, 2, 206, {"Content-Length": "4", "Content-Range": "bytes 2-3/4"})])
def test_resume_status_range_and_length_must_all_match(size, offset, status, framing):
    with pytest.raises(ArtifactError, match="HTTP"):
        acquisition._framing(Response(b"ab", status=status, headers=framing), size, offset)


def test_deadline_stops_before_root_mutation(tmp_path, publisher):
    plan, _, _, _, request = publisher
    ticks = iter((0, 31))
    with pytest.raises(ArtifactError, match="deadline"):
        acquisition._acquire(tmp_path / "work", plan["total_bytes"], 30000, request=request, clock=lambda: next(ticks))
    assert not (tmp_path / "work").exists()


def test_owned_unpublished_atomic_temporary_cleanup_is_exact(tmp_path, publisher):
    plan, _, _, _, request = publisher
    workspace = acquisition.private_directory(tmp_path / "work")
    root = acquisition._marked_directory(workspace, "source-" + acquisition.NVIDIA_REVISION, plan["digest"], modes=(0o700,))
    orphan = root / (".ria-" + "a" * 32)
    orphan.write_bytes(b"interrupted unpublished metadata")
    orphan.chmod(0o600)
    keep = root / ".ria-other-owned-evidence"
    keep.write_bytes(b"keep")
    acquisition._acquire(workspace, plan["total_bytes"], 30000, request=request)
    assert not orphan.exists() and keep.read_bytes() == b"keep"


def test_unpublished_temporary_hardlink_rejected_without_outside_mutation(tmp_path, publisher):
    plan, _, _, _, request = publisher
    workspace = acquisition.private_directory(tmp_path / "work")
    root = acquisition._marked_directory(workspace, "source-" + acquisition.NVIDIA_REVISION, plan["digest"], modes=(0o700,))
    outside = tmp_path / "outside"
    outside.write_bytes(b"keep")
    outside.chmod(0o600)
    orphan = root / (".ria-" + "a" * 32)
    os.link(outside, orphan)
    with pytest.raises(ArtifactError, match="safe owned single inode"):
        acquisition._acquire(workspace, plan["total_bytes"], 30000, request=request)
    assert outside.read_bytes() == b"keep" and orphan.exists()
