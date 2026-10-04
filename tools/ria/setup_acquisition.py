"""Opt-in pinned checkpoint acquisition; no model code is imported or executed.

Only reviewed NVIDIA config/index/shards are fetched. Native compact metadata
comes from packaged locks. Each request uses direct verified HTTPS, bounded
redirects with credentials removed, identity encoding and strict range framing.
Private partials survive interruption; only authenticated complete files become
visible and the sealed acquisition record is published last. The public helper
supervises an owned child so DNS and network stalls obey the total deadline.

Primary contracts: https://huggingface.co/docs/hub/en/api ;
https://huggingface.co/docs/hub/en/local-cache ;
https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/utils/_http.py
"""

import argparse
from contextlib import contextmanager
import hashlib
import http.client
import os
from pathlib import Path
import re
import shutil
import ssl
import stat
import subprocess
import sys
import time
from urllib.parse import quote, urljoin, urlsplit

from .identity import (ArtifactError, DIGEST, _open_directory, atomic_output, canonical,
                       hash_file, loads, open_regular, read_json, seal, verify_identity)
from .setup_artifacts import _marked_directory, _publish, private_directory
from .target import MODEL, NVIDIA, NVIDIA_REVISION, SOURCE_REVISION, _locked_metadata

SHA1 = re.compile(r"[0-9a-f]{40}\Z")
MAX_API_BYTES = 16 << 20
BLOCK_BYTES = 1 << 20


class _ContentMismatch(ArtifactError):
    """A valid owned partial's bytes fail its declared publisher identity."""


def acquisition_plan():
    """Derive exactly the reviewed shard population before any network call."""
    repository = read_json(_locked_metadata(NVIDIA, NVIDIA_REVISION, "repository-index.json"))
    index = loads(_locked_metadata(NVIDIA, NVIDIA_REVISION, "model.safetensors.index.json").read_bytes(),
                  project=False, max_nodes=2000000)
    if repository.get("sha") != NVIDIA_REVISION or repository.get("id") != NVIDIA or not isinstance(index.get("weight_map"), dict):
        raise ArtifactError("packaged publisher revision/index differs from pinned source")
    names = {"config.json", "model.safetensors.index.json", *index["weight_map"].values()}
    entries = {}
    for item in repository["siblings"]:
        name = item.get("rfilename")
        if name not in names:
            continue
        if name in entries or not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or (
                name not in ("config.json", "model.safetensors.index.json") and not name.endswith(".safetensors")):
            raise ArtifactError("publisher required-file path/identity is invalid")
        size, blob = item.get("size"), item.get("blobId")
        lfs = item.get("lfs")
        if type(size) is not int or not 0 < size < 1 << 63 or not isinstance(blob, str) or not SHA1.fullmatch(blob):
            raise ArtifactError("publisher required-file size/blob identity is invalid")
        sha = None
        if lfs is not None:
            if not isinstance(lfs, dict) or lfs.get("size") != size or not isinstance(lfs.get("sha256"), str) or not DIGEST.fullmatch(lfs["sha256"]):
                raise ArtifactError("publisher LFS size/hash identity is invalid")
            sha = lfs["sha256"]
        elif name not in ("config.json", "model.safetensors.index.json"):
            raise ArtifactError("required model shard has no reviewed LFS identity")
        content = sha if sha is not None else hash_file(_locked_metadata(NVIDIA, NVIDIA_REVISION, name))
        entries[name] = {"path": name, "bytes": size, "blob_id": blob, "lfs_sha256": sha, "sha256": content}
    if entries.keys() != names:
        raise ArtifactError("publisher inventory omits required source files")
    local = []
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        path = _locked_metadata(MODEL, SOURCE_REVISION, name)
        local.append({"path": name, "bytes": path.stat().st_size, "sha256": hash_file(path)})
    return seal({"schema_revision": 1, "repository": NVIDIA, "revision": NVIDIA_REVISION,
                 "files": [entries[name] for name in sorted(entries)], "native_metadata": local,
                 "total_bytes": sum(item["bytes"] for item in entries.values()) + sum(item["bytes"] for item in local)})


def _remaining(deadline, clock):
    value = deadline - clock()
    if value <= 0:
        raise ArtifactError("pinned source acquisition exceeded its declared deadline")
    return min(value, 30.0)


def _url(url):
    value = urlsplit(url)
    host = value.hostname
    if value.scheme != "https" or value.username is not None or value.password is not None or value.port not in (None, 443) or (
            value.fragment or not host or not (host == "huggingface.co" or host.endswith(".huggingface.co") or host.endswith(".hf.co"))) or (
            any(ord(character) < 33 or ord(character) > 126 for character in url)):
        raise ArtifactError("source redirect requires an approved HTTPS Hub/storage host")
    return value


class _Response:
    def __init__(self, response, socket, deadline, clock):
        self.response, self.socket, self.deadline, self.clock = response, socket, deadline, clock
        self.status = response.status
        self.headers = response.headers

    def read(self, count):
        self.socket.settimeout(_remaining(self.deadline, self.clock))
        return self.response.read1(count)


@contextmanager
def _request(url, headers, deadline, clock):
    # No environment proxy, cookie jar, implicit HF token or custom endpoint.
    # Each redirect starts a new TLS connection and loses Authorization.
    current, outgoing = url, dict(headers)
    connection = None
    try:
        for _ in range(6):
            value = _url(current)
            connection = http.client.HTTPSConnection(value.hostname, timeout=_remaining(deadline, clock),
                                                     context=ssl.create_default_context())
            connection.connect()
            socket = connection.sock
            connection.request("GET", value.path + ("?" + value.query if value.query else ""), headers=outgoing)
            response = connection.getresponse()
            if response.status not in (301, 302, 303, 307, 308):
                try:
                    yield _Response(response, socket, deadline, clock)
                finally:
                    response.close()
                return
            location = response.getheader("Location")
            if not location:
                raise ArtifactError("source redirect lacks a target")
            current = urljoin(current, location)
            outgoing.pop("Authorization", None)
            response.close()
            connection.close()
        raise ArtifactError("source acquisition exceeded its bounded redirect count")
    except (OSError, http.client.HTTPException, ValueError) as exc:
        # Exception chains may contain signed URLs or credentials; callers get
        # an actionable class without serializing those diagnostics.
        raise ArtifactError("pinned source HTTPS request failed") from exc
    finally:
        if connection is not None:
            connection.close()


def _header(response, name):
    values = response.headers.get_all(name) if hasattr(response.headers, "get_all") else [response.headers.get(name)]
    values = [value for value in values or [] if value is not None]
    if len(values) > 1:
        raise ArtifactError("source HTTP response repeats a framing header")
    return values[0] if values else None


def _framing(response, size, offset):
    if response.status != (206 if offset else 200) or _header(response, "Content-Encoding") not in (None, "identity") or _header(response, "Transfer-Encoding") is not None:
        raise ArtifactError("source HTTP status/encoding cannot preserve exact bytes")
    if _header(response, "Content-Length") != str(size - offset):
        raise ArtifactError("source HTTP length differs from pinned remaining bytes")
    expected = f"bytes {offset}-{size - 1}/{size}" if offset else None
    if _header(response, "Content-Range") != expected:
        raise ArtifactError("source HTTP range differs from private partial offset")


def _verify_file(path, item, deadline, clock):
    sha = hashlib.sha256()
    git = hashlib.sha1(b"blob " + str(item["bytes"]).encode("ascii") + b"\0")
    with open_regular(path) as stream:
        info = os.fstat(stream.fileno())
        if info.st_uid != os.geteuid() or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size != item["bytes"]:
            raise ArtifactError("owned acquired file size/type/mode differs from its plan")
        while block := stream.read(BLOCK_BYTES):
            _remaining(deadline, clock)
            sha.update(block)
            if item["lfs_sha256"] is None:
                git.update(block)
    if sha.hexdigest() != item["sha256"] or (
            item["lfs_sha256"] is None and git.hexdigest() != item["blob_id"]):
        raise _ContentMismatch("acquired file differs from pinned publisher content identity")
    return sha.hexdigest()


def _cleanup_unpublished(rootfd, deadline, clock):
    """Reclaim only atomic_output's exact private temporary names in our inode."""
    changed = False
    with os.scandir(rootfd) as entries:
        for entry in entries:
            _remaining(deadline, clock)
            if not re.fullmatch(r"\.ria-[0-9a-f]{32}", entry.name):
                continue
            fd = os.open(entry.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, dir_fd=rootfd)
            try:
                info = os.fstat(fd)
                observed = os.stat(entry.name, dir_fd=rootfd, follow_symlinks=False)
                if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_nlink != 1 or (
                        stat.S_IMODE(info.st_mode) != 0o600 or (info.st_dev, info.st_ino) != (observed.st_dev, observed.st_ino)):
                    raise ArtifactError("unpublished source temporary is not a safe owned single inode")
                os.unlink(entry.name, dir_fd=rootfd)
                changed = True
            finally:
                os.close(fd)
    if changed:
        os.fsync(rootfd)


def _acquire(workspace, max_bytes, deadline_ms, token_file=None, *, request=_request, clock=time.monotonic):
    if type(max_bytes) is not int or not 1 <= max_bytes < 1 << 63 or type(deadline_ms) is not int or not 1 <= deadline_ms <= 604800000:
        raise ArtifactError("source acquisition requires explicit positive byte and at-most-seven-day time bounds")
    deadline = clock() + deadline_ms / 1000
    plan = acquisition_plan()
    if plan["total_bytes"] > max_bytes:
        raise ArtifactError("pinned source population exceeds declared acquisition byte limit")
    token = None
    if token_file is not None:
        with open_regular(token_file) as stream:
            info = os.fstat(stream.fileno())
            if stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1 or not 1 <= info.st_size <= 4096:
                raise ArtifactError("HF token must be a private bounded single regular file")
            raw = stream.read(4097).rstrip(b"\r\n")
        if not raw or any(byte < 33 or byte > 126 for byte in raw):
            raise ArtifactError("HF token must be nonwhitespace ASCII")
        token = raw.decode("ascii")
    headers = {"Accept-Encoding": "identity", "User-Agent": "RIA-pinned-source/1"}
    if token is not None:
        headers["Authorization"] = "Bearer " + token
    api = f"https://huggingface.co/api/models/{NVIDIA}/revision/{NVIDIA_REVISION}?blobs=true"
    with request(api, headers, deadline, clock) as response:
        if response.status != 200:
            raise ArtifactError("pinned publisher revision metadata request failed")
        data = bytearray()
        while block := response.read(min(BLOCK_BYTES, MAX_API_BYTES + 1 - len(data))):
            _remaining(deadline, clock)
            data.extend(block)
            if len(data) > MAX_API_BYTES:
                raise ArtifactError("publisher revision metadata exceeds its byte bound")
    remote = loads(bytes(data), project=False)
    if not isinstance(remote, dict) or remote.get("sha") != NVIDIA_REVISION or remote.get("id") != NVIDIA or not isinstance(remote.get("siblings"), list):
        raise ArtifactError("remote publisher metadata differs from pinned repository/revision")
    siblings = {}
    for item in remote["siblings"]:
        if not isinstance(item, dict) or not isinstance(item.get("rfilename"), str) or item["rfilename"] in siblings:
            raise ArtifactError("remote publisher metadata repeats/invalidates a file identity")
        siblings[item["rfilename"]] = item
    for item in plan["files"]:
        peer = siblings.get(item["path"], {})
        lfs = peer.get("lfs")
        if (lfs is not None and not isinstance(lfs, dict)) or type(peer.get("size")) is not int or (
                (peer.get("size"), peer.get("blobId"), lfs.get("sha256") if lfs else None) != (item["bytes"], item["blob_id"], item["lfs_sha256"])) or (
                lfs is not None and lfs.get("size") != item["bytes"]):
            raise ArtifactError("remote publisher file identity differs from reviewed lock")
    root = _marked_directory(private_directory(workspace), "source-" + NVIDIA_REVISION, plan["digest"], modes=(0o700,))
    partial = private_directory(root / ".download")
    rootfd, partialfd = _open_directory(root), _open_directory(partial)
    try:
        _cleanup_unpublished(rootfd, deadline, clock)
        remaining = sum(item["bytes"] for item in plan["native_metadata"] if not (root / item["path"]).exists())
        for item in plan["files"]:
            if (root / item["path"]).exists():
                continue
            temporary = partial / (hashlib.sha256(item["path"].encode()).hexdigest() + ".partial")
            resumed = 0
            if temporary.exists() or temporary.is_symlink():
                with open_regular(temporary) as stream:
                    info = os.fstat(stream.fileno())
                    if info.st_uid != os.geteuid() or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size > item["bytes"]:
                        raise ArtifactError("source partial has invalid ownership/type/size")
                    resumed = info.st_size
            remaining += item["bytes"] - resumed
        # Atomic metadata publication can retain one prior complete copy and
        # its replacement stream. Include that transient file and the bounded
        # sealed record rather than treating final bytes as the disk peak.
        remaining += max((item["bytes"] for item in plan["native_metadata"]), default=0) + (1 << 20)
        if shutil.disk_usage(root).free < remaining:
            raise ArtifactError("source acquisition lacks disk space for its pinned remaining population")
        completed = []
        for item in plan["files"]:
            name, final = item["path"], root / item["path"]
            temporary = hashlib.sha256(name.encode()).hexdigest() + ".partial"
            if final.exists() or final.is_symlink():
                # Reconcile a crash between link publication and private-name
                # unlink only when these two names account for both links.
                observed = os.stat(name, dir_fd=rootfd, follow_symlinks=False)
                if stat.S_ISREG(observed.st_mode) and observed.st_nlink == 2:
                    candidate = os.stat(temporary, dir_fd=partialfd, follow_symlinks=False)
                    if (candidate.st_dev, candidate.st_ino, candidate.st_nlink, candidate.st_uid) != (
                            observed.st_dev, observed.st_ino, 2, os.geteuid()):
                        raise ArtifactError("source publication links cannot be reconciled safely")
                    os.unlink(temporary, dir_fd=partialfd)
                    os.fsync(partialfd)
                sha = _verify_file(final, item, deadline, clock)
            else:
                fd = os.open(temporary, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, 0o600, dir_fd=partialfd)
                try:
                    info = os.fstat(fd)
                    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size > item["bytes"]:
                        raise ArtifactError("source partial has invalid ownership/type/size")
                    offset = info.st_size
                    if offset < item["bytes"]:
                        outgoing = {**headers, "Range": f"bytes={offset}-"} if offset else dict(headers)
                        url = f"https://huggingface.co/{NVIDIA}/resolve/{NVIDIA_REVISION}/{quote(name, safe='')}"
                        os.lseek(fd, offset, os.SEEK_SET)
                        with request(url, outgoing, deadline, clock) as response:
                            _framing(response, item["bytes"], offset)
                            while offset < item["bytes"]:
                                _remaining(deadline, clock)
                                block = response.read(min(BLOCK_BYTES, item["bytes"] - offset))
                                if not block:
                                    raise ArtifactError("source HTTP body ended before pinned file length")
                                if len(block) > item["bytes"] - offset:
                                    raise ArtifactError("source HTTP body exceeds pinned file length")
                                view = memoryview(block)
                                while view:
                                    written = os.write(fd, view)
                                    if written <= 0:
                                        raise ArtifactError("source partial write did not progress")
                                    view = view[written:]
                                offset += len(block)
                            if response.read(1):
                                raise ArtifactError("source HTTP body exceeds pinned file length")
                    os.fsync(fd)
                finally:
                    os.close(fd)
                try:
                    sha = _verify_file(partial / temporary, item, deadline, clock)
                except _ContentMismatch:
                    os.unlink(temporary, dir_fd=partialfd)
                    os.fsync(partialfd)
                    raise
                # Link without replacement: concurrent publication cannot be
                # overwritten. The controller serializes its owned workspace.
                os.link(temporary, name, src_dir_fd=partialfd, dst_dir_fd=rootfd, follow_symlinks=False)
                os.unlink(temporary, dir_fd=partialfd)
                os.fsync(rootfd)
                os.fsync(partialfd)
            completed.append({**item, "sha256": sha})
        for item in plan["native_metadata"]:
            path = _locked_metadata(MODEL, SOURCE_REVISION, item["path"])
            with open_regular(path) as source, atomic_output(root / item["path"], mode=0o600, immutable=True) as destination:
                copied = 0
                while block := source.read(BLOCK_BYTES):
                    _remaining(deadline, clock)
                    copied += len(block)
                    if copied > item["bytes"]:
                        raise ArtifactError("packaged native metadata changed during acquisition")
                    destination.write(block)
            if copied != item["bytes"] or hash_file(root / item["path"]) != item["sha256"]:
                raise ArtifactError("packaged native metadata differs from reviewed hash")
        _remaining(deadline, clock)
        facts = seal({"schema_revision": 1, "kind": "setup_source_acquisition", "repository": NVIDIA,
                      "revision": NVIDIA_REVISION, "source_dir": str(root), "plan_digest": plan["digest"],
                      "total_bytes": plan["total_bytes"], "files": completed, "native_metadata": plan["native_metadata"]})
        return _publish(root / "source-acquisition.json", facts)
    finally:
        os.close(rootfd)
        os.close(partialfd)


def acquire_source(workspace, *, max_bytes, deadline_ms, token_file=None):
    """Run the bounded source-only worker; terminate/reap only its owned process."""
    if type(max_bytes) is not int or not 1 <= max_bytes < 1 << 63 or type(deadline_ms) is not int or not 1 <= deadline_ms <= 604800000:
        raise ArtifactError("source acquisition requires positive bytes and an at-most-seven-day deadline")
    command = [sys.executable, "-m", "ria.setup_acquisition", "--worker", "--workspace", str(workspace),
               "--max-bytes", str(max_bytes), "--deadline-ms", str(deadline_ms)]
    if token_file is not None:
        command.extend(("--token-file", str(token_file)))
    environment = {"PATH": "/usr/local/bin:/usr/bin:/bin", "PYTHONPATH": str(Path(__file__).resolve().parents[1]), "LANG": "C.UTF-8"}
    try:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                env=environment, cwd=Path(__file__).resolve().parents[1], timeout=deadline_ms / 1000, check=False)
    except subprocess.TimeoutExpired as exc:
        raise ArtifactError("source acquisition child exceeded its declared deadline and was reaped") from exc
    if len(result.stdout) > 1 << 20 or len(result.stderr) > 4096:
        raise ArtifactError("source acquisition child exceeded bounded diagnostics")
    if result.returncode:
        raise ArtifactError("source acquisition failed: " + result.stderr.decode("utf-8", errors="replace").strip())
    facts = loads(result.stdout)
    verify_identity(facts)
    return facts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--max-bytes", type=int, required=True)
    parser.add_argument("--deadline-ms", type=int, required=True)
    parser.add_argument("--token-file")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        function = _acquire if args.worker else acquire_source
        facts = function(args.workspace, max_bytes=args.max_bytes, deadline_ms=args.deadline_ms, token_file=args.token_file)
        sys.stdout.buffer.write(canonical(facts) + b"\n")
        return 0
    except (ArtifactError, OSError) as exc:
        print(str(exc)[:2048], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
