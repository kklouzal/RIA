"""Bounded setup RPC, separate from DSER and from measured native child ownership.

The network worker starts through fresh exec with one broker descriptor. It has
no controller objects, CA key, Docker descriptors or inherited parent heap.
Only locally registered exports and fixed dispatcher operations reach the
controller. TLS authenticates its pinned server; MACs authenticate both roles
and every reply in TLS and explicitly selected trusted-network mode.
"""

import ctypes
import copy
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import hmac
import http.client
from http.server import BaseHTTPRequestHandler, HTTPServer
import ipaddress
import os
from pathlib import Path, PurePosixPath
import secrets
import signal
import shutil
import socket
import ssl
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time

from .identity import (
    ArtifactError,
    canonical as identity_canonical,
    check_json,
    digest,
    loads,
    open_regular,
    seal,
    sync_directory,
    verify_identity,
)
from .setup_journal import SetupJournal
from .setup_security import endpoint, validate_invitation
from .setup_limits import (
    CHUNK,
    MAX_FILES,
    MAX_MESSAGE,
    MAX_NODES,
    MAX_RECEIPT,
    MAX_CHUNK_METADATA,
    MAX_SETUP_SECONDS,
    MAX_TRANSFER_SECONDS,
    MAX_TREE,
)

JSON_TYPE = "application/vnd.ria.setup.rpc+json;version=1"
BINARY_TYPE = "application/vnd.ria.setup.chunk;version=1"
BROKER_HEADER = struct.Struct("!BBII")  # version, kind, metadata bytes, binary bytes


@dataclass(frozen=True)
class _Raw:
    value: dict
    data: bytes


def _descriptor(data):
    if not isinstance(data, bytes) or len(data) > CHUNK:
        raise ArtifactError("setup binary chunk exceeds its byte bound")
    return {
        "schema_revision": 1,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _raw_descriptor(value):
    if (
        isinstance(value, dict)
        and value.get("operation") == "import-chunk"
        and isinstance(value.get("payload"), dict)
    ):
        descriptor = value["payload"].get("data")
    elif (
        isinstance(value, dict)
        and isinstance(value.get("response"), dict)
        and value["response"].get("ok") is True
        and isinstance(value["response"].get("result"), dict)
        and set(value["response"]["result"]) == {"data", "sha256"}
    ):
        descriptor = value["response"]["result"]["data"]
        if (
            not isinstance(descriptor, dict)
            or descriptor.get("sha256") != value["response"]["result"]["sha256"]
        ):
            raise ArtifactError("setup binary reply digests differ")
    else:
        raise ArtifactError(
            "binary setup bytes require a fixed import/export operation"
        )
    if (
        not isinstance(descriptor, dict)
        or set(descriptor) != {"schema_revision", "bytes", "sha256"}
        or type(descriptor["schema_revision"]) is not int
        or descriptor["schema_revision"] != 1
        or type(descriptor["bytes"]) is not int
        or not 0 <= descriptor["bytes"] <= CHUNK
        or not isinstance(descriptor["sha256"], str)
        or len(descriptor["sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in descriptor["sha256"])
    ):
        raise ArtifactError("invalid setup binary descriptor")
    return descriptor


def _split_raw(value):
    if (
        isinstance(value, dict)
        and value.get("operation") == "import-chunk"
        and isinstance(value.get("payload"), dict)
        and isinstance(value["payload"].get("data"), bytes)
    ):
        raw = value["payload"]["data"]
        return _Raw(
            {**value, "payload": {**value["payload"], "data": _descriptor(raw)}}, raw
        )
    if (
        isinstance(value, dict)
        and isinstance(value.get("response"), dict)
        and value["response"].get("ok") is True
        and isinstance(value["response"].get("result"), dict)
        and isinstance(value["response"]["result"].get("data"), bytes)
    ):
        result = value["response"]["result"]
        raw = result["data"]
        metadata = {
            **value,
            "response": {
                **value["response"],
                "result": {**result, "data": _descriptor(raw)},
            },
        }
        _raw_descriptor(metadata)
        return _Raw(metadata, raw)
    return None


def _validate_raw(value, raw):
    descriptor = _raw_descriptor(value)
    if (
        len(raw) != descriptor["bytes"]
        or hashlib.sha256(raw).hexdigest() != descriptor["sha256"]
    ):
        raise ArtifactError("setup binary tail length/hash differs")


def _attach_raw(value, raw):
    """Attach immutable bytes whose metadata/length/SHA was proven at this hop.

    Only the private validated exchange result reaches the client attachment;
    its metadata has no shared writers while the client owns its operation lock.
    """
    if value.get("operation") == "import-chunk":
        return {**value, "payload": {**value["payload"], "data": raw}}
    return {
        **value,
        "response": {
            **value["response"],
            "result": {**value["response"]["result"], "data": raw},
        },
    }


def _restore_raw(value, raw):
    _validate_raw(value, raw)
    return _attach_raw(value, raw)


OPERATIONS = frozenset(
    {
        "offer",
        "client-offer",
        "commit-plan",
        "registration",
        "fixture-start",
        "probe",
        "native-fixture",
        "transport-fixture",
        "collect",
        "finalize",
        "launch",
        "health",
        "stop",
        "finish",
        "sign-csr",
    }
)


def canonical(value):
    check_json(value, max_nodes=MAX_NODES, max_depth=24)
    return identity_canonical(value)


class SetupNotReady(ArtifactError):
    """Only this classified environmental state may be polled by orchestration."""


def _timeout(value, maximum=MAX_TRANSFER_SECONDS):
    if type(value) not in (int, float) or not 0 < value <= maximum:
        raise ArtifactError("setup timeout is outside its positive bounded policy")
    return value


def _publish_tree(staging, output):
    # Linux is the declared setup target; never overwrite an unrelated directory.
    library = ctypes.CDLL(None, use_errno=True)
    rename = library.renameat2
    rename.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(staging), -100, os.fsencode(output), 1):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    sync_directory(output.parent)


def _private_directory(path):
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise ArtifactError("setup output requires an owned private real directory")


def _relative(value):
    if (
        not isinstance(value, str)
        or not 0 < len(value) <= 4096
        or "\\" in value
        or any(ord(c) < 32 or ord(c) == 127 for c in value)
    ):
        raise ArtifactError("invalid setup artifact path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or len(path.parts) > 32
        or len(value.encode()) > 4095
        or any(part in (".", "..") or len(part.encode()) > 255 for part in path.parts)
    ):
        raise ArtifactError("setup artifact path must be canonical and relative")
    if path.suffix.lower() in (".key", ".pem", ".token") or path.name.startswith(
        ".env"
    ):
        raise ArtifactError("credentials must never be exported as setup artifacts")
    return path


def _file_state(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _bounded_hash(source, length, check, label):
    """Hash exactly the declared size, polling between bounded file reads."""
    hasher, remaining = hashlib.sha256(), length
    if check is not None:
        check()
    while remaining:
        count = min(CHUNK, remaining)
        data = source.read(count)
        if len(data) != count:
            raise ArtifactError(label + " truncated during bounded hashing")
        hasher.update(data)
        remaining -= count
        if check is not None:
            check()
    if source.read(1):
        raise ArtifactError(label + " grew during bounded hashing")
    return hasher.hexdigest()


class ExportTree:
    def __init__(
        self, name, path, *, max_bytes=MAX_TREE, check=None, reject_hardlinks=False
    ):
        """Caller owns immutable inputs and a non-reentrant optional stop callback."""
        if check is not None and not callable(check):
            raise ArtifactError("setup transfer check must be callable")
        self.check = check
        if (
            not isinstance(name, str)
            or not name
            or len(name) > 96
            or any(
                c
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                for c in name
            )
        ):
            raise ArtifactError("invalid registered export name")
        self.root = Path(path).absolute()
        if self.root.is_symlink() or not self.root.is_dir():
            raise ArtifactError("export must be an existing real directory")
        root_info = self.root.stat()
        self.root_identity = root_info.st_dev, root_info.st_ino
        if type(max_bytes) is not int or not 0 <= max_bytes <= MAX_TREE:
            raise ArtifactError("invalid registered export byte quota")
        self.states = {}
        files, total, pending, directories, manifest_bound = [], 0, [()], 0, 512
        while pending:
            if check is not None:
                check()
            directory = pending.pop()
            directories += 1
            if directories > MAX_FILES:
                raise ArtifactError("export directory population exceeds its bound")
            descriptor = self._directory(directory)
            try:
                with os.scandir(descriptor) as entries:
                    for entry in entries:
                        if check is not None:
                            check()
                        parts = (*directory, entry.name)
                        relative = "/".join(parts)
                        _relative(relative)
                        if entry.is_symlink():
                            raise ArtifactError("export contains a symlink")
                        if entry.is_dir(follow_symlinks=False):
                            if directories + len(pending) >= MAX_FILES:
                                raise ArtifactError(
                                    "export directory population exceeds its bound"
                                )
                            pending.append(parts)
                            continue
                        if (
                            not entry.is_file(follow_symlinks=False)
                            or len(files) >= MAX_FILES
                        ):
                            raise ArtifactError(
                                "export contains special files or too many files"
                            )
                        manifest_bound += 2 * len(relative.encode()) + 192
                        if manifest_bound > MAX_MESSAGE - 65536:
                            raise ArtifactError(
                                "setup export manifest exceeds its message envelope"
                            )
                        fd = os.open(
                            entry.name,
                            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
                            dir_fd=descriptor,
                        )
                        with os.fdopen(fd, "rb") as source:
                            before = os.fstat(source.fileno())
                            if not stat.S_ISREG(before.st_mode):
                                raise ArtifactError(
                                    "export leaf changed to a special file"
                                )
                            total += before.st_size
                            if reject_hardlinks and before.st_nlink != 1:
                                raise ArtifactError(
                                    "setup report snapshot forbids hardlinks"
                                )
                            if total > max_bytes:
                                raise ArtifactError(
                                    "registered export exceeds its byte quota"
                                )
                            sha = _bounded_hash(source, before.st_size, check, "export")
                            if _file_state(before) != _file_state(
                                os.fstat(source.fileno())
                            ):
                                raise ArtifactError(
                                    "export changed during registration"
                                )
                        self.states[relative] = _file_state(before)
                        files.append(
                            {"path": relative, "bytes": before.st_size, "sha256": sha}
                        )
            finally:
                os.close(descriptor)
        if not files:
            raise ArtifactError("registered export is empty")
        self.manifest = seal(
            {
                "schema_revision": 1,
                "name": name,
                "files": sorted(files, key=lambda f: f["path"]),
                "total_bytes": total,
            }
        )

    def _directory(self, parts):
        directory = os.open(
            self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        )
        try:
            info = os.fstat(directory)
            if (info.st_dev, info.st_ino) != self.root_identity:
                raise ArtifactError("export root changed")
            for part in parts:
                child = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=directory,
                )
                os.close(directory)
                directory = child
            return directory
        except BaseException:
            os.close(directory)
            raise

    def chunk(self, path, offset, length):
        if self.check is not None:
            self.check()
        parts = _relative(path).parts
        if (
            path not in self.states
            or type(offset) is not int
            or type(length) is not int
            or not 0 <= offset <= self.states[path][2]
            or not 0 < length <= CHUNK
        ):
            raise ArtifactError("chunk is outside its registered export")
        directory = self._directory(parts[:-1])
        try:
            fd = os.open(
                parts[-1],
                os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
                dir_fd=directory,
            )
        finally:
            os.close(directory)
        with os.fdopen(fd, "rb") as source:
            if _file_state(os.fstat(source.fileno())) != self.states[path]:
                raise ArtifactError("export changed after registration")
            source.seek(offset)
            data = source.read(min(length, self.states[path][2] - offset))
            if _file_state(os.fstat(source.fileno())) != self.states[path]:
                raise ArtifactError("export changed during transfer")
        return data


def export_tree(name, path, *, max_bytes=MAX_TREE, check=None, reject_hardlinks=False):
    return ExportTree(
        name, path, max_bytes=max_bytes, check=check, reject_hardlinks=reject_hardlinks
    )


def _manifest(manifest, max_bytes):
    if (
        not isinstance(manifest, dict)
        or set(manifest)
        != {"schema_revision", "name", "files", "total_bytes", "digest"}
        or type(manifest["schema_revision"]) is not int
        or manifest["schema_revision"] != 1
    ):
        raise ArtifactError("invalid export manifest")
    verify_identity(manifest)
    if (
        not isinstance(manifest["name"], str)
        or not manifest["name"]
        or len(manifest["name"]) > 96
        or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for c in manifest["name"]
        )
    ):
        raise ArtifactError("invalid setup manifest name")
    files = manifest["files"]
    if (
        type(max_bytes) is not int
        or not isinstance(files, list)
        or not 0 < len(files) <= MAX_FILES
        or type(manifest["total_bytes"]) is not int
        or not 0 <= manifest["total_bytes"] <= max_bytes <= MAX_TREE
    ):
        raise ArtifactError("export manifest exceeds import quota")
    seen, directories, total = set(), set(), 0
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}:
            raise ArtifactError("invalid export file entry")
        _relative(item["path"])
        if (
            item["path"] in seen
            or type(item["bytes"]) is not int
            or item["bytes"] < 0
            or not isinstance(item["sha256"], str)
            or len(item["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in item["sha256"])
        ):
            raise ArtifactError("duplicate or invalid export file")
        seen.add(item["path"])
        directories.update(str(p) for p in PurePosixPath(item["path"]).parents)
        if len(directories) > MAX_FILES:
            raise ArtifactError("setup manifest directory population exceeds its bound")
        total += item["bytes"]
    if total != manifest["total_bytes"] or any(
        str(parent) in seen for p in seen for parent in PurePosixPath(p).parents
    ):
        raise ArtifactError("export totals or file/directory paths conflict")
    return files


def import_tree(manifest, output, reader, *, max_bytes=MAX_TREE, check=None):
    """Publish a complete tree once; caller holds its private workspace process lock."""
    files = _manifest(manifest, max_bytes)
    if check is not None:
        check()
    total = manifest["total_bytes"]
    output = Path(output).absolute()
    _private_directory(output.parent)
    if output.exists() or output.is_symlink():
        raise ArtifactError("setup import destination must be new")
    if shutil.disk_usage(output.parent).free < total:
        raise ArtifactError("insufficient disk capacity for bounded setup import")
    staging = Path(tempfile.mkdtemp(prefix=".ria-import-", dir=output.parent))
    try:
        # Metadata cannot expose an incomplete model tree, even within staging.
        for item in sorted(
            files, key=lambda f: (f["path"] == "manifest.json", f["path"])
        ):
            target = staging.joinpath(*PurePosixPath(item["path"]).parts)
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd = os.open(
                target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
            )
            sha, offset = hashlib.sha256(), 0
            with os.fdopen(fd, "wb") as stream:
                while offset < item["bytes"]:
                    if check is not None:
                        check()
                    length = min(CHUNK, item["bytes"] - offset)
                    data = reader(item["path"], offset, length)
                    if not isinstance(data, bytes) or len(data) != length:
                        raise ArtifactError(
                            "setup file transfer was truncated or oversized"
                        )
                    stream.write(data)
                    sha.update(data)
                    offset += length
                stream.flush()
                os.fsync(stream.fileno())
            if sha.hexdigest() != item["sha256"]:
                raise ArtifactError("setup file transfer hash mismatch")
        for directory in sorted(
            (p for p in staging.rglob("*") if p.is_dir()), reverse=True
        ):
            sync_directory(directory)
        sync_directory(staging)
        if check is not None:
            check()
        _publish_tree(staging, output)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


class _Upload:
    """One locally allowed target; retries only continue the identical manifest.

    Progress is owned by this controller lifetime. A leftover staging directory
    after a crash fails closed and requires explicit local recovery; it is never
    mistaken for a completed proof tree or silently erased.
    """

    def __init__(self, output, manifest, maximum, check=None):
        self.check = check
        if check is not None:
            check()
        self.files = {f["path"]: f for f in _manifest(manifest, maximum)}
        self.manifest, self.output = manifest, Path(output).absolute()
        _private_directory(self.output.parent)
        if self.output.exists() or self.output.is_symlink():
            raise ArtifactError("setup upload destination must be new")
        if shutil.disk_usage(self.output.parent).free < manifest["total_bytes"]:
            raise ArtifactError("insufficient capacity for bounded setup upload")
        self.staging = self.output.parent / (".ria-upload-" + manifest["digest"])
        self.staging.mkdir(mode=0o700, exist_ok=False)
        self.offsets, self.completed = dict.fromkeys(self.files, 0), False
        for name in self.files:
            path = self.staging.joinpath(*PurePosixPath(name).parts)
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with os.fdopen(
                os.open(
                    path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
                ),
                "wb",
            ):
                pass

    def status(self):
        return {
            "digest": self.manifest["digest"],
            "offsets": dict(self.offsets),
            "completed": self.completed,
        }

    def chunk(self, path, offset, data):
        if self.check is not None:
            self.check()
        if (
            self.completed
            or not isinstance(path, str)
            or path not in self.files
            or type(offset) is not int
            or offset < 0
            or not isinstance(data, bytes)
            or len(data) > CHUNK
        ):
            raise ArtifactError("invalid setup upload chunk")
        raw = data
        if (
            not 0 < len(raw) <= CHUNK
            or offset > self.offsets[path]
            or offset + len(raw) > self.files[path]["bytes"]
        ):
            raise ArtifactError("setup upload chunk is outside declared file")
        target = self.staging.joinpath(*PurePosixPath(path).parts)
        with open(target, "r+b") as stream:
            stream.seek(offset)
            if offset < self.offsets[path]:
                if (
                    offset + len(raw) > self.offsets[path]
                    or stream.read(len(raw)) != raw
                ):
                    raise ArtifactError("setup upload retry has different bytes")
            else:
                stream.write(raw)
                stream.flush()
                self.offsets[path] += len(raw)
        return self.status()

    def finish(self):
        if self.check is not None:
            self.check()
        if self.completed:
            return self.status()
        for name, item in self.files.items():
            if self.offsets[name] != item["bytes"]:
                raise ArtifactError("setup upload is incomplete")
            with open_regular(
                self.staging.joinpath(*PurePosixPath(name).parts)
            ) as stream:
                if self.check is not None:
                    self.check()
                os.fsync(stream.fileno())
                if (
                    _bounded_hash(stream, item["bytes"], self.check, "setup upload")
                    != item["sha256"]
                ):
                    raise ArtifactError("setup upload file hash mismatch")
        for directory in sorted(
            (p for p in self.staging.rglob("*") if p.is_dir()), reverse=True
        ):
            if self.check is not None:
                self.check()
            sync_directory(directory)
        sync_directory(self.staging)
        if self.check is not None:
            self.check()
        _publish_tree(self.staging, self.output)
        self.completed = True
        return self.status()


def _send(stream, value):
    binary = _split_raw(value)
    metadata, raw = (value, b"") if binary is None else (binary.value, binary.data)
    data = canonical(metadata)
    if len(data) > (MAX_MESSAGE if binary is None else MAX_CHUNK_METADATA):
        raise ArtifactError("setup broker message exceeds its bound")
    stream.sendall(BROKER_HEADER.pack(1, int(binary is not None), len(data), len(raw)))
    stream.sendall(data)
    if raw:
        stream.sendall(raw)


def _receive(stream):
    def exact(length):
        chunks = bytearray()
        while len(chunks) < length:
            data = stream.recv(length - len(chunks))
            if not data:
                raise EOFError("setup broker disconnected")
            chunks.extend(data)
        return bytes(chunks)

    version, kind, length, raw_length = BROKER_HEADER.unpack(exact(BROKER_HEADER.size))
    if (
        version != 1
        or kind not in (0, 1)
        or not 0 < length <= (MAX_MESSAGE if kind == 0 else MAX_CHUNK_METADATA)
        or not 0 <= raw_length <= CHUNK
        or (kind == 0 and raw_length != 0)
    ):
        raise ArtifactError("setup broker length exceeds its bound")
    value = loads(
        exact(length), max_bytes=MAX_MESSAGE, max_nodes=MAX_NODES, max_depth=24
    )
    if kind == 0:
        return value
    descriptor = _raw_descriptor(value)
    if descriptor["bytes"] != raw_length:
        raise ArtifactError("setup broker binary declaration differs")
    return _restore_raw(value, exact(raw_length))


def _mac(invitation, label, value):
    return hmac.new(
        bytes.fromhex(invitation["pair_secret"]),
        label.encode() + b"\x00" + canonical(value),
        hashlib.sha256,
    ).hexdigest()


class PeerServer:
    def __init__(
        self,
        invitation,
        security,
        dispatcher,
        exports=None,
        journal=None,
        *,
        worker_uid=10001,
        worker_gid=10001,
        timeout=30,
    ):
        self.invitation = validate_invitation(invitation)
        if security.invitation != invitation or security.role != "expert":
            raise ArtifactError(
                "peer server security differs from its fixed invitation"
            )
        self.security, self.dispatcher = security, dispatcher
        self.exports = dict(exports or {})
        self.journal = journal or SetupJournal(security.directory / "journal")
        self.worker_uid, self.worker_gid, self.timeout = (
            worker_uid,
            worker_gid,
            _timeout(timeout),
        )
        if worker_uid != 10001 or worker_gid != 10001:
            raise ArtifactError(
                "setup requires a privileged controller and exact isolated UID/GID10001 worker"
            )
        self.process = self.broker = self.thread = None
        self.tasks, self.mutex = {}, threading.Lock()
        self.lifecycle = threading.Condition(threading.RLock())
        self.inflight = None
        self.reply_failure = None
        self.task_errors = {}
        self._stopping = False  # All publication/reads use the lifecycle lock.
        self.imports = {}
        self._rpc_deadline = None  # Set/read only by the serialized broker operation.

    @property
    def stopping(self):
        with self.lifecycle:
            return self._stopping

    @stopping.setter
    def stopping(self, value):
        if type(value) is not bool:
            raise ArtifactError("setup stop state must be a boolean")
        with self.lifecycle:
            self._stopping = value

    def pending_tasks(self, *, timeout=0):
        """Return a locked Thread snapshot; failure retains controller ownership.

        Before shutdown this is observational. After the broker has joined and
        admission stopped, the snapshot includes every possible task publisher.
        """
        if (
            type(timeout) not in (int, float)
            or not 0 <= timeout <= MAX_TRANSFER_SECONDS
        ):
            raise ArtifactError("invalid bounded task snapshot timeout")
        if not self.mutex.acquire(timeout=timeout):
            raise ArtifactError(
                "setup task snapshot unavailable; controller work remains owned"
            )
        try:
            return tuple(self.tasks.values())
        finally:
            self.mutex.release()

    def submit(self, stepid, name, payload):
        with self.lifecycle:
            return self._operation(
                "submit", {"step_id": stepid, "name": name, "payload": payload}
            )

    def status(self, stepid):
        value = self.journal.status(stepid)
        with self.mutex:
            if stepid in self.task_errors:
                raise ArtifactError(
                    "setup task durable failure: " + self.task_errors[stepid]
                )
        return value

    def check_health(self):
        """Fail the fixed job promptly; caller retains all task/resource ownership.

        Call only during normal running, after start, before deliberate shutdown.
        Lifecycle, task and journal checks take their own locks without nesting.
        """
        with self.lifecycle:
            if self.reply_failure is not None:
                raise ArtifactError(self.reply_failure)
            if self.stopping:
                raise ArtifactError("setup peer is stopping or its broker failed")
            if self.process is None or self.thread is None:
                raise ArtifactError("setup peer has not started")
            code = self.process.poll()
            if code is not None:
                raise ArtifactError("setup network worker exited: " + str(code))
            if not self.thread.is_alive():
                raise ArtifactError("setup controller broker is no longer running")
        with self.mutex:
            if self.task_errors:
                step = min(self.task_errors)
                raise ArtifactError(
                    "setup task " + step + " durable failure: " + self.task_errors[step]
                )
        self.journal.check_health()

    def allow_import(self, name, target, *, max_bytes=MAX_TREE, check=None):
        if (
            not isinstance(name, str)
            or not name
            or len(name) > 96
            or any(
                c
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                for c in name
            )
            or type(max_bytes) is not int
            or not 0 <= max_bytes <= MAX_TREE
            or (check is not None and not callable(check))
        ):
            raise ArtifactError("invalid locally allowed setup import")
        target = Path(target).absolute()
        _private_directory(target.parent)
        if name in self.imports or len(self.imports) >= 16:
            raise ArtifactError(
                "setup import name is immutable or population exhausted"
            )

        def bounded_check():
            if self.stopping:
                raise ArtifactError("setup upload cancelled while controller stops")
            if (
                self._rpc_deadline is not None
                and time.monotonic() >= self._rpc_deadline
            ):
                raise ArtifactError("setup upload request deadline elapsed")
            if check is not None:
                check()

        self.imports[name] = {
            "target": target,
            "maximum": max_bytes,
            "upload": None,
            "check": bounded_check,
        }

    def export_tree(
        self, name, path, *, max_bytes=MAX_TREE, check=None, reject_hardlinks=False
    ):
        with self.mutex:
            if len(self.exports) >= 16 and name not in self.exports:
                raise ArtifactError("setup export population exhausted")
        tree = export_tree(
            name,
            path,
            max_bytes=max_bytes,
            check=check,
            reject_hardlinks=reject_hardlinks,
        )
        with self.mutex:
            if len(self.exports) >= 16 and name not in self.exports:
                raise ArtifactError("setup export population exhausted")
            if name in self.exports and self.exports[name].manifest != tree.manifest:
                raise ArtifactError("registered setup export is immutable")
            self.exports[name] = tree
        return tree.manifest

    def _export(self, name):
        if not isinstance(name, str) or not 0 < len(name) <= 96:
            raise ArtifactError("unknown locally registered setup export")
        with self.mutex:
            tree = self.exports.get(name)
        if tree is None:
            raise ArtifactError("unknown locally registered setup export")
        return tree

    def _operation(self, operation, payload):
        if self.stopping:
            raise ArtifactError("setup controller is stopping; new work rejected")
        if not isinstance(payload, dict):
            raise ArtifactError("setup operation requires an object payload")
        if operation in ("import-begin", "import-chunk", "import-finish"):
            keys = {
                "import-begin": {"name", "manifest"},
                "import-chunk": {"name", "digest", "path", "offset", "data"},
                "import-finish": {"name", "digest"},
            }
            if (
                set(payload) != keys[operation]
                or not isinstance(payload["name"], str)
                or payload["name"] not in self.imports
            ):
                raise ArtifactError("setup import is not locally authorized")
            target = self.imports[payload["name"]]
            if operation == "import-begin":
                manifest = payload["manifest"]
                if (
                    not isinstance(manifest, dict)
                    or manifest.get("name") != payload["name"]
                ):
                    raise ArtifactError(
                        "setup upload manifest has another artifact name"
                    )
                if target["upload"] is None:
                    target["upload"] = _Upload(
                        target["target"], manifest, target["maximum"], target["check"]
                    )
                elif target["upload"].manifest != manifest:
                    raise ArtifactError("setup upload identity is immutable")
                return target["upload"].status()
            upload = target["upload"]
            if upload is None or payload["digest"] != upload.manifest["digest"]:
                raise ArtifactError("setup upload digest differs")
            if operation == "import-chunk":
                return upload.chunk(payload["path"], payload["offset"], payload["data"])
            return upload.finish()
        if operation == "export-manifest" and set(payload) == {"name"}:
            return self._export(payload["name"]).manifest
        if operation == "export-chunk" and set(payload) == {
            "name",
            "path",
            "offset",
            "length",
        }:
            data = self._export(payload["name"]).chunk(
                payload["path"], payload["offset"], payload["length"]
            )
            return {
                "data": data,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        if operation == "status" and set(payload) == {"step_id"}:
            return self.status(payload["step_id"])
        if operation == "submit" and set(payload) == {"step_id", "name", "payload"}:
            if (
                not isinstance(payload["name"], str)
                or payload["name"] not in OPERATIONS
                or not isinstance(payload["payload"], dict)
            ):
                raise ArtifactError("unknown fixed setup task")
            # The accepted digest and asynchronous execution own one snapshot;
            # local callers may reuse/mutate their original nested JSON objects.
            payload = copy.deepcopy(payload)
            with self.mutex:
                previous = self.journal.status(payload["step_id"])
                if previous["status"] == "unknown" and any(
                    t.is_alive() for t in self.tasks.values()
                ):
                    raise ArtifactError("another setup task already owns this role")
                value, created = self.journal.accept(
                    payload["step_id"], payload["name"], payload["payload"]
                )
                if created:
                    task = threading.Thread(
                        target=self._task, args=(payload,), daemon=False
                    )
                    self.tasks[payload["step_id"]] = task
                    try:
                        task.start()
                    except Exception as error:
                        context = "setup task could not start: " + str(error)
                        try:
                            self.journal.transition(
                                payload["step_id"], "failed", error=context
                            )
                        except Exception as cleanup:
                            context += "; failed journal commit: " + str(cleanup)
                        self.task_errors[payload["step_id"]] = context[:4096]
                        raise ArtifactError(context) from error
                return value
        if operation not in OPERATIONS:
            raise ArtifactError("unknown fixed setup operation")
        if operation == "sign-csr":
            if set(payload) != {"csr"}:
                raise ArtifactError("invalid client signing payload")
            return {"certificate": self.security.sign_client_csr(payload["csr"])}
        return self.dispatcher(operation, payload)

    def _task(self, payload):
        step = payload["step_id"]
        try:
            self.journal.transition(step, "running")
            result = self._operation(payload["name"], payload["payload"])
            # Validate receipt size before committing completion to disk.
            if len(canonical(result)) > MAX_RECEIPT - 65536:
                raise ArtifactError("setup result exceeds its bound")
            self.journal.transition(step, "completed", result=result)
        except Exception as error:
            try:
                self.journal.transition(step, "failed", error=str(error))
            except Exception as cleanup:
                with self.mutex:
                    self.task_errors[step] = (
                        str(error) + "; failed journal commit: " + str(cleanup)
                    )[:4096]

    def _serve_broker(self):
        stream = self.broker
        try:
            while not self.stopping:
                request = _receive(stream)
                if not isinstance(request, dict) or set(request) != {
                    "operation",
                    "payload",
                    "request_digest",
                    "deadline_ms",
                }:
                    raise ArtifactError("invalid setup broker request")
                with self.lifecycle:
                    self.inflight = request["request_digest"]
                try:
                    deadline_ms = request["deadline_ms"]
                    now = time.monotonic()
                    if (
                        type(deadline_ms) is not int
                        or not 0 < deadline_ms <= 9007199254740991
                        or not now < deadline_ms / 1000 <= now + self.timeout
                    ):
                        raise ArtifactError(
                            "setup broker operation deadline expired/invalid"
                        )
                    self._rpc_deadline = deadline_ms / 1000
                    result = {
                        "ok": True,
                        "result": self._operation(
                            request["operation"], request["payload"]
                        ),
                    }
                except Exception as error:
                    result = {
                        "ok": False,
                        "error": str(error)[:4096],
                        "code": "not_ready"
                        if isinstance(error, SetupNotReady)
                        else "failed",
                    }
                finally:
                    self._rpc_deadline = None
                _send(
                    stream,
                    {"request_digest": request["request_digest"], "response": result},
                )
                receipt = _receive(stream)
                if receipt != {"reply_written": request["request_digest"]}:
                    raise ArtifactError(
                        "setup worker reply receipt has another association"
                    )
                with self.lifecycle:
                    self.inflight = None
                    self.lifecycle.notify_all()
        except (EOFError, OSError, ArtifactError):
            # Process health, rather than a fabricated result, exposes loss.
            self.stopping = True
        finally:
            with self.lifecycle:
                if self.inflight is not None:
                    self.reply_failure = self.reply_failure or (
                        "setup worker lost an in-flight authenticated response"
                    )
                self.inflight = None
                self.lifecycle.notify_all()

    def start(self):
        if os.geteuid() != 0:
            raise ArtifactError(
                "setup network isolation requires a privileged local controller"
            )
        if self.process is not None:
            raise ArtifactError("setup peer already started")
        docker = Path("/var/run/docker.sock")
        if docker.exists() and docker.stat().st_mode & stat.S_IWOTH:
            raise ArtifactError("setup refuses a world-writable Docker socket")
        parent, child = socket.socketpair()
        self.broker = parent
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "ria.setup_peer",
                "--worker-fd",
                str(child.fileno()),
            ],
            pass_fds=(child.fileno(),),
            close_fds=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
            },
        )
        child.close()
        parent.settimeout(self.timeout)
        try:
            _send(
                parent,
                {
                    "invitation": self.invitation,
                    "credentials": self.security.worker_credentials(),
                    "uid": self.worker_uid,
                    "gid": self.worker_gid,
                    "timeout": self.timeout,
                },
            )
            ready = _receive(parent)
            if ready != {"ready": True}:
                raise ArtifactError("setup network worker failed to initialize")
            parent.settimeout(None)
            self.thread = threading.Thread(target=self._serve_broker, daemon=False)
            self.thread.start()
            return self
        except Exception:
            try:
                self.close()
            except Exception:
                pass  # Preserve initialization failure; no tasks exist yet.
            raise

    def begin_close(self):
        """Stop admission; caller still owns every outstanding controller task."""
        deadline = time.monotonic() + self.timeout + 1
        with self.lifecycle:
            self.stopping = True
            while self.inflight is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.reply_failure = (
                        "setup response drain deadline elapsed; completion is ambiguous"
                    )
                    break
                self.lifecycle.wait(timeout=remaining)
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.broker is not None:
            try:
                self.broker.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass  # An already disconnected broker owns no pending response.
            self.broker.close()
            self.broker = None

    def close(self, *, timeout=5):
        """Quiesce before releasing workspace; a timeout retains live ownership.

        Call begin_close, cancel/stop owned fixtures through local policy, then
        close. Raising here never grants permission to free those live resources.
        """
        deadline = time.monotonic() + _timeout(timeout)
        self.begin_close()
        errors = []
        if self.thread is not None:
            try:
                self.thread.join(timeout=max(0, deadline - time.monotonic()))
            except RuntimeError as error:
                errors.append("broker join: " + str(error))
        # A live broker may still publish a task after it passed admission. Do
        # not inspect a task map whose mutex it may hold in foreign filesystem IO.
        if self.thread is not None and self.thread.is_alive():
            raise ArtifactError(
                "setup broker work remains owned; caller must quiesce its resources"
                + ("; " + "; ".join(errors) if errors else "")
            )
        tasks = self.pending_tasks(timeout=max(0, deadline - time.monotonic()))
        for task in tasks:
            try:
                task.join(timeout=max(0, deadline - time.monotonic()))
            except RuntimeError as error:
                errors.append("task join: " + str(error))
        if any(task.is_alive() for task in tasks):
            raise ArtifactError(
                "setup controller work remains owned; caller must quiesce its resources"
                + ("; " + "; ".join(errors) if errors else "")
            )
        if self.reply_failure is not None:
            errors.insert(0, self.reply_failure)
        if errors:
            raise ArtifactError("; ".join(errors))


class PeerClient:
    def __init__(self, invitation, peerendpoint=None, security=None, *, timeout=30):
        self.invitation = validate_invitation(invitation)
        self.host, self.port = endpoint(peerendpoint or invitation["endpoint"])
        if (self.host, self.port) != endpoint(invitation["endpoint"]):
            raise ArtifactError("setup peer endpoint differs from trusted invitation")
        self.security, self.timeout = security, _timeout(timeout)
        self.nonce, self.sequence, self.lock = (
            secrets.token_hex(32),
            0,
            threading.RLock(),
        )
        self.challenge = None
        self.transport, self.reuse = None, False
        self.ssl_context = None
        if self.invitation["tls_enabled"]:
            self.ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            self.ssl_context.minimum_version = ssl.TLSVersion.TLSv1_3
            self.ssl_context.check_hostname = (
                False  # Every fresh connection requires its exact leaf pin.
            )
            self.ssl_context.load_verify_locations(cadata=self.invitation["ca_pem"])

    def _connection(self, timeout):
        if not self.invitation["tls_enabled"]:
            return http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        return http.client.HTTPSConnection(
            self.host, self.port, timeout=timeout, context=self.ssl_context
        )

    def _exchange(self, path, payload, *, deadline=None, binary=None):
        deadline = deadline or time.monotonic() + self.timeout
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ArtifactError("setup request deadline elapsed")
        connection = (
            self.transport
            if self.reuse and self.transport is not None
            else self._connection(remaining)
        )
        timer, retained = None, False
        try:
            fresh = connection.sock is None
            if fresh:
                # Endpoint validation fixed numeric IPv4 at initialization. Own
                # the socket before TCP/TLS so the absolute guard covers both.
                connection.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            guarded_socket = connection.sock

            def expire():
                try:
                    guarded_socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

            timer = threading.Timer(max(0, deadline - time.monotonic()), expire)
            timer.daemon = True
            timer.start()

            def remaining_timeout():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ArtifactError("setup request deadline elapsed")
                connection.sock.settimeout(remaining)

            remaining_timeout()
            if fresh:
                connection.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                connection.sock.connect((self.host, self.port))
                remaining_timeout()
                if self.invitation["tls_enabled"]:
                    connection.sock = self.ssl_context.wrap_socket(
                        connection.sock,
                        server_hostname=self.host,
                        do_handshake_on_connect=False,
                    )
                    guarded_socket = connection.sock
                    remaining_timeout()
                    connection.sock.do_handshake()
                    remaining_timeout()
                    if (
                        hashlib.sha256(
                            connection.sock.getpeercert(binary_form=True)
                        ).hexdigest()
                        != self.invitation["server_leaf_sha256"]
                    ):
                        raise ArtifactError(
                            "setup TLS leaf differs from invitation pin"
                        )
            data = canonical(payload)
            if len(data) > (MAX_MESSAGE if binary is None else MAX_CHUNK_METADATA) or (
                binary is not None
                and (not isinstance(binary, bytes) or len(binary) > CHUNK)
            ):
                raise ArtifactError("setup request exceeds its bound")
            if binary is not None:
                data = struct.pack("!I", len(data)) + data + binary
            remaining_timeout()
            connection.request(
                "POST",
                path,
                body=data,
                headers={
                    "Content-Type": JSON_TYPE if binary is None else BINARY_TYPE,
                    "Connection": "keep-alive" if self.reuse else "close",
                },
            )
            result = connection.getresponse()
            if (
                result.status != 200
                or result.getheader("Transfer-Encoding") is not None
            ):
                raise ArtifactError("setup peer rejected bounded request")
            lengths = result.headers.get_all("Content-Length", [])
            length = lengths[0] if len(lengths) == 1 else None
            if (
                length is None
                or not length.isdecimal()
                or not 0 < int(length) <= MAX_MESSAGE
            ):
                raise ArtifactError("invalid setup response length")
            types = result.headers.get_all("Content-Type", [])
            if len(types) != 1 or types[0] not in (JSON_TYPE, BINARY_TYPE):
                raise ArtifactError("unsupported setup response content version")
            if types[0] == BINARY_TYPE:
                if (
                    path != "/rpc"
                    or payload.get("body", {}).get("operation") != "export-chunk"
                    or not 4 < int(length) <= CHUNK + MAX_CHUNK_METADATA + 4
                ):
                    raise ArtifactError(
                        "binary response requires the fixed export operation"
                    )
                prefix = result.read(4)
                if len(prefix) != 4:
                    raise ArtifactError("truncated setup binary prefix")
                metadata_length = struct.unpack("!I", prefix)[0]
                if (
                    not 0 < metadata_length <= MAX_CHUNK_METADATA
                    or metadata_length + 4 > int(length)
                ):
                    raise ArtifactError("setup binary metadata exceeds its bound")
                packet = loads(
                    result.read(metadata_length),
                    max_bytes=MAX_CHUNK_METADATA,
                    max_nodes=MAX_NODES,
                    max_depth=24,
                )
                if (
                    not isinstance(packet, dict)
                    or set(packet) != {"body", "mac"}
                    or not isinstance(packet["mac"], str)
                    or not hmac.compare_digest(
                        packet["mac"],
                        _mac(self.invitation, "expert-response", packet["body"]),
                    )
                    or not isinstance(packet["body"], dict)
                    or set(packet["body"]) != {"request_digest", "response"}
                    or packet["body"]["request_digest"] != digest(payload["body"])
                ):
                    raise ArtifactError(
                        "binary response authentication/association failed"
                    )
                descriptor = _raw_descriptor(packet["body"])
                if int(length) != 4 + metadata_length + descriptor["bytes"]:
                    raise ArtifactError("setup binary response declaration differs")
                raw = result.read(descriptor["bytes"] + 1)
                _validate_raw(packet["body"], raw)
                value = _Raw(packet, raw)
            else:
                raw = result.read(int(length) + 1)
                if len(raw) != int(length):
                    raise ArtifactError("truncated or oversized setup response")
                value = loads(
                    raw, max_bytes=MAX_MESSAGE, max_nodes=MAX_NODES, max_depth=24
                )
            if time.monotonic() >= deadline:
                raise ArtifactError("setup request deadline elapsed")
            if self.reuse and not result.will_close:
                self.transport, retained = connection, True
            return value
        finally:
            if timer is not None:
                timer.cancel()
                timer.join()
            if not retained:
                connection.close()
                if self.transport is connection:
                    self.transport = None

    @contextmanager
    def _burst(self, timeout):
        """A transfer owns at most one connection; no ambiguous request is retried."""
        deadline = time.monotonic() + _timeout(timeout)
        if not self.lock.acquire(timeout=max(0, deadline - time.monotonic())):
            raise ArtifactError("setup transfer admission deadline elapsed")
        try:
            if self.reuse:
                raise ArtifactError("nested setup transfers are forbidden")
            self.reuse = True
            yield deadline
        finally:
            self.reuse = False
            if self.transport is not None:
                self.transport.close()
                self.transport = None
            self.lock.release()

    def call(self, operation, payload, *, timeout=None):
        deadline = time.monotonic() + _timeout(
            self.timeout if timeout is None else timeout
        )
        if not self.lock.acquire(timeout=max(0, deadline - time.monotonic())):
            raise ArtifactError("setup client admission deadline elapsed")
        try:
            if self.challenge is None:
                challenge = self._exchange("/challenge", {}, deadline=deadline)
                if (
                    not isinstance(challenge, dict)
                    or set(challenge) != {"body", "mac"}
                    or not isinstance(challenge["mac"], str)
                    or not hmac.compare_digest(
                        challenge["mac"],
                        _mac(self.invitation, "expert-challenge", challenge["body"]),
                    )
                ):
                    raise ArtifactError("setup peer challenge authentication failed")
                body = challenge["body"]
                if (
                    not isinstance(body, dict)
                    or set(body) != {"job_id", "tls_enabled", "server_nonce"}
                    or body["job_id"] != self.invitation["job_id"]
                    or body["tls_enabled"] is not self.invitation["tls_enabled"]
                    or not isinstance(body["server_nonce"], str)
                    or len(body["server_nonce"]) != 64
                    or any(c not in "0123456789abcdef" for c in body["server_nonce"])
                ):
                    raise ArtifactError("setup challenge has another job/mode")
                self.challenge = dict(body)
            body = self.challenge
            self.sequence += 1
            request = {
                "job_id": body["job_id"],
                "tls_enabled": body["tls_enabled"],
                "server_nonce": body["server_nonce"],
                "client_nonce": self.nonce,
                "sequence": self.sequence,
                "operation": operation,
                "payload": payload,
            }
            chunk = _split_raw(request)
            if operation == "import-chunk" and chunk is None:
                raise ArtifactError("setup import chunk requires raw binary bytes")
            if chunk is not None:
                request = chunk.value
            reply = self._exchange(
                "/rpc",
                {
                    "body": request,
                    "mac": _mac(self.invitation, "client-request", request),
                },
                deadline=deadline,
                binary=None if chunk is None else chunk.data,
            )
            raw_reply = reply if isinstance(reply, _Raw) else None
            if raw_reply is not None:
                reply = raw_reply.value
            if (
                not isinstance(reply, dict)
                or set(reply) != {"body", "mac"}
                or not isinstance(reply["mac"], str)
                or not hmac.compare_digest(
                    reply["mac"],
                    _mac(self.invitation, "expert-response", reply["body"]),
                )
            ):
                raise ArtifactError("setup response authentication failed")
            response = reply["body"]
            if (
                not isinstance(response, dict)
                or set(response) != {"request_digest", "response"}
                or response["request_digest"] != digest(request)
            ):
                raise ArtifactError("setup response has another request association")
            result = response["response"]
            if raw_reply is not None:
                result = _attach_raw(response, raw_reply.data)["response"]
            if not isinstance(result, dict) or result.get("ok") is not True:
                if isinstance(result, dict) and result.get("code") == "not_ready":
                    raise SetupNotReady(str(result.get("error", "setup not ready")))
                raise ArtifactError(
                    "setup operation failed: "
                    + str(
                        result.get("error", "unknown error")
                        if isinstance(result, dict)
                        else "invalid response"
                    )
                )
            return result["result"]
        finally:
            self.lock.release()

    def submit(self, stepid, name, payload):
        return self.call(
            "submit", {"step_id": stepid, "name": name, "payload": payload}
        )

    def status(self, stepid):
        return self.call("status", {"step_id": stepid})

    def wait(self, stepid, *, timeout=600):
        deadline = time.monotonic() + _timeout(timeout, maximum=MAX_SETUP_SECONDS)
        while time.monotonic() < deadline:
            value = self.call(
                "status",
                {"step_id": stepid},
                timeout=min(self.timeout, deadline - time.monotonic()),
            )
            if value["status"] == "completed":
                return value["result"]
            if value["status"] in ("failed", "unknown"):
                raise ArtifactError(
                    "setup task did not complete: "
                    + str(value.get("error", value["status"]))
                )
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))
        raise ArtifactError(
            "setup task wait deadline elapsed; task ownership remains with controller"
        )

    def fetch_tree(self, name, output, *, max_bytes=MAX_TREE, timeout=3600):
        with self._burst(timeout) as deadline:
            return self._fetch_tree(
                name, output, max_bytes=max_bytes, deadline=deadline
            )

    def _fetch_tree(self, name, output, *, max_bytes, deadline):

        def check():
            if time.monotonic() >= deadline:
                raise ArtifactError("setup transfer deadline elapsed")

        def call(operation, payload):
            return self.call(
                operation,
                payload,
                timeout=min(self.timeout, max(0, deadline - time.monotonic())),
            )

        manifest = call("export-manifest", {"name": name})
        if manifest.get("name") != name:
            raise ArtifactError("setup export response has another artifact identity")

        def reader(path, offset, length):
            reply = call(
                "export-chunk",
                {"name": name, "path": path, "offset": offset, "length": length},
            )
            if not isinstance(reply["data"], bytes):
                raise ArtifactError("setup export did not return binary bytes")
            return reply["data"]

        return import_tree(manifest, output, reader, max_bytes=max_bytes, check=check)

    def upload_tree(self, name, path, *, max_bytes=MAX_TREE, timeout=3600):
        with self._burst(timeout) as deadline:
            return self._upload_tree(name, path, max_bytes=max_bytes, deadline=deadline)

    def _upload_tree(self, name, path, *, max_bytes, deadline):
        def check():
            if time.monotonic() >= deadline:
                raise ArtifactError("setup transfer deadline elapsed")

        tree = export_tree(name, path, max_bytes=max_bytes, check=check)

        def call(operation, payload):
            return self.call(
                operation,
                payload,
                timeout=min(self.timeout, max(0, deadline - time.monotonic())),
            )

        progress = call("import-begin", {"name": name, "manifest": tree.manifest})
        if progress["digest"] != tree.manifest["digest"] or set(
            progress["offsets"]
        ) != set(tree.states):
            raise ArtifactError(
                "setup upload progress has another artifact association"
            )
        if progress["completed"] is True:
            return tree.manifest
        for item in tree.manifest["files"]:
            offset = progress["offsets"][item["path"]]
            if type(offset) is not int or not 0 <= offset <= item["bytes"]:
                raise ArtifactError("invalid setup upload offset")
            while offset < item["bytes"]:
                data = tree.chunk(
                    item["path"], offset, min(CHUNK, item["bytes"] - offset)
                )
                call(
                    "import-chunk",
                    {
                        "name": name,
                        "digest": tree.manifest["digest"],
                        "path": item["path"],
                        "offset": offset,
                        "data": data,
                    },
                )
                offset += len(data)
        result = call(
            "import-finish", {"name": name, "digest": tree.manifest["digest"]}
        )
        if (
            result["digest"] != tree.manifest["digest"]
            or result["completed"] is not True
        ):
            raise ArtifactError("setup upload was not completely published")
        return tree.manifest


def _worker(fd):
    broker = socket.socket(fileno=fd)
    config = _receive(broker)
    uid, gid = config["uid"], config["gid"]
    if os.geteuid() == 0:
        os.setgroups([])
        os.setgid(gid)
        os.setuid(uid)
    if os.geteuid() != uid or os.getegid() != gid or uid == 0 or os.getgroups():
        raise ArtifactError(
            "setup network worker must use an unprivileged exact identity"
        )
    # Linux no_new_privs is irreversible and prevents set-ID/file-capability
    # executables from restoring authority. Check every capability set rather
    # than assuming a parent used the default securebits/keepcaps policy.
    library = ctypes.CDLL(None, use_errno=True)
    prctl = library.prctl
    prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    prctl.restype = ctypes.c_int
    if prctl(38, 1, 0, 0, 0):  # PR_SET_NO_NEW_PRIVS; man2/prctl, fixed Linux ABI.
        raise ArtifactError("setup worker could not prohibit new privileges")
    with open("/proc/self/status", encoding="ascii") as status:
        raw_status = status.read(16385)
    if len(raw_status) > 16384:
        raise ArtifactError("setup worker status exceeded its metadata bound")
    fields = dict(line.split(":", 1) for line in raw_status.splitlines() if ":" in line)
    if (
        any(
            int(fields.get(name, "-1"), 16) != 0
            for name in ("CapInh", "CapPrm", "CapEff", "CapAmb")
        )
        or fields.get("NoNewPrivs", "").strip() != "1"
    ):
        raise ArtifactError("setup network worker must retain no capabilities")
    if os.access("/var/run/docker.sock", os.W_OK):
        raise ArtifactError("setup network worker must have no Docker socket authority")
    invitation = validate_invitation(config["invitation"])
    host, port = endpoint(invitation["endpoint"])
    server_nonce, sessions = secrets.token_hex(32), {}

    def expired(signum, frame):
        raise TimeoutError("setup accepted request deadline elapsed")

    signal.signal(signal.SIGALRM, expired)

    def authenticate(value):
        if (
            not isinstance(value, dict)
            or set(value) != {"body", "mac"}
            or not isinstance(value["mac"], str)
            or not hmac.compare_digest(
                value["mac"], _mac(invitation, "client-request", value["body"])
            )
        ):
            raise ArtifactError("setup authentication failed")
        body = value["body"]
        fields = {
            "job_id",
            "tls_enabled",
            "server_nonce",
            "client_nonce",
            "sequence",
            "operation",
            "payload",
        }
        if (
            not isinstance(body, dict)
            or set(body) != fields
            or body["job_id"] != invitation["job_id"]
            or body["tls_enabled"] is not invitation["tls_enabled"]
            or body["server_nonce"] != server_nonce
            or not isinstance(body["operation"], str)
            or not isinstance(body["payload"], dict)
        ):
            raise ArtifactError("setup job/mode/session differs")
        nonce, sequence = body["client_nonce"], body["sequence"]
        if (
            not isinstance(nonce, str)
            or len(nonce) != 64
            or any(c not in "0123456789abcdef" for c in nonce)
            or type(sequence) is not int
            or sequence <= sessions.get(nonce, 0)
            or sequence > 9007199254740991
        ):
            raise ArtifactError("setup replay/sequence rejected")
        if nonce not in sessions and len(sessions) >= 32:
            raise ArtifactError("setup session population exhausted")
        return body

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        # Small authenticated metadata precedes each raw tail; delayed ACK must
        # not serialize those writes. Retain the original connection wall budget.
        disable_nagle_algorithm = True

        def log_message(self, *arguments):
            pass  # Never send credential-bearing parser input to diagnostics.

        def setup(self):
            self.request.settimeout(config["timeout"])
            self.messages = 0
            super().setup()

        def do_POST(self):
            self.messages += 1
            if self.messages >= 4:
                self.close_connection = True
            dispatched = False
            raw_input = raw_output = None
            try:
                lengths = self.headers.get_all("Content-Length", [])
                if (
                    len(lengths) != 1
                    or not lengths[0].isdecimal()
                    or not 0 < int(lengths[0]) <= MAX_MESSAGE
                    or self.headers.get("Transfer-Encoding") is not None
                ):
                    raise ArtifactError("invalid setup request framing")
                types = self.headers.get_all("Content-Type", [])
                if len(types) != 1 or types[0] not in (JSON_TYPE, BINARY_TYPE):
                    raise ArtifactError("unsupported setup request content version")
                if types[0] == BINARY_TYPE:
                    if (
                        self.path != "/rpc"
                        or not 4 < int(lengths[0]) <= CHUNK + MAX_CHUNK_METADATA + 4
                    ):
                        raise ArtifactError("invalid setup binary request length")
                    prefix = self.rfile.read(4)
                    if len(prefix) != 4:
                        raise ArtifactError("truncated setup binary prefix")
                    metadata_length = struct.unpack("!I", prefix)[0]
                    if (
                        not 0 < metadata_length <= MAX_CHUNK_METADATA
                        or metadata_length + 4 >= int(lengths[0])
                    ):
                        raise ArtifactError("setup binary metadata exceeds its bound")
                    value = loads(
                        self.rfile.read(metadata_length),
                        max_bytes=MAX_CHUNK_METADATA,
                        max_nodes=MAX_NODES,
                        max_depth=24,
                    )
                    body = authenticate(value)
                    if body["operation"] != "import-chunk":
                        raise ArtifactError(
                            "binary request requires fixed import operation"
                        )
                    descriptor = _raw_descriptor(body)
                    if (
                        descriptor["bytes"] == 0
                        or int(lengths[0]) != 4 + metadata_length + descriptor["bytes"]
                    ):
                        raise ArtifactError("setup binary request declaration differs")
                    raw_input = self.rfile.read(descriptor["bytes"])
                    dispatch_body = _restore_raw(body, raw_input)
                else:
                    data = self.rfile.read(int(lengths[0]))
                    if len(data) != int(lengths[0]):
                        raise ArtifactError("truncated setup request")
                    value = loads(
                        data, max_bytes=MAX_MESSAGE, max_nodes=MAX_NODES, max_depth=24
                    )
                if self.path == "/challenge" and value == {}:
                    body = {
                        "job_id": invitation["job_id"],
                        "tls_enabled": invitation["tls_enabled"],
                        "server_nonce": server_nonce,
                    }
                    result = {
                        "body": body,
                        "mac": _mac(invitation, "expert-challenge", body),
                    }
                elif self.path == "/rpc":
                    body = (
                        value["body"] if raw_input is not None else authenticate(value)
                    )
                    nonce, sequence = body["client_nonce"], body["sequence"]
                    sessions[nonce] = sequence
                    if body["operation"] == "import-chunk" and raw_input is None:
                        raise ArtifactError(
                            "setup import chunks require raw binary framing"
                        )
                    if raw_input is None:
                        dispatch_body = body
                    dispatched = True
                    association = digest(body)
                    _send(
                        broker,
                        {
                            "operation": body["operation"],
                            "payload": dispatch_body["payload"],
                            "request_digest": association,
                            # Processes share Linux CLOCK_MONOTONIC. Flooring
                            # milliseconds preserves the original accepted
                            # connection deadline across broker hashing/IO.
                            "deadline_ms": int(
                                (
                                    time.monotonic()
                                    + signal.getitimer(signal.ITIMER_REAL)[0]
                                )
                                * 1000
                            ),
                        },
                    )
                    response = _receive(broker)
                    if (
                        not isinstance(response, dict)
                        or set(response) != {"request_digest", "response"}
                        or response["request_digest"] != association
                    ):
                        os._exit(
                            1
                        )  # Retire an ambiguous broker; never associate a late reply to new work.
                    chunk = _split_raw(response)
                    reply = response if chunk is None else chunk.value
                    raw_output = None if chunk is None else chunk.data
                    result = {
                        "body": reply,
                        "mac": _mac(invitation, "expert-response", reply),
                    }
                else:
                    raise ArtifactError("unknown setup endpoint")
                encoded = canonical(result)
                if len(encoded) > (
                    MAX_MESSAGE if raw_output is None else MAX_CHUNK_METADATA
                ):
                    raise ArtifactError("setup response exceeds its bound")
                self.send_response(200)
                size = (
                    len(encoded)
                    if raw_output is None
                    else 4 + len(encoded) + len(raw_output)
                )
                self.send_header("Content-Length", str(size))
                self.send_header(
                    "Content-Type", JSON_TYPE if raw_output is None else BINARY_TYPE
                )
                if self.close_connection:
                    self.send_header("Connection", "close")
                self.end_headers()
                if raw_output is not None:
                    self.wfile.write(struct.pack("!I", len(encoded)))
                self.wfile.write(encoded)
                if raw_output is not None:
                    self.wfile.write(raw_output)
                self.wfile.flush()
                if dispatched:
                    _send(broker, {"reply_written": association})
            except (ValueError, TypeError, KeyError, EOFError, OSError):
                self.close_connection = True
                if dispatched:
                    os._exit(
                        1
                    )  # Parent retains tasks/fixtures until explicit local cleanup.
                self.send_error(400, "Setup request rejected")

    class Server(HTTPServer):
        address_family = (
            socket.AF_INET6
            if ipaddress.ip_address(host).version == 6
            else socket.AF_INET
        )
        request_queue_size = 2

        def get_request(self):
            connection, address = super().get_request()
            signal.setitimer(signal.ITIMER_REAL, config["timeout"])
            connection.settimeout(config["timeout"])
            if invitation["tls_enabled"]:
                try:
                    connection = self.context.wrap_socket(connection, server_side=True)
                except Exception:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    connection.close()
                    raise
            return connection, address

        def finish_request(self, request, address):
            try:
                super().finish_request(request, address)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)

    with Server((host, port), Handler) as server:
        if invitation["tls_enabled"]:
            with tempfile.TemporaryDirectory(prefix="ria-setup-leaf-") as temporary:
                paths = []
                for name in ("certificate", "key"):
                    path = Path(temporary) / name
                    descriptor = os.open(
                        path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
                    )
                    with os.fdopen(descriptor, "w") as file:
                        file.write(config["credentials"][name])
                    paths.append(path)
                server.context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                server.context.minimum_version = ssl.TLSVersion.TLSv1_3
                server.context.load_cert_chain(*map(str, paths))
        _send(broker, {"ready": True})
        server.serve_forever(poll_interval=0.1)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker-fd":
        raise SystemExit(2)
    _worker(int(sys.argv[2]))
