"""Bounded duplicate-aware JSON, RFC 8785 identities and durable publication."""

import hashlib
import json
import math
import os
import re
import stat
import tempfile
from pathlib import Path

import rfc8785

SAFE_INTEGER = 9007199254740991
U64_MAX = (1 << 64) - 1
DIGEST = re.compile(r"[0-9a-f]{64}\Z")
DECIMAL = re.compile(r"(?:0|[1-9][0-9]*)\Z")


class ArtifactError(ValueError):
    """An input cannot satisfy the immutable artifact or deployment contract."""


def u64(value):
    if not isinstance(value, str) or not DECIMAL.fullmatch(value) or len(value) > 20:
        raise ArtifactError("u64 must be a canonical decimal string")
    number = int(value)
    if number > U64_MAX:
        raise ArtifactError("u64 exceeds its range")
    return number


def checked_product(values, limit=U64_MAX):
    total = 1
    for value in values:
        if type(value) is not int or value < 0 or value > limit:
            raise ArtifactError("invalid dimension")
        if value and total > limit // value:
            raise ArtifactError("dimension product overflow")
        total *= value
    return total


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def check_json(value, *, project=True, max_depth=64, max_nodes=32000000):
    work = [(value, 0)]
    count = 0
    while work:
        item, depth = work.pop()
        count += 1
        if count > max_nodes or depth > max_depth:
            raise ArtifactError("JSON complexity limit exceeded")
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ArtifactError("JSON object key must be a string")
            work.extend((key, depth + 1) for key in item)
            work.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, (list, tuple)):
            work.extend((child, depth + 1) for child in item)
        elif isinstance(item, str):
            try:
                item.encode("utf-8", errors="strict")
            except UnicodeError as exc:
                raise ArtifactError("malformed Unicode") from exc
        elif type(item) is int:
            if project and abs(item) > SAFE_INTEGER:
                raise ArtifactError("ordinary JSON integer exceeds safe range")
        elif type(item) is float:
            if not math.isfinite(item):
                raise ArtifactError("nonfinite JSON number")
        elif item is not None and type(item) is not bool:
            raise ArtifactError("unsupported JSON value")


def loads(raw, *, project=True, max_bytes=16 << 20, max_depth=64, max_nodes=2000000):
    if isinstance(raw, str):
        raw = raw.encode("utf-8", errors="strict")
    if len(raw) > max_bytes:
        raise ArtifactError("JSON byte limit exceeded")
    # Check depth before the recursive standard decoder allocates nested objects.
    depth = 0
    quoted = escaped = False
    for byte in raw:
        if quoted:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quoted = False
        elif byte == 34:
            quoted = True
        elif byte in (91, 123):
            depth += 1
            if depth > max_depth:
                raise ArtifactError("JSON nesting limit exceeded")
        elif byte in (93, 125):
            depth -= 1
    try:
        result = json.loads(raw.decode("utf-8", errors="strict"), object_pairs_hook=_pairs,
                            parse_constant=lambda _: (_ for _ in ()).throw(ArtifactError("nonfinite JSON")))
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ArtifactError(str(exc)) from exc
    check_json(result, project=project, max_depth=max_depth, max_nodes=max_nodes)
    return result


def read_json(path, **kwargs):
    max_bytes = kwargs.get("max_bytes", 16 << 20)
    with open_regular(path) as stream:
        raw = stream.read(max_bytes + 1)
    return loads(raw, **kwargs)


def canonical(value):
    check_json(value)
    try:
        return rfc8785.dumps(value)
    except (rfc8785.CanonicalizationError, ValueError) as exc:
        raise ArtifactError(str(exc)) from exc


def digest(value):
    # Only top-level fields are removed. Every referenced child digest stays bound.
    if not isinstance(value, dict):
        raise ArtifactError("identity document must be an object")
    return hashlib.sha256(canonical({k: v for k, v in value.items() if k not in ("digest", "signatures")})).hexdigest()


def seal(value):
    result = dict(value)
    result["digest"] = digest(result)
    return result


def verify_identity(value, expected=None):
    claimed = value.get("digest")
    if not isinstance(claimed, str) or not DIGEST.fullmatch(claimed):
        raise ArtifactError("missing or malformed digest")
    actual = digest(value)
    if actual != claimed or (expected is not None and actual != expected):
        raise ArtifactError("JSON identity mismatch")
    return actual


def open_regular(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ArtifactError("expected a regular file")
        return os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise


def within(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ArtifactError("artifact path must be relative")
    if any(part in ("..", ".") for part in relative.split("/")) or "\\" in relative:
        raise ArtifactError("invalid artifact path")
    root = Path(root).resolve(strict=True)
    candidate = root / relative
    # Reject symlinks at every component, even if their destination is inside root.
    current = root
    for part in Path(relative).parts:
        current /= part
        if current.is_symlink():
            raise ArtifactError("artifact path contains symlink")
    if not candidate.resolve().is_relative_to(root):
        raise ArtifactError("artifact escapes authorized root")
    return candidate


def hash_file(path, block_bytes=1 << 20):
    result = hashlib.sha256()
    with open_regular(path) as stream:
        while block := stream.read(block_bytes):
            result.update(block)
    return result.hexdigest()


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_bytes(path, data, *, mode=0o644):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".ria-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            os.fchmod(stream.fileno(), mode)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    atomic_bytes(path, canonical(value) + b"\n")
