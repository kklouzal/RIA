"""Defensive safetensors inspection and authenticated bounded byte-range reads."""

import hashlib
import os
import struct
from dataclasses import dataclass

from .identity import ArtifactError, DIGEST, U64_MAX, checked_product, loads, open_regular

DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "U16": 2, "I16": 2, "U32": 4,
               "I32": 4, "U64": 8, "I64": 8, "F16": 2, "BF16": 2, "F32": 4,
               "F64": 8, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1}


@dataclass(frozen=True)
class Tensor:
    name: str
    dtype: str
    shape: tuple
    offset: int
    length: int


@dataclass(frozen=True)
class Shard:
    path: object
    size: int
    data_start: int
    tensors: dict
    snapshot: tuple


def _snapshot(value):
    # Local filesystem/admin metadata is trusted; the independently supplied
    # SHA256 authenticates content. ctime catches mutate-and-restore, while inode
    # identity catches pathname replacement. Do not use atime: reads change it.
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def validate_source_snapshot(shard, stream=None):
    """Reject changes to a verified source before bytes or provenance are used."""
    try:
        current = _snapshot(os.stat(shard.path, follow_symlinks=False))
        if current != shard.snapshot or stream is not None and _snapshot(os.fstat(stream.fileno())) != shard.snapshot:
            raise ArtifactError("verified source snapshot changed")
    except OSError as exc:
        raise ArtifactError("verified source snapshot is unavailable") from exc


def exact_read(stream, size):
    result = bytearray()
    while len(result) < size:
        block = stream.read(size - len(result))
        if not block:
            raise ArtifactError("truncated tensor data")
        result.extend(block)
    return bytes(result)


def inspect(path, *, max_header_bytes=16 << 20, max_tensors=100000, max_dimensions=8,
            expected_sha256=None):
    with open_regular(path) as stream:
        snapshot = _snapshot(os.fstat(stream.fileno()))
        size = snapshot[2]
        if expected_sha256 is not None:
            if not isinstance(expected_sha256, str) or not DIGEST.fullmatch(expected_sha256):
                raise ArtifactError("source SHA256 identity is invalid")
            actual = hashlib.sha256()
            while block := stream.read(1 << 20):
                actual.update(block)
            if actual.hexdigest() != expected_sha256:
                raise ArtifactError("source shard hash mismatch")
            stream.seek(0)
        if size < 8:
            raise ArtifactError("truncated safetensors prefix")
        header_bytes = struct.unpack("<Q", exact_read(stream, 8))[0]
        if not 2 <= header_bytes <= max_header_bytes or header_bytes > size - 8:
            raise ArtifactError("invalid safetensors header size")
        raw = exact_read(stream, header_bytes)
        if raw[:1] != b"{":
            raise ArtifactError("safetensors header must begin with an object")
        header = loads(raw, project=False, max_bytes=max_header_bytes, max_nodes=max_tensors * 30 + 100)
        witness = Shard(path, size, 8 + header_bytes, {}, snapshot)
        validate_source_snapshot(witness, stream)
    if not isinstance(header, dict) or len(header) > max_tensors + 1:
        raise ArtifactError("invalid safetensors tensor inventory")
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items()):
        raise ArtifactError("safetensors metadata must be string pairs")
    data_start = 8 + header_bytes
    tensors = {}
    intervals = []
    for name, descriptor in header.items():
        if not name or not isinstance(descriptor, dict) or set(descriptor) != {"dtype", "shape", "data_offsets"}:
            raise ArtifactError("invalid safetensors tensor descriptor")
        dtype, shape, offsets = (descriptor[key] for key in ("dtype", "shape", "data_offsets"))
        if dtype not in DTYPE_BYTES or not isinstance(shape, list) or len(shape) > max_dimensions:
            raise ArtifactError("unsupported dtype or tensor rank")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(v) is not int or not 0 <= v <= U64_MAX for v in offsets):
            raise ArtifactError("invalid safetensors data offsets")
        start, end = offsets
        length = checked_product([checked_product(shape), DTYPE_BYTES[dtype]])
        if end < start or end - start != length or end > size - data_start:
            raise ArtifactError("tensor byte shape or bounds mismatch")
        tensors[name] = Tensor(name, dtype, tuple(shape), start, length)
        if length:
            intervals.append((start, end))
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ArtifactError("overlap or unindexed bytes in safetensors data")
        cursor = end
    if cursor != size - data_start:
        raise ArtifactError("unindexed trailing safetensors data")
    result = Shard(path, size, data_start, tensors, snapshot)
    validate_source_snapshot(result)
    return result


def tensor_blocks(shard, tensor, *, block_bytes=1 << 20):
    if not 0 < block_bytes <= 16 << 20:
        raise ArtifactError("invalid streaming block bound")
    with open_regular(shard.path) as stream:
        validate_source_snapshot(shard, stream)
        stream.seek(shard.data_start + tensor.offset)
        remaining = tensor.length
        while remaining:
            block = exact_read(stream, min(remaining, block_bytes))
            remaining -= len(block)
            validate_source_snapshot(shard, stream)
            yield block
        validate_source_snapshot(shard, stream)


def chunk_index(path, chunk_size):
    if type(chunk_size) is not int or not 1 <= chunk_size <= 4 << 20:
        raise ArtifactError("chunk size exceeds authenticated bulk allowance")
    full = hashlib.sha256()
    hashes = []
    size = 0
    with open_regular(path) as stream:
        expected_size = os.fstat(stream.fileno()).st_size
        if (expected_size + chunk_size - 1) // chunk_size > 1000000:
            raise ArtifactError("chunk index exceeds bounded inventory capacity")
        while block := stream.read(chunk_size):
            full.update(block)
            size += len(block)
            hashes.append(hashlib.sha256(block).hexdigest())
        if size != expected_size:
            raise ArtifactError("shard changed during chunk indexing")
    return {"length": str(size), "sha256": full.hexdigest(), "chunk_size": chunk_size, "chunk_hashes": hashes}


def authenticated_range(path, descriptor, start, length, grants, *, max_bytes=16 << 20):
    """Verify complete chunks first; every returned chunk byte must be authorized.

    Grants are absolute half-open file ranges from a trusted preparation policy.
    This intentionally charges overfetch and refuses tensor-only grants that leak
    a neighboring tensor through chunk verification.
    """
    from .identity import u64
    size, chunk = u64(descriptor["length"]), descriptor["chunk_size"]
    hashes = descriptor["chunk_hashes"]
    if type(chunk) is not int or not 1 <= chunk <= 4 << 20 or len(hashes) != (size + chunk - 1) // chunk:
        raise ArtifactError("invalid chunk index")
    if any(type(v) is not int or v < 0 for v in (start, length)) or start > size or length > size - start:
        raise ArtifactError("range outside shard")
    if length == 0:
        return b""
    first, last = start // chunk, (start + length - 1) // chunk
    begin, finish = first * chunk, min((last + 1) * chunk, size)
    if finish - begin > max_bytes:
        raise ArtifactError("authenticated transfer scratch limit exceeded")
    normalized = sorted(grants)
    cursor = begin
    for lower, upper in normalized:
        if type(lower) is not int or type(upper) is not int or lower < 0 or upper < lower or upper > size:
            raise ArtifactError("invalid preparation grant")
        if lower <= cursor < upper:
            cursor = max(cursor, upper)
    if cursor < finish:
        raise ArtifactError("grant does not authorize complete verification chunks")
    data = bytearray()
    with open_regular(path) as stream:
        if os.fstat(stream.fileno()).st_size != size:
            raise ArtifactError("shard size changed")
        stream.seek(begin)
        for number in range(first, last + 1):
            block = exact_read(stream, min(chunk, size - number * chunk))
            if hashlib.sha256(block).hexdigest() != hashes[number]:
                raise ArtifactError("verification chunk hash mismatch")
            data.extend(block)
    relative = start - begin
    return bytes(data[relative:relative + length])
