"""Compact role extraction from a trusted prepared root, with chunk authentication.

This path never hashes or imports an entire source weight shard. A caller must
supply the server root identity through its approved provisioning channel. Whole
verification chunks are authorized against the root's complete tensor inventory
before any selected bytes are emitted; cache selections are complete experts.
"""

import hashlib
import re
from pathlib import Path

from .identity import (ArtifactError, atomic_bytes, atomic_json, canonical, digest,
                       open_regular, read_json, seal, u64, verify_identity, within)
from .preparation import (_write_bundle, publish_tensor_pages, stable_id,
                          verify_metadata_graph, verify_package)
from .safetensors import authenticated_range, inspect
from .schemas import validate


def _inventory(root, manifest):
    tensors, shards = list(manifest["tensors"]), list(manifest["shards"])
    for reference in manifest["tensor_pages"]:
        page = read_json(within(root, reference["path"]))
        verify_identity(page, reference["digest"])
        validate("tensor-page", page)
        tensors.extend(page["tensors"])
        shards.extend(page["shards"])
    validate("physical-layout", {"schema_revision": 1, "logical_model_digest": manifest["logical_model_digest"],
        "operator_contract_digest": manifest["operator_contract_digest"], "backend": "source", "tensors": tensors,
        "tensor_pages": [], "digest": manifest["layout_digest"]})
    shard_map = {item["id"]: item for item in shards}
    if len(shard_map) != len(shards) or len({item["path"] for item in shards}) != len(shards):
        raise ArtifactError("duplicate prepared shard identity")
    return {item["name"]: item for item in tensors}, shard_map


def _selection(tensors, selected_names):
    selected = set(selected_names)
    if any(name not in tensors or tensors[name]["placement"] != "cache" for name in selected):
        raise ArtifactError("cache selection must name explicitly cacheable tensors")
    expert_keys = set()
    for name in selected:
        match = re.fullmatch(r"(layers\.[0-9]+\.ffn\.experts\.[0-9]+)\.w[123]\.weight", name)
        if not match:
            raise ArtifactError("client cache selection requires complete routed expert projections")
        expert_keys.add(match[1])
    for key in expert_keys:
        names = {key + "." + projection + ".weight" for projection in ("w1", "w2", "w3")}
        if not names <= selected:
            raise ArtifactError("client cache selection omits a gate/up/down projection")
    selected.update(name for name, item in tensors.items() if item["placement"] in ("client", "both"))
    by_id = {item["id"]: item for item in tensors.values()}
    pending = list(selected)
    while pending:
        item = tensors[pending.pop()]
        references = item["scale_ids"] + ([item["alias_of"]] if item["alias_of"] is not None else [])
        for reference in references:
            target = by_id.get(reference)
            if target is None or target["placement"] not in ("client", "both", "cache"):
                raise ArtifactError("selected client tensor has unavailable dependencies")
            if target["name"] not in selected:
                selected.add(target["name"])
                pending.append(target["name"])
    if not selected:
        raise ArtifactError("client package has no owned tensors")
    return selected


def client_package(server_root, trusted_digest, output, *, selected_names=(), chunk_size=4 << 20,
                   max_shard_bytes=1 << 30):
    server_root, output = Path(server_root), Path(output)
    manifest = read_json(server_root / "manifest.json", max_bytes=256 << 10)
    validate("manifest", manifest)
    verify_identity(manifest, trusted_digest)
    if manifest["role"] != "server":
        raise ArtifactError("compact client extraction requires an authenticated server package")
    verify_metadata_graph(server_root, manifest["metadata"])
    tensors, shards = _inventory(server_root, manifest)
    selected = _selection(tensors, selected_names)
    selection_digest = digest({"names": sorted(selected)})
    if type(chunk_size) is not int or not 1 <= chunk_size <= 4 << 20 or type(max_shard_bytes) is not int or max_shard_bytes < 2 * chunk_size:
        raise ArtifactError("invalid compact bundle bounds")
    output.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        raise ArtifactError("client output must not be a symlink")
    if (output / "manifest.json").exists():
        existing = verify_package(output)
        provenance = read_json(output / "client-source.json")
        if (provenance.get("server_manifest_digest"), provenance.get("selection_digest")) != (trusted_digest, selection_digest):
            raise ArtifactError("existing client package has another source or selection")
        return existing
    (output / "tensors").mkdir(exist_ok=True)
    grants = {key: [(0, u64(item["data_start"]))] for key, item in shards.items()}
    for item in tensors.values():
        if item["shard"] not in shards:
            raise ArtifactError("inventory references an absent shard")
        # Public zero padding can be authenticated alongside any adjacent owned
        # population. Server-only values never enter the client grant union.
        public_padding = item["operation"] == "inactive" and item["name"].startswith("__ria_padding_")
        if item["placement"] in ("client", "both", "cache") or public_padding:
            shard = shards[item["shard"]]
            lower = u64(shard["data_start"]) + u64(item["offset"])
            grants[item["shard"]].append((lower, lower + u64(item["length"])))
    inspected, tasks = {}, []
    for name in sorted(selected):
        item = tensors[name]
        source = shards[item["shard"]]
        source_path = within(server_root, source["path"])
        if item["shard"] not in inspected:
            inspected[item["shard"]] = inspect(source_path)
        actual = inspected[item["shard"]]
        value = actual.tensors.get(name)
        if (actual.size, actual.data_start) != (u64(source["length"]), u64(source["data_start"])) or value is None or (value.offset, value.length, value.dtype, list(value.shape)) != (u64(item["offset"]), u64(item["length"]), item["dtype"], item["physical_shape"]):
            raise ArtifactError("trusted tensor descriptor disagrees with source header")
        def blocks(item=item, source=source, source_path=source_path):
            start, remaining = u64(source["data_start"]) + u64(item["offset"]), u64(item["length"])
            while remaining:
                # At most two extra complete chunks are retained. Authentication
                # precedes this yield and immutable publication follows all yields.
                count = min(remaining, 4 << 20)
                yield authenticated_range(source_path, source, start, count, grants[item["shard"]])
                start += count
                remaining -= count
        tasks.append({"name": name, "dtype": item["dtype"], "shape": item["physical_shape"], "length": u64(item["length"]),
            "blocks": blocks, "descriptor": item, "grant_group": "client-authorized-population", "padding": False})
    bundles, current, length = [], [], chunk_size
    for task in tasks:
        allowance = task["length"] + len(task["name"].encode()) * 4 + 512
        if current and length + allowance > max_shard_bytes:
            bundles.append(current)
            current, length = [], chunk_size
        if length + allowance > max_shard_bytes:
            raise ArtifactError("selected tensor exceeds compact package shard cap")
        current.append(task)
        length += allowance
    if current:
        bundles.append(current)
    if len(bundles) > 4096:
        raise ArtifactError("compact package exceeds coarse shard cap")
    from .safetensors import chunk_index
    descriptors, output_shards = [], []
    for index, bundle in enumerate(bundles):
        shard_id = stable_id(f"client:{trusted_digest}:{selection_digest}:{index}")
        relative = f"tensors/{shard_id}.safetensors"
        target = within(output, relative)
        records, data_start = _write_bundle(target, bundle, chunk_size, max_shard_bytes)
        for task, offset, payload_hash in records:
            if task["padding"]:
                descriptors.append({"id": stable_id(relative + ":" + task["name"]), "name": task["name"],
                    "logical_shape": task["shape"], "physical_shape": task["shape"], "dtype": "U8", "format": "plain",
                    "layout": "row_major_le", "shard": shard_id, "offset": str(offset), "length": str(task["length"]),
                    "sha256": payload_hash, "source_sha256": payload_hash, "scale_ids": [], "operation": "inactive",
                    "placement": "inactive", "alias_of": None, "byte_order": "little", "bytes_per_block": 1,
                    "block_values": 1, "group_shape": []})
                continue
            if payload_hash != task["descriptor"]["sha256"]:
                raise ArtifactError("authenticated client tensor payload identity mismatch")
            descriptors.append({**task["descriptor"], "shard": shard_id, "offset": str(offset)})
        output_shards.append({"id": shard_id, "path": relative, "data_start": str(data_start), **chunk_index(target, chunk_size)})
    pages = publish_tensor_pages(output, descriptors, output_shards)
    layout = seal({"schema_revision": 1, "logical_model_digest": manifest["logical_model_digest"],
        "operator_contract_digest": manifest["operator_contract_digest"], "backend": "source", "tensors": [], "tensor_pages": pages})
    atomic_json(output / "layout.json", layout)
    metadata = [{"path": "layout.json", "digest": layout["digest"], "kind": "layout"}, *pages]
    for reference in manifest["metadata"]:
        document = read_json(within(server_root, reference["path"]))
        if reference["kind"] == "operator":
            atomic_json(output / "operator.json", document)
            metadata.append({"path": "operator.json", "digest": document["digest"], "kind": "operator"})
        elif set(document) == {"schema_revision", "path", "source_path", "sha256", "length", "digest"}:
            if document["source_path"].endswith("model.safetensors.index.json"):
                continue  # Entire checkpoint index is unnecessary client payload.
            with open_regular(within(server_root, document["path"])) as stream:
                data = stream.read((16 << 20) + 1)
            if len(data) > 16 << 20 or hashlib.sha256(data).hexdigest() != document["sha256"]:
                raise ArtifactError("client compact metadata integrity mismatch")
            relative = f"metadata/{len(metadata)}.bin"
            wrapper_relative = f"metadata/{len(metadata)}.json"
            atomic_bytes(output / relative, data)
            wrapper = seal({**document, "path": relative})
            atomic_json(output / wrapper_relative, wrapper)
            metadata.append({"path": wrapper_relative, "digest": wrapper["digest"], "kind": "metadata"})
    provenance = seal({"schema_revision": 1, "server_manifest_digest": trusted_digest, "selection_digest": selection_digest,
        "excluded_inventory": "server-only and unselected cache tensors remain authenticated by the provisioned server root"})
    atomic_json(output / "client-source.json", provenance)
    metadata.append({"path": "client-source.json", "digest": provenance["digest"], "kind": "metadata"})
    result = seal({**manifest, "role": "client", "layout_digest": layout["digest"], "tensors": [], "shards": [],
        "tensor_pages": pages, "metadata": metadata, "feature_exclusions": manifest["feature_exclusions"] + [provenance["excluded_inventory"]]})
    if len(canonical(result)) > 256 << 10:
        raise ArtifactError("client manifest exceeds compact root cap")
    verify_package(output, result)
    atomic_json(output / "manifest.json", result)
    return result
