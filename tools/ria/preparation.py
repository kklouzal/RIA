"""Streaming, manifest-last preparation from an explicit reviewed tensor recipe."""

import hashlib
import os
import shutil
import stat
import struct
import tempfile
import math
import re
import sys
from pathlib import Path

from .identity import (ArtifactError, atomic_json, canonical, checked_product, digest,
                       hash_file, loads, open_regular, read_json, read_verified_bytes, seal, sync_directory,
                       u64, verify_identity, within)
from .numeric import (decode_matrix_rows, encode_bf16, encode_fp8_block32)
from .safetensors import (DTYPE_BYTES, chunk_index, exact_read, inspect, tensor_blocks,
                         validate_source_snapshot)
from .schemas import (FORMAT, OPERATION, PROFILE, SHA, SHAPE, TEXT, POS, REV, array,
                      obj, validate)

RECIPE_SCHEMA = obj({"schema_revision": REV, "model_id": {"const": "deepseek-ai/DeepSeek-V4.1-Flash"},
    "source_revision": {"type": "string", "pattern": "^[0-9a-f]{40}$"}, "profile": PROFILE,
    "logical_model_digest": SHA, "tokenizer_digest": SHA, "encoding_digest": SHA,
    "operator_contract": TEXT, "sources": array(obj({"path": TEXT, "sha256": SHA}), minimum=1),
    "tensors": array(obj({"name": TEXT, "operation": OPERATION, "placement": {"enum": ["client", "server", "both", "cache", "inactive"]},
        "format": FORMAT, "logical_shape": SHAPE, "scale_names": array(TEXT, maximum=32),
        "convert": {"type": "boolean"}, "grant_group": TEXT, "repack": {"enum": ["none", "engram_packed"]}, "weight_global_scale_bits": {"type": "string", "pattern": "^[0-9a-f]{8}$"},
        "activation_global_scale_bits": {"type": "string", "pattern": "^[0-9a-f]{8}$"},
        "inactive_reason": TEXT, "weight_global_scale_tensor": TEXT, "activation_scale_tensor": TEXT,
        "activation_scale_group": TEXT}, ("weight_global_scale_bits", "activation_global_scale_bits", "inactive_reason", "repack", "grant_group",
        "weight_global_scale_tensor", "activation_scale_tensor", "activation_scale_group")), minimum=1),
    "metadata": array(obj({"path": TEXT, "sha256": SHA}), maximum=4096), "feature_exclusions": array(TEXT, maximum=100),
    "max_shard_bytes": POS, "chunk_size": POS, "scratch_bytes": POS, "digest": SHA})


def stable_id(name):
    return str(int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "big"))


def bits_float(text):
    return struct.unpack("<f", struct.pack("<I", int(text, 16)))[0]


def target_operation(name):
    """Fixed V4.1 tensor-to-operation mapping; unknown essential names fail."""
    if name.startswith("mtp."):
        return "inactive"
    if name in ("engram.token_map", "engram.primes", "engram.offsets", "engram.multipliers", "engram.pad_id"):
        return "engram"
    base = re.sub(r"\.(?:scale|weight_scale(?:_2)?|input_scale)$", ".weight", name)
    is_scale = base != name
    expert = re.fullmatch(r"layers\.([0-9]+)\.ffn\.(experts\.([0-9]+)|shared_experts)\.(w[123])\.weight", base)
    if expert:
        if int(expert[1]) >= 40 or (expert[3] is not None and int(expert[3]) >= 384):
            raise ArtifactError("target expert ID exceeds pinned inventory")
        if is_scale:
            return "scale"
        return ("expert_" if expert[3] is not None else "shared_") + {"w1": "gate", "w2": "down", "w3": "up"}[expert[4]]
    layer = re.fullmatch(r"layers\.([0-9]+)\.(.+)", base)
    if layer:
        if int(layer[1]) >= 40:
            raise ArtifactError("target backbone layer exceeds pinned inventory")
        suffix = layer[2]
        if suffix in ("attn_norm.weight", "ffn_norm.weight"):
            return "norm"
        if suffix in ("hc_attn_base", "hc_attn_fn", "hc_attn_scale", "hc_ffn_base", "hc_ffn_fn", "hc_ffn_scale"):
            return "mhc"
        if suffix in ("ffn.gate.weight", "ffn.gate.bias", "ffn.gate.bias_vl"):
            return "router"
        if suffix in ("engram.embed.weight", "engram.q_weight", "engram.k_weight", "engram.wkv.weight"):
            return "scale" if is_scale else "engram"
        if suffix in ("attn.indexer.k_norm.weight", "attn.indexer.weights_proj.weight", "attn.indexer.wk.weight", "attn.indexer.wq_b.weight"):
            return "scale" if is_scale else "index"
        if suffix in ("attn.attn_sink", "attn.compressor.norm.weight", "attn.compressor.wgate.weight", "attn.compressor.wkv.weight",
                      "attn.kv_norm.weight", "attn.q_norm.weight", "attn.wkv.weight", "attn.wo_a.weight", "attn.wo_b.weight", "attn.wq_a.weight", "attn.wq_b.weight"):
            return "scale" if is_scale else "attention"
    if name == "embed.weight":
        return "embedding"
    if name == "head.weight":
        return "output"
    if name == "norm.weight":
        return "norm"
    if name in ("image_start", "image_end", "image_newline") or re.fullmatch(r"aligner\.w[12]\.(?:weight|bias)", name):
        return "vision"
    if name == "vision.norm.weight" or re.fullmatch(r"vision\.patch_embed\.proj\.(?:weight|bias)", name):
        return "vision"
    vision = re.fullmatch(r"vision\.blocks\.([0-9]+)\.(norm[12]\.weight|attn\.(wqkv|wo)\.(weight|bias)|mlp\.w[12]\.weight)", name)
    if vision and int(vision[1]) < 32:
        return "vision"
    raise ArtifactError(f"unrecognized required target tensor: {name}")


def _raw_slice(shard, tensor, start, length):
    if start < 0 or length < 0 or start > tensor.length or length > tensor.length - start:
        raise ArtifactError("conversion slice exceeds source tensor")
    with open_regular(shard.path) as stream:
        validate_source_snapshot(shard, stream)
        stream.seek(shard.data_start + tensor.offset + start)
        result = exact_read(stream, length)
        validate_source_snapshot(shard, stream)
        return result


def _validate_sources(tensors):
    for shard in {str(shard.path): shard for shard, _ in tensors.values()}.values():
        validate_source_snapshot(shard)


def _conversion_blocks(rule, source, tensors, profile, scratch_bytes):
    shard, tensor = source
    logical = rule["logical_shape"]
    if len(logical) != 2 or rule["format"] not in ("source_mxfp4", "nvfp4", "fp8_block32", "bf16"):
        raise ArtifactError("conversion requires a recognized row-major matrix")
    rows, columns = logical
    if not rows or not columns or columns * 32 * 80 > scratch_bytes:
        raise ArtifactError("conversion tile exceeds declared scratch bound")
    if rule["format"] in ("source_mxfp4", "nvfp4", "fp8_block32") and len(rule["scale_names"]) != 1:
        raise ArtifactError("conversion needs one explicit block scale tensor")
    scale_source = tensors[rule["scale_names"][0]] if rule["scale_names"] else None
    stride = (columns + 1) // 2 if rule["format"] in ("source_mxfp4", "nvfp4") else columns * (2 if rule["format"] == "bf16" else 1)
    if tensor.length != rows * stride:
        raise ArtifactError("source value layout does not match logical matrix")
    global_scale = bits_float(rule.get("weight_global_scale_bits", "3f800000"))
    for row in range(0, rows, 32):
        count = min(32, rows - row)
        values = _raw_slice(shard, tensor, row * stride, count * stride)
        scales = b""
        if scale_source:
            group = 16 if rule["format"] == "nvfp4" else 32
            scale_columns = (columns + group - 1) // group
            if rule["format"] == "fp8_block32":
                scale_offset, scale_length = (row // 32) * scale_columns, scale_columns
            else:
                scale_offset, scale_length = row * scale_columns, count * scale_columns
            scales = _raw_slice(*scale_source, scale_offset, scale_length)
        decoded = decode_matrix_rows(values, scales, count, columns, rule["format"], global_scale, first_row=row)
        if profile == "bf16":
            yield encode_bf16(decoded), b""
        elif profile == "fp8":
            yield encode_fp8_block32(decoded)
        else:
            raise ArtifactError("NVFP4 strict preparation only imports publisher bytes")


def _engram_blocks(source, scale_source, rows, scratch_bytes):
    shard, tensor = source
    scale_shard, scale_tensor = scale_source
    if tensor.dtype != "F8_E4M3" or tensor.shape != (rows, 256) or scale_tensor.dtype not in ("F8_E8M0", "U8") or scale_tensor.shape != (rows, 8):
        raise ArtifactError("Engram packing requires exact FP8/UE8M0 row256+8 layout")
    count = max(1, min(4096, scratch_bytes // 1024))
    for lower in range(0, rows, count):
        size = min(count, rows - lower)
        values = _raw_slice(shard, tensor, lower * 256, size * 256)
        scales = _raw_slice(scale_shard, scale_tensor, lower * 8, size * 8)
        output = bytearray(size * 264)
        for row in range(size):
            output[row * 264:row * 264 + 256] = values[row * 256:row * 256 + 256]
            output[row * 264 + 256:(row + 1) * 264] = scales[row * 8:(row + 1) * 8]
        yield output


def publish_tensor_pages(output, tensors, shards, *, max_bytes=16 << 20):
    pages, current_tensors, current_shards = [], [], []
    shard_map = {item["id"]: item for item in shards}
    used = set()
    current_size = 128
    def flush():
        page = seal({"schema_revision": 1, "tensors": list(current_tensors), "shards": list(current_shards)})
        validate("tensor-page", page)
        if len(canonical(page)) > max_bytes:
            raise ArtifactError("tensor page exceeds metadata bound")
        path = f"index/{len(pages)}.json"
        atomic_json(output / path, page)
        pages.append({"path": path, "digest": page["digest"], "kind": "index"})
    for tensor in tensors:
        shard = shard_map[tensor["shard"]]
        new_shard = shard["path"] not in used
        size = len(canonical(tensor)) + (len(canonical(shard)) if new_shard else 0) + 8
        if size + 128 > max_bytes:
            raise ArtifactError("single tensor chunk index exceeds metadata page bound")
        if current_tensors and current_size + size > max_bytes - 128:
            flush()
            current_tensors, current_shards, current_size = [], [], 128
        current_tensors.append(tensor)
        if new_shard:
            current_shards.append(shard)
            used.add(shard["path"])
        current_size += size
    if current_tensors:
        flush()
    return pages


def publish_provenance(output, recipe, converter_digest, selection_digest, rules):
    """Bound the complete inactive/reduction inventory instead of one huge JSON."""
    pages, entries, size = [], [], 128
    def flush():
        page = seal({"schema_revision": 1, "entries": list(entries)})
        relative = f"provenance/{len(pages)}.json"
        if len(canonical(page)) > 16 << 20:
            raise ArtifactError("provenance page exceeds metadata bound")
        atomic_json(output / relative, page)
        pages.append({"path": relative, "digest": page["digest"], "kind": "metadata"})
    for rule in rules.values():
        records = []
        if rule["operation"] == "inactive":
            records.append({"kind": "inactive", "name": rule["name"], "reason": rule["inactive_reason"]})
        if "activation_scale_group" in rule:
            records.append({"kind": "scale_reduction", "name": rule["name"], "group": rule["activation_scale_group"],
                "source_tensor": rule["activation_scale_tensor"], "result_f32_bits": rule["activation_global_scale_bits"]})
        for record in records:
            length = len(canonical(record)) + 8
            if entries and size + length > (16 << 20) - 128:
                flush()
                entries, size = [], 128
            entries.append(record)
            size += length
    if entries:
        flush()
    provenance = seal({"schema_revision": 1, "recipe_digest": recipe["digest"], "converter_digest": converter_digest,
        "source_files": recipe["sources"], "selection_digest": selection_digest, "metadata": pages})
    atomic_json(output / "provenance.json", provenance)
    return provenance


def _validate_recipe(root, recipe):
    from .schemas import StrictValidator
    error = next(StrictValidator(RECIPE_SCHEMA).iter_errors(recipe), None)
    if error:
        raise ArtifactError(f"recipe: {error.message}")
    verify_identity(recipe)
    if not 1 <= recipe["chunk_size"] <= 4 << 20:
        raise ArtifactError("chunk size exceeds bulk data limit")
    operator = read_json(within(root, recipe["operator_contract"]))
    validate("operator-contract", operator)
    verify_identity(operator)
    if operator["profile"] != recipe["profile"]:
        raise ArtifactError("operator and preparation profile mismatch")
    seen = {}
    source_bytes = 0
    for source in recipe["sources"]:
        path = within(root, source["path"])
        shard = inspect(path, expected_sha256=source["sha256"])
        source_bytes += shard.size
        for name, tensor in shard.tensors.items():
            if name in seen:
                raise ArtifactError("duplicate tensor across source shards")
            seen[name] = (shard, tensor)
    rules = {rule["name"]: rule for rule in recipe["tensors"]}
    if len(rules) != len(recipe["tensors"]) or set(rules) != set(seen):
        raise ArtifactError("every required source tensor needs exactly one explained operation")
    ids = [stable_id(name) for name in seen]
    if len(set(ids)) != len(ids):
        raise ArtifactError("stable tensor ID collision")
    metadata_bytes = 0
    for item in recipe["metadata"]:
        length = within(root, item["path"]).stat().st_size
        if length > 16 << 20:
            raise ArtifactError("compact metadata file exceeds bounded byte cap")
        metadata_bytes += length
    if metadata_bytes > 512 << 20:
        raise ArtifactError("compact metadata population exceeds bounded byte cap")
    # A supplied official source index constrains the whole active/deferred
    # population. Comparing only successfully opened shards could otherwise
    # label an incomplete target checkpoint as prepared.
    for item in recipe["metadata"]:
        if item["path"].endswith("model.safetensors.index.json"):
            path = within(root, item["path"])
            index = loads(read_verified_bytes(path, expected_sha256=item["sha256"]),
                          project=False, max_nodes=2000000)
            auxiliary = {"engram.token_map", "engram.primes", "engram.offsets", "engram.multipliers", "engram.pad_id"}
            if not isinstance(index.get("weight_map"), dict) or set(index["weight_map"]) != set(seen) - auxiliary:
                raise ArtifactError("source shard inventory is incomplete for its official index")
    activation_groups = {}
    activation_members = {}
    for name, rule in rules.items():
        rule = dict(rule)
        rules[name] = rule
        for field, output_field in (("weight_global_scale_tensor", "weight_global_scale_bits"), ("activation_scale_tensor", "activation_global_scale_bits")):
            if field in rule:
                scale_name = rule[field]
                if scale_name not in seen or seen[scale_name][1].dtype != "F32" or seen[scale_name][1].length != 4:
                    raise ArtifactError("published global factor must reference one exact F32 scalar tensor")
                value = struct.unpack("<f", _raw_slice(*seen[scale_name], 0, 4))[0]
                if not math.isfinite(value) or value <= 0:
                    raise ArtifactError("published global factor must be finite and positive")
                bits = f"{struct.unpack('<I', struct.pack('<f', value))[0]:08x}"
                if output_field in rule and rule[output_field] != bits:
                    raise ArtifactError("declared global factor disagrees with published scalar")
                rule[output_field] = bits
        if "activation_scale_group" in rule:
            if "activation_scale_tensor" not in rule:
                raise ArtifactError("full-population scale reduction requires a published source scalar")
            group = rule["activation_scale_group"]
            activation_groups[group] = max(activation_groups.get(group, 0), bits_float(rule["activation_global_scale_bits"]))
            activation_members.setdefault(group, set()).add(rule["activation_scale_tensor"])
    for rule in rules.values():
        operation = target_operation(rule["name"])
        if operation != rule["operation"] and not (operation == "scale" and rule["operation"] == "inactive" and rule.get("inactive_reason")):
            raise ArtifactError("tensor-to-operation mapping disagrees with fixed source graph")
        if "activation_scale_group" in rule:
            value = activation_groups[rule["activation_scale_group"]]
            rule["activation_global_scale_bits"] = f"{struct.unpack('<I', struct.pack('<f', value))[0]:08x}"
    for rule in rules.values():
        if any(scale not in rules or rules[scale]["operation"] not in ("scale", "inactive") for scale in rule["scale_names"]):
            raise ArtifactError("unrecognized or missing scale tensor")
        if rule["operation"] == "inactive" and not rule.get("inactive_reason"):
            raise ArtifactError("inactive tensors require a reason")
        if rule["placement"] == "cache" and not rule.get("grant_group"):
            raise ArtifactError("cache populations require an explicit shared authorization group")
        checked_product(rule["logical_shape"])
        if rule["format"] == "nvfp4" and ("weight_global_scale_bits" not in rule or "activation_global_scale_bits" not in rule):
            raise ArtifactError("NVFP4 requires exact global weight and calibrated activation multipliers")
        if rule["format"] == "nvfp4" and not all(key in rule for key in ("weight_global_scale_tensor", "activation_scale_tensor", "activation_scale_group")):
            raise ArtifactError("NVFP4 global factors require published scalar references and full-population reduction proof")
        if rule["format"] == "nvfp4" and any(not math.isfinite(bits_float(rule[key])) or bits_float(rule[key]) <= 0 for key in ("weight_global_scale_bits", "activation_global_scale_bits")):
            raise ArtifactError("NVFP4 factors must be finite positive dequant multipliers")
        if rule["operation"] in ("expert_gate", "expert_up", "expert_down"):
            match = re.fullmatch(r"layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.weight", rule["name"])
            expected = {"w1": "expert_gate", "w2": "expert_down", "w3": "expert_up"}
            if not match or int(match[1]) >= 40 or int(match[2]) >= 384 or rule["operation"] != expected[match[3]]:
                raise ArtifactError("unrecognized target routed expert tensor")
            if recipe["profile"] == "nvfp4" and rule["format"] != "nvfp4":
                raise ArtifactError("strict NVFP4 routed experts must import publisher NVFP4 layout")
        if rule["convert"] and recipe["profile"] == "nvfp4":
            raise ArtifactError("strict NVFP4 cannot invent calibration")
    for group, members in activation_members.items():
        match = re.fullmatch(r"layers\.([0-9]+)\.(w13|w2)", group)
        if not match or int(match[1]) >= 40:
            raise ArtifactError("activation reduction group is outside the pinned graph")
        projections = ("w1", "w3") if match[2] == "w13" else ("w2",)
        expected = {f"layers.{match[1]}.ffn.experts.{expert}.{projection}.input_scale" for expert in range(384) for projection in projections}
        if members != expected:
            raise ArtifactError("activation scale reduction must cover the full original expert population before selection")
    return seen, rules, operator, source_bytes


def _partition_tasks(tasks, chunk_size, max_shard_bytes):
    tasks.sort(key=lambda task: task["grant_group"])
    import itertools
    units = [list(unit) for _, unit in itertools.groupby(tasks, key=lambda task: task["atomic_group"])]
    bundles, bundle, data_bytes, header_upper, previous_grant = [], [], 0, 64, None
    for unit in units:
        proposed_data, proposed_header, proposed_grant = data_bytes, header_upper, previous_grant
        for task in unit:
            if proposed_grant is not None and proposed_grant != task["grant_group"]:
                proposed_data += (-proposed_data) % chunk_size
                proposed_header += 256
            proposed_data += task["length"]
            proposed_header += len(task["name"].encode()) * 4 + 256
            proposed_grant = task["grant_group"]
        header_bytes = ((proposed_header + 8 + chunk_size - 1) // chunk_size) * chunk_size
        if bundle and (proposed_data + header_bytes > max_shard_bytes or header_bytes > 16 << 20):
            bundles.append(bundle)
            bundle, data_bytes, header_upper, previous_grant = [], 0, 64, None
            proposed_data = sum(task["length"] for task in unit)
            proposed_header = 64 + sum(len(task["name"].encode()) * 4 + 256 for task in unit)
            proposed_grant = unit[-1]["grant_group"]
            header_bytes = ((proposed_header + 8 + chunk_size - 1) // chunk_size) * chunk_size
        if proposed_data + header_bytes > max_shard_bytes or header_bytes > 16 << 20:
            raise ArtifactError("atomic tensor/scale group exceeds finite shard/header cap")
        bundle.extend(unit)
        data_bytes, header_upper, previous_grant = proposed_data, proposed_header, proposed_grant
    if bundle:
        bundles.append(bundle)
    if len(bundles) > 4096:
        raise ArtifactError("prepared bundle count exceeds coarse-mapping cap; increase finite shard cap")
    return bundles


def estimate(root, recipe):
    tensors, rules, _, source_bytes = _validate_recipe(root, recipe)
    tasks = []
    for name, (_, tensor) in sorted(tensors.items()):
        rule = rules[name]
        if rule["operation"] == "inactive":
            continue
        shape, dtype = list(tensor.shape), tensor.dtype
        if rule.get("repack") == "engram_packed":
            shape, dtype = [rule["logical_shape"][0], 264], "U8"
        elif rule["convert"]:
            shape = rule["logical_shape"]
            dtype = "BF16" if recipe["profile"] == "bf16" else "F8_E4M3"
        task = {"name": name, "shape": shape, "dtype": dtype, "length": checked_product(shape) * DTYPE_BYTES[dtype],
            "grant_group": rule.get("grant_group", rule["placement"]), "atomic_group": name}
        tasks.append(task)
        if rule["convert"] and recipe["profile"] == "fp8":
            scale_shape = [(value + 31) // 32 for value in shape]
            tasks.append({**task, "name": name + ".ria_scale", "shape": scale_shape,
                "dtype": "F8_E8M0", "length": checked_product(scale_shape)})
    bundles = _partition_tasks(tasks, recipe["chunk_size"], recipe["max_shard_bytes"])
    sizes = [_bundle_header(Path("18446744073709551615.safetensors"), bundle,
             recipe["chunk_size"], recipe["max_shard_bytes"])[2] for bundle in bundles]
    output, largest = sum(sizes) + (512 << 20), max(sizes, default=0)
    pending, visited, retained = [recipe, tensors, rules], set(), 0
    while pending:
        item = pending.pop()
        if id(item) in visited:
            continue
        visited.add(id(item))
        retained += sys.getsizeof(item)
        if isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, (tuple, list)):
            pending.extend(item)
        elif hasattr(item, "__dict__"):
            pending.append(vars(item))
    # Include the actual retained complete inventory, JSON encoding/decoding
    # temporaries, one bounded source header/index and tensor-page publication.
    metadata_working = retained * 3 + len(canonical(recipe)) * 2 + (128 << 20)
    return {"source_bytes": source_bytes, "output_upper_bytes": output,
            "in_progress_upper_bytes": largest, "calibration_scratch_bytes": recipe["scratch_bytes"],
            "disk_peak_upper_bytes": source_bytes + output + largest + recipe["scratch_bytes"],
            "retained_metadata_bytes": retained,
            "working_memory_upper_bytes": recipe["scratch_bytes"] + metadata_working,
            "classification": "conservative preparation estimate, not runtime admission"}


def _bundle_header(path, tasks, chunk_size, max_shard_bytes):
    header, cursor, ordered = {}, 0, []
    for index, task in enumerate(tasks):
        if index:
            previous = tasks[index - 1]
            changed_grant = task["grant_group"] != previous["grant_group"]
            # Per-expert pages are private placement units. This 4 KiB boundary
            # does not expand the independently reviewed bootstrap grant.
            current_group = re.match(r"layers\.[0-9]+\.ffn\.experts\.[0-9]+\.", task["name"])
            previous_group = re.match(r"layers\.[0-9]+\.ffn\.experts\.[0-9]+\.", previous["name"])
            changed_expert = (current_group.group() if current_group else None) != (previous_group.group() if previous_group else None)
            alignment = chunk_size if changed_grant else (4096 if changed_expert else 1)
            padding = (-cursor) % alignment
            if padding:
                name = f"__ria_padding_{path.stem}_{index}"
                task_padding = {"name": name, "dtype": "U8", "shape": [padding], "length": padding,
                                "blocks": lambda padding=padding: iter([b"\0" * padding]), "padding": True}
                header[name] = {"dtype": "U8", "shape": [padding], "data_offsets": [cursor, cursor + padding]}
                ordered.append((task_padding, cursor))
                cursor += padding
        header[task["name"]] = {"dtype": task["dtype"], "shape": task["shape"], "data_offsets": [cursor, cursor + task["length"]]}
        ordered.append((task, cursor))
        cursor += task["length"]
    encoded = canonical(header)
    # The entire header is public metadata and independently authenticated. Keep
    # data groups aligned to absolute file verification chunks, not guessed
    # tensor-relative offsets. Padding is explicitly indexed safetensors data.
    header_length = ((len(encoded) + 8 + chunk_size - 1) // chunk_size) * chunk_size - 8
    encoded += b" " * (header_length - len(encoded))
    if len(encoded) > 16 << 20 or cursor + len(encoded) + 8 > max_shard_bytes:
        raise ArtifactError("bundle exceeds bounded header/shard allowance")
    if (cursor + len(encoded) + 8 + chunk_size - 1) // chunk_size > 1000000:
        raise ArtifactError("bundle chunk index exceeds bounded inventory capacity")
    return encoded, ordered, cursor + len(encoded) + 8


def _write_bundle(path, tasks, chunk_size, max_shard_bytes):
    """One immutable coarse shard; authorization groups start at chunk boundaries."""
    encoded, ordered, _ = _bundle_header(path, tasks, chunk_size, max_shard_bytes)
    fd, temporary = tempfile.mkstemp(prefix=".ria-data-", dir=path.parent)
    records = []
    try:
        with os.fdopen(fd, "wb") as stream:
            os.fchmod(stream.fileno(), 0o644)
            stream.write(struct.pack("<Q", len(encoded)))
            stream.write(encoded)
            for task, offset in ordered:
                result, written = hashlib.sha256(), 0
                for block in task["blocks"]():
                    written += len(block)
                    if written > task["length"]:
                        raise ArtifactError("prepared tensor overrun")
                    result.update(block)
                    stream.write(block)
                if written != task["length"]:
                    raise ArtifactError("prepared tensor truncated")
                records.append((task, offset, result.hexdigest()))
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            if hash_file(path) != hash_file(temporary):
                raise ArtifactError("existing immutable bundle differs")
            os.unlink(temporary)
        else:
            os.rename(temporary, path)
        sync_directory(path.parent)
        return records, len(encoded) + 8
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prepare(root, recipe, output, *, role="server", selected_names=None):
    root, output = Path(root), Path(output)
    tensors, rules, operator, source_bytes = _validate_recipe(root, recipe)
    if role not in ("client", "server"):
        raise ArtifactError("unknown prepared package role")
    selected = (set(tensors) if role == "server" else set()) if selected_names is None else set(selected_names)
    if not selected <= set(tensors):
        raise ArtifactError("unknown client cache tensor")
    if role == "client":
        selected = {name for name in selected if rules[name]["placement"] in ("client", "both", "cache")}
        selected.update(name for name, rule in rules.items() if rule["placement"] in ("client", "both"))
        pending = list(selected)
        while pending:
            for scale in rules[pending.pop()]["scale_names"]:
                if scale not in selected:
                    selected.add(scale)
                    pending.append(scale)
    selected = {name for name in selected if rules[name]["operation"] != "inactive"}
    if not selected:
        raise ArtifactError("empty role package")
    output.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        raise ArtifactError("output root must not be a symlink")
    selection_digest = digest({"names": sorted(selected)})
    state_path = output / ".prepare-state.json"
    converter_digest = digest({path.name: hash_file(path) for path in Path(__file__).parent.glob("*.py")})
    if state_path.exists():
        permissions = state_path.stat()
        if permissions.st_uid != os.geteuid() or stat.S_IMODE(permissions.st_mode) & 0o077:
            raise ArtifactError("resume checkpoint must be private to its preparing user")
    state = read_json(state_path) if state_path.exists() else {"recipe_digest": recipe["digest"], "role": role,
        "selection_digest": selection_digest, "converter_digest": converter_digest, "completed": {}}
    if (state.get("recipe_digest"), state.get("role"), state.get("selection_digest"), state.get("converter_digest")) != (recipe["digest"], role, selection_digest, converter_digest):
        raise ArtifactError("resume recipe or selection mismatch")
    if (output / "manifest.json").exists():
        if not state["completed"]:
            raise ArtifactError("output already contains an unrelated committed manifest")
        existing = verify_package(output)
        _validate_sources(tensors)
        return existing
    data_dir = output / "tensors"
    data_dir.mkdir(exist_ok=True)
    tasks = []
    for name in sorted(selected):
        rule, source = rules[name], tensors[name]
        tensor = source[1]
        source_hasher = hashlib.sha256()
        for block in tensor_blocks(*source):
            source_hasher.update(block)
        shape, dtype, format_name = list(tensor.shape), tensor.dtype, rule["format"]
        grant = rule.get("grant_group", rule["placement"])
        generated = None
        repack = rule.get("repack") == "engram_packed"
        source_scale_hash = None
        if repack:
            if rule["convert"] or len(rule["logical_shape"]) != 2 or rule["logical_shape"][1] != 256 or len(rule["scale_names"]) != 1:
                raise ArtifactError("Engram repack requires one exact source row-scale tensor")
            shape, dtype, format_name = [rule["logical_shape"][0], 264], "U8", "engram_packed"
            scale_source = tensors[rule["scale_names"][0]]
            scale_hasher = hashlib.sha256()
            for block in tensor_blocks(*scale_source):
                scale_hasher.update(block)
            source_scale_hash = scale_hasher.hexdigest()
            def blocks(source=source, scale_source=scale_source, rows=shape[0]):
                return _engram_blocks(source, scale_source, rows, recipe["scratch_bytes"])
        elif rule["convert"]:
            shape = rule["logical_shape"]
            dtype, format_name = ("BF16", "bf16") if recipe["profile"] == "bf16" else ("F8_E4M3", "fp8_block32")
            generated = (name + ".ria_scale", bytearray()) if recipe["profile"] == "fp8" else None
            def blocks(rule=rule, source=source, generated=generated):
                if generated is not None:
                    generated[1].clear()
                for values, scale_values in _conversion_blocks(rule, source, tensors, recipe["profile"], recipe["scratch_bytes"]):
                    if generated is not None:
                        generated[1].extend(scale_values)
                        if len(generated[1]) > recipe["scratch_bytes"]:
                            raise ArtifactError("generated scale buffer exceeds scratch cap")
                    yield values
        else:
            def blocks(source=source):
                return tensor_blocks(*source)
        length = checked_product(shape) * DTYPE_BYTES[dtype]
        descriptor = {"id": stable_id(name), "name": name, "logical_shape": rule["logical_shape"], "physical_shape": shape,
            "dtype": dtype, "format": format_name, "layout": "row_major_le", "shard": "0", "offset": "0", "length": str(length),
            "sha256": "0" * 64, "source_sha256": source_hasher.hexdigest(),
            "scale_ids": [] if rule["convert"] or repack else [stable_id(scale) for scale in rule["scale_names"]],
            "operation": rule["operation"], "placement": rule["placement"], "alias_of": None, "byte_order": "little",
            "bytes_per_block": 1 if format_name in ("nvfp4", "source_mxfp4", "fp8_block32", "engram_packed") else DTYPE_BYTES[dtype],
            "block_values": 2 if format_name in ("nvfp4", "source_mxfp4") else 1,
            "group_shape": [1, 16] if format_name == "nvfp4" else ([1, 32] if format_name == "source_mxfp4" else ([32, 32] if format_name == "fp8_block32" else []))}
        if source_scale_hash:
            descriptor["source_scale_sha256"] = source_scale_hash
        if len(shape) == 2:
            descriptor["value_row_stride"] = str(length // shape[0]) if shape[0] else "0"
            if descriptor["scale_ids"] and format_name in ("nvfp4", "source_mxfp4", "fp8_block32"):
                descriptor["scale_row_stride"] = str((rule["logical_shape"][1] + (15 if format_name == "nvfp4" else 31)) // (16 if format_name == "nvfp4" else 32))
        if not rule["convert"]:
            descriptor.update({key: rule[key] for key in ("weight_global_scale_bits", "activation_global_scale_bits") if key in rule})
        tasks.append({"name": name, "dtype": dtype, "shape": shape, "length": length, "blocks": blocks,
                      "descriptor": descriptor, "grant_group": grant, "padding": False, "atomic_group": name})
        if generated is not None:
            scale_name, scale_data = generated
            scale_shape = [(value + 31) // 32 for value in shape]
            scale_id = stable_id(scale_name)
            descriptor["scale_ids"] = [scale_id]
            descriptor["scale_row_stride"] = str(scale_shape[1])
            descriptor["weight_global_scale_bits"] = "3f800000"
            descriptor["activation_global_scale_bits"] = "3f800000"
            scale_descriptor = {**descriptor, "id": scale_id, "name": scale_name, "logical_shape": scale_shape,
                "physical_shape": scale_shape, "dtype": "F8_E8M0", "format": "plain", "length": str(checked_product(scale_shape)),
                "scale_ids": [], "operation": "scale", "value_row_stride": str(scale_shape[1]), "bytes_per_block": 1, "block_values": 1, "group_shape": []}
            def scale_blocks(scale_data=scale_data):
                yield scale_data
                scale_data.clear()
            tasks.append({"name": scale_name, "dtype": "F8_E8M0", "shape": scale_shape, "length": checked_product(scale_shape),
                          "blocks": scale_blocks, "descriptor": scale_descriptor,
                          "grant_group": grant, "padding": False, "atomic_group": name})
    bundles = _partition_tasks(tasks, recipe["chunk_size"], recipe["max_shard_bytes"])
    sizes = [_bundle_header(Path(f"{index}.safetensors"), bundle, recipe["chunk_size"], recipe["max_shard_bytes"])[2]
             for index, bundle in enumerate(bundles)]
    # Charge actual planned payload/header/alignment bytes, one in-progress
    # atomic shard and the bounded complete metadata graph. Per-tensor chunk
    # padding is neither emitted nor a reason to reject sufficient storage.
    output_upper, largest = sum(sizes), max(sizes)
    if shutil.disk_usage(output).free < output_upper + largest + (512 << 20):
        raise ArtifactError("insufficient disk for final output plus in-progress bundle")
    descriptors, shards = [], []
    for index, bundle in enumerate(bundles):
        shard_id = stable_id(f"{recipe['digest']}:{role}:bundle:{index}")
        relative = f"tensors/{shard_id}.safetensors"
        target = within(output, relative)
        # A completed shard is usable only after independently rechecking its
        # identity; generation remains deterministic so interrupted work resumes.
        checkpoint = state["completed"].get(relative)
        resumed = bool(checkpoint and target.exists())
        if resumed and hash_file(target) != checkpoint["sha256"]:
            raise ArtifactError("completed resume shard integrity mismatch")
        if resumed:
            checkpoint_document = read_json(within(output, checkpoint["checkpoint_path"]))
            verify_identity(checkpoint_document, checkpoint["checkpoint_digest"])
            bundle_descriptors = checkpoint_document["tensors"]
            data_start = u64(checkpoint_document["shard"]["data_start"])
            records = []
        else:
            records, data_start = _write_bundle(target, bundle, recipe["chunk_size"], recipe["max_shard_bytes"])
            bundle_descriptors = []
        for task, offset, payload_hash in records:
            if task["padding"]:
                padding_name = f"{relative}:{task['name']}"
                descriptor = {"id": stable_id(padding_name), "name": task["name"], "logical_shape": task["shape"], "physical_shape": task["shape"],
                    "dtype": "U8", "format": "plain", "layout": "row_major_le", "shard": shard_id, "offset": str(offset), "length": str(task["length"]),
                    "sha256": payload_hash, "source_sha256": payload_hash, "scale_ids": [], "operation": "inactive", "placement": "inactive",
                    "alias_of": None, "byte_order": "little", "bytes_per_block": 1, "block_values": 1, "group_shape": []}
            else:
                descriptor = {**task["descriptor"], "shard": shard_id, "offset": str(offset), "sha256": payload_hash}
            bundle_descriptors.append(descriptor)
        descriptors.extend(bundle_descriptors)
        shard = {"id": shard_id, "path": relative, "data_start": str(data_start), **chunk_index(target, recipe["chunk_size"])}
        shards.append(shard)
        checkpoint_document = seal({"schema_revision": 1, "recipe_digest": recipe["digest"], "converter_digest": converter_digest,
            "tensors": bundle_descriptors, "shard": shard})
        checkpoint_path = f".prepare-resume/{shard_id}.json"
        from .identity import atomic_bytes
        atomic_bytes(output / checkpoint_path, canonical(checkpoint_document) + b"\n", mode=0o600)
        state["completed"][relative] = {"sha256": shard["sha256"], "checkpoint_path": checkpoint_path,
            "checkpoint_digest": checkpoint_document["digest"]}
        atomic_bytes(state_path, canonical(state) + b"\n", mode=0o600)
    pages = publish_tensor_pages(output, descriptors, shards)
    layout = seal({"schema_revision": 1, "logical_model_digest": recipe["logical_model_digest"],
        "operator_contract_digest": operator["digest"], "backend": "source", "tensors": [], "tensor_pages": pages})
    validate("physical-layout", layout)
    atomic_json(output / "layout.json", layout)
    atomic_json(output / "operator.json", operator)
    metadata = [{"path": "layout.json", "digest": layout["digest"], "kind": "layout"},
                {"path": "operator.json", "digest": operator["digest"], "kind": "operator"}, *pages]
    provenance = publish_provenance(output, recipe, converter_digest, selection_digest, rules)
    metadata.append({"path": "provenance.json", "digest": provenance["digest"], "kind": "metadata"})
    from .identity import atomic_bytes
    for index, item in enumerate(recipe["metadata"]):
        source_path = within(root, item["path"])
        data = read_verified_bytes(source_path, expected_sha256=item["sha256"])
        relative = f"metadata/{index}.bin"
        atomic_bytes(output / relative, data)
        wrapper = seal({"schema_revision": 1, "path": relative, "source_path": item["path"], "sha256": item["sha256"], "length": str(len(data))})
        wrapper_relative = f"metadata/{index}.json"
        atomic_json(output / wrapper_relative, wrapper)
        metadata.append({"path": wrapper_relative, "digest": wrapper["digest"], "kind": "metadata"})
    manifest = seal({"schema_revision": 1, "role": role, "model_id": recipe["model_id"], "source_revision": recipe["source_revision"],
        "profile": recipe["profile"], "logical_model_digest": recipe["logical_model_digest"], "operator_contract_digest": operator["digest"],
        "tokenizer_digest": recipe["tokenizer_digest"], "encoding_digest": recipe["encoding_digest"], "layout_digest": layout["digest"],
        "tensors": [], "shards": [], "tensor_pages": pages, "metadata": metadata,
        "feature_exclusions": recipe["feature_exclusions"] + ([f"remote-owned inventory bound by logical model {recipe['logical_model_digest']}"] if role == "client" else []),
        "unexplained_required_tensors": []})
    validate("manifest", manifest)
    if len(canonical(manifest)) > 256 << 10:
        raise ArtifactError("compact manifest root exceeds 256 KiB")
    verify_package(output, manifest)
    _validate_sources(tensors)
    atomic_json(output / "preparation-estimate.json", {"source_bytes": source_bytes,
        "output_bytes": sum(u64(shard["length"]) for shard in shards), "scratch_bytes": recipe["scratch_bytes"]})
    atomic_json(output / "manifest.json", manifest)
    return manifest


def verify_metadata_graph(root, references, *, max_nodes=4096, max_bytes=512 << 20, max_depth=32):
    seen, active, binary_seen = set(), set(), set()
    total = 0
    def visit(reference, depth):
        nonlocal total
        path = reference["path"]
        if path in active:
            raise ArtifactError("metadata identity graph cycle")
        key = (path, reference["digest"])
        if key in seen:
            return
        if len(seen) >= max_nodes or depth > max_depth:
            raise ArtifactError("metadata graph bound exceeded")
        target = within(root, path)
        total += target.stat().st_size
        if total > max_bytes:
            raise ArtifactError("metadata graph byte bound exceeded")
        document = read_json(target)
        verify_identity(document, reference["digest"])
        active.add(path)
        if set(document) == {"schema_revision", "path", "source_path", "sha256", "length", "digest"}:
            binary = within(root, document["path"])
            if document["path"] not in binary_seen:
                total += binary.stat().st_size
                binary_seen.add(document["path"])
            if total > max_bytes:
                raise ArtifactError("metadata graph byte bound exceeded")
            if binary.stat().st_size != u64(document["length"]) or hash_file(binary) != document["sha256"]:
                raise ArtifactError("opaque metadata file integrity mismatch")
        for child in document.get("metadata", []) + document.get("tensor_pages", []):
            visit(child, depth + 1)
        active.remove(path)
        seen.add(key)
    for reference in references:
        visit(reference, 0)


def verify_package(root, manifest=None):
    root = Path(root)
    manifest = read_json(root / "manifest.json") if manifest is None else manifest
    validate("manifest", manifest)
    verify_identity(manifest)
    verify_metadata_graph(root, manifest["metadata"])
    for kind, expected in (("layout", manifest["layout_digest"]), ("operator", manifest["operator_contract_digest"])):
        references = [item for item in manifest["metadata"] if item["kind"] == kind]
        if len(references) != 1 or references[0]["digest"] != expected:
            raise ArtifactError("root does not bind exactly one matching layout/operator contract")
        document = read_json(within(root, references[0]["path"]))
        validate("physical-layout" if kind == "layout" else "operator-contract", document)
        if kind == "layout" and (document["logical_model_digest"], document["operator_contract_digest"], document.get("tensor_pages", []), document["tensors"]) != (manifest["logical_model_digest"], manifest["operator_contract_digest"], manifest["tensor_pages"], manifest["tensors"]):
            raise ArtifactError("physical layout and root inventory disagree")
        if kind == "operator" and document["profile"] != manifest["profile"]:
            raise ArtifactError("operator contract and root semantics disagree")
    tensor_descriptors = list(manifest["tensors"])
    shard_descriptors = list(manifest["shards"])
    for reference in manifest["tensor_pages"]:
        page = read_json(within(root, reference["path"]))
        validate("tensor-page", page)
        verify_identity(page, reference["digest"])
        tensor_descriptors.extend(page["tensors"])
        shard_descriptors.extend(page["shards"])
    validate("physical-layout", {"schema_revision": 1, "logical_model_digest": manifest["logical_model_digest"],
        "operator_contract_digest": manifest["operator_contract_digest"], "backend": "source", "tensors": tensor_descriptors,
        "tensor_pages": [], "digest": manifest["layout_digest"]})
    shards = {}
    for item in shard_descriptors:
        path = within(root, item["path"])
        if item["id"] in shards:
            raise ArtifactError("duplicate shard path")
        shard = inspect(path)
        actual = chunk_index(path, item["chunk_size"])
        if actual != {key: item[key] for key in actual} or shard.data_start != u64(item["data_start"]):
            raise ArtifactError("prepared shard integrity mismatch")
        shards[item["id"]] = shard
    for item in tensor_descriptors:
        expected_format = {"nvfp4": "nvfp4", "fp8": "fp8_block32", "bf16": "bf16"}[manifest["profile"]]
        if item["operation"] in ("expert_gate", "expert_up", "expert_down") and item["format"] != expected_format:
            raise ArtifactError("routed projection encoding disagrees with numerical profile")
        shard = shards.get(item["shard"])
        if shard is None or item["name"] not in shard.tensors:
            raise ArtifactError("tensor references absent shard or tensor")
        tensor = shard.tensors[item["name"]]
        if (tensor.offset, tensor.length, tensor.dtype, list(tensor.shape)) != (u64(item["offset"]), u64(item["length"]), item["dtype"], item["physical_shape"]):
            raise ArtifactError("tensor physical descriptor mismatch")
        actual = hashlib.sha256()
        for block in tensor_blocks(shard, tensor):
            actual.update(block)
        if actual.hexdigest() != item["sha256"]:
            raise ArtifactError("prepared tensor hash mismatch")
    indexed_names = {item["id"]: set() for item in shard_descriptors}
    for item in tensor_descriptors:
        indexed_names[item["shard"]].add(item["name"])
    if any(indexed_names[key] != set(shard.tensors) for key, shard in shards.items()):
        raise ArtifactError("prepared shard contains unexplained physical tensors")
    return manifest
