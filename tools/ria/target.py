"""Generate a reviewed target recipe from local official NVIDIA checkpoint files."""

import contextlib
import hashlib
import math
import struct
from pathlib import Path

from .identity import (ArtifactError, atomic_json, canonical, digest, hash_file, loads, read_json,
                       read_verified_bytes, seal, within)
from .preparation import _validate_sources, target_operation
from .safetensors import exact_read, inspect, validate_source_snapshot
from .identity import open_regular

SOURCE_REVISION = "2cba9e42aa026125f3ed06c6d98c1db82f7ca027"
NVIDIA_REVISION = "3431dde3247c13b5957f682b1e3c6fcae2566079"
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
NVIDIA = "nvidia/DeepSeek-V4.1-Flash-NVFP4"


def _locked_metadata(repository, revision, name):
    directory = Path(__file__).resolve().parents[2] / "locks" / "metadata" / repository / revision
    path = directory / name
    if not path.is_file():
        raise ArtifactError(f"reviewed compact source metadata is unavailable: {name}")
    return path


def _f32_scalar(streams, tensor_sources, name):
    if name not in tensor_sources:
        raise ArtifactError("published calibration/global scale tensor is absent")
    shard, tensor = tensor_sources[name]
    if tensor.dtype != "F32" or tensor.length != 4:
        raise ArtifactError("global calibration requires the published F32 scalar layout")
    stream = streams[Path(shard.path)]
    validate_source_snapshot(shard, stream)
    stream.seek(shard.data_start + tensor.offset)
    raw = exact_read(stream, 4)
    validate_source_snapshot(shard, stream)
    value = struct.unpack("<f", raw)[0]
    if not math.isfinite(value) or value <= 0:
        raise ArtifactError("invalid published calibration/global scale scalar")
    return value, f"{struct.unpack('<I', raw)[0]:08x}", hashlib.sha256(raw).hexdigest()


def create_recipe(source_dir, profile, *, chunk_size=4 << 20, max_shard_bytes=128 << 30,
                  scratch_bytes=64 << 20):
    """No weight acquisition. Read and verify only operator-provisioned local files.

    The initial accepted input is the immutable official NVIDIA artifact. Derived
    FP8/BF16 keep its provenance; widening never claims recovered master weights.
    """
    if profile not in ("nvfp4", "fp8", "bf16"):
        raise ArtifactError("unknown numerical profile")
    root = Path(source_dir).resolve(strict=True)
    config_path, index_path = within(root, "config.json"), within(root, "model.safetensors.index.json")
    config_sha256 = hash_file(_locked_metadata(NVIDIA, NVIDIA_REVISION, "config.json"))
    index_sha256 = hash_file(_locked_metadata(NVIDIA, NVIDIA_REVISION, "model.safetensors.index.json"))
    config = loads(read_verified_bytes(config_path, expected_sha256=config_sha256))
    text = config["text_config"]
    expected = {"num_hidden_layers": 40, "hidden_size": 5120, "moe_intermediate_size": 2304,
                "n_routed_experts": 384, "num_experts_per_tok": 6, "engram_layer_ids": [1, 14], "hc_mult": 4,
                "kv_source_layer_ids": [2, 8, 14, 20], "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36]}
    if any(text.get(key) != value for key, value in expected.items()):
        raise ArtifactError("target graph configuration discrepancy")
    quantized = config["quantization_config"].get("quantized_layers", {})
    if set(quantized) != {f"layers.{layer}.ffn.experts" for layer in range(40)} or any(value != {"group_size": 16, "quant_algo": "NVFP4"} for value in quantized.values()):
        raise ArtifactError("per-module mixed precision descriptors are not the required NVIDIA NVFP4 inventory")
    index = loads(read_verified_bytes(index_path, expected_sha256=index_sha256), project=False, max_nodes=2000000)
    weight_map = index["weight_map"]
    if not isinstance(weight_map, dict):
        raise ArtifactError("source index lacks its exact tensor inventory")
    repository = read_json(_locked_metadata(NVIDIA, NVIDIA_REVISION, "repository-index.json"))
    expected_sources = {item["rfilename"]: item["lfs"] for item in repository["siblings"] if item["rfilename"].endswith(".safetensors")}
    sources, tensor_sources = [], {}
    for relative in sorted(set(weight_map.values())):
        if relative not in expected_sources:
            raise ArtifactError("source shard is not in the pinned publisher LFS inventory")
        path = within(root, relative)
        expected_source = expected_sources[relative]
        if path.stat().st_size != expected_source["size"]:
            raise ArtifactError("local checkpoint shard differs from the pinned publisher LFS content identity")
        shard = inspect(path, expected_sha256=expected_source["sha256"])
        sources.append({"path": relative, "sha256": expected_source["sha256"]})
        for name, tensor in shard.tensors.items():
            if name in tensor_sources or weight_map.get(name) != relative:
                raise ArtifactError("source index/header tensor ownership mismatch")
            tensor_sources[name] = (shard, tensor)
    if set(tensor_sources) != set(weight_map):
        raise ArtifactError("incomplete required checkpoint population")
    scalar_values, groups = {}, {}
    with contextlib.ExitStack() as stack:
        # One handle per coarse source shard, not one file-open per calibration scalar.
        unique_paths = {Path(shard.path) for shard, _ in tensor_sources.values()}
        streams = {path: stack.enter_context(open_regular(path)) for path in unique_paths}
        for layer in range(40):
            for expert in range(384):
                for projection in ("w1", "w3", "w2"):
                    prefix = f"layers.{layer}.ffn.experts.{expert}.{projection}"
                    for suffix in ("weight_scale_2", "input_scale"):
                        name = prefix + "." + suffix
                        scalar_values[name] = _f32_scalar(streams, tensor_sources, name)
                    group = f"layers.{layer}.{'w13' if projection != 'w2' else 'w2'}"
                    groups.setdefault(group, []).append(prefix + ".input_scale")
    _validate_sources(tensor_sources)
    calibration_groups = []
    for group, names in sorted(groups.items()):
        expected_count = 768 if group.endswith("w13") else 384
        if len(names) != expected_count:
            raise ArtifactError("calibration reduction population is incomplete")
        maximum = max(scalar_values[name][0] for name in names)
        calibration_groups.append({"group": group, "count": len(names), "reduction": "maximum",
            "maximum_f32_bits": f"{struct.unpack('<I', struct.pack('<f', maximum))[0]:08x}",
            "members_digest": digest({"members": [{"name": name, "f32_bits": scalar_values[name][1], "sha256": scalar_values[name][2]} for name in sorted(names)]})})
    calibration = seal({"schema_revision": 1, "source_revision": NVIDIA_REVISION,
        "integration_revision": "da64c5cbb8cf6bfd39be19da43573fdfd484c43a", "integration_branch": "flashinfer_cutlass_or_trtllm_scale_hierarchy",
        "coefficient_position": "source_before_down_quantizer", "scope": "full_original_384_experts_and_both_gate_up_per_layer",
        "source_index_sha256": index_sha256, "groups": calibration_groups,
        "qualification": "source-derived calibration identity; numerical/target qualification is separate"})
    directory = root / ".ria-recipes" / calibration["digest"] / profile
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "calibration.json", calibration)
    operator = seal({"schema_revision": 1, "profile": profile, "source_revision": SOURCE_REVISION,
        "graph": "deepseek_v41_flash", "weight_format": "nvfp4" if profile == "nvfp4" else ("fp8_block32" if profile == "fp8" else "bf16"),
        "activation_group": 16 if profile == "nvfp4" else (32 if profile == "fp8" else 0),
        "weight_scale_block": [1, 16] if profile == "nvfp4" else ([32, 32] if profile == "fp8" else []),
        "activation_quantizer": {"nvfp4": "nvfp4_dynamic16_calibrated", "fp8": "fp8_e4m3fn_ue8m0_32", "bf16": "bf16_rne"}[profile],
        "clamp_f32_bits": "41200000", "gate_clamp": "upper_only", "up_clamp": "two_sided",
        "coefficient_position": "before_down_quantizer", "accumulator": "fp32", "reduction_order": "increasing_expert_id_then_shared",
        "scale_reduction_domain": "full_original_population", "calibration_digest": calibration["digest"] if profile == "nvfp4" else None,
        "rounding": "ties_to_even", "nibble_order": "low_first"})
    atomic_json(directory / "operator.json", operator)
    tokenizer_path = within(root, "tokenizer.json")
    tokenizer_sha256 = hash_file(_locked_metadata(MODEL, SOURCE_REVISION, "tokenizer.json"))
    read_verified_bytes(tokenizer_path, expected_sha256=tokenizer_sha256)
    from .engram import prepare_metadata
    engram_metadata = prepare_metadata(config_path, tokenizer_path, directory,
                                       configuration_sha256=config_sha256, tokenizer_sha256=tokenizer_sha256)
    engram_path = directory / "engram-metadata.safetensors"
    engram_shard = inspect(engram_path, expected_sha256=engram_metadata["engram_metadata_sha256"])
    sources.append({"path": str(engram_path.relative_to(root)), "sha256": engram_metadata["engram_metadata_sha256"]})
    for name, tensor in engram_shard.tensors.items():
        tensor_sources[name] = (engram_shard, tensor)
    rules, used_runtime_scales = [], set()
    for name, (_, tensor) in sorted(tensor_sources.items()):
        operation = target_operation(name)
        routed = operation in ("expert_gate", "expert_up", "expert_down")
        format_name, logical_shape, scales, convert, repack = "plain", list(tensor.shape), [], False, "none"
        weight_like = operation not in ("scale", "inactive", "norm", "mhc", "router")
        if routed:
            if tensor.dtype != "U8" or len(tensor.shape) != 2:
                raise ArtifactError("required NVIDIA routed weights must be packed U8 matrices")
            logical_shape[-1] *= 2
            expected_shape = [5120, 2304] if operation == "expert_down" else [2304, 5120]
            if logical_shape != expected_shape:
                raise ArtifactError("target routed projection shape discrepancy")
            format_name, scales, convert = "nvfp4", [name.removesuffix(".weight") + ".weight_scale"], profile != "nvfp4"
        elif tensor.dtype == "F8_E4M3" and weight_like:
            format_name, scales = "fp8_block32", [name.removesuffix(".weight") + ".scale"]
            convert = profile == "bf16"
            if name.endswith(".engram.embed.weight"):
                if tensor.shape[1:] != (256,):
                    raise ArtifactError("Engram row width discrepancy")
                convert, repack = False, "engram_packed"
        elif tensor.dtype == "BF16":
            format_name = "bf16"
        placement = "inactive" if operation == "inactive" else ("cache" if routed or ".ffn.experts." in name else ("server" if ".engram.embed." in name else "both"))
        rule = {"name": name, "operation": operation, "placement": placement, "format": format_name,
                "logical_shape": logical_shape, "scale_names": scales, "convert": convert, "repack": repack}
        if ".ffn.experts." in name:
            # The reviewed model grant authorizes the entire routed bank. A
            # separate boundary per expert would add ~50 GiB of zero padding;
            # only server-owned tables need isolation from this grant domain.
            rule["grant_group"] = "routed-bank"
        elif ".engram.embed." in name:
            rule["grant_group"] = name.rsplit(".embed", 1)[0]
        if routed:
            prefix = name.removesuffix(".weight")
            rule["weight_global_scale_tensor"] = prefix + ".weight_scale_2"
            rule["activation_scale_tensor"] = prefix + ".input_scale"
            rule["activation_scale_group"] = prefix.split(".ffn.experts.")[0] + (".w2" if operation == "expert_down" else ".w13")
        if operation == "inactive":
            rule["inactive_reason"] = "DSpark/MTP is explicitly disabled in the target-only runtime package"
        if not convert and repack == "none":
            used_runtime_scales.update(scales)
        rules.append(rule)
    for rule in rules:
        if rule["operation"] == "scale" and rule["name"] not in used_runtime_scales:
            rule["operation"], rule["placement"] = "inactive", "inactive"
            rule["inactive_reason"] = "source scale consumed offline by conversion/repacking or embedded frozen global factor"
    metadata = [{"path": "config.json", "sha256": config_sha256}, {"path": "model.safetensors.index.json", "sha256": index_sha256},
                {"path": str((directory / "calibration.json").relative_to(root)), "sha256": hashlib.sha256(canonical(calibration) + b"\n").hexdigest()},
                {"path": str((directory / "engram-metadata.json").relative_to(root)), "sha256": hashlib.sha256(canonical(engram_metadata) + b"\n").hexdigest()}]
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        source = within(root, name)
        pinned = _locked_metadata(MODEL, SOURCE_REVISION, name)
        expected_sha256 = hash_file(pinned)
        read_verified_bytes(source, expected_sha256=expected_sha256)
        metadata.append({"path": name, "sha256": expected_sha256})
    encoding_source = _locked_metadata(MODEL, SOURCE_REVISION, "encoding/encoding.py")
    from .identity import atomic_bytes
    encoding_data = read_verified_bytes(encoding_source)
    encoding_sha256 = hashlib.sha256(encoding_data).hexdigest()
    atomic_bytes(directory / "encoding.py", encoding_data)
    metadata.append({"path": str((directory / "encoding.py").relative_to(root)), "sha256": encoding_sha256})
    logical_model_digest = digest({"model_id": MODEL, "source_revision": SOURCE_REVISION, "artifact_revision": NVIDIA_REVISION,
        "source_index_sha256": index_sha256, "configuration_sha256": config_sha256, "profile": profile,
        "operator_contract_digest": operator["digest"], "engram_metadata_digest": engram_metadata["digest"]})
    recipe = seal({"schema_revision": 1, "model_id": MODEL, "source_revision": NVIDIA_REVISION, "profile": profile,
        "logical_model_digest": logical_model_digest, "tokenizer_digest": tokenizer_sha256,
        "encoding_digest": encoding_sha256, "operator_contract": str((directory / "operator.json").relative_to(root)),
        "sources": sources, "tensors": rules, "metadata": metadata,
        "feature_exclusions": ["DSpark/MTP target-only omission; no backbone, Engram or vision pruning",
                               "widened FP8/BF16 do not recover unavailable original master weights"],
        "max_shard_bytes": max_shard_bytes, "chunk_size": chunk_size, "scratch_bytes": scratch_bytes})
    _validate_sources(tensor_sources)
    atomic_json(directory / "recipe.json", recipe)
    return recipe
