"""Offline immutable Engram lookup metadata using pinned normalization/PCG64."""

import importlib.metadata
import hashlib
import math
import struct
from pathlib import Path

from .identity import (ArtifactError, atomic_bytes, atomic_json, canonical, loads,
                       read_verified_bytes, seal)

NUMPY_VERSION = "2.5.3"
TOKENIZERS_VERSION = "0.23.2"


def _prime(value):
    if value < 2 or (value != 2 and value % 2 == 0):
        return False
    return all(value % divisor for divisor in range(3, math.isqrt(value) + 1, 2))


def tables(config, tokenizer_data):
    import numpy as np
    from tokenizers import Regex, Tokenizer, normalizers
    if importlib.metadata.version("numpy") != NUMPY_VERSION or importlib.metadata.version("tokenizers") != TOKENIZERS_VERSION:
        raise ArtifactError("Engram metadata requires the pinned offline NumPy/tokenizers versions")
    text = config.get("text_config")
    required = {"vocab_size": 129280, "engram_compressed_vocab_size": 99092,
        "engram_layer_ids": [1, 14], "engram_max_ngram_size": 4, "engram_n_heads": 8,
        "engram_vocab_size": 16000000, "engram_num_embeddings": [384006168, 384016682]}
    if not isinstance(text, dict) or any(text.get(key) != value for key, value in required.items()) or type(config.get("pad_token_id")) is not int or not 0 <= config["pad_token_id"] < 129280:
        raise ArtifactError("Engram derivation is restricted to the bounded pinned V4.1 configuration")
    loads(tokenizer_data, project=False, max_nodes=2000000)
    tokenizer = Tokenizer.from_str(tokenizer_data.decode("utf-8", errors="strict"))
    vocab = tokenizer.get_vocab_size(with_added_tokens=True)
    if vocab != text["vocab_size"] or vocab > 200000:
        raise ArtifactError("tokenizer vocabulary differs from pinned target")
    sentinel = "\ue000"
    normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(),
        normalizers.Lowercase(), normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), sentinel), normalizers.Strip(), normalizers.Replace(sentinel, " ")])
    token_map, identities = [], {}
    for token_id in range(vocab):
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in decoded:
            key = tokenizer.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(decoded)
            key = normalized if normalized else decoded
        if key not in identities:
            identities[key] = len(identities)
        token_map.append(identities[key])
    if len(identities) != text["engram_compressed_vocab_size"]:
        raise ArtifactError("Engram compressed vocabulary differs from source; no silent Unicode approximation")
    seen, all_primes, all_offsets, multipliers = set(), [], [], []
    for layer_index, layer in enumerate(text["engram_layer_ids"]):
        prime_layer = []
        for _ in range(text["engram_max_ngram_size"] - 1):
            current = text["engram_vocab_size"] - 1
            for _ in range(text["engram_n_heads"]):
                current += 1
                while current in seen or not _prime(current):
                    current += 1
                seen.add(current)
                prime_layer.append(current)
        if sum(prime_layer) != text["engram_num_embeddings"][layer_index]:
            raise ArtifactError("prime table population disagrees with source Engram rows")
        offset, offsets = 0, []
        for prime in prime_layer:
            offsets.append(offset)
            offset += prime
        bound = max(1, (((1 << 63) - 1) // len(identities)) // 2)
        generator = np.random.default_rng(10007 * layer)
        values = generator.integers(low=0, high=bound, size=(text["engram_max_ngram_size"],), dtype=np.int64)
        multipliers.append([int(value) * 2 + 1 for value in values])
        all_primes.append(prime_layer)
        all_offsets.append(offsets)
    return {"token_map": token_map, "primes": all_primes, "offsets": all_offsets,
            "multipliers": multipliers, "compressed_vocab": len(identities),
            "pad_compressed_id": token_map[config["pad_token_id"]]}


def prepare_metadata(config_path, tokenizer_path, output_dir, *, configuration_sha256=None, tokenizer_sha256=None):
    config_data = read_verified_bytes(config_path, expected_sha256=configuration_sha256)
    tokenizer_data = read_verified_bytes(tokenizer_path, expected_sha256=tokenizer_sha256)
    config, output = loads(config_data), Path(output_dir)
    result = tables(config, tokenizer_data)
    tensors = [
        ("engram.token_map", "U32", [len(result["token_map"])], struct.pack("<" + "I" * len(result["token_map"]), *result["token_map"])),
        ("engram.primes", "U64", [2, 3, 8], b"".join(struct.pack("<Q", value) for row in result["primes"] for value in row)),
        ("engram.offsets", "U64", [2, 24], b"".join(struct.pack("<Q", value) for row in result["offsets"] for value in row)),
        ("engram.multipliers", "I64", [2, 4], b"".join(struct.pack("<q", value) for row in result["multipliers"] for value in row)),
        ("engram.pad_id", "U32", [1], struct.pack("<I", result["pad_compressed_id"]))]
    header, data = {}, bytearray()
    for name, dtype, shape, values in tensors:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(data), len(data) + len(values)]}
        data.extend(values)
    encoded = canonical(header)
    encoded += b" " * ((-len(encoded)) % 8)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "engram-metadata.safetensors"
    atomic_bytes(path, struct.pack("<Q", len(encoded)) + encoded + data)
    provenance = seal({"schema_revision": 1, "algorithm": "pinned_deepseek_normalization_pcg64_and_unique_ascending_primes",
        "numpy_version": NUMPY_VERSION, "tokenizers_version": TOKENIZERS_VERSION,
        "configuration_sha256": hashlib.sha256(config_data).hexdigest(), "tokenizer_sha256": hashlib.sha256(tokenizer_data).hexdigest(),
        "engram_metadata_sha256": hashlib.sha256(struct.pack("<Q", len(encoded)) + encoded + data).hexdigest(), "compressed_vocab": result["compressed_vocab"],
        "pad_compressed_id": result["pad_compressed_id"], "multipliers": [[str(value) for value in row] for row in result["multipliers"]],
        "primes": result["primes"], "offsets": result["offsets"]})
    atomic_json(output / "engram-metadata.json", provenance)
    return provenance
