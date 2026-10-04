#!/usr/bin/env python3
"""Inspect/estimate/prepare explicit reviewed local artifacts; never download weights."""

import argparse
import sys

from ria.identity import ArtifactError, atomic_json, canonical, read_json
from ria.preparation import estimate, prepare, verify_package
from ria.safetensors import inspect


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    inspect_parser = sub.add_parser("inspect")
    inspect_parser.add_argument("path")
    recipe_parser = sub.add_parser("recipe")
    recipe_parser.add_argument("--source-dir", required=True)
    recipe_parser.add_argument("--profile", choices=("nvfp4", "fp8", "bf16"), required=True)
    recipe_parser.add_argument("--output", required=True)
    recipe_parser.add_argument("--chunk-size", type=int, default=4 << 20)
    recipe_parser.add_argument("--max-shard-bytes", type=int, default=128 << 30)
    recipe_parser.add_argument("--scratch-bytes", type=int, default=64 << 20)
    engram_parser = sub.add_parser("engram-metadata")
    engram_parser.add_argument("--config", required=True)
    engram_parser.add_argument("--tokenizer", required=True)
    engram_parser.add_argument("--output-dir", required=True)
    client_parser = sub.add_parser("client-package")
    client_parser.add_argument("--server-package", required=True)
    client_parser.add_argument("--trusted-manifest-digest", required=True)
    client_parser.add_argument("--output-dir", required=True)
    client_parser.add_argument("--selected-tensors")
    for name in ("estimate", "prepare"):
        command = sub.add_parser(name)
        command.add_argument("--source-root", "--source-dir", dest="source_root", required=True)
        command.add_argument("--recipe")
        command.add_argument("--profile", choices=("nvfp4", "fp8", "bf16"))
        if name == "prepare":
            command.add_argument("--output-dir", required=True)
            command.add_argument("--role", choices=("server", "client"), default="server")
            command.add_argument("--selected-tensors", help="JSON array of allowed immutable cache tensor names")
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("directory")
    args = parser.parse_args(argv)
    try:
        if args.operation == "recipe":
            from ria.target import create_recipe
            result = create_recipe(args.source_dir, args.profile, chunk_size=args.chunk_size,
                max_shard_bytes=args.max_shard_bytes, scratch_bytes=args.scratch_bytes)
            atomic_json(args.output, result)
            result = {"recipe_digest": result["digest"], "profile": result["profile"]}
        elif args.operation == "client-package":
            from ria.client import client_package
            selected = read_json(args.selected_tensors) if args.selected_tensors else []
            if not isinstance(selected, list) or any(not isinstance(name, str) for name in selected):
                raise ArtifactError("selected-tensors must be a JSON string array")
            result = client_package(args.server_package, args.trusted_manifest_digest, args.output_dir,
                selected_names=selected)
            result = {"prepared": True, "digest": result["digest"], "role": result["role"]}
        elif args.operation == "engram-metadata":
            from ria.engram import prepare_metadata
            result = prepare_metadata(args.config, args.tokenizer, args.output_dir)
        elif args.operation == "inspect":
            shard = inspect(args.path)
            result = {"size": str(shard.size), "data_start": str(shard.data_start), "tensors": [
                {"name": item.name, "dtype": item.dtype, "shape": list(item.shape), "offset": str(item.offset), "length": str(item.length)}
                for item in shard.tensors.values()]}
        elif args.operation == "verify":
            manifest = verify_package(args.directory)
            result = {"verified": True, "digest": manifest["digest"]}
        else:
            if args.recipe:
                recipe = read_json(args.recipe, max_bytes=256 << 20, max_nodes=16000000)
            elif args.profile:
                from ria.target import create_recipe
                recipe = create_recipe(args.source_root, args.profile)
            else:
                raise ArtifactError("provide --recipe or --profile for automatic local target preparation")
            if args.operation == "estimate":
                result = estimate(args.source_root, recipe)
            else:
                selected = read_json(args.selected_tensors) if args.selected_tensors else None
                if selected is not None and (not isinstance(selected, list) or any(not isinstance(name, str) for name in selected)):
                    raise ArtifactError("selected-tensors must be a JSON string array")
                result = prepare(args.source_root, recipe, args.output_dir, role=args.role, selected_names=selected)
                result = {"prepared": True, "digest": result["digest"], "role": result["role"]}
        sys.stdout.buffer.write(canonical(result) + b"\n")
        return 0
    except (ArtifactError, OSError, ImportError) as exc:
        print(f"prepare_ria: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
