#!/usr/bin/env python3
"""Verify bounded local schema/identity/artifact/source-lock contracts."""

import argparse
import sys
from pathlib import Path

from ria.identity import ArtifactError, canonical, hash_file, read_json, verify_identity, within
from ria.preparation import verify_package
from ria.schemas import SCHEMAS, validate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", choices=sorted(SCHEMAS))
    parser.add_argument("--document")
    parser.add_argument("--package")
    parser.add_argument("--source-lock")
    args = parser.parse_args(argv)
    try:
        if args.package:
            result = {"verified": True, "manifest_digest": verify_package(args.package)["digest"]}
        elif args.source_lock:
            path = Path(args.source_lock).absolute()
            document = read_json(path)
            verify_identity(document)
            root = path.parent.parent
            for reference in document["references"]:
                if "error" in reference:
                    raise ArtifactError("source lock contains an unresolved retrieval failure")
                if hash_file(within(root, reference["local_path"])) != reference["sha256"]:
                    raise ArtifactError("locked metadata/reference file hash mismatch")
            for item in document.get("derived_metadata", []):
                if hash_file(within(root, item["path"])) != item["sha256"]:
                    raise ArtifactError("derived immutable metadata hash mismatch")
            result = {"verified": True, "source_lock_digest": document["digest"], "classification": document["classification"]}
        elif args.schema and args.document:
            document = read_json(args.document)
            validate(args.schema, document)
            if "digest" in document:
                verify_identity(document)
            result = {"verified": True, "schema": args.schema}
        else:
            parser.error("provide --package, --source-lock, or --schema and --document")
        sys.stdout.buffer.write(canonical(result) + b"\n")
        return 0
    except (ArtifactError, OSError) as exc:
        print(f"verify_ria: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
