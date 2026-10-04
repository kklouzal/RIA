#!/usr/bin/env python3
"""Metadata-only memory inventory; no weights, model, NUMA probe or CUDA init."""

import argparse
import sys

from ria.identity import ArtifactError, read_json
from ria.inventory import build_inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        build_inventory(read_json(args.request), args.manifest, args.output)
    except (ArtifactError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
