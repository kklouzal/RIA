#!/usr/bin/env python3
"""Reproduce the narrow NUMA extension of the exact Docker Engine profile."""
import argparse
import hashlib
import json
from pathlib import Path


def generate(directory):
    lock = json.loads((directory / "seccomp-source.json").read_text())
    for record in lock["inputs"]:
        data = (directory / record["path"]).read_bytes()
        if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError("pinned Engine seccomp source changed")
    source = json.loads((directory / "seccomp-engine-default.json").read_bytes())
    if source["defaultAction"] != "SCMP_ACT_ERRNO":
        raise ValueError("source no longer has the reviewed default-deny policy")
    def argument(index, value):
        return {"index": index, "value": value, "op": "SCMP_CMP_EQ"}
    # These interfaces target only this process's own memory policy. mbind
    # applies before first touch, with no migration flags. Locality queries
    # cannot specify another process or move a page.
    source["syscalls"].extend([
        {"names": ["get_mempolicy"], "action": "SCMP_ACT_ALLOW"},
        {"names": ["set_mempolicy"], "action": "SCMP_ACT_ALLOW", "args": [argument(0, 0)]},
        {"names": ["set_mempolicy"], "action": "SCMP_ACT_ALLOW", "args": [argument(0, 2)]},
        {"names": ["mbind"], "action": "SCMP_ACT_ALLOW", "args": [argument(2, 2), argument(5, 0)]},
        {"names": ["move_pages"], "action": "SCMP_ACT_ALLOW", "args": [argument(0, 0), argument(3, 0), argument(5, 0),
            {"index": 1, "value": 64, "op": "SCMP_CMP_LE"}]},
    ])
    return (json.dumps(source, indent=2, sort_keys=True) + "\n").encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    content = generate(directory)
    output = directory / "seccomp-numa.json"
    if args.check:
        if output.read_bytes() != content:
            parser.exit(1, "generated NUMA seccomp profile is stale\n")
    else:
        output.write_bytes(content)


if __name__ == "__main__":
    main()
