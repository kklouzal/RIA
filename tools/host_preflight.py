#!/usr/bin/env python3
"""Record target-host topology, or verify a running container's hidden cgroup parents."""

import argparse
import subprocess
import sys

from ria.host import save_host, verify_container_ancestors
from ria.identity import ArtifactError, atomic_json, canonical, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    observe = commands.add_parser("observe")
    observe.add_argument("--cgroup-parent", required=True)
    observe.add_argument("--gpu-uuid")
    observe.add_argument("--client", action="store_true")
    observe.add_argument("--output", required=True)
    verify = commands.add_parser("verify-container")
    verify.add_argument("--pid", required=True, type=int)
    verify.add_argument("--host-report", required=True)
    verify.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.command == "observe":
            value = save_host(args.cgroup_parent, args.output, args.gpu_uuid, client=args.client)
        else:
            value = verify_container_ancestors(args.pid, read_json(args.host_report))
            atomic_json(args.output, value)
        sys.stdout.buffer.write(canonical({"digest": value["digest"], "hardware_qualified": False}) + b"\n")
        return 0
    except (ArtifactError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"host_preflight: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
