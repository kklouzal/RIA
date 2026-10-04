#!/usr/bin/env python3
"""Freeze, validate or explicitly execute a bounded private loopback replay plan."""

import argparse
from pathlib import Path
import sys

from ria.identity import ArtifactError, atomic_json, canonical, read_json
from ria.release_runner import execute_plan, freeze_plan, validate_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--register-plan", action="store_true")
    modes.add_argument("--validate-plan", action="store_true")
    modes.add_argument("--execute", action="store_true")
    for name in ("plan", "policy", "workload-root"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--bearer-token-file")
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        plan = read_json(args.plan, max_bytes=256 << 10, max_nodes=20000, max_depth=32)
        policy = read_json(args.policy, max_bytes=256 << 10)
        root = Path(args.workload_root)
        if args.register_plan:
            if args.output is None:
                raise ArtifactError("registration requires output")
            result = freeze_plan(plan, root, policy, args.output)
            summary = {"digest": result["digest"], "registered": True, "executed": False}
        elif args.validate_plan:
            validate_plan(plan, root, policy)
            summary = {"digest": plan["digest"], "valid": True, "executed": False}
        else:
            if args.bearer_token_file is None or args.output is None:
                raise ArtifactError("execution requires credential file and output")
            relative_plan = Path(args.plan).absolute().relative_to(root.resolve()).as_posix()
            result = execute_plan(plan, root, policy, args.bearer_token_file, execute=True, plan_path=relative_plan)
            atomic_json(args.output, result)
            summary = {"digest": result["digest"], "passed": result["passed"], "qualified": False}
        sys.stdout.buffer.write(canonical(summary) + b"\n")
        return 0 if not args.execute or result["passed"] else 1
    except (ArtifactError, OSError, ValueError):
        # Decoder/network exception text may contain private input. Publish a
        # static code; detailed response bodies never become diagnostics.
        print("run_ria_release: invalid contract or execution failure", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
