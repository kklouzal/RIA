#!/usr/bin/env python3
"""Render immutable host-local bootstrap/final deployment packages."""

import argparse
import sys
from pathlib import Path

from ria.deployment import bootstrap, finalize, fixture_start, fixture_stop, fixture_exec, launch
from ria.identity import ArtifactError, canonical, read_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    for operation in ("bootstrap", "finalize", "launch", "fixture-start", "fixture-stop", "fixture-exec"):
        command = sub.add_parser(operation)
        command.add_argument("--request", required=True)
        command.add_argument("--output-dir", required=True)
        if operation == "fixture-exec":
            command.add_argument("--run", required=True, choices=("probe", "native_server", "native_client", "transport_server", "transport_client"))
        if operation == "finalize":
            command.add_argument("--probe", required=True)
            command.add_argument("--inventory", required=True)
            command.add_argument("--calibration", required=True)
            command.add_argument("--probe-evidence", required=True)
            command.add_argument("--calibration-evidence", required=True)
            command.add_argument("--policy", required=True)
    args = parser.parse_args(argv)
    try:
        request = read_json(args.request)
        if args.operation == "bootstrap":
            result = bootstrap(request, args.output_dir)
        elif args.operation == "finalize":
            result = finalize(request, read_json(args.probe), read_json(args.inventory), read_json(args.calibration), args.output_dir,
                              probe_evidence=read_json(args.probe_evidence), calibration_evidence=read_json(args.calibration_evidence),
                              calibration_evidence_dir=Path(args.calibration_evidence).absolute().parent, policy=read_json(args.policy))
        elif args.operation == "fixture-start":
            result = fixture_start(request, args.output_dir)
        elif args.operation == "fixture-stop":
            result = fixture_stop(request, args.output_dir)
        elif args.operation == "fixture-exec":
            fixture_exec(request, args.output_dir, args.run)
            result = {"completed": True, "run": args.run}
        else:
            launch(request, args.output_dir)
            result = {"started": True}
        sys.stdout.buffer.write(canonical(result) + b"\n")
        return 0
    except (ArtifactError, OSError, ValueError) as exc:
        print(f"render_deployment: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
