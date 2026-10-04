#!/usr/bin/env python3
"""Preregister explicit tolerances, compare saved logits and enumerate hardware gates."""

import argparse
from pathlib import Path
import sys
import subprocess

from ria.identity import ArtifactError, atomic_json, canonical, read_json
from ria.qualification import calibration_report, compare_files, freeze_policy, release_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--policy", required=True)
    freeze.add_argument("--output", required=True)
    matrix = commands.add_parser("matrix")
    matrix.add_argument("--output", required=True)
    compare = commands.add_parser("compare-logits")
    for name in ("reference", "candidate", "input", "policy", "output"):
        compare.add_argument("--" + name, required=True)
    compare.add_argument("--axis", required=True, choices=["same_realization", "native_source"])
    calibration = commands.add_parser("calibration")
    for name in ("evidence", "policy", "output"):
        calibration.add_argument("--" + name, required=True)
    registration = commands.add_parser("register-fixtures")
    for name in ("input", "policy", "server-build-info", "client-build-info", "server-environment", "client-environment", "output"):
        registration.add_argument("--" + name, required=True)
    requests = commands.add_parser("fixture-requests")
    for name in ("registration", "policy", "output-dir"):
        requests.add_argument("--" + name, required=True)
    run = commands.add_parser("run-fixture")
    for name in ("registration", "policy", "run", "executable", "build-info", "output-dir", "environment", "probe-config", "probe", "probe-evidence"):
        run.add_argument("--" + name, required=True)
    run.add_argument("--transport-config")
    components = commands.add_parser("components")
    for name in ("registration", "policy", "admission-role", "native-server", "native-client", "transport-server", "transport-client", "output-dir"):
        components.add_argument("--" + name, required=True)
    args = parser.parse_args()
    try:
        if args.command == "freeze":
            result = freeze_policy(read_json(args.policy), args.output)
        elif args.command == "matrix":
            result = release_matrix()
            atomic_json(args.output, result)
        elif args.command == "calibration":
            result = calibration_report(read_json(args.evidence), Path(args.evidence).resolve().parent, read_json(args.policy))
            atomic_json(args.output, result)
        elif args.command == "register-fixtures":
            from ria.fixture_runner import freeze_registration
            result = freeze_registration(read_json(args.input), read_json(args.policy),
                {"server": args.server_build_info, "client": args.client_build_info},
                {"server": args.server_environment, "client": args.client_environment}, args.output)
        elif args.command == "fixture-requests":
            from ria.fixture_runner import publish_requests
            result = publish_requests(read_json(args.registration), read_json(args.policy), args.output_dir)
        elif args.command == "run-fixture":
            from ria.fixture_runner import execute_run
            result = execute_run(read_json(args.registration), args.run, read_json(args.policy), args.executable,
                                 args.build_info, args.output_dir, transport_config=args.transport_config,
                                 environment=args.environment, probe_config=args.probe_config, probe=args.probe, probe_evidence=args.probe_evidence)
        elif args.command == "components":
            from ria.fixture_runner import produce_components
            result = produce_components(read_json(args.registration), read_json(args.policy),
                {"native_server": args.native_server, "native_client": args.native_client,
                 "transport_server": args.transport_server, "transport_client": args.transport_client},
                args.output_dir, role=args.admission_role)
        else:
            result = compare_files(args.reference, args.candidate, args.input, args.policy, args.axis, args.output)
        sys.stdout.buffer.write(canonical({"digest": result.get("digest", result.get("raw_digest")), "passed": result.get("passed", False)}) + b"\n")
        return 0 if args.command != "compare-logits" or result["passed"] else 1
    except (ArtifactError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"qualify_ria: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
