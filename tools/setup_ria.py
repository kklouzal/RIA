#!/usr/bin/env python3
"""Prepare, pair, qualify and launch hardened RIA services from a setup container."""

import argparse
import json
from pathlib import Path
import sys

from ria.identity import ArtifactError, loads, read_json


def override(document, assignment):
    """Typed existing-key overrides; external text is never executable code."""
    path, separator, raw = assignment.partition("=")
    keys = path.split(".")
    if not separator or any(not key for key in keys):
        raise ArtifactError("--set requires existing.dotted.key=JSON-value")
    parent = document
    for key in keys[:-1]:
        if not isinstance(parent, dict) or key not in parent:
            raise ArtifactError("--set path does not exist in settings")
        parent = parent[key]
    if not isinstance(parent, dict) or keys[-1] not in parent:
        raise ArtifactError("--set path does not exist in settings")
    parent[keys[-1]] = loads(raw, max_bytes=65536, max_depth=8, max_nodes=1000)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate settings/policy without Docker, weights or hardware")
    run = commands.add_parser("run", help="run the locally authorized paired setup workflow")
    for command in (validate, run):
        command.add_argument("--settings", required=True)
        command.add_argument("--qualification", help="expert-only accepted policy and fixture tuning")
        command.add_argument("--invitation", help="client-only private trusted expert invitation")
        command.add_argument("--set", action="append", default=[], metavar="KEY=JSON")
    status = commands.add_parser("status", help="show stage receipts without revealing pairing/API credentials")
    status.add_argument("--workspace", required=True)
    stop = commands.add_parser("stop", help="stop/remove only containers owned by the recorded workspace")
    stop.add_argument("--workspace", required=True)
    template = commands.add_parser("template", help="print settings with null policy fields to fill explicitly")
    template.add_argument("--role", required=True, choices=("expert", "client"))
    template.add_argument("--executor", required=True, choices=("cpu", "cuda"))
    commands.add_parser("qualification-template", help="print explicit policy/fixture fields; fill nulls before use")
    try:
        args = parser.parse_args(argv)
        if args.command == "template":
            from ria.setup_config import template_settings
            value = template_settings(args.role, args.executor)
        elif args.command == "qualification-template":
            from ria.setup_qualification import template_tuning
            value = template_tuning()
        elif args.command == "stop":
            from ria.setup import stop_setup
            value = stop_setup(args.workspace)
        elif args.command == "status":
            root = Path(args.workspace)
            value = {"workspace": str(root), "stages": {}}
            for name in ("local-journal", "peer-journal"):
                directory = root / name
                if directory.is_dir():
                    files = list(directory.glob("*.json"))
                    if len(files) > 128:
                        raise ArtifactError("setup journal population exceeded")
                    value["stages"][name] = [{"step_id": record["step_id"], "status": record["status"]}
                        for record in (read_json(path, max_bytes=1 << 20) for path in sorted(files))]
        else:
            from ria.setup_config import validate_settings
            from ria.setup_qualification import validate_tuning
            settings = read_json(args.settings, max_bytes=2 << 20)
            for assignment in args.set:
                override(settings, assignment)
            settings = validate_settings(settings)
            tuning = validate_tuning(read_json(args.qualification)) if args.qualification else None
            from ria.setup_security import read_invitation
            invitation = read_invitation(args.invitation) if args.invitation else None
            if settings["role"] == "expert" and (tuning is None or invitation is not None):
                raise ArtifactError("expert requires --qualification and creates its own invitation")
            if settings["role"] == "client" and ((args.command == "run" and invitation is None) or tuning is not None):
                raise ArtifactError("client requires --invitation; expert supplies the frozen qualification policy")
            if args.command == "validate":
                value = {"valid": True, "role": settings["role"], "executor": settings["executor"], "hardware_executed": False}
            else:
                from ria.setup import run_setup
                value = run_setup(settings, tuning=tuning, invitation=invitation)
        print(json.dumps(value, sort_keys=True, indent=2 if args.command in ("template", "qualification-template") else None))
        return 0
    except (ArtifactError, OSError, ValueError, KeyError) as error:
        print(str(error), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("setup interrupted; owned cleanup was attempted and evidence remains in the workspace", file=sys.stderr)
        from ria.setup import SetupTerminated
        return 143 if isinstance(sys.exception(), SetupTerminated) else 130


if __name__ == "__main__":
    sys.exit(main())
