#!/usr/bin/env python3
"""Internal bounded preparation worker; no Engine, peer or hardware authority."""

import argparse
import sys

from ria.identity import ArtifactError, canonical, read_json
from ria.setup_artifacts import prepare_expert
from ria.setup_config import validate_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", required=True)
    parser.add_argument("--selection", required=True)
    args = parser.parse_args()
    try:
        settings = validate_settings(read_json(args.settings))
        if settings["role"] != "expert":
            raise ArtifactError("preparation worker requires expert settings")
        selection = read_json(args.selection)
        if set(selection) != {"local_experts"}:
            raise ArtifactError("invalid preparation selection fields")
        if settings["model"]["mode"] == "download":
            from ria.setup_acquisition import acquire_source
            model = settings["model"]
            facts = acquire_source(settings["workspace"], max_bytes=model["max_download_bytes"],
                deadline_ms=model["download_deadline_ms"], token_file=model.get("hf_token_file"))
            settings["model"] = {"mode": "source", "source_dir": facts["source_dir"],
                **{key: model[key] for key in ("chunk_size", "max_shard_bytes", "scratch_bytes")}}
        result = prepare_expert(settings, settings["workspace"], local_experts=selection["local_experts"])
        sys.stdout.buffer.write(canonical(result) + b"\n")
        return 0
    except (ArtifactError, OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
