"""Setup inputs reuse the serving schemas; generated JSON has one authority."""

from copy import deepcopy

from .setup_limits import MAX_SETUP_SECONDS, MAX_TRANSFER_SECONDS, MAX_TREE


def setup_schemas():
    # Called after ria.schemas has defined the shared serving contracts. Keeping
    # this import deferred also lets the canonical generator include this input.
    from .schemas import (API, CAPS, EXECUTOR, EXPERT_POLICY, IMAGE, NETWORK,
                          PATH, PLACEMENT_PLAN, POS, PROFILE, REV, ROLE, SHA,
                          nullable, obj)

    deadline = {**POS, "maximum": 3600000}
    network = deepcopy(NETWORK)
    for field in ("control_address", "bulk_address", "server_executor"):
        network["properties"].pop(field)
        network["required"].remove(field)
    for field in network["properties"]:
        if field.endswith("timeout_ms"):
            network["properties"][field] = deadline
    api = deepcopy(API)
    for field in ("bind_address", "bearer_token_file", "max_active_generations",
                  "max_queued_generations", "allow_remote_image_urls", "cors_allowed_origins"):
        api["properties"].pop(field)
        api["required"].remove(field)
    # These are the native API's accepted resource domains. Serving schemas
    # also bind admission evidence; setup rejects impossible inputs earlier.
    for field, maximum in (("max_body_bytes", 64 << 20), ("max_header_bytes", 65536),
                           ("max_json_depth", 64), ("max_json_nodes", 200000),
                           ("max_messages", 10000), ("max_tools", 1024),
                           ("max_images", 16), ("max_encoded_image_bytes", 64 << 20),
                           ("max_http_connections", 64),
                           ("header_timeout_ms", 2147483647),
                           ("body_timeout_ms", 2147483647),
                           ("stream_write_timeout_ms", 2147483647)):
        api["properties"][field] = {**api["properties"][field], "maximum": maximum}
    api["properties"]["max_header_bytes"]["minimum"] = 128
    runtime = deepcopy(PLACEMENT_PLAN["properties"]["runtime"])
    for field in ("tokenizer_file", "tokenizer_sha256", "prefill_rows"):
        runtime["properties"].pop(field)
        runtime["required"].remove(field)
    placement = PLACEMENT_PLAN["properties"]
    client = obj({"runtime": runtime, **{field: deepcopy(placement[field]) for field in (
        "host_expert_cache_bytes", "device_expert_cache_bytes", "engram_cache_bytes", "local_experts")}})
    source = obj({"mode": {"const": "source"}, "source_dir": PATH,
                  "chunk_size": {**POS, "maximum": 4 << 20}, "max_shard_bytes": POS,
                  "scratch_bytes": POS}, ("chunk_size", "max_shard_bytes", "scratch_bytes"))
    prepared = obj({"mode": {"const": "prepared"}, "package_dir": PATH, "trusted_manifest_digest": SHA})
    download = obj({"mode": {"const": "download"}, "max_download_bytes": POS,
        "download_deadline_ms": {**POS, "maximum": MAX_SETUP_SECONDS * 1000}, "hf_token_file": nullable(PATH),
        "chunk_size": {**POS, "maximum": 4 << 20}, "max_shard_bytes": POS, "scratch_bytes": POS},
        ("hf_token_file", "chunk_size", "max_shard_bytes", "scratch_bytes"))
    properties = {"schema_revision": REV, "role": ROLE, "executor": EXECUTOR,
        "profile": PROFILE, "workspace": PATH, "service_image": IMAGE,
        "bind_address": {"type": "string", "minLength": 1, "maxLength": 15},
        "peer_address": {"type": "string", "minLength": 1, "maxLength": 15},
        "server_executor": EXECUTOR,
        "gpu_uuid": nullable({"type": "string", "pattern": "^GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$"}),
        "planning": obj({"context_positions": {**POS, "maximum": 1048576},
                         "prefill_rows": {**POS, "maximum": 64}, "caps": deepcopy(CAPS)}),
        "environment": obj({"cpuset": {"type": "string", "maxLength": 4096,
                            "pattern": "^[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*$"},
            **{field: POS for field in ("cgroup_bytes", "memlock_bytes", "pids_limit",
                "start_period_seconds", "stop_grace_seconds")},
            "api_port": {**POS, "maximum": 65535}}),
        "probe": obj({"max_host_test_bytes": POS, "max_device_test_bytes": {**POS, "minimum": 0},
                      "max_pinned_test_bytes": {**POS, "minimum": 0}, "deadline_ms": deadline}),
        "network": network, "deadline_ms": deadline, "max_transfer_bytes": {**POS, "maximum": MAX_TREE},
        "transfer_deadline_ms": {**POS, "maximum": MAX_TRANSFER_SECONDS * 1000},
        "setup_deadline_ms": {**POS, "maximum": MAX_SETUP_SECONDS * 1000}, "setup_port": {**POS, "minimum": 1024, "maximum": 65535},
        "security": obj({"mode": {"enum": ["tls", "trusted_network"]},
                         "api_token_file": nullable(PATH)}, ("mode", "api_token_file")),
        "model": {"oneOf": [source, prepared, download]}, "expert": deepcopy(EXPERT_POLICY),
        "client_runtime": client, "api": api}
    settings = obj(properties, ("peer_address", "gpu_uuid", "setup_port", "security",
                               "model", "expert", "client_runtime", "api"))
    settings["allOf"] = [{"if": {"properties": {"role": {"const": "expert"}}},
        "then": {"required": ["model", "expert"], "not": {"anyOf": [
            {"required": ["client_runtime"]}, {"required": ["api"]}]}},
        "else": {"required": ["client_runtime", "api"], "not": {"anyOf": [
            {"required": ["model"]}, {"required": ["expert"]}]}}}]
    return {"setup-settings": settings}
