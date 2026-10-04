"""Derive fixture registration from accepted policy and actual role facts.

Only tuning and hard limits are operator input. Model, source, binary, build,
environment and certificate identities are derived; no evidence is synthesized.
"""

from copy import deepcopy
from pathlib import Path

from .fixture_runner import (HARD_LIMITS, RUNS, freeze_registration,
                             registration_schemas)
from .identity import ArtifactError, check_json, read_json, verify_identity
from .qualification import freeze_policy
from .qualification_schema import POLICY, record


def setup_qualification_schema():
    registration = registration_schemas()["fixture-registration-request"]
    runs = registration["properties"]["runs"]["properties"]
    native = runs["native_server"]["properties"]["request_body"]["properties"]
    transport = runs["transport_server"]["properties"]["request_body"]["properties"]
    policy = deepcopy(POLICY)
    for name in ("logical_model_digest", "source_lock_digest", "registered_at", "digest"):
        policy["properties"].pop(name)
        if name in policy["required"]:
            policy["required"].remove(name)
    limits = deepcopy(HARD_LIMITS)
    for name in ("host_bytes", "device_bytes", "pinned_bytes", "max_rss_bytes"):
        limits["required"].remove(name)
    return record({"schema_revision": {"const": 1}, "policy": policy,
        "native": record({name: deepcopy(native[name]) for name in
            ("deadline_ms", "warmup", "repeats", "fixture_seed")}),
        "transport": record({name: deepcopy(transport[name]) for name in
            ("deadline_ms", "warmup", "repeats", "fixture_seed", "max_frame_bytes",
             "control_credit", "expert_credit", "row_credit", "bulk_credit")}),
        "hard_limits": record({name: deepcopy(limits) for name in RUNS})})


def validate_tuning(document):
    from .schemas import StrictValidator
    check_json(document, max_nodes=10000, max_depth=16)
    error = next(StrictValidator(setup_qualification_schema()).iter_errors(document), None)
    if error:
        raise ArtifactError(f"setup qualification: {error.message}")
    if document["native"]["repeats"] < 2 or document["transport"]["repeats"] < 2:
        raise ArtifactError("admission requires at least two measured repetitions")
    return deepcopy(document)


def template_tuning():
    """Show every accepted input without inventing thresholds or hard limits."""
    def blank(schema):
        if "const" in schema:
            return schema["const"]
        if schema.get("type") == "object":
            return {name: blank(schema["properties"][name]) for name in schema["required"]}
        if schema.get("type") == "array":
            return [None]
        return None
    return blank(setup_qualification_schema())


def register(tuning, model, builds, environments, security, output):
    """Publish one policy and registration after both environments are frozen."""
    tuning = validate_tuning(tuning)
    output = Path(output)
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    for role in ("server", "client"):
        verify_identity(builds[role])
        verify_identity(environments[role])
    if builds["server"]["source_lock_digest"] != builds["client"]["source_lock_digest"]:
        raise ArtifactError("paired images have different source locks")
    policy_path = output / "policy.json"
    freeze_policy({**tuning["policy"], "logical_model_digest": model["logical_model_digest"],
        "source_lock_digest": builds["server"]["source_lock_digest"]}, policy_path)
    policy = read_json(policy_path)
    identities = {"policy_digest": policy["digest"], "logical_model_digest": model["logical_model_digest"],
        "source_lock_digest": policy["source_lock_digest"], "operator_contract_digest": model["operator_contract_digest"]}
    realizations = {role: {"environment_digest": environments[role]["digest"],
        "build_digest": builds[role]["digest"]} for role in ("server", "client")}
    runs = {}
    for name in RUNS:
        role = name.rsplit("_", 1)[1]
        environment = environments[role]
        default_limits = {key: str(environment["planning_request"]["caps"][key])
            for key in ("host_bytes", "device_bytes", "pinned_bytes")}
        default_limits["max_rss_bytes"] = str(environment["environment"]["cgroup_bytes"])
        if name.startswith("transport_"):
            default_limits.update(device_bytes="0", pinned_bytes="0")
        limits = {**default_limits, **tuning["hard_limits"][name]}
        common = {"schema_revision": 1, **identities, **realizations[role]}
        if name.startswith("native_"):
            body = {**common, **tuning["native"], "kind": "native_fixture_request",
                "profile": model["profile"], "executor": environments["server"]["planning_request"]["executor"],
                "role": role, "runner_executor": environment["planning_request"]["executor"],
                "gpu_uuid": environment["environment"]["gpu_uuid"], "expert_shape": "target",
                "host_budget": limits["host_bytes"], "device_budget": limits["device_bytes"],
                "pinned_budget": limits["pinned_bytes"],
                "relative_floor": policy["thresholds"]["same_realization"]["relative_floor"]}
            runs[name] = {"request_body": body, "hard_limits": limits}
        else:
            tls = environment["tls"]
            if tls.get("enabled", True):
                bootstrap_tls = {key: tls[key] for key in
                    ("ca_file", "certificate_file", "private_key_file", "expected_peer_name")}
                bootstrap_tls["authorized_peer_sha256"] = security["client" if role == "server" else "server"]
            else:
                bootstrap_tls = {"enabled": False}
            network = {key: environment["network"][key] for key in
                ("control_address", "bulk_address", "connect_timeout_ms", "handshake_timeout_ms",
                 "operation_timeout_ms", "frame_io_timeout_ms", "write_timeout_ms")}
            runs[name] = {"request_body": {**common, **tuning["transport"], "kind": "transport_request"},
                "hard_limits": limits, "bootstrap_body": {"schema_revision": 1,
                    "role": "expert" if role == "server" else "client", **realizations[role],
                    "build_info_file": "/usr/share/dwarfstar/build-info.json", "network": network,
                    "tls": bootstrap_tls}}
    document = {"schema_revision": 1, "kind": "fixture_registration_request", "qualification_scope": "initial_fixture",
        "profile": model["profile"], "server_executor": environments["server"]["planning_request"]["executor"],
        **identities, "realizations": realizations, "runs": runs}
    from .identity import atomic_json
    paths = {"builds": {}, "environments": {}}
    for role in ("server", "client"):
        for group, values in (("builds", builds), ("environments", environments)):
            path = output / f"{role}-{group}.json"
            atomic_json(path, values[role])
            paths[group][role] = path
    result = freeze_registration(document, policy, paths["builds"], paths["environments"], output / "registration.json")
    return {"policy": policy, "registration": result}
