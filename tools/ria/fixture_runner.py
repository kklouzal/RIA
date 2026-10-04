"""Preregister and supervise bounded native measurements before model admission.

The four request bodies omit the registration identity. Sealing that immutable
population first and deriving executable requests afterwards keeps provenance
acyclic. Raw native reports remain unqualified; every admitted component is
rederived from authenticated reports and complete-lifetime wait4 observations.
"""

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import os
import shutil
import stat
import subprocess
import tempfile
import time

from .identity import (ArtifactError, SAFE_INTEGER, atomic_bytes, atomic_json,
                       check_json, digest, hash_file, loads, read_json,
                       seal, sync_directory, u64, verify_identity, within)
from .native_fixture_schema import (EXACT_CASES, REPEATED_CASES, TIMING_GROUPS, NATIVE_REQUEST,
                                   NATIVE_SEALED_MEASUREMENTS, expected_cases, expected_dimensions,
                                   expected_samples, expected_warmup)
from .process import run_bounded
from .fixture_environment import PREFLIGHT, observe_runtime, validate_preflight
from .qualification_schema import SHA, record

RUNS = ("native_server", "native_client", "transport_server", "transport_client")
LIMIT_NAMES = ("host_bytes", "device_bytes", "pinned_bytes", "max_rss_bytes",
               "max_elapsed_ns", "max_startup_ns", "max_case_latency_ns")
DECIMAL = {"type": "string", "pattern": "^(0|[1-9][0-9]{0,19})$"}
REALIZATION = record({"environment_digest": SHA, "build_digest": SHA})
HARD_LIMITS = record({name: DECIMAL for name in LIMIT_NAMES})
IDENTITIES = ("policy_digest", "logical_model_digest", "source_lock_digest", "operator_contract_digest")
SUPERVISION = record({"schema_revision": {"const": 1}, "kind": {"const": "fixture_supervision"},
    "run": {"enum": list(RUNS)}, "registration_digest": SHA, "request_digest": SHA,
    "raw_digest": SHA, "binary_sha256": SHA, "build_digest": SHA,
    "wall_ns": DECIMAL, "peak_rss_bytes": DECIMAL, "stdout_bytes": DECIMAL, "stderr_bytes": DECIMAL,
    "rss_scope": {"const": "native wait4 complete child lifetime; descendants require their own cgroup cap"},
    "exit_code": {"const": 0}, "preflight": PREFLIGHT, "digest": SHA})


def registration_schemas():
    from .schemas import TRANSPORT_REQUEST, TRANSPORT_BOOTSTRAP
    native = deepcopy(NATIVE_REQUEST)
    transport = deepcopy(TRANSPORT_REQUEST)
    for schema in (native, transport):
        for field in ("preregistration_digest", "digest"):
            schema["properties"].pop(field, None)
            if field in schema["required"]:
                schema["required"].remove(field)
    runs = {name: record({"request_body": native if name.startswith("native_") else transport,
                         "binary_sha256": SHA, "hard_limits": HARD_LIMITS}) for name in RUNS}
    bootstrap = deepcopy(TRANSPORT_BOOTSTRAP)
    del bootstrap["properties"]["request_digest"]
    bootstrap["required"].remove("request_digest")
    for name in ("transport_server", "transport_client"):
        runs[name]["properties"]["bootstrap_body"] = bootstrap
        runs[name]["required"].append("bootstrap_body")
    registration = record({"schema_revision": {"const": 1}, "kind": {"const": "fixture_registration"},
        "qualification_scope": {"const": "initial_fixture"}, "profile": NATIVE_REQUEST["properties"]["profile"],
        "server_executor": NATIVE_REQUEST["properties"]["executor"], **{name: SHA for name in IDENTITIES},
        "realizations": record({"server": REALIZATION, "client": REALIZATION}),
        "runs": record(runs), "registered_at": {"type": "string", "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"},
        "digest": SHA})
    request = deepcopy(registration)
    request["properties"]["kind"] = {"const": "fixture_registration_request"}
    for field in ("registered_at", "digest"):
        del request["properties"][field]
        request["required"].remove(field)
    for run in request["properties"]["runs"]["properties"].values():
        del run["properties"]["binary_sha256"]
        run["required"].remove("binary_sha256")
    return {"fixture-registration": registration, "fixture-registration-request": request,
            "fixture-supervision": SUPERVISION}


def _schema(schema, document, context):
    from .schemas import StrictValidator
    check_json(document, max_nodes=100000, max_depth=32)
    error = next(StrictValidator(schema).iter_errors(document), None)
    if error:
        raise ArtifactError(f"{context}: {error.message}")


def _integer(value):
    return u64(value) if isinstance(value, str) else value


def _run_role(name):
    return name.rsplit("_", 1)[-1]


def _registration_relations(document, policy):
    from .qualification import policy_validate
    policy_validate(policy)
    if document["policy_digest"] != policy["digest"] or any(document[name] != policy[name] for name in ("logical_model_digest", "source_lock_digest")):
        raise ArtifactError("fixture registration differs from frozen policy/model/source")
    for name, run in document["runs"].items():
        body, limits = run["request_body"], run["hard_limits"]
        role = _run_role(name)
        if any(body[key] != document[key] for key in IDENTITIES) or any(body[key] != document["realizations"][role][key] for key in ("environment_digest", "build_digest")):
            raise ArtifactError("fixture body has another role realization or policy identity")
        amounts = {key: u64(value) for key, value in limits.items()}
        if any(value > SAFE_INTEGER for value in amounts.values()) or any(not amounts[key] for key in LIMIT_NAMES if key not in ("device_bytes", "pinned_bytes")):
            raise ArtifactError("fixture hard limits must be explicit positive bounded native values")
        if amounts["max_elapsed_ns"] > body["deadline_ms"] * 1000000 or any(amounts[key] > amounts["max_elapsed_ns"] for key in ("max_startup_ns", "max_case_latency_ns")):
            raise ArtifactError("fixture timing limits exceed the supervised request deadline")
        if body["repeats"] < 2:
            raise ArtifactError("fixture qualification requires at least two measured repetitions")
        if name.startswith("native_"):
            if (body["profile"], body["executor"], body["role"], body["runner_executor"]) != (document["profile"], document["server_executor"], role, "cuda" if role == "client" else document["server_executor"]):
                raise ArtifactError("native fixture arithmetic role/profile differs from registration")
            if body["expert_shape"] != "target" or body["relative_floor"] != policy["thresholds"]["same_realization"]["relative_floor"]:
                raise ArtifactError("initial admission requires target-shaped fixtures and the frozen relative floor")
            for field, bound in (("host_budget", "host_bytes"), ("device_budget", "device_bytes"), ("pinned_budget", "pinned_bytes")):
                if _integer(body[field]) != amounts[bound]:
                    raise ArtifactError("native owner caps differ from the explicit registered limits")
            _integer(body["fixture_seed"])
        elif amounts["device_bytes"] or amounts["pinned_bytes"]:
            raise ArtifactError("transport fixture must not allocate neural device/pinned workspaces")
        else:
            bootstrap = run["bootstrap_body"]
            from .schemas import validate
            validate("transport-bootstrap", {**bootstrap, "request_digest": "0" * 64})
            if bootstrap["role"] != ("expert" if role == "server" else "client") or any(bootstrap[key] != body[key] for key in ("environment_digest", "build_digest")):
                raise ArtifactError("transport bootstrap differs from its frozen role realization")
    left, right = (document["runs"][name]["request_body"] for name in ("transport_server", "transport_client"))
    for key in ("deadline_ms", "warmup", "repeats", "fixture_seed", "max_frame_bytes", "control_credit", "expert_credit", "row_credit", "bulk_credit"):
        if left[key] != right[key]:
            raise ArtifactError("paired transport requests have different fixture/credit populations")
    floors = {"control_credit": 131328, "expert_credit": 33160, "row_credit": 432, "bulk_credit": 1024}
    if not 4096 <= u64(left["max_frame_bytes"]) <= 16777216 or sum(u64(left[key]) for key in floors) > 67108864 or any(u64(left[key]) < minimum for key, minimum in floors.items()):
        raise ArtifactError("paired transport credits cannot guarantee bounded protected progress")


def validate_registration(document, policy):
    _schema(registration_schemas()["fixture-registration"], document, "fixture registration")
    verify_identity(document)
    _registration_relations(document, policy)
    try:
        registered = datetime.strptime(document["registered_at"], "%Y-%m-%dT%H:%M:%SZ")
        frozen = datetime.strptime(policy["registered_at"], "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ArtifactError("invalid fixture preregistration timestamp") from exc
    if registered < frozen:
        raise ArtifactError("fixture registration predates the acceptance policy")
    return document


def freeze_registration(document, policy, build_info, environments, output):
    _schema(registration_schemas()["fixture-registration-request"], document, "fixture registration input")
    _registration_relations(document, policy)
    result = deepcopy(document)
    for role in ("server", "client"):
        environment = read_json(environments[role], max_bytes=2 << 20)
        from .schemas import validate
        validate("deployment-environment", environment)
        verify_identity(environment, document["realizations"][role]["environment_digest"])
        if environment.get("environment", {}).get("build_digest") != document["realizations"][role]["build_digest"]:
            raise ArtifactError("actual frozen environment does not match fixture role/build")
        planning = environment["planning_request"]
        if (planning["role"], planning["executor"], planning["profile"], planning["logical_model_digest"], planning["operator_contract_digest"]) != (
                "expert" if role == "server" else "client", document["server_executor"] if role == "server" else "cuda",
                document["profile"], document["logical_model_digest"], document["operator_contract_digest"]):
            raise ArtifactError("frozen environment has another role/profile/model/operator realization")
        body = document["runs"]["native_" + role]["request_body"]
        limits = document["runs"]["native_" + role]["hard_limits"]
        if any(_integer(body[name]) > planning["caps"][bound] for name, bound in (("host_budget", "host_bytes"), ("device_budget", "device_bytes"), ("pinned_budget", "pinned_bytes"))) or u64(limits["max_rss_bytes"]) > environment["environment"]["cgroup_bytes"]:
            raise ArtifactError("fixture caps exceed the frozen role's memory realization")
        if body["gpu_uuid"] != environment["environment"]["gpu_uuid"] or environment["network"]["server_executor"] != document["server_executor"]:
            raise ArtifactError("frozen environment changes selected GPU or server executor")
        bootstrap = document["runs"]["transport_" + role]["bootstrap_body"]
        if any(value != environment["network"].get(key) for key, value in bootstrap["network"].items()) or any(bootstrap["tls"][key] != environment["tls"].get(key) for key in ("ca_file", "certificate_file", "private_key_file", "expected_peer_name")):
            raise ArtifactError("transport bootstrap changes the actual frozen network/TLS settings")
        build = read_json(build_info[role], max_bytes=16 << 20)
        verify_identity(build, document["realizations"][role]["build_digest"])
        if build.get("source_lock_digest") != document["source_lock_digest"] or build.get("image_kind") != (document["server_executor"] if role == "server" else "cuda"):
            raise ArtifactError("actual build information differs from registered image/source")
        for prefix, binary in (("native", "ds4-ria-qualify"), ("transport", "ds4ctl")):
            claimed = build.get("binaries", {}).get(binary)
            _schema(SHA, claimed, "locked native executable hash")
            result["runs"][prefix + "_" + role]["binary_sha256"] = claimed
    result.update(kind="fixture_registration", registered_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    result = seal(result)
    validate_registration(result, policy)
    if Path(output).exists():
        raise ArtifactError("fixture registration is immutable; choose a new destination")
    atomic_json(output, result)
    return result


def derive_request(registration, name):
    if name not in RUNS:
        raise ArtifactError("unknown preregistered fixture run")
    result = {**registration["runs"][name]["request_body"], "preregistration_digest": registration["digest"]}
    return seal(result) if name.startswith("transport_") else result


def derive_bootstrap(registration, name):
    if name not in ("transport_server", "transport_client"):
        raise ArtifactError("only registered transport roles have a bootstrap configuration")
    return {**registration["runs"][name]["bootstrap_body"], "request_digest": derive_request(registration, name)["digest"]}


def publish_requests(registration, policy, output):
    validate_registration(registration, policy)
    output = Path(output).absolute()
    if output.exists():
        raise ArtifactError("derived fixture requests are immutable")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".ria-requests-", dir=output.parent))
    try:
        for name in RUNS:
            atomic_json(staging / (name + "-request.json"), derive_request(registration, name))
            if name.startswith("transport_"):
                atomic_json(staging / (name + "-bootstrap.json"), derive_bootstrap(registration, name))
        os_publish(staging, output)
        return registration
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _bound(value, maximum, label, *, positive=False):
    amount = _integer(value)
    if type(amount) is not int or amount < int(positive) or amount > maximum:
        raise ArtifactError(f"fixture {label} violates its preregistered bound")
    return amount


def validate_native(raw, registration, name, policy):
    _schema(NATIVE_SEALED_MEASUREMENTS, raw, "native fixture measurements")
    verify_identity(raw)
    body, limits = registration["runs"][name]["request_body"], registration["runs"][name]["hard_limits"]
    request = derive_request(registration, name)
    if raw["request_digest"] != digest(request):
        raise ArtifactError("native report does not identify the exact derived request")
    for key in (*IDENTITIES, "environment_digest", "build_digest", "profile", "executor", "runner_executor", "role", "gpu_uuid", "warmup", "repeats", "relative_floor"):
        if raw[key] != request[key]:
            raise ArtifactError("native measurements have another request identity")
    if raw["preregistration_digest"] != registration["digest"] or u64(raw["fixture_seed"]) != _integer(request["fixture_seed"]):
        raise ArtifactError("native measurements have another frozen seed/population")
    cases = raw["cases"]
    ids = [item["id"] for item in cases]
    expected = expected_cases(body["role"], body["runner_executor"])
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ArtifactError("native report does not contain exactly every required case")
    thresholds = policy["thresholds"]["same_realization"]
    maxima = {key: 0 for key in ("max_abs_error", "max_relative_error", "max_rms_error")}
    for case in cases:
        key = case["id"]
        expert = key in ("expert_dense_clamp_route", "expert_unit_palette", "expert_resident_dense_clamp_route", "expert_resident_unit_palette")
        transfer = key == "pinned_multichunk_odd_bytes"
        expected_component = "experts" if expert else "transfers" if transfer else "graph_state"
        expected_path = "vram" if key.startswith("expert_resident_") else "host" if expert else "pinned" if transfer else "gpu"
        if (case["component"], case["path"]) != (expected_component, expected_path):
            raise ArtifactError("native case altered its authoritative component or residency path")
        if tuple(case[field] for field in ("rows", "input_features", "intermediate_features", "output_features", "elements")) != expected_dimensions(key, body["expert_shape"], pinned_bytes=u64(case["pinned_bytes"])):
            raise ArtifactError("native case dimensions differ from the immutable target fixture")
        if (case["requires_exact_match"], case["timing_group"], case["warmup"]) != (key in EXACT_CASES, TIMING_GROUPS.get(key, key), expected_warmup(key, body["warmup"])):
            raise ArtifactError("native case altered its exactness/timing/warmup contract")
        exact = case["oracle_digest"] == case["actual_digest"]
        if case["exact_match"] != exact or (case["requires_exact_match"] and not exact) or (exact and any(case[key] for key in maxima)):
            raise ArtifactError("native exact comparison/digests/errors are inconsistent")
        if case["repeat_identical"] is False or case["placement_identical"] is False:
            raise ArtifactError("native repeated or host/VRAM placement results differ")
        if not case["latency_ns"]:
            raise ArtifactError("native case lacks a measured execution duration")
        for value in case["latency_ns"]:
            _bound(value, u64(limits["max_case_latency_ns"]), "case latency", positive=True)
        if len(case["latency_ns"]) != expected_samples(key, body["repeats"]) or (key in REPEATED_CASES and case["repeat_identical"] is not True) or (key not in REPEATED_CASES and case["repeat_identical"] is not None):
            raise ArtifactError("native repetition evidence disagrees with measured samples")
        if case["placement_identical"] != (True if key.startswith("expert_resident_") else None):
            raise ArtifactError("native case altered its required placement parity proof")
        for field, bound in (("owned_host_bytes", "host_bytes"), ("device_bytes", "device_bytes"), ("pinned_bytes", "pinned_bytes"), ("startup_ns", "max_startup_ns")):
            _bound(case[field], u64(limits[bound]), field)
        for field, peak in (("owned_host_bytes", "owned_host_peak_bytes"), ("device_bytes", "device_peak_bytes"), ("pinned_bytes", "pinned_peak_bytes")):
            if u64(case[field]) > u64(raw[peak]):
                raise ArtifactError("native case allocation exceeds the observed process owner peak")
        if any(u64(value) > u64(raw["elapsed_ns"]) for value in (*case["latency_ns"], case["startup_ns"])):
            raise ArtifactError("native case duration exceeds its complete measured run")
        for key in maxima:
            if case[key] > thresholds[key]:
                raise ArtifactError("native numerical errors exceed preregistered same-realization tolerance")
            maxima[key] = max(maxima[key], case[key])
    by_id = {case["id"]: case for case in cases}
    for child, parent in TIMING_GROUPS.items():
        if child in by_id and by_id[child]["latency_ns"] != by_id[parent]["latency_ns"]:
            raise ArtifactError("native shared timing observation differs from its completed invocation")
    for field, bound in (("owned_host_peak_bytes", "host_bytes"), ("device_peak_bytes", "device_bytes"), ("pinned_peak_bytes", "pinned_bytes"),
                         ("process_peak_rss_bytes", "max_rss_bytes"), ("elapsed_ns", "max_elapsed_ns"), ("graph_startup_ns", "max_startup_ns")):
        _bound(raw[field], u64(limits[bound]), field, positive=field in ("process_peak_rss_bytes", "elapsed_ns"))
    if raw["graph_gpu_fixtures"] != ("executed" if body["role"] == "client" else "unexecuted") or raw["transfer_gpu_fixtures"] != ("executed" if body["runner_executor"] == "cuda" else "unexecuted"):
        raise ArtifactError("native GPU component execution scope is inconsistent")
    return maxima


def validate_transport(raw, registration, name, policy):
    from .transport_fixture_schema import TRANSPORT_SEALED_MEASUREMENTS, EXPECTED_CHECKS, CASE_KINDS, CASE_STATUSES
    _schema(TRANSPORT_SEALED_MEASUREMENTS, raw, "transport measurements")
    verify_identity(raw)
    request, limits = derive_request(registration, name), registration["runs"][name]["hard_limits"]
    for key in (*IDENTITIES, "environment_digest", "build_digest", "preregistration_digest"):
        if raw[key] != request[key]:
            raise ArtifactError("transport measurements have another preregistered identity")
    if raw["request_digest"] != request["digest"] or raw["role"] != ("expert" if name.endswith("server") else "client"):
        raise ArtifactError("transport measurements have another exact derived request/role")
    checks = raw["checks"]
    if len(checks) != len(EXPECTED_CHECKS) or {item["id"] for item in checks} != set(EXPECTED_CHECKS) or not all(item["passed"] for item in checks):
        raise ArtifactError("transport report lacks an actual required passing contract check")
    if (raw["warmup_completed"], raw["iterations_completed"]) != (request["warmup"], request["repeats"]):
        raise ArtifactError("paired transport did not complete the registered population")
    expected = [(iteration, kind) for iteration in range(request["repeats"]) for kind in CASE_KINDS]
    actual = [(item["iteration"], item["id"]) for item in raw["cases"]]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ArtifactError("transport cases do not cover each registered measured iteration exactly once")
    for case in raw["cases"]:
        if case["kind"] != CASE_KINDS[case["id"]] or case["status"] != CASE_STATUSES[case["id"]]:
            raise ArtifactError("transport case opcode/status differs from the exact fixture")
        _bound(case["elapsed_ns"], u64(limits["max_case_latency_ns"]), "transport case latency", positive=True)
        for key in ("request_bytes", "reply_bytes"):
            _bound(case[key], u64(request["max_frame_bytes"]) + 64, "transport frame bytes", positive=True)
            if u64(case[key]) < 64:
                raise ArtifactError("transport measured frame omits its authenticated wire header")
    for field, bound in (("owned_buffers_peak_bytes", "host_bytes"), ("max_rss_bytes", "max_rss_bytes"), ("elapsed_ns", "max_elapsed_ns"), ("startup_ns", "max_startup_ns")):
        _bound(raw[field], u64(limits[bound]), field, positive=field in ("max_rss_bytes", "elapsed_ns"))
    if raw["peer_certificate_digest"] != registration["runs"][name]["bootstrap_body"]["tls"]["authorized_peer_sha256"]:
        raise ArtifactError("transport measurements authenticated a different exact peer certificate")
    frame_timeout = registration["runs"][name]["bootstrap_body"]["network"]["frame_io_timeout_ms"] * 1000000
    if not frame_timeout <= raw["timeout_elapsed_ns"] <= u64(limits["max_elapsed_ns"]):
        raise ArtifactError("transport partial-header deadline was not actually observed within the registered bound")
    if raw["elapsed_ns"] < raw["startup_ns"] + raw["timeout_elapsed_ns"] or any(case["elapsed_ns"] > raw["elapsed_ns"] for case in raw["cases"]):
        raise ArtifactError("transport durations disagree with its complete measured run")
    if _integer(raw["measured_request_bytes"]) != sum(u64(case["request_bytes"]) for case in raw["cases"]) or _integer(raw["measured_response_bytes"]) != sum(u64(case["reply_bytes"]) for case in raw["cases"]):
        raise ArtifactError("transport measured byte totals do not match its complete case population")
    return {}


def _supervision(document, raw, registration, name):
    _schema(SUPERVISION, document, "native process supervision")
    verify_identity(document)
    validate_preflight(document["preflight"], registration, name)
    request, run = derive_request(registration, name), registration["runs"][name]
    if (document["run"], document["registration_digest"], document["request_digest"], document["raw_digest"], document["binary_sha256"], document["build_digest"]) != (
            name, registration["digest"], digest(request), raw["digest"], run["binary_sha256"], run["request_body"]["build_digest"]):
        raise ArtifactError("supervisor observation differs from exact request/raw/build")
    _bound(document["peak_rss_bytes"], u64(run["hard_limits"]["max_rss_bytes"]), "completed process RSS", positive=True)
    _bound(document["wall_ns"], u64(run["hard_limits"]["max_elapsed_ns"]), "supervised elapsed time", positive=True)
    _bound(document["stdout_bytes"], 2 << 20, "native stdout")
    _bound(document["stderr_bytes"], 65536, "native stderr")
    reported_rss = raw["process_peak_rss_bytes"] if name.startswith("native_") else raw["max_rss_bytes"]
    if u64(document["peak_rss_bytes"]) < _integer(reported_rss):
        raise ArtifactError("native self RSS exceeds its complete-lifetime wait4 observation")
    if u64(document["wall_ns"]) < _integer(raw["elapsed_ns"]):
        raise ArtifactError("native elapsed time exceeds the complete supervised process lifetime")


def execute_run(registration, name, policy, executable, build_info, output, *, environment, probe_config, probe, probe_evidence,
                transport_config=None, runner=run_bounded, observer=observe_runtime):
    validate_registration(registration, policy)
    executable = Path(executable).absolute()
    if executable.is_symlink() or not stat.S_ISREG(executable.stat().st_mode) or not os.access(executable, os.X_OK):
        raise ArtifactError("native executable must be an executable regular file without a symlink")
    executable = executable.resolve(strict=True)
    build = read_json(build_info, max_bytes=16 << 20)
    run, request = registration["runs"][name], derive_request(registration, name)
    verify_identity(build, request["build_digest"])
    binary = "ds4-ria-qualify" if name.startswith("native_") else "ds4ctl"
    if not stat.S_ISREG(executable.stat().st_mode) or hash_file(executable) != run["binary_sha256"] or build.get("binaries", {}).get(binary) != run["binary_sha256"]:
        raise ArtifactError("native executable differs from the actual locked build")
    preflight = {key: read_json(value, max_bytes=2 << 20) for key, value in (
        ("environment", environment), ("probe_config", probe_config), ("probe", probe), ("probe_evidence", probe_evidence))}
    preflight["observation"] = observer()
    validate_preflight(preflight, registration, name)
    output = Path(output).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ArtifactError("fixture run outputs are immutable")
    staging = Path(tempfile.mkdtemp(prefix=".ria-fixture-", dir=output.parent))
    try:
        atomic_json(staging / "request.json", request)
        args = [str(executable), str(staging / "request.json")]
        if name.startswith("transport_"):
            if transport_config is None:
                raise ArtifactError("paired transport execution requires its explicit TLS bootstrap configuration")
            from .schemas import validate
            config = read_json(transport_config)
            validate("transport-bootstrap", config)
            if config != derive_bootstrap(registration, name):
                raise ArtifactError("transport bootstrap differs from its exact frozen derived request")
            args = [str(executable), "qualify-transport", "--config", str(Path(transport_config).resolve(strict=True)), "--request", str(staging / "request.json")]
        start = time.monotonic_ns()
        result = runner(args, timeout=request["deadline_ms"] / 1000, env={"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
                        max_stdout=2 << 20, max_stderr=65536, max_rss_bytes=u64(run["hard_limits"]["max_rss_bytes"]), cwd=staging)
        wall = time.monotonic_ns() - start
        if hash_file(executable) != run["binary_sha256"]:
            raise ArtifactError("native executable changed while the registered run was executing")
        atomic_bytes(staging / "native-stdout.json", result.stdout, mode=0o600)
        atomic_bytes(staging / "native-stderr.log", result.stderr, mode=0o600)
        if result.returncode:
            raise ArtifactError(f"native fixture exited {result.returncode}: {result.stderr[:4096].decode('utf-8', errors='replace')}")
        raw = seal(loads(result.stdout, max_bytes=2 << 20, max_nodes=100000, max_depth=32))
        (validate_native if name.startswith("native_") else validate_transport)(raw, registration, name, policy)
        supervision = seal({"schema_revision": 1, "kind": "fixture_supervision", "run": name,
            "registration_digest": registration["digest"], "request_digest": digest(request), "raw_digest": raw["digest"],
            "binary_sha256": run["binary_sha256"], "build_digest": request["build_digest"], "wall_ns": str(wall),
            "peak_rss_bytes": str(result.peak_rss_bytes), "stdout_bytes": str(len(result.stdout)), "stderr_bytes": str(len(result.stderr)),
            "rss_scope": result.rss_scope, "exit_code": 0, "preflight": preflight})
        _supervision(supervision, raw, registration, name)
        atomic_json(staging / "measurements.json", raw)
        atomic_json(staging / "supervision.json", supervision)
        os_publish(staging, output)
        return {"run": name, "raw_digest": raw["digest"], "supervision_digest": supervision["digest"]}
    except (ArtifactError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        # Preserve bounded diagnostics under an explicitly failed owned name.
        failed = output.with_name(output.name + ".failed")
        atomic_json(staging / "failure.json", {"schema_revision": 1, "run": name, "qualified": False,
            "registration_digest": registration["digest"], "error": str(exc)[:4096]})
        if not failed.exists():
            os_publish(staging, failed)
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def os_publish(staging, output):
    staging.rename(output)
    sync_directory(output.parent)


def _source_population(registration, sources, policy):
    if set(sources) != set(RUNS):
        raise ArtifactError("initial proof production requires exactly four registered actual runs")
    for name in RUNS:
        raw, supervision = sources[name]
        (validate_native if name.startswith("native_") else validate_transport)(raw, registration, name, policy)
        _supervision(supervision, raw, registration, name)
    server, client = sources["transport_server"][0], sources["transport_client"][0]
    if server["measured_request_bytes"] != client["measured_request_bytes"] or server["measured_response_bytes"] != client["measured_response_bytes"]:
        raise ArtifactError("paired transport measured different byte populations")
    left = {(item["iteration"], item["id"]): item for item in server["cases"]}
    for item in client["cases"]:
        other = left[(item["iteration"], item["id"])]
        if any(item[key] != other[key] for key in ("request_bytes", "reply_bytes", "request_sha256", "response_sha256", "status")):
            raise ArtifactError("paired authenticated transport responses differ from their exact payload association")


def _component(registration, sources, policy, name, role, references, registration_path):
    selected = "server" if role == "expert" else "client"
    realization = registration["realizations"][selected]
    metrics, checks = {}, []
    for run in RUNS:
        raw, observed = sources[run]
        metrics[run + ".peak_rss_bytes"] = u64(observed["peak_rss_bytes"])
        metrics[run + ".wall_ns"] = u64(observed["wall_ns"])
        checks.append({"id": run + ":authenticated_population_and_bounds", "passed": True})
        if run.startswith("native_"):
            cases = [case for case in raw["cases"] if case["component"] == name]
            if name == "experts" and run != "native_server":
                cases = []
            for metric in ("max_abs_error", "max_relative_error", "max_rms_error"):
                if cases:
                    metrics[run + "." + metric] = max(case[metric] for case in cases)
    return seal({"schema_revision": 1, "kind": "qualification_component", "qualification_scope": "initial_fixture",
        **realization, "policy_digest": policy["digest"], "profile": registration["profile"],
        "executor": registration["server_executor"] if role == "expert" else "cuda",
        "operator_contract_digest": registration["operator_contract_digest"], "component": name,
        "registration_digest": registration["digest"], "registration_path": registration_path, "raw_evidence": references,
        "fixture_digest": digest({"registration": registration["digest"], "component": name, "sources": references}),
        "checks": checks, "measurements": metrics, "qualified": all(check["passed"] for check in checks)})


def produce_components(registration, policy, files, output, *, role):
    validate_registration(registration, policy)
    if role not in ("client", "expert") or set(files) != set(RUNS):
        raise ArtifactError("component production requires an explicit admission role and all four run directories")
    sources = {name: (read_json(Path(files[name]) / "measurements.json", max_bytes=2 << 20), read_json(Path(files[name]) / "supervision.json")) for name in RUNS}
    _source_population(registration, sources, policy)
    output = Path(output).absolute()
    if output.exists():
        raise ArtifactError("initial proof packages are immutable")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".ria-components-", dir=output.parent))
    try:
        atomic_json(staging / "registration.json", registration)
        references = []
        for name, (raw, supervision) in sources.items():
            for suffix, document in (("measurements", raw), ("supervision", supervision)):
                relative = name + "-" + suffix + ".json"
                atomic_json(staging / relative, document)
                references.append({"path": relative, "digest": document["digest"]})
        from .qualification_schema import COMPONENT_NAMES
        refs = []
        for name in COMPONENT_NAMES:
            proof = _component(registration, sources, policy, name, role, references, "registration.json")
            atomic_json(staging / (name + ".json"), proof)
            refs.append({"name": name, "evidence_path": name + ".json", "evidence_digest": proof["digest"], "passed": proof["qualified"]})
        realization = registration["realizations"]["server" if role == "expert" else "client"]
        aggregate = seal({"schema_revision": 1, "kind": "calibration_evidence", "qualification_scope": "initial_fixture", **realization,
            "policy_digest": policy["digest"], "profile": registration["profile"], "executor": registration["server_executor"] if role == "expert" else "cuda",
            "operator_contract_digest": registration["operator_contract_digest"], "components": refs, "comparisons": [],
            "release_matrix_digest": None, "release_matrix_path": None, "soak_seconds": 0, "passed": all(ref["passed"] for ref in refs)})
        from .qualification import calibration_report
        report = calibration_report(aggregate, staging, policy)
        atomic_json(staging / "calibration-evidence.json", aggregate)
        atomic_json(staging / "calibration.json", report)
        atomic_json(staging / "policy.json", policy)
        os_publish(staging, output)
        return aggregate
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def validate_component_sources(proof, evidence_dir, policy):
    registration = read_json(within(evidence_dir, proof["registration_path"]), max_bytes=2 << 20)
    verify_identity(registration, proof["registration_digest"])
    validate_registration(registration, policy)
    references = proof["raw_evidence"]
    if len(references) != 8 or len({item["path"] for item in references}) != 8:
        raise ArtifactError("component needs exactly four raw measurements and four supervisor observations")
    sources = {name: [None, None] for name in RUNS}
    for reference in references:
        raw = read_json(within(evidence_dir, reference["path"]), max_bytes=2 << 20)
        verify_identity(raw, reference["digest"])
        if raw.get("kind") == "fixture_supervision":
            name, slot = raw.get("run"), 1
        else:
            native = raw.get("kind") == "native_fixture_measurements"
            if raw.get("kind") not in ("native_fixture_measurements", "native_transport_measurements"):
                raise ArtifactError("unknown initial proof source kind")
            name = ("native_" if native else "transport_") + ("server" if raw.get("role") in ("server", "expert") else "client")
            slot = 0
        if name not in sources or sources[name][slot] is not None:
            raise ArtifactError("duplicate initial proof source population")
        sources[name][slot] = raw
    if any(any(value is None for value in pair) for pair in sources.values()):
        raise ArtifactError("missing actual initial proof source")
    _source_population(registration, sources, policy)
    roles = [role for role in ("expert", "client") if proof["environment_digest"] == registration["realizations"]["server" if role == "expert" else "client"]["environment_digest"] and
             proof["build_digest"] == registration["realizations"]["server" if role == "expert" else "client"]["build_digest"] and
             proof["executor"] == (registration["server_executor"] if role == "expert" else "cuda")]
    expected = [_component(registration, sources, policy, proof["component"], role, references, proof["registration_path"]) for role in roles]
    if not any(proof == value for value in expected):
        raise ArtifactError("component checks/metrics/identity were not derived from its complete actual proof population")
    return [{"path": proof["registration_path"], "digest": registration["digest"]}, *references]
