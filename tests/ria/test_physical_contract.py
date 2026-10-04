"""Synthetic contract fixtures only; these tests execute no physical workload."""

import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from ria.identity import ArtifactError, canonical, seal
from ria.physical_contract import validate_physical_contract
import ria.physical_contract as physical


def fixture_bundle(tmp_path, domain="number"):
    thresholds = {"max_abs_error": 0, "max_relative_error": 0, "max_rms_error": 0,
                  "max_loss_delta": 0, "relative_floor": 1e-12}
    policy = seal({"schema_revision": 1, "logical_model_digest": "1" * 64,
        "source_lock_digest": "2" * 64,
        "thresholds": {"same_realization": thresholds, "native_source": thresholds},
        "minimum_soak_seconds": 3600, "ordered_objectives": ["synthetic contract test"],
        "registered_at": "2026-01-01T00:00:00Z"})
    common = {"policy_digest": policy["digest"], "logical_model_digest": "1" * 64,
        "source_lock_digest": "2" * 64, "environment_digest": "3" * 64,
        "build_digest": "4" * 64, "operator_contract_digest": "5" * 64,
        "runtime_config_digest": "6" * 64, "profile": "nvfp4", "server_executor": "cpu"}
    cell = {"profile": "nvfp4", "server_executor": "cpu",
        "client_residency": "resident_reference", "placement": "all_remote",
        "phase": "prefill", "numa_policy": "sharded"}
    cells, gates = [cell], ["G01"]
    runtime = {"schema_revision": 1, "kind": "physical_runtime_identity", **common}
    plan = {"schema_revision": 1, "kind": "physical_contract_plan", **common,
        "qualification_scope": "final_release", "registered_at": "2026-01-01T00:00:01Z",
        "binary_sha256": "7" * 64, "max_elapsed_ns": "1000",
        "runtime_reference": {"path": "runtime.json", "digest": "0" * 64},
        "required_cells": copy.deepcopy(cells), "required_gates": gates[:],
        "checks": [{"id": "synthetic_latency", "unit": "ns", "domain": domain,
            "minimum": "1" if domain == "u64" else 1,
            "maximum": "10" if domain == "u64" else 10,
            "cells": copy.deepcopy(cells), "gates": gates[:]}]}
    raw = {"schema_revision": 1, "kind": "physical_contract_measurements", **common,
        "classification": "physical", "plan_digest": "0" * 64,
        "execution": {"run_id": "8" * 64, "started_at": "2026-01-01T00:00:02Z",
            "clock": "CLOCK_MONOTONIC", "elapsed_ns": "50", "pid": 123,
            "complete_child_lifetime": True, "child_exit_code": 0, "child_signal": None,
            "binary_sha256": "7" * 64},
        "measurements": [{"check_id": "synthetic_latency", "measurement_id": "synthetic_run_latency",
            "unit": "ns", "value": "5" if domain == "u64" else 5,
            "cells": copy.deepcopy(cells), "gates": gates[:]}]}
    proof = {"schema_revision": 1, "kind": "physical_contract_evidence", **common,
        "classification": "physical", "qualification_scope": "final_release",
        "plan_reference": {"path": "plan.json", "digest": "0" * 64},
        "raw_evidence": [{"path": "raw.json", "digest": "0" * 64}],
        "covered_cells": copy.deepcopy(cells), "covered_gates": gates[:],
        "checks": [{"id": "synthetic_latency", "measurement_id": "synthetic_run_latency",
                    "raw_digest": "0" * 64, "passed": True}], "passed": True}
    bundle = {"policy": policy, "runtime": runtime, "plan": plan, "raw": raw, "proof": proof}
    publish(tmp_path, bundle)
    return bundle


def publish(tmp_path, bundle):
    """Seal synthetic fixtures, preserving deliberately changed identities/values."""
    bundle["runtime"] = seal(bundle["runtime"])
    bundle["plan"]["runtime_reference"]["digest"] = bundle["runtime"]["digest"]
    bundle["plan"] = seal(bundle["plan"])
    bundle["raw"]["plan_digest"] = bundle["plan"]["digest"]
    bundle["raw"] = seal(bundle["raw"])
    proof = bundle["proof"]
    proof["plan_reference"]["digest"] = bundle["plan"]["digest"]
    if proof["raw_evidence"]:
        proof["raw_evidence"][0]["digest"] = bundle["raw"]["digest"]
    for check in proof["checks"]:
        check["raw_digest"] = bundle["raw"]["digest"]
    bundle["proof"] = seal(proof)
    for name in ("runtime", "plan", "raw"):
        (tmp_path / f"{name}.json").write_bytes(canonical(bundle[name]))


def validate(bundle, tmp_path):
    return validate_physical_contract(bundle["proof"], tmp_path, bundle["policy"])


@pytest.mark.parametrize("domain", ["number", "u64"])
def test_complete_measurement_bound_report_returns_authenticated_refs(tmp_path, domain):
    bundle = fixture_bundle(tmp_path, domain)
    assert validate(bundle, tmp_path) == [
        {"path": "plan.json", "digest": bundle["plan"]["digest"]},
        {"path": "raw.json", "digest": bundle["raw"]["digest"]},
        {"path": "runtime.json", "digest": bundle["runtime"]["digest"]}]


def test_u64_domain_above_safe_integer_is_compared_without_float_conversion(tmp_path):
    bundle = fixture_bundle(tmp_path, "u64")
    bound = "9007199254740993"
    bundle["plan"]["checks"][0].update(minimum=bound, maximum=bound)
    bundle["raw"]["measurements"][0]["value"] = bound
    publish(tmp_path, bundle)
    assert validate(bundle, tmp_path)
    bundle["raw"]["measurements"][0]["value"] = "9007199254740992"
    publish(tmp_path, bundle)
    with pytest.raises(ArtifactError, match="pass claims"):
        validate(bundle, tmp_path)
    bundle["proof"]["checks"][0]["passed"] = False
    bundle["proof"]["passed"] = False
    bundle["proof"] = seal(bundle["proof"])
    assert validate(bundle, tmp_path)  # Valid measured failure is never a pass.


@pytest.mark.parametrize("value", [1, 10, 1.5])
def test_number_bounds_are_inclusive(tmp_path, value):
    bundle = fixture_bundle(tmp_path)
    bundle["raw"]["measurements"][0]["value"] = value
    publish(tmp_path, bundle)
    assert validate(bundle, tmp_path)


def test_one_sided_bound_and_gate_only_report(tmp_path):
    bundle = fixture_bundle(tmp_path)
    bundle["plan"]["checks"][0]["minimum"] = None
    for document, field in [("plan", "required_cells"), ("proof", "covered_cells")]:
        bundle[document][field] = []
    bundle["plan"]["checks"][0]["cells"] = []
    bundle["raw"]["measurements"][0]["cells"] = []
    publish(tmp_path, bundle)
    assert validate(bundle, tmp_path)


@pytest.mark.parametrize("document", ["runtime", "plan", "raw", "proof"])
@pytest.mark.parametrize("field", physical.IDENTITY_FIELDS)
def test_identity_changes_never_cross_realization_boundaries(tmp_path, document, field):
    bundle = fixture_bundle(tmp_path)
    changed = "bf16" if field == "profile" else "cuda" if field == "server_executor" else "9" * 64
    bundle[document][field] = changed
    publish(tmp_path, bundle)
    with pytest.raises(ArtifactError):
        validate(bundle, tmp_path)


@pytest.mark.parametrize("document", ["runtime", "plan", "raw", "proof"])
def test_unknown_schema_and_undeclared_fields_are_rejected(tmp_path, document):
    bundle = fixture_bundle(tmp_path)
    bundle[document]["kind"] = "unknown_physical_report"
    publish(tmp_path, bundle)
    with pytest.raises(ArtifactError, match="strict schema"):
        validate(bundle, tmp_path)
    bundle = fixture_bundle(tmp_path)
    bundle[document]["extra"] = "never printed externally supplied content"
    publish(tmp_path, bundle)
    with pytest.raises(ArtifactError) as error:
        validate(bundle, tmp_path)
    assert "never printed" not in str(error.value)


@pytest.mark.parametrize("document", ["runtime", "plan", "raw"])
def test_sealed_reference_hash_and_document_identity_are_required(tmp_path, document):
    bundle = fixture_bundle(tmp_path)
    changed = copy.deepcopy(bundle[document])
    changed["operator_contract_digest"] = "f" * 64
    (tmp_path / f"{document}.json").write_bytes(canonical(changed))
    with pytest.raises(ArtifactError, match="identity mismatch"):
        validate(bundle, tmp_path)
    changed = seal(changed)
    (tmp_path / f"{document}.json").write_bytes(canonical(changed))
    with pytest.raises(ArtifactError, match="identity mismatch"):
        validate(bundle, tmp_path)


@pytest.mark.parametrize("case", ["bare_boolean", "empty_checks", "empty_measurements", "empty_refs",
    "missing_measurement", "unplanned_measurement", "duplicate_check", "duplicate_measurement",
    "no_coverage", "unassigned_gate", "unknown_gate", "unplanned_gate", "unplanned_cell",
    "mismatched_profile_cell", "mismatched_executor_cell", "wrong_unit", "wrong_domain",
    "no_bounds", "reversed_bounds", "bool_value", "overflow_u64", "noncanonical_u64",
    "false_pass", "wrong_measurement_reference", "wrong_raw_reference", "wrong_binary",
    "incomplete_child", "nonzero_exit", "signal_exit", "zero_elapsed", "over_deadline",
    "before_plan", "before_policy", "invalid_utc", "wrong_clock", "missing_policy_registration"])
def test_malformed_physical_reports_fail_closed(tmp_path, case):
    bundle = fixture_bundle(tmp_path)
    plan, raw, proof = bundle["plan"], bundle["raw"], bundle["proof"]
    check, measured = plan["checks"][0], raw["measurements"][0]
    if case == "bare_boolean":
        bundle["proof"] = seal({"classification": "physical", "passed": True,
                               "covered_cells": proof["covered_cells"], "covered_gates": ["G01"]})
        with pytest.raises(ArtifactError):
            validate(bundle, tmp_path)
        return
    if case == "empty_checks":
        plan["checks"] = []
    elif case == "empty_measurements":
        raw["measurements"] = []
    elif case == "empty_refs":
        proof["raw_evidence"] = []
    elif case == "missing_measurement":
        extra = copy.deepcopy(check)
        extra["id"] = "synthetic_missing_check"
        plan["checks"].append(extra)
    elif case == "unplanned_measurement":
        measured["check_id"] = "synthetic_unplanned_check"
    elif case == "duplicate_check":
        extra = copy.deepcopy(check)
        extra["maximum"] = 9
        plan["checks"].append(extra)
    elif case == "duplicate_measurement":
        extra = copy.deepcopy(measured)
        extra["value"] = 6
        raw["measurements"].append(extra)
    elif case == "no_coverage":
        plan.update(required_cells=[], required_gates=[])
        proof.update(covered_cells=[], covered_gates=[])
        check.update(cells=[], gates=[])
    elif case == "unassigned_gate":
        plan["required_gates"].append("G02")
        proof["covered_gates"].append("G02")
    elif case == "unknown_gate":
        proof["covered_gates"].append("G99")
    elif case == "unplanned_gate":
        measured["gates"].append("G02")
    elif case == "unplanned_cell":
        measured["cells"][0]["placement"] = "vram_hit"
    elif case == "mismatched_profile_cell":
        proof["covered_cells"][0]["profile"] = "fp8"
    elif case == "mismatched_executor_cell":
        proof["covered_cells"][0]["server_executor"] = "cuda"
    elif case == "wrong_unit":
        measured["unit"] = "ms"
    elif case == "wrong_domain":
        measured["value"] = "5"
    elif case == "no_bounds":
        check.update(minimum=None, maximum=None)
    elif case == "reversed_bounds":
        check.update(minimum=11, maximum=10)
    elif case == "bool_value":
        measured["value"] = True
    elif case == "overflow_u64":
        check.update(domain="u64", minimum="0", maximum="18446744073709551616")
        measured["value"] = "5"
    elif case == "noncanonical_u64":
        check.update(domain="u64", minimum="0", maximum="10")
        measured["value"] = "05"
    elif case == "false_pass":
        measured["value"] = 11
    elif case == "wrong_measurement_reference":
        proof["checks"][0]["measurement_id"] = "synthetic_wrong_id"
    elif case == "wrong_raw_reference":
        proof["checks"][0]["raw_digest"] = "f" * 64
    elif case == "wrong_binary":
        raw["execution"]["binary_sha256"] = "f" * 64
    elif case == "incomplete_child":
        raw["execution"]["complete_child_lifetime"] = False
    elif case == "nonzero_exit":
        raw["execution"]["child_exit_code"] = 1
    elif case == "signal_exit":
        raw["execution"]["child_signal"] = 9
    elif case == "zero_elapsed":
        raw["execution"]["elapsed_ns"] = "0"
    elif case == "over_deadline":
        raw["execution"]["elapsed_ns"] = "1001"
    elif case == "before_plan":
        raw["execution"]["started_at"] = "2026-01-01T00:00:00Z"
    elif case == "before_policy":
        plan["registered_at"] = "2025-12-31T23:59:59Z"
    elif case == "invalid_utc":
        plan["registered_at"] = "2026-02-30T00:00:00Z"
    elif case == "wrong_clock":
        raw["execution"]["clock"] = "CLOCK_REALTIME"
    elif case == "missing_policy_registration":
        del bundle["policy"]["registered_at"]
        bundle["policy"] = seal(bundle["policy"])
        for document in ("runtime", "plan", "raw", "proof"):
            bundle[document]["policy_digest"] = bundle["policy"]["digest"]
    publish(tmp_path, bundle)
    if case == "empty_refs":
        bundle["proof"] = seal({**bundle["proof"], "raw_evidence": []})
    elif case == "wrong_raw_reference":
        bundle["proof"]["checks"][0]["raw_digest"] = "f" * 64
        bundle["proof"] = seal(bundle["proof"])
    with pytest.raises(ArtifactError):
        validate(bundle, tmp_path)


@pytest.mark.parametrize("path", ["../raw.json", "/raw.json", "./raw.json", "sub/../raw.json",
    "sub//raw.json", "raw.json/", "missing.json"])
def test_traversal_noncanonical_or_missing_references_are_rejected(tmp_path, path):
    bundle = fixture_bundle(tmp_path)
    bundle["proof"]["raw_evidence"][0]["path"] = path
    bundle["proof"] = seal(bundle["proof"])
    with pytest.raises(ArtifactError):
        validate(bundle, tmp_path)


@pytest.mark.parametrize("directory_link", [False, True])
def test_reference_symlinks_are_not_followed(tmp_path, directory_link):
    bundle = fixture_bundle(tmp_path)
    if directory_link:
        real = tmp_path / "real"
        real.mkdir()
        (real / "raw.json").write_bytes((tmp_path / "raw.json").read_bytes())
        (tmp_path / "alias").symlink_to(real, target_is_directory=True)
        path = "alias/raw.json"
    else:
        (tmp_path / "alias.json").symlink_to(tmp_path / "raw.json")
        path = "alias.json"
    bundle["proof"]["raw_evidence"][0]["path"] = path
    bundle["proof"] = seal(bundle["proof"])
    with pytest.raises(ArtifactError, match="read safely"):
        validate(bundle, tmp_path)


def test_raw_plan_digest_and_duplicate_aliases_are_rejected(tmp_path):
    bundle = fixture_bundle(tmp_path)
    bundle["raw"]["plan_digest"] = "f" * 64
    bundle["raw"] = seal(bundle["raw"])
    (tmp_path / "raw.json").write_bytes(canonical(bundle["raw"]))
    bundle["proof"]["raw_evidence"][0]["digest"] = bundle["raw"]["digest"]
    bundle["proof"]["checks"][0]["raw_digest"] = bundle["raw"]["digest"]
    bundle["proof"] = seal(bundle["proof"])
    with pytest.raises(ArtifactError, match="preregistered plan"):
        validate(bundle, tmp_path)
    bundle = fixture_bundle(tmp_path)
    (tmp_path / "alias.json").write_bytes((tmp_path / "raw.json").read_bytes())
    bundle["proof"]["raw_evidence"].append({"path": "alias.json", "digest": bundle["raw"]["digest"]})
    bundle["proof"] = seal(bundle["proof"])
    with pytest.raises(ArtifactError, match="distinct completed runs"):
        validate(bundle, tmp_path)


def test_duplicate_json_and_nonfinite_measurements_are_rejected(tmp_path):
    bundle = fixture_bundle(tmp_path)
    raw = (tmp_path / "raw.json").read_text()
    (tmp_path / "raw.json").write_text(raw.replace('"value":5', '"value":NaN'))
    with pytest.raises(ArtifactError, match="nonfinite JSON"):
        validate(bundle, tmp_path)
    (tmp_path / "raw.json").write_text('{"value":1,"value":2}')
    with pytest.raises(ArtifactError, match="duplicate JSON"):
        validate(bundle, tmp_path)
    assert json.loads(raw)["measurements"][0]["value"] == 5


@pytest.mark.parametrize("bound,value", [("MAX_DOCUMENT_BYTES", 128), ("MAX_TOTAL_BYTES", 1024),
    ("MAX_REFERENCES", 2), ("MAX_DOCUMENT_NODES", 10)])
def test_parser_resource_bounds_are_enforced(tmp_path, monkeypatch, bound, value):
    bundle = fixture_bundle(tmp_path)
    monkeypatch.setattr(physical, bound, value)
    with pytest.raises(ArtifactError):
        validate(bundle, tmp_path)


def test_more_than_4096_raw_references_is_rejected_before_io(tmp_path):
    bundle = fixture_bundle(tmp_path)
    bundle["proof"]["raw_evidence"] = [{"path": f"raw{index}.json", "digest": "a" * 64}
                                       for index in range(4097)]
    bundle["proof"] = seal(bundle["proof"])
    with pytest.raises(ArtifactError, match="strict schema"):
        validate(bundle, tmp_path)
