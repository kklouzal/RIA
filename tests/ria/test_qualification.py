"""Offline malformed-artifact and independent shifted-label comparison fixtures."""

import hashlib
import math
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from ria.identity import ArtifactError, atomic_json, canonical, seal
from ria.qualification import compare_logits, freeze_policy, policy_validate, release_matrix, validate_calibration_evidence
from fixture_measurements import registration, write_component_fixture


def policy(tmp_path):
    exact = {"max_abs_error": 0, "max_relative_error": 0, "max_rms_error": 0,
             "max_loss_delta": 0, "relative_floor": 1e-12}
    return freeze_policy({"schema_revision": 1, "logical_model_digest": "1" * 64,
        "source_lock_digest": "2" * 64,
        "thresholds": {"same_realization": exact, "native_source": exact},
        "minimum_soak_seconds": 3600, "ordered_objectives": ["p99 inter-token latency"]},
        tmp_path / "policy.json")


def logits(path, document, *, operator="3" * 64, changed=False):
    count = len(document["tokens"])
    header = b"RIALOG1\0" + struct.pack("<IIQQ", 1, 129280, count,
        sum(document.get("label_mask", [True] * (count - 1))))
    header += bytes.fromhex("1" * 64 + operator) + hashlib.sha256(canonical(document)).digest()
    rows = np.zeros((count, 129280), dtype="<f4")
    for index in range(count - 1):
        rows[index, document["tokens"][index + 1]] = 2
    if changed:
        rows[0, 0] = 1
    path.write_bytes(header + rows.tobytes())


def test_exact_parity_and_independent_shifted_loss(tmp_path):
    p = policy(tmp_path)
    document = {"schema_revision": 1, "tokens": [1, 2, 3], "label_mask": [True, False]}
    left, right = tmp_path / "left", tmp_path / "right"
    logits(left, document)
    logits(right, document)
    result = compare_logits(left, right, document, p, "same_realization")
    assert result["passed"] and result["labels"] == 1
    expected_loss = math.log(129279 + math.exp(2)) - 2
    assert result["reference_nll"] == pytest.approx(expected_loss, abs=1e-12)
    assert result["metrics"] == {"max_abs_error": 0, "max_relative_error": 0,
        "max_rms_error": 0, "max_loss_delta": 0}
    logits(right, document, changed=True)
    assert not compare_logits(left, right, document, p, "same_realization")["passed"]


@pytest.mark.parametrize("offset", [0, 1e20, -1e20, float(np.finfo(np.float32).max), -float(np.finfo(np.float32).max)])
def test_shifted_loss_preserves_large_common_offsets(tmp_path, offset):
    p = policy(tmp_path)
    document = {"schema_revision": 1, "tokens": [1, 2]}
    left, right = tmp_path / "left", tmp_path / "right"
    logits(left, document)
    uniform = np.full((2, 129280), offset, dtype="<f4").tobytes()
    left.write_bytes(left.read_bytes()[:128] + uniform)
    right.write_bytes(left.read_bytes())
    result = compare_logits(left, right, document, p, "same_realization")
    assert result["passed"]
    assert result["reference_nll"] == pytest.approx(math.log(129280), abs=1e-12)
    assert result["candidate_nll"] == result["reference_nll"]


def test_two_fidelity_axes_do_not_conflate_operator_identity(tmp_path):
    p = policy(tmp_path)
    document = {"schema_revision": 1, "tokens": [1, 2]}
    left, right = tmp_path / "left", tmp_path / "right"
    logits(left, document)
    logits(right, document, operator="4" * 64)
    with pytest.raises(ArtifactError, match="identical operator"):
        compare_logits(left, right, document, p, "same_realization")
    assert compare_logits(left, right, document, p, "native_source")["passed"]


@pytest.mark.parametrize("malformation", ["truncated", "extra", "nonfinite", "wrong_input", "wrong_model", "wrong_labels"])
def test_invalid_logits_fail_closed(tmp_path, malformation):
    p = policy(tmp_path)
    document = {"schema_revision": 1, "tokens": [1, 2]}
    left, right = tmp_path / "left", tmp_path / "right"
    logits(left, document)
    logits(right, document)
    raw = bytearray(right.read_bytes())
    if malformation == "truncated":
        del raw[-1:]
    elif malformation == "extra":
        raw.append(0)
    elif malformation == "nonfinite":
        struct.pack_into("<f", raw, 128, float("nan"))
    elif malformation == "wrong_input":
        raw[96] ^= 1
    elif malformation == "wrong_model":
        raw[32] ^= 1
    else:
        struct.pack_into("<Q", raw, 24, 0)
    right.write_bytes(raw)
    with pytest.raises(ArtifactError):
        compare_logits(left, right, document, p, "same_realization")


def test_policy_and_matrix_never_imply_hardware_pass(tmp_path):
    p = policy(tmp_path)
    assert policy_validate(p) == p
    with pytest.raises(ArtifactError):
        freeze_policy(p, tmp_path / "second.json")
    with pytest.raises(ArtifactError):
        policy_validate(seal({**p, "minimum_soak_seconds": 3599}))
    matrix = release_matrix()
    assert len(matrix["cells"]) == 540 and len(matrix["gates"]) == 28
    assert not matrix["hardware_qualified"]
    assert all(cell["status"] == "unexecuted" and cell["evidence"] == [] for cell in matrix["cells"])


def calibration_fixture(tmp_path):
    """Sealed synthetic contracts for refusal tests, never physical evidence."""
    p = policy(tmp_path)
    reg = registration(p)
    common = {**reg["realizations"]["server"],
        "policy_digest": p["digest"], "profile": "bf16", "executor": "cpu",
        "operator_contract_digest": "3" * 64, "qualification_scope": "initial_fixture"}
    refs = write_component_fixture(tmp_path, p, reg)
    document = seal({"schema_revision": 1, "kind": "calibration_evidence", **common,
        "components": refs, "comparisons": [], "release_matrix_digest": None,
        "release_matrix_path": None, "soak_seconds": 0, "passed": True})
    return p, document


def test_initial_admission_needs_no_full_model_or_soak(tmp_path):
    p, document = calibration_fixture(tmp_path)
    references = validate_calibration_evidence(document, tmp_path, p)
    assert len(references) == 14  # registration, eight raw/supervision documents, five derived components


def test_final_release_requires_actual_feature_and_soak_runs(tmp_path):
    p, document = calibration_fixture(tmp_path)
    teacher = {"schema_revision": 1, "tokens": [1, 2]}
    left, right = tmp_path / "left", tmp_path / "right"
    logits(left, teacher)
    logits(right, teacher)
    for axis in ("same_realization", "native_source"):
        result = compare_logits(left, right, teacher, p, axis)
        atomic_json(tmp_path / (axis + ".json"), result)
        document["comparisons"].append({"axis": axis, "evidence_path": axis + ".json",
            "evidence_digest": result["digest"], "passed": True})
    document.update(qualification_scope="final_release", soak_seconds=3600,
        release_matrix_path="matrix.json", release_matrix_digest="f" * 64)
    with pytest.raises(ArtifactError, match="authenticated feature and soak"):
        validate_calibration_evidence(seal(document), tmp_path, p)


@pytest.mark.parametrize("failure", ["missing_component", "duplicate_component", "failed_component", "changed_environment",
    "path_traversal", "missing_file", "unqualified", "false_soak", "fake_final", "minimal_fidelity_claim", "other_candidate_operator"])
def test_incomplete_or_mismatched_calibration_cannot_admit(tmp_path, failure):
    p, document = calibration_fixture(tmp_path)
    if failure == "missing_component":
        document["components"].pop()
    elif failure == "duplicate_component":
        document["components"][0] = document["components"][1].copy()
    elif failure == "failed_component":
        document["components"][0]["passed"] = False
    elif failure == "changed_environment":
        document["environment_digest"] = "d" * 64
    elif failure == "path_traversal":
        document["components"][0]["evidence_path"] = "../experts.json"
    elif failure == "missing_file":
        (tmp_path / "experts.json").unlink()
    elif failure == "unqualified":
        document["passed"] = False
    elif failure == "false_soak":
        document["soak_seconds"] = 3600
    elif failure == "fake_final":
        document["qualification_scope"] = "final_release"
    elif failure == "other_candidate_operator":
        teacher = {"schema_revision": 1, "tokens": [1, 2]}
        left, right = tmp_path / "left", tmp_path / "right"
        logits(left, teacher, operator="4" * 64)
        logits(right, teacher, operator="4" * 64)
        proof = compare_logits(left, right, teacher, p, "same_realization")
        atomic_json(tmp_path / "comparison.json", proof)
        document["comparisons"] = [{"axis": "same_realization", "evidence_path": "comparison.json", "evidence_digest": proof["digest"], "passed": True}]
    else:
        proof = seal({"axis": "same_realization", "passed": True,
            "logical_model_digest": p["logical_model_digest"], "policy_digest": p["digest"]})
        atomic_json(tmp_path / "comparison.json", proof)
        document["comparisons"] = [{"axis": "same_realization", "evidence_path": "comparison.json", "evidence_digest": proof["digest"], "passed": True}]
    with pytest.raises((ArtifactError, OSError)):
        validate_calibration_evidence(seal(document), tmp_path, p)
