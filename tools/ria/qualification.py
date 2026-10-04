"""Stream independent logits comparisons; keep unavailable release cells explicit."""

from datetime import datetime, timezone
import hashlib
import itertools
import math
from pathlib import Path
import struct

from .identity import ArtifactError, atomic_json, canonical, check_json, hash_file, open_regular, read_json, seal, verify_identity, within
from .qualification_schema import CALIBRATION_EVIDENCE, COMPONENT, COMPONENT_NAMES, LOGITS_COMPARISON, MATRIX_AXES, POLICY, RELEASE_MATRIX


def policy_validate(value, *, frozen=True):
    # Reuse the repository's strict numeric types and duplicate-aware decoding.
    from .schemas import StrictValidator

    check_json(value)
    error = next(StrictValidator(POLICY).iter_errors(value), None)
    if error:
        raise ArtifactError(f"qualification policy: {error.message}")
    if frozen:
        verify_identity(value)
        if "registered_at" not in value:
            raise ArtifactError("acceptance policy has not been preregistered")
        try:
            datetime.strptime(value["registered_at"], "%Y-%m-%dT%H:%M:%SZ")
        except ValueError as exc:
            raise ArtifactError("invalid policy registration time") from exc
    return value


def freeze_policy(value, output):
    if "digest" in value or "registered_at" in value:
        raise ArtifactError("freeze accepts a new unsealed policy only")
    policy_validate(value, frozen=False)
    result = seal({**value, "registered_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    atomic_json(output, result)
    return result


def release_matrix():
    axes = MATRIX_AXES
    cells = [dict(zip(axes, values, strict=True), status="unexecuted", evidence=[])
             for values in itertools.product(*axes.values())]
    return seal({"schema_revision": 1, "hardware_qualified": False, "axes": axes, "cells": cells,
                 "gates": [{"id": f"G{index:02}", "status": "unexecuted", "evidence": []}
                           for index in range(1, 29)],
                 "required_features": ["text", "reasoning", "tools", "images", "exact_continuation"],
                 "minimum_soak_seconds": 3600})


def _header(stream, document):
    raw = stream.read(128)
    if len(raw) != 128 or raw[:8] != b"RIALOG1\0":
        raise ArtifactError("invalid or truncated RIALOG1 header")
    revision, vocab, positions, labels = struct.unpack_from("<IIQQ", raw, 8)
    expected_labels = sum(document.get("label_mask", [True] * (len(document["tokens"]) - 1)))
    if revision != 1 or vocab != 129280 or positions != len(document["tokens"]) or labels != expected_labels:
        raise ArtifactError("RIALOG1 dimensions/label count differ from teacher-forced input")
    if raw[96:128] != hashlib.sha256(canonical(document)).digest():
        raise ArtifactError("RIALOG1 input identity differs from teacher-forced input")
    return raw, vocab, positions


def _input(document):
    if set(document) not in ({"schema_revision", "tokens"}, {"schema_revision", "tokens", "label_mask"}):
        raise ArtifactError("unknown or missing teacher-forced input field")
    tokens = document["tokens"]
    if type(document["schema_revision"]) is not int or document["schema_revision"] != 1:
        raise ArtifactError("unsupported teacher-forced input revision")
    if not isinstance(tokens, list) or not 2 <= len(tokens) <= 1048576:
        raise ArtifactError("teacher-forced input needs 2..1048576 tokens")
    if any(type(token) is not int or not 0 <= token < 129280 for token in tokens):
        raise ArtifactError("invalid teacher-forced token ID")
    mask = document.get("label_mask", [True] * (len(tokens) - 1))
    if not isinstance(mask, list) or len(mask) != len(tokens) - 1 or any(type(item) is not bool for item in mask) or not any(mask):
        raise ArtifactError("label mask needs at least one shifted-label position")
    return tokens, mask


def compare_logits(reference, candidate, input_document, policy, axis):
    """Bound memory to two FP32 vocabulary rows, with no model dependency."""
    import numpy as np

    policy_validate(policy)
    if axis not in ("same_realization", "native_source"):
        raise ArtifactError("unknown fidelity axis")
    tokens, mask = _input(input_document)
    limits = policy["thresholds"][axis]
    reference, candidate = Path(reference), Path(candidate)
    before = (hash_file(reference), hash_file(candidate))
    absolute = relative = squares = reference_loss = candidate_loss = 0.0
    elements = 0
    with open_regular(reference) as left, open_regular(candidate) as right:
        lh, vocab, positions = _header(left, input_document)
        rh, _, _ = _header(right, input_document)
        if lh[32:64] != rh[32:64] or lh[32:64].hex() != policy["logical_model_digest"]:
            raise ArtifactError("logical model identity differs from accepted policy")
        if axis == "same_realization" and lh[64:96] != rh[64:96]:
            raise ArtifactError("placement parity requires identical operator realization")
        for position in range(positions):
            a, b = left.read(vocab * 4), right.read(vocab * 4)
            if len(a) != vocab * 4 or len(b) != vocab * 4:
                raise ArtifactError("truncated logits row")
            av, bv = np.frombuffer(a, dtype="<f4").astype(np.float64), np.frombuffer(b, dtype="<f4").astype(np.float64)
            if not np.isfinite(av).all() or not np.isfinite(bv).all():
                raise ArtifactError("nonfinite logits")
            difference = np.abs(av - bv)
            absolute = max(absolute, float(difference.max()))
            relative = max(relative, float((difference / np.maximum(np.abs(av), limits["relative_floor"])).max()))
            squares += float(np.dot(difference, difference))
            elements += vocab
            if position + 1 < positions and mask[position]:
                label = tokens[position + 1]
                for values, which in ((av, 0), (bv, 1)):
                    peak = float(values.max())
                    loss = math.log(float(np.exp(values - peak).sum())) + (peak - float(values[label]))
                    if which:
                        candidate_loss += loss
                    else:
                        reference_loss += loss
        if left.read(1) or right.read(1):
            raise ArtifactError("unexpected bytes after complete logits artifact")
    after = (hash_file(reference), hash_file(candidate))
    if before != after:
        raise ArtifactError("logits artifacts changed during comparison")
    label_count = sum(mask)
    metrics = {"max_abs_error": absolute, "max_relative_error": relative,
               "max_rms_error": math.sqrt(squares / elements),
               "max_loss_delta": abs(candidate_loss - reference_loss) / label_count}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ArtifactError("comparison metric overflow")
    return seal({"schema_revision": 1, "kind": "teacher_forced_comparison", "axis": axis,
        "policy_digest": policy["digest"], "logical_model_digest": lh[32:64].hex(),
        "reference_operator_digest": lh[64:96].hex(), "candidate_operator_digest": rh[64:96].hex(),
        "input_digest": lh[96:128].hex(), "reference_sha256": after[0], "candidate_sha256": after[1],
        "positions": positions, "labels": label_count, "metrics": metrics,
        "reference_nll": reference_loss / label_count, "candidate_nll": candidate_loss / label_count,
        "passed": all(value <= limits[name] for name, value in metrics.items()),
        "qualification_scope": "supplied teacher-forced corpus only; other release gates remain separate"})


def compare_files(reference, candidate, inputs, policy_path, axis, output):
    result = compare_logits(reference, candidate, read_json(inputs), read_json(policy_path), axis)
    atomic_json(output, result)
    return result


def validate_calibration_evidence(document, evidence_dir, policy):
    """Authenticate an acyclic calibration proof set before deployment admission.

    Returns relative path/digest records for immutable package staging. Initial
    fixture admission deliberately precedes full-model execution; final release
    additionally requires every physical matrix cell, fidelity axis and soak.
    """
    from .schemas import StrictValidator

    policy_validate(policy)
    check_json(document)
    error = next(StrictValidator(CALIBRATION_EVIDENCE).iter_errors(document), None)
    if error:
        raise ArtifactError(f"calibration evidence: {error.message}")
    verify_identity(document)
    if not document["passed"] or document["policy_digest"] != policy["digest"]:
        raise ArtifactError("calibration has not passed the preregistered policy")
    names = [item["name"] for item in document["components"]]
    if len(set(names)) != 5 or set(names) != set(COMPONENT_NAMES):
        raise ArtifactError("calibration needs all five independent component proofs")
    references = []
    registration_digest = None
    for item in document["components"]:
        proof = read_json(within(evidence_dir, item["evidence_path"]))
        error = next(StrictValidator(COMPONENT).iter_errors(proof), None)
        if error:
            raise ArtifactError(f"qualification component: {error.message}")
        verify_identity(proof, item["evidence_digest"])
        if not item["passed"] or not proof["qualified"] or proof["component"] != item["name"] or not all(check["passed"] for check in proof["checks"]):
            raise ArtifactError("a required calibration component did not pass")
        if len({check["id"] for check in proof["checks"]}) != len(proof["checks"]):
            raise ArtifactError("duplicate component contract check")
        for name in ("environment_digest", "build_digest", "policy_digest", "profile", "executor", "operator_contract_digest"):
            if proof[name] != document[name]:
                raise ArtifactError("component identity differs from calibration realization")
        if proof["qualification_scope"] != "initial_fixture":
            raise ArtifactError("component must describe its bounded initial fixture execution")
        if registration_digest is None:
            registration_digest = proof["registration_digest"]
        elif registration_digest != proof["registration_digest"]:
            raise ArtifactError("components use different joint preregistrations")
        from .fixture_runner import validate_component_sources
        references.extend(validate_component_sources(proof, evidence_dir, policy))
        references.append({"path": item["evidence_path"], "digest": item["evidence_digest"]})
    axes = [item["axis"] for item in document["comparisons"]]
    if len(set(axes)) != len(axes):
        raise ArtifactError("duplicate fidelity axis")
    for item in document["comparisons"]:
        proof = read_json(within(evidence_dir, item["evidence_path"]))
        check_json(proof)
        error = next(StrictValidator(LOGITS_COMPARISON).iter_errors(proof), None)
        if error:
            raise ArtifactError(f"fidelity comparison: {error.message}")
        verify_identity(proof, item["evidence_digest"])
        if not item["passed"] or not proof.get("passed") or proof.get("axis") != item["axis"] or proof.get("policy_digest") != policy["digest"] or proof.get("logical_model_digest") != policy["logical_model_digest"]:
            raise ArtifactError("fidelity proof failed or has another policy/model identity")
        if (proof["labels"] >= proof["positions"] or
                proof["candidate_operator_digest"] != document["operator_contract_digest"] or
                any(value > policy["thresholds"][item["axis"]][name] for name, value in proof["metrics"].items()) or
                (item["axis"] == "same_realization" and proof["reference_operator_digest"] != proof["candidate_operator_digest"])):
            raise ArtifactError("fidelity metrics or operator identities violate the policy")
        references.append({"path": item["evidence_path"], "digest": item["evidence_digest"]})
    if document["qualification_scope"] == "initial_fixture":
        if document["release_matrix_digest"] is not None or document["release_matrix_path"] is not None or document["soak_seconds"] or document.get("release_runs"):
            raise ArtifactError("initial fixture admission cannot claim full release qualification")
    else:
        if set(axes) != {"same_realization", "native_source"} or document["soak_seconds"] < policy["minimum_soak_seconds"]:
            raise ArtifactError("final release lacks both fidelity axes or the required soak")
        if not document["release_matrix_path"] or not document["release_matrix_digest"]:
            raise ArtifactError("final release needs its physical matrix evidence")
        if not document.get("release_runs"):
            raise ArtifactError("final release needs authenticated feature and soak executions")
        from .release_runner import validate_evidence
        features = set()
        measured_soak = 0
        for reference in document["release_runs"]:
            proof = read_json(within(evidence_dir, reference["path"]))
            verify_identity(proof, reference["digest"])
            plan_reference = proof.get("plan_reference")
            if not isinstance(plan_reference, dict) or not isinstance(plan_reference.get("path"), str):
                raise ArtifactError("release execution lacks its preregistered workload plan")
            plan = read_json(within(evidence_dir, plan_reference["path"]))
            validate_evidence(proof, plan, evidence_dir, policy)
            if (proof["kind"] != "physical_replay_evidence" or not proof["passed"] or
                    proof["realization"]["profile"] != document["profile"] or
                    proof["realization"]["server_executor"] != document["executor"] or
                    any(proof[key] != document[key] for key in
                        ("policy_digest", "environment_digest", "build_digest", "operator_contract_digest"))):
                raise ArtifactError("release feature/soak execution differs from admitted realization")
            features.update(proof["observed_features"])
            measured_soak = max(measured_soak, int(proof["soak_ns"]) // 1000000000)
            references.extend((reference, plan_reference))
            references.extend({"path": step["request_path"], "digest": step["request_digest"]} for step in plan["steps"])
            references.append({"path": plan["runtime_config_path"], "digest": plan["runtime_config_digest"]})
        if set(release_matrix()["required_features"]) - features or document["soak_seconds"] > measured_soak:
            raise ArtifactError("final release feature or soak claims lack actual measured coverage")
        matrix = read_json(within(evidence_dir, document["release_matrix_path"]))
        verify_identity(matrix, document["release_matrix_digest"])
        error = next(StrictValidator(RELEASE_MATRIX).iter_errors(matrix), None)
        if error:
            raise ArtifactError(f"release matrix: {error.message}")
        if matrix["minimum_soak_seconds"] < policy["minimum_soak_seconds"]:
            raise ArtifactError("release matrix weakens the preregistered soak")
        required = release_matrix()
        keys = list(required["axes"])
        expected = {tuple(cell[key] for key in keys) for cell in required["cells"]}
        cells = matrix.get("cells", [])
        actual = {tuple(cell.get(key) for key in keys) for cell in cells}
        if matrix.get("axes") != required["axes"] or not matrix.get("hardware_qualified") or len(cells) != len(expected) or actual != expected:
            raise ArtifactError("release matrix does not cover every required physical cell")
        gates = matrix.get("gates", [])
        if len(gates) != 28 or {gate.get("id") for gate in gates} != {gate["id"] for gate in required["gates"]}:
            raise ArtifactError("release matrix lacks a required gate family")
        from .physical_contract import validate_physical_contract
        checked_physical = {}
        for row in [*cells, *gates]:
            if row.get("status") != "passed" or not row.get("evidence"):
                raise ArtifactError("release matrix contains an incomplete gate or cell")
            for reference in row["evidence"]:
                identity = (reference["path"], reference["digest"])
                if identity not in checked_physical:
                    proof = read_json(within(evidence_dir, reference["path"]))
                    verify_identity(proof, reference["digest"])
                    references.extend(validate_physical_contract(proof, evidence_dir, policy))
                    if proof["passed"] is not True:
                        raise ArtifactError("matrix proof contains a failed measured physical contract")
                    checked_physical[identity] = proof
                proof = checked_physical[identity]
                if "id" in row:
                    covered = row["id"] in proof.get("covered_gates", [])
                else:
                    cell_identity = {key: row[key] for key in keys}
                    covered = cell_identity in proof.get("covered_cells", [])
                if not covered:
                    raise ArtifactError("physical proof does not cover this matrix row")
                references.append(reference)
        references.append({"path": document["release_matrix_path"], "digest": document["release_matrix_digest"]})
    if len({item["path"] for item in references}) != len(references):
        # Multiple matrix rows may deliberately reference one sealed integrated
        # run. Conflicting identities for one path are never permitted.
        seen = {}
        for item in references:
            if item["path"] in seen and seen[item["path"]] != item["digest"]:
                raise ArtifactError("one proof path has conflicting identities")
            seen[item["path"]] = item["digest"]
        references = [{"path": path, "digest": digest} for path, digest in sorted(seen.items())]
    return references


def calibration_report(evidence, evidence_dir, policy):
    validate_calibration_evidence(evidence, evidence_dir, policy)
    return seal({"schema_revision": 1, "profile": evidence["profile"],
        "operator_contract_digest": evidence["operator_contract_digest"],
        "executor": evidence["executor"], "qualified": True,
        "environment_digest": evidence["environment_digest"], "build_digest": evidence["build_digest"],
        "evidence_digest": evidence["digest"], "policy_digest": policy["digest"]})
