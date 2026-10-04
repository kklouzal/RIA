"""Validate independent instrumented physical reports; hashes are not attestation.

This module does not execute workloads or establish that a report describes real
hardware. The independent producer owns that obligation. It authenticates the
reported bytes, preregistered bounds and identities, and measurement-backed
coverage before the release aggregator may use a physical report.
"""

import math
import os
import stat
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .identity import ArtifactError, canonical, check_json, loads, u64, verify_identity
from .qualification_schema import MATRIX_AXES, POLICY, record

MAX_CHECKS = 4096
MAX_REFERENCES = 4096
MAX_DOCUMENT_BYTES = 8 << 20
MAX_TOTAL_BYTES = 64 << 20
MAX_DOCUMENT_NODES = 250000
MAX_DEPTH = 16

SHA = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
U64 = {"type": "string", "pattern": "^(0|[1-9][0-9]*)$", "maxLength": 20}
NUMBER = {"type": "number"}
VALUE = {"anyOf": [NUMBER, U64]}
BOUND = {"anyOf": [VALUE, {"type": "null"}]}
ID = {"type": "string", "pattern": "^[A-Za-z][A-Za-z0-9_.:-]{0,127}$"}
UNIT = {"type": "string", "pattern": "^[A-Za-z][A-Za-z0-9_./^-]{0,63}$"}
UTC = {"type": "string", "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"}
REFERENCE = record({"path": {"type": "string", "minLength": 1, "maxLength": 1024,
    "pattern": "^[A-Za-z0-9][A-Za-z0-9._/-]*$"}, "digest": SHA})
CELL = record({key: {"enum": values} for key, values in MATRIX_AXES.items()})
GATE = {"enum": [f"G{index:02}" for index in range(1, 29)]}


def _array(items, maximum, minimum=0):
    return {"type": "array", "items": items, "minItems": minimum,
            "maxItems": maximum, "uniqueItems": True}


CELLS = _array(CELL, 540)
GATES = _array(GATE, 28)
IDENTITY_FIELDS = ("policy_digest", "logical_model_digest", "source_lock_digest",
    "environment_digest", "build_digest", "operator_contract_digest", "runtime_config_digest",
    "profile", "server_executor")
COMMON = {**{key: SHA for key in IDENTITY_FIELDS[:-2]},
    "profile": {"enum": ["nvfp4", "fp8", "bf16"]},
    "server_executor": {"enum": ["cpu", "cuda"]}}
RUNTIME_IDENTITY = record({"schema_revision": {"const": 1},
    "kind": {"const": "physical_runtime_identity"}, **COMMON, "digest": SHA})
PLANNED_CHECK = record({"id": ID, "unit": UNIT, "domain": {"enum": ["number", "u64"]},
    "minimum": BOUND, "maximum": BOUND, "cells": CELLS, "gates": GATES})
PLAN = record({"schema_revision": {"const": 1}, "kind": {"const": "physical_contract_plan"},
    "qualification_scope": {"const": "final_release"}, **COMMON,
    "registered_at": UTC, "binary_sha256": SHA, "max_elapsed_ns": U64,
    "runtime_reference": REFERENCE, "required_cells": CELLS, "required_gates": GATES,
    "checks": _array(PLANNED_CHECK, MAX_CHECKS, 1), "digest": SHA})
EXECUTION = record({"run_id": SHA, "started_at": UTC,
    "clock": {"const": "CLOCK_MONOTONIC"}, "elapsed_ns": U64,
    "pid": {"type": "integer", "minimum": 1, "maximum": 2147483647},
    "complete_child_lifetime": {"const": True}, "child_exit_code": {"const": 0},
    "child_signal": {"type": "null"}, "binary_sha256": SHA})
MEASUREMENT = record({"check_id": ID, "measurement_id": ID, "unit": UNIT,
    "value": VALUE, "cells": CELLS, "gates": GATES})
MEASUREMENTS = record({"schema_revision": {"const": 1},
    "kind": {"const": "physical_contract_measurements"}, "classification": {"const": "physical"},
    **COMMON, "plan_digest": SHA, "execution": EXECUTION,
    "measurements": _array(MEASUREMENT, MAX_CHECKS, 1), "digest": SHA})
RESULT = record({"id": ID, "measurement_id": ID, "raw_digest": SHA,
    "passed": {"type": "boolean"}})
EVIDENCE = record({"schema_revision": {"const": 1},
    "kind": {"const": "physical_contract_evidence"}, "classification": {"const": "physical"},
    "qualification_scope": {"const": "final_release"}, **COMMON,
    "plan_reference": REFERENCE, "raw_evidence": _array(REFERENCE, MAX_REFERENCES, 1),
    "covered_cells": CELLS, "covered_gates": GATES,
    "checks": _array(RESULT, MAX_CHECKS, 1), "passed": {"type": "boolean"}, "digest": SHA})
PHYSICAL_CONTRACT_SCHEMAS = {
    "physical-runtime-identity": RUNTIME_IDENTITY,
    "physical-contract-plan": PLAN,
    "physical-contract-measurements": MEASUREMENTS,
    "physical-contract-evidence": EVIDENCE,
}


def _document(value, schema, expected=None):
    # Imported lazily because the canonical registry imports our schemas.
    from .schemas import StrictValidator

    check_json(value, max_depth=MAX_DEPTH, max_nodes=MAX_DOCUMENT_NODES)
    if len(canonical(value)) > MAX_DOCUMENT_BYTES:
        raise ArtifactError("physical contract document exceeds its byte bound")
    if next(StrictValidator(schema).iter_errors(value), None) is not None:
        # Do not quote externally supplied values, credentials or document bodies.
        raise ArtifactError("physical contract does not satisfy its strict schema")
    verify_identity(value, expected)


def _same_identity(value, expected):
    if any(value[key] != expected[key] for key in IDENTITY_FIELDS):
        raise ArtifactError("physical contract realization identities disagree")


def _utc(value):
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError) as exc:
        raise ArtifactError("physical contract UTC timestamp is invalid") from exc


def _quantity(value, domain):
    if domain == "u64":
        return u64(value)
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ArtifactError("physical measurement requires a finite number in its planned domain")
    return value


def _cell_key(cell):
    return tuple(cell[key] for key in MATRIX_AXES)


def _coverage(cells, gates, candidate):
    if any(cell["profile"] != candidate["profile"] or
           cell["server_executor"] != candidate["server_executor"] for cell in cells):
        raise ArtifactError("physical cell does not belong to its candidate realization")
    cell_keys, gate_keys = {_cell_key(cell) for cell in cells}, set(gates)
    if len(cell_keys) != len(cells) or len(gate_keys) != len(gates):
        raise ArtifactError("physical coverage contains duplicates")
    return cell_keys, gate_keys


class _References:
    """Authenticate bounded JSON snapshots without following path-component links."""

    def __init__(self, root, initial_bytes):
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise ArtifactError("physical evidence root must be a directory")
        self.references = {}
        self.documents = {}
        self.total_bytes = initial_bytes

    def read(self, reference, schema):
        relative, expected = reference["path"], reference["digest"]
        parts = PurePosixPath(relative).parts
        if (not parts or str(PurePosixPath(relative)) != relative or
                any(part in (".", "..") for part in relative.split("/"))):
            raise ArtifactError("physical evidence reference is not a canonical relative path")
        if relative in self.references:
            if self.references[relative] != expected:
                raise ArtifactError("physical evidence path has conflicting identities")
            value = self.documents[relative]
            _document(value, schema, expected)
            return value
        if len(self.references) >= MAX_REFERENCES:
            raise ArtifactError("physical evidence exceeds the total reference bound")
        remaining = min(MAX_DOCUMENT_BYTES, MAX_TOTAL_BYTES - self.total_bytes)
        if remaining <= 0:
            raise ArtifactError("physical evidence exceeds the total byte bound")
        root_fd = fd = None
        try:
            root_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            fd = root_fd
            for part in parts[:-1]:
                next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                  dir_fd=fd)
                if fd != root_fd:
                    os.close(fd)
                fd = next_fd
            leaf_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
                              dir_fd=fd)
            with os.fdopen(leaf_fd, "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise ArtifactError("physical evidence reference requires a regular file")
                raw = stream.read(remaining + 1)
            if len(raw) > remaining:
                raise ArtifactError("physical evidence exceeds its byte bound")
            self.total_bytes += len(raw)
            value = loads(raw, max_bytes=MAX_DOCUMENT_BYTES, max_depth=MAX_DEPTH,
                          max_nodes=MAX_DOCUMENT_NODES)
            _document(value, schema, expected)
        except OSError as exc:
            raise ArtifactError("physical evidence reference cannot be read safely") from exc
        finally:
            if fd is not None and fd != root_fd:
                os.close(fd)
            if root_fd is not None:
                os.close(root_fd)
        self.references[relative] = expected
        self.documents[relative] = value
        return value

    def result(self):
        return [{"path": path, "digest": expected} for path, expected in sorted(self.references.items())]


def validate_physical_contract(proof, evidence_dir, frozen_policy):
    """Return authenticated relative ``{path, digest}`` dependencies for staging.

    Accept a complete report whose pass bits equal recomputed inclusive bounds;
    a measured failing report remains valid evidence with ``passed=False``.
    Callers require ``passed=True`` before crediting release coverage. All
    referenced JSON is selfsealed and bounded. Selfhash integrity is not hardware
    attestation, independent provenance verification, or a substitute for the
    producer actually running every instrumented obligation.

    A proof covers one profile/server-executor realization. Matrix aggregation
    can combine distinct proofs for other candidates; each proof binds its own
    environment, build, operator and runtime identities throughout.
    """
    _document(frozen_policy, POLICY)
    _document(proof, EVIDENCE)
    if any(proof[key] != frozen_policy[key] for key in ("logical_model_digest", "source_lock_digest")) or \
            proof["policy_digest"] != frozen_policy["digest"]:
        raise ArtifactError("physical evidence does not bind the frozen policy and source model")
    if "registered_at" not in frozen_policy:
        raise ArtifactError("physical evidence requires a preregistered frozen policy")
    policy_time = _utc(frozen_policy["registered_at"])
    references = _References(evidence_dir, len(canonical(proof)) + len(canonical(frozen_policy)))
    plan = references.read(proof["plan_reference"], PLAN)
    _same_identity(plan, proof)
    plan_time = _utc(plan["registered_at"])
    if plan_time < policy_time:
        raise ArtifactError("physical plan predates its frozen policy")
    deadline = u64(plan["max_elapsed_ns"])
    if deadline == 0:
        raise ArtifactError("physical plan requires a positive monotonic execution bound")
    runtime = references.read(plan["runtime_reference"], RUNTIME_IDENTITY)
    _same_identity(runtime, plan)
    required_cells, required_gates = _coverage(plan["required_cells"], plan["required_gates"], plan)
    if not required_cells and not required_gates:
        raise ArtifactError("physical plan requires at least one matrix cell or gate")
    if _coverage(proof["covered_cells"], proof["covered_gates"], proof) != (required_cells, required_gates):
        raise ArtifactError("physical proof coverage differs from its preregistered obligations")

    planned = {}
    assigned_cells, assigned_gates = set(), set()
    for check in plan["checks"]:
        if check["id"] in planned:
            raise ArtifactError("physical plan check identifiers must be unique")
        minimum = None if check["minimum"] is None else _quantity(check["minimum"], check["domain"])
        maximum = None if check["maximum"] is None else _quantity(check["maximum"], check["domain"])
        if minimum is None and maximum is None or minimum is not None and maximum is not None and minimum > maximum:
            raise ArtifactError("physical check must declare consistent nonempty bounds")
        cells, gates = _coverage(check["cells"], check["gates"], plan)
        if not cells and not gates or not cells <= required_cells or not gates <= required_gates:
            raise ArtifactError("physical check assignments must cover only planned obligations")
        assigned_cells.update(cells)
        assigned_gates.update(gates)
        planned[check["id"]] = (check, minimum, maximum, cells, gates)
    if assigned_cells != required_cells or assigned_gates != required_gates:
        raise ArtifactError("physical plan contains an obligation without a planned measurement")

    measured, measurement_ids, raw_digests, run_ids = {}, set(), set(), set()
    for reference in proof["raw_evidence"]:
        raw = references.read(reference, MEASUREMENTS)
        _same_identity(raw, plan)
        if raw["plan_digest"] != plan["digest"]:
            raise ArtifactError("physical measurement does not bind its preregistered plan")
        execution = raw["execution"]
        elapsed = u64(execution["elapsed_ns"])
        if execution["binary_sha256"] != plan["binary_sha256"] or elapsed == 0 or elapsed > deadline:
            raise ArtifactError("physical execution violates its binary identity or monotonic bound")
        if _utc(execution["started_at"]) < plan_time:
            raise ArtifactError("physical execution started before preregistration")
        if raw["digest"] in raw_digests or execution["run_id"] in run_ids:
            raise ArtifactError("physical raw evidence must identify distinct completed runs")
        raw_digests.add(raw["digest"])
        run_ids.add(execution["run_id"])
        for measurement in raw["measurements"]:
            check_id, measurement_id = measurement["check_id"], measurement["measurement_id"]
            if check_id not in planned or check_id in measured or measurement_id in measurement_ids:
                raise ArtifactError("physical measurements must identify each planned check exactly once")
            if len(measured) >= MAX_CHECKS:
                raise ArtifactError("physical evidence exceeds its total measurement bound")
            check, minimum, maximum, cells, gates = planned[check_id]
            value = _quantity(measurement["value"], check["domain"])
            if measurement["unit"] != check["unit"] or \
                    _coverage(measurement["cells"], measurement["gates"], plan) != (cells, gates):
                raise ArtifactError("physical measurement units or assigned coverage differ from its plan")
            passed = (minimum is None or value >= minimum) and (maximum is None or value <= maximum)
            measured[check_id] = (measurement_id, raw["digest"], passed)
            measurement_ids.add(measurement_id)
    if set(measured) != set(planned):
        raise ArtifactError("physical report omits a planned measurement")
    declared = {}
    for check in proof["checks"]:
        if check["id"] in declared:
            raise ArtifactError("physical proof check identifiers must be unique")
        declared[check["id"]] = (check["measurement_id"], check["raw_digest"], check["passed"])
    if declared != measured or proof["passed"] != all(item[2] for item in measured.values()):
        raise ArtifactError("physical proof pass claims disagree with recomputed numeric bounds")
    return references.result()
