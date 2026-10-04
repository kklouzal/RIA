"""Offline producer availability is distinct from executing physical tests."""

from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from ria.identity import read_json, verify_identity
from ria.qualification_readiness import gate_catalog, qualification_readiness
from ria.qualification_schema import MATRIX_AXES


def test_fixed_gate_and_matrix_obligations_require_semantic_software():
    catalog = gate_catalog()
    verify_identity(catalog)
    assert [gate["id"] for gate in catalog["gates"]] == [f"G{index:02}" for index in range(1, 29)]
    assert all(len(gate["obligations"]) >= 3 and gate["specification_sections"] for gate in catalog["gates"])
    result = qualification_readiness()
    verify_identity(result)
    assert not result["executed"] and not result["release_ready"] and not result["hardware_qualified"]
    assert result["gate_catalog_digest"] == catalog["digest"]
    assert result["matrix"]["required_cells"] == 540 and result["matrix"]["axes"] == MATRIX_AXES
    assert result["matrix"]["status"] == "blocked_missing_software"
    assert all(gate["semantic_release_adapter"] == "missing" for gate in result["gates"])
    assert result["gates"][1]["id"] == "G02"
    assert result["gates"][1]["status"] == "available_software_unexecuted"
    assert {row["status"] for row in result["gates"][24]["obligation_software"]} == {"available", "partial", "missing"}
    assert {row["scope"] for row in result["partial_producers"]} == {"initial_fixture", "http_replay", "supplied_corpus"}


def test_readiness_command_is_runnable_and_returns_failure_for_missing_software(tmp_path):
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "qualification readiness.json"
    result = subprocess.run([sys.executable, str(root / "tools/qualify_ria.py"), "readiness",
                             "--output", str(output)], cwd=root, capture_output=True, timeout=20)
    assert result.returncode == 1
    assert result.stderr == b""
    report = read_json(output)
    assert report == qualification_readiness()
    assert report["gates"][21]["id"] == "G22"
    assert any("fault" in value for value in report["required_external_inputs"])
