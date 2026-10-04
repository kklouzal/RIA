"""Model-free workflow ordering, durable ownership and registration derivation."""

from copy import deepcopy
from pathlib import Path
import sys
import threading
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from fixture_measurements import registration, synthetic_environment
from test_qualification import policy
from ria.identity import ArtifactError, read_json, seal
from ria.setup import SetupController, role_lock, workspace_lock
from ria.setup_journal import SetupJournal
from ria.setup_qualification import register, validate_tuning
from setup_ria import override


def qualification_input(p, reg):
    return {"schema_revision": 1,
        "policy": {key: value for key, value in p.items() if key not in
            ("logical_model_digest", "source_lock_digest", "registered_at", "digest")},
        "native": {key: reg["runs"]["native_server"]["request_body"][key] for key in
            ("deadline_ms", "warmup", "repeats", "fixture_seed")},
        "transport": {key: reg["runs"]["transport_server"]["request_body"][key] for key in
            ("deadline_ms", "warmup", "repeats", "fixture_seed", "max_frame_bytes", "control_credit",
             "expert_credit", "row_credit", "bulk_credit")},
        "hard_limits": {name: item["hard_limits"] for name, item in reg["runs"].items()}}


@pytest.mark.parametrize("tls", [True, False])
@pytest.mark.parametrize("executor", ["cpu", "cuda"])
def test_registration_uses_actual_role_builds_and_environments(tmp_path, tls, executor):
    p = policy(tmp_path)
    builds = {role: seal({"image_kind": "cuda" if role == "client" else executor,
        "source_lock_digest": p["source_lock_digest"], "binaries": {"ds4ctl": ("1" if role == "server" else "2") * 64,
            "ds4-ria-qualify": ("3" if role == "server" else "4") * 64}}) for role in ("server", "client")}
    environments = {role: synthetic_environment(p["logical_model_digest"], p["source_lock_digest"], "3" * 64,
        "bf16", executor, role, builds[role]["digest"], tls_enabled=tls) for role in builds}
    reg = registration(p, executor=executor, environments=environments,
        server_build=builds["server"]["digest"], client_build=builds["client"]["digest"], tls_enabled=tls)
    tuning = qualification_input(p, reg)
    result = register(tuning, {"logical_model_digest": p["logical_model_digest"], "operator_contract_digest": "3" * 64,
        "profile": "bf16"}, builds, environments, {"server": "5" * 64, "client": "6" * 64}, tmp_path / "derived")
    actual = result["registration"]
    assert actual["runs"]["native_server"]["binary_sha256"] == "3" * 64
    assert actual["runs"]["native_client"]["binary_sha256"] == "4" * 64
    assert actual["realizations"] == {role: {"environment_digest": environments[role]["digest"],
        "build_digest": builds[role]["digest"]} for role in builds}
    for name in ("transport_server", "transport_client"):
        bootstrap = actual["runs"][name]["bootstrap_body"]
        if tls:
            assert bootstrap["tls"]["authorized_peer_sha256"] == ("6" if name.endswith("server") else "5") * 64
        else:
            assert bootstrap["tls"] == {"enabled": False}
    assert tuning == qualification_input(p, reg)


def test_omitted_fixture_byte_limits_derive_from_explicit_role_caps(tmp_path):
    p = policy(tmp_path)
    builds = {role: seal({"image_kind": "cuda" if role == "client" else "cpu",
        "source_lock_digest": p["source_lock_digest"], "binaries": {"ds4ctl": "1" * 64,
            "ds4-ria-qualify": "2" * 64}}) for role in ("server", "client")}
    environments = {role: synthetic_environment(p["logical_model_digest"], p["source_lock_digest"], "3" * 64,
        "bf16", "cpu", role, builds[role]["digest"], tls_enabled=False) for role in builds}
    tuning = qualification_input(p, registration(p, environments=environments,
        server_build=builds["server"]["digest"], client_build=builds["client"]["digest"], tls_enabled=False))
    for limits in tuning["hard_limits"].values():
        for key in ("host_bytes", "device_bytes", "pinned_bytes", "max_rss_bytes"):
            limits.pop(key)
    result = register(tuning, {"logical_model_digest": p["logical_model_digest"], "operator_contract_digest": "3" * 64,
        "profile": "bf16"}, builds, environments, {"server": None, "client": None}, tmp_path / "derived")
    for name, run in result["registration"]["runs"].items():
        environment = environments[name.rsplit("_", 1)[1]]
        limits = run["hard_limits"]
        assert int(limits["host_bytes"]) == environment["planning_request"]["caps"]["host_bytes"]
        assert int(limits["max_rss_bytes"]) == environment["environment"]["cgroup_bytes"]
        if name.startswith("transport_") or name.endswith("server"):
            assert limits["device_bytes"] == limits["pinned_bytes"] == "0"


@pytest.mark.parametrize("mutation", ["threshold", "unknown", "repeats", "identity"])
def test_tuning_rejects_missing_policy_or_derived_identity(tmp_path, mutation):
    p = policy(tmp_path)
    tuning = qualification_input(p, registration(p))
    if mutation == "threshold":
        del tuning["policy"]["thresholds"]["same_realization"]["max_abs_error"]
    elif mutation == "unknown":
        tuning["force_admission"] = True
    elif mutation == "repeats":
        tuning["native"]["repeats"] = 1
    else:
        tuning["policy"]["logical_model_digest"] = "0" * 64
    with pytest.raises(ArtifactError):
        validate_tuning(tuning)


def controller(tmp_path):
    result = SetupController.__new__(SetupController)
    result.workspace = tmp_path
    result.journal = SetupJournal(tmp_path / "journal")
    result.stage_lock = threading.RLock()
    result.settings_digest = "1" * 64
    result.settings = {"role": "client"}
    result.invitation = {"job_id": "2" * 64}
    result.deadline = time.monotonic() + 600
    result.cancelled = threading.Event()
    return result


def test_stage_intent_precedes_effect_and_lost_reply_never_reruns(tmp_path):
    owner = controller(tmp_path)
    calls = []
    def operation():
        receipt = read_json(tmp_path / "journal" / "probe.json")
        assert receipt["status"] == "running"
        calls.append(1)
        return {"actual_digest": "3" * 64}
    first = owner.step("probe", operation)
    assert owner.step("probe", operation) == first
    assert calls == [1]
    owner.settings["peer_address"] = "127.0.0.1"
    assert owner.step("probe", operation) == first
    assert calls == [1]


def test_completed_stage_cannot_accept_changed_payload(tmp_path):
    owner = controller(tmp_path)
    owner.step("client-offer", lambda: {"accepted": True}, inputs={"layout": "3" * 64})
    with pytest.raises(ArtifactError, match="different inputs"):
        owner.step("client-offer", lambda: {"accepted": True}, inputs={"layout": "4" * 64})


def test_failed_or_interrupted_measurement_cannot_be_reexecuted(tmp_path):
    owner = controller(tmp_path)
    with pytest.raises(ArtifactError, match="fixture failed"):
        owner.step("native-fixture", lambda: (_ for _ in ()).throw(ArtifactError("fixture failed")))
    with pytest.raises(ArtifactError, match="previously failed"):
        owner.step("native-fixture", lambda: {"manufactured": True})
    owner.journal.accept("transport-fixture", "transport-fixture", {})
    recovered = SetupJournal(tmp_path / "journal")
    assert recovered.status("transport-fixture")["status"] == "failed"
    assert recovered.interrupted == ["transport-fixture"]


def test_workspace_and_role_exclusion_and_symlink_rejection(tmp_path):
    root = tmp_path / "workspace"
    with workspace_lock(root):
        with pytest.raises(ArtifactError, match="another controller"):
            with workspace_lock(root):
                pytest.fail("duplicate workspace controller admitted")
    with role_lock("expert", directory=tmp_path):
        with pytest.raises(ArtifactError, match="another setup job"):
            with role_lock("expert", directory=tmp_path):
                pytest.fail("another workspace could replace this role")
        with role_lock("client", directory=tmp_path):
            pass
    (tmp_path / "alias").symlink_to(root, target_is_directory=True)
    with pytest.raises(ArtifactError):
        with workspace_lock(tmp_path / "alias"):
            pytest.fail("aliased workspace admitted")


def test_launch_requires_both_admission_and_fixture_quiescence(tmp_path):
    owner = controller(tmp_path)
    with pytest.raises(ArtifactError, match="finalize"):
        owner.launch()
    owner.step("finalize", lambda: {"admitted": True})
    with pytest.raises(ArtifactError, match="stop-fixture"):
        owner.launch()


def test_external_overrides_are_typed_existing_paths_without_execution(tmp_path):
    target = tmp_path / "must-not-exist"
    original = {"profile": "bf16", "planning": {"context_positions": 42}}
    settings = deepcopy(original)
    override(settings, 'planning.context_positions=64')
    override(settings, 'profile="$(touch ' + str(target) + ')"')
    assert settings["planning"]["context_positions"] == 64
    assert not target.exists()
    assert original["planning"]["context_positions"] == 42
    with pytest.raises(ArtifactError):
        override(settings, "missing.control=true")
    with pytest.raises(ArtifactError):
        override(settings, "planning.context_positions=True")
