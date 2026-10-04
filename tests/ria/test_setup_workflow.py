"""Genuine controller sequencing over recorded, explicitly unqualified boundaries.

No Engine, model inference, GPU, publisher HTTP or container is executed. Tiny
synthetic packages use the real preparer/extractor and filesystem publications.
Host discovery, measurements, admission and Engine operations are recording
oracles; their artifacts are never claimed as production qualification.
"""

from copy import deepcopy
from contextlib import contextmanager
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import threading
import time
from types import MethodType

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from fixture_measurements import synthetic_environment
from ria import setup as workflow, setup_artifacts as artifacts
from ria.identity import ArtifactError, atomic_json, canonical, read_json, seal, verify_identity
from ria.setup_security import SetupSecurity
from test_setup_artifacts import prepared, settings as artifact_settings
from test_setup_config_host import settings_fixture


HEALTHY_ORDER = [
    "local.context", "remote.submit.commit-plan", "remote.wait.commit-plan", "remote.call.offer", "local.discover",
    "remote.fetch.client-model", "local.bootstrap", "remote.submit.client-offer", "remote.wait.client-offer", "remote.submit.registration",
    "remote.wait.registration", "local.registration", "remote.submit.fixture-start", "remote.wait.fixture-start",
    "local.fixture-start", "remote.submit.probe", "remote.wait.probe", "local.probe", "remote.submit.native-fixture",
    "remote.wait.native-fixture", "local.native-fixture", "remote.submit.transport-fixture", "local.transport-fixture",
    "remote.wait.transport-fixture", "remote.upload.client-runs", "remote.submit.collect", "remote.wait.collect",
    "remote.fetch.expert-runs", "remote.submit.finalize", "remote.wait.finalize", "local.finalize", "remote.submit.stop",
    "remote.wait.stop", "local.stop-fixture", "remote.submit.launch", "remote.wait.launch", "remote.submit.health",
    "remote.wait.health", "local.launch", "local.health", "remote.call.finish",
]


@pytest.fixture
def flow(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "RUNTIME_UID", os.geteuid())
    monkeypatch.setattr(artifacts, "RUNTIME_GID", os.getegid())
    client = artifacts.private_directory(tmp_path / "client")
    settings, _, request = settings_fixture(client, "client", "cuda")
    # Remove only paths created by this exact synthetic fixture; incoming model
    # publication and marked secrets must be exercised rather than prepopulated.
    for name in ("model", "secrets", "reports"):
        shutil.rmtree(client / name)
    source = artifacts.private_directory(tmp_path / "model-fixture")
    _, _, server, servermanifest = prepared(source)
    facts = artifacts.prepare_expert(artifact_settings(server, servermanifest), tmp_path / "expert-artifacts")
    invitation = {"schema_revision": 1, "job_id": "1" * 64, "pair_secret": "2" * 64,
                  "endpoint": "192.168.10.3:9010", "tls_enabled": False, "ca_pem": None, "server_leaf_sha256": None}
    settings["peer_address"] = "192.168.10.3"
    builds = {role: seal({"image_kind": "cpu" if role == "server" else "cuda", "source_lock_digest": "4" * 64,
                         "sources": {"synthetic-workflow-boundary": "5" * 64}}) for role in ("server", "client")}
    state = {"events": [], "fail": None, "triggered": False, "launched": False, "fixture": False,
             "command_deadlines": [], "facts": facts, "builds": builds}
    def record(name):
        state["events"].append(name)
        if name == state.get("signal"):
            os.kill(os.getpid(), signal.SIGTERM)
        if name == state["fail"] and not state["triggered"]:
            state["triggered"] = True
            raise ArtifactError("injected boundary failure: " + name)
    def runner(arguments, deadline, **kwargs):
        state["command_deadlines"].append(deadline)
        assert type(deadline) is int and deadline > 0
        if arguments[3] == "ps":
            return b""
        if arguments[3] == "exec":
            assert arguments[4] == "a" * 64 and arguments[5] == "/usr/local/bin/ds4ctl"
            record("local.health")
            return b""
        if arguments[3] in ("stop", "rm"):
            assert arguments[-1] == "a" * 64
            record("cleanup.model." + arguments[3])
            state["launched"] = False
            return b""
        raise AssertionError("unexpected Engine operation: " + repr(arguments))
    owner = workflow.SetupController(settings, invitation=invitation, runner=runner)
    monkeypatch.setattr(workflow, "validate_controller_context", lambda *args, **kwargs: record("local.context"))
    state["owner"] = owner
    environment = synthetic_environment(facts["logical_model_digest"], builds["client"]["source_lock_digest"],
        facts["operator_contract_digest"], "bf16", "cpu", "client", builds["client"]["digest"], tls_enabled=False)
    def discover(*args, **kwargs):
        record("local.discover")
        return {"build_info": builds["client"], "host_report": seal({"classification": "unqualified workflow boundary"})}
    monkeypatch.setattr(workflow, "discover_local", discover)
    def bootstrap(self, manifest, policy_path):
        record("local.bootstrap")
        self.request = deepcopy(request)
        artifacts._publish(Path(self.paths["operator"]) / "request.json", self.request)
        artifacts.runtime_report_directory(self.workspace)
        directory = artifacts.private_directory(self.paths["bootstrap"])
        atomic_json(directory / "bootstrap.json", seal({"classification": "unqualified workflow boundary"}))
        atomic_json(directory / "environment.json", environment)
        return environment
    owner._bootstrap = MethodType(bootstrap, owner)
    def install_registration(self, result):
        record("local.registration")
        return {"registration_digest": "6" * 64}
    owner.install_registration = MethodType(install_registration, owner)
    def fixture_start(*args, **kwargs):
        record("local.fixture-start")
        marker = read_json(Path(owner.paths["bootstrap"]) / "bootstrap.json")
        atomic_json(Path(owner.paths["reports"]) / ("fixture-" + marker["digest"] + ".json"),
                    {"classification": "unqualified workflow ownership boundary"})
        state["fixture"] = True
        return {"container_id": "b" * 64}
    monkeypatch.setattr(workflow.deployment, "fixture_start", fixture_start)
    def fixture_exec(request, directory, name, **kwargs):
        operation = "probe" if name == "probe" else name.split("_")[0] + "-fixture"
        record("local." + operation)
        document = seal({"classification": "unqualified workflow measurement boundary", "operation": operation})
        reports = Path(owner.paths["reports"])
        if name == "probe":
            atomic_json(reports / "probe.json", document)
        else:
            path = reports / name.replace("_", "-")
            path.mkdir(mode=0o700)
            atomic_json(path / "measurements.json", document)
            atomic_json(path / "supervision.json", document)
    monkeypatch.setattr(workflow.deployment, "fixture_exec", fixture_exec)
    def finalize(self):
        self.require("collect")
        record("local.finalize")
        artifacts.private_directory(self.paths["final"])
        return {"deployment_lock_digest": "7" * 64, "qualification_scope": "initial_fixture", "final_release_qualified": False}
    owner.finalize = MethodType(finalize, owner)
    def fixture_stop(*args, **kwargs):
        name = "cleanup.fixture" if state["owner"].cleaning else "local.stop-fixture"
        record(name)
        state["fixture"] = False
        return {"stopped": True}
    monkeypatch.setattr(workflow.deployment, "fixture_stop", fixture_stop)
    def launch(*args, **kwargs):
        owner.require("finalize")
        owner.require("stop-fixture")
        assert not state["fixture"]
        record("local.launch")
        state["launched"] = True
    monkeypatch.setattr(workflow.deployment, "launch", launch)
    def observed(self, *, allow_absent=False, deadline_ms=None, require_owner=False):
        if not state["launched"]:
            assert allow_absent
            return None
        return {"Id": "a" * 64, "State": {"Running": True, "Pid": 1234, "Health": {"Status": "healthy"}}}
    owner._owned_service = MethodType(observed, owner)
    class Peer:
        def __init__(self, invitation, **kwargs):
            assert invitation["job_id"] == "1" * 64
            self.pending = {}
        def call(self, operation, payload, *, timeout=None):
            if timeout is not None:
                assert timeout > 0
            if operation == "submit":
                return self.submit(payload["step_id"], payload["name"], payload["payload"])
            abort = operation == "stop" and payload == {"abort": True}
            record("remote.call." + ("abort" if abort else operation))
            if operation == "offer":
                return {"model": facts, "host": {"build_info": builds["server"]}, "bind_address": settings["peer_address"]}
            if operation == "client-offer":
                assert payload["facts"]["client_manifest_digest"] == facts["client_manifest_digest"]
                return {"accepted": True}
            if operation == "finish":
                assert state["launched"] and owner.journal.status("health")["status"] == "completed"
            return {}
        def submit(self, step, operation, payload):
            assert step == operation
            record("remote.submit." + operation)
            if operation == "client-offer":
                assert payload["facts"]["client_manifest_digest"] == facts["client_manifest_digest"]
            self.pending[step] = payload
            return {"step_id": step}
        def wait(self, step, *, timeout):
            assert timeout > 0 and step in self.pending
            record("remote.wait." + step)
            return {}
        def fetch_tree(self, name, output, **kwargs):
            record("remote.fetch." + name)
            if name == "client-model":
                shutil.copytree(facts["export_root"], output)
            else:
                Path(output).mkdir(mode=0o700)
            return {"classification": "unqualified transfer boundary"}
        def upload_tree(self, name, path, **kwargs):
            record("remote.upload." + name)
            assert (Path(path) / "native-client" / "measurements.json").is_file()
            assert (Path(path) / "transport-client" / "measurements.json").is_file()
            return {"uploaded": True}
    monkeypatch.setattr(workflow, "PeerClient", Peer)
    # run_setup keeps its genuine locks, initialization, receipts and exception
    # cleanup; only construction is redirected to our recorded boundary owner.
    monkeypatch.setattr(workflow, "SetupController", lambda *args, **kwargs: state["owner"])
    locks = artifacts.private_directory(tmp_path / "locks")
    original_role_lock = workflow.role_lock
    monkeypatch.setattr(workflow, "role_lock", lambda role: original_role_lock(role, directory=locks))
    state.update(settings=settings, invitation=invitation, record=record, environment=environment, request=request)
    return state


def test_genuine_client_healthy_stage_order_and_unqualified_completion(flow):
    result = workflow.run_setup(flow["settings"], invitation=flow["invitation"])
    assert flow["events"] == HEALTHY_ORDER
    assert result["healthy"] is True and result["qualification_scope"] == "initial_fixture"
    assert result["final_release_qualified"] is False
    owner = flow["owner"]
    receipt = read_json(owner.workspace / "completion.json")
    assert receipt["result"] == result
    assert owner.journal.status("collect")["status"] == "completed"
    assert flow["events"].index("local.transport-fixture") < flow["events"].index("remote.wait.transport-fixture")


@pytest.mark.parametrize("stage", [stage for stage in HEALTHY_ORDER if stage != "local.context"])
def test_every_external_stage_failure_aborts_and_cleans_only_owned_effects(flow, stage):
    flow["fail"] = stage
    with pytest.raises(ArtifactError, match="injected boundary failure"):
        workflow.run_setup(flow["settings"], invitation=flow["invitation"])
    events = flow["events"]
    failure = events.index(stage)
    assert events[:failure + 1] == HEALTHY_ORDER[:HEALTHY_ORDER.index(stage) + 1]
    assert "remote.call.abort" in events[failure + 1:]
    assert all(item.startswith("cleanup.") or item == "remote.call.abort" for item in events[failure + 1:])
    assert not (flow["owner"].workspace / "completion.json").exists()
    assert not flow["fixture"] and not flow["launched"]
    if "local.launch" not in events[:failure]:
        assert not any(item.startswith("cleanup.model.") for item in events)
    else:
        assert "cleanup.model.stop" in events and "cleanup.model.rm" in events


def fresh_recording_controller(flow):
    previous = flow["owner"]
    owner = type(previous)(flow["settings"], invitation=flow["invitation"], runner=previous.base_runner)
    owner._owned_service = MethodType(previous._owned_service.__func__, owner)
    flow["owner"] = owner


def restart_boundaries(flow, monkeypatch):
    """Record pure package/build/host boundaries; none qualify this toy job."""
    workflow.run_setup(flow["settings"], invitation=flow["invitation"])
    fresh_recording_controller(flow)
    flow["events"].clear()
    host_path = Path(flow["owner"].paths["host_report"])
    atomic_json(host_path, seal({"classification": "unqualified current-host boundary"}))
    atomic_json(flow["owner"].paths["build_info"], flow["builds"]["client"])
    monkeypatch.setattr(workflow, "discover_local", lambda *args, **kwargs: pytest.fail("completed job rediscovered"))
    from ria import host
    monkeypatch.setattr(workflow, "validate_controller_context", lambda *args, **kwargs: flow["record"]("restart.context"))
    def package(request, directory):
        assert request == flow["owner"].request and directory == flow["owner"].paths["final"]
        flow["record"]("restart.package")
    monkeypatch.setattr(workflow.deployment, "verify_deployment_files", package)
    original_read = workflow.read_json
    def read(path, **kwargs):
        if str(path) == "/usr/share/dwarfstar/build-info.json":
            flow["record"]("restart.build")
            return flow.get("current_build", flow["builds"]["server"])
        return original_read(path, **kwargs)
    monkeypatch.setattr(workflow, "read_json", read)
    monkeypatch.setattr(host, "revalidate_host_report", lambda value: flow["record"]("restart.host"))
    monkeypatch.setattr(host, "verify_container_ancestors", lambda pid, value: flow["record"]("restart.ancestors"))


def test_completed_job_checks_current_service_and_host_instead_of_receipt(flow, monkeypatch):
    restart_boundaries(flow, monkeypatch)
    result = workflow.run_setup(flow["settings"], invitation=flow["invitation"])
    assert result["setup_previously_completed"] is True
    assert flow["events"] == ["restart.context", "restart.context", "restart.package", "restart.build", "restart.host", "restart.ancestors", "local.health"]
    assert result["final_release_qualified"] is False


@pytest.mark.parametrize("stage", ["restart.package", "restart.build", "restart.host", "restart.ancestors", "local.health"])
def test_completed_job_current_boundary_failure_cannot_return_old_health(flow, monkeypatch, stage):
    restart_boundaries(flow, monkeypatch)
    old_receipt = (flow["owner"].workspace / "completion.json").read_bytes()
    flow["fail"] = stage
    with pytest.raises(ArtifactError, match="injected boundary failure"):
        workflow.run_setup(flow["settings"], invitation=flow["invitation"])
    assert flow["triggered"] and not flow["launched"]
    assert (flow["owner"].workspace / "completion.json").read_bytes() == old_receipt
    assert "remote.call.abort" not in flow["events"]
    assert "cleanup.model.stop" in flow["events"] and "cleanup.model.rm" in flow["events"]
    if stage != "local.health":
        assert "local.health" not in flow["events"]


@pytest.mark.parametrize("field", ["sources", "source_lock_digest"])
def test_completed_job_source_publication_change_fails_before_current_health(flow, monkeypatch, field):
    restart_boundaries(flow, monkeypatch)
    change = {"synthetic-other-publication": "9" * 64} if field == "sources" else "9" * 64
    flow["current_build"] = seal({**flow["builds"]["server"], field: change})
    with pytest.raises(ArtifactError, match="another setup/source publication"):
        workflow.run_setup(flow["settings"], invitation=flow["invitation"])
    assert "restart.package" in flow["events"] and "restart.build" in flow["events"]
    assert "restart.host" not in flow["events"] and "local.health" not in flow["events"]
    assert not flow["launched"]


@pytest.mark.parametrize("condition", ["replacement", "missing", "dangling", "journal-replacement"])
def test_owned_service_observation_rejects_same_config_replacement_id(flow, monkeypatch, condition):
    owner = flow["owner"]
    owner.initialize()
    owner.request = deepcopy(flow["request"])
    directory = artifacts.private_directory(owner.paths["final"])
    atomic_json(directory / "compose-effective.json", {"name": "ria-client"})
    receipt = Path(owner.paths["operator"]) / "service-owner.json"
    if condition == "replacement":
        atomic_json(receipt, {"container_id": "a" * 64, "image": owner.settings["service_image"]})
    elif condition == "dangling":
        receipt.symlink_to(owner.workspace / "missing-receipt")
    elif condition == "journal-replacement":
        owner.step("launch", lambda: {"container_id": "a" * 64})
    container_id = ("b" if condition in ("replacement", "journal-replacement") else "a") * 64
    observed = {"Id": container_id, "Config": {"Image": owner.settings["service_image"], "Labels": {
        "com.docker.compose.project": "ria-client", "com.docker.compose.service": "client"}}}
    def runner(arguments, deadline, **kwargs):
        assert deadline > 0
        return container_id.encode() + b"\n" if "ps" in arguments else canonical(observed)
    owner.base_runner = runner
    from ria import container_inspection
    monkeypatch.setattr(container_inspection, "validate_container_inspection", lambda *args: None)
    # Exercise the genuine production method, bypassing only the independently
    # tested full inspect-shape validator, not its ID/receipt ownership check.
    with pytest.raises(ArtifactError, match="replacement|ownership|launch receipt"):
        type(owner)._owned_service(owner, require_owner=condition in ("missing", "dangling"))


@pytest.mark.parametrize("component", ["manifest", "shard", "tokenizer", "provenance"])
def test_imported_client_tamper_prevents_bootstrap_or_any_service_launch(flow, monkeypatch, component):
    original_verify = workflow.verify_client_import
    def tamper_then_verify(root, facts):
        root = Path(root)
        if component == "manifest":
            atomic_json(root / "manifest.json", seal({**facts["client_manifest"], "tokenizer_digest": "f" * 64}))
        else:
            path = (next((root / "tensors").glob("*.safetensors")) if component == "shard" else
                    root / (facts["tokenizer_file"][len("/model/"):] if component == "tokenizer" else "client-source.json"))
            data = path.read_bytes()
            path.write_bytes(data[:-1] + bytes((data[-1] ^ 1,)))
        return original_verify(root, facts)
    monkeypatch.setattr(workflow, "verify_client_import", tamper_then_verify)
    with pytest.raises(ArtifactError):
        workflow.run_setup(flow["settings"], invitation=flow["invitation"])
    assert "remote.fetch.client-model" in flow["events"] and "remote.call.abort" in flow["events"]
    assert "local.bootstrap" not in flow["events"] and "local.fixture-start" not in flow["events"]
    assert "local.launch" not in flow["events"] and not flow["launched"]
    assert not (flow["owner"].workspace / "completion.json").exists()


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo"])
def test_export_cannot_follow_runtime_report_symlink_to_host_secret(flow, kind):
    owner = flow["owner"]
    owner.initialize()
    reports = Path(artifacts.runtime_report_directory(owner.workspace))
    native = artifacts.private_directory(reports / "native-client")
    artifacts.private_directory(reports / "transport-client")
    outside = owner.workspace.parent / "outside-controller-secret"
    outside.write_bytes(b"private host bytes must never enter exported evidence")
    outside.chmod(0o600)
    if kind == "symlink":
        (native / "unexpected.json").symlink_to(outside)
    elif kind == "hardlink":
        os.link(outside, native / "unexpected.json")
    else:
        os.mkfifo(native / "unexpected.json")
    for stage in ("native-fixture", "transport-fixture"):
        owner.step(stage, lambda: {"classification": "unqualified workflow boundary"})
    with pytest.raises(ArtifactError, match="symlink|hard|unsafe|regular|special"):
        owner.export_runs()
    assert outside.read_bytes() == b"private host bytes must never enter exported evidence"
    copied = owner.workspace / "exports/runs/native-client/unexpected.json"
    assert not copied.exists()


def test_runtime_evidence_snapshot_caps_total_before_second_subtree_copy(flow):
    owner = flow["owner"]
    owner.initialize()
    owner.settings["max_transfer_bytes"] = 1024
    reports = Path(artifacts.runtime_report_directory(owner.workspace))
    for name in ("native-client", "transport-client"):
        artifacts.private_directory(reports / name)
        (reports / name / "bounded.bin").write_bytes(b"x" * 700)
    for stage in ("native-fixture", "transport-fixture"):
        owner.step(stage, lambda: {"classification": "unqualified workflow boundary"})
    with pytest.raises(ArtifactError, match="byte quota|byte|bound"):
        owner.export_runs()
    assert not (owner.workspace / "exports/runs/transport-client").exists()


def test_unfinished_controller_restart_fails_closed_after_quiescence(flow):
    flow["fail"] = "local.probe"
    with pytest.raises(ArtifactError):
        workflow.run_setup(flow["settings"], invitation=flow["invitation"])
    fresh_recording_controller(flow)
    flow["events"].clear()
    with pytest.raises(ArtifactError, match="unfinished previous setup"):
        workflow.run_setup(flow["settings"], invitation=flow["invitation"])
    assert not any(item in HEALTHY_ORDER and item != "local.context" for item in flow["events"])


def test_controller_authority_failure_precedes_locks_and_workspace_mutation(flow):
    flow["fail"] = "local.context"
    with pytest.raises(ArtifactError, match="boundary failure"):
        workflow.run_setup(flow["settings"], invitation=flow["invitation"])
    owner = flow["owner"]
    assert flow["events"] == ["local.context"]
    assert owner.journal is None
    assert not (owner.workspace / "settings.json").exists()
    assert not (owner.workspace / ".controller.lock").exists()
    assert not (owner.workspace / "pki").exists()


def test_initialization_security_mismatch_never_contacts_or_aborts_unclaimed_peer(flow):
    flow["owner"].settings["security"]["mode"] = "tls"
    with pytest.raises(ArtifactError, match="different transport security"):
        workflow.run_setup(flow["settings"], invitation=flow["invitation"])
    assert flow["events"] == ["local.context"]
    assert not flow["fixture"] and not flow["launched"]
    assert not (flow["owner"].workspace / "pki").exists()


def test_controller_end_to_end_deadline_and_cleanup_residuals(flow, monkeypatch):
    owner = flow["owner"]
    owner.deadline = time.monotonic() + .050
    owner._run(["docker", "--host", "unix:///var/run/docker.sock", "ps"], 30000)
    assert 0 < flow["command_deadlines"][-1] <= 50
    owner.deadline = time.monotonic() - 1
    with pytest.raises(ArtifactError, match="deadline"):
        owner._run(["docker", "--host", "unix:///var/run/docker.sock", "ps"], 30000)
    count = len(flow["command_deadlines"])
    owner.cleaning = True
    owner.cleanup_thread = threading.get_ident()
    owner._run(["docker", "--host", "unix:///var/run/docker.sock", "ps"], 1000)
    assert len(flow["command_deadlines"]) == count + 1 and flow["command_deadlines"][-1] == 1000


def test_cleanup_owner_bypass_never_admits_background_stage_runner(flow):
    owner = flow["owner"]
    owner.initialize()
    owner.quiesce()
    assert owner.cleanup_thread == threading.get_ident() and owner.cancelled.is_set()
    failures = []
    def background():
        try:
            owner._run(["docker", "--host", "unix:///var/run/docker.sock", "ps"], 1000)
        except ArtifactError as error:
            failures.append(str(error))
    thread = threading.Thread(target=background)
    thread.start()
    thread.join(2)
    assert not thread.is_alive() and failures and "cancelled" in failures[0]
    assert flow["command_deadlines"] == []
    owner._run(["docker", "--host", "unix:///var/run/docker.sock", "ps"], 1000)
    assert flow["command_deadlines"] == [1000]


def test_production_runner_cancels_and_reaps_running_child_while_cleanup_still_runs(flow, monkeypatch):
    from ria import process
    owner = flow["owner"]
    owner.initialize()
    owner.base_runner = workflow.deployment._run
    started, failures, children = owner.workspace / "owned-child-started", [], []
    original_popen = process.subprocess.Popen
    def popen(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(process.subprocess, "Popen", popen)
    def background():
        try:
            owner._run([sys.executable, "-c",
                "import pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text('started'); time.sleep(60)", str(started)], 30000)
        except ArtifactError as error:
            failures.append(str(error))
    thread = threading.Thread(target=background)
    thread.start()
    try:
        deadline = time.monotonic() + 2
        while not started.exists() and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(.005)
        assert started.exists()
        owner.quiesce()
        cleanup_output = owner._run([sys.executable, "-c", "print('cleanup owner only')"], 1000)
        assert cleanup_output == b"cleanup owner only\n"
    finally:
        owner.cancelled.set()
        thread.join(3)
    assert not thread.is_alive() and failures and "cancelled" in failures[0]
    assert len(children) == 2 and children[0].returncode == -signal.SIGKILL and children[1].returncode == 0


def test_cancelled_controller_never_publishes_completed_late_stage(flow):
    owner = flow["owner"]
    owner.initialize()
    def cancellation_during_stage():
        owner.cancelled.set()
        return {"not_completed": True}
    with pytest.raises(ArtifactError, match="cancelled"):
        owner.step("probe", cancellation_during_stage)
    assert owner.journal.status("probe")["status"] == "failed"


def test_actual_dispatch_abort_wakes_owner_and_rejects_later_operations(flow):
    owner = flow["owner"]
    owner.initialize()
    result = owner.dispatch("stop", {"abort": True})
    assert result == {"aborted": True}
    assert owner.finished.is_set() and owner.cancelled.is_set()
    with pytest.raises(ArtifactError, match="aborting"):
        owner.dispatch("offer", {})


@pytest.mark.parametrize("field", ["client_manifest_digest", "client_layout_digest", "selected_tensors"])
def test_actual_dispatch_rejects_resealed_alternate_extraction_before_grants(flow, field):
    owner = flow["owner"]
    owner.initialize()
    owner.model = flow["facts"]
    owner.step("prepare", lambda: {"fixture": "prepared without hardware"})
    facts = seal({**owner.model, field: ["layers.0.ffn.experts.1.w1.weight"] if field == "selected_tensors" else "f" * 64})
    with pytest.raises(ArtifactError, match="exact prepared compact extraction"):
        owner.dispatch("client-offer", {"facts": facts})
    assert not Path(owner.paths["grants"]).exists()
    assert owner.journal.status("client-offer")["status"] == "failed"


def test_prepare_receipt_waits_for_verified_model_and_export_publication(flow, monkeypatch):
    owner = flow["owner"]
    owner.initialize()
    owner.host = {"build_info": flow["builds"]["server"]}
    owner.step("discover", lambda: owner.host)
    prepared_facts = deepcopy(flow["facts"])
    entered, release = threading.Event(), threading.Event()
    results, failures = [], []
    def recorded_preparation():
        # The small genuine package was already prepared by this fixture. Only
        # external preparation execution is replaced; publication stays real.
        workflow._publish_once(owner.workspace / "operator" / "model-facts.json", prepared_facts)
        return {"model_facts_digest": prepared_facts["digest"]}
    monkeypatch.setattr(owner, "_prepare_model", recorded_preparation)
    class RecordedExportBoundary:
        def __init__(self):
            self.exports = {}
        def export_tree(self, name, path, *, max_bytes, check):
            assert name == "client-model" and path == prepared_facts["export_root"]
            entered.set()
            released = release.wait(2)
            if not released:
                raise ArtifactError("fixture export barrier was not released")
            # Real manifest and byte hashing, with no network worker started.
            tree = workflow.export_tree(name, path, max_bytes=max_bytes, check=check)
            self.exports[name] = tree
            return tree.manifest
    owner.server = RecordedExportBoundary()
    def commit():
        try:
            results.append(owner.dispatch("commit-plan", {"local_experts": []}))
        except BaseException as error:
            failures.append(error)
    thread = threading.Thread(target=commit)
    thread.start()
    try:
        arrived = entered.wait(2)
        assert arrived
        assert owner.journal.status("prepare")["status"] == "running"
        assert not owner.server.exports
        with pytest.raises(ArtifactError, match="requires completed prepare"):
            owner.dispatch("offer", {})
        with pytest.raises(ArtifactError, match="requires completed prepare"):
            owner.dispatch("client-offer", {"facts": prepared_facts})
        assert owner.journal.status("client-offer")["status"] == "unknown"
    finally:
        release.set()
        thread.join(2)
    assert not thread.is_alive() and not failures
    assert results == [{"prepared": True, "client_manifest_digest": prepared_facts["client_manifest_digest"]}]
    assert owner.journal.status("prepare")["result"] == {"model_facts_digest": prepared_facts["digest"]}
    assert owner.journal.status("prepare")["status"] == "completed"
    offer = owner.dispatch("offer", {})
    assert offer["model"] == prepared_facts and offer["host"] == owner.host
    assert set(owner.server.exports) == {"client-model"}
    exported = owner.server.exports["client-model"]
    verify_identity(exported.manifest)
    files = {item["path"]: item for item in exported.manifest["files"]}
    manifest_bytes = (Path(prepared_facts["export_root"]) / "manifest.json").read_bytes()
    assert exported.chunk("manifest.json", 0, len(manifest_bytes)) == manifest_bytes
    assert files["manifest.json"]["bytes"] == len(manifest_bytes)


def test_actual_dispatch_lost_reply_input_binding_rejects_regrant(flow, monkeypatch):
    owner = flow["owner"]
    owner.initialize()
    owner.model = flow["facts"]
    owner.host = {"build_info": flow["builds"]["server"]}
    owner.step("prepare", lambda: {"fixture": "prepared without hardware"})
    monkeypatch.setattr(workflow, "validate_settings", lambda value: value)
    owner.settings["peer_address"] = "192.168.10.2"
    owner.settings["server_executor"] = "cpu"
    owner.security = SetupSecurity(owner.workspace / "pki", flow["invitation"], "expert")
    plan = artifacts.make_placement(owner.settings, owner.model["client_manifest"], owner.model, owner.workspace)
    facts = seal({**owner.model, "placement_plan_digest": plan["digest"], "environment": flow["environment"],
                  "build_info": flow["builds"]["client"], "bind_address": "192.168.10.2", "certificate_sha256": None})
    first = owner.dispatch("client-offer", {"facts": facts})
    assert owner.dispatch("client-offer", {"facts": facts}) == first
    changed = seal({**facts, "placement_plan_digest": "f" * 64})
    with pytest.raises(ArtifactError, match="different inputs"):
        owner.dispatch("client-offer", {"facts": changed})
    assert flow["events"].count("local.bootstrap") == 1


def test_health_uses_remaining_startup_time_for_admin_command(flow):
    owner = flow["owner"]
    owner.initialize()
    owner.step("launch", lambda: {"container_id": "a" * 64})
    flow["launched"] = True
    owner.deadline = time.monotonic() + .025
    result = owner.health()
    assert result["healthy"] is True
    assert 0 < flow["command_deadlines"][-1] <= 25


def test_abort_during_running_stage_wakes_and_preserves_failed_receipt(flow):
    owner = flow["owner"]
    owner.initialize()
    entered, release = threading.Event(), threading.Event()
    failures = []
    def work():
        entered.set()
        released = release.wait(2)
        assert released
        return {"late_result": "cannot become completed evidence"}
    def run():
        try:
            owner.step("prepare", work)
        except ArtifactError as error:
            failures.append(str(error))
    thread = threading.Thread(target=run)
    thread.start()
    try:
        arrived = entered.wait(2)
        assert arrived
        owner.dispatch("stop", {"abort": True})
    finally:
        release.set()
        thread.join(2)
    assert not thread.is_alive() and failures and "cancelled" in failures[0]
    assert owner.journal.status("prepare")["status"] == "failed"


@pytest.mark.parametrize("stage", ["allow-import", "start"])
def test_expert_startup_failure_closes_owner_and_preserves_primary_cleanup_context(flow, monkeypatch, stage):
    owner = flow["owner"]
    owner.initialize()
    calls = []
    class FailedStartupPeer:
        def __init__(self, *args, **kwargs):
            self.tasks, self.thread = {}, None
        def allow_import(self, *args, **kwargs):
            calls.append("allow-import")
            if stage == "allow-import":
                raise ArtifactError("primary startup failure")
        def start(self):
            calls.append("start")
            raise ArtifactError("primary startup failure")
        def begin_close(self):
            calls.append("begin-close")
            raise ArtifactError("begin-close secondary failure")
        def close(self, **kwargs):
            calls.append("close")
            raise ArtifactError("close secondary failure")
    monkeypatch.setattr(workflow, "PeerServer", FailedStartupPeer)
    with pytest.raises(ArtifactError) as failure:
        owner.run_expert()
    assert all(message in str(failure.value) for message in (
        "primary startup failure", "begin-close secondary failure", "close secondary failure"))
    assert calls == (["allow-import"] if stage == "allow-import" else ["allow-import", "start"]) + ["begin-close", "close"]
    assert owner.cancelled.is_set() and owner.cleaning and owner.cleanup_thread == threading.get_ident()


def test_sigterm_quiesces_running_workflow_in_isolated_process(flow):
    if os.environ.get("RIA_WORKFLOW_SIGTERM_CHILD") != "1":
        environment = dict(os.environ)
        environment["RIA_WORKFLOW_SIGTERM_CHILD"] = "1"
        result = subprocess.run([sys.executable, "-m", "pytest", "-q",
            str(Path(__file__).resolve()) + "::test_sigterm_quiesces_running_workflow_in_isolated_process"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment,
            cwd=Path(__file__).resolve().parents[2], timeout=20, check=False)
        assert result.returncode == 0, result.stdout[-4096:] + result.stderr[-4096:]
        return
    previous = signal.getsignal(signal.SIGTERM)
    flow["signal"] = "local.probe"
    with pytest.raises(workflow.SetupTerminated, match="SIGTERM"):
        workflow.run_setup(flow["settings"], invitation=flow["invitation"])
    owner = flow["owner"]
    assert owner.cancelled.is_set() and not flow["fixture"] and not flow["launched"]
    assert owner.journal.status("probe")["status"] == "failed"
    assert "cleanup.fixture" in flow["events"] and "remote.call.abort" in flow["events"]
    assert signal.getsignal(signal.SIGTERM) == previous
    assert not (owner.workspace / "completion.json").exists()


@pytest.mark.parametrize("stderr_failure,cleanup_failure,survivor", [
    pytest.param("none", None, "broker", id="stderr-ok"),
    pytest.param("write", None, "broker", id="stderr-write-failure"),
    pytest.param("flush", None, "broker", id="stderr-flush-failure"),
    pytest.param("none", "begin-close", "broker", id="unexpected-begin-close-failure"),
    pytest.param("none", "quiesce", "broker", id="unexpected-quiesce-failure"),
    pytest.param("none", "close", "broker", id="unexpected-close-failure"),
    pytest.param("none", None, "process", id="recorded-process-only-survivor"),
    pytest.param("none", None, "task", id="task-only-stable-journal-id"),
    pytest.param("none", None, "contended-task", id="bounded-task-ownership-contention"),
])
def test_live_worker_fatal_containment_precedes_lease_unwind_in_isolated_process(flow, monkeypatch, stderr_failure, cleanup_failure, survivor, request):
    if os.environ.get("RIA_WORKFLOW_FATAL_CHILD") != "1":
        child_base = artifacts.private_directory(flow["owner"].workspace.parent / "fatal-child-fixtures")
        environment = dict(os.environ)
        environment["RIA_WORKFLOW_FATAL_CHILD"] = "1"
        result = subprocess.run([sys.executable, "-m", "pytest", "-q", "--basetemp", str(child_base),
            str(Path(__file__).resolve()) + "::test_live_worker_fatal_containment_precedes_lease_unwind_in_isolated_process[" + request.node.callspec.id + "]"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment,
            cwd=Path(__file__).resolve().parents[2], timeout=20, check=False)
        assert result.returncode == 125, result.stdout[-4096:] + result.stderr[-4096:]
        receipts = list(child_base.rglob("fatal-containment.json"))
        assert len(receipts) == 1
        info = receipts[0].stat()
        assert info.st_uid == os.geteuid() and info.st_mode & 0o777 == 0o600
        value = read_json(receipts[0])
        verify_identity(value)
        assert value["completion_ambiguous"] is True and value["owned_cleanup_complete"] is False
        assert value["broker_alive"] is (survivor == "broker")
        assert value["worker_alive"] is (survivor == "process")
        assert value["ownership_observation_failed"] is (survivor == "contended-task")
        assert value["surviving_task_ids"] == (["native-fixture"] if survivor == "task" else [])
        if survivor == "contended-task":
            assert any("bounded lock" in message for message in value["shutdown_failures"])
        if cleanup_failure:
            assert any("RuntimeError:" in message for message in value["shutdown_failures"])
        job = receipts[0].parent
        assert (job / ("recorded-process-started" if survivor == "process" else "owned-thread-started")).exists()
        assert not any((job / name).exists() for name in ("completion.json", "normal-lease-unwind", "late-worker-mutation"))
        # The process has terminated, so its Python worker cannot use this
        # released lease. Reacquisition succeeds only after that containment.
        with workflow.workspace_lock(job):
            assert not (job / "late-worker-mutation").exists()
        return
    owner = flow["owner"]
    original_lock = workflow.workspace_lock
    @contextmanager
    def observed_lock(path):
        with original_lock(path):
            try:
                yield
            finally:
                (Path(path) / "normal-lease-unwind").write_bytes(b"a live worker must prevent this path")
    monkeypatch.setattr(workflow, "workspace_lock", observed_lock)
    if cleanup_failure == "quiesce":
        def failed_quiesce():
            raise RuntimeError("synthetic unexpected quiesce failure")
        monkeypatch.setattr(owner, "quiesce", failed_quiesce)
    if stderr_failure != "none":
        class FailedFatalDiagnostics:
            def __init__(self, sink):
                self.sink, self.fatal = sink, False
            def write(self, value):
                self.fatal = self.fatal or "fatal containment" in value
                if self.fatal and stderr_failure == "write":
                    raise OSError("synthetic closed fatal diagnostic pipe")
                return self.sink.write(value)
            def flush(self):
                if self.fatal and stderr_failure == "flush":
                    raise OSError("synthetic broken fatal diagnostic flush")
                return self.sink.flush()
            def __getattr__(self, name):
                return getattr(self.sink, name)
        monkeypatch.setattr(workflow.sys, "stderr", FailedFatalDiagnostics(sys.stderr))
    class StuckPeer:
        def __init__(self, *args, **kwargs):
            self.tasks, self.thread, self.process = {}, None, None
            self.mutex = threading.Lock()
        def allow_import(self, *args, **kwargs):
            pass
        def start(self):
            if survivor == "process":
                # An explicitly unqualified recorded process boundary checks
                # process-only containment without leaving a real grandchild.
                class RecordedRunningProcess:
                    def poll(self):
                        return None
                self.process = RecordedRunningProcess()
                (owner.workspace / "recorded-process-started").write_bytes(b"recorded live")
                owner.finished.set()
                return
            entered = threading.Event()
            def mutate_after_gate():
                (owner.workspace / "owned-thread-started").write_bytes(b"started")
                entered.set()
                time.sleep(4 if survivor == "contended-task" else 2)
                (owner.workspace / "late-worker-mutation").write_bytes(b"must never happen")
            def worker():
                if survivor == "contended-task":
                    # Own the actual mutex beyond the controller's 1s snapshot
                    # budget; unknown task ownership must remain process-fatal.
                    with self.mutex:
                        mutate_after_gate()
                else:
                    mutate_after_gate()
            task = threading.Thread(target=worker, name="not-the-fixed-journal-step-id")
            if survivor == "broker":
                self.thread = task
            else:
                self.tasks["native-fixture"] = task
            task.start()
            arrived = entered.wait(1)
            assert arrived
            if cleanup_failure == "quiesce":
                owner.aborted = True
            owner.finished.set()
        def begin_close(self):
            if cleanup_failure == "begin-close":
                raise RuntimeError("synthetic unexpected begin-close failure")
        def close(self, **kwargs):
            error = RuntimeError if cleanup_failure == "close" else ArtifactError
            raise error("synthetic live owned worker exceeded its cancellation bound")
    monkeypatch.setattr(workflow, "PeerServer", StuckPeer)
    owner.run_client = MethodType(lambda self: self.run_expert(), owner)
    workflow.run_setup(flow["settings"], invitation=flow["invitation"])
    pytest.fail("a surviving worker returned through the controller lease")
