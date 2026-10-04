"""Local policy owner for the paired initial-admission setup workflow.

The peer chooses only fixed stage names. Settings, host paths, commands and
resource/numerical policy are local immutable inputs. Production services run
independently after setup completes. Interrupted measurements are never resumed.
"""

from contextlib import contextmanager
import fcntl
import hashlib
import os
from pathlib import Path
import re
import secrets
import signal
import ssl
import stat
import sys
import threading
import time

from . import deployment
from .fixture_runner import derive_bootstrap, produce_components, validate_registration
from .identity import (ArtifactError, atomic_bytes, canonical, digest,
                       loads, read_json, seal, verify_identity)
from .inventory import build_inventory
from .setup_artifacts import (make_grants, make_placement,
    private_directory, provision_api_token, runtime_readable, runtime_reports,
    runtime_secret, runtime_secret_directory, runtime_report_directory, verify_client_import)
from .setup_config import build_request, validate_settings, workspace_paths
from .setup_host import discover_local, validate_controller_context
from .setup_journal import SetupJournal
from .setup_peer import PeerClient, PeerServer, export_tree, import_tree
from .setup_qualification import register, validate_tuning
from .setup_security import SetupSecurity, endpoint, read_invitation, write_invitation
from .process import run_bounded
from .schemas import validate
from .setup_limits import MAX_MESSAGE, MAX_NODES


@contextmanager
def workspace_lock(workspace):
    """Exclude other controllers before inspecting or mutating any job state."""
    workspace = private_directory(workspace)
    fd = os.open(workspace / ".controller.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise ArtifactError("unsafe setup workspace lock")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ArtifactError("another controller owns this setup workspace") from error
        yield
    finally:
        os.close(fd)


@contextmanager
def role_lock(role, *, directory=Path("/run/lock")):
    """Serialize setup jobs across workspaces without nesting deployment locks."""
    if role not in ("expert", "client"):
        raise ArtifactError("invalid setup role")
    fd = os.open(Path(directory) / ("ria-" + role + ".setup.lock"), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise ArtifactError("unsafe setup role lock")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ArtifactError("another setup job owns this host role") from error
        yield
    finally:
        os.close(fd)


def _publish_once(path, value):
    if Path(path).exists():
        if read_json(path) != value:
            raise ArtifactError("immutable setup inputs changed; use a new workspace")
    else:
        atomic_bytes(path, canonical(value) + b"\n", mode=0o600)
    return value


class SetupController:
    def __init__(self, settings, *, tuning=None, invitation=None, runner=deployment._run):
        self.settings = validate_settings(settings)
        self.settings_digest = digest(self.settings)
        self.workspace = Path(self.settings["workspace"])
        self.paths = workspace_paths(self.workspace)
        self.base_runner = runner
        self.runner = self._run
        self.deadline = time.monotonic() + self.settings["setup_deadline_ms"] / 1000
        self.cleaning = False
        self.cleanup_thread = None
        self.tuning = validate_tuning(tuning) if tuning is not None else None
        self.invitation = invitation
        self.journal = self.security = self.server = self.request = None
        self.host = self.model = None
        self.finished = threading.Event()
        self.cancelled = threading.Event()
        self.aborted = False
        self.stage_lock = threading.RLock()
        self.completed = False
        self.peer_participating = False

    def remaining_ms(self):
        remaining = int((self.deadline - time.monotonic()) * 1000)
        if self.cancelled.is_set() or remaining <= 0:
            raise ArtifactError("setup cancelled or its end-to-end deadline elapsed")
        return remaining

    def _run(self, arguments, deadline_ms, *, cwd=None):
        cleanup = self.cleaning and self.cleanup_thread == threading.get_ident()
        if not cleanup:
            deadline_ms = min(deadline_ms, self.remaining_ms())
        if self.base_runner is deployment._run:
            return self.base_runner(arguments, deadline_ms, cwd=cwd,
                cancelled=None if cleanup else self.cancelled.is_set)
        return self.base_runner(arguments, deadline_ms, cwd=cwd)

    def peer_call(self, peer, operation, payload):
        return peer.call(operation, payload, timeout=min(self.settings["deadline_ms"], self.remaining_ms()) / 1000)

    def peer_submit(self, peer, name, payload):
        return self.peer_call(peer, "submit", {"step_id": name, "name": name, "payload": payload})

    def initialize(self):
        """Called only with workspace_lock held for the entire controller life."""
        _publish_once(self.workspace / "settings.json", self.settings)
        self.journal = SetupJournal(self.workspace / "local-journal")
        self._load_request()
        completion = self.workspace / "completion.json"
        if completion.exists():
            receipt = read_json(completion)
            verify_identity(receipt)
            if receipt.get("settings_digest") != self.settings_digest:
                raise ArtifactError("completed setup has different settings")
            self.completed = True
        elif self.journal.records:
            self._load_request()
            self.quiesce()
            raise ArtifactError("unfinished previous setup stopped; preserve evidence and start a new job/workspace")
        for name in ("operator", "imports", "exports"):
            private_directory(self.workspace / name)
        role = self.settings["role"]
        path = self.workspace / "invitation.json"
        if role == "expert":
            if self.tuning is None:
                raise ArtifactError("expert setup requires accepted qualification policy and fixture limits")
            _publish_once(self.workspace / "qualification-input.json", self.tuning)
            if path.exists():
                self.invitation = read_invitation(path)
                self.security = SetupSecurity(self.workspace / "pki", self.invitation, "expert")
            else:
                self.security, self.invitation = SetupSecurity.create_expert(self.workspace / "pki", secrets.token_hex(32),
                    f"{self.settings['bind_address']}:{self.settings['setup_port']}",
                    tls_enabled=self.settings["security"]["mode"] == "tls")
                write_invitation(path, self.invitation)
        else:
            if self.invitation is None:
                raise ArtifactError("client setup requires its trusted invitation file")
            if self.invitation["tls_enabled"] != (self.settings["security"]["mode"] == "tls"):
                raise ArtifactError("invitation and local settings select different transport security")
            address, _ = endpoint(self.invitation["endpoint"])
            if self.settings.get("peer_address", address) != address:
                raise ArtifactError("configured expert address differs from trusted invitation")
            self.settings["peer_address"] = address
            if path.exists():
                if read_invitation(path) != self.invitation:
                    raise ArtifactError("workspace has another setup invitation")
                self.security = SetupSecurity(self.workspace / "pki", self.invitation, "client")
            else:
                self.security = SetupSecurity.create_client(self.workspace / "pki", self.invitation)
                write_invitation(path, self.invitation)

    def step(self, name, function, *, inputs=None):
        with self.stage_lock:
            self.remaining_ms()
            record, created = self.journal.accept(name, name, {"settings_digest": self.settings_digest,
                "job_id": self.invitation["job_id"], "input_digest": digest(inputs) if inputs is not None else None})
            if not created:
                if record["status"] != "completed":
                    raise ArtifactError(f"setup stage {name} previously failed; new job required")
                return record["result"]
            self.journal.transition(name, "running")
            print(f"RIA setup {self.settings['role']}: {name}", file=sys.stderr, flush=True)
            try:
                result = function()
                self.remaining_ms()
                self.journal.transition(name, "completed", result=result)
                return result
            except BaseException as error:
                try:
                    self.journal.transition(name, "failed", error=error)
                except (ArtifactError, OSError) as receipt_error:
                    raise ArtifactError(f"setup stage {name} failed: {error}; failure receipt could not be written: {receipt_error}") from error
                raise

    def require(self, name):
        if self.journal.status(name)["status"] != "completed":
            raise ArtifactError(f"setup stage requires completed {name}")

    def prepare(self):
        self.host = self.step("discover", lambda: discover_local(self.settings, self.workspace, runner=self.runner))
        return self.host

    def _prepare_model(self):
        result = run_bounded([sys.executable, str(Path(__file__).resolve().parents[1] / "setup_prepare.py"),
            "--settings", str(self.workspace / "settings.json"), "--selection",
            str(self.workspace / "operator" / "client-selection.json")],
            timeout=self.remaining_ms() / 1000, max_stdout=MAX_MESSAGE, max_stderr=65536,
            cancelled=self.cancelled.is_set, env=deployment.controlled_environment())
        if result.returncode:
            raise ArtifactError("preparation failed: " + result.stderr[:4096].decode("utf-8", errors="replace"))
        facts = loads(result.stdout, max_bytes=MAX_MESSAGE, max_nodes=MAX_NODES)
        from .setup_artifacts import verify_model_facts
        verify_model_facts(facts)
        _publish_once(self.workspace / "operator" / "model-facts.json", facts)
        return {"model_facts_digest": facts["digest"]}

    def _service_security(self):
        root = Path(runtime_secret_directory(self.workspace))
        if self.security.tls_enabled:
            for name in ("ca.pem", "peer.pem", "peer.key"):
                destination = root / name
                raw = (self.security.directory / name).read_bytes()
                if destination.exists():
                    if destination.read_bytes() != raw:
                        raise ArtifactError("runtime credential changed")
                else:
                    atomic_bytes(destination, raw, mode=0o600)
                    runtime_secret(destination)
            tls = {"ca_file": "/run/secrets/ca.pem", "certificate_file": "/run/secrets/peer.pem",
                "private_key_file": "/run/secrets/peer.key", "minimum_version": "TLS1.3", "early_data": False,
                "expected_peer_name": self.security.peer_name("client" if self.settings["role"] == "expert" else "expert")}
        else:
            tls = {"enabled": False}
        # Only the exported runtime secrets tree is mounted in inference.
        # The CA signing key stays in the private setup PKI directory.
        return {"tls": tls, "expected_peer_name": self.security.peer_name("client")}

    def _certificate_digest(self):
        if not self.security.tls_enabled:
            return None
        pem = (self.security.directory / "peer.pem").read_text(encoding="ascii")
        return hashlib.sha256(ssl.PEM_cert_to_DER_cert(pem)).hexdigest()

    def _load_request(self):
        path = self.workspace / "operator" / "request.json"
        if path.exists():
            self.request = read_json(path)

    def _check_peer_build(self, build, executor):
        verify_identity(build)
        local = self.host["build_info"]
        if build.get("image_kind") != executor or build.get("source_lock_digest") != local["source_lock_digest"] or (
                build.get("sources") != local["sources"]):
            raise ArtifactError("peer image has different implementation sources or executor")

    def _bootstrap(self, manifest, policy_path):
        runtime_report_directory(self.workspace)
        self.request = build_request(self.settings, self.workspace, manifest, self.host,
            self._service_security(), str(policy_path), model_root=(
                self.model["server_root"] if self.settings["role"] == "expert" else self.paths["model"]))
        _publish_once(self.workspace / "operator" / "request.json", self.request)
        if not Path(self.paths["bootstrap"]).exists():
            deployment.bootstrap(self.request, self.paths["bootstrap"], runner=self.runner)
            runtime_readable(self.paths["bootstrap"])
        environment = read_json(Path(self.paths["bootstrap"]) / "environment.json")
        verify_identity(environment, digest(deployment.frozen_environment(self.request)))
        return environment

    def client_offer(self, peer, model):
        placement = make_placement(self.settings, model, self.model, self.workspace)
        _publish_once(self.paths["placement"], placement)
        self.api_token_path = provision_api_token(self.workspace, self.settings["security"]["api_token_file"])
        environment = self._bootstrap(model, Path(self.paths["placement"]))
        facts = seal({**{key: value for key, value in self.model.items() if key != "digest"},
            "kind": "setup_client_facts", "client_manifest": model,
            "client_layout_digest": model["layout_digest"], "placement_plan_digest": placement["digest"],
            "bind_address": self.settings["bind_address"], "build_info": self.host["build_info"],
            "environment": environment, "certificate_sha256": self._certificate_digest()})
        _publish_once(self.workspace / "operator" / "client-facts.json", facts)
        self.peer_submit(peer, "client-offer", {"facts": facts})
        return peer.wait("client-offer", timeout=self.remaining_ms() / 1000)

    def install_registration(self, result):
        policy, registration = result["policy"], result["registration"]
        validate_registration(registration, policy)
        role = "server" if self.settings["role"] == "expert" else "client"
        environment = read_json(Path(self.paths["bootstrap"]) / "environment.json")
        realization = registration["realizations"][role]
        if realization != {"environment_digest": environment["digest"], "build_digest": self.host["build_info"]["digest"]}:
            raise ArtifactError("peer registration changes the local frozen realization")
        planning = self.request["planning_request"]
        for prefix in ("native_", "transport_"):
            limits = registration["runs"][prefix + role]["hard_limits"]
            if int(limits["max_rss_bytes"]) > self.request["environment"]["cgroup_bytes"] or any(
                int(limits[key]) > planning["caps"][key] for key in ("host_bytes", "device_bytes", "pinned_bytes")):
                raise ArtifactError("peer fixture limits exceed local declared caps")
        reports = Path(self.paths["reports"])
        if not reports.exists():
            private_directory(reports)
        runtime_reports(reports)
        inputs = reports / "inputs"
        inputs.mkdir(mode=0o700, exist_ok=True)
        for name, document in (("policy.json", policy), ("registration.json", registration),
                ("transport_" + role + "-bootstrap.json", derive_bootstrap(registration, "transport_" + role))):
            _publish_once(inputs / name, document)
        runtime_readable(inputs)
        _publish_once(self.workspace / "operator" / "registration-bundle.json", result)
        return {"registration_digest": registration["digest"]}

    def fixture(self, operation):
        self.require("registration")
        if operation == "fixture-start":
            def start():
                self._reject_existing_project()
                return deployment.fixture_start(self.request, self.paths["bootstrap"], runner=self.runner)
            return self.step(operation, start)
        self.require("fixture-start")
        if operation != "probe":
            self.require("probe")
        name = operation if operation == "probe" else ("native_" if operation == "native-fixture" else "transport_") + (
            "server" if self.settings["role"] == "expert" else "client")
        def execute():
            deployment.fixture_exec(self.request, self.paths["bootstrap"], name, runner=self.runner)
            if name == "probe":
                result = read_json(Path(self.paths["reports"]) / "probe.json")
                verify_identity(result)
                return {"probe_digest": result["digest"]}
            root = Path(self.paths["reports"]) / name.replace("_", "-")
            raw, supervision = read_json(root / "measurements.json"), read_json(root / "supervision.json")
            verify_identity(raw)
            verify_identity(supervision)
            return {"raw_digest": raw["digest"], "supervision_digest": supervision["digest"]}
        return self.step(operation, execute)

    def export_runs(self):
        self.require("native-fixture")
        self.require("transport-fixture")
        destination = self.workspace / "exports" / "runs"
        if not destination.exists():
            destination.mkdir(mode=0o700)
            suffix = "server" if self.settings["role"] == "expert" else "client"
            remaining = self.settings["max_transfer_bytes"]
            for prefix in ("native-", "transport-"):
                name = prefix + suffix
                source = export_tree(name, Path(self.paths["reports"]) / name,
                    max_bytes=remaining, check=self.remaining_ms, reject_hardlinks=True)
                import_tree(source.manifest, destination / name, source.chunk,
                    max_bytes=remaining, check=self.remaining_ms)
                remaining -= source.manifest["total_bytes"]
        return str(destination)

    def finalize(self):
        self.require("collect")
        result = read_json(self.workspace / "operator" / "registration-bundle.json")
        role = self.settings["role"]
        suffix = "server" if role == "expert" else "client"
        peer_suffix = "client" if role == "expert" else "server"
        files = {prefix + "_" + suffix: Path(self.paths["reports"]) / (prefix + "-" + suffix) for prefix in ("native", "transport")}
        files.update({prefix + "_" + peer_suffix: self.workspace / "imports" / "peer-runs" / (prefix + "-" + peer_suffix)
            for prefix in ("native", "transport")})
        calibration = self.workspace / "operator" / "calibration"
        produce_components(result["registration"], result["policy"], files, calibration, role=role)
        inventory = build_inventory(self.request, Path(self.request["environment"]["model_dir"]) / "manifest.json",
            self.workspace / "operator" / "inventory.json", runner=self.runner)
        reports = Path(self.paths["reports"])
        lock = deployment.finalize(self.request, read_json(reports / "probe.json"), inventory,
            read_json(calibration / "calibration.json"), self.paths["final"],
            probe_evidence=read_json(reports / "probe.json.details.json"),
            calibration_evidence=read_json(calibration / "calibration-evidence.json"),
            calibration_evidence_dir=calibration, policy=result["policy"], runner=self.runner)
        runtime_readable(self.paths["final"])
        return {"deployment_lock_digest": lock["digest"], "qualification_scope": "initial_fixture", "final_release_qualified": False}

    def _owned_service(self, *, allow_absent=False, deadline_ms=None, require_owner=False):
        role = self.settings["role"]
        directory = Path(self.paths["final"])
        deadline = time.monotonic() + (deadline_ms or self.settings["deadline_ms"]) / 1000
        def remaining():
            result = int((deadline - time.monotonic()) * 1000)
            if result <= 0:
                raise ArtifactError("owned container observation deadline elapsed")
            return result
        raw = self.runner(deployment.compose_arguments(self.request, directory, directory / (role + ".env"),
            "ps", "--all", "--quiet", role), remaining(), cwd=directory)
        container_id = raw.decode("ascii", errors="strict").strip()
        if not container_id and allow_absent:
            return None
        if not re.fullmatch(r"[0-9a-f]{64}", container_id):
            raise ArtifactError("expected exactly one owned service container")
        owner = self.workspace / "operator" / "service-owner.json"
        if require_owner and not owner.exists():
            raise ArtifactError("recorded service ownership receipt is missing")
        if owner.exists() and container_id != read_json(owner)["container_id"]:
            raise ArtifactError("refusing a replacement service container")
        if self.journal is not None and "launch" in self.journal.records:
            launch = self.journal.status("launch")
            if launch["status"] == "completed" and launch["result"]["container_id"] != container_id:
                raise ArtifactError("service container differs from its authoritative launch receipt")
        observed = loads(self.runner(["docker", "--host", "unix:///var/run/docker.sock", "inspect", "--type", "container",
            "--format", "{{json .}}", container_id], remaining(), cwd=directory))
        compose = read_json(directory / "compose-effective.json")
        labels = observed.get("Config", {}).get("Labels", {})
        if observed.get("Id") != container_id or observed.get("Config", {}).get("Image") != self.settings["service_image"] or (
                labels.get("com.docker.compose.project"), labels.get("com.docker.compose.service")) != (compose["name"], role):
            raise ArtifactError("service ownership differs from the reviewed deployment")
        from .container_inspection import validate_container_inspection
        validate_container_inspection(observed, self.request, directory)
        return observed

    def _reject_existing_project(self):
        role = self.settings["role"]
        ids = self.runner(["docker", "--host", "unix:///var/run/docker.sock", "ps", "--all", "--quiet", "--no-trunc",
            "--filter", "label=com.docker.compose.project=ria-" + role,
            "--filter", "label=com.docker.compose.service=" + role], self.settings["deadline_ms"])
        if ids.strip():
            raise ArtifactError("a prior role container exists; stop its owning workspace before starting a new setup job")

    def launch(self):
        self.require("finalize")
        self.require("stop-fixture")
        self._reject_existing_project()
        deployment.launch(self.request, self.paths["final"], runner=self.runner)
        observed = self._owned_service()
        _publish_once(self.workspace / "operator" / "service-owner.json",
            {"container_id": observed["Id"], "image": self.settings["service_image"]})
        return {"container_id": observed["Id"]}

    def health(self):
        self.require("launch")
        deadline = min(self.deadline, time.monotonic() + self.settings["environment"]["start_period_seconds"])
        while True:
            remaining_ms = int((deadline - time.monotonic()) * 1000)
            if remaining_ms <= 0:
                raise ArtifactError("owned service startup deadline elapsed")
            observed = self._owned_service(deadline_ms=min(self.settings["deadline_ms"], remaining_ms), require_owner=True)
            state = observed.get("State", {})
            if state.get("Running") is not True:
                raise ArtifactError("owned model service exited before readiness")
            status = state.get("Health", {}).get("Status")
            if status == "healthy":
                remaining_ms = int((deadline - time.monotonic()) * 1000)
                if remaining_ms <= 0:
                    raise ArtifactError("owned service startup deadline elapsed")
                timeout_ms = min(5000, max(1, remaining_ms - 1))
                self.runner(["docker", "--host", "unix:///var/run/docker.sock", "exec", observed["Id"],
                    "/usr/local/bin/ds4ctl", "health", "--socket", "/run/dwarfstar/admin.sock", "--timeout-ms", str(timeout_ms)], remaining_ms)
                return {"container_id": observed["Id"], "healthy": True}
            if status == "unhealthy" or time.monotonic() >= deadline:
                raise ArtifactError("owned service did not become healthy within its declared startup period")
            time.sleep(min(1, max(0, deadline - time.monotonic())))

    def stop_fixture(self):
        self.require("finalize")
        deployment.fixture_stop(self.request, self.paths["bootstrap"], runner=self.runner)
        return {"stopped": True}

    def quiesce(self):
        """Stop only exact owned resources; report cleanup failures to the caller."""
        errors = []
        self.cleaning = True
        self.cleanup_thread = threading.get_ident()
        self.cancelled.set()
        if self.request is not None and Path(self.paths["bootstrap"]).exists():
            try:
                marker = read_json(Path(self.paths["bootstrap"]) / "bootstrap.json")
                lock = Path(self.paths["reports"]) / ("fixture-" + marker["digest"] + ".json")
                if lock.exists():
                    deployment.fixture_stop(self.request, self.paths["bootstrap"], runner=self.runner)
            except (ArtifactError, OSError) as error:
                errors.append(str(error))
        owner = self.workspace / "operator" / "service-owner.json"
        # The validated final directory identifies the exact model container
        # even when publication of its ownership receipt was interrupted.
        if owner.exists() or (self.request is not None and Path(self.paths["final"]).exists()):
            try:
                observed = self._owned_service(allow_absent=True)
                if observed is not None and observed.get("State", {}).get("Running"):
                    grace = self.settings["environment"]["stop_grace_seconds"]
                    self.runner(["docker", "--host", "unix:///var/run/docker.sock", "stop", "--timeout", str(grace), observed["Id"]], grace * 1000 + 5000)
                if observed is not None:
                    self.runner(["docker", "--host", "unix:///var/run/docker.sock", "rm", observed["Id"]], self.settings["deadline_ms"])
            except (ArtifactError, OSError) as error:
                errors.append(str(error))
        if errors:
            raise ArtifactError("owned cleanup failed: " + "; ".join(errors))

    def dispatch(self, operation, payload):
        """Expert fixed operations; called by the authenticated local broker."""
        if operation == "stop" and payload == {"abort": True}:
            self.aborted = True
            self.cancelled.set()
            self.finished.set()
            return {"aborted": True}
        if self.aborted:
            raise ArtifactError("setup job is aborting")
        if operation == "commit-plan" and set(payload) == {"local_experts"}:
            self.require("discover")
            from .schemas import StrictValidator, PLACEMENT_PLAN
            error = next(StrictValidator(PLACEMENT_PLAN["properties"]["local_experts"]).iter_errors(payload["local_experts"]), None)
            if error:
                raise ArtifactError("invalid requested local expert selection")
            _publish_once(self.workspace / "operator" / "client-selection.json", payload)
            def prepare():
                receipt = self._prepare_model()
                self.model = read_json(self.workspace / "operator" / "model-facts.json", max_bytes=MAX_MESSAGE, max_nodes=MAX_NODES)
                verify_identity(self.model, receipt["model_facts_digest"])
                self.server.export_tree("client-model", self.model["export_root"], max_bytes=self.settings["max_transfer_bytes"], check=self.remaining_ms)
                return receipt
            # A completed preparation receipt proves both model facts and the
            # allowed export are ready before another stage may observe them.
            self.step("prepare", prepare)
            return {"prepared": True, "client_manifest_digest": self.model["client_manifest_digest"]}
        if operation == "offer" and not payload:
            self.require("prepare")
            return {"model": self.model, "host": self.host, "bind_address": self.settings["bind_address"]}
        if operation == "client-offer" and set(payload) == {"facts"}:
            def accept():
                facts = payload["facts"]
                verify_identity(facts)
                for key in ("client_manifest_digest", "client_layout_digest", "selected_tensors"):
                    if facts[key] != self.model[key]:
                        raise ArtifactError("client facts differ from the exact prepared compact extraction")
                validate("deployment-environment", facts["environment"])
                verify_identity(facts["environment"])
                self._check_peer_build(facts["build_info"], "cuda")
                if facts["certificate_sha256"] != self.security.expected_client_leaf_digest():
                    raise ArtifactError("client certificate differs from the actually issued identity")
                validate_settings({**self.settings, "peer_address": facts["bind_address"]})
                if self.settings.get("peer_address", facts["bind_address"]) != facts["bind_address"]:
                    raise ArtifactError("client address differs from local configured peer")
                self.settings["peer_address"] = facts["bind_address"]
                grants = make_grants(self.settings, self.model["server_manifest"], facts, self.security, self.workspace)
                _publish_once(self.paths["grants"], grants)
                environment = self._bootstrap(self.model["server_manifest"], Path(self.paths["grants"]))
                _publish_once(self.workspace / "operator" / "client-facts.json", facts)
                return {"environment": environment, "build_info": self.host["build_info"], "grants_digest": grants["digest"]}
            self.require("prepare")
            return self.step("client-offer", accept, inputs=payload)
        if operation == "registration" and not payload:
            self.require("client-offer")
            def registration():
                client = read_json(self.workspace / "operator" / "client-facts.json")
                result = register(self.tuning, self.model, {"server": self.host["build_info"], "client": client["build_info"]},
                    {"server": read_json(Path(self.paths["bootstrap"]) / "environment.json"), "client": client["environment"]},
                    {"server": self._certificate_digest(), "client": client["certificate_sha256"]}, self.workspace / "operator" / "qualification")
                self.install_registration(result)
                return result
            return self.step("registration", registration)
        if operation in ("fixture-start", "probe", "native-fixture", "transport-fixture") and not payload:
            return self.fixture(operation)
        if operation == "collect" and not payload:
            self.require("transport-fixture")
            self.server.export_tree("expert-runs", self.export_runs(), max_bytes=self.settings["max_transfer_bytes"], check=self.remaining_ms)
            if not (self.workspace / "imports" / "peer-runs").is_dir():
                raise ArtifactError("complete client evidence has not been transferred")
            return self.step("collect", lambda: {"complete": True})
        if operation == "finalize" and not payload:
            return self.step("finalize", self.finalize)
        if operation == "stop" and payload == {"fixture": True}:
            return self.step("stop-fixture", self.stop_fixture)
        if operation == "launch" and not payload:
            return self.step("launch", self.launch)
        if operation == "health" and not payload:
            return self.step("health", self.health)
        if operation == "finish" and not payload:
            self.require("health")
            self.finished.set()
            return {"finished": True, "final_release_qualified": False}
        raise ArtifactError("unexpected setup operation or payload for this local stage")

    def run_expert(self):
        self.prepare()
        self.server = PeerServer(self.invitation, self.security, self.dispatch,
            journal=SetupJournal(self.workspace / "peer-journal"), timeout=self.settings["deadline_ms"] / 1000)
        try:
            self.server.allow_import("client-runs", self.workspace / "imports" / "peer-runs", max_bytes=self.settings["max_transfer_bytes"], check=self.remaining_ms)
            self.server.start()
            print("Expert setup is ready. Transfer the private invitation file at " + str(self.workspace / "invitation.json") +
                " through your trusted operator channel, then launch client setup.", file=sys.stderr, flush=True)
            deadline = self.deadline
            while not self.finished.wait(min(1, max(0, deadline - time.monotonic()))):
                self.server.check_health()
                if time.monotonic() >= deadline:
                    raise ArtifactError("paired setup exceeded its declared coordination deadline")
            if self.aborted:
                raise ArtifactError("client aborted paired setup")
            return {"role": "expert", "healthy": True, "qualification_scope": "initial_fixture", "final_release_qualified": False}
        finally:
            primary, errors = sys.exception(), []
            try:
                self.server.begin_close()
            except BaseException as error:
                errors.append(f"{type(error).__name__}: {error}")
            try:
                if not self.finished.is_set() or self.aborted:
                    self.cancelled.set()
                    self.quiesce()
            except BaseException as error:
                errors.append(f"{type(error).__name__}: {error}")
            try:
                self.server.close(timeout=self.settings["deadline_ms"] / 1000 + 10)
            except BaseException as error:
                errors.append(f"{type(error).__name__}: {error}")
            # Never release either lease while a Python worker can still
            # mutate this job. All owned subprocesses have cancellation and
            # bounded reap; surviving controller work is process-fatal.
            thread = self.server.thread
            broker_alive = worker_alive = ownership_ambiguous = False
            surviving = []
            try:
                broker_alive = thread is not None and thread.is_alive()
                process = getattr(self.server, "process", None)
                worker_alive = process is not None and process.poll() is None
                # A live broker may still publish tasks. Its failed join
                # already requires exit; otherwise use a bounded snapshot.
                if not broker_alive:
                    mutex = getattr(self.server, "mutex", None)
                    if mutex is not None and not mutex.acquire(timeout=1):
                        raise ArtifactError("setup task ownership snapshot could not acquire its bounded lock")
                    try:
                        surviving = [name for name, task in self.server.tasks.items() if task.is_alive()]
                    finally:
                        if mutex is not None:
                            mutex.release()
            except BaseException as error:
                ownership_ambiguous = True
                errors.append(f"{type(error).__name__}: {error}")
            if broker_alive or worker_alive or surviving or ownership_ambiguous:
                try:
                    def reason(value):
                        text = str(value)
                        if self.invitation is not None:
                            text = text.replace(self.invitation["pair_secret"], "[redacted]")
                        return "".join(character if 32 <= ord(character) != 127 else " " for character in text[:1024])
                    failure = {"primary_failure": reason(primary) if primary is not None else None,
                        "shutdown_failures": [reason(error) for error in errors],
                        "surviving_task_ids": [reason(name) for name in surviving],
                        "broker_alive": broker_alive, "worker_alive": worker_alive,
                        "ownership_observation_failed": ownership_ambiguous}
                    try:
                        atomic_bytes(self.workspace / "fatal-containment.json", canonical(seal({
                            "schema_revision": 1, "kind": "setup_fatal_containment",
                            "settings_digest": self.settings_digest, "completion_ambiguous": True,
                            "owned_cleanup_complete": False, **failure})) + b"\n", mode=0o600)
                    except (ArtifactError, OSError) as error:
                        print("fatal containment marker could not be written: " + reason(error), file=sys.stderr, flush=True)
                    print("setup fatal containment: controller work did not quiesce; process terminating, owned effects require reconciliation",
                        file=sys.stderr, flush=True)
                    print("failure context: " + reason(primary) + "; " + "; ".join(failure["shutdown_failures"]),
                        file=sys.stderr, flush=True)
                finally:
                    # Diagnostics, marker publication and even stderr may fail.
                    # None can allow leases to unwind with live mutating work.
                    os._exit(125)
            if errors:
                context = f"setup failed: {primary}; " if primary is not None else ""
                raise ArtifactError(context + "setup shutdown failed: " + "; ".join(errors)) from primary

    def run_client(self):
        peer = PeerClient(self.invitation, security=self.security, timeout=self.settings["deadline_ms"] / 1000)
        # Initialization has accepted this workspace's exact invitation. Only
        # this participating job may request remote containment on failure.
        self.peer_participating = True
        if self.security.tls_enabled:
            self.security.install_client_certificate(self.peer_call(peer, "sign-csr", {"csr": self.security.client_csr()})["certificate"])
        self.peer_submit(peer, "commit-plan", {"local_experts": self.settings["client_runtime"]["local_experts"]})
        peer.wait("commit-plan", timeout=self.remaining_ms() / 1000)
        offer = self.peer_call(peer, "offer", {})
        self.model = offer["model"]
        verify_identity(self.model)
        if offer["bind_address"] != self.settings["peer_address"] or self.model["profile"] != self.settings["profile"]:
            raise ArtifactError("expert offer differs from declared endpoint/profile")
        self.prepare()
        self._check_peer_build(offer["host"]["build_info"], self.settings["server_executor"])
        model_root = Path(self.paths["model"])
        if not model_root.exists():
            peer.fetch_tree("client-model", model_root, max_bytes=self.settings["max_transfer_bytes"],
                timeout=min(self.settings["transfer_deadline_ms"], self.remaining_ms()) / 1000)
        model = verify_client_import(model_root, self.model)
        runtime_readable(model_root)
        self.step("client-offer", lambda: self.client_offer(peer, model))
        self._load_request()
        def registration():
            self.peer_submit(peer, "registration", {})
            return self.install_registration(peer.wait("registration", timeout=self.remaining_ms() / 1000))
        self.step("registration", registration)
        def remote(name, payload=None):
            self.peer_submit(peer, name, payload or {})
            return peer.wait(name, timeout=self.remaining_ms() / 1000)
        remote("fixture-start")
        self.fixture("fixture-start")
        remote("probe")
        self.fixture("probe")
        remote("native-fixture")
        self.fixture("native-fixture")
        self.peer_submit(peer, "transport-fixture", {})
        self.fixture("transport-fixture")
        peer.wait("transport-fixture", timeout=self.remaining_ms() / 1000)
        peer.upload_tree("client-runs", self.export_runs(), max_bytes=self.settings["max_transfer_bytes"],
            timeout=min(self.settings["transfer_deadline_ms"], self.remaining_ms()) / 1000)
        remote("collect")
        peer.fetch_tree("expert-runs", self.workspace / "imports" / "peer-runs", max_bytes=self.settings["max_transfer_bytes"],
            timeout=min(self.settings["transfer_deadline_ms"], self.remaining_ms()) / 1000)
        self.step("collect", lambda: {"complete": True})
        remote("finalize")
        self.step("finalize", self.finalize)
        remote("stop", {"fixture": True})
        self.step("stop-fixture", self.stop_fixture)
        remote("launch")
        remote("health")
        self.step("launch", self.launch)
        self.step("health", self.health)
        self.peer_call(peer, "finish", {})
        return {"role": "client", "healthy": True, "qualification_scope": "initial_fixture", "final_release_qualified": False,
            "api_token_file": str(Path(self.paths["secrets"]) / "api.token")}


def run_setup(settings, *, tuning=None, invitation=None):
    controller = SetupController(settings, tuning=tuning, invitation=invitation)
    # Observation/authority validation precedes both persistent host locks and
    # any workspace writes, including restart cleanup or completed-job checks.
    validate_controller_context(controller.settings, runner=controller.runner)
    with setup_signals(controller), role_lock(controller.settings["role"]), workspace_lock(controller.workspace):
        try:
            controller.initialize()
            if controller.completed:
                validate_controller_context(controller.settings, runner=controller.runner)
                deployment.verify_deployment_files(controller.request, controller.paths["final"])
                current_build = read_json("/usr/share/dwarfstar/build-info.json", max_bytes=2 << 20)
                verify_identity(current_build)
                frozen_build = read_json(controller.paths["build_info"], max_bytes=2 << 20)
                verify_identity(frozen_build)
                if current_build["sources"] != frozen_build["sources"] or current_build["source_lock_digest"] != frozen_build["source_lock_digest"]:
                    raise ArtifactError("completed job belongs to another setup/source publication")
                from .host import revalidate_host_report, verify_container_ancestors
                host = read_json(controller.paths["host_report"])
                revalidate_host_report(host)
                observed = controller._owned_service(require_owner=True)
                verify_container_ancestors(observed["State"]["Pid"], host)
                return {"role": controller.settings["role"], "current_health": controller.health(),
                    "setup_previously_completed": True, "qualification_scope": "initial_fixture", "final_release_qualified": False}
            result = controller.run_expert() if controller.settings["role"] == "expert" else controller.run_client()
            atomic_bytes(controller.workspace / "completion.json", canonical(seal({"schema_revision": 1,
                "settings_digest": controller.settings_digest, "result": result})) + b"\n", mode=0o600)
            return result
        except BaseException as primary:
            original = primary
            if controller.settings["role"] == "client" and controller.invitation is not None and controller.peer_participating:
                try:
                    PeerClient(controller.invitation, timeout=5).call("stop", {"abort": True})
                except (ArtifactError, OSError) as abort_error:
                    primary = ArtifactError(f"{primary}; peer abort could not be acknowledged: {abort_error}")
            try:
                controller.quiesce()
            except (ArtifactError, OSError) as cleanup:
                raise ArtifactError(f"setup failed: {primary}; {cleanup}") from primary
            if primary is not original:
                raise primary from original
            raise


def stop_setup(workspace):
    settings = validate_settings(read_json(Path(workspace) / "settings.json"))
    if settings["workspace"] != str(Path(workspace)):
        raise ArtifactError("stop workspace differs from its recorded settings")
    controller = SetupController(settings)
    validate_controller_context(settings, runner=controller.runner)
    with setup_signals(controller), role_lock(settings["role"]), workspace_lock(workspace):
        controller._load_request()
        controller.quiesce()
        return {"role": settings["role"], "owned_containers_stopped": True}


class SetupTerminated(KeyboardInterrupt):
    """SIGTERM requests owned cancellation and a distinct143 exit status."""


@contextmanager
def setup_signals(controller):
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGTERM)
    def terminate(number, frame):
        controller.cancelled.set()
        raise SetupTerminated("setup received SIGTERM")
    signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)
