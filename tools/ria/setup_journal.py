"""Durable fixed-step ownership; interrupted work never becomes completed evidence."""

import os
import copy
from pathlib import Path
import re
import stat
import threading

from .identity import ArtifactError, atomic_bytes, canonical, digest, read_json
from .setup_limits import MAX_RECEIPT

STEP_ID = re.compile(r"[a-zA-Z0-9_-]{1,96}\Z")


class SetupJournal:
    """One process owns this directory and serializes every transition.

    Callers must hold their workspace process lock. Recovery marks accepted or
    running work failed; the controller must stop its owned fixture container.
    A completed result is a receipt, never a substitute for evidence validation.
    """

    def __init__(self, directory, *, maximum=128):
        self.directory = Path(directory).absolute()
        self.directory.mkdir(mode=0o700, parents=False, exist_ok=True)
        info = self.directory.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o700
        ):
            raise ArtifactError("setup journal must be an owned private real directory")
        if type(maximum) is not int or not 1 <= maximum <= 128:
            raise ArtifactError("invalid setup journal population bound")
        self.maximum = maximum
        self.lock = threading.Lock()
        self.records = {}
        files = list(self.directory.iterdir())
        if len(files) > maximum:
            raise ArtifactError("setup journal exceeds its fixed step population")
        self.interrupted = []
        for path in files:
            if path.suffix != ".json" or not STEP_ID.fullmatch(path.stem):
                raise ArtifactError("unexpected setup journal entry")
            value = read_json(path, max_bytes=MAX_RECEIPT)
            info = path.lstat()
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o600
            ):
                raise ArtifactError(
                    "setup journal records must be owned regular0600 files"
                )
            fields = {"schema_revision", "step_id", "name", "input_digest", "status"}
            if (
                not isinstance(value, dict)
                or not fields <= set(value) <= fields | {"result", "error"}
                or type(value.get("schema_revision")) is not int
                or value.get("schema_revision") != 1
                or value.get("step_id") != path.stem
                or not isinstance(value.get("name"), str)
                or not isinstance(value.get("input_digest"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", value["input_digest"])
                or value.get("status")
                not in ("accepted", "running", "completed", "failed")
            ):
                raise ArtifactError("invalid setup journal record")
            if (value["status"] == "completed" and "result" not in value) or (
                value["status"] == "failed" and not isinstance(value.get("error"), str)
            ):
                raise ArtifactError("setup terminal journal record lacks its receipt")
            if value["status"] in ("accepted", "running"):
                value.update(
                    status="failed",
                    error="controller interrupted; owned resources require quiescence",
                )
                self._write(path, value)
                self.interrupted.append(path.stem)
            self.records[path.stem] = value

    @staticmethod
    def _write(path, value):
        data = canonical(value) + b"\n"
        if len(data) > MAX_RECEIPT:
            raise ArtifactError("setup journal receipt exceeds its bound")
        atomic_bytes(path, data, mode=0o600)

    def accept(self, step_id, name, payload):
        if not isinstance(step_id, str) or not STEP_ID.fullmatch(step_id):
            raise ArtifactError("invalid fixed setup step identity")
        if (
            not isinstance(name, str)
            or not STEP_ID.fullmatch(name)
            or not isinstance(payload, dict)
        ):
            raise ArtifactError("invalid fixed setup journal inputs")
        identity = digest({"name": name, "payload": payload})
        with self.lock:
            if step_id in self.records:
                value = self.records[step_id]
                if value["input_digest"] != identity:
                    raise ArtifactError(
                        "setup step identity already has different inputs"
                    )
                return copy.deepcopy(value), False
            if len(self.records) >= self.maximum:
                raise ArtifactError("setup step population exhausted")
            value = {
                "schema_revision": 1,
                "step_id": step_id,
                "name": name,
                "input_digest": identity,
                "status": "accepted",
            }
            self._write(self.directory / (step_id + ".json"), value)
            self.records[step_id] = value
            return copy.deepcopy(value), True

    def transition(self, step_id, status, *, result=None, error=None):
        with self.lock:
            old = self.records[step_id]
            allowed = {
                "accepted": {"running", "failed"},
                "running": {"completed", "failed"},
            }
            if status not in allowed.get(old["status"], set()):
                raise ArtifactError("invalid setup step transition")
            value = {**copy.deepcopy(old), "status": status}
            if status == "completed":
                value["result"] = copy.deepcopy(result)
            if error is not None:
                value["error"] = str(error)[:4096]
            self._write(self.directory / (step_id + ".json"), value)
            self.records[step_id] = value
            return copy.deepcopy(value)

    def status(self, step_id):
        if not isinstance(step_id, str) or not STEP_ID.fullmatch(step_id):
            raise ArtifactError("invalid fixed setup step identity")
        with self.lock:
            if step_id not in self.records:
                return {"step_id": step_id, "status": "unknown"}
            return copy.deepcopy(self.records[step_id])

    def check_health(self):
        """A failed fixed stage makes this job unusable; it is never retried."""
        with self.lock:
            for step in sorted(self.records):
                value = self.records[step]
                if value["status"] == "failed":
                    raise ArtifactError(
                        "setup stage " + step + " failed: " + value["error"]
                    )
