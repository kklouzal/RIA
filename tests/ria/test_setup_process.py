"""Cancellation contains a real owned subprocess group without model work."""

from pathlib import Path
import subprocess
import sys
import time

import pytest

from ria.identity import ArtifactError
from ria.process import run_bounded


def test_cancelled_operation_never_acquires_a_process(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("cancelled operation acquired a process")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    with pytest.raises(ArtifactError, match="cancelled"):
        run_bounded([sys.executable, "-c", "pass"], cancelled=lambda: True)


def test_cancellation_terminates_owned_parent_and_descendant(tmp_path):
    ready = tmp_path / "ready.json"
    program = """
import json,os,subprocess,sys,time
from pathlib import Path
child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])
Path(sys.argv[1]).write_text(json.dumps([os.getpid(),child.pid]))
child.wait()
"""
    started = time.monotonic()
    with pytest.raises(ArtifactError, match="cancelled"):
        run_bounded([sys.executable, "-c", program, str(ready)],
                    timeout=5, cancelled=ready.exists)
    import json
    parent, child = json.loads(ready.read_text())
    assert not Path(f"/proc/{parent}").exists()  # The direct owner reaped it.
    deadline = time.monotonic() + 2
    while Path(f"/proc/{child}/stat").exists():
        state = Path(f"/proc/{child}/stat").read_text().rsplit(")", 1)[1].split()[0]
        if state == "Z":  # Reparented dead descendants are reaped by host init.
            break
        if time.monotonic() >= deadline:
            pytest.fail("owned descendant survived cancellation")
        time.sleep(.01)
    assert time.monotonic() - started < 4
