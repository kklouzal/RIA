"""Synthetic cgroup/proc trees and bounded subprocess tests; no physical probe."""

from pathlib import Path
import resource
import subprocess
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from ria.host import cgroup_ancestors, verify_container_ancestors
from ria.identity import ArtifactError, seal
from ria.process import run_bounded
import ria.process as process_module


def group(path, memory="max"):
    path.mkdir(parents=True, exist_ok=True)
    for name, value in {"memory.max": memory, "memory.swap.max": "max", "pids.max": "max",
                        "cpuset.cpus.effective": "0-3", "cpuset.mems.effective": "0"}.items():
        (path / name).write_text(value + "\n")


def test_hidden_parent_change_and_pid_membership(tmp_path):
    root, proc = tmp_path / "cgroup", tmp_path / "proc"
    group(root)
    parent = root / "system.slice"
    group(parent, "1048576")
    baseline = seal({"kind": "host_preflight", "rootful": True, "cgroup_version": 2,
                     "parent_ancestors": cgroup_ancestors(parent, root=root)})
    process = proc / "123"
    process.mkdir(parents=True)
    (process / "cgroup").write_text("0::/system.slice/docker-fixture.scope\n")
    assert verify_container_ancestors(123, baseline, proc_root=proc, cgroup_root=root)["verified"]
    (parent / "memory.max").write_text("524288\n")
    with pytest.raises(ArtifactError, match="re-probe"):
        verify_container_ancestors(123, baseline, proc_root=proc, cgroup_root=root)
    (process / "cgroup").write_text("0::/../../other\n")
    with pytest.raises(ArtifactError):
        verify_container_ancestors(123, baseline, proc_root=proc, cgroup_root=root)


def test_physical_root_missing_controllers_are_unlimited(tmp_path):
    root = tmp_path / "cgroup"
    group(root)
    for name in ("memory.max", "memory.swap.max", "pids.max"):
        (root / name).unlink()
    assert cgroup_ancestors(root, root=root)[0]["memory.max"] == "max"
    parent = root / "private"
    group(parent)
    (parent / "memory.max").unlink()
    with pytest.raises(FileNotFoundError):
        cgroup_ancestors(parent, root=root)


def test_native_arguments_output_bounds_and_deadline():
    arguments = ["space here", "$HOME", "*", "", "中文"]
    result = run_bounded([sys.executable, "-c", "import sys; print(repr(sys.argv[1:]))", *arguments])
    assert result.returncode == 0 and result.stdout.decode().strip() == repr(arguments)
    with pytest.raises(ArtifactError, match="output"):
        run_bounded([sys.executable, "-c", "import sys; sys.stderr.write('x'*1000000)"], max_stderr=1024)
    with pytest.raises(subprocess.TimeoutExpired):
        run_bounded([sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.05)
    result = run_bounded([sys.executable, "-c", "import sys; print('failure'); sys.exit(7)"])
    assert result.returncode == 7 and result.stdout == b"failure\n"


def test_complete_child_rss_and_invalid_limits():
    code = "import resource; x=bytearray(32*1024*1024); del x; print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)"
    # Linux wait4 includes the child's inherited fork image before exec. The
    # fixture's startup budget must account for the actual pytest parent, whose
    # NumPy/tokenizer state varies with test order and architecture.
    startup_budget = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 + (128 << 20)
    result = run_bounded([sys.executable, "-c", code], max_rss_bytes=startup_budget)
    assert result.peak_rss_bytes >= int(result.stdout) * 1024 >= 32 << 20
    assert result.elapsed_seconds > 0 and "complete child lifetime" in result.rss_scope
    with pytest.raises(ArtifactError, match="RSS"):
        run_bounded([sys.executable, "-c", code], max_rss_bytes=16 << 20)
    for bad in (float("nan"), float("inf"), False, -1):
        with pytest.raises(ArtifactError):
            run_bounded([sys.executable, "-c", "pass"], timeout=bad)
    with pytest.raises(ArtifactError):
        run_bounded([sys.executable, "-c", "pass"], max_rss_bytes=True)


def test_process_creation_is_included_in_the_deadline(monkeypatch):
    original = subprocess.Popen
    children = []

    def delayed_spawn(*arguments, **keywords):
        child = original(*arguments, **keywords)
        children.append(child)
        time.sleep(0.05)
        return child

    monkeypatch.setattr(process_module.subprocess, "Popen", delayed_spawn)
    with pytest.raises(subprocess.TimeoutExpired):
        run_bounded([sys.executable, "-c", "pass"], timeout=0.01)
    assert len(children) == 1 and children[0].returncode is not None
