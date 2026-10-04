"""Actual Python syntax checks must not need writable source caches.

All other offline checks are recorded boundaries; no build, container, GPU or
model command runs. Root callers drop only the isolated compiler fixture to the
runtime UID/GID, retaining root-owned source caches to reproduce mixed owners.
"""

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile

import pytest


ROOT = Path(__file__).resolve().parents[2]
CHILD = r'''
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys

root, evidence = map(Path, sys.argv[1:])
assert os.geteuid() != 0, "the permission fixture must run unprivileged"
for source in (root / "tools", root / "deploy"):
    try:
        with (source / "__pycache__" / "forbidden-write").open("xb"):
            pass
    except PermissionError:
        pass
    else:
        raise RuntimeError("source cache is writable; fixture is invalid")

spec = importlib.util.spec_from_file_location("isolated_offline_checks", root / "deploy" / "offline_checks.py")
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)
actual_execute = checks.execute

def execute(arguments, directory, log, timeout):
    if "compileall" not in arguments:
        log.write_text("Synthetic recorded boundary; no external check executed.\n")
        return 0
    assert arguments[1] == "-X"
    assert arguments[2].startswith("pycache_prefix=")
    cache = Path(arguments[2].split("=", 1)[1])
    assert cache.parent == evidence and cache.name.startswith("python-cache-")
    info = cache.stat()
    assert stat.S_IMODE(info.st_mode) == 0o700 and info.st_uid == os.geteuid()
    code = actual_execute(arguments, directory, log, timeout)
    compiled = [path for path in cache.rglob("*.pyc") if path.name.startswith(("tool_ok.", "deploy_ok."))]
    (evidence / "compile-observation.json").write_text(json.dumps({
        "cache": str(cache), "mode": stat.S_IMODE(info.st_mode), "uid": info.st_uid,
        "compiled": [str(path) for path in compiled], "source_cache_writable": False,
    }))
    return code

checks.execute = execute
sys.argv = [str(root / "deploy" / "offline_checks.py"), "--output", str(evidence / "offline.json")]
raise SystemExit(checks.main())
'''


def source_files(root):
    """Include the unavailable source caches, which Git need not track."""
    return {
        str(path.relative_to(root)): (hashlib.sha256(path.read_bytes()).hexdigest(),
                                     stat.S_IMODE(path.stat().st_mode), path.stat().st_uid)
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(root).parts
    }


@pytest.mark.parametrize("invalid_syntax", [False, True], ids=["valid", "invalid"])
def test_main_compiles_in_private_evidence_cache_without_touching_source(invalid_syntax):
    # A standalone owned /tmp root stays readable when the root-run branch
    # drops UID. Pytest's root-owned0700 ancestors would invalidate that case.
    with tempfile.TemporaryDirectory(prefix="ria-offline-permissions-", dir="/tmp") as temporary:
        base = Path(temporary)
        base.chmod(0o755)
        root, evidence = base / "read only sources", base / "evidence with spaces"
        root.mkdir(mode=0o755)
        evidence.mkdir(mode=0o700)
        for name in ("tools", "deploy"):
            directory = root / name
            directory.mkdir()
            cache = directory / "__pycache__"
            cache.mkdir()
            (cache / "existing.pyc").write_bytes(b"unavailable source cache sentinel\n")
        (root / "tools" / "tool_ok.py").write_text("value = 41\n")
        (root / "deploy" / "deploy_ok.py").write_text("value = 42\n")
        (root / "deploy" / "offline_checks.py").write_bytes((ROOT / "deploy" / "offline_checks.py").read_bytes())
        if invalid_syntax:
            (root / "tools" / "invalid.py").write_text("def invalid(:\n")
        environment = dict(os.environ)
        environment.pop("PYTHONPYCACHEPREFIX", None)
        environment["GIT_CONFIG_GLOBAL"] = "/dev/null"
        environment["GIT_CONFIG_NOSYSTEM"] = "1"
        subprocess.run(["git", "init", "--quiet"], cwd=root, env=environment, check=True, capture_output=True, timeout=10)
        process_identity = {}
        if os.geteuid() == 0:
            uid = gid = 10001
            # Keep the source cache root-owned. Git additionally checks its
            # own directory ownership, so give this synthetic repo to the child.
            for path in [root, root / ".git", *(root / ".git").rglob("*")]:
                os.chown(path, uid, gid)
            os.chown(evidence, uid, gid)
            process_identity = {"user": uid, "group": gid, "extra_groups": []}
        else:
            uid = os.geteuid()
        for directory in (root / "tools", root / "deploy"):
            for path in directory.rglob("*"):
                path.chmod(0o555 if path.is_dir() else 0o444)
            directory.chmod(0o555)
        root.chmod(0o555)
        original = source_files(root)
        result = subprocess.run([sys.executable, "-B", "-c", CHILD, str(root), str(evidence)],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=environment, timeout=20, check=False, **process_identity)
        expected = 1 if invalid_syntax else 0
        assert result.returncode == expected, result.stdout[-4096:] + result.stderr[-4096:]
        document = json.loads((evidence / "offline.json").read_text())
        assert document["passed"] is not invalid_syntax
        assert document["changed_during_checks"] == []
        compile_check = next(check for check in document["checks"] if check["name"] == "python-compile")
        assert compile_check["exit_code"] == expected
        assert compile_check["command"][1] == "-X" and compile_check["command"][3:] == ["-m", "compileall", "-q", "tools", "deploy"]
        observation = json.loads((evidence / "compile-observation.json").read_text())
        cache = Path(observation["cache"])
        assert compile_check["command"][2] == "pycache_prefix=" + str(cache)
        assert cache.parent == evidence and observation["mode"] == 0o700 and observation["uid"] == uid
        assert observation["source_cache_writable"] is False
        assert {Path(path).name.split(".", 1)[0] for path in observation["compiled"]} == {"tool_ok", "deploy_ok"}
        assert not cache.exists() and not list(evidence.glob("python-cache-*"))
        assert source_files(root) == original
        assert all(stat.S_IMODE((root / name / "__pycache__").stat().st_mode) == 0o555 for name in ("tools", "deploy"))
        log = Path(compile_check["log"])
        assert hashlib.sha256(log.read_bytes()).hexdigest() == compile_check["log_sha256"]
        if invalid_syntax:
            assert b"SyntaxError" in log.read_bytes() and b"invalid.py" in log.read_bytes()
