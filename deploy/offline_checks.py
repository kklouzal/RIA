#!/usr/bin/env python3
"""Run bounded offline checks and retain complete logs plus source identities."""
import argparse
import hashlib
import json
from pathlib import Path
import plistlib
import os
import signal
import subprocess
import sys
import time


MAX_LOG_BYTES = 32 * 1024 * 1024


def source_snapshot(root):
    paths = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root, capture_output=True, check=True, timeout=30,
    ).stdout.split(b"\0")
    return {
        os.fsdecode(name): hashlib.sha256((root / os.fsdecode(name)).read_bytes()).hexdigest()
        for name in sorted(set(paths))
        if name and (root / os.fsdecode(name)).is_file()
    }


def execute(arguments, root, log, timeout):
    """Bound logs and wall time, including compiler subprocess descendants."""
    with log.open("wb") as output:
        try:
            process = subprocess.Popen(
                arguments, cwd=root, stdout=output, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as error:
            output.write((str(error) + "\n").encode())
            return 127
        deadline = time.monotonic() + timeout
        try:
            while process.poll() is None:
                code = (125 if log.stat().st_size > MAX_LOG_BYTES else
                        124 if time.monotonic() >= deadline else None)
                if code is not None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
                    output.write(f"\nCheck aborted: exit {code}.\n".encode())
                    return code
                time.sleep(0.05)
            if log.stat().st_size > MAX_LOG_BYTES:
                output.truncate(MAX_LOG_BYTES)
                output.seek(0, os.SEEK_END)
                output.write(b"\nCheck output exceeded its bound; exit 125.\n")
                return 125
            return process.returncode
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output = args.output.absolute()
    evidence = args.output.parent
    evidence.mkdir(parents=True, exist_ok=True)
    initial_sources = source_snapshot(root)
    results = []
    def run(name, arguments, timeout=180):
        log = evidence / (name + ".log")
        started = time.monotonic()
        code = execute(arguments, root, log, timeout)
        result = {"name": name, "command": arguments, "cwd": str(root), "exit_code": code,
                  "elapsed_seconds": round(time.monotonic() - started, 3), "log": str(log),
                  "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest()}
        results.append(result)
        print(name, "passed" if code == 0 else f"failed ({code}); see {log}", flush=True)
        return code == 0
    ok = True
    for name, command in (
        ("ruff", ["ruff", "check", "tools", "tests/ria", "deploy"]),
        ("python-compile", [sys.executable, "-m", "compileall", "-q", "tools", "deploy"]),
        ("seccomp-generated", [sys.executable, "deploy/generate_seccomp.py", "--check"]),
        ("source-lock", [sys.executable, "tools/verify_ria.py", "--source-lock", "locks/source-lock.json"]),
        ("workflow-lint", ["build/ria/check-tools/actionlint", ".github/workflows/ria.yml"]),
        ("container-lint", ["build/ria/check-tools/hadolint", "deploy/Dockerfile.cpu", "deploy/Dockerfile.cuda", "deploy/Dockerfile.setup"]),
        ("container-base-lock", [sys.executable, "deploy/check_container_lock.py"]),
        ("native-static", ["make", "ria-static"]),
        ("compiler-versions", ["clang", "--version"]),
        ("cppcheck-version", ["build/ria/check-tools/cppcheck", "--version"]),
    ):
        passed = run(name, command)
        ok = passed and ok
    sources = sorted((root / "ria").glob("*.c"))
    for source in sources:
        name = "analyzer-" + source.stem
        report = evidence / (name + ".plist")
        flags = ["-DRIA_QUALIFY_CPU_ONLY"] if source.stem == "qualify" else []
        passed = run(name, ["clang", "--analyze", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", *flags,
            "-I.", "-Iria", "-Ithird_party/ryu", "-D_FILE_OFFSET_BITS=64", "-fno-fast-math", "-ffp-contract=off",
            "-Xanalyzer", "-analyzer-output=plist", "-o", str(report), str(source.relative_to(root))])
        findings = None
        if passed and report.is_file():
            with report.open("rb") as stream:
                findings = len(plistlib.load(stream).get("diagnostics", []))
            passed = findings == 0
        else:
            passed = False
        results[-1]["analyzer_findings"] = findings
        if not passed:
            print(name, "has findings or missing machine-readable evidence", flush=True)
        ok = passed and ok
    source_hashes = source_snapshot(root)
    changed = sorted(name for name in initial_sources.keys() | source_hashes.keys()
                     if initial_sources.get(name) != source_hashes.get(name))
    if changed:
        print("Source changes invalidated this check run:", ", ".join(changed), flush=True)
        ok = False
    document = {"schema_revision": 1, "classification": "offline/static only; no model or GPU execution",
                "passed": ok, "sources": source_hashes, "changed_during_checks": changed, "checks": results,
                "deferred": ["physical deployment", "GPU numerical execution", "model fidelity", "performance", "one-hour soak"]}
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
