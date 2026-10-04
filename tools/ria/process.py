"""Finite native-argument subprocesses with bounded output and owned cleanup."""

import math
import os
from pathlib import Path
import signal
import selectors
import subprocess
import time

from .identity import ArtifactError


def run_bounded(arguments, *, cwd=None, env=None, timeout=30,
                max_stdout=1048576, max_stderr=65536, max_rss_bytes=None):
    if not arguments or any(not isinstance(item, (str, Path)) for item in arguments):
        raise ArtifactError("subprocess requires an explicit native argument list")
    if (type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0 or
            type(max_stdout) is not int or type(max_stderr) is not int or
            not 0 < max_stdout <= 32 << 20 or not 0 < max_stderr <= 32 << 20):
        raise ArtifactError("invalid subprocess deadline/output bound")
    if max_rss_bytes is not None and (type(max_rss_bytes) is not int or not 0 < max_rss_bytes <= 9007199254740991):
        raise ArtifactError("invalid subprocess RSS bound")
    process = subprocess.Popen(arguments, cwd=cwd, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, start_new_session=True)
    started = time.monotonic()
    deadline = started + timeout
    output, diagnostic = bytearray(), bytearray()
    complete = False
    usage = None

    def reap():
        nonlocal usage
        if process.returncode is None:
            # This function is the sole child owner. Reap through the native
            # Linux wait4 contract so the complete lifetime's peak includes
            # startup, work, report emission and teardown. Recording the
            # documented returncode prevents Popen's cleanup from reaping again.
            pid, status, result = os.wait4(process.pid, os.WNOHANG)
            if pid:
                process.returncode = os.waitstatus_to_exitcode(status)
                usage = result
                if max_rss_bytes is not None and result.ru_maxrss * 1024 > max_rss_bytes:
                    raise ArtifactError("subprocess complete-lifetime peak RSS exceeded its admitted bound")

    def observe():
        if max_rss_bytes is not None and process.returncode is None:
            try:
                with open(f"/proc/{process.pid}/status", "rb") as status:
                    data = status.read(8192)
                for line in data.splitlines():
                    if line.startswith((b"VmHWM:", b"VmRSS:")):
                        fields = line.split()
                        if len(fields) != 3 or fields[2] != b"kB":
                            raise ArtifactError("unexpected native RSS observation format")
                        if int(fields[1]) * 1024 > max_rss_bytes:
                            raise ArtifactError("subprocess observed RSS exceeded its admitted bound")
            except FileNotFoundError:
                pass  # wait4 below remains the authoritative lifetime peak.

    try:
        with selectors.DefaultSelector() as selector:
            for stream, buffer, limit in ((process.stdout, output, max_stdout),
                                          (process.stderr, diagnostic, max_stderr)):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, (buffer, limit))
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(arguments, timeout)
                observe()
                for key, _ in selector.select(min(remaining, .025)):
                    buffer, limit = key.data
                    try:
                        chunk = os.read(key.fileobj.fileno(), min(65536, limit - len(buffer) + 1))
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    buffer.extend(chunk)
                    if len(buffer) > limit:
                        raise ArtifactError("subprocess output exceeded its admitted bound")
        while process.returncode is None:
            observe()
            reap()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(arguments, timeout)
            if process.returncode is None:
                time.sleep(min(.025, remaining))
        complete = True
        result = subprocess.CompletedProcess(arguments, process.returncode, bytes(output), bytes(diagnostic))
        result.peak_rss_bytes = usage.ru_maxrss * 1024
        result.elapsed_seconds = time.monotonic() - started
        result.rss_scope = "native wait4 complete child lifetime; descendants require their own cgroup cap"
        return result
    finally:
        if not complete:
            # Before reaping, the child PID/group identity cannot be reused.
            # After reaping with both result pipes closed, never signal that
            # reusable identity as a cleanup action.
            if process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait(timeout=10)
        process.stdout.close()
        process.stderr.close()
