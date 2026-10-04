"""Root-only localhost setup comparison; no Docker, GPU, weights or calibration.

Install requirements-ria-dev.txt with its hashes, then run::

    sudo -n .venv/bin/python tests/ria/setup_peer_benchmark.py \
        --output build/ria/evidence/setup-peer-benchmark.json

The source-only original512KiB archive is an immutable test oracle, never a
runtime fallback. Each before/after sample executes a fresh interpreter and an
actual UID/GID10001 setup worker. Cache eviction is advisory, not a global drop.
CPU includes identical5ms RSS sampling; its cost and process tick granularity
are reported. Three observations per mode/cache cannot establish true p95/p99.
"""

import argparse
from contextlib import contextmanager
import gzip
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import socket
import signal
import ssl
import statistics
import subprocess
import sys
import tarfile
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).with_name("fixtures")
MEMBERS = {
    "tools/ria/__init__.py",
    "tools/ria/identity.py",
    "tools/ria/setup_journal.py",
    "tools/ria/setup_limits.py",
    "tools/ria/setup_peer.py",
    "tools/ria/setup_security.py",
}
PAYLOAD_BYTES = 64 << 20
QUOTA_BYTES = PAYLOAD_BYTES + 4096
MEMORY_BUDGET_BYTES = 256 << 20
TRANSFER_BUDGET_SECONDS = 60


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes(source):
    return {name: sha(source / name) for name in sorted(MEMBERS)}


def unpack_baseline(destination):
    """Verify fixed bytes before bounded extraction of six regular source files."""
    manifest_path = FIXTURE / "setup-peer-baseline.json"
    archive = FIXTURE / "setup-peer-baseline.tar.gz"
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema_revision"] != 1 or set(manifest["members"]) != MEMBERS:
        raise RuntimeError("baseline source manifest is not the exact fixed closure")
    packed = archive.read_bytes()
    if (
        len(packed) != manifest["archive_bytes"]
        or len(packed) > 1 << 20
        or sha(archive) != manifest["archive_sha256"]
    ):
        raise RuntimeError("baseline source archive hash/size differs")
    with gzip.GzipFile(fileobj=io.BytesIO(packed)) as file:
        raw = file.read((1 << 20) + 1)
    if len(raw) > 1 << 20:
        raise RuntimeError("baseline source archive expansion exceeds its bound")
    seen = set()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive_file:
        for item in archive_file:
            if (
                item.name not in MEMBERS
                or item.name in seen
                or not item.isfile()
                or item.size != manifest["members"][item.name]["bytes"]
                or not 0 <= item.size <= 128 << 10
            ):
                raise RuntimeError(
                    "baseline archive member is outside its fixed contract"
                )
            stream = archive_file.extractfile(item)
            if stream is None:
                raise RuntimeError("baseline archive source unavailable")
            with stream:
                data = stream.read(item.size + 1)
            if (
                len(data) != item.size
                or hashlib.sha256(data).hexdigest()
                != manifest["members"][item.name]["sha256"]
            ):
                raise RuntimeError("baseline archive member digest differs")
            target = destination / item.name
            target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
            with target.open("xb") as file:
                file.write(data)
            target.chmod(0o644)
            seen.add(item.name)
    if seen != MEMBERS:
        raise RuntimeError("baseline archive source closure incomplete")
    return {
        "archive_sha256": manifest["archive_sha256"],
        "manifest_sha256": sha(manifest_path),
    }


def proc_status(pid):
    with open(f"/proc/{pid}/status", encoding="ascii") as file:
        raw = file.read(16385)
    if len(raw) > 16384:
        raise RuntimeError("process status exceeds fixed fixture bound")
    return dict(line.split(":", 1) for line in raw.splitlines() if ":" in line)


def cpu(pid):
    with open(f"/proc/{pid}/stat", encoding="ascii") as file:
        raw = file.read(16385)
    if len(raw) > 16384:
        raise RuntimeError("process counters exceed fixed fixture bound")
    fields = raw.rpartition(") ")[2].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


def address():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return "127.0.0.1:" + str(sock.getsockname()[1])


def digest_file(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


@contextmanager
def owned_server(server, timings):
    """Retain primary context while always joining/reaping fixture ownership."""
    primary = None
    try:
        yield
    except BaseException as error:
        primary = error
        raise
    finally:
        started = time.perf_counter()
        try:
            server.close()
        except BaseException as cleanup:
            if primary is None:
                raise
            primary.add_note("fixture cleanup also failed: " + str(cleanup))
        timings["shutdown_s"] = time.perf_counter() - started


def run_sample(arguments):
    # The process group includes the separately executed UID10001 worker. A
    # failed/hung comparison must not orphan a localhost listener/controller.
    process = subprocess.Popen(
        arguments,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        output, error = process.communicate(timeout=180)
    except BaseException:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise
    if process.returncode != 0:
        raise RuntimeError("bounded source comparison failed: " + error[-8192:])
    if len(output) > 65536:
        raise RuntimeError("bounded fixture result exceeded its ceiling")
    return json.loads(output)


def measure(args):
    source_tree = args.source.resolve()
    expected_sources = source_hashes(source_tree)
    sys.path.insert(0, str(source_tree / "tools"))
    from ria.setup_security import SetupSecurity
    from ria.setup_peer import PeerClient, PeerServer
    from ria.setup_limits import CHUNK, MAX_MESSAGE

    with tempfile.TemporaryDirectory(prefix="ria-setup-measure-") as name:
        root = Path(name)
        source = root / "source"
        source.mkdir(mode=0o700)
        payload = source / "weights.synthetic"
        block = bytes(range(256)) * 4096
        with payload.open("wb") as output:
            for _ in range(PAYLOAD_BYTES // len(block)):
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        (source / "manifest.json").write_bytes(b'{"fixture":"synthetic-transfer-only"}')
        (source / "zero-byte-file").touch()
        expected = digest_file(payload)
        outputs = root / "output"
        outputs.mkdir(mode=0o700)
        start = time.perf_counter()
        security, invitation = SetupSecurity.create_expert(
            root / "security", "a" * 64, address(), tls_enabled=bool(args.tls)
        )
        server = PeerServer(
            invitation, security, lambda name, payload: {}, timeout=30
        ).start()
        timings = {}
        with owned_server(server, timings):
            client = PeerClient(invitation, timeout=30)
            startup_s = time.perf_counter() - start
            worker = proc_status(server.process.pid)
            if (
                set(worker["Uid"].split()) != {"10001"}
                or set(worker["Gid"].split()) != {"10001"}
                or worker["Groups"].split()
                or any(
                    int(worker[field], 16) != 0
                    for field in ("CapInh", "CapPrm", "CapEff", "CapAmb")
                )
            ):
                raise RuntimeError(
                    "measured worker retained privileged identity/authority"
                )
            if args.cache == "cold-advisory":
                with payload.open("rb") as file:
                    os.posix_fadvise(file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            else:
                digest_file(payload)
            stop = threading.Event()
            peaks = {"parent": 0, "worker": 0, "combined": 0, "samples": 0}
            sample_errors = []

            def sample():
                began = time.thread_time()
                try:
                    while not stop.is_set():
                        parent_rss = (
                            int(proc_status(os.getpid())["VmRSS"].split()[0]) * 1024
                        )
                        worker_rss = (
                            int(proc_status(server.process.pid)["VmRSS"].split()[0])
                            * 1024
                        )
                        peaks["parent"] = max(peaks["parent"], parent_rss)
                        peaks["worker"] = max(peaks["worker"], worker_rss)
                        peaks["combined"] = max(
                            peaks["combined"], parent_rss + worker_rss
                        )
                        peaks["samples"] += 1
                        stop.wait(0.005)
                except Exception as error:
                    sample_errors.append(str(error))
                finally:
                    peaks["sampler_cpu_s"] = time.thread_time() - began

            monitor = threading.Thread(target=sample, daemon=False)
            monitor.start()
            try:
                before_cpu, before_worker = time.process_time(), cpu(server.process.pid)
                before = time.perf_counter()
                manifest = server.export_tree(
                    "client-model", source, max_bytes=QUOTA_BYTES
                )
                registered = time.perf_counter()
                fetched = client.fetch_tree(
                    "client-model",
                    outputs / "fetched",
                    max_bytes=QUOTA_BYTES,
                    timeout=TRANSFER_BUDGET_SECONDS,
                )
                downloaded = time.perf_counter()
                server.allow_import(
                    "client-runs", outputs / "uploaded", max_bytes=QUOTA_BYTES
                )
                uploaded_manifest = client.upload_tree(
                    "client-runs",
                    source,
                    max_bytes=QUOTA_BYTES,
                    timeout=TRANSFER_BUDGET_SECONDS,
                )
                uploaded = time.perf_counter()
                used_cpu, used_worker = (
                    time.process_time() - before_cpu,
                    cpu(server.process.pid) - before_worker,
                )
                if (
                    fetched != manifest
                    or uploaded_manifest["total_bytes"] != manifest["total_bytes"]
                ):
                    raise RuntimeError("complete transfer manifest association failed")
                for destination in (outputs / "fetched", outputs / "uploaded"):
                    if (
                        digest_file(destination / payload.name) != expected
                        or (destination / "manifest.json").read_bytes()
                        != (source / "manifest.json").read_bytes()
                        or (destination / "zero-byte-file").stat().st_size != 0
                    ):
                        raise RuntimeError(
                            "independent complete published-byte oracle differs"
                        )
            finally:
                stop.set()
                monitor.join()
            if (
                sample_errors
                or peaks["samples"] == 0
                or peaks["combined"] > MEMORY_BUDGET_BYTES
            ):
                raise RuntimeError(
                    "fixture RSS/sampling hard budget failed: " + str(sample_errors)
                )
        shutdown_s = timings["shutdown_s"]
        record = {
            "variant": args.variant,
            "tls_enabled": bool(args.tls),
            "cache": args.cache,
            "repeat": args.repeat,
            "startup_s": startup_s,
            "register_s": registered - before,
            "fetch_s": downloaded - registered,
            "upload_s": uploaded - downloaded,
            "total_s": uploaded - before,
            "shutdown_s": shutdown_s,
            "parent_cpu_s": used_cpu,
            "worker_cpu_s": used_worker,
            "combined_cpu_s": used_cpu + used_worker,
            "peak_rss_bytes": peaks,
            "payload_bytes": PAYLOAD_BYTES,
            "payload_sha256": expected,
            "correct": True,
            "worker_identity": {
                key: worker[key].strip()
                for key in (
                    "Uid",
                    "Gid",
                    "Groups",
                    "CapInh",
                    "CapPrm",
                    "CapEff",
                    "CapAmb",
                    "NoNewPrivs",
                )
            },
            "chunk_bytes": CHUNK,
            "rpc_limit": MAX_MESSAGE,
            "source_hashes": expected_sources,
        }
        if source_hashes(source_tree) != expected_sources:
            raise RuntimeError("source changed while its measured process was active")
        print(json.dumps(record, sort_keys=True))


def summaries(records):
    result = []
    for tls in (False, True):
        for cache in ("cold-advisory", "warm"):
            row = {"tls_enabled": tls, "cache": cache, "variants": {}}
            for variant in ("original512KiB", "current"):
                samples = [
                    r
                    for r in records
                    if r["variant"] == variant
                    and r["tls_enabled"] == tls
                    and r["cache"] == cache
                ]
                metrics = {}
                for metric in (
                    "startup_s",
                    "register_s",
                    "fetch_s",
                    "upload_s",
                    "total_s",
                    "shutdown_s",
                    "combined_cpu_s",
                ):
                    values = sorted(r[metric] for r in samples)
                    metrics[metric] = {
                        "median": statistics.median(values),
                        "min": min(values),
                        "max": max(values),
                        "empirical_p95": values[math.ceil(0.95 * len(values)) - 1],
                    }
                metrics["combined_peak_rss_bytes"] = max(
                    r["peak_rss_bytes"]["combined"] for r in samples
                )
                row["variants"][variant] = {
                    "sample_count": len(samples),
                    "metrics": metrics,
                }
            row["median_transfer_speedup"] = (
                row["variants"]["original512KiB"]["metrics"]["total_s"]["median"]
                / row["variants"]["current"]["metrics"]["total_s"]["median"]
            )
            result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--measure", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--variant", help=argparse.SUPPRESS)
    parser.add_argument("--tls", type=int, choices=(0, 1), help=argparse.SUPPRESS)
    parser.add_argument(
        "--cache", choices=("cold-advisory", "warm"), help=argparse.SUPPRESS
    )
    parser.add_argument("--repeat", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if os.geteuid() != 0 or platform.system() != "Linux":
        parser.error(
            "actual UID10001 peer comparison requires a privileged Linux controller"
        )
    if args.measure:
        if (
            args.source is None
            or args.variant not in ("original512KiB", "current")
            or args.tls is None
            or args.cache is None
            or args.repeat is None
        ):
            parser.error("incomplete private measurement invocation")
        measure(args)
        return
    if args.output is None or not 3 <= args.repetitions <= 10:
        parser.error("--output required and repetitions must be3..10")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    current_sources = source_hashes(ROOT)
    openssl_cli = subprocess.run(
        ["openssl", "version"],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=10,
        env={
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        },
    ).stdout.strip()
    if len(openssl_cli) > 4096:
        raise RuntimeError("OpenSSL version metadata exceeds its bound")
    with tempfile.TemporaryDirectory(prefix="ria-setup-source-oracle-") as name:
        sources = Path(name)
        sources.chmod(0o755)  # Source-only, no credentials/data in this directory.
        baseline = sources / "baseline"
        baseline.mkdir(mode=0o755)
        archive = unpack_baseline(baseline)
        for repeat in range(args.repetitions):
            for cache_index, cache in enumerate(("cold-advisory", "warm")):
                for tls in (0, 1):
                    order = (("original512KiB", baseline), ("current", ROOT))
                    if (repeat + cache_index + tls) % 2:
                        order = order[::-1]
                    for variant, source in order:
                        record = run_sample(
                            [
                                sys.executable,
                                str(Path(__file__).resolve()),
                                "--measure",
                                "--source",
                                str(source),
                                "--variant",
                                variant,
                                "--tls",
                                str(tls),
                                "--cache",
                                cache,
                                "--repeat",
                                str(repeat),
                            ]
                        )
                        if record["source_hashes"] != source_hashes(source):
                            raise RuntimeError(
                                "comparison source changed before receipt"
                            )
                        if (
                            variant == "current"
                            and record["source_hashes"] != current_sources
                        ):
                            raise RuntimeError(
                                "current source changed between comparisons"
                            )
                        if (
                            record["variant"] != variant
                            or record["tls_enabled"] is not bool(tls)
                            or record["cache"] != cache
                            or record["repeat"] != repeat
                            or record["correct"] is not True
                        ):
                            raise RuntimeError(
                                "measurement identity/correctness differs"
                            )
                        records.append(record)
                        print(
                            json.dumps(record, sort_keys=True),
                            file=sys.stderr,
                            flush=True,
                        )
        result = {
            "schema_revision": 1,
            "purpose": "model-free isolated software setup transfer comparison",
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "machine": platform.machine(),
                "openssl": ssl.OPENSSL_VERSION,
                "openssl_cli": openssl_cli,
                "rfc8785": importlib.metadata.version("rfc8785"),
                "cpu_tick_seconds": 1 / os.sysconf("SC_CLK_TCK"),
            },
            "benchmark_sha256": sha(__file__),
            "baseline": archive,
            "dependency_input_sha256": {
                p: sha(ROOT / p)
                for p in ("requirements-ria.txt", "requirements-ria-dev.txt")
            },
            "budgets": {
                "payload_bytes": PAYLOAD_BYTES,
                "job_byte_quota": QUOTA_BYTES,
                "transfer_deadline_seconds": TRANSFER_BUDGET_SECONDS,
                "accepted_connection_deadline_seconds": 30,
                "combined_sampled_rss_bytes": MEMORY_BUDGET_BYTES,
                "repetitions_per_region": args.repetitions,
            },
            "objective_order": [
                "complete transfer wall time",
                "CPU service cost",
                "bounded peak memory",
            ],
            "uncertainty": "localhost software only; warm/cold-advisory cache;5ms sampled RSS can miss shorter peaks; CPU includes recorded sampler overhead and quantized worker ticks; balanced before/after order; small samples cannot establish true p95/p99 or physical remote-model/link performance",
            "records": records,
            "summaries": summaries(records),
        }
        with args.output.open("w", encoding="utf-8") as file:
            json.dump(result, file, sort_keys=True, indent=2)
            file.write("\n")
    print(str(args.output))


if __name__ == "__main__":
    main()
