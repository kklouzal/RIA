#!/usr/bin/env python3
"""Install hash-pinned static check tools into an explicitly owned build directory."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import tarfile
import tempfile
import urllib.request

from offline_checks import execute


def download(entry, cache):
    if cache.is_file():
        raw = cache.read_bytes()
        if len(raw) <= 64 << 20 and hashlib.sha256(raw).hexdigest() == entry["sha256"]:
            return raw
    with urllib.request.urlopen(entry["url"], timeout=30) as response:
        raw = response.read((64 << 20) + 1)
    if len(raw) > 64 << 20 or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError("diagnostic tool download differs from its frozen hash")
    cache.write_bytes(raw)
    return raw


def install_cppcheck(entry, output):
    """Build the pinned upstream source without editing an installed dependency."""
    marker = output / "cppcheck-install.json"
    if marker.is_file() and (output / "cppcheck").is_file():
        previous = json.loads(marker.read_bytes())
        if (previous.get("source_sha256") == entry["sha256"] and
                previous.get("binary_sha256") == hashlib.sha256((output / "cppcheck").read_bytes()).hexdigest()):
            return
    raw = download(entry, output / ("cppcheck-" + entry["version"] + ".tar.gz"))
    with tempfile.TemporaryDirectory(prefix=".cppcheck-build-", dir=output) as temporary:
        directory = Path(temporary)
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
            members = archive.getmembers()
            if len(members) > 10000 or sum(member.size for member in members) > 64 << 20:
                raise ValueError("diagnostic source archive exceeds its bounds")
            for member in members:
                path = Path(member.name)
                if (path.is_absolute() or ".." in path.parts or
                        not (member.isfile() or member.isdir()) or member.size > 16 << 20):
                    raise ValueError("diagnostic source archive has an unsafe member")
            archive.extractall(directory, members=members, filter="data")
        source = directory / ("cppcheck-" + entry["commit"])
        if not (source / "Makefile").is_file():
            raise ValueError("diagnostic source archive has the wrong root")
        data = output / "cppcheck-data"
        data.mkdir(exist_ok=True)
        log = output / "cppcheck-build.log"
        code = execute(["make", "-j2", "MATCHCOMPILER=yes", "FILESDIR=" + str(data), "cppcheck"], source, log, 1200)
        if code:
            raise ValueError(f"diagnostic compiler build failed ({code}); see {log}")
        for name in ("cfg", "platforms", "addons"):
            shutil.copytree(source / name, data / name, dirs_exist_ok=True)
        shutil.copyfile(source / "cppcheck", output / "cppcheck")
        (output / "cppcheck").chmod(0o755)
        marker.write_text(json.dumps({"version": entry["version"], "source_sha256": entry["sha256"],
            "binary_sha256": hashlib.sha256((output / "cppcheck").read_bytes()).hexdigest()}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    architecture = platform.machine()
    if platform.system() != "Linux" or architecture not in ("arm64", "aarch64", "x86_64"):
        raise ValueError("unsupported diagnostic build host")
    architecture = "arm64" if architecture == "aarch64" else architecture
    lock = json.loads(Path(__file__).with_name("check-tools-lock.json").read_bytes())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("actionlint", "hadolint"):
        entry = lock[name][architecture]
        raw = download(entry, args.output_dir / (name + "-" + architecture + ".download"))
        if name == "actionlint":
            with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
                member = archive.getmember("actionlint")
                if not member.isfile() or member.size > 32 << 20:
                    raise ValueError("diagnostic tool archive has an invalid binary")
                with archive.extractfile(member) as binary:
                    raw = binary.read((32 << 20) + 1)
                if len(raw) != member.size:
                    raise ValueError("truncated diagnostic tool archive")
        fd, temporary = tempfile.mkstemp(prefix=".ria-check-", dir=args.output_dir)
        try:
            with os.fdopen(fd, "wb") as stream:
                os.fchmod(stream.fileno(), 0o755)
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, args.output_dir / name)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        print(name, lock[name]["version"], entry["sha256"])
    install_cppcheck(lock["cppcheck"], args.output_dir)
    print("cppcheck", lock["cppcheck"]["version"], lock["cppcheck"]["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
