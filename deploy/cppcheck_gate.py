#!/usr/bin/env python3
"""Run the pinned analyzer with bounded time and inspect its structured findings."""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

from offline_checks import execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    executable = root / "build/ria/check-tools/cppcheck"
    sources = [str(path.relative_to(root)) for path in sorted((root / "ria").glob("*.c"))]
    passed = True
    for name, definition in (("cpu", "RIA_QUALIFY_CPU_ONLY"), ("cuda-host", "RIA_WITH_CUDA")):
        log = args.output_dir / ("cppcheck-" + name + ".xml")
        code = execute([str(executable), "--quiet", "--xml", "--xml-version=2",
            "--enable=warning,performance,portability", "--error-exitcode=1", "--std=c99",
            "--inline-suppr", "--library=posix", "-Iria", "-Ithird_party/ryu", "-D" + definition,
            *sources], root, log, 180)
        try:
            document = ET.fromstring(log.read_bytes())
            if document.find("cppcheck").get("version") != "2.22.0":
                raise ValueError("wrong analyzer version")
            findings = [item for item in document.findall("errors/error")
                        if item.get("severity") != "information" or item.get("id") in
                        ("internalError", "syntaxError", "preprocessorErrorDirective")]
        except (ET.ParseError, AttributeError, ValueError) as error:
            print(name, "invalid analyzer output:", error)
            passed = False
            continue
        if code or findings:
            print(name, "failed:", code, [item.attrib for item in findings])
            passed = False
        else:
            print(name, "passed; structured evidence:", log)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
