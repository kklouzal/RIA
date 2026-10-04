"""Selected-toolkit provenance boundaries; no toolkit execution or CUDA loading."""
import json
import os
from pathlib import Path
import sys
import subprocess

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deploy"))
import cuda_toolkit
import build_info
from check_cuda_artifacts import runtime_dependencies


@pytest.fixture
def toolkit(tmp_path, monkeypatch):
    home = tmp_path / "selected toolkit"
    (home / "bin").mkdir(parents=True)
    (home / "include").mkdir()
    (home / "lib64").mkdir()
    (home / "bin/nvcc").write_bytes(b"fixture nvcc")
    (home / "bin/cuobjdump").write_bytes(b"fixture cuobjdump")
    (home / "include/cuda_runtime_api.h").write_text("#define CUDART_VERSION 13040\n")
    (home / "lib64/libcudart_static.a").write_bytes(b"!<arch>\nfixture runtime")
    monkeypatch.delenv("CUDA_VERSION", raising=False)
    calls = []

    def report(arguments):
        calls.append(arguments)
        if arguments[0] == str(home / "bin/nvcc"):
            return "Cuda compilation tools, release 13.4, V13.4.101"
        if arguments[0] == "nm":
            return "000000 T cudaRuntimeGetVersion"
        if arguments[0] == "dpkg-query":
            return "installed\t13.4.2-1"
        raise AssertionError("unexpected executable: " + str(arguments))

    monkeypatch.setattr(cuda_toolkit, "tool_report", report)
    return home, calls


def test_selected_inputs_and_standalone_version_are_recorded(toolkit):
    home, calls = toolkit
    result = cuda_toolkit.inspect_toolkit(home)
    assert result["runtime_api_version"] == 13040
    assert result["sdk_version"] is None and result["container_cuda_version"] is None
    assert result["runtime_linkage"] == "static"
    assert result["nvcc_executable"] == str(home / "bin/nvcc")
    assert set(result["files"]) == {"bin/nvcc", "bin/cuobjdump", "include/cuda_runtime_api.h", "lib64/libcudart_static.a"}
    assert calls == [[str(home / "bin/nvcc"), "--version"],
                     ["nm", "-g", "--defined-only", str(home / "lib64/libcudart_static.a")]]


@pytest.mark.parametrize("tool", ["nvcc", "cuobjdump"])
def test_tools_from_another_home_are_rejected(toolkit, tmp_path, tool):
    home, calls = toolkit
    other = tmp_path / tool
    other.write_bytes(b"another toolkit")
    with pytest.raises(ValueError, match="tools and toolkit home differ"):
        cuda_toolkit.inspect_toolkit(home, **{tool: other})
    assert not calls


def test_tool_symlink_outside_home_is_rejected(toolkit, tmp_path):
    home, calls = toolkit
    other = tmp_path / "old-nvcc"
    other.write_bytes(b"old toolkit")
    (home / "bin/nvcc").unlink()
    (home / "bin/nvcc").symlink_to(other)
    with pytest.raises(ValueError, match="outside the selected toolkit"):
        cuda_toolkit.inspect_toolkit(home)
    assert not calls


@pytest.mark.parametrize("defect", ["header", "archive", "symbols", "nvcc"])
def test_inconsistent_native_inputs_fail_closed(toolkit, monkeypatch, defect):
    home, _ = toolkit
    if defect == "header":
        (home / "include/cuda_runtime_api.h").write_text("#define CUDART_VERSION 13010\n")
    elif defect == "archive":
        (home / "lib64/libcudart_static.a").write_bytes(b"not an archive")
    else:
        original = cuda_toolkit.tool_report

        def report(arguments):
            if defect == "symbols" and arguments[0] == "nm":
                return "000000 T unrelated"
            if defect == "nvcc" and arguments[1] == "--version":
                return "Cuda compilation tools, release 13.4, V13.1.101"
            return original(arguments)

        monkeypatch.setattr(cuda_toolkit, "tool_report", report)
    with pytest.raises(ValueError):
        cuda_toolkit.inspect_toolkit(home)


@pytest.mark.parametrize("metadata", [False, True])
def test_container_exact_sdk_patch_is_verified(toolkit, monkeypatch, metadata):
    home, calls = toolkit
    monkeypatch.setenv("CUDA_VERSION", "13.4.2")
    if metadata:
        (home / "version.json").write_text(json.dumps({"cuda": {"version": "13.4.2"}}))
    result = cuda_toolkit.inspect_toolkit(home)
    assert result["sdk_version"] == "13.4.2"
    assert result["sdk_version_source"] == ("version.json" if metadata else "dpkg:cuda-minimal-build-13-4")
    assert bool([command for command in calls if command[0] == "dpkg-query"]) is not metadata


@pytest.mark.parametrize("version", ["13.4.1", "13.4.92"])
def test_component_or_stale_patch_cannot_claim_pinned_sdk(toolkit, monkeypatch, version):
    home, _ = toolkit
    monkeypatch.setenv("CUDA_VERSION", "13.4.2")
    (home / "version.json").write_text(json.dumps({"cuda": {"version": version}}))
    with pytest.raises(ValueError, match="SDK patch differs"):
        cuda_toolkit.inspect_toolkit(home)


@pytest.mark.parametrize("package", ["installed\t13.4.1-1", "installed\t13.4.92-1", "not-installed\t13.4.2-1"])
def test_package_metadata_must_prove_the_exact_sdk_patch(toolkit, monkeypatch, package):
    home, _ = toolkit
    original = cuda_toolkit.tool_report
    monkeypatch.setenv("CUDA_VERSION", "13.4.2")
    monkeypatch.setattr(cuda_toolkit, "tool_report", lambda argv: package if argv[0] == "dpkg-query" else original(argv))
    with pytest.raises(ValueError):
        cuda_toolkit.inspect_toolkit(home)


@pytest.mark.parametrize("metadata", [{}, {"cuda": []}, {"cuda": {"version": "13.1.1"}}, {"cuda": {"version": 13040}}])
def test_invalid_or_other_sdk_metadata_is_rejected(toolkit, metadata):
    home, _ = toolkit
    (home / "version.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        cuda_toolkit.inspect_toolkit(home)


@pytest.mark.parametrize("version", ["", "13.1.1", "13.4.1"])
def test_container_version_cannot_override_pin(toolkit, monkeypatch, version):
    home, _ = toolkit
    monkeypatch.setenv("CUDA_VERSION", version)
    with pytest.raises(ValueError, match="CUDA_VERSION differs"):
        cuda_toolkit.inspect_toolkit(home)


def test_toolkit_stamp_changes_only_with_selected_inputs(toolkit, monkeypatch, tmp_path):
    home, _ = toolkit
    output = tmp_path / "stamp.json"
    monkeypatch.setattr(sys, "argv", ["cuda_toolkit.py", "--cuda-home", str(home), "--output", str(output)])
    cuda_toolkit.main()
    original = output.read_bytes()
    os.utime(output, ns=(1, 1))
    cuda_toolkit.main()
    assert output.stat().st_mtime_ns == 1 and output.read_bytes() == original
    (home / "bin/nvcc").write_bytes(b"changed compiler")
    cuda_toolkit.main()
    assert output.stat().st_mtime_ns != 1 and output.read_bytes() != original


def test_static_runtime_allows_os_and_injected_driver_dependencies():
    libraries = ["libssl.so.3", "libstdc++.so.6", "libgcc_s.so.1", "libcuda.so.1", "libc.so.6"]
    report = "\n".join("0x1 (NEEDED) Shared library: [" + item + "]" for item in libraries)
    assert runtime_dependencies(report) == libraries


@pytest.mark.parametrize("library", ["libcudart.so.13", "libcublas.so.13", "libcudnn.so.9", "libnvrtc.so.13",
                                     "libnvJitLink.so.13", "libtorch_cuda.so", "libtensorflow.so", ""])
def test_runtime_linkage_rejects_dynamic_toolkit_or_framework(library):
    report = "(NEEDED) Shared library: [" + library + "]" if library else "No dynamic section."
    with pytest.raises(ValueError, match="shared CUDA/framework linkage"):
        runtime_dependencies(report)


def test_make_rebuilds_cuda_objects_after_toolkit_switch(tmp_path):
    """Exercise actual Make dependency/timestamp semantics with fake compilers."""
    root = Path(__file__).resolve().parents[2]
    recipe = tmp_path / "Makefile.ria"
    recipe.write_bytes((root / "Makefile.ria").read_bytes())
    (tmp_path / "deploy").mkdir()
    (tmp_path / "ria").mkdir()
    (tmp_path / "deploy/container-lock.json").write_text("{}")
    (tmp_path / "ria/fixture.cu").write_text("fixture; never compiled as CUDA\n")
    (tmp_path / "deploy/cuda_toolkit.py").write_text(
        "import argparse,hashlib,pathlib\n"
        "p=argparse.ArgumentParser()\n"
        "for name in ('cuda-home','nvcc','cuobjdump','output'):p.add_argument('--'+name,required=True)\n"
        "a=p.parse_args();out=pathlib.Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)\n"
        "value=hashlib.sha256(pathlib.Path(a.nvcc).read_bytes()).digest()\n"
        "if not out.exists() or out.read_bytes()!=value:out.write_bytes(value)\n")
    home = tmp_path / "toolkit with spaces"
    (home / "bin").mkdir(parents=True)
    compiler = home / "bin/nvcc"
    compiler.write_text(
        "#!/usr/bin/env python3\nimport json,pathlib,sys\n"
        "assert '--cudart=static' in sys.argv\n"
        "with pathlib.Path('compiler-calls.log').open('a') as f:f.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "pathlib.Path(sys.argv[sys.argv.index('-o')+1]).write_bytes(b'fixture object')\n")
    compiler.chmod(0o700)
    command = ["make", "-f", str(recipe), "build/ria/ria/fixture.cu.o",
               "RIA_PYTHON=" + sys.executable, "RIA_CUDA_HOME=" + str(home)]

    def build():
        subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, timeout=10)

    build()
    build()
    calls = tmp_path / "compiler-calls.log"
    first = [json.loads(row) for row in calls.read_text().splitlines()]
    assert len(first) == 1 and "-O2" in first[0]
    updated = recipe.read_text().replace("RIA_NVCCFLAGS = -O2 -g", "RIA_NVCCFLAGS = -O1 -g", 1)
    assert updated != recipe.read_text()
    recipe.write_text(updated)
    build()
    second = [json.loads(row) for row in calls.read_text().splitlines()]
    assert len(second) == 2 and "-O1" in second[1] and "-O2" not in second[1]
    with compiler.open("a") as output:
        output.write("# Different selected toolkit input\n")
    build()
    third = [json.loads(row) for row in calls.read_text().splitlines()]
    assert len(third) == 3 and third[2] == second[1]


def test_cpu_build_info_never_inspects_or_executes_cuda(tmp_path, monkeypatch):
    (tmp_path / "deploy").mkdir()
    (tmp_path / "locks").mkdir()
    (tmp_path / "bin").mkdir()
    for name in ("Makefile", "Makefile.ria", "LICENSE", ".dockerignore"):
        (tmp_path / name).write_text("fixture\n")
    (tmp_path / "locks/source-lock.json").write_text(json.dumps({
        "upstream_commits": {"antirez/ds4": "a" * 40}, "digest": "b" * 64}))
    for name in ("ds4ctl", "ds4-expert-server", "ds4-ria-qualify"):
        (tmp_path / "bin" / name).write_bytes(b"fixture CPU binary")
    commands = []

    def report(arguments):
        commands.append(arguments)
        return "fixture build tool report"

    def forbidden(*arguments):
        raise AssertionError("CPU provenance must not inspect CUDA")

    output = tmp_path / "result.json"
    monkeypatch.setattr(build_info, "__file__", str(tmp_path / "deploy/build_info.py"))
    monkeypatch.setattr(build_info, "run", report)
    monkeypatch.setattr(build_info, "inspect_toolkit", forbidden)
    monkeypatch.setattr(sys, "argv", ["build_info.py", "--role", "cpu", "--compiler", "cc", "--output", str(output)])
    build_info.main()
    result = json.loads(output.read_bytes())
    assert "cuda_toolkit" not in result and "nvcc" not in result
    assert [command[0] for command in commands] == ["cc", "dpkg-query", "dpkg"]
