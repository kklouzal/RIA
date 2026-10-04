# CUDA toolkit and container provenance

The CUDA client and CUDA expert share one Linux amd64 image. Its builder and
final base are NVIDIA NGC CUDA **13.4.2** on Ubuntu 24.04, verified as the latest
matching release on 2026-10-04 using the [official NGC catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags)
and the complete anonymous registry tag list. The authoritative
[container lock](../deploy/container-lock.json) records tags, index digests,
amd64 manifest/config digests and inherited driver constraints. The CPU expert
and separate administrative setup image keep their CUDA-free Ubuntu bases.
Neither requires a CUDA driver. The [setup controller](ria-container-setup.md)
packages three CPU native binaries with checked ELF dependencies, hash-locked
Python preparation dependencies and pinned Docker/Compose clients. Its root
management entrypoint is distinct from the UID10001 native serving images.

The `devel` builder supplies NVCC, runtime headers, libdevice and static cudart.
RIA links `--cudart=static`, emits only SM120a cubins and retains strict numerical
flags. The smaller matching `base` final image supplies the CUDA environment;
explicit Ubuntu libraries supply the remaining host dependencies. The full NGC
`runtime` alternative has approximately 1.7 GB more compressed library layers
without a current RIA consumer. No inference-speed improvement is claimed.

`deploy/check_container_lock.py` rejects mismatched recipes, SDKs or image
identities. `deploy/cuda_toolkit.py` validates the actual selected NVCC/cuobjdump
paths, header release, static archive/native API and exact container SDK patch;
it records content hashes without loading CUDA. GNU Make preserves the toolkit
stamp when unchanged and rebuilds affected objects after a toolkit change.
`RIA_CUDA_HOME`, `RIA_NVCC` and `RIA_CUOBJDUMP` must identify one coherent toolkit.
Standalone local builds may use another installed SDK and record its actual
identity; they do not establish the pinned container's build result.

`deploy/check_cuda_artifacts.py` inspects every executable's ELF dependencies,
cubins and production projection SASS. Shared CUDA/framework linkage and PTX
are rejected. `deploy/build_info.py` binds the same toolkit and AOT report to
source/package/compiler and executable hashes. Its result is installed at
`/usr/share/dwarfstar/build-info.json`.

Before the final image can build, `deploy/check_runtime_linkage.py` verifies
all six executable hashes and traces the trusted native ELF loader's library
resolution. Missing libraries fail even if `ldd` exits zero. This runs neither
the executable's main function nor CUDA initialization or kernels. Its report
is installed at `/usr/share/dwarfstar/runtime-linkage.json`. GitHub Actions
exports build-info, AOT/raw inspection logs and runtime linkage in the
`ria-cuda-toolchain-COMMIT` artifact, retained for seven days. The separate
scratch exporter contains metadata only; the default Docker build remains the
runtime image. CI builds and publishes CPU, CUDA and setup images after the
offline gates and bounded model-free amd64 setup-transfer comparison. The setup
image uses the existing public `ria-cpu` package with a distinct `setup-sha-`
tag and immutable digest; it does not use the CUDA toolchain exporter.

NVIDIA's [CUDA 13.4 release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/#cuda-driver)
identify the corresponding driver branch as R615. CUDA 13.x [minor compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
has an R580 baseline with feature restrictions; it is not a physically qualified
RIA driver floor. Use a supported host driver for the GPU and exact image, plus
NVIDIA Container Toolkit. The host CUDA SDK is not used by these executables;
replacing it cannot update their statically linked runtime. Preserve
`NVIDIA_REQUIRE_CUDA`; the operator must select exactly the admitted GPU UUID.

Every toolkit/image change requires fresh hardware qualification and admission.
Strict math flags do not prove cross-toolchain bit identity: NVIDIA changed
[`erff` in CUDA 13.2](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/#cuda-math-release-13-2),
and RIA's vision GELU uses it. Build/loader/static evidence proves neither GPU
numerical fidelity nor performance. The [README](../README.md) and
[handoff](ria-handoff.md) carry only actually verified published image pins;
existing full-model/release qualification software gaps remain listed there.
