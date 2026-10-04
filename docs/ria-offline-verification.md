# Offline implementation verification

Verified 2026-10-04 UTC on the development ARM64 Linux host. The declared release
target remains Linux x86-64 with the specified physical client/server hardware.
No checkpoint weights were downloaded, no CUDA kernel was launched, and no
live model inference, physical target probe or engineering soak was run.

The final available offline checks passed:

| Check | Result |
| --- | --- |
| Independent Python, boundary and cross-language fixtures | 360 passed |
| Ruff E/F/B rules, Python compilation, generated schemas | Passed |
| GCC strict C99 and CUDA host branches | Passed |
| Clang static analyzer, all 25 owned C modules | Zero findings |
| Isolated, checksum-pinned Cppcheck 2.22.0, CPU/CUDA-host analyses | Zero findings |
| ASan, UBSan and leak detection | Passed |
| Paired TLS, cancellation, teardown and malformed protocol fixtures | Passed normally and under sanitizers |
| CPU binaries | Three compiled and linked |
| CUDA binaries | Six compiled and linked for SM120a |
| Production CUDA projection SASS | Required HMMA/QMMA/OMMA found in every executable; PTX absent |
| CPU/CUDA container recipes | Both built for ARM64 without GPU/checkpoint; image sources matched working sources |
| Workflow/container lint and pinned source metadata | Passed |
| Available donor CPU regressions | Sampling, Engram, GGUF extents, argument parsing, prompt prefixes, web recovery, evaluator checks passed |
| Original specification and artifact audit | Source hash unchanged; no weights/private credentials committed |

The complete static aggregate contains 34 successful checks with unchanged
sources during execution. Full command logs and machine-readable evidence are
under ignored `build/ria/evidence/`; GitHub Actions retains its own corresponding
evidence artifacts for seven days. The compact tracked
[checkpoint](../locks/verification/offline-integration.json) records log hashes,
source identities, exact local image IDs and the scope of these results.

The Pillow reference emits its documented warning for a deliberately exercised
palette-transparency input; the pixel comparison passes. ThreadSanitizer binaries
compiled, but both PIE and non-PIE attempts failed to initialize on this ARM64
host with `unexpected memory mapping` (exit 66). Those attempts are unavailable
instrumentation, not successful race checks. The concurrency fixtures and
ASan/UBSan tests do not replace a working race detector on a supported test host.

The workflow builds the actual Linux amd64 images on standard hosted runners and
publishes immutable commit-tagged CPU/CUDA GHCR images with SBOM and provenance
on authorized repository pushes. Publication/run results belong to the actual
[Actions execution](https://github.com/kklouzal/RIA/actions), not this local build
record. Documentation and verification-record-only pushes skip rebuilding.

GPU numerical behavior, source/model fidelity, physical capacity and NUMA,
the 540-cell matrix, G01–G28, candidate performance and the required one-hour soak
remain unexecuted. Follow the [hardware handoff](ria-handoff.md) for bootstrap,
initial admission, measured execution and final release qualification.
