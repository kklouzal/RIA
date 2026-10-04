# Offline implementation verification

Verified 2026-10-04 UTC on the development ARM64 Linux host. The declared release
target remains Linux x86-64 with the specified physical client/server hardware.
No checkpoint weights were downloaded, no CUDA kernel was launched, and no
live model inference, physical target probe or engineering soak was run.

The independent second review corrected numerical scheduling, preparation,
protocol, resource accounting and ownership defects. Its frozen local sources
passed413 Python tests, the34-case production socket lifecycle fixture, Ruff,
all35 static checks (26 owned C modules, zero Clang findings), ASan/UBSan/leaks,
and all three Clang18 ThreadSanitizer fixtures. All three CPU and six SM120a
CUDA executables compiled and linked; production HMMA/QMMA/OMMA instructions
were found in every CUDA executable and PTX was absent. Available donor CPU
regressions passed. Exact commands, source identities and complete log hashes
are recorded in [the review checkpoint](../locks/verification/implementation-review.json).

The [corrected hosted amd64 run](https://github.com/kklouzal/RIA/actions/runs/37188961273)
passed all three jobs for commit `f37d418ed146283a98004e02925621c82573b9e9`:
the 413 Python tests, 35 static checks, 26 zero-finding Clang analyses,
ASan/UBSan/leaks, all three Clang18 ThreadSanitizer fixtures, unchanged schema
regeneration, and both native container builds. Downloaded evidence hashes and
all 3714 tracked source files matched before the documentation-only checkpoint.
Anonymous registry metadata verified the immutable amd64 image/config identities,
source revision, runtime UID and SBOM/SLSA descriptors. No image filesystem layers
were pulled or executed. The [review publication record](../locks/verification/implementation-review-publication.json)
retains those results; the [handoff](ria-handoff.md) lists the corrected images.

These checks do not establish complete implementation. Grouped prompt prefill
and complete semantic physical matrix/fault/gate producers remain missing
software. [The review report](ria-implementation-review.md) and
[producer readiness](../planning/qualification-software-readiness.json) list
the remaining work separately from hardware execution. Generic numeric reports
grant no release coverage. The prior complete-software claim is withdrawn.

The earlier pre-review baseline checks below are retained as historical evidence:

| Check | Result |
| --- | --- |
| Independent Python, boundary and cross-language fixtures | 370 passed |
| Ruff E/F/B rules, Python compilation, generated schemas | Passed |
| GCC strict C99 and CUDA host branches | Passed |
| Clang static analyzer, all 25 owned C modules | Zero findings |
| Isolated, checksum-pinned Cppcheck 2.22.0, CPU/CUDA-host analyses | Zero findings |
| ASan, UBSan and leak detection | Passed |
| Clang 18 ThreadSanitizer, model-free admin/server fixtures | Passed, with per-process diagnostic ASLR policy |
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
palette-transparency input; the pixel comparison passes. Earlier GCC
ThreadSanitizer attempts failed to initialize with `unexpected memory mapping`
(exit 66). Installing the Clang 18 runtime and using its supported per-process
diagnostic ASLR policy resolved that limitation: both instrumented fixtures pass
without suppressions or retries. The separate `make ria-thread-sanitize` target
is also required in CI. These fixtures do not prove all production interleavings.

The [actual hosted amd64 run](https://github.com/kklouzal/RIA/actions/runs/37181861323)
passed all three jobs for implementation commit
`c987a34618efe6ccc4aca5fed26e3db0aac863be`: the370 fixtures,34 static checks,
ASan/UBSan/leaks, Clang18 ThreadSanitizer, schema regeneration and both containers.
The images were published to GHCR with SBOM and SLSA provenance. Direct anonymous
registry requests verified the exact OCI index hashes, amd64 manifests and
attestation descriptors. The tracked
[hosted publication record](../locks/verification/hosted-publication.json)
contains job URLs, immutable digests and downloaded evidence hashes.
Documentation and verification-record-only pushes skip rebuilding.

GPU numerical behavior, source/model fidelity, physical capacity and NUMA,
the 540-cell matrix, G01–G28, candidate performance and the required one-hour soak
remain unexecuted. Follow the [hardware handoff](ria-handoff.md) for bootstrap,
initial admission, measured execution and final release qualification.
