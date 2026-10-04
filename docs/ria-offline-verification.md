# Offline implementation verification

Container-managed setup passed **1,043 Python fixtures with no privileged
skips**, Ruff, 37 static checks and 27 zero-finding Clang analyses. All 47 schema
files regenerate unchanged. Full ASan/UBSan/leak checks, five Clang18 TSan
fixtures, three CPU binaries and six SM120a-only CUDA binaries passed. The
[four-job hosted build](https://github.com/kklouzal/RIA/actions/runs/37240824381)
passed for `d0288edbaa35c55d3fae43988bc69568cbc5615e` and published matching
Linux amd64 CPU, CUDA and administrative setup images. Downloaded evidence
matches all 3,782 committed source files and 3,713 frozen runtime inputs;
actual CUDA 13.4.2 compiler/header/static-runtime/SDK, AOT/raw inspection and
final-image ELF dependency identities were verified without GPU execution.
Anonymous registry metadata verifies all three distinct images, their source,
role/entrypoint, serving UID10001 versus administrative root, and SPDX/SLSA
descriptors. Filesystem/attestation layers and signatures were not downloaded
or verified. See [local evidence](../locks/verification/container-setup.json),
[publication evidence](../locks/verification/container-setup-publication.json),
[automated setup](ria-container-setup.md) and [handoff](ria-handoff.md).

The hosted model-free transfer comparison independently verified 24 paired
64MiB fetch/upload cases across TLS/plaintext and advisory-cold/warm cases, with three
before/after samples per region. Median complete-transfer speedups over the
retained test-only prototype were 10.13–11.10×. Every case met its 60-second
transfer and 256MiB sampled combined-RSS budgets; larger chunks raised maximum
sampled RSS from 67,350,528 to 100,380,672 bytes. Five-millisecond sampling can
miss shorter peaks, and three samples cannot establish population p95/p99.
This is isolated amd64 loopback setup-transfer evidence, not physical LAN or
model performance. The authoritative producer hashes both copied payloads;
the separate evidence reader verifies its source, report and independently
recomputed summaries, without claiming to rehash unretained payload files.

The first hosted attempt failed only on Python source-cache permissions after
root-owned fixture execution. A private per-invocation compilation cache fixes
that mixed-owner case without changing source ownership or weakening syntax
checks. Actual valid/invalid syntax regressions verify write denial, generated
cache ownership, cleanup, unchanged sources and retained failure status. The
failed artifact and corrective evidence remain bound in the local record.
These builds and initial-admission tools remain unqualified for final release;
no checkpoint/model/GPU/physical/soak execution occurred here, and the existing
full-model and semantic release qualification software gaps remain.

The preceding NVIDIA NGC CUDA migration passed 608 Python fixtures/Ruff, 37 static checks
and 27 zero-finding Clang analyses, all 45 unchanged schemas, and CPU/CUDA AOT
build gates. All three jobs in the [hosted NGC build](https://github.com/kklouzal/RIA/actions/runs/37227441606)
passed for `d9d0b6e1692c005772358abdd5ed01e1fc03cb4a`, repeating full
ASan/UBSan/leaks and five TSan fixtures on Linux amd64. The actual CUDA 13.4.2
builder exported compiler/header/static-runtime/SDK identities, six SM120a-only
binaries with no PTX/shared CUDA linkage, 25 raw inspection logs and successful
final-image ELF dependency resolution without main/GPU execution. The local
installed CUDA 13.1 compilation is recorded separately and does not prove the
new SDK build. See [local evidence](../locks/verification/ngc-cuda.json),
[publication evidence](../locks/verification/ngc-cuda-publication.json) and the
[toolchain guide](ria-cuda-toolchain.md). Images are build-verified; hardware,
model numerical/performance and full release qualification remain unrun.

The preceding optional-certificate implementation passed 560 Python fixtures, Ruff, all36
static checks/27 zero-finding Clang analyses, full ASan/UBSan/leak checking,
five Clang18 TSan fixtures, all45 unchanged generated schemas, three CPU and
six SM120a CUDA builds. Default mTLS and explicit certificate-free TCP both
exercise the production transport and server lifecycle; mode disagreement,
wrong peer/capability/session, framing/deadline, exact grants and shutdown
failures are covered. [The checkpoint](../locks/verification/optional-transport.json)
binds commands, source/log identities and the mirrored updated specification.
All three jobs in the [hosted optional-transport run](https://github.com/kklouzal/RIA/actions/runs/37223845971)
passed for `37756b57c885b46f16d431d3cf112518b4c4cf59`. Downloaded evidence matches
all3740 tracked source files and3680 frozen runtime inputs; anonymous registry
metadata verifies both new amd64 image/config/source/UID and SPDX/SLSA descriptor
identities without downloading filesystem or attestation layers. The
[publication record](../locks/verification/optional-transport-publication.json) and
[handoff](ria-handoff.md) bind the new pins. No checkpoint/model/GPU/physical/performance
execution or complete release qualification is claimed.

The preceding [full implementation once-over](ria-once-over.md) corrected additional
boundary/startup/lifecycle defects and passed 492 Python tests, 36 static checks,
ASan/UBSan/leaks, five TSan fixtures and CPU/SM120a build gates. Complete semantic
qualification software remains incomplete. Earlier checkpoints below retain their
historical scopes.

The [once-over hosted run](https://github.com/kklouzal/RIA/actions/runs/37215164710)
passed all three jobs for `3614e21aad6465c1af41a2fbeccad2991a6243d2`; [its publication
record](../locks/verification/once-over-publication.json) binds the preceding verified
amd64 images and immutable source/evidence identities.

Verified 2026-10-04 UTC on the development ARM64 Linux host. The declared release
target remains Linux x86-64 with the specified physical client/server hardware.
No checkpoint weights were downloaded, no CUDA kernel was launched, and no
live model inference, physical target probe or engineering soak was run.

The later specification 7.3 grouped-prefill implementation passed 430 Python
tests, 39 production socket lifecycle cases, Ruff, all 36 static checks, 27
zero-finding Clang C analyses, ASan/UBSan/leak detection and all four Clang18
ThreadSanitizer fixtures. Actual graph, engine and worker fixtures cover causal
row state, grouped original-slot scatter, mixed image rows, continuation,
aliasing, cancellation and failures. All three CPU and six SM120a CUDA
executables built; required projection instructions were present and PTX was
absent. Regenerating all 45 schemas produced no changes. Every frozen source
file's bytes matched across these checks.

The [prompt-prefill checkpoint](../locks/verification/prompt-prefill.json)
records commands, source/log identities and explicit limitations. See the
[prefill contract](ria-prompt-prefill.md) for configuration regeneration and
numerical scope. This closes the scheduling software gap; it does not establish
GPU/model parity, performance superiority or complete release qualification.

All three jobs in the [prefill hosted run](https://github.com/kklouzal/RIA/actions/runs/37208400490)
passed for commit `97310cd58ebe3b7079c4bdc6acaa2413d701dea6`: the 430 Python
tests, 36 static checks, 27 zero-finding Clang analyses, ASan/UBSan/leaks, four
ThreadSanitizer fixtures, unchanged schemas and both native amd64 containers.
Downloaded evidence matched all 3,727 tracked source files in that immutable
commit and every locally validated changed runtime input. Anonymous registry
metadata verified both image/config identities, revision, UID and SPDX/SLSA
descriptors without downloading filesystem or attestation layers. The
[prefill publication record](../locks/verification/prompt-prefill-publication.json)
retains job URLs, immutable digests and complete downloaded evidence hashes;
the [handoff](ria-handoff.md) lists these current images.

The preceding independent second review corrected numerical scheduling, preparation,
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

The later [grouped prompt-prefill implementation](ria-prompt-prefill.md) closes
that scheduling gap. Complete semantic physical matrix/fault/gate producers
remain missing software. [The review report](ria-implementation-review.md) and
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
