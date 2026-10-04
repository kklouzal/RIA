# RIA task ledger

Task: implement the reviewed RIA plan on the canonical antirez/ds4 fork, with complete available offline/static/build checks and a hardware handoff. The initial planning request is complete; the current implementation request supersedes its original scope.

Source: `/home/kklouzal/DwarfStar-Remote-Inference-Architecture.md`, 2,133 lines, 245,262 bytes; SHA-256 `14224cdb33476944111e14f69a5679f0597c192a44d67b48f048910f326f3f6e`. Planning date: 2026-10-03. Governing guidance: `/home/kklouzal/AGENTS.md` and the user's supplied instructions. RIA was empty at task start; no deeper AGENTS.md was present.

## Invariants

- The handoff is authoritative; source audits, proposed interfaces, working code, and hardware qualification are distinct evidence classes.
- Preserve all mandatory requirements, both expert executors, all three profiles, exact state/continuation, client CUDA computation, required Docker deployment and feature/release coverage.
- Do not invent source pins, budgets, hardware capabilities, performance, numerical tolerances or completed checks.
- Preserve the original handoff. Keep an explicit prerequisite/gate for unresolved external inputs; complete the planning work with available evidence.

## Work

- [x] Primary reads all source lines, with coverage recorded below.
- [x] Model/operator/source review: `planning/model-review.md`.
- [x] Executor/protocol/admission review: `planning/protocol-review.md`.
- [x] Deployment/delivery/release review: `planning/delivery-review.md`.
- [x] Integrate contracts, dependency order, deliverables, acceptance gates and measured decisions into `IMPLEMENTATION_PLAN.md`.
- [x] Build source-section/requirement traceability and verify coverage, internal links, no fabricated execution evidence, and original source hash.
- [x] Independently review the integrated plan, resolve findings and recheck changed sections with all three reviewers.

Planning status: complete. Software implementation W00–W12 and all runtime/model/physical qualification remain unexecuted project work. The user subsequently provided the fork and authorized commit/push; publish these reviewed documentation artifacts on `codex/implementation-plan` without changing the main branch.

## Active implementation request

The user now authorizes complete implementation and free GitHub Actions/GHCR automation. Do not download model weights or run live inference/hardware tests on this host; prepare the software and offline/static/build evidence for the friend's later hardware handoff. Required physical/numerical/performance release gates remain explicit and deferred by that instruction. Model reference source and compact metadata may be inspected without acquiring checkpoints.

Implementation branch: `codex/ria-implementation`, based on planning commit `e2af41cc173b76cba26321e0d18ba7c46c5ddf77`. Runtime source changes, offline fixture tests, static analysis, Ruff, native/CPU/CUDA/container builds, CI and GHCR publication are authorized. Preserve unrelated donor functionality and do not claim deferred gates passed.

- [x] Freeze shared descriptors/source contracts and implement native boundary validation, identities and protocol.
- [x] Implement bounded prepared tensor ingestion/profile preparation and strict schemas/tooling.
- [x] Implement CPU/CUDA expert math, admission, caches and server role.
- [x] Integrate the CUDA graph with bounded client backing, exact state, remote work and features.
- [x] Implement probes/admin/deployment renderer/templates/containers/runbooks.
- [x] Add offline verification and free CI/GHCR build/publication, run available checks and fix findings.
- [x] Complete integration review, preserve evidence and commit/push the handoff.

## Read coverage and checkpoint

- Primary consumed all lines 1–2133 in consecutive reads: 1–305, 306–610, 611–856, 857–1086, 1087–1331, 1332–1551, 1552–1792, 1793–1927 and 1928–2133. Separately reread 434–453 to recover the only truncated ownership table.
- Remaining primary read: none.
- Three independent reviewers are assigned disjoint principal section ranges with shared source-register access, as permitted by D11.

## Evidence and decisions

- Verified RIA directory exists and is initially empty; verified source size, headings and SHA-256. No project build/test command exists yet.
- The implementation plan will preserve full-reference scheduling as the correctness baseline; faster exact CED scheduling requires equivalence evidence.
- Numerical operator contracts and an independent strict W4A4 fixture precede distributed optimization; native W4A8 cannot stand in for that oracle.
- Source/model/toolchain pins and actual hardware capacity remain implementation prerequisites, not facts established by this planning pass.
- Read-only development-host evidence is recorded in `planning/development-host.json`: ARM64 Linux, GB10 compute capability 12.1, approximately 121.6 GiB total host memory, Docker Engine 29.6.2, no nvcc on PATH. This is not the specified x86-64/RTX 5090 release environment; installed tooling is not a qualified dependency lock.
- Bounded primary-source browsing confirmed selected base and target model/card availability. The canonical base working tree was subsequently materialized after the user's fork instruction; complete dependency/artifact acquisition, kernel compilation, model evaluation and production-container tests have not been performed.
- User steering: the canonical repository is `antirez/ds4`. Adopt it as the implementation base, superseding the handoff's selection of `stefandsl/DwarfStar`; preserve the handoff unchanged and treat stefandsl only as an optional donor requiring an evidenced need. All other target/ownership/numerical/deployment requirements remain authoritative. Reviewers notified; plan/source-lock/refactor boundaries must reflect this correction.
- User subsequently created `kklouzal/RIA` and authorized repository work with commit/push. Verified GitHub fork parent is `antirez/ds4` and authenticated account has push permission. Materialized the base in the existing RIA directory while preserving planning files. Both observed main remote heads and the initial checkout are `0aaea5a238fb41a35106a551e73c8409dfb751ac`; origin/upstream are configured accordingly. Full source/dependency/model/toolchain qualification is still W00/W02 work.
- Read the checkout's `AGENT.md` and `CONTRIBUTING.md`; no deeper `AGENTS.md` was present. Native C host modules/CUDA remain the chosen integration contract. The user's production CPU expert and strict RAM/state requirements supersede conflicting donor reference-only CPU/disk defaults for the scoped RIA path.
- Preserved exact source bytes in `docs/ria-specification.md`, with the same source hash; the plan records the user's canonical-repository correction explicitly instead of altering historical input.
- Resolved integrated-review findings: full-reference assembly belongs to W08 after W04 contracts and W05 manager fixtures; early operator/transfer/TLS calibration and W10a admission precede full-bank/full-model execution; W11 evidence feeds W10b requalification. Dependency diagram is acyclic (15 nodes, 36 edges).
- Protocol decisions now explicitly protect progress bytes, separate shared credit charges from endpoint allocations, retain established bulk until whole binding Close, fail ambiguous bulk Bind through fresh binding, distinguish epoch/private generation/invocation scopes, retire counters before wrap and require fresh-session replay after a failed required operation.
- Exact FP8 scale/group domains and 2/4 GiB plus beyond-1-TiB checked-address gates are explicit. Independent reviewers rechecked corrections and reported no residual issue in their revised scope.
- `planning/verification.json` retains 13 passing planning-artifact checks: source/snapshot identity, W00–W12 and all 15 requirements/23 sections, FP8/address gates, acyclic/ordered dependencies, document fences/local links, development evidence JSON and no inherited runtime-source edits. These are structural/planning checks, not runtime correctness/performance results.
- Staged whitespace check flags seven intentional two-space Markdown hard breaks at lines 5–11 of the byte-identical historical specification. Triaged without modifying that input; the scoped whitespace check for all newly authored documentation passes (exit 0). No runtime source/build/test command was changed or executed.

## Historical prerequisites at the end of planning

- W00/W02: complete coherent source/model/publisher/kernel/toolchain/image pins, import provenance and licenses; independent strict W4A4 oracle and preregistered profile/operator/quality policies.
- W01/W04/W05: actual verified tensor inventory and prepared populations; complete operation/state dependencies and measured growth/peak bounds; bounded exact prefill/continuation.
- W02/W06/W10/W12: actual x86-64 CPU/NUMA and RTX 5090/server GPU hosts, sufficient profile/reference/replica capacities, required Compose-v2 environment, network/PKI/paths/permissions and real probes.
- W03/W07/W10/W11: measured workspace and protected resources, schema-bound credit costs, complete lifecycle tables, fixture calibration, context/feature/resource/deadline inputs and operational targets where applicable.
- All required builds, operator/model/TLS/Docker/physical-host tests, fidelity/capacity measurements, candidate comparisons and at-least-one-hour engineering soak remain unrun. The implementation plan owns their dependencies/gates; none is counted as a pass.

## Implementation checkpoint — 2026-10-03

- Source pins and compact publisher metadata are recorded under locks/; no checkpoint weights have been downloaded. Python preparation, trusted metadata pages, full-population profile calibration, compact client extraction and deployment schemas are implemented and undergoing final integration checks.
- Native JCS, protocol, admission, TLS and shared contracts pass GCC/Clang, ASan/UBSan and isolated offline mTLS fixtures. These are synthetic boundary checks, not live inference qualification. Full evidence is retained under build/ria/evidence/.
- CPU expert math and independent profile oracles pass offline checks; CUDA expert and client graph compile for SM120a. CUDA production matrix acceleration, full graph/frontends, vision fidelity and executable linking remain unresolved work. Compilation is not a GPU execution or numerical/performance pass.
- Parent-owned tensor ingestion authenticates root/pages/shards/chunks, copies active bytes into protected anonymous RAM, validates safetensors boundaries and exposes immutable views. A prepared Python fixture loads in C. Generic exact private-state paging fixtures pass. Further integration, sanitizer and adversarial artifact review remain open.
- Bank descriptors, service/deployment identity checks and persistent remote binding code compile with strict C diagnostics; executable server, administrative CLI, probes, containers and CI remain in progress. No running production service is claimed.
- Installed development tooling includes Ruff, Clang/static analysis, C sanitizers, OpenSSL/NUMA/Oniguruma headers and CUDA 13.1 compiler/disassembler. This ARM64 CC12.1 host remains outside the declared x86-64/RTX5090 release region. No model/GPU live test will run here.
- Root owns integration/probe/CLI/build/deployment/CI and evidence; native_contracts owns frontend/tokenizer/prompt/evaluator; expert_math owns CUDA graph/vision; artifact_tools owns preparation/schema/renderer, then expert-service integration. Shared interfaces are coordinated explicitly.
- GitHub repository push is authorized. Read access to Actions permission settings returned HTTP403 for the current fine-grained token; workflow publication/CI access is not yet tested and must be reported precisely if blocked.
- All physical, fidelity, capacity, candidate-performance and at-least-one-hour soak gates remain deferred by the user. Required preregistered thresholds and actual host admission evidence must be supplied before a deployment can be admitted.

## Integration checkpoint — 2026-10-03

- Artifact tooling checkpoint: `locks/verification/artifact-tools.json` records 28 passing offline Python checks, Ruff, source/derived metadata verification and actual native admission integration (fixture host peak 13,312 bytes). Subsequent per-expert page-alignment changes exposed a client padding-record regression; it remains open until fixed and reverified.
- Native CLI `bin/ds4ctl` builds and supports strict plan/validate/probe/health/drain. Bounded probe configuration/publication fixtures pass without running a physical probe. The common runtime restriction inspector is being integrated into role startup; no target-host observation has been manufactured.
- Shared Unix admin code compiles and offline fixtures pass peer authorization, strict malformed input, actual callback drain and finite shutdown. Client/expert consumers are integrating callbacks; neither role is claimed ready yet.
- Production CUDA expert code now dispatches actual native BF16/FP8/NVFP4 MMA, with generated SASS instructions independently inspected by the math agent. Source cast/quantization contracts and fragment-layout fixtures remain under review; no kernel was launched here. The client graph requires the exact RTX5090/CC12.0/UUID contract.
- Image recipes, strict role Compose templates, pinned Engine-derived NUMA seccomp generation and hash-locked isolated Python test environment are present. Native source/toolchain build metadata and pinned-action CI/GHCR workflow are drafted. Container builds, effective/actual deployment review, CI execution and publication remain open.
- Broad integration checks found and are fixing strict frontend compiler diagnostics, native analyzer boundary findings, the client extraction regression and a Cppcheck loop-shadowing false finding. Source mutations invalidate earlier build results; final verification will rerun after integration stabilizes.
- Shared ownership update: artifact_tools owns server/NUMA and coordinated service/tensor extensions for explicit budgets and NUMA binding before first touch; native_contracts owns API/tokenizer/prompt/engine/evaluator/admin client callbacks; expert_math owns graph/vision/native MMA; root owns runtime/probe/admin/CLI/build/container/CI integration and final evidence.

## Offline integration checkpoint — 2026-10-04 UTC

- CPU expert service, CPU/CUDA administrative tools, CUDA client CLI/HTTP/evaluator and standalone bounded operator qualifier compile and link. SM120a cubins are present and PTX is absent in the inspected qualifier; no CUDA kernel was launched. Source/build mutations still require a final repeat.
- Sixteen cross-language Python-to-native TensorStore fixtures pass, including actual authenticated loading, header/descriptor shape equality, rank/stride/overflow, aliases, paths, truncation and hash corruption. Twenty qualification comparison/admission fixtures passed before mandatory raw-provenance fields were added; those inputs must be migrated and reverified.
- Actual Compose config parsing exposed decimal-string memory limits and empty default core/network/placement objects omitted by mocked fixtures. Those production normalization corrections and actual Engine inspection policy checks are now being integrated. A CPU image was created and inspected without starting it, then removed; no production service was run.
- Review found and corrected NUMA/seccomp drift: bind mode is plain MPOL_BIND=2 with no migration flags, and self-query batches are at most64 pages. A pure fixture now compares the production constants to the generated allow rules.
- Exact native child-lifetime peak RSS is obtained with owned wait4, with25ms sampled early containment and finite process-group cleanup; four subprocess/host-boundary fixtures pass. Physical container cgroup limits remain the hard memory enclosure.
- Existing donor CPU binaries built; offline Engram, GGUF extent, web recovery, GPU-argument, prompt-prefix, sampling and evaluation self-checks pass. No model file was acquired.
- All25 owned C modules passed Clang analyzer in an integration run, but source changes and an in-progress transport compiler warning invalidated that aggregate. Cppcheck2.13 builtin analysis stalls on TensorStore; alternate parser investigation is ongoing and no incomplete analysis is counted as a pass.
- Remaining software work: complete paired model-free TLS qualifier and preregistered five-component proof producer, handoff/replay/soak driver, final docs, stable-source checks, final containers, commit/push and actual hosted CI/GHCR verification. Physical NUMA/GPU/model parity/capacity/performance/soak remain deferred by the user.

## Final software integration checkpoint — 2026-10-04 UTC

- All previously open production modules and offline handoff producers are implemented. The actual paired mTLS qualifier, joint preregistration, same-image managed fixture containers, raw-backed five-component proof derivation, source preparation, CUDA frontends/evaluator, bounded replay/soak driver and inspected launch controller are present. Physical matrix proofs additionally require preregistered measured contracts; content hashes are integrity checks, not attestations of execution.
- A fresh forced build of all three CPU and six CUDA executables passed. Production projection functions in every CUDA executable contain the required BF16 HMMA, FP8 QMMA and NVFP4 OMMA instructions; only SM120a cubins are retained, with no PTX. No kernel was launched.
- The integration suite passed 258 offline Python tests before the last physical-proof validator addition. Strict Ruff E/F/B checks passed. ASan/UBSan with leak detection passed all native contract, expert, state, graph, probe, admin, server and actual socket fixtures, plus nine frontend, sixteen TensorStore, six qualifier and three real paired TLS fixtures. The final Python/schema/static aggregate still needs to include the final proof validator.
- Donor CPU binaries build, and sampling, Engram, GGUF extents, GPU-argument parsing, prompt-prefix, web recovery and evaluator case/extractor checks pass without a model. Those fixtures do not establish live donor or RIA inference correctness.
- Cppcheck2.13's stall and alternate-parser crash were resolved by pinning and building isolated official Cppcheck2.22.0. Its full CPU/CUDA-host analyses pass with narrow documented POSIX mmap model suppressions. The source is unmodified, hash checked and kept outside production dependencies. The installer now kills and reaps timed-out compiler process groups.
- Review fixed stale OpenSSL error queues, fail-stop TLS cleanup, SIGPIPE on OpenSSL socket writes, large-common-offset NLL cancellation, scope/environment digest cycles, stale actual host reports, candidate-operator identity checks, release profile/executor identity checks, build flag overrides and probe-only CUDA instruction certification. Negative fixtures cover these boundaries.
- All native/source owners are frozen. Remaining work is final physical-proof integration, generated schemas, stable-source aggregate checks, refreshed local container builds, tracked verification report, commit/push and actual hosted amd64 Actions/GHCR verification. Model weights, hardware execution, fidelity, physical matrix, resource/performance evidence and the one-hour soak remain deferred by the user's instruction.

## Handoff verification checkpoint — 2026-10-04 UTC

- Final proof integration and generated schemas are complete. All360 offline Python tests and Ruff pass. The stable-source static aggregate has34 passing checks, including zero findings in all25 owned Clang analyzers and both Cppcheck2.22.0 configurations. Final native compilation, AOT SASS inspection, available donor regressions and ASan/UBSan/leaks pass.
- Both ARM64 build-only containers were rebuilt after source freeze; stopped-container inspection confirms exact source/build identities and UID10001. No production container or GPU workload was started. Hosted amd64 builds/GHCR publication are checked after the authorized branch push; actual results are recorded in GitHub Actions and the final handoff response.
- ThreadSanitizer compiled but could not initialize on this host (unexpected memory mapping, exit66, both PIE and non-PIE). It is recorded as unavailable, never passed. The single Pillow palette-transparency warning is from the intentional reference fixture and its pixel comparison passes.
- `docs/ria-offline-verification.md`, `locks/verification/offline-integration.json` and ignored complete logs retain the evidence and limits. The source/credential/artifact audit accepts only the authenticated locally generated integer Engram lookup metadata as a safetensors file; no neural weights or private credentials are committed. The unchanged original specification retains its exact input hash.

## Hosted ThreadSanitizer gate checkpoint — 2026-10-04 UTC

- Added the separate `make ria-thread-sanitize` gate after ASan/UBSan in the hosted Ubuntu24 workflow. Explicit `clang-18` and `libclang-rt-18-dev` dependencies build both existing model-free admin/server fixtures and all their owned C dependencies with strict diagnostics, `-O1 -fsanitize=thread -fPIE -pie -fno-omit-frame-pointer -pthread`. Serving binaries and runtime restrictions remain unchanged.
- Each fixture runs with `TSAN_OPTIONS=halt_on_error=1:exitcode=66 timeout --signal=TERM --kill-after=5s 60s setarch -R build/ria/thread-sanitize/{admin,server}`. ASLR changes affect only that diagnostic process. Races, initialization failures, unsupported personality changes and timeouts fail the gate; there are no suppressions, retries, silent skips or whole-host sysctl changes.
- Reviewed [LLVM18.1.3's Linux TSan address-layout implementation](https://github.com/llvm/llvm-project/blob/llvmorg-18.1.3/compiler-rt/lib/tsan/rtl/tsan_platform_linux.cpp#L231-L277), which itself sets `ADDR_NO_RANDOMIZE` and re-executes for incompatible high-entropy ASLR, [util-linux2.39.3's per-process implementation](https://github.com/util-linux/util-linux/blob/v2.39.3/sys-utils/setarch.c#L467-L468), and [GitHub's hosted VM contract](https://docs.github.com/en/actions/concepts/runners/github-hosted-runners). This supports the chosen diagnostic setup; hosted success still requires actual execution.
- After installing the missing Clang18 runtime package, strict compilation and both instrumented fixtures passed locally on aarch64 with this exact gate (exit0, zero TSan findings). ELF inspection confirms PIE and native `__tsan_init`/access instrumentation. Workflow actionlint, existing unsanitized admin/server fixtures and whitespace checks pass. The earlier optional TSan mapping failure remains historical evidence for that earlier compiler/setup; it is not the result of this new Clang18 gate.
- Complete logs are `build/ria/evidence/thread-sanitize-local-{build,run}.log`, `thread-sanitize-{admin,server,environment,actionlint}.log` and `thread-sanitize-local-runtime-install.log`. Hosted amd64 TSan execution remains unrun at this checkpoint. These fixtures do not execute a model, physical NUMA policy, CUDA or GPU work, and do not prove complete production concurrency coverage.

## Hosted integration corrections — 2026-10-04 UTC

- Actual initial hosted run37180927646 rejected six legitimate Compose2 mount serializations and one test's fixed128MiB child RSS budget; all353 other Python tests passed. That run was a failure, not a release/build pass. Failure logs are retained locally; the workflow now creates evidence before tool installation and captures installer/native fixture output with checked pipelines.
- Primary-source review distinguishes Compose2's Boolean/omitempty false omission from Compose5's OptOut/omitzero true omission. Only verified released Compose2 versions through2.40.3 can accept a present empty bind object. New/unknown versions require explicit Boolean false; missing/null binds, true/numeric creation flags, unknown options and access/path changes fail. Actual config-only fixtures record the actual CLI version into synthetic host metadata. The sealed source review is `locks/verification/compose-bind.json`.
- Linux wait4 legitimately accounts the inherited child fork image before exec. The test now derives startup allowance from the actual pytest parent RSS rather than assuming an empty parent; production admission caps remain strict. The supervisor's monotonic deadline now includes process creation, with a delayed-spawn containment/reaping regression fixture.
- All370 offline Python tests and Ruff pass after these corrections. Both current-source ARM64 build-only image recipes compile. Clang18 ThreadSanitizer admin/server fixtures pass; earlier GCC initialization failures are preserved as historical diagnostic evidence. Actual amd64 CI and publication require the corrected authorized push.

## Completed software handoff — 2026-10-04 UTC

- Implementation commit `c987a34618efe6ccc4aca5fed26e3db0aac863be` is pushed on `codex/ria-implementation`. [Actual hosted run37181861323](https://github.com/kklouzal/RIA/actions/runs/37181861323) completed successfully: offline verification, CPU container and CUDA container jobs all passed. Hosted370 fixtures,34 static checks, all25 zero-finding Clang analyses, ASan/UBSan/leaks, Clang18 ThreadSanitizer and unchanged schema regeneration are independently downloaded and checked. Current software inputs match the tested hosted source snapshot.
- Both actual Linux amd64 images were published with SBOM and SLSA provenance. CPU index digest: `sha256:7dfd92857d8ba35ade385c0bce08fa9f81254381898ab9cd0365d99c6803374a`; CUDA index digest: `sha256:aa1db3b05d561198d4d62ec76f74fa94c3c3bff3e2396c9472f3a9b4ef93dcf5`. Direct anonymous registry requests verified exact index hashes, amd64 manifests and both attestation predicate descriptors without pulling model data or executing an image.
- `locks/verification/hosted-publication.json` preserves actual job URLs, implementation commit, immutable image identities, downloaded log hashes and explicit unqualified status. The operational handoff lists exact image references and the tested source branch. Documentation-only publication records do not change the tested runtime inputs or trigger another build.
- All authorized software implementation, available offline/static/build verification and CI/GHCR publication work is complete. No model weights were downloaded and no live model/GPU/target probe was run here. Actual source/model fidelity, GPU numerical execution, physical NUMA/capacity/performance, all540 cells/G01–G28 and the one-hour soak remain deferred to the friend's hardware, as the user directed. These are unexecuted qualification gates, not passed results.
