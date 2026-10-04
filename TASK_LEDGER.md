# RIA planning ledger

Task: read the complete implementation handoff and produce a reasoned, reviewable implementation plan. Software implementation is outside this request.

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

## Implementation prerequisites still open

- W00/W02: complete coherent source/model/publisher/kernel/toolchain/image pins, import provenance and licenses; independent strict W4A4 oracle and preregistered profile/operator/quality policies.
- W01/W04/W05: actual verified tensor inventory and prepared populations; complete operation/state dependencies and measured growth/peak bounds; bounded exact prefill/continuation.
- W02/W06/W10/W12: actual x86-64 CPU/NUMA and RTX 5090/server GPU hosts, sufficient profile/reference/replica capacities, required Compose-v2 environment, network/PKI/paths/permissions and real probes.
- W03/W07/W10/W11: measured workspace and protected resources, schema-bound credit costs, complete lifecycle tables, fixture calibration, context/feature/resource/deadline inputs and operational targets where applicable.
- All required builds, operator/model/TLS/Docker/physical-host tests, fidelity/capacity measurements, candidate comparisons and at-least-one-hour engineering soak remain unrun. The implementation plan owns their dependencies/gates; none is counted as a pass.
