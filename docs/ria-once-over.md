# Full implementation once-over

Reviewed from `fc7c5d86e59b26376fd0191ede120d34526def34` against the complete
implementation specification, SHA256
`14224cdb33476944111e14f69a5679f0597c192a44d67b48f048910f326f3f6e`.
Independent reviews covered graph/operator/state/image semantics, native
protocol/frontend/startup ownership, artifact/deployment/CI boundaries and
qualification producers. All reproduced findings below were corrected.
No further actionable defect remained in those scoped source reviews.

| Finding | Correction and regression |
| --- | --- |
| Artifact output could escape through directory links | Held no-follow directory descriptors, descriptor-relative durable publication, non-replacing immutable publication, ownership-safe cleanup; adversarial link/replacement/mutation cases |
| Official preparation emitted incompatible source revisions | Recipe/root/operator bind the native revision; independent NVIDIA publisher identity stays in provenance; actual native loader accepts the resulting package |
| Tensor metadata underestimated transient JSON allocations | Shared parser/duplicate-key/canonical equations and retained-DOM accounting; actual native allocation-boundary fixtures |
| Manifest, index, shard, tokenizer and bearer FIFOs could stall before type checks | Nonblocking no-follow opens, bounded regular-file validation and stable reads; bounded FIFO and mutation regressions |
| Frontend accepted embedded-NUL configuration strings | Reject incomplete C string representations before use; actual frontend policy fixtures |
| Startup ignored cancellation during population | Explicit construction callback, phase and at-most4MiB TensorStore progress checks, typed cancellation and cleanup; actual constructor with explicit model/runtime boundary shims |
| Expired channels left closed descriptors in poll snapshots | Discard the snapshot before polling/accepting; three actual-loop regressions plus existing real TLS/socket cases |
| Health remained ready after SIGTERM and signal words lacked a concurrency contract | Lock-free compiler atomics, unique listener-close ownership, synchronized readiness and cancellation-aware health; actual frontend cross-thread/second-signal fixtures |
| CPU/CUDA expert stride boundaries disagreed | One exact-span equation; normalize unused one-row CUDA pitches; production CUDA validation prefix with no device operation |
| Saved-logit evidence could identify bytes different from compared rows | Hash bytes actually consumed through held regular descriptors and verify descriptor/path stability; ABA and in-place restoration regressions |
| Malformed teacher/policy JSON escaped typed errors | Validate object/schema boundaries before membership; typed errors with no output publication |
| Strict verification exposed missing printf forwarding contracts | Correct indexed compiler attributes, unchanged warning flags; full GCC/Clang source checks and invalid-call rejection |
| Prefill test read a peer worker before initialization | Bind recorder TLS to its executing worker; deterministically force worker1-first startup; old predicate fails, corrected ASan/TSan/NDEBUG fixtures pass |

The [graph review](../planning/once-over-graph-review.md),
[protocol review](../planning/once-over-protocol-review.md),
[delivery review](../planning/once-over-delivery-review.md) and
[qualification review](../planning/once-over-qualification-review.md) retain
scope, contracts, focused regressions and limitations.

Final frozen local gates passed:492 Python tests,42 actual socket lifecycle
cases, Ruff,36 static checks including27 zero-finding Clang C analyses,
ASan/UBSan/leak detection and all five Clang18 TSan fixtures. Three CPU and six
SM120a CUDA executables built; production HMMA/QMMA/OMMA were present and PTX
was absent. All45 generated schemas were unchanged. All frozen source bytes
matched across these gates. The [local checkpoint](../locks/verification/once-over.json)
records exact commands, source/log identities and the preserved failed runs
that led to the printf and worker-fixture corrections. The Pillow warning
comes from the intentional palette-transparency reference case; it passes.

Hosted amd64 publication is recorded separately after its jobs complete. Local
ARM64/compiler/fixture results do not establish physical target readiness.
No checkpoint download, GPU query/initialization/kernel, live model execution,
physical probe, controlled performance comparison or soak was performed.
There is no claim of target performance superiority or cross-platform bit identity.

Complete semantic540-cell matrix and fault/gate producers and validators
remain missing software. Full-model preregistration still needs an explicit
oracle, corpus, selected positions, profile and determinism contract plus the
corresponding execution producers. The existing threshold/identity/objective
policy does not implement that registration. These gaps remain distinct from
external inputs and hardware tests. [Producer readiness](../planning/qualification-software-readiness.json)
returns exit1 with `release_ready=false`; generic numeric reports grant no
release coverage. Full software implementation and final release qualification
remain incomplete. See the [handoff](ria-handoff.md) for provisioned target work.
