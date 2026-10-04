# Independent grouped prompt verification

Scope: specification 7.3 at `docs/ria-specification.md`, exact source revision
`2cba9e42aa026125f3ed06c6d98c1db82f7ca027`, production graph scheduling,
CPU/CUDA expert row domains, and server worker grouping. No checkpoint download,
CUDA runtime initialization, GPU execution, physical NUMA probe, or throughput
measurement was performed.

## Invariants checked

The new scheduler retains the pinned source's sequential attention realization:
one initial window slot, then 128 sparse ring slots with leading invalid holes.
The BF16 probability cast remains scoped to its original 64-slot tiles. Moving
independent FFN work does not authorize compacting these slots, reading future
compressed rows, using another row's selection/candidate set, or resetting an
incomplete compressor pool at a prompt boundary.

Each row retains its mHC residual/pre/post/comb data, normalized expert input,
Engram token/image history, original expert IDs, selected slots, coefficients,
and absolute causal identity. One original expert is evaluated over bounded
independent rows; its results scatter into the original row/slot. Routed sums
follow original expert ID order and the shared branch executes once per row.
All three expert profiles retain their individual activation quantizer domains
under row order and subgroup-size changes.

Cancellation, failed required expert work, nonfinite results, and incorrectly
rounded results poison the graph and restore the prior public chunk position.
Private caches may have changed and cannot be reused until a successful reset.
The worker request owner must retain input/output backing until every gathered
row is quiescent, including cancellation while an executor is active.

## Executable checks

`tests/ria/test_graph_prefill.c` includes the actual `ria/graph.c`, replacing CUDA
with bounded independent host arithmetic and a strict association oracle. It
compares grouped execution with the actual one-row graph caller at admitted
sizes 1, 2, 3, 8, 17, and 64 across 130 causal positions. The routing fixture
touches all 384 original experts, repeats experts across rows, varies original
slot order, and includes zero coefficients. It compares every logit position,
all 40 sparse windows, packed compressed/index history, incomplete pools,
selected indices, candidate block snapshots, and token/image history. Further
checks cover the next decode token, host/VRAM local membership and phase changes,
mixed image/text rows, odd continuation chunks, final-head-only execution,
invalid later-chunk tokens, invalid images, late subgroup failures, and
cancellation after initialization, attention, expert evaluation, scatter, and
the vocabulary head. Every poisoned case resets and compares against fresh
execution. Both prefill APIs also permit an exact final-row `last_logits` alias
and reject partial or earlier-row output overlap before any CUDA/state change.
The native fixtures use always-executed checks, so required test operations do
not disappear when language assertions are disabled. This is a
scheduling/state fixture, not learned CUDA numerical
qualification; at this length the source candidate block population is below
its 2,048-block selection cap.

`tests/ria/test_prefill_expert_oracle.py` uses the separate rational palettes and
explicit scalar casts in `tests/ria/reference_expert.py`. It verifies 64 rows of
real production CPU BF16/FP8/NVFP4 expert evaluation against that oracle, under
permuted original row order and microbatch sizes 1, 2, 3, 8, 16, and 64. A second
case extracts the exact production CUDA activation quantizer body, substitutes
only its scalar CUDA builtins, and runs it as host C. Widths 17, 33, and 65,
64 differently scaled rows, all three profiles, and microbatches 1, 3, 8, 17,
and 64 match the independent per-row quantizer exactly. CUDA libraries and
drivers are not linked by either case.

`tests/ria/test_server_prefill.c` executes the production worker loop and real
CPU expert executor with miniature verified BF16 matrices and explicit NUMA
affinity/lookup shims. One- and two-worker schedules process two interleaved
request owners, each with 64 rows and two experts. All 256 contributions retain
their exact original response slot and match a separate scalar BF16 oracle;
group sizes fit the fixed 2 MiB worker arena. Adversarial barriers cancel an
active group or fail its executor. Requests remain outstanding and keep their
borrowed backing until all group members and queued work finish. The shims do
not prove physical NUMA placement or target-shaped learned arithmetic.

Existing graph contract/step/reference tests retain their prior checks and now
declare mandatory `prefill_rows=1`; rejection cases cover zero, above-64, and
above-context bounds. The source reference fixture links the authoritative pure
prefill accounting helper. Independent review also identified two boundary
details subsequently corrected in production: whole-call text token validation
before the first chunk, each teacher-forced head's absolute causal position,
and an explicit checked output-alias contract.

## Local evidence

On the development ARM64 environment, strict GCC C99 fixture compilation,
normal graph and actual-worker runs, Ruff, both new independent expert-oracle
cases, and the 11 existing graph reference cases passed. Actual worker tests
also passed ASan/UBSan/leak checks and Clang 18 ThreadSanitizer with the existing
per-process `setarch -R` policy, no sanitizer suppressions or retries. Full
canonical build/static/sanitizer/hosted checks are recorded by the root task,
not inferred from these targeted results.

Targeted logs live in the ignored evidence directory:

- `build/ria/evidence/prefill-graph-verification.log`
- `build/ria/evidence/prefill-graph-sanitize.log`
- `build/ria/evidence/prefill-worker-verification.log`
- `build/ria/evidence/prefill-worker-sanitize.log`
- `build/ria/evidence/prefill-worker-thread.log`
- `build/ria/evidence/prefill-expert-oracle.log`
- `build/ria/evidence/prefill-reference-verification.log`

Physical grouped CUDA numerical/fidelity and performance gates remain unrun as
requested. No speedup, target GPU parity, complete matrix, or final release
qualification is claimed by this review.
