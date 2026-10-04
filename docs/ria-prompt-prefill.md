# Expert-grouped prompt prefill

Specification section7.3 is implemented in the native RIA serving path.
`ria_graph_prefill_rows` processes one admitted prompt chunk; the text wrapper
and engine split longer prompts into those chunks. The maximum is64 rows.
Ordinary generation keeps the one-position decode graph.

## Causal schedule and numerical contract

Each chunk runs layer by layer. Attention visits its rows in ascending absolute
position, preserving source KV aliases, sparse window holes, compressed-block
publication, incomplete pools, mHC initialization and Engram history. Each row
retains its own residual, FFN input, mixes, router choices and query-specific
index candidate snapshot. A later row's candidate set cannot replace an earlier
query's snapshot.

The independent FFN rows are then grouped by original expert ID. Explicit phase
membership selects local host-backed CUDA, local VRAM CUDA or remote execution.
One group carries absolute row IDs, coefficients and original selected slots;
results scatter into the original row and slot. The shared branch executes once
per row and the final merge follows the source slot order, including zero
coefficients. A client miss never fetches weights or runs CPU expert arithmetic.
Remote CPU workers also gather same-owner, same-expert jobs into bounded groups.

The numerical realization retains the pinned source's single-position operator
population, quantizer scope, BF16 cast points and sparse attention tile layout.
Grouping changes scheduling and ownership, not precision. Optional `all_logits`
records every teacher-forced row; without it, only the chunk's final vocabulary
head runs. `last_logits` may equal the final `all_logits` row exactly; other
overlapping output spans are rejected before state mutation.

Mixed chunks accept text, encoded image and learned delimiter rows. Image rows
suppress Engram and use the image router bias without resetting causal history.
Healthy continuation preserves packed state, candidate aliases and incomplete
compression groups. An unchanged prefix performs no additional prefill work;
changed token/image identity requires fresh binding and complete replay.

## Capacity and transport

`prefill_rows` is mandatory in the planning request, memory plan, placement
runtime and service configuration. Its range is1–64 and it cannot exceed context
capacity. Preparation, generated native inventory, admission and runtime compare
the same value. Changing it requires regeneration of identity-bound plans,
bootstrap, probes, calibration evidence and deployment locks; packages lacking
the field fail closed.

The authoritative layout and pure allocation equation live in `ria/prefill.h`
and `ria/prefill.c`. Protected host storage holds the row snapshots and two
row-major input/result staging arrays, plus row/slot association metadata.
Device and pinned admission include the multirow projection context and the
source-candidate ID buffer. Existing state, vision, TensorStore, local expert
and Engram cache pools remain separately accounted. Native admission rederives
the allocation proof rather than trusting an editable allocation list.

The remote owner reserves its full admitted result staging pool and bounds
transient control/data allocations, including JSON parsing/canonicalization.
Negotiated row, frame and shared-credit limits determine each wire subgroup
before dispatch. Splitting preserves associations, unique invocation IDs and
the protected progress unit. Results publish only after every subgroup succeeds.
The expert worker's fixed NUMA arena determines its group capacity; the same
pure equation also sizes CUDA pinned admission. Device allocation must succeed
before the server becomes ready.

## Failure and cancellation

All prompt tokens and image embeddings are validated before a chunk mutates
state. Cancellation is checked between initialization, layer rows, expert
groups, scatter rows, vocabulary heads and final commit. The callback must be
nonblocking, must not re-enter or destroy the graph, and owns a context that
outlives the synchronous call.

A chunk that fails during execution or is cancelled retains its prior public
position, poisons private state and retires the binding. Boundary rejection
before mutation leaves the graph unchanged. Partial execution output is unusable. Engine prefix and
logits publication occurs only after the entire chunk succeeds; recovery uses
a fresh binding and complete replay. Worker request backing remains owned until
all admitted work drains. Failed CUDA quiescence retains borrowed backing and
requires the serving owner to terminate before teardown.

## Offline evidence and limits

The fixtures execute actual graph scheduling with independent synthetic
arithmetic, the actual engine synchronization, production worker threads and
real loopback mTLS request framing. They cover chunk sizes1/2/3/8/17/64,
130 causal positions across all40 layers, all384 expert IDs, all three profiles,
original slots and coefficients, explicit residency/phase membership, mixed
image rows, continuation, output alias boundaries and staged cancellation.
Independent expert oracles check64-row CPU groups and host-simulated production
CUDA quantizer bodies. The CUDA host boundary fixture stops before the first
CUDA operation and checks the actual production validation prefix.

The [offline verification report](ria-offline-verification.md) records final
checks and publication evidence. No checkpoint download, physical GPU operation,
live model inference, target capacity/NUMA probe, performance comparison or soak
is part of this evidence. CUDA compilation is separate from numerical execution.
No measured performance superiority, cross-platform bit identity or CPU weight
tile reuse benefit is claimed. The complete semantic540-cell matrix/fault/gate
tooling remains a separate software gap, and these images remain unqualified
for model/physical release.
