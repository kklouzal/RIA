# Grouped prompt graph implementation checkpoint

The graph retains the pinned sequential `seqlen=1` numerical realization while
transposing execution into bounded layer-major microbatches. For each layer,
rows visit attention in increasing absolute causal position, then routed FFN
work is grouped by original expert ID. The original selected slot and absolute
position travel through local/remote execution and determine scatter locations.
Shared expert execution remains exactly once per row. Source-ID merge order and
every single-row attention/mHC/Engram kernel/cast boundary are unchanged.

`ria_graph_options.prefill_rows` is an authenticated positive cap in `[1,64]`,
no larger than the context. `ria_graph_prefill_rows()` accepts one mixed
text/image microbatch; nonnull row embedding pointers include learned image
delimiters and suppress Engram just as in decode. `ria_graph_prefill()` splits a
text prompt into those admitted chunks. A failed chunk retains its old public
position and poisons its private state; output buffers are unspecified. A reset
drains CUDA before clearing all private state/snapshots. With no teacher-forced
`all_logits`, only the final row's vocabulary projection runs in each chunk.
Output spans must be disjoint except that `last_logits` may exactly equal the
final row of `all_logits`. Full text-call and bounded mixed-chunk spans are
checked before any state mutation; partial/earlier-row overlap and address
overflow reject with `RIA_INVALID_REQUEST`. The exact alias skips the final
copy, avoiding overlapping `memcpy` while preserving all teacher-forced rows.
Optional cancellation is polled before row initialization, each causal layer
row, each populated expert group, row scatter, vocabulary head and final chunk
commit. It returns `RIA_CANCELLED`, restores the pre-chunk public position and
poisons/drains private state. Callback/context lifetime is the synchronous
operation; there is no hidden global or retained callback.
The cancellation observer is nonblocking and may not re-enter or destroy the
graph; its context remains valid throughout the call.

Each protected row owns H, FFN residual/input/shared/PRE/post/comb, routed
contributions/IDs/coefficients, exact compressed-token history, the latest 512
selected positions, and the layer-20 candidate hierarchy's original 2,048 block
IDs. Layer-major execution may have already published later compressed rows;
each consumer uses `(absolute_position+1)/ratio` to limit its visible population.
Per-row selected positions are restored before every layer. Compact candidate
IDs are restored before later reindex consumers, rather than accidentally using
the last row's hierarchy. Window slots retain the source's invalid leading
holes and its 64-slot BF16 online-softmax boundaries.

`ria/prefill.h` is the authoritative host snapshot layout and
`ria_graph_prefill_required_bytes()` in `ria/prefill.c` is the shared native
admission/allocator equation. Extra host bytes are page-rounded
`rows * (sizeof(ria_graph_prefill_row) + 2*5120*sizeof(float) + sizeof(uint64_t) +
sizeof(float) + 2*sizeof(uint16_t))`, including contiguous expert group inputs,
results, positions, coefficients, original slots, and row scatter indices.
Those bytes occupy a separate page-rounded tail within the existing locked,
non-dumpable graph private mapping. Existing state/owner/cache bytes are added
to that allocation, with overflow checks.

The projection arena's exact additional device bytes over its previous
one-row bound are `(rows-1)*1117188 + 2048*sizeof(uint32_t)`. The per-row term
comes from the owned input/quantized/factor/activation-code/activation-scale,
gate/up/output/coefficient arrays for the declared 32768/32768/129280 arena.
Weight tiles and owner metadata do not multiply by prompt rows. Pinned bytes
come from the existing page-rounded projection pool equation with `max_rows`
equal to the admitted prefill cap; the vision pool remains separately reserved.
CUDA's allocator still checks the complete actual cap before every allocation.

`ria_expert_cuda_evaluate_resident()` shares the existing host input/result API
and production per-row quantizers. Explicit local VRAM entries use their owned
resident view; explicit host entries stage each bounded channel tile once for
the complete group. It never promotes/fetches a missing client expert and never
executes learned expert arithmetic on the client CPU. Decode's original
single-row local expert entry point remains unchanged.

Host expert input/coefficient/output spans use one pure address predicate,
`ria_expert_ranges_disjoint()`, in both CPU and CUDA boundaries. CUDA checks
exact strided spans `(rows-1)*stride+width`, including internal row padding,
before the first device operation. It rejects overlap or unrepresentable
address arithmetic with `RIA_INVALID_REQUEST`; permitted adjacent spans do not
include unused padding after the final row. The driver-free
`test_prefill_cuda_host_boundary.py` executes the actual CUDA host validation
prefix/private context layout with a sentinel at its first CUDA boundary.
Seventeen alias/stride/coefficient/resident-owner cases and real CPU/range
comparisons passed; this never invokes a CUDA library, device, or kernel.

Validation at this checkpoint: strict C99 host syntax (including pedantic,
conversion and shadow diagnostics) passed; both owned graph/expert CUDA sources
compiled with the production strict SM120a flags. A separate compilation and
execution of `tests/ria/test_graph_prefill.c` passed and is recorded at
`build/ria/evidence/prefill-graph-independent.log`. The independent fixture
compares actual graph orchestration with synthetic scalar arithmetic for chunks
1/2/3/8/17/64, 130-token sparse rings, retained source/compressor/candidate state,
all 384 original experts across rows, mixed local/remote ownership, image spans,
continued prompts, subsequent decode, boundary validation and failed-chunk
poison/reset. This is software scheduling evidence, not numerical GPU or
performance qualification. No model weights, GPU initialization, GPU kernels,
or live model inference were used. Integrated static/sanitizer/build gates are
owned by the verifier/root and will supersede this checkpoint with final results.
