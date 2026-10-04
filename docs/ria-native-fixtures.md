# Native bounded fixture bootstrap

The standalone qualifiers use deterministic synthetic inputs and the production
operator APIs. They load no model weights, TensorStore, tokenizer or bank. This
breaks the admission cycle: initial operator qualification can precede a full
bank, while full-model graph parity, both fidelity axes, the release matrix and
the one-hour soak remain separate gates.

Build with `make ria-cpu` and `make ria-cuda`. The driver-free executable is
`bin/ds4-ria-qualify`; the native SM120a executable is
`bin/cuda/ds4-ria-qualify`. Builds do not run CUDA. CUDA C compilation uses
`-DRIA_WITH_CUDA -isystem /usr/local/cuda/include`; CUDA kernels compile AOT
for `compute_120a` to `sm_120a`, without fast math or implicit FMA contraction.

Both executables accept one strict JSON request path and write one raw JSON
measurement record to stdout. Diagnostics and failures use stderr and nonzero
status; failures publish no success-shaped record. An outer supervisor must
enforce the complete process deadline and host cap, because a blocked CUDA call
cannot be canceled safely by a host control-point check. Use the repository's
supervised qualification workflow to seal measurements and judge the accepted
policy. A direct binary invocation is measurement only:

```sh
bin/ds4-ria-qualify /absolute/path/server-request.json > /absolute/path/server-raw.json
bin/cuda/ds4-ria-qualify /absolute/path/client-request.json > /absolute/path/client-raw.json
```

Offline diagnostics do not query or execute CUDA:

```sh
bin/ds4-ria-qualify --validate-request /absolute/path/server-request.json
bin/ds4-ria-qualify --oracle-self-test
python3 tests/ria/test_qualify.py
```

The authoritative request/raw schemas and exact case populations are in
`tools/ria/native_fixture_schema.py`. All request fields are required. Supply
the actual preregistered logical-model, source-lock, joint environment, build,
policy, operator-contract and preregistration SHA256 digests. The binary binds
and echoes these identities; the trusted workflow authenticates them. It does
not accept a caller's digest as proof of preregistration or of physical policy.

`executor` identifies the **server policy** (`cpu` or `cuda`).
`runner_executor` identifies the arithmetic actually executed by this process.
A server request sets them equal. A client request always sets
`runner_executor="cuda"`, including when its server policy is CPU. `role` is
`server` or `client`. This permits client CUDA graph/transfer measurements and
server CPU expert measurements to form one realization under a frozen joint
environment. Client startup requires exactly one visible device zero, the
locked canonical UUID, physical CC12.0 and `NVIDIA GeForce RTX 5090`. Server
CUDA startup requires the locked UUID, one device zero and physical CC12.0;
the provisioned inventory/qualification policy owns its product selection.
CPU requests require `gpu_uuid=null`, `device_budget=0` and `pinned_budget=0`.

`host_budget`, `device_budget`, `pinned_budget` and `fixture_seed` accept unsigned
decimal strings or exact JSON integers. Host budget is at most the JSON safe
integer limit; decimal uint64 values are range checked natively. Positive
`deadline_ms` is at most 600000, `repeats` is 1–32 and `warmup` is 0–8.
`relative_floor` must be positive and must equal the floor in the accepted
policy. There are no default acceptance tolerances. `expert_shape="ragged"`
uses 17 rows, K65, intermediate33 and output19; `"target"` uses one row,
K5120, intermediate2304 and output5120. The latter is synthetic original-domain
operator data, not a source checkpoint replay.

Each executor exercises dense palette weights, asymmetric gate/up clamps,
zero and nonzero routing coefficients applied before the down quantizer, and
a unit-diagonal closed form. CPU production evaluates all three projections
against a separately written scalar FMA oracle. That oracle uses explicit
nearest-palette searches, independently decoded E4M3/E2M1 tables, adjacent
BF16 rounding and per-group scale boundaries; it does not include `numeric.h`
or call the production quantizer. Existing Fraction fixtures independently
check the production palette, midpoint and fragment contracts. CUDA additionally exercises
both bounded host staging and immutable VRAM residency with identical inputs,
recording actual placement parity. NVFP4 fixtures use explicit frozen synthetic
global factors; they do not stand in for full-population source calibration.

The client CUDA population invokes the actual graph kernels for E4M3/UE8M0
act32 window state, E2M1/E4M3 group16 compressed state, E2M1/UE8M0 group32
index state, flattened20480 mHC with20 Sinkhorn iterations and post-mix
orientation, FP32 ratio2 compressor pooling, forward/inverse window and YaRN
RoPE, 65/129-key sparse attention and BF16 probabilities, and32×128 index
scoring. Attention materializes one extra future key and changes it to verify
that the authoritative causal extent excludes it. Hierarchical index fixtures
use16409 entries, an incomplete newest eight-position block pinned at infinity,
2048 candidate blocks and512 selected positions. A separate closed-form
index score and host stable rank oracle verify Full and Reindex behavior;
the source graph's Reuse aliases and exact token schedule are covered by the
host graph contracts. These fixtures do not assert full-model state parity.

Additional bounded client cases decode source Engram rows with32-element
UE8M0 scale groups, check per-stream learned fusion and image-mask pass-through,
separate routing selection bias from unbiased coefficients, verify original-ID
expert summation with the shared contribution once, and retain unrounded FP32
inputs in sensitive dense projections. These source-domain operator fixtures
do not require learned checkpoint weights.

Transfers use an owned reusable pinned pool and an odd byte count larger than
the pool, exercising multiple complete chunks and a partial tail. No full-bank
host registration occurs. Requests enforce allocation caps before owned
allocations; raw evidence reports heap metadata and pinned ownership, actual
CUDA allocation peaks, startup costs, elapsed time and process peak RSS.
Runtime/allocator/stack overhead is observable in process RSS and governed by
the outer process cap; owned-byte counters alone do not claim to measure it.
Latency samples include synchronous input transfer, operator work and output
transfer. Expert samples include all three projections. Graph fixtures have
explicit bounded samples rather than an inferred model throughput result.
The schema module declares exact dimensions, sample counts, warmups and timing
groups. mHC and router observation records share a completed invocation's
timing group; their latency must not be counted as another independent call.

Every raw result has `qualified=false`. Numeric errors are measurements; byte
state, selected IDs and transfers separately require exact matching through
`requires_exact_match`. Repeat and placement parity are null unless actually
checked. The supervisor binds the raw record to the accepted policy, evaluates
the exact required population and publishes sealed component evidence. Initial
admission still needs all five components: experts, graph/state, transfers,
transport and resource bounds. The latter two are supplied by their independent
native supervisor/transport fixtures, not fabricated by this operator harness.

No CUDA fixture, physical benchmark or model replay was executed during this
implementation. Compilation and native SASS inspection demonstrate code paths;
they do not qualify hardware arithmetic, source quality or performance.
