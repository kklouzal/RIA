# Independent native graph and operator once-over

Baseline: `fc7c5d86e59b26376fd0191ede120d34526def34`. The immutable
specification is `docs/ria-specification.md`, SHA256
`14224cdb33476944111e14f69a5679f0597c192a44d67b48f048910f326f3f6e`.
This audit read specification sections 3–13 and the operator/validation
requirements, the complete graph, CUDA graph, CPU/CUDA expert, native image
preprocessor, CUDA vision and private-state implementations, their public
contracts and current fixtures. It independently compared the pinned
DeepSeek `2cba9e42aa026125f3ed06c6d98c1db82f7ca027` model, kernel, Engram,
vision and image source. Historical review and prefill notes remain unchanged.

## Confirmed finding and correction

The CUDA host expert boundary validated `rows*stride*sizeof(float)` rather
than the used extent `((rows-1)*stride+width)*sizeof(float)`. This rejected a
valid one-row call whose unused stride was `UINT64_MAX`; the actual CPU
executor accepted the same three-float input/output buffers. The independent
reproducer executed the production CUDA validation prefix and stopped before
its first CUDA operation. Ordinary engine/service calls use stride 5120 and
were unaffected; this was a low-severity public CPU/CUDA contract mismatch.

The corrected boundaries share `ria_expert_float_span_bytes()` in
`ria/expert.c:39`, including CPU evaluation and diagnostic projection. The
helper checks exact element multiplication/addition and representable byte
conversion. `ria/expert_cuda.cu:456` uses it before any device operation and
normalizes an unused one-row stride to logical width before byte-pitch
conversion. Inter-row padding remains protected by the alias check; final-row
padding remains outside the span. Empty, short, overflowing and overlapping
spans still fail. Headers document this contract.

The expanded `test_prefill_cuda_host_boundary.py` executes 20 production-prefix
cases, actual CPU comparisons and nine pure span-boundary assertions. It covers
huge unused strides, exact last-row extents, representable byte-limit equality,
one-float overflow, multiplication overflow, zero rows/width and missing output.

## Independently traced contracts

- Every layer retains its own 128-slot window. Four compressed-KV/index owners
  use the declared source aliases and exact floor-based capacities. Ratio-two
  FP32 incomplete pools survive chunk boundaries, and each causal row observes
  only `(absolute_position+1)/ratio` published rows despite layer-major execution.
- Grouped prefill snapshots per-row selected positions, history and the layer-20
  hierarchy; later reindex sources restore that row's candidate mask. The newest
  partial block remains mandatory. Reuse layers preserve the preceding index
  selection; subsequent decode replaces query-dependent selection correctly.
- Per-token one-hot mHC initialization, flattened-stream normalization,
  Sinkhorn order, transposed residual mixing and sublayer premix ownership match
  the pinned source. Engram uses causal dead-token boundaries, exact packed
  rows, original global IDs and learned GPU fusion. Images and delimiters
  suppress Engram and select the image router bias.
- Routed expert IDs/coefficients retain original slot association. Each row's
  shared branch runs once, with source precision, before increasing-expert-ID
  FP32 routed/shared merge and BF16 output rounding. Gate/up clamp asymmetry,
  coefficient position and down-input quantization boundaries are preserved.
- CPU and CUDA quantizers retain K32 FP8 and K16 NVFP4 row domains and frozen
  calibrated global factors. Output-channel tiles preserve FP8 N/32 scale
  coordinates, including non-block-aligned tile bases. Native fragment and
  NVFP4 scale selectors were rechecked against current
  [NVIDIA PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16864).
- Vision retains the source full bidirectional attention domain, split-half 2D
  RoPE, BF16 operation boundaries, channel-major padded 3×3 unfold and aligner.
  PNG/JPEG preprocessing bounds raster and ancillary/coefficient work, and
  declares its source RGB/bicubic behavior. Host/private, device and pinned
  owners remain separate; failed CUDA quiescence retains borrowed backing.

No additional unconditional graph numerical, causal-state, allocation or
lifetime defect was confirmed in these paths. An independent read of the
current cross-agent credential-file and TensorStore JSON-budget changes found
no new C/API defect; the integration owner must validate the frozen whole tree.

## Targeted checks and practical limits

All checks used the ARM64 development host, no shared make targets:

- `/tmp/ria-validation-venv/bin/python -m pytest -q` over
  `test_prefill_cuda_host_boundary.py`, `test_prefill_expert_oracle.py`,
  `test_graph_reference.py`, `test_cuda_lifetime.py` and
  `test_attention_slots.py`: **17 passed** in 6.58 seconds. Full log:
  `build/ria/evidence/once-over-graph-targeted.log`. One expected Pillow palette
  transparency warning came from the independent reference decoder.
- `tests/ria/test_expert.c` compiled and passed with strict C99/pedantic,
  conversion/shadow/format/prototype diagnostics, no fast math and contraction
  disabled. Its separate ASan/UBSan build passed with leak detection and
  halt-on-error enabled.
- `ria/expert.c` passed the same strict syntax diagnostics.
- `ria/expert_cuda.cu` compiled to `/tmp/ria-once-over-expert_cuda.o` with the
  production `compute_120a/sm_120a`, CUDA C++17, strict host diagnostics,
  `--fmad=false --ftz=false --prec-div=true --prec-sqrt=true` flags. No device
  discovery, CUDA initialization or kernel execution followed compilation.
- Scoped `git diff --check` passed for the five changed expert/boundary files.

Actual GPU MMA/SDPA arithmetic, target-size device accesses and resource peaks,
checkpoint/profile quality, complete learned image parity and long-context
model execution remain unrun. In particular, PTX documents MMA accumulation
order and subnormal handling as unspecified: strict compiler flags and a host
fragment simulation do not certify physical numerical behavior. No performance
improvement or full-model fidelity is claimed by these software checks. No
model weights were downloaded and no GPU query, initialization, kernel, live
model, physical probe or soak was run.

## Follow-up startup and saved-logit review

An independent read of `tools/ria/qualification.py` and its current tests found
no additional unconditional metric or identity defect. The comparison hashes
the same header/row bytes it consumes for metrics, validates complete file
dimensions, and checks descriptor/path identities before and after the read.
Non-object policies and teacher inputs now fail at their declared artifact
boundary. The revised shifted loss remains finite across the tested common
FP32 offsets. All **40 qualification tests** passed.

The explicit `ds4.h`/`ria/engine.h` startup callback is borrowed only during
construction, checked before allocation and at staged copy/hash/configuration,
peer-binding and final graph publication boundaries, then cleared. Callback
cancellation preserves `RIA_CANCELLED`; cleanup retains backing and terminates
when safe CUDA completion cannot be proved. The production TensorStore copy
and payload-hash blocks are bounded by `RIA_BULK_MAX` (4MiB). No physical startup
or cancellation-latency guarantee was exercised.

Two additional findings were confirmed:

- `ria/tokenizer.c` used `fopen()` before file admission. Actual
  `ria_tokenizer_open()` blocked for more than 500ms on a writerless private
  FIFO; the bounded child was killed and reaped. It now uses nonblocking,
  close-on-exec/no-follow open, requires a nonempty regular file of at most
  16MiB before allocation, reads its exact admitted extent, and rejects growth,
  truncation or changed descriptor identity. The consumed source SHA and pinned
  tokenizer semantics are unchanged. A valid output pointer is cleared on
  every failure. The existing BPE local `edge` variable was renamed to pass the
  owned source's strict shadow diagnostics.
- Client health could still report ready after a stop signal, because it
  ignored the process stop bit until later drainage. Shared `volatile
  sig_atomic_t` stop/mode/listener words also lacked a C thread synchronization
  contract. This was handed to the protocol owner for a lock-free signal-safe
  atomic correction, readiness gating and early immutable signal-mode
  selection. Its final source review remains an integration checkpoint.

The new `test_tokenizer_boundary.c/.py` uses actual tokenizer/JSON/hash code and
private file fixtures, including FIFO, directory/device, empty/oversized source,
final symlink, wrong SHA, valid pinned metadata, and a restored-byte/mtime rewrite.
The injected read wrappers retain both actual fortified and unfortified reads.
`RIA_TOKENIZER_BOUNDARY_FIXTURE` selects a canonical-built executable; standalone
use compiles an isolated scratch fixture.

Current follow-up checks:

- Standalone tokenizer plus qualification pytest files: **48 passed** in 2.11s;
  full log `build/ria/evidence/once-over-tokenizer-qualification-targeted.log`.
- Explicit scratch ASan/UBSan executable through all eight tokenizer tests:
  **8 passed**, leak detection and halt-on-error enabled; full log
  `build/ria/evidence/once-over-tokenizer-sanitize.log`.
- Owned tokenizer and new fixture compiled with strict C99/pedantic,
  conversion/shadow/format/prototype diagnostics, and no fast math/contraction.
  The pinned Ryu dependency used the canonical project's diagnostics rather
  than imposing new conversion diagnostics on vendor code.
- Ruff and scoped whitespace checks passed. No shared make target was invoked.

The startup constructor fixture uses explicit foreign-boundary shims, so it is
control-flow/ownership evidence only. Root was advised that required operations
inside its new `assert()` calls should use always-evaluating checks to satisfy
the repository's diagnostic-side-effect rule. Full integration, physical CUDA
initialization, startup/drain timing and model tests remain separate checks.
