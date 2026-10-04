# Independent model implementation review

This review began at `bd76be469c45c6e672481125023090577b08cdf3` and assesses the current working tree. The immutable specification is `docs/ria-specification.md`, SHA256 `14224cdb33476944111e14f69a5679f0597c192a44d67b48f048910f326f3f6e`. The user's selection of `antirez/ds4` and the `kklouzal/RIA` maintained fork supersedes the original handoff's repository preference. Host implementation remains native C; CUDA translation units use CUDA C++.

No model weights were downloaded, physical GPU queried, CUDA kernel executed, or live inference run. Compilation, synthetic host execution, source inspection and mocked lifetime tests are evidence only for their stated contracts.

## Scope and independent sources

The original planning review consumed specification lines 1–610 and 1928–2133 completely. This implementation audit additionally checked the attention/state/continuation, memory/lifetime, numerical-validation and qualification requirements, especially lines 200, 257, 302, 476–480, 575–593, 769, 951, 1016–1018, 1165–1177 and 1209–1233.

Reviewed complete graph/CUDA graph, vision/CUDA vision, state, CPU expert and CUDA expert implementations; engine open/placement/prompt synchronization/continuation/image/teardown/rebind callers; numeric, target and Engram preparation helpers; and the critical preparation conversion/calibration/resume/verification paths. Independently compared the pinned DeepSeek revision `2cba9e42aa026125f3ed06c6d98c1db82f7ca027` inference `model.py`, `kernel.py`, `vision.py`, `engram.py` and `image_processor.py`. NVIDIA metadata and its cited SGLang `da64c5cbb8cf6bfd39be19da43573fdfd484c43a` implementation inform the strict NVFP4 scale contract rather than treating the original MXFP4 source as a W4A4 oracle.

The final additional read-only audit covered `ria/inventory.c`, TensorStore metadata inspection/accounting, expert-policy extraction, shared NUMA accounting, Python inventory orchestration and finalize's native rederivation.

## Confirmed defects corrected

1. **mHC premix persisted between tokens.** Source `model.py:1257–1260` initializes the identity premix on every forward, with helper `1159–1163` producing `[1,0,0,0]`. The old native graph reused the last FFN premix at the next token's first collapse. `ria/graph.c:549–555` now begins each token with the narrow CUDA premix reset; retained attention state, candidate masks and Engram history are preserved. The actual graph recorder uses 130 tokens and deliberately leaves a nonidentity final premix. Removing just the new reset from a temporary source copy reproduces a second-token failure.

2. **Sparse-slot compaction changed BF16 rounding boundaries.** Source `model.py:410–429` uses one initial slot, then 128 physical window slots in oldest-first ring order, including leading invalid slots while filling. Source `kernel.py:351–378` casts unnormalized exponent weights to BF16 within each 64-slot tile. Compacting valid rows moved compressed rows across these cast boundaries. `ria/graph.c:500–513` now preserves extent, physical order and mask; `ria/graph_cuda.cu:192–216` skips masked KV loads in both score and value accumulation. Callers in `ria/qualify.c` supply explicit empty masks. A valid packed synthetic example produces source `-3.453125` versus compacted `-3.796875`; it demonstrates an operator defect, not a full-model quality estimate. The graph recorder is compared with the pinned source helper across 130 positions, including partial/full/wrapped rings. Host execution of the exact kernel body checks poisoned holes, an entirely masked tile after valid rows and a wholly masked row.

3. **Failed CUDA quiescence released borrowed backing.** CUDA destructors formerly continued releases after failed synchronization; graph/vision/engine callers then released model or pinned source memory. Destructors now stop before releasing anything after device selection or drain failure, preserve owners on CUDA release failure, and clear successfully released fields. Graph and vision parents retain their owners when child cleanup fails. Engine close retains the graph/TensorStore on failed graph cleanup; reconnect and construction policy owners terminate before losing unsafe ownership. The server and CLI top-level policy was integrated by their respective owners. Model-free fixtures exercise the exact four destructor bodies and actual graph/engine close/reconnect callers, with shared release counters and a child fail-stop assertion.

   Current primary CUDA documentation supports requiring successful completion before reclamation and process termination after sticky fatal errors: [stream API](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__STREAM.html), [runtime error types](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__TYPES.html). The conservative contract does not assert that every synchronization error proves outstanding DMA; it refuses to infer safe completion from a failed wait.

A last-read concern that host `munlock`/`munmap` failures contradicted a general owner-retention promise was withdrawn: the current `graph.h` promise specifically applies to CUDA drain/release failures, which return before host teardown. Vision's host `free` is not fallible. Policy owners fail-stop on any false teardown result. This is not an additional unresolved finding.

## Mandatory software still absent

Specification line 476 requires bounded prompt chunks, grouping independent rows by expert and scattering results to their original row/slot. `ria/graph.c:593–599` still implements prefill by repeated complete one-token steps; engine prompt synchronization likewise invokes the single-row graph/callback. This is a confirmed software gap, separate from unrun hardware validation. The retained all-layer schedule is an allowed correctness baseline under lines 302 and 589, but it does not satisfy the expert-grouped execution requirement.

A correct implementation needs bounded per-row hidden/residual/mHC/Engram/image state; causal per-query window/compressor/index/candidate state; multirow client executor and remote callbacks; original `(row, slot, coefficient)` scatter identities; canonical per-row reduction; and admitted batch workspaces/transfer resources. Compressor partial groups and window/source aliases must survive chunk boundaries. Candidate masks and selected positions currently belong to a single mutable query and cannot simply be reused for simultaneous rows. Chunk schedule identity also affects source sparse-slot/BF16 cast layout, so sequential-versus-grouped validation must explicitly define that numerical comparison rather than silently inheriting a one-token layout.

The broader complete semantic matrix/fault/gate producer gap is recorded by the independent qualification reviewer and `planning/qualification-software-readiness.json`; this model review does not count those absent producers as hardware-only blockers.

## Inspected contracts and limits

No further unconditional graph-math defect was identified in the inspected baseline after the fixes. Checks covered source KV/index aliases and compression ratios; candidate newest-partial-block preservation; RoPE and compressed positions; sink denominator and BF16 attention probability casts; flattened mHC normalization, Sinkhorn order and transposed residual mixing; modality router behavior; source Engram hash/dead boundaries, packed lossless rows and learned CUDA fusion; expert clamp, coefficient-before-down-quantizer and increasing-expert-ID reduction; and the full bidirectional native vision path, 2D rotation, unfold order and image processing.

Preparation distinguishes the original MXFP4 source from the strict NVFP4 W4A4 operator, retains published global factors and local E4M3 scales, and records FP8 32×32 weight blocks/group-32 UE8M0 activation scales. Widening source weights does not recover master weights. Sensitive FP32 modules and source BF16 casts remain operator-specific. Actual full-checkpoint mapping, CUDA MMA/kernel arithmetic, profile quality and teacher-forced loss, long-prompt retained-state parity, native-image numerical parity and cross-target comparisons remain unproved by this audit. Synthetic/static checks cannot substitute for them.

Inventory inspection stops before payload population and reuses authenticated descriptor, page-rounding, alias and NUMA replica accounting. All phase allocations are conservative reservations. Client pinned capacity is counted once within host capacity. Expert local/startup headroom and client runtime capacities are explicitly selected budgets, not measured library peaks or derived physical availability. Finalize rejects an inventory differing from a fresh native derivation. Fixed-bound output, constant/generated names and protected atomic publication introduce no confirmed new boundary or ownership defect. Future capacity, physical NUMA placement, GPU workspace peaks and performance still require qualification.

## Exact available verification

- `python -m unittest discover -s tests/ria -p test_graph_reference.py -v`: 11 passed, including actual graph schedule recording and pinned-source ring helper comparison.
- `python -m unittest discover -s tests/ria -p test_cuda_lifetime.py -v`: 2 passed, exact destructor bodies and actual graph/engine backing preservation/fail-stop.
- `python -m unittest discover -s tests/ria -p test_attention_slots.py -v`: 1 passed, exact kernel body hosted on 256 pthread lanes; no CUDA runtime.
- `python -m pytest -q tests/ria/test_inventory.py`: 16 passed, authenticated metadata-only native CLI, absent payloads, all three NUMA policies, pinned counting, tamper rejection and failure publication preservation.
- Strict C99 `RIA_WITH_CUDA` syntax for graph/engine/vision/qualifier passed. Expert, graph and vision CUDA units compiled with SM120a, strict host diagnostics, precise divide/sqrt, `--fmad=false`, no fast math and explicit architecture target. Graph CUDA compilation was repeated after the mask API change.
- Actual graph-step and graph/engine cleanup fixtures passed ASan/UBSan with leak detection. The step fixture was repeated after extending to 130 tokens. `git diff --check` passed.

Python commands above used `/tmp/ria-validation-venv/bin/python`. CUDA compilation used `/usr/local/cuda/bin/nvcc`; no GPU execution follows from successful compilation. Root owns the final stable-source aggregate, generated-schema checks and hosted integration evidence.

The three focused model suites total **14 tests** (11 graph + 2 lifetime + 1 kernel), in addition to 16 inventory tests. The CUDA compilation command for each changed unit was:

```sh
/usr/local/cuda/bin/nvcc -I. -Iria -Ithird_party/ryu -I/usr/local/cuda/include/cccl -O2 -g -lineinfo -std=c++17 --compiler-bindir=g++ --generate-code=arch=compute_120a,code=sm_120a --fmad=false --ftz=false --prec-div=true --prec-sqrt=true --Werror=cross-execution-space-call,reorder,deprecated-declarations,default-stream-launch,ext-lambda-captures-this -Xcompiler=-Wall,-Wextra,-Werror,-fno-fast-math,-ffp-contract=off,-pthread -DRIA_WITH_CUDA -c ria/graph_cuda.cu -o /tmp/ria-audit-graph_cuda.o
```

Expert and vision used the corresponding `ria/expert_cuda.cu` and `ria/vision_cuda.cu` source/output names with the same flags.
