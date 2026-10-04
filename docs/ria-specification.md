# An amalgamation of FreeToken and BlackStar with disaggregated attention

## DwarfStar distributed inference — implementation handoff

**Requirements date:** October 3, 2026  
**Document status:** Consolidated implementation specification and acceptance contract. Architecture and scope are selected; implementation, numerical fidelity, physical memory fit, and performance remain to be demonstrated.  
**Acceptance model:** DeepSeek-V4.1-Flash.  
**Deployment:** Docker on two physical Linux hosts: one Blackwell attention-client container and one RAM-backed expert-server container, with one active generation session. The same CUDA runtime image can serve either GPU role; a separate CPU-only runtime image must work without NVIDIA software.  
**Required server executors:** CPU-only and Blackwell CUDA, selected explicitly before admission.  
**Required client memory:** Separately managed VRAM and system RAM; client neural computation is CUDA-only.  
**Audit scope:** The complete supplied specification was read and reconciled with primary model, runtime, kernel, Linux, Docker, and NVIDIA Container Toolkit sources. Source-level evidence, proposed interfaces, document-level tests, and hardware acceptance are distinct. This pass also verifies accelerated CUDA target requirements, numerical scale domains, protocol lifecycle, canonical identities, two-stage deployment preflight, and container resource/security contracts. No repository compilation, container build/start, full checkpoint download, or inference benchmark is claimed.  
**Implementation status:** This is a normative development specification. Proposed commands, package targets, schemas, and container entry points below are interfaces the fork must implement, not features available by launching an upstream image.

> **Repository identity:** “BlackStar” is retained from the requested headline. No distinct, relevant BlackStar repository was established during this audit. The explicitly selected implementation base is **`stefandsl/DwarfStar`**. **`antirez/ds4`**, also presented as DwarfStar, is a related implementation donor. **`FlashML-org/FreeToken`** supplies offload, expert-cache, CPU-execution, and bandwidth-planning precedents. Do not substitute an unverified BlackStar project or assume these repositories already form one working runtime. [R01][R02][R03]

This document is the authoritative handoff. Its requirements supersede conflicting statements in earlier drafts. It contains design contracts, not working launch commands or a claim of completed development. **MUST** denotes an acceptance requirement; **SHOULD** denotes a preferred implementation subject to a documented, measured alternative; **MAY** denotes an optional feature. Proposed interfaces and configuration names are not claims about existing upstream APIs.

## Contents

1. [System contract and boundaries](#1-system-contract-and-boundaries)
2. [Evidence review and upstream reuse](#2-evidence-review-and-upstream-reuse)
3. [Target model and operator inventory](#3-target-model-and-operator-inventory)
4. [Numerical profiles and quantization](#4-numerical-profiles-and-quantization)
5. [Runtime organization and source locking](#5-runtime-organization-and-source-locking)
6. [Checkpoint preparation and tensor identity](#6-checkpoint-preparation-and-tensor-identity)
7. [Computation placement and layer execution](#7-computation-placement-and-layer-execution)
8. [Client VRAM and system-memory management](#8-client-vram-and-system-memory-management)
9. [V4.1 attention, state, and exact continuation](#9-v41-attention-state-and-exact-continuation)
10. [Engram lookup and local row caching](#10-engram-lookup-and-local-row-caching)
11. [CPU-only expert server and NUMA](#11-cpu-only-expert-server-and-numa)
12. [Blackwell CUDA expert server](#12-blackwell-cuda-expert-server)
13. [Cache planning and transfer scheduling](#13-cache-planning-and-transfer-scheduling)
14. [Internal interfaces and remote protocol](#14-internal-interfaces-and-remote-protocol)
15. [Admission, configuration, and operations](#15-admission-configuration-and-operations)
16. [Failure, cancellation, and security](#16-failure-cancellation-and-security)
17. [Performance model and measurement](#17-performance-model-and-measurement)
18. [Validation and acceptance matrix](#18-validation-and-acceptance-matrix)
19. [Implementation sequence and deliverables](#19-implementation-sequence-and-deliverables)
20. [Risk register and decisions to measure](#20-risk-register-and-decisions-to-measure)
21. [Traceability and definition of completion](#21-traceability-and-definition-of-completion)
22. [Docker deployment and operating contract](#22-docker-deployment-and-operating-contract)
23. [Source register](#23-source-register)

---

## 1. System contract and boundaries

### 1.1 Objective

Make the server's usable system memory the primary limit on **stored model capacity**, while using the client's Blackwell GPU and fast local system RAM to reduce inference latency. Avoid requiring a complete checkpoint on the client. Move compact intermediate inputs and computed expert contributions across the network instead of transferring expert matrices on ordinary client cache misses.

This is **attention/expert disaggregation with heterogeneous memory**, not a passive storage server, a shared virtual GPU, or an assertion that attention has no weights. The expert server performs input-dependent neural computation. Its executor is either CPU or CUDA; its authoritative weights remain RAM-backed in both modes.

The implementation must distinguish three results: **numerically qualified for the declared realization**, **fits the admitted memory hierarchy**, and **meets the declared performance target**. Numerical qualification includes stated tolerances; it is not a proof of bitwise equivalence across all kernels. A successful load does not establish useful speed. More total experts need not imply proportionally more active experts, but increasing width, depth, active count, or context can increase computation, state, and traffic.

### 1.2 Locked requirements

| ID | Requirement |
|---|---|
| MODEL | DeepSeek-V4.1-Flash is the development and release acceptance model. Other models and synthetic tensors may aid tests but cannot replace it. |
| GPU | Blackwell is the GPU family. RTX 5090 / SM120 client qualification is mandatory. Other Blackwell targets are qualified separately. |
| CLIENT | All client neural operations execute on CUDA, including attention, routing, local experts, mHC, and learned Engram fusion. |
| HOST | Client system RAM is a required managed backing/cache tier for weights and exact sequence state, not a CPU inference route. |
| SERVER | CPU-only and Blackwell CUDA expert-server modes are both required, using one mathematical service contract. |
| MODE | Server mode is explicit and fixed for a session. No automatic hybrid selection or silent device substitution. |
| NUMA | CPU serving supports sharded storage, replicated expert banks, and replicated server-required tensor sets with real per-node admission. |
| PROFILE | NVFP4, explicitly defined NVIDIA-compatible FP8, and BF16 are required mixed-precision profiles. |
| ROUTE | Preserve the model's selected experts, coefficients, masks, and operation semantics. Placement determines where work runs, not whether it runs. |
| MISS | A miss in both client expert-cache tiers causes remote evaluation, not a synchronous network fetch of that expert's matrices. |
| STATE | Sequence state belongs to the client across its VRAM and RAM. Paging must preserve exact dependencies and valid copies. |
| CAPACITY | Minimum implemented computation tiles must fit VRAM; backing objects, buffers, and replicas must fit their own memory budgets. |
| SAFETY | Invalid configurations fail admission. Incomplete computations never become plausible output through zeros, skipped experts, stale state, or hidden approximations. |
| EVIDENCE | Source pins, artifact identities, operator tests, placement tests, and physical-host measurements are required release evidence. |
| DOCKER | Docker is the production deployment boundary. Two host-local Compose deployments, CPU/CUDA image separation, resource/security settings, and containerized acceptance tests are required. |

### 1.3 Physical scope

The initial client is a Linux machine with one RTX 5090 and an intended class of 64 GB or more of system RAM. This is an intended hardware class, not a claim that 64 GB is sufficient for every profile and context. The server starts from an approximately 512 GB memory concept and may grow to multi-terabyte DDR3 systems. Its actual CPU ISA, DIMM arrangement, usable per-node capacity, network, and PCIe topology are not yet established.

One optional server GPU is in scope. Its advertised VRAM, possibly 16 GB, is cache/workspace capacity, not a promise that every kernel fits. Multiple NUMA nodes within the server are in scope; multiple attention clients, physical expert servers, or GPUs within one host are deferred. One active generation session does not prohibit bounded prompt-row batching within that session.

### 1.4 Ownership overview

```text
CLIENT: Blackwell GPU + fast system RAM       EXPERT SERVER: large system RAM

GPU: attention, indexing, routing, mHC        RAM: complete authoritative packed
     shared/local operations, cached experts      experts, Engram, manifest
RAM: admitted weight/state backing,
     additional expert and Engram caches      Executor selected at startup:
                                             CPU: NUMA-local packed expert math
Double client-cache miss:                    CUDA: local RAM -> VRAM -> expert math
    activation + exact selected slots ------>
    <---------------- ordered contributions

Engram row miss: row IDs -------------------->
    <---------------------- packed rows/scales

Bootstrap/cache preparation only:
    authorized immutable tensor ranges <-----
```

Normal learned client operations never run on its CPU. Host tokenization, parsing, address/ID bookkeeping, hashing for deterministic lookups, byte gathering, image decoding, memory copies, scheduling, and sampling/control are permitted. Image preprocessing must be declared and tested; learned vision computation remains CUDA. Sampling cannot be used as a label for moving the learned output projection to CPU.

### 1.5 Non-goals

The initial release does not implement a storage-only network weight-streaming alternative, NFS inference dependency, arbitrary model support, inter-host tensor parallelism, remote private-state backing, automatic CPU/GPU hybrid serving, speculative duplicate expert execution, approximate routing, expert pruning, semantic-output caching, automatic context truncation, or compulsory RDMA. It does not require a graph database or a general execution-graph service.

DSpark/MTP speculative decoding, adapters, steering, multiple-user batching, and dynamically rebalancing a cluster are deferred. Source tensors for deferred features must be preserved or excluded through an explicitly identified target-only runtime package. No feature may silently invoke a different full-model runtime. Ada/RTX 4090 support is not part of the Blackwell release baseline.

## 2. Evidence review and upstream reuse

### 2.1 Audit boundary

The audit read every section of the supplied Markdown and rechecked load-bearing claims against primary model/reference, runtime, numerical-kernel, Linux and container sources. The report's CED and bounded-replay pages were also inspected as rendered pages. Earlier video observations remain attributed to the earlier transcript/visual audit; this pass does not claim another video replay. The audit did **not** clone/build the complete dependencies, download the full checkpoint, recover the video's executable, execute Docker/Compose, or run model/GPU/NUMA benchmarks. Repository access through the working container and a Docker executable were unavailable. Reproducible document/protocol/configuration checks are reported separately from those outstanding implementation gates. No finite source audit proves all inputs or hardware configurations correct.

Public branches and documentation can represent different revisions. They are evidence and implementation leads, not a mutually tested dependency lock. Section 5 defines the required immutable source lock. No commit identifier is invented to conceal that work.

### 2.2 What each upstream contributes

| Upstream | Useful contribution | Integration boundary |
|---|---|---|
| `stefandsl/DwarfStar` | Selected native runtime base, quantized CUDA operations, model loading, cache and transport material | Audit V4 assumptions, full-model mapping dependencies, and fixed capacities. Its existing layer-distributed mode is not an expert RPC. [R01][R04][R05][R44] |
| `antirez/ds4` | Related DwarfStar evolution documenting V4.1 text CUDA support | Candidate graph/operator port. GGUF execution does not establish strict NVFP4/FP8/BF16 support or this topology. [R02] |
| `FlashML-org/FreeToken` | Host expert banks, GPU slot caches, packed evaluators, offload scheduling, CPU execution, bandwidth calibration | Reuse selected mechanisms and tests. Do not run a second complete FreeToken model behind the DwarfStar client. [R03][R06][R07] |
| FreeToken V4.1 pull request | Concrete experimental V4.1 graph, cache, quantization, and image work | Review pinned changes individually; neither an open PR nor author-reported tests establishes our acceptance matrix. [R08][R09] |
| DeepSeek reference | Architecture, graph semantics, packed source layouts, tokenizer/encoding, exact operator fixtures | Primary semantic authority, subject to identifying documented deployment approximations. [R10][R11][R12][R13] |
| NVIDIA target conversion | Required NVFP4 routed-expert values and calibrated W4A4 contract | Published qualification is not RTX 5090 or this distributed implementation. [R14] |
| LARQL | Demonstrates the remote FFN/expert service boundary | No requirement to import its database format, approximations, or service implementation. [R15][R16] |
| FlashMLA / FlashInfer / CUTLASS | Attention and low-precision CUDA building blocks | Qualify exact SM, dimensions, cache layouts, activation rules, and APIs before selection. [R17][R18][R19][R20] |

### 2.3 FreeToken code reuse and mandatory differences

The reviewed FreeToken files include `python/freetoken/moe/host_banks.py`, `offload_cache.py`, `cpu_executor.py`, `expert_banks.py`, `fused_nvfp4.py`, `offload_kernels.py`, and `benchbw.py`. Relevant concepts are addressed sources, packed bank schemas, reusable slots, representative transfer-versus-compute calibration, and safe overlap. These are source paths to inspect at the pinned revision, not a promise of ABI stability. [R21][R22][R23][R24]

Four differences are mandatory:

| Hazard | Required implementation response |
|---|---|
| Bank residency can imply whole-bank CUDA registration or a CPU execution choice | Use bounded staging/registration. Client host backing never selects CPU neural math, and CUDA-server misses never silently invoke CPU experts. |
| Prefill can reserve an entire expert layer or double-buffer entire layers | Execute bounded expert/row groups and supported tiles. Admission cannot depend on fitting the whole 384-expert layer. |
| Existing low-bit expert paths can be W4A16 | Implement the required NVIDIA W4A4 quantizer contract; do not equate shared weight bytes with identical arithmetic. |
| Default automatic strategies can select hybrid/CPU work | Expose only the explicitly selected server mode and client CUDA policy required here. |

FreeToken's documented CPU format coverage must be checked per revision; an existing BF16/FP4 CPU path does not establish the required FP8 CPU path. Its current framework/dependency choices are not automatically the dependencies of the native fork. [R06]

### 2.4 Experimental V4.1 status

At the inspected snapshot, FreeToken PR **#460** was open and described experimental V4.1 support with eager-only limitations. Its author-reported SM120 execution used a community-derived artifact, not the required NVIDIA checkpoint. The branch's documented low-RAM example uses lossy FP4 Engram tables and reclaimable file-cache residency; neither satisfies this specification's artifact identity or strict RAM-resident server contract. The branch documentation identifies a clamped-SwiGLU Triton path and rejects its generic b12x/Marlin alternatives for that implementation. Its actual activation quantization must be established from the pinned code, not inferred from the NVFP4 label or from a BF16 input/output signature. Mainline FreeToken also contains weight-only-style low-bit paths; that does not establish that every backend uses the same arithmetic. Graph/MTP qualification is separate. Treat it as a donor and test source, not a release oracle. [R08][R09]

In particular, do not substitute a smaller lossy Engram conversion, a different cache representation, or an approximate prefill path to obtain a favorable memory or speed result. Any such experiment must have its own explicit numerical identity and cannot satisfy the baseline contract.

### 2.5 Video and research conclusions

The supplied Chris Hay video, **“I Decoupled Attention from Weights - Gemma 4 26B,”** demonstrates local attention with CPU expert services. The earlier retrieved audit identified approximately 24–25 tokens/s with `localhost` services and approximately 1.8 tokens/s with Fly.io endpoints. The fast case is not a physical DDR3-server/LAN benchmark; the WAN case changes hardware as well as transport. No target-system speed is inferred. [R15]

LARQL's distributed FFN documentation supports the activation/contribution boundary and preserves layer dependencies. [R51]

LARQL's CLI distinguishes sequential dispatch from layer predispatch/refinement. The latter is not merely packet batching and must not be imported as an exact-forward-pass optimization. Database/KNN terminology does not eliminate input-dependent expert arithmetic. [R16]

Fiddler, FreeToken's paper, KTransformers, FlexGen, FlashAttention, PagedAttention, and MegaScale-Infer are useful supporting references for placement, offloading, attention tiling, or disaggregation. Their hardware, model, quality, and throughput results are not acceptance evidence for this system. [R33][R34][R35][R36][R37][R38][R39]

## 3. Target model and operator inventory

### 3.1 Required artifacts

The semantic source is `deepseek-ai/DeepSeek-V4.1-Flash`, including its checkpoint index, configuration, reference inference code, native encoding implementation, tokenizer, image processing, and Engram metadata. The initial NVFP4 artifact is `nvidia/DeepSeek-V4.1-Flash-NVFP4`. Use full repository revisions and verified file/tensor hashes. A filename, model display name, or quantization label is not an identity. [R10][R14]

**Observed source snapshot:** the publisher's model API reported revision `2cba9e42aa026125f3ed06c6d98c1db82f7ca027` on the audit date. This is a retrieved metadata identity, not a hash-verified download or qualified release lock. The same inventory includes `chat_template.jinja`, `encoding/encoding.py`, and native encoding test fixtures. Do not state that the current artifact has no Jinja template. Fetch and hash the selected revision's actual files before accepting their contents; metadata-level access did not validate every file at that revision. [R79]

The source model contains a 552B backbone and 196B Engram conditional memory; published active counts distinguish prefill from decode. These parameter counts do not determine physical bytes read per token. The serving package must inventory actual packed tensors and scales, not multiply a headline parameter count by one global dtype. [R10]

### 3.2 Configuration-derived audit checklist

The following is the inspected target configuration, not a substitute for the pinned manifest. A discrepancy must stop preparation until explained; it must not be repaired by importing V4 constants. Layer numbers below are zero-based. [R11]

| Property | Target value / invariant |
|---|---|
| Backbone | 40 layers, 20-layer causal encoder and 20-layer decoder |
| Residual / expert intermediate width | 5,120 / 2,304 |
| Attention heads / head width | 64 / 512 |
| Routed / selected / shared experts | 384 / 6 / 1 per backbone MoE layer |
| KV source layers | 2, 8, 14, 20 |
| Index source layers | 2, 8, 14, 20, 24, 28, 32, 36 |
| Sliding window / sparse selected entries | 128 / 512 |
| Hierarchical candidate source / block size | Layer 20 / 8 entries |
| Candidate block limit | 2,048, with source-defined handling of newest and partial blocks |
| Engram injection layers | 1, 14 |
| Router score / routed scaling / expert clamp | `sqrtsoftplus` / 1.5 / 10.0 |
| mHC streams | 4 |
| Vocabulary | 129,280; special IDs come from the pinned tokenizer/configuration |
| Configured position ceiling | 1,048,576; not a guaranteed admitted context |

Do not use the small default shapes in the reference's test configuration as the released model dimensions. Do not infer image routing bias, RoPE dimensions, norm epsilon, compressor settings, or Engram hash constants from this summary; import and verify their exact fields.

The generated operator inventory must additionally record query low-rank width, output-group/low-rank factorization, RoPE dimensions/base/factor and original positions, per-layer compression factors, index head dimensions, norm and mHC epsilons, router temperature and modality biases, exact Engram table/scale shapes, and vision patch/downsample/token-expansion metadata. These are extracted from the pinned files and are not inferred from the summary table. Include an explicit tensor-to-operation map with every required tensor consumed exactly where specified, every tied alias identified, and every deliberately inactive tensor explained. A nonempty unexplained-required-tensor list fails preparation.

### 3.3 Required functional components

The client implementation MUST include CED scheduling; CSA2 Full, Reindex, and Reuse behavior; source-layer KV/index aliases; hierarchical candidate selection; sliding-window attention; RoPE and compressed-position handling; attention sinks where present; cache quantization; Single-Pass mHC; router modality behavior; and learned Engram fusion. Each component receives an operator/state fixture before distributed execution is considered correct.

Preserve the distinction between the full mHC residual stream and the normalized expert input. RPC width comes from the expert descriptor, not from multiplying the residual width by the mHC stream count. Shared experts, output projections, embeddings, and vision tensors are real memory consumers, not negligible omissions.

### 3.4 Text, reasoning, tools, and images

Text generation, native reasoning-effort encoding, multi-turn continuation, and tool-call formatting are required core tests. Use the source encoding implementation or a tested equivalent; do not borrow another model's chat template. DeepSeek also identifies `deepseek-recipe` as an encoding/decoding implementation lead; it is not an inference server. Pin its schema and token-level fixtures before reuse. [R55] Tool output is returned as model output; this runtime must not execute tool calls. [R10][R13]

The published Jinja template is an implementation candidate, not an automatic replacement for native encoding. Establish token-ID parity for reasoning settings, history dropping, image placeholders, generation prefixes, tool schemas and tool-call/result ordering. Normalize API tool arguments through a declared adapter: a JSON-encoded argument string and a parsed argument object must not accidentally serialize as different model instructions. The retrieved template treats mapping arguments differently from other values, so test this boundary explicitly. A template's presence does not establish full image preprocessing or inference compatibility. [R79]

Text-only bring-up is a milestone. Preserve vision metadata/tensors, then qualify native image processing, encoder, projector, position handling, image-span routing bias, and generation before advertising complete multimodal support. CPU decoding of an image file is permitted; learned vision operations run on the client GPU. Host backing must cover vision work too when it exceeds its device reservation.

DSpark/MTP remains disabled in ordinary decoding baselines. Its presence in a checkpoint is not permission to allocate a second draft engine, alter state commitment, or report speculative throughput as ordinary decoding. A target-only runtime package may omit inactive draft tensors when that omission is explicit, hashed, and unrelated to backbone/Engram pruning.

## 4. Numerical profiles and quantization

### 4.1 Required profile definitions

A **profile** identifies storage, quantizers, scales, operations, accumulation, and output rounding—not just weight bit width. Each profile must run with both server executors and both fully resident and host-backed client tests.

| Profile | Locked baseline interpretation | Scope |
|---|---|---|
| `nvfp4` | NVIDIA artifact's routed-expert W4A4 contract, including its activation calibration and complete scale hierarchy | Non-routed components retain the artifact's declared mixed source precision. |
| `fp8` | Derived E4M3FN FP8 linear/expert representation using the source-compatible block-scaled W8A8 arithmetic contract; scales and activation block boundaries are explicit | Preserve source FP8 components where already correct. Convert the declared FP4 expert tensors reproducibly. Do not call unspecified per-tensor and block-scaled FP8 interchangeable. |
| `bf16` | BF16 storage/inputs for the declared weight operators, with required FP32 sensitive operations and accumulation/rounding rules | Derive weights reproducibly from their actual source. Native attention-state quantization and integer/row metadata remain independently specified. |

For the FP8 baseline, follow the inspected reference's **32-by-32 weight scale blocks, 32-element activation groups, E4M3 values, and UE8M0 scale semantics**, after validating the pinned source. A native microscaling instruction with another scale axis/grouping is not automatically a drop-in implementation. Repacking or a different hardware arithmetic path must preserve this contract or receive a separately identified realization. [R12]

BF16 is the preferred 16-bit profile; IEEE FP16 is not an alias. An additional FP16 or W8A16 experiment may be implemented only with an explicit identity and separate tests. It does not replace required BF16 or the defined FP8 profile. NVIDIA documents distinct FP8, microscaled, and NVFP4 representations. [R25][R26]

### 4.2 Native source versus NVIDIA conversion

The inspected native reference uses packed FP4 expert weights **with FP8-quantized inputs to each quantized linear operation**: its `linear()` dispatch calls `act_quant()` before both `fp4_gemm()` and `fp8_gemm()`. Thus native expert linear arithmetic is W4A8, not automatically NVIDIA NVFP4 W4A4 or a weight-only W4A16 path. These labels describe individual linear operations; clamp, coefficient, casts, and down-projection input quantization still belong to the full expert graph. [R12][R53]

The native source already contains low-bit expert weights. Expanding them to FP8 or BF16 does not recover unavailable high-precision master weights. Record this provenance and evaluate quality relative to the pinned source, rather than marketing a widened checkpoint as an original BF16 model.

NVIDIA describes its conversion as routed experts W4A4, retaining other components at source precision, and reports validation on GB300. Its reported weight conversion can preserve dequantized values while the activation scheme still changes inference. Those statements do not establish exact output equivalence or SM120 qualification. [R14]

Two different comparisons are mandatory:

1. **Placement fidelity:** distributed execution versus a local/reference implementation of the same profile and operator realization.
2. **Profile fidelity:** each derived/optimized profile versus the pinned native source across a declared evaluation corpus.

A CPU or CUDA path that uses NVFP4 weights with BF16 activations but skips required A4 quantization is W4A16, not the strict W4A4 path. This distinction applies to client hits, server hits, misses, and prefill equally.

### Native, conversion, and kernel identity checklist

The native model, published NVIDIA conversion, and a candidate optimized kernel are three separately checked objects. Do not infer a successful match from a shared checkpoint name. The import report must identify source FP4 values and UE8M0 scales, NVIDIA FP4 values plus per-16 E4M3/global scales, all calibrated input scales, and the exact operation used after decoding each representation. A lossless weight transcode does not establish identical activation arithmetic.

For `fp8`, the specified 32-by-32 weight-scale grouping is a source-compatible project profile. Hardware MXFP8 instructions commonly consume one-dimensional groups along K; expanding a shared source scale into a supported physical layout is only a layout operation when the resulting arithmetic and scale domains agree. Never replace a 2D source quantizer with a different per-row quantizer during repacking. The reference `act_quant()` rounds scales to powers of two when its MX-format control is enabled; preserve its zero, saturation, and rounding behavior rather than estimating scales from a generic maximum alone. [R12][R53]

`bf16` conversion is a deliberate prepared artifact, not whole-bank expansion at service startup. Record precisely which linear weights are widened, which source FP32 tensors remain FP32, and which attention/Engram/cache representations remain mixed. The strict NVFP4 baseline uses the publisher's calibration; a new calibration dataset, missing scale repair, or changed global multiplier creates a different realization and requires both accuracy comparisons.

The native W4A8 implementation is not itself a strict W4A4 oracle. The source-lock milestone must inspect the pinned NVIDIA-tested runtime/converter integration to locate activation quantizers, coefficient application and casts, and build an independent W4A4 expert fixture. The project preserves the declared source graph with the selected profile quantizers; any publisher runtime that changes the coefficient/quantizer order is an explicitly different realization, not silently interchangeable proof of parity. Resolve such a discrepancy in the committed operator contract before optimizing either CPU or CUDA. Published model-card accuracy is evidence about the publisher's tested setup, not a replacement for this decision. [R12][R14]

### Publisher integration and frozen quantizer scope

The NVIDIA artifact is mixed: its configuration can retain a top-level FP8 selector while separately describing NVFP4 routed layers. Import using the actual per-module descriptors, tensors and scales; one top-level `quant_method` must not override the routed-expert inventory. The publication cites SGLang commit `da64c5cbb8cf6bfd39be19da43573fdfd484c43a`. Its `modelopt_quant.py` is an inspected, immutable implementation lead: the FlashInfer CUTLASS/TRT-LLM path reduces loaded activation scales with a layer-wide maximum and preserves separate gate/up weight factors. Other branches choose different scale behavior. [R14][R84][R85]

A realization adopting those reductions must prepare them once over the **full declared original expert/scale population**, including source gate/up distinctions, and place their hashes and scope in the operator manifest. Never derive a replacement maximum from a client cache, NUMA shard, selected-expert subset, or current prefill microbatch. Per-block dynamic quantization still operates on the correct runtime values; a calibrated global factor does not make every block scale constant. Generic fallback scale values, training-only random transforms, stochastic rounding recipes, and environment-driven W4A16 switches are not part of the strict profile unless explicitly present in its approved graph.

The reference fixture must trace both projections and their scale/cast boundaries through the exact selected publisher integration. Preserve the source clamp and routing-weight position when defining this project's realization. If a publisher backend applies a different operation order, record that difference and do not use it as an identical oracle. The evidence package must resolve this at the operator-fixture gate; the handoff does not claim to have run or numerically certified the publisher backend. The pinned integration is a source of small kernels/tests, not a requirement to deploy SGLang as another model server.

### 4.3 Representation contract

The manifest records, for every quantized tensor and operator:

- Logical shape, physical byte shape, signedness, value dtype, nibble/byte ordering, padding, transposition, and packing.
- Scale tensor identities, dimensions, numeric types, direction (scale versus inverse), block axes, group sizes, global scale, zero/special-value behavior, and any swizzle.
- Input/output dtypes, activation quantizer, clipping, calibration provenance, quantizer reduction domain, rounding, saturation, intermediate casts, accumulator precision, and reduction order.
- Source values and conversion/repacking recipe, kernel realization, prepared byte hashes, and deterministic handling of padding.

NVFP4 weight values and scales are not ordinary integer Q4_K blocks. Source MXFP4, NVFP4 expert weights, source compressed attention state, and FP8 index/state representations must have different explicit format identities. The expert profile must never globally cast the KV cache, scales, routers, norms, or Engram tables.

### 4.4 Quantization must survive partitioning

Splitting selected experts between devices must not change a quantizer's reduction domain. A scale computed across an entire input row, projection, or calibrated tensor cannot be recomputed over only the local subset and assumed equivalent. If exact quantization needs a shared statistic, compute it once at the required scope or implement an equivalent bounded two-pass operation.

The same rule applies to prompt-row microbatching, matrix tiling, output-channel partitioning, and NUMA workers. Scratch limits cannot justify changing scale granularity. Per-expert scales stay associated with their expert and projection; shared input values do not imply identical calibrated input scales for every projection.

Compiler/library math settings are part of the profile: record FMA contraction, fast-math approximations, TF32 permissions, denormal handling, rounding mode, and deterministic reduction settings where they affect outputs. Sensitive FP32 operations must not silently use a reduced-mantissa matrix path because it is a library default.

Wire FP32 is a transport choice. It carries the declared input values and necessary quantizer metadata; it does not bypass internal quantization. The wire descriptor identifies the logical input dtype and required rounding before the first linear operation. Local and remote execution must consume the same logical values.

### 4.5 Exact expert semantics

Implement the pinned V4.1 operation, not a generic SwiGLU convenience kernel. The reviewed reference distinguishes an upper-only gate clamp from the two-sided up clamp, applies the routing coefficient to the gated intermediate before the down projection, and combines routed and shared results in a defined sequence. The router's selection bias is not itself a contribution weight. [R12]

The implementation contract is:

- The client supplies the original selected expert IDs, selected-slot associations, and coefficients. The server does not rerun routing.
- Clamp placement, SiLU, multiplication, intermediate rounding, down-projection quantization, and output casts follow the profile's declared graph.
- Reference-sensitive operations use their required precision; do not fuse away a meaningful rounding or quantization boundary.
- Each requested selected slot is evaluated once. A shared expert is evaluated once per required row, not once per remote shard.
- **Reduction order is derived from the pinned numerical reference, not network arrival order.** The inspected single-rank reference accumulates routed contributions by increasing expert ID in FP32, then adds the shared result and applies the declared cast. Selected-slot IDs identify contributions but do not by themselves define their accumulation order. [R12]

Do not promise bitwise equality across CPU, CUDA, or different GEMM tilings. Exact transport/identity/selection fixtures must match exactly; numerical outputs use declared tolerances. Small numerical changes can alter later near-tied routing decisions, so model-level drift must be measured rather than hidden by a claim of universal route identity.

### 4.6 Strict state semantics versus deployment approximations

**SWA Bounded Replay is not a lossless paging mechanism.** DeepSeek's report explicitly describes reconstruction from a bounded recent segment as approximate and distinguishes it from exact reconstruction. [R27]

The correctness baseline therefore retains necessary state or reconstructs it using the complete required dependency history. The initial correctness oracle is an explicitly identified full-reference schedule. The inspected reference loops through all backbone layers in `Transformer.forward`; it does not itself prove an exact encoder-only prefill optimization. Retaining sufficient state or computing the missing decoder history may require more work than the advertised half-prefill path. A faster exact CED schedule is admitted only after its decoder SWA initialization and continued-prompt semantics are demonstrated equivalent. Report the schedule identity alongside profile, throughput, and state footprint. [R12][R27]

Do not import approximate bounded replay, speculative layer predispatch, lower-quality Engram tables, or altered sparse selection as invisible performance fixes. A later approved approximation must have a separate quality profile, evidence, and user-visible declaration. It cannot satisfy the baseline fidelity gate.

## 5. Runtime organization and source locking

### 5.1 One model engine and a narrow expert boundary

Retain a single DwarfStar-derived native model/session engine. Refactor the target execution path directly rather than layering full FreeToken, LARQL, and DwarfStar servers together. Use C/C++ host components and CUDA kernels compatible with the existing native base. Offline Python tools and Python reference tests are appropriate; they must not create a mandatory full-model Python runtime on either serving host.

A small compiled kernel dependency, including an AOT-produced kernel when support is proven, is acceptable behind the same operator interface. Otherwise port the required operation into the native CUDA backend. No general plugin framework is required. The CPU and CUDA expert implementations are two explicit implementations of one narrow interface because both are requirements, not compatibility fallbacks. The Docker CPU target must not discover/link a CUDA package just to compile an unused branch.

The first execution path is eager and event-driven. CUDA graph capture is optional after pointer lifetime, slot layout, input/output shape, network boundaries, and paged-state tests pass. Graph capture must not be a prerequisite for correctness or a reason to reserve the whole model.

### 5.2 Source lock deliverable

Before implementing against donor code, create an immutable source lock containing:

| Item | Required identity |
|---|---|
| Selected DwarfStar base and related donor | Repository URL, full commit, exact imported files/patches, license |
| FreeToken main and experimental PR material | Full base/head commits, imported files/tests, unresolved upstream limitations |
| DeepSeek model/reference | Repository revision, file hashes, config/tokenizer/encoding identities |
| NVIDIA checkpoint/conversion | Artifact revision, shard/index hashes, calibration metadata, converter identity |
| CUDA kernels/libraries | Full version/commit, enabled SM targets, compiler flags, local patches |
| Toolchain | Compiler/linker, CUDA toolkit, minimum tested driver, OS/libc, CPU ISA build target |
| Build output | Build recipe, dependency lock, binary hashes, kernel support matrix |

Disagreement among the source graph, conversion metadata, and candidate kernels must be recorded in the numerical contract before choosing an oracle. A reference correction is a reviewed change with its own fixtures, not permission to select whichever implementation produces a favorable result.

The source lock MUST distinguish an author-reported test commit from this project's tested build. An upstream old-layout compatibility commit must not be mistaken for the current V4.1 implementation. Record new findings before porting; do not silently follow moving branches.

### 5.3 Blackwell qualification

SM120 client execution is required. Datacenter Blackwell and consumer Blackwell are not one interchangeable compiled target. FlashMLA's inspected requirements name SM100/SM103 and CUDA 13.1+, while FlashInfer exposes separate SM120 capabilities. This is a qualification gap, not an instruction to load an SM100 binary on a 5090. [R17][R18][R28]

Select the minimum coherent toolchain that compiles and passes all required kernels; record it instead of assuming the newest toolkit fixes semantics. Inspect architecture-specific instructions, shared-memory limits, launch constraints, shapes, and quantized cache layouts. Source constants and precompiled kernel inventories must be validated on the actual device.

### Accelerated SM120 compilation contract

The physical RTX 5090 target remains compute capability 12.0 / SM120. That is **not** the same as selecting every required instruction feature in the compiler. NVIDIA's PTX specification requires `sm_120a` for the SM120 block-scaled `mma` forms used by native low-bit Tensor Core kernels; compatible family-target forms are available as `sm_120f` with the specified PTX/toolchain support. Plain `sm_120` is not a sufficient build contract for those operations. [R80]

For this exact RTX 5090 release, compile the relevant native CUDA translation units with the architecture-accelerated target **`sm_120a`**, for example `-gencode arch=compute_120a,code=sm_120a`, using the tested toolkit. Packaging exposes `CUDA_ARCH=sm_120a`. A separately qualified `sm_120f` build is allowed only when every instruction and library involved supports that family target; it is not silently substituted. Ordinary kernels may have their own appropriate target, recorded in the kernel inventory. This suffix denotes instruction compatibility, not a different advertised GPU or an application-version suffix.

The image probe must actually launch small native block-scaled FP4/FP8 operations with the required scale layout as well as BF16, not just query device capability. Inspect emitted code and launch resources. Architecture-accelerated binaries and PTX are not assumed portable to SM100/SM103 or another consumer target; compile and qualify those separately. Compilation success proves neither numerical semantics nor enough workspace for the target dimensions.

### 5.4 Licenses and distribution

Preserve notices and licenses for all copied source, kernels, checkpoint conversions, tokenizers, and tests. FreeToken presents an Apache-2.0 license. The selected DwarfStar license and the related donor both present MIT licensing; the selected license includes ds4 and ggml notices. Preserve those notices and verify every imported component at its pinned revision, including model weights and ancillary dependencies. Do not describe the combined work as solely MIT if Apache-licensed material is included. [R40][R41][R50]

The release contains source patches, reproducible build instructions, lockfiles, operator fixtures that may legally be redistributed, and attribution. It must not bundle restricted model weights, user data, private credentials, or proprietary binaries without appropriate rights.

## 6. Checkpoint preparation and tensor identity

### 6.1 Input and prepared representation

Ingest the official sharded safetensors inventory and configuration without executing untrusted checkpoint code. Offline reference/conversion code runs only from the reviewed source lock. GGUF remains useful for comparison or existing DwarfStar functionality, but a GGUF-only loader is insufficient for the required source/NVIDIA artifacts. A four-bit GGUF label is not NVFP4 acceptance.

Use one prepared representation: an immutable manifest plus indexed, bounded-size tensor shards. Safetensors shards may store packed bytes and explicit scale tensors; the manifest supplies logical quantization and operator information missing from a byte-array dtype. Reuse a valid source shard without conversion where possible. Do not create a new general-purpose container, database, or required chain of intermediate formats.

Kernel-specific packed layouts are declared alternate tensor realizations, prepared deterministically with complete identities. They may share unchanged source bytes/scales but cannot masquerade as the source layout. Production must not require simultaneous resident copies of every source and every repacked model.

### 6.2 Manifest contents

| Group | Required fields |
|---|---|
| Identity | Model family, source revision, profile, operator-contract digest, prepared-manifest digest, tokenizer/encoding digests |
| Graph | Layer roles, all shapes, source-layer aliases, expert counts, routed/shared graph, Engram configuration, feature exclusions |
| Tensor | Stable ID/name, logical and physical shape, dtype, payload shard/ranges, scales, layout, hashes, tied/shared identity |
| Operator | Inputs/outputs, quantizer scope, casts, accumulation, reduction/tie behavior, required backend capability |
| Placement | Authoritative tensor population, allowed client subset, server-required set, optional cache entries, residency eligibility |
| State | Object classes, producer/readers, page layout, growth function, alias graph, update granularity, profile identity |
| Provenance | Converter source/build, recipe, calibration source/seed, rounding, changes from native source |

Separate **common logical model/profile identity** from **backend-specific prepared-layout identity**. CPU and CUDA, or SM120 and SM100, may have different checked physical layouts for the same declared operation. Peers must agree on logical tensors, values/scales, quantizer semantics, and operator contract; they must publish their respective approved physical-layout digests and conversion mappings. Do not require physically different CPU/GPU banks to have one identical byte digest. A byte-identical bank is likewise not proof of identical arithmetic. State and cached objects carry the exact physical representation actually stored.

The manifest describes data and a fixed supported model, not executable arbitrary graphs. Unknown essential fields, tensor kinds, or graph variants fail preparation. Metadata sizes, element products, byte offsets, and lengths use overflow-checked arithmetic. File locations are resolved within an authorized preparation root; network clients never supply arbitrary filesystem paths.

### Preparation integrity and partial transfer

Use SHA-256 for source, prepared-shard, and manifest integrity. Source-file hashes cover the exact retrieved bytes. Project JSON identity objects use **RFC 8785 JCS** UTF-8 canonicalization, not an unspecified `sort_keys` operation. Reject duplicate keys, malformed Unicode/lone surrogates and nonfinite numbers before canonicalization; preserve string values without Unicode normalization. Compute a manifest's digest over its canonical object after removing only the top-level `digest` and `signatures` members. Child/index/source digests remain covered. Digest text is exactly 64 lowercase hexadecimal characters. Verify the RFC's sorting/string/number fixtures in the chosen C++ and preparation implementations. [R81]

All ordinary integer-valued project JSON numbers are exact and restricted to `0..9007199254740991` (or an explicitly allowed signed safe-integer range). Full-width u64 **handles, request/invocation IDs, epochs and unrestricted file offsets** use schema-declared decimal strings matching `0|[1-9][0-9]*`, checked against their unsigned 64-bit range; they are never parsed through binary64. Field types are fixed by the schema, not changed opportunistically by value. Runtime arithmetic still uses checked integers. Numerical-contract scalars whose exact floating bits matter are typed IEEE bit strings or tensor bytes, not ambiguous decimal reserialization; ordinary user sampling parameters use their separately specified numeric conversion. These rules preserve multi-terabyte capacity without JSON precision loss. [R81]

Canonical encoding is used for manifest/plan/bind identity; a stored pretty-printed file may be parsed and canonicalized before verification. Hashing a submitted JSON byte string without applying its declared identity scheme is not equivalent. Signed release metadata, when used, signs the canonical digest. A content hash is not an authenticity proof: the initial root/model/profile trust anchor is provisioned from the reviewed source/release lock, not from the untrusted service being authenticated.

A complete-shard hash cannot independently verify a small range before the rest of that shard is present. Prepared shards therefore include a manifest-authenticated fixed-size chunk-hash index, with a declared chunk size no larger than the bulk transfer payload allowance. Bootstrap/cache reads request complete verification chunks or an explicitly validated final partial chunk, check the expected digest, then extract the authorized tensor subrange. Hash bytes before interpreting scales or exposing an entry. Count overfetch and scratch; the implementation must not download a whole huge shard merely to validate one client cache entry.

Safetensors stores offsets relative to its data area, after its little-endian header-length field and JSON header. Validate against that format, not absolute offsets guessed from a tensor name. Use a maintained parser or a fuzz-tested small parser; reject malformed metadata, duplicate names/keys, overflows, invalid dtype byte sizes, out-of-file intervals, and illegal overlap. Tied logical tensors are represented by an explicit alias rather than contradictory duplicate payload ranges. [R54]

The preparer uses bounded streaming and atomic publication on the same filesystem: write temporary outputs, close and synchronize completed data as required for restart durability, rename completed shards, and publish the final manifest last. An interrupted build can resume at verified shard boundaries. Preserve the disk-space admission estimate for source plus output plus in-progress shard and calibration scratch. Runtime containers never run an unreviewed model script downloaded from a mutable branch.

### Noncircular trust and bounded metadata bootstrap

The trust graph is directed: reviewed source identities and conversion recipe → tensor/chunk hashes and logical operator contract → prepared layout manifests → native build/artifact descriptors → externally generated image/deployment locks. A manifest does not hash itself. An image does not embed a deployment lock containing that image's final digest. Role-specific deployment locks may differ while binding the same logical model/profile. Prepared CPU and CUDA layouts are independently hashed and mapped to that logical identity.

Provision the compact trusted root and the hash-index metadata needed to authenticate the client's permitted preparation objects before network bootstrap. A large index can itself be split into authenticated metadata objects referenced by this root, fetched before their dependent data; it must not depend on a data chunk whose hash is obtainable only by trusting that same chunk. Bound the metadata graph, node count, nesting, index bytes and traversal; reject cycles. Bootstrap may transfer metadata and authorized chunks, never run downloaded code.

Authorization is evaluated at the **actual verification-chunk boundary**. A whole chunk can contain neighboring tensor bytes. The preparation grant must cover all bytes returned, or the preparer must place differently authorized populations in separate chunks/shards. Do not claim tensor-level confidentiality while serving unauthorized neighboring bytes as hash-verification overfetch. All startup/offline metadata, hash indexes and partial-transfer buffers count toward the relevant host budget.

### 6.3 Bounded preparation pipeline

The preparer streams tensor blocks, checks their source hashes and ranges, converts/reorders only necessary blocks, writes final shards atomically, and verifies their output hashes. It must not instantiate the complete model in FP32/BF16 to export a packed profile. Offline CPU arithmetic is permitted and is separate from the serving-time client CUDA policy.

Process shard headers defensively: limit header bytes and tensor counts, validate dimensions and nonoverlapping/authorized ranges, handle tied storage explicitly, verify exact reads, and reject truncation or unrecognized packing. Short reads, interrupted reads, and alignment boundaries must be handled deliberately rather than treated as zero-filled data.

Calibration for a new quantized realization is reproducible and receives a separate artifact identity. Reusing published calibration is preferable for the strict NVIDIA artifact. A fallback calibration entry in a publisher artifact is part of that artifact's provenance; do not invent replacement values during loading.

Startup staging must account for source mappings, page-cache residence, prepared buffers, and deliberate copies at peak. A loader that eventually frees duplicate model memory may still fail the peak-memory contract. Save restartable preparation progress at completed immutable shard boundaries, never a misleading completed manifest for a partial artifact.

### Address-width and alignment contract

Checked large offsets must survive **every** boundary: shard offsets, prepared-bank strides, host pointer arithmetic, CUDA kernel arguments, device address calculations, transfer lengths and library descriptors. Promote operands to the required 64-bit type **before** multiplication; casting an already-overflowed 32-bit product does not fix it. Kernel-local dimensions may use narrower verified ranges, but byte strides into multi-GiB banks must not inherit a signed 32-bit limit. A single packed projection bank or GPU cache can exceed 2 GiB even when its expert count fits u16.

Validate logical/physical alignment, endianness, nibble order, padded extents and byte bounds before launching. Backend packing must not turn an aligned parent allocation into a misaligned per-expert slice. Require sentinel/canary tests at offsets around 2 GiB and 4 GiB, representative actual device allocations where available, and synthetic checked-address tests above 1 TiB. Report actual device-address tests separately from synthetic host arithmetic; neither a small tensor nor a file-size test proves large kernel strides correct.

### 6.4 Server loading

Load the declared active tensor population into the chosen RAM arrangement. Retain one canonical logical copy plus explicitly budgeted NUMA replicas or necessary alternate layouts. Prefault and verify residency for the strict resident configuration. No full-population CUDA registration is required.

Before accepting a session, report loaded bytes by tensor class, per-NUMA physical placement, retained alternate layouts, intentional replicas, reserved scratch, host/VRAM pools, and startup phase durations. The complete runtime model includes required non-expert modules; target-only exclusions apply only to explicitly disabled features, not to active experts or required Engram rows.

The on-disk immutable checkpoint is restart storage. Steady-state server expert evaluation and Engram lookup must not depend on disk reads, swap, or reclaimable pages silently reloaded from files. If requested residency cannot be established, fail readiness. OS memory residency and CUDA host registration are different controls and must be reported separately. [R29][R30]

### 6.5 Client preparation

The client receives the compact graph/tokenizer metadata, its admitted non-routed backing tensors, and selected expert/row-cache entries. It need not download or map the complete model. Cache preparation and bootstrap can use authenticated tensor-range requests, but neither is the inference expert-miss path.

Do not create a huge sparse local file, fake model pointer, or dummy allocation to satisfy a loader that expects every expert. Refactor tensor access into checked handles so an absent client expert has metadata and a remote execution route without pretending its bytes are local.

A client cache entry becomes executable only when every necessary projection, scale, and layout descriptor is complete and verified. Interrupted preparation produces no partial valid entry. Persistent client cache files, if implemented, are immutable verified preparation artifacts; they are not a third paging tier during inference.

## 7. Computation placement and layer execution

### 7.1 Complete ownership table

| Component | Canonical bytes | Computation / owner |
|---|---|---|
| Attention, query/output projections, norms, positional operations | Server manifest; admitted client host/device copies | Client CUDA |
| Sparse indexing, compression, mHC, residual mixing, router | Server manifest; client copies | Client CUDA |
| KV/index/SWA/compressor/Engram continuation state | Client session memory | Client CUDA updates; client owns host/device validity |
| Shared experts | Server canonical copy; client backing/residency by plan | Client CUDA by default; a declared shared-expert service operation may execute at the selected server executor |
| Embeddings and learned vocabulary head | Server canonical copy; admitted client host/device backing | Client CUDA, including tiled/staged execution; not a CPU operation disguised as sampling |
| Routed experts cached in client VRAM | Server canonical plus client cache | Client CUDA |
| Routed experts cached only in client RAM | Server canonical plus client host cache | Client CUDA after staging when selected by plan; otherwise server |
| Routed experts absent from both client tiers | Server RAM | Selected CPU or CUDA server executor |
| Engram table rows and scales | Server RAM | Host byte lookup; client CUDA learned fusion |
| Vision encoder/projector | Server canonical; admitted client backing | Client CUDA after vision qualification |
| Tokenizer, transport, bookkeeping, scheduling, output parsing | Small local metadata | Host control code |

Moving an allowed shared/dense expert operation to the server must be explicit in the placement manifest and supported by both the selected executor and its numerical contract. The server still does not host another attention loop, router, vocabulary sampler, or full model. Prefer keeping embedding/head math on client CUDA with host backing; do not expand remote operations merely to avoid implementing client staging.

### 7.2 Exact selected-expert execution

Conceptual control flow, to be integrated into the actual V4.1 graph:

```text
run required client graph up to normalized expert input and routing
record input logical dtype, quantizer metadata, selected IDs/slots/coefficients
partition selected work into client-CUDA and remote sets using admitted plan
submit bounded remote request when its set is nonempty
stage host-cached client operands and execute local selected experts
execute the shared branch at its declared owner exactly once
receive and validate every remote contribution
assemble contributions in canonical reference reduction order
perform the required cast, residual/mHC transition, and next graph operation
```

This is not permission to reorder the graph around Engram or mHC boundaries. The model-specific caller supplies the correct input point and resumes at the correct output point. Remote expert evaluation is stateless mathematically; the client's overall generation is not stateless.

For the same operation, each `(session epoch, invocation, input row, layer, selected slot)` has exactly one execution owner. Original slots remain identifiable even when work is grouped by expert, sent out of order, or assigned to a NUMA replica. Unknown, missing, repeated, or mismatched contributions abort the request. Nonfinite expert inputs, coefficients or returned contributions are failures unless an explicit operator contract permits a particular sentinel; attention's internal masked-score negative infinities do not authorize NaN expert results on the wire. Check asynchronously on the executing device where appropriate without moving client neural math to CPU.

### 7.3 Prefill

Prefill MUST be implemented, not deferred behind a decoding demo. Process bounded chunks in the actual CED dependency order, group independent rows by expert for reuse, and scatter results back to their original row/slot association. One long prompt may touch all experts across its rows even though each row selects only six.

Choose expert/row subgroups and microbatches so weights, intermediate values, quantizer scratch, result storage, network frames, and transfer slots remain bounded. Keep per-row quantization semantics invariant under microbatch changes. Never allocate all routed expert matrices, all dequantized prompt experts, or the complete layer union merely to call a fused prefill routine.

Profile prefill separately from decoding. Protect useful decode cache entries from one-time prompt-wide streaming where possible, but do not reserve so much protected cache that mandatory prefill cannot execute. Cache preparation and deferred kernel compilation are counted in startup or first-token timing, not silently omitted.

### 7.4 Legal concurrency

Local and remote branches can overlap within a layer. Known independent copies can overlap computation. Engram rows for known token history can be prefetched before their consumption. Different selected experts or independent prompt rows can execute concurrently on server workers.

The next dependent layer waits for its required result. A single exact autoregressive token does not turn all layers into independent requests. Avoid speculative route prediction, early advancement on partial contributions, and evaluating the same expert on two devices to select whichever finishes first. These add complexity and do not belong to the initial contract.

The host networking path must continue progressing while CUDA work executes. Use bounded queues and completion notifications; do not put a blocking network wait inside a device callback that prevents the transfer needed to finish the request.

## 8. Client VRAM and system-memory management

### 8.1 Separate budgets, one owner

Client RAM is a required storage tier. A 32 GB GPU plus 64 GB RAM is not a 96 GB GPU; bandwidth and execution constraints remain distinct. The RTX 5090 supports PCIe Gen 5, but actual bandwidth depends on its negotiated host link and memory subsystem. Record both rather than infer link speed from DDR5. [R31]

Maintain independent caps for total host allocation, registered/pinned memory, device allocation, private-state backing, immutable host weights, expert caches, row caches, and transport/staging. Budgets are resolved before session admission and recorded in a plan. Installed RAM is not free usable RAM; account for OS reserve, other processes, cgroups, and startup peaks.

Mandatory executable buffers and state reservations take priority over optional caches. Above those floors, select residency by measured latency saved per byte. Do not force attention state out of VRAM merely to improve expert-hit counters.

### 8.2 Client expert tiers

| Residency | Normal action |
|---|---|
| Complete valid VRAM entry | Execute on client CUDA. |
| Complete valid host entry only | Execute locally after explicit staging when the phase-specific plan favors it; otherwise request remote evaluation. |
| Neither local tier | Request remote evaluation; no demand network expert-weight fetch. |

Client host expert entries remain packed. CUDA performs serving-time numerical unpacking, activation quantization, and expert math. The client CPU may gather/copy prepared bytes, not run a numerical dequantize-and-GEMM fallback. Offline artifact conversion is a different phase.

A GPU copy of immutable weights can be discarded without writeback when a valid source exists. Retain host membership independently of transient device residency. Cache keys include checkpoint, profile, tensor realization, layer, expert, and all scale/layout dependencies. Never cache input-dependent expert outputs as interchangeable with weights.

### 8.3 State-object model

Each private state object records:

| Field | Purpose |
|---|---|
| Session and epoch | Prevent cross-session or stale reuse. |
| Logical identity and producer | Identify shared KV/index objects and their real source layer/phase. |
| Logical token/element range | Associate pages with model positions, not just allocation offsets. |
| Representation and layout | Include cache dtype, scales, stride, masks, and any ring-buffer interpretation. |
| Generation / dirty ranges | Identify the current value and host/device synchronization requirement. |
| Host and device locations | Bounded handles; either can be absent when permitted. |
| Producer and copy events | Publish validity only after completion. |
| Reader leases | Prevent eviction or overwrite while kernels or DMA still read the object. |

A minimal explicit state machine distinguishes unallocated, host-valid, copy-in-progress, device-valid, dirty-device, writeback-in-progress, and invalid states. It is valid to have both copies current. A canceled or failed copy never advances validity. In-flight device access cannot outlive its buffer lease.

### 8.4 Eviction and writeback

For an immutable state page with current host backing, device eviction needs no writeback. For new or dirty state, write back the actual changed range before discarding its last valid device copy. If the kernel rewrites a complete page, track that complete dirty page; do not assume every update is append-only.

A ring buffer must carry logical positions/generations so a recycled slot cannot be read as an earlier token. Partially filled compression groups require their live accumulator/input state, not only their finalized packed outputs. Shared source-layer objects remain live until every model-defined consumer finishes.

Optional immutable caches may be reclaimed safely when an admitted state reservation needs the space. Private state cannot be discarded merely because it has a lower cache priority. If the admitted RAM/VRAM hierarchy cannot support the workload, fail rather than adding remote private-state backing, swap, or client CPU attention.

### 8.5 Transfer implementation

Use explicit migration through bounded reusable pinned pools and device execution slots. Ordinary resident host arenas may feed those pools through a measured gather/copy; bounded registration of selected long-lived ranges is a qualified alternative. Do not pin all client RAM, repeatedly register the same buffers per token, or use Unified Memory page faults as the main admission mechanism. [R32]

Separate transfer intent: required state reads, required weight reads, dirty writeback, expert staging, row retrieval, activation replies, and optional promotion. Prioritize operation progress and necessary writeback above speculative work. Count staging duplicates and H2D/D2H independently.

Double buffering is used only when there is legal overlap and its extra memory fits. CUDA streams and events express dependencies; they do not create additional PCIe bandwidth. Every kernel consumes stable, completed buffers. Captured graph pointers must be updated safely or use fixed slots; a freed/reused address is not valid simply because a graph still references it.

Mapped host memory/UVA may be evaluated for sparse, coalesced one-touch retrieval. It is not permission to treat discrete-GPU host access as VRAM-speed or bypass tracking of the bytes consumed. The baseline remains explicit staging.

### 8.6 Attention-side weight staging

Small, frequently reused attention/router/norm/mHC weights should remain device-resident when beneficial. Larger attention/shared/head/vision tensors can remain in client RAM and stage through bounded operation/layer/tile slots. Prefetch according to the actual CED schedule, not a simplistic universal next-layer assumption.

If a kernel requires a contiguous tensor larger than the available execution slot, implement valid tiling or reject that configuration. Matrix partial sums, softmax normalization, sparse candidate selection, and quantizer scales must remain mathematically correct across tiles. Splitting a softmax into independently normalized fragments is not an attention implementation.

Where a reference attention operation permits streaming exact score blocks, preserve a running maximum `m`, denominator `l`, and weighted numerator `z`. Combining an existing aggregate with a new block uses the common maximum `m_new`, rescales both aggregates by their exponential maximum differences, and adds their denominators/numerators before computing `z/l`. Handle all-masked blocks, sinks, causal/window overlap, precision, and sparse index order explicitly. This is a mathematical tiling rule to validate for each operator, not a claim that an existing backend already implements it.

Precisely, for existing `(m,l,z)` and a new nonempty valid-score block `(m_b,l_b,z_b)`, use `m_next=max(m,m_b)`, `a=exp(m-m_next)`, `b=exp(m_b-m_next)`, `l_next=a*l+b*l_b`, and `z_next=a*z+b*z_b`; normalize only once as `z_next/l_next`. Here `z_b=sum(exp(score-m_b)*value)` and `l_b=sum(exp(score-m_b))`. An empty/all-masked block contributes no update, so do not evaluate `exp(-inf-(-inf))`; initialize from the first valid block. Sinks are accounted exactly as the reference defines, including any denominator-only contribution. A query with no valid score/sink follows an explicitly tested source behavior, not an invented uniform distribution. Tile dimensions must not change the mask, candidate set or required scale/cast boundaries.

## 9. V4.1 attention, state, and exact continuation

### 9.1 Backend selection is an operator matrix

The client needs qualified implementations for sparse attention, sparse indexing, compressed-cache production, mHC, projection GEMMs, and each transition—not one global `attention_backend` label. Select a tested implementation for each operation and phase, then bind that support matrix into the build/manifest.

FlashMLA's V4.1 interface is an important semantic/layout source. FlashInfer exposes SM120 sparse-MLA configurations and Blackwell expert/GEMM interfaces. Matching a name, head dimension, or nominal dtype does not establish compatible packed rows or quantizers. Existing donor kernels must pass exact V4.1 layout, mask, clamp, and state tests. CUTLASS is another candidate implementation source, not a replacement for that qualification. [R17][R18][R19][R20][R45]

No server attention backend is required. The CUDA server initializes only expert operations and permitted declared non-routed work, not duplicated client KV/index state.

### 9.2 Ownership-aware paging

KV reuse and index reuse have different source-layer sets. Allocate each shared logical cache once per actual host/device copy, with aliases for its readers. Keep query-side computation and learned indexing on CUDA even when their operand population is in host RAM.

Where correct, keep compact index metadata and hot window state resident, select candidates on GPU, then gather the exact needed compressed-KV pages. Some indexers must scan a broader key population to choose those candidates. Account for those reads; do not report only the final top-k gather as all attention traffic.

Packing includes scales, positional meaning, sinks, sparse validity counts, causal masks, and partial groups. Preserve already-quantized history by moving its bytes; repeatedly decoding and requantizing state can introduce drift. Cache precision is independent of the expert profile.

### 9.3 CED and continuation

Implement a tested execution schedule for initial prefill, token decoding, appended user/tool input, repeated turns, and reset. Encoder outputs used to create decoder global KV and decoder-local SWA state have distinct lifetimes. A source-layer cache being shared does not imply that every decoder state can be reconstructed from that one object.

For the baseline, preserve required SWA/encoder continuation state in client RAM when it cannot remain in VRAM. Exact reconstruction is permitted only from a sufficient recorded dependency history and must be compared against retained-state execution. Do not select the report's approximate bounded replay just because it is an upstream deployment optimization. [R27]

A model's published long-context limit does not waive admission. The reported compressed global cache density is not the full allocation: include SWA, indexers, compression intermediates, encoder/decoder boundary state, Engram history, vision state where used, graph buffers, and live arithmetic.

### Reference limitations that the fork must address

The inspected minimal reference is not a turnkey arbitrary-chunk server: its nonzero-position window helper assumes the decode-form query, and its image embedding path asserts initial-position processing. Required bounded prefill, appended prompt chunks, image spans crossing chunks, and exact continued conversations need dedicated implementation and parity tests. Do not remove an assertion and assume the cache indexing is correct. A reference used for teacher-forced loss must expose logits at the necessary positions, rather than compare only the final position that an inference helper returns. [R12][R13]

The parser/state planner must distinguish normalization domains. `hc_mixes()` uses the flattened **hc × residual-width** vector for its normalization. Engram fusion normalizes each hc-stream vector over its own residual width before its matching score; treating the entire Engram hc stack as one normalized vector changes the model. This is a required negative-test fixture. Learned fusion stays on CUDA; token-derived row hashing and byte gathering may run on the host. [R12][R42]

The strict schedule may initially compute and retain all decoder-window state during full-reference prefill. Host-backed exact reconstruction is allowed only with sufficient dependency inputs and complete replay accounting. No minimal replay length is inferred solely from the sliding-window size; dependencies propagate through decoder layers and any other stateful operations. The report's bounded approximate replay cannot satisfy retained-state parity. [R27]

### Conversation re-encoding and live-prefix reuse

Conversation continuation must compare the **newly rendered native input**, not only the previous visible text. Reasoning-history removal, system/tool definitions, generation prefixes, image expansion, and message normalization can change tokens earlier than the appended user turn. Reuse the current state only when its complete incorporated token prefix, image/content identities, position assignments, encoding/profile/schedule identity and logical history exactly match the new input prefix. Include the sampled-but-not-yet-incorporated token boundary from Section 16.

The initial engine supports exact append to that current prefix. It does not advertise arbitrary-prefix rollback merely because a longest common prefix can be found. If the new input diverges before the incorporated-state boundary, or its prior images/encoding settings differ, discard private state and run exact full re-prefill under a fresh binding/session epoch as specified in Section 14. Immutable weights/verified row caches may survive. A future prefix snapshot requires snapshots of **all** dependent state, not just KV, and is outside the baseline. This makes native history dropping correct without silently reusing stale mHC, compressor, Engram or window state.

Teacher-forced comparison must specify the exact encoded corpus, all compared positions and the shifted next-token labels, with padding/image/nonprediction positions masked consistently. Reference helpers returning only last-position logits need an instrumented evaluation path. Compare state and loss across chunk boundaries and re-encoding resets, not merely one generated answer.

### 9.4 Ties and sparse selection

Specify deterministic tie handling in test fixtures and record any difference from the reference implementation's unspecified ties. Preserve selected-score semantics, modality-aware router bias, newest/partial candidate blocks, causal position ordering, and window overlap handling. A lower-index tie rule is a deliberate numerical convention, not proof that every upstream `topk` uses it.

Paging must not change which entries are considered. A page missing from device memory is fetched or the operation fails; it is never removed from the candidate set to avoid a transfer. Candidate/index caching includes all dependencies on the current query and state generation.

### 9.5 Context limits

The maximum admitted context is derived from the current profile and measured allocation functions. Reserve for prompt plus requested output, continued-prefill requirements, image expansion, and state-update peaks. Give an explicit rejection with the limiting object/bytes when the request cannot fit.

Remote private-state backing, SSD prefix stores, and automatic spill to operating-system swap remain out of scope. This preserves the initial one-client ownership and failure model while using both local memory tiers fully.

## 10. Engram lookup and local row caching

### 10.1 Distinct operation, same immutable identity

Engram is not a routed expert. The server stores its authoritative packed table rows and scales. The client computes the source-defined token-history/row selection and performs learned fusion on CUDA. Byte gathering on a CPU is allowed in both server modes; it is not CPU expert evaluation.

Use a dedicated bounded row-lookup operation. Requests identify table, row IDs, expected representation, and row association. Replies contain exactly those packed rows/scales with explicit lengths. Do not expose arbitrary addresses or reimplement table lookup as an expert GEMM request.

Native token normalization, compressed vocabulary, hashing, special-token handling, n-gram padding/history, row dimensions, and layer-specific table identities must be preserved. Repeated requested rows may share a transferred payload while preserving their original associations. Validate overflow and signedness in hash/index arithmetic against the source fixtures. [R42]

### 10.2 Host and device row caches

Provide separately budgeted client-RAM and optional VRAM caches. A zero-cache budget is valid. Host hits stage rows locally; misses retrieve only the required rows and may populate the bounded cache. GPU fusion consumes the exact source representation or an explicitly validated prepared layout.

Known token history can permit row prefetch before the corresponding learned layer runs. This does not permit predicting unknown future expert routes. Newly sampled token IDs become eligible only under the correct commitment/history rules. Canceling a generation must not reuse a speculative n-gram history in another session.

vLLM's documented host-resident Engram path is a useful placement precedent. It is not a ready-made remote table protocol or permission to map the complete table onto a smaller client. [R43]

### 10.3 Table placement and integrity

Engram placement is independent of expert replication: shared, row-sharded, or explicitly replicated populations may be chosen. Do not multiply a large sparse table for every CPU worker unless its locality benefit is measured and each replica fits. The server's own table operation remains bounded regardless of request size.

No lower-bit community Engram table may replace the required artifact without a separately approved model/profile change. Quantizing tables to save capacity is a quality change, not cache eviction. A missing or corrupt row fails the request; it is never replaced with a zero row.

## 11. CPU-only expert server and NUMA

### 11.1 Production CPU executor

The CPU build must serve the target expert/profile inventory without a usable GPU, NVIDIA driver, CUDA initialization, or mandatory GPU-only Python/package dependency. It receives already-selected work and returns per-expert contributions. It never runs a second transformer or selects different experts.

Implement bounded packed decode/scaling, input quantization where required, expert projections, clamps, routing-weight placement, and output rounding. BF16, FP8, and NVFP4 storage support does not imply native CPU instructions for those dtypes. A valid CPU implementation may use software unpacking and FP32 arithmetic with explicitly modeled quantizer/rounding effects. Test it against the same-realization reference. Use a bounded number of coarse arenas per node, not one operating-system mapping per expert/tensor view. Record mapping count, allocation alignment, page policy and allocator fragmentation in admission; sufficient byte capacity does not waive process mapping or allocation limits. Do not change host-wide huge-page or mapping limits silently.

The actual CPU ISA is a qualification input. DDR3 does not imply AVX2, AVX512, AMX, or native BF16. Compile separate server/client targets; do not copy a workstation-native binary onto an older server. Runtime dispatch among tested CPU ISA kernels is permitted inside the CPU backend, but unsupported instructions must never execute before capability checks.

The DwarfStar diagnostic CPU path can establish semantics but is not a sufficient performance implementation by declaration. FreeToken and KT-Kernel are useful CPU-kernel leads with their own format/ISA restrictions. Native NVIDIA activation quantization and this model's clamp behavior still require integration. [R01][R06][R23][R35]

### 11.2 Worker structure

Use one bounded pool per selected NUMA node, with node-local scratch and controlled affinity. One expert-service process dispatches jobs; do not load a complete independent model engine per worker or fork after initializing CUDA elsewhere.

For a row/layer, different selected experts may be assigned to different nodes. Replicas provide alternative local sources, not duplicate computation. The initial whole-expert scheduler creates at most the number of remaining selected routed experts in routed-expert jobs per row; an explicitly remote shared branch is separate. When the client executes some of the six target experts, fewer jobs remain. Additional nodes need independent rows or a separately validated intra-expert partition to contribute useful parallelism.

Avoid nested BLAS/thread-pool oversubscription. Reserve service/network progress capacity rather than allowing all CPUs to block behind compute pools. Cost balancing uses measured expert/profile/row-count latency and queue load, not the assumption that all jobs cost the same.

### 11.3 Required placement policies

| Policy | Physical layout | Expected use |
|---|---|---|
| `sharded` | One logical bank, each expert owned by a node; other tensor classes explicitly placed | Capacity-first baseline; execute near the owning weights. |
| `replicated_experts` | Complete packed routed-expert bank on every participating replica node | Flexible assignment of selected experts using local weight reads. |
| `replicated_server_model` | Complete declared server-required tensor population on each participating node | Maximum local scheduling freedom when capacity permits. |

A literal full-checkpoint replica may be benchmarked when requested and affordable, but does not require loading unused draft/vision/client-only tensors into every evaluator. Count canonical source copies, active runtime copies, and replicated classes separately. The server model's complete logical identity must not depend on worker placement.

Replication is not necessary to exploit several nodes' bandwidth. A sharded layer can already run concurrently when selected experts reside on different nodes. Replication principally removes ownership restrictions and can reduce imbalance. It does not reduce arithmetic or multiply the useful work in one dependent layer.

### 11.4 Capacity and physical locality

Let `E` be one packed expert bank, `R` its physical replica count, `G` separately placed table bytes, `O` other nonreplicated active bytes, and `W_i` worker scratch/reserve:

```text
expert-replicated RAM = R*E + G + O + sum(W_i) + transport/host reserve
fully replicated server RAM = R*S + sum(W_i) + transport/host reserve
per-node requirement = local shard/replica + local scratch + local reserve
```

Here `S` is the explicitly declared server-required population, and `R` counts all retained physical copies of the replicated population, including the original when it is one of those copies. Canonical identity is logical; it does not require an additional uncounted bank. Every node must fit its local allocation; enough aggregate chassis memory is not enough. A supposedly local replica spilling onto another node does not pass replicated-local admission.

Allocate node-bound arenas or another verifiably equivalent arrangement, initialize on their owning nodes, and inspect physical placement. Multiple mappings of one file can share physical pages; different CPU affinities do not create local weight replicas. NUMA domain count also need not equal socket count or independent full-bandwidth memory channels. Verify real topology and memory-controller use. [R29]

Keep intentional replicas immutable and identically hashed. Prevent deduplication/migration from invalidating the selected policy through scoped configuration and verification, not hidden global OS changes. Allocation policy must not silently relax on failure. Account for page-cache, first-touch, copy, alignment, and huge-page behavior at startup and under load.

### 11.5 Why the 512 GB concept needs admission

NVIDIA reports approximately **492 GiB** for its NVFP4 artifact. That is about **528.3 decimal GB**, before runtime overhead. Even 512 GiB leaves only about 20 GiB before reserve if the full artifact is retained at that size. Four such complete copies are about 1.92 TiB; eight are about 3.84 TiB. These are artifact-based arithmetic illustrations, not measured runtime footprints or expert-bank-only sizes. [R14]

Use the actual active tensor inventory to size each configuration. FP8/BF16 derived weights and NUMA replicas can require substantially more RAM. Preparation may identify legitimate inactive tensors, but it must not prune required experts or quantize Engram differently merely to force the original server to pass.

### 11.6 CPU measurements

Measure packed expert throughput, scaling/unpacking, DRAM bytes, local versus remote-node access, per-node queues, synchronization, and end-to-end layer completion. A benchmark repeatedly evaluating one expert that remains in CPU cache is not evidence for a multi-terabyte bank. Include sufficiently broad expert traces and real prefill row groups.

Sum memory-channel bandwidth only when the workload actually uses independent local channels concurrently. If a scalar reference kernel is compute-bound, additional RAM channels will not automatically help. Record instruction support, clocks, power limits, DIMM channels/population, and actual memory rates instead of attributing speed to DDR generation alone.

## 12. Blackwell CUDA expert server

### 12.1 Expert-only initialization

The server's CUDA mode initializes expert kernels, input/output slots, bounded transfer pools, mandatory workspace, and an optional expert cache. It must not instantiate attention, client KV/index state, a second router, or an unnecessary vocabulary head to reuse a full-model code path.

Its host CPU handles service logic, authorization, memory placement, gathering/copies, and launches. Expert projection, nonlinear activation, quantization, and assigned reduction mathematics remain CUDA. Failure to pin a whole bank is not a reason to invoke CPU experts; no whole-bank pin is required.

### 12.2 Cache miss path

For one admitted subgroup:

```text
validate selected work and obtain model/profile handles
upload each distinct input row and required quantizer metadata once
resolve complete cache hits and allocate transient execution slots
stage missing packed weights/scales from server RAM to those slots
wait for their copy events, then execute qualified expert operations
retain or discard cache entries according to policy and leases
copy complete tagged contributions into bounded response buffers
publish the response only after successful completion
```

Weights move over the **server's local PCIe connection**. The client sees only the remote operation's inputs/results. Bounded expert subgroups may execute sequentially when all selected experts do not fit simultaneously. A single oversized projection needs a qualified tiled kernel; it cannot be made executable by allocating fake sparse storage.

### 12.3 Format and kernel requirements

Qualify NVFP4 W4A4, source-compatible FP8 W8A8, and BF16 for the exact server SM and target shapes. Validate source clamps, quantizer scopes, accumulator behavior, and output casts. A fast generic fused MoE API that omits clamping or changes scale semantics is not a usable kernel for the strict profile.

Use vector/matrix paths appropriate to decode and grouped prefill. Tensor Core support is not a guarantee that a large-batch kernel performs well for one row. Keep baseline correctness independent of graph capture or a fixed full-layer launch shape.

### 12.4 Host staging and topology

Place reusable pinned pools and their service threads near the GPU root complex where practical. A multi-terabyte bank can span other nodes; count inter-socket reads and staging copies. Do not require replicating the entire bank onto the GPU-local node.

Measure actual negotiated PCIe generation/width and effective transfers with representative expert sizes, simultaneous NIC traffic, and host-memory contention. Approximately 12 GB/s is NVIDIA's example for pinned Gen3 x16 transfers, not a universal Blackwell limit. A faster arithmetic kernel does not remove required PCIe bytes. [R32]

The server cache should be tuned to work left after **both** client tiers, not the original model's global expert frequencies. A server cache hit saves local H2D weight bytes but does not eliminate its network round trip. A miss remains a valid CUDA operation, not a CPU fallback trigger.

## 13. Cache planning and transfer scheduling

### 13.1 Optimize layer completion

The planner optimizes time to first token and inter-token latency under capacity limits. It must not use allocated bytes, GPU utilization, or expert-hit percentage alone as the objective.

For a host-cached client expert, compare local gathering/staging, client H2D, CUDA work, and contention against remote queue/execution/reply cost. Another remote expert may already retain the layer's network dependency. Local CUDA work also competes with client attention/shared work and state transfers. Choose the whole-layer plan, not isolated per-expert arithmetic speed.

Calibrate per profile, server mode, CPU/NUMA arrangement, client memory budget, and prefill/decode phase. Keep the initial protected cache membership and preferred host-hit execution route reproducible within each phase. Safe optional-cache reclamation follows the admitted state-growth plan; it does not become arbitrary unrecorded online adaptation.

### 13.2 Distinct locality benefits

| Event | Benefit | Remaining cost |
|---|---|---|
| Client VRAM expert hit | Avoids remote evaluation and client weight transfer | Client expert compute and contention |
| Client RAM expert hit selected locally | Avoids remote evaluation | Client H2D/staging and CUDA compute |
| Server GPU expert hit | Avoids server H2D weight transfer | Network, server compute, result copy |
| Server CPU NUMA-local assignment | Avoids cross-node weight reads | CPU expert arithmetic and local DRAM reads |
| Client Engram host/VRAM hit | Avoids server row lookup | Local staging when needed and CUDA fusion |
| Client state/weight VRAM hit | Avoids client H2D and possible gather | Device operation |

Record byte-weighted misses, fully local layers, actual copy volume, and phase-specific reuse. One required remote expert or remote shared branch preserves that layer's remote completion dependency even when most experts hit locally.

### 13.3 Admission and cache fill policy

Fill selected immutable client weights/experts during preparation. Count their network transfer and startup cost, and measure how much reuse amortizes it. Server CUDA caches may use a simple deterministic replacement policy with protected entries and transient execution slots; preserve ongoing kernel leases.

Engram row demand misses may populate a bounded row cache because the operation is already row retrieval. Expert demand misses may not be disguised as weight retrieval. Later cache promotion can be added at safe boundaries with an explicit traffic budget, but must not block demand replies or change the server's fixed execution mode.

Use inclusive host backing where beneficial but do not require every device entry to consume an extra client-RAM copy when the budget favors server-backed preparation. Count actual duplicated layouts and scales. Immutable data needs no distributed write coherence; identity and lease validity are sufficient.

### 13.4 Scheduling priority and starvation control

Prioritize demand computation, required page/weight reads, and dirty writeback needed to free execution slots. Then schedule known next-phase prefetch, and only then optional cache fill. Bound each queue and the total bytes reserved by it. Bulk transfers must not consume all pinned slots or socket buffers needed for a reply.

A lease must reserve all resources needed to finish its admitted subgroup; do not create a cycle where a dirty page waits for a pinned buffer held by a prefetch waiting for the dirty page's device slot. Admission reserves progress buffers that optional caches cannot consume. Cancellation releases them only after DMA and readers are quiescent.

### 13.5 Limits of prediction

Next-layer fixed weights can often be known, but expert identities depend on intermediate activations. Engram row IDs may be known from committed token history; future generated token IDs are not. Do not label speculative route predictions as exact scheduling. Any future speculative work must not affect correctness or consume reserved demand capacity.

## 14. Internal interfaces and remote protocol

### 14.1 Internal boundaries

Use narrow interfaces around concrete work, not a generic distributed graph engine. Suggested internal contracts are:

| Interface | Required behavior |
|---|---|
| `TensorStore` | Resolve immutable verified tensor/range handles, representation, and placement without requiring local bytes for every tensor. |
| `ResidencyManager` | Admit and lease bounded host/device objects; track private-state generations and valid copies. |
| `ExpertExecutor` | Evaluate selected row/expert work under one numerical contract; explicit CPU or CUDA implementation on the server. |
| `ClientExpertDispatcher` | Partition original selected work across client CUDA and remote evaluation according to the admitted plan. |
| `EngramStore` | Resolve exact packed table rows and scales under authorized table handles. |
| `RemoteTransport` | Authenticate, bind identity/limits, exchange bounded requests/results, report errors and cancellation. |

These names describe responsibilities, not a mandatory hierarchy of classes. Simple structs, functions, and a small explicit executor selection are preferred. Do not add generic providers, compatibility layers, or a distributed cache-coherence subsystem.

An expert descriptor identifies its projections/scales and their operation contract. Inputs include row association, logical dtype, exact selected expert and slot, coefficient, and applicable quantizer metadata. Outputs include the same association and complete contribution. The descriptor cannot contain an untrusted function pointer, arbitrary kernel name, or client-specified code.

### 14.2 Transport baseline

Use persistent TCP connections with an explicit deployment transport policy. Mutual TLS is the default: TLS 1.3 from a maintained standard library with mutual certificate authentication. For an operator-approved trusted network, `tls: {"enabled": false}` instead selects ordinary TCP with no certificates, encryption or cryptographic peer authentication. The disabled object accepts no credential/name/fingerprint fields; both roles and channels must select the same mode. Missing credentials or a TLS failure never select plaintext automatically. In TLS mode disable early data, verify the configured peer DNS/IP against its certificate SAN, and authorize the client identity explicitly; a certificate signed by the CA alone is not sufficient authorization for every model. In trusted-network mode retain exact model/operator/encoding/layout/placement grants with `expected_peer_name: null`; these grants apply to the trusted network and do not represent certificate authentication. One latency-sensitive connection handles control, expert work, Engram lookup, and bounded replies. A separate bounded bulk connection handles bootstrap/client-cache tensor ranges. Both bind to the same model session and selected transport policy. TLS binds the same authenticated leaf identity; trusted-network TCP requires matching numeric peer IPs (ignoring ephemeral ports, retaining IPv6 scope) and the same fresh one-use bulk capability. IP pairing is a routing/session check, not authentication.

A production listener is restricted to the intended network/peer. Default internal ports are TCP 7443 for control/expert/Engram traffic and TCP 7444 for bounded bulk transfers; deployment may map different host ports explicitly. There is no UDP discovery, multicast, or requirement for inbound connections to the attention client. Certificate-free trusted-network TCP is an explicit supported production policy; enable it only where the operator accepts absence of network confidentiality and cryptographic peer identity. HTTP API bearer authentication remains independent and required. Loopback tests may use test credentials. Configure short-packet behavior deliberately and measure application latency rather than equating ping with expert RPC time.

RDMA/GPUDirect is not required for correctness or initial deployment. The baseline stages network data through host buffers; direct NIC/GPU integrations can be separate measured optimizations without changing the protocol's mathematical meaning.

### 14.3 Session binding

Before inference, exchange and verify:

- Protocol revision and required capabilities; reject incompatible revisions rather than provide compatibility shims.
- Common logical model, tokenizer/encoding, numerical-profile, and operator-contract digests; logical tensor IDs and content integrity.
- Each peer's approved prepared-layout manifest and mapping to the common logical tensors. Different CPU/CUDA byte layouts need not have equal physical digests; unapproved numerical changes are rejected.
- Server executor mode, CPU/SM kernel support, per-request limits, accepted tensor/table handles, and readiness.
- Session identifier, fresh generation epoch, allowed row/shape ranges, context/phase plan identity, and peer authorization.
- Hard limits for frame bytes, in-flight requests/bytes, bulk chunks, worker jobs, response slots, and timeout/cancellation behavior.

The model and execution mode are immutable for the binding. Repacking that changes kernel interpretation, changing a precision realization, or reloading weights requires a new binding. Do not send a giant manifest on every token; bind checked handles once.

### 14.4 Required operations

| Operation | Request | Response |
|---|---|---|
| Bind | Identity/capability/limit agreement | Bound model and authorized handles or a typed error |
| Evaluate experts | Layer/operation handle, input rows, original selected slots/IDs/coefficients, quantizer context | Tagged complete per-expert contributions |
| Lookup Engram rows | Table handles and exact row associations | Packed requested rows/scales with their associations |
| Read immutable tensor ranges | Authorized handles/ranges for preparation or controlled cache admission | Verified bounded chunks; not arbitrary filesystem reads |
| Evaluate declared shared operation | Only an explicitly configured supported non-routed expert operation | Its declared contribution; no arbitrary graph execution |
| Cancel | Session epoch and request ID | Cancellation state; resources released after safe quiescence |
| Health/metrics and close | Bound authorized control request | Readiness/counters or orderly unbind |

There is no initial remote KV/state-write operation. Client host/device paging is local. Adding remote private-state backing later requires a separate ownership, admission, and failure design.

### 14.5 Wire framing

Define fixed-width fields and explicit byte order; do not serialize native structs, padding, pointers, `size_t`, or compiler enums. The initial exact header and payload registry below define magic, protocol revision, message kind, flags, payload length, request ID, bound session/epoch identity, and response status. Use checked 64-bit lengths/offsets/counters and reject overflow before allocation.

The baseline floating payload is IEEE-754 binary32 in a declared byte order. Other numeric fields have equally explicit signedness and size. Model quantized weights/rows retain their declared byte layout and scale metadata. Header/schema encoding may be compact binary; configuration/manifest JSON is not justification for JSON arrays of inference floats.

The initial maximum frame **payload** is **16 MiB** (excluding the fixed header and TLS record overhead), with **4 MiB** maximum bulk data chunks and a configurable expert microbatch no larger than **64 input rows** for the inspected target dimensions. These are proposed safety defaults, not performance guarantees. The actual admitted byte limit also accounts for coefficients, metadata, logical dtype information, and result expansion. Lower limits must work; a larger future negotiated limit remains bounded and requires tests.

Limit in-flight expert requests to a small fixed window, initially no more than two admitted subrequests, unless the documented prefill plan needs another tested bound. Reserve response memory before accepting its work. Chunk large prefill/row requests at legal independent-row boundaries, not across an unrecorded quantizer reduction domain.

### 14.6 Expert request shape and association

For each request, transmit the normalized source input rows once, not once per selected expert. A compact row map identifies variable numbers of remote selected slots per row. Each entry carries the original slot, global expert ID, coefficient, and profile-required quantizer context. Reuse identical input values inside the server without recomputing an invalid common scale.

Validate layer ranges, row counts, dimensions, expert membership, unique selected slots, coefficient finiteness and expected constraints, profile identity, and total response size. Padding rows and masked tokens must not generate spurious expert work. Zero remote work is handled locally and does not require a meaningless evaluation request.

Responses are keyed by request/epoch and original row/slot. A server may compute subresults out of order. The baseline sends one complete bounded response per subrequest; it does not expose partially successful contributions for early model advancement. Collect and reduce using the canonical operation order. Never multiply routing coefficients twice or reapply a coefficient already incorporated before the down projection.

Aggregate-only replies can reduce traffic, but are not the baseline. Such an optimization needs a separate reduction contract and same-realization numerical validation; it cannot silently regroup partial sums across placements.

### 14.7 Progress, backpressure, and cancellation

All queues have byte and job bounds. The server declines a request that cannot reserve its mandatory resources; it must not accept unbounded work and hope allocations succeed later. The client stops issuing additional subrequests when credits are exhausted while continuing to receive replies.

Cancellation does not promise immediate kernel preemption. Mark the work abandoned, prevent result application, then drain or terminate safely according to executor capability. Do not return buffers to the allocator while a kernel, copy engine, socket sender, or NUMA worker still uses them.

A timed-out request fails the generation. The initial runtime does not automatically retry an uncertain expert operation or race it against a second backend. Protocol cancellation targets control-channel inference/row work; bulk-fill cancellation closes or explicitly abandons that bulk transfer without publishing a partial cache entry. Request IDs on the separate connection are not confused with control-channel IDs. Request identifiers prevent stale/duplicate application; they do not by themselves provide transactional replay of client state.

### 14.8 Exact initial framing and schema registry

This subsection fixes the initial protocol rather than leaving the transport implementer to invent incompatible structs. Protocol revision `1` is a wire identity, not a request to maintain a compatibility layer. Reject other revisions until deliberately specified. Use little-endian unsigned integers and IEEE-754 binary32 for the baseline float payload. Every arithmetic size calculation is checked before allocation.

**Fixed header: 64 bytes.** Offsets are from the first application byte; TLS framing, when selected, is external.

| Offset | Bytes | Field | Contract |
|---:|---:|---|---|
| 0 | 4 | Magic | ASCII `DSER` |
| 4 | 2 | Protocol revision | `1` |
| 6 | 2 | Message kind | Registry below |
| 8 | 4 | Flags | Bit 0 means response; all other bits zero |
| 12 | 4 | Status | Zero in requests/success; typed nonzero error in responses |
| 16 | 8 | Payload length | Excludes this header; maximum 16,777,216 bytes |
| 24 | 8 | Request ID | Requests use nonzero IDs increasing in send order per connection; responses echo their request ID and may arrive out of order |
| 32 | 16 | Session ID | Cryptographically random server-issued bytes; zero only before initial binding |
| 48 | 8 | Epoch | Server-issued generation epoch; zero before initial binding |
| 56 | 8 | Reserved | Must be zero |

The parser consumes exactly the declared bytes and handles fragmented/coalesced socket reads. It never casts the byte buffer to a native struct. Requests and responses use the same kind and request ID. A response carrying another session/epoch is rejected before its values reach model state, except for the explicitly specified initial Bind transition below. The server checks the selected transport policy and exact provisioned model/layout grant before admitting handles or expensive operations; TLS additionally authenticates the peer. Bounds also apply to control JSON, connection admission, and, in TLS mode, certificates and handshake concurrency. Mutual TLS and disabled early data follow the authenticated policy; trusted-network TCP omits only TLS. Both modes retain application-level model/session identity, grant checks, payload validation, credits, cancellation, deadlines and failure containment. TLS integrity does not replace those checks. [R76]

| Kind | Operation | Payload form |
|---:|---|---|
| 1 | Bind control connection | Canonical bounded UTF-8 JSON, at most 262,144 bytes |
| 2 | Bind bulk connection | Same session/epoch and authorized peer, with a fresh bind capability |
| 10 | Evaluate selected experts | Fixed descriptor plus binary arrays below |
| 11 | Lookup Engram rows | Binary table/row descriptor and packed data |
| 12 | Read preparation chunk | Binary authorized shard/chunk identity and bytes |
| 13 | Evaluate declared shared branch | The expert descriptor form with its explicitly bound non-routed operation handle |
| 20 | Cancel | Target request ID and epoch; no arbitrary state mutation |
| 21 | Health/metrics | Small bounded control JSON; no activations or secret data |
| 22 | Close binding | Empty request; acknowledgement after safe drainage |

Use status codes `0=OK`, `1=INVALID_REQUEST`, `2=UNAUTHORIZED`, `3=IDENTITY_MISMATCH`, `4=UNSUPPORTED`, `5=RESOURCE_LIMIT`, `6=NOT_READY`, `7=DEADLINE_EXCEEDED`, `8=CANCELLED`, `9=INTEGRITY_ERROR`, `10=EXECUTOR_ERROR`, and `11=INTERNAL_ERROR`. Error JSON is at most 16 KiB, contains a stable code and safe diagnostic, and does not expose paths, tensor contents, credentials, or an internal stack trace. A protocol corruption closes the connection; an executor/state-affecting error invalidates its generation. There is no automatic fallback on an error code.

The bind response confirms checked 64-bit handles already assigned by the verified manifest and returns compact permissions, supported dimensions and limits. It never embeds the complete tensor inventory or chunk-hash index in the 256 KiB control message. Provision the compact client metadata and authenticated hash index during preparation; larger metadata objects use bounded verified chunks. All handle-to-logical-object mappings are immutable for the binding, collision-checked and authorized by class/range. It returns the session and a fresh one-use bulk-bind capability. The bulk channel must use the same authorized certificate identity in TLS mode, or matching numeric peer IP in trusted-network mode, bind the same logical/profile contract, and present that capability; accepting only a guessed session ID is insufficient. Hold at most one control and one bulk channel for the initially admitted client. No compression or arbitrary extension blob is negotiated in the baseline. A reset/reload after invalidation requires a fresh binding/session epoch; the process may retain immutable model banks, but cannot retain authorization to apply old responses to new mutable state.

#### Expert payload layout

Start with a 40-byte descriptor: `operation_handle:u64`, `invocation_id:u64`, `row_count:u32`, `input_width:u32`, `output_width:u32`, `entry_count:u32`, `quantizer_context_bytes:u32`, and `reserved:u32=0`. These dimensions must agree with the bound operation; they are not authority to allocate arbitrary tensor shapes.

The descriptor is followed, without native padding, by:

1. `row_count` row identifiers as `u64`, associating rows with positions inside the logical invocation.
2. A CSR-style array of `row_count + 1` `u32` entry offsets, beginning at zero, monotone, and ending at `entry_count`.
3. `entry_count` eight-byte entries: `expert_id:u16`, `original_slot:u16`, and `routing_coefficient:f32`. The fixed supported model fits these ranges; a future larger model needs an explicit protocol change, not truncation.
4. `row_count * input_width` row-major `f32` input values. Where the logical input is BF16, these encode exactly the BF16-rounded values widened to FP32; they must not substitute earlier higher-precision intermediates.
5. The declared quantizer context, with a fixed binary layout and exact byte length named by the bound operator contract. The initial per-row/per-expert realizations use zero context bytes when the full-row inputs and manifest-frozen calibration determine every scale; an operation needing an external shared statistic must declare the complete typed layout, producer/reduction domain and an independent fixture before it is advertised as supported. Arbitrary opaque blobs or implicit context-schema negotiation are not supported. All local and remote consumers use the same statistic; recomputing it on a subset is forbidden.

For kind 13, the bound shared-operation descriptor requires one entry per row, slot `65535`, expert ID zero, and coefficient `1.0`; it never applies routed scaling to the shared branch. Other uses of that reserved slot are invalid. The caller specifies each shared evaluation once and performs the final combination at the client.

A successful expert/shared response has a 32-byte prefix: `operation_handle:u64`, `invocation_id:u64`, `row_count:u32`, `output_width:u32`, `entry_count:u32`, `reserved:u32=0`. It then returns entries in request-entry order as `entry_count * output_width` `f32` values. It carries no independently chosen expert IDs. The outstanding request retains the authoritative row/slot association. The client checks complete length and membership before application. Down-projection outputs that the reference rounds to BF16 are rounded at that boundary before wire widening; the wire does not increase their effective precision. The client's final combination still uses canonical expert-ID order, not request order.

Microbatch splitting occurs before these messages are created. Requests may contain zero entries for a row but must contain at least one total entry. Remote coefficients are the original coefficients for that subset; never renormalize them or require a partial subset to sum to the full routed scale. Check individual finiteness/range according to the model, while the client validates full-selection normalization against the reference. The implementation must preserve quantizer reduction domains across those splits. No response may successfully omit a failed entry. Credits reserve the worst-case request plus response, intermediate results, and executor workspace; the 16 MiB frame allowance is not the total memory allowance.

#### Row and chunk payloads

An Engram request starts with `table_handle:u64`, `row_count:u32`, `reserved:u32=0`, then `row_count` pairs of `row_id:u64` and `association:u64`. Its response starts with `table_handle:u64`, `row_count:u32`, `packed_row_stride:u32`, and `representation_handle:u64`, followed by the same row-ID/association pairs and then their row-major packed bytes. The returned stride and representation must match the bound descriptor. The baseline may return duplicate rows directly; deduplication must preserve a deterministic row map if later implemented. Table metadata determines width and encoding, not client-provided strides.

A preparation request is `shard_handle:u64` and `chunk_index:u64`. The response prefix is exactly 64 bytes: `shard_handle:u64`, `chunk_index:u64`, `logical_offset:u64`, `data_length:u32`, `reserved:u32=0`, and 32 SHA-256 bytes; the data follows. The expected hash comes from the previously verified manifest, not merely from the response. Maximum chunk data is 4,194,304 bytes; descriptor overhead remains within the frame ceiling. Preparation privileges do not authorize arbitrary files or a demand expert-miss path.

Define machine-readable `protocol/schema.json` and byte-level fixtures for every message, both byte orders in parser-negative tests, truncated headers/payloads, reserved-bit failures, multiplication overflow, unknown kinds, bad quantizer contexts, and invalid shared slots. Tests must verify the 64-byte header with explicit offsets, not the compiler's `sizeof` a packed C struct.


### 14.9 Binding, readiness and connection lifecycle

The following rules are part of protocol revision 1, not implementation-defined conventions.

**Initial binding:** after selected-mode peer/policy checks and model-grant authorization, the first control request is kind 1 with zero session/epoch, request ID 1 and response flag clear. A successful reply is the **only** transition that installs nonzero server-generated session bytes and epoch; its request ID still matches 1. Its JSON echoes the requested logical identities and supplies the agreed limits, server mode and physical layout. A failed initial bind returns zero session/epoch and a typed error, then closes. A normal response must never silently install a new session.

**Identity field schema:** the kind-1 request contains `role="client"`, `logical_model_digest`, `operator_contract_digest`, `encoding_digest`, `client_layout_digest`, `placement_plan_digest`, `profile` (`nvfp4|fp8|bf16`), `server_executor` (`cpu|cuda`), and `limits`. Digests have the Section 6 SHA-256 text form. `limits` contains positive safe-integer `frame_payload_bytes`, `bulk_data_bytes`, `expert_rows`, `expert_requests`, `row_lookup_rows`, `inflight_payload_bytes`, `operation_timeout_ms`, `frame_io_timeout_ms`, and `write_timeout_ms`. Required maxima cannot exceed this document's ceilings. The server applies the smaller acceptable peer limit and rejects a limit below its minimum viable unit rather than enlarging it silently. The client retains its stricter deadline when a peer advertises a longer one; neither endpoint silently increases its configured resource ceilings. The fixed non-expert outstanding limits specified below also apply.

The successful bind JSON echoes those fields, adds `server_layout_digest`, `session_id` (32 lowercase hexadecimal characters), `epoch` (u64 decimal string), and `bulk_capability` (64 lowercase hexadecimal characters representing 32 fresh random bytes). JSON session/epoch must equal the header. Physical manifests are matched through the logical mapping, not equal CPU/GPU layout hashes. The compact permission grant names already provisioned handle sets by digest; do not put every tensor in this control object. Bind unknown fields, malformed identities and duplicate keys are rejected. This compact schema must be emitted as strict `protocol/schema.json` definitions and fixtures.

**Bulk binding:** its first request is kind 2 with the established header session/epoch and a JSON object containing `session_id`, `epoch`, `logical_model_digest`, `operator_contract_digest`, and `bulk_capability`. TLS certificate identity must match the control peer; trusted-network TCP instead matches numeric peer IP. Both modes retain the fresh one-use capability. The server consumes the capability atomically and returns the same session/epoch with `{"bound":true}`. Replay or a second simultaneous bulk channel is rejected. The capability is never logged. Control and bulk request-ID spaces are independent; each increases only for requests, not for arrival order of responses.

**Readiness without a cycle:** expert readiness means local artifacts, executor, resource reservations and listener authorization are ready to accept Bind; it does not require a client already connected. Client readiness additionally requires its successful control binding, required bootstrap completion and its local memory/kernel plan. Being busy is not unhealthy. An idle binding can serve consecutive nonoverlapping admitted generations; invocation IDs are never reused within its epoch, and private client state is reset or continued by the declared history contract.

**Disconnect/rebind:** unexpected loss of the control or an established bulk connection invalidates the binding and any active generation. There is no automatic in-flight retransmission or capability reuse. Close/drain both channels, then perform a fresh control/bulk handshake before future work. Successfully verified immutable preparation chunks may be retained and reauthorized; private state and partially valid cache entries may not. The baseline deliberately avoids a separate bulk-reconnect protocol. A pre-readiness connect attempt may be retried within an explicit startup deadline because no inference was admitted; it is not a retry of uncertain neural work.

**Unknown input:** requests with an unknown kind, reserved flag/field, nonmonotone/reused request ID, unexpected response flag or invalid session close that connection without accepting work. Responses may complete out of request order but must match one live outstanding request. Completed request IDs cannot be applied twice. A bounded tombstone set handles the documented cancellation race below; it is not an unlimited replay log.

### 14.10 Deadlines, cancellation and credits

Use monotonic clocks for durations; do not compare absolute wall-clock timestamps between hosts. Each admitted request's end-to-end deadline includes queueing, copies, expert execution and complete reply receipt. The chosen `operation_timeout_ms` is fixed in the binding. The sender may abandon earlier; the server's local accepted-work timeout is a resource bound, not proof that cross-host elapsed time is identical. In TLS mode certificate validity uses the host's trusted wall clock separately.

Frame parsing has an explicit partial-frame deadline and rate/byte bound. Waiting for the first response byte is governed by the request deadline, not a short partial-header timeout that would kill a legitimately long CPU operation. Once a frame starts, its declared header/payload must finish within the admitted I/O deadline. Writes and TLS handshakes also have finite deadlines. Bound small operations separately: at most four outstanding lightweight control requests, two Engram lookup requests and one bulk chunk request per bound channel pair, in addition to the at-most-two expert requests. Reserve their progress slots independently of expert-work credits; each accepted payload and its maximum response still consume the negotiated total byte budget. Handle partial TLS reads/writes and WANT_READ/WANT_WRITE without busy-spinning or assuming one TLS record equals one application message.

A kind-20 cancel request has exactly 16 payload bytes: `target_request_id:u64`, `target_epoch:u64`. It targets kind 10, 11 or 13 on the **same control connection**. The caller must retain that target's association/resources while cancellation drains. A successful cancel reply is exactly 24 bytes: the same two u64 fields followed by `cancel_state:u32` and zero `reserved:u32`. States are `0=ACCEPTED_PENDING`, `1=ALREADY_TERMINAL`, `2=UNKNOWN_TARGET`. Cancellation acknowledgement is not a neural result and does not return the target's memory credit.

Every accepted target request has **one terminal reply**: a complete successful result or a typed terminal error such as CANCELLED. If execution completed before cancellation took effect, its successful terminal reply may win the race; the already-abandoned client generation must discard it, not apply it. ACCEPTED_PENDING is followed by the target's terminal error only after its buffers are safe to release. ALREADY_TERMINAL never causes a duplicate target reply. A cancel target that was never admitted returns UNKNOWN_TARGET without allocating work. Retain a bounded cache of the 16 most recent terminal inference/row request IDs until evicted in terminal-completion order; active targets are tracked independently. An evicted terminal target is UNKNOWN_TARGET and is never executed again. The cancel target ID must be smaller than its cancel request ID, and its epoch must match; do not wait for a future request or retain an unbounded cancellation history.

Expert-work credits are negotiated fixed counts and bytes: sending reserves one request slot plus its known worst-case payload/result/execution storage; receiving and validating its one terminal reply returns that credit. There is no additional unimplemented credit-update message. A request declined before admission returns a typed terminal error and releases the sender's reservation. No successful error response can excuse exceeding a bound before validation. Connection invalidation destroys its credits and requires fresh admission. A request cannot contain the same global expert twice for one row under different slots; validate both slot and expert uniqueness against the target's top-k contract.

Kind 21 accepts `{}` and returns a bounded non-sensitive JSON object with `state`, `ready`, and declared counters; diagnostics requiring expensive scans are forbidden. Kind 22 uses empty request/success payloads: mark the binding draining immediately and acknowledge only after earlier admitted work is terminal/quiescent. Cancelling or closing cannot preempt an arbitrary GPU kernel; a bounded process-level failure path handles an executor that cannot quiesce. No new inference is accepted while draining.

### 14.11 Exact lengths, row records and atomic result application

For an expert request with `R` rows, `E` entries, input width `D` and `Q` quantizer-context bytes, the exact payload length is `40 + 8*R + 4*(R+1) + 8*E + 4*R*D + Q`. Its response length is `32 + 4*E*output_width`. Validate these equations, each operation's maximum width/rows, CSR monotonicity, unique row IDs within the invocation subset, and quantizer-context schema **before** allocating or executing. Top-k for one row cannot exceed the manifest's six routed experts. Shared kind 13 has its independently declared one-entry rule.

For an Engram request the length is `16 + 16*R`; the reply length is `24 + 16*R + R*packed_row_stride`. A row record includes every needed value byte, scale and required deterministic padding, with offsets/dtypes defined by the bound representation. It cannot silently omit scales stored in a separate checkpoint tensor. Association values are opaque u64 IDs echoed exactly, not unchecked pointers. Validate the table row range and the derived reply length. `row_lookup_rows` is negotiated and must fit both frame and response-pool budgets. Duplicate rows may be returned separately without changing their associations.

Chunk requests are 16 bytes and replies are `64 + data_length`, with the declared final-chunk exception checked against the trusted shard length. Type-specific bounds apply before a generic 16 MiB payload allocation: bind JSON ≤256 KiB, error JSON ≤16 KiB, bulk data ≤4 MiB. Require exact payload consumption; trailing data is invalid. No codec compression is implicit.

Each local/remote contribution buffer becomes publishable only after complete successful device/CPU computation and validation. The layer merge consumes each expected contribution once in its reference order; partial network success never becomes a partially committed layer. Streaming the user-facing token occurs only after all model operations for that generation step have succeeded.

## 15. Admission, configuration, and operations

### 15.1 Required resolved configuration

Configuration is validated into one explicit plan before serving. Suggested field groups are below. They are implementation requirements, not currently available upstream flags.

| Group | Required settings |
|---|---|
| Model | Artifact/manifest path or bound identity, profile, tokenizer/encoding identity, declared features |
| Client device | Exact CUDA device, required SM support, VRAM cap, runtime reserve |
| Client host | Host-memory cap, OS headroom, pinned cap, state backing cap, weight/expert/row budgets |
| Client execution | Placement plan per phase, prefill microbatch, current/next execution slots, capture policy |
| Server | Exactly `cpu` or `cuda`, authoritative RAM cap, source/active tensor set, readiness requirements |
| CPU mode | Allowed NUMA nodes, placement policy, local replica/shard budgets, threads and scratch per node |
| CUDA mode | Exact GPU, device-memory cap, expert cache, transient slots, pinned staging, support matrix |
| Session | Maximum admitted input/output positions, image limits, seed/sampling controls, reasoning mode |
| Transport | Listen/connect addresses, TLS credentials/trust, peer authorization, frame/chunk/credit limits, deadlines |
| Profiling | Calibration identity, trace output policy, metrics interval, opt-in sensitive fixtures |

Resolve units to bytes; distinguish decimal GB and binary GiB. Missing essential budgets must produce an explicit dry-run planning result or an error, not an unsafe attempt to use all free memory. A plan produced from measurements is saved with its exact resolved values and hashes.

No client `cpu` neural-backend setting exists. No server `auto` or implicit hybrid mode exists. CPU ISA dispatch and client-local-versus-remote placement remain permitted inside their explicitly selected contracts.

### 15.2 Memory equations

For the client:

```text
host weights + host expert cache + host row cache + private host state
  + additional pinned staging + metadata/runtime <= client host budget

current device weights/state tiles + live activations + accumulators
  + required kernel workspace + copy slots + protected caches
  + graph/runtime allocation + reserve <= client VRAM budget
```

For the server:

```text
canonical active packed tensors + explicit alternate layouts
  + intentional replicas + worker scratch + network/staging
  + runtime/operating reserve <= usable server RAM

per NUMA node: local assigned bytes + local scratch/reserve <= local capacity

CUDA server: expert cache + executable group/tiles + quantizer intermediates
  + input/output/copy buffers + runtime reserve <= server VRAM budget
```

Pinned bytes already counted as a backing allocation are not charged again solely for registration. A separate staging copy is a separate allocation. Host and device copies occupy different budgets. Shared tensor aliases are counted once per actual physical copy; independent repacked layouts and replicas are counted separately.

### 15.3 Peak and progress admission

Admission uses the maximum simultaneous allocation during startup, prefill, decode, cache transitions, continuation, image processing, cancellation drainage, and expected state growth. It must include minimum progress buffers that caches cannot consume.

State growth is reserved for the declared maximum prompt plus generation and continuation plan, not merely the first token. Optional caches can occupy otherwise unused reservation only with a proven safe reclamation schedule. A full cache cannot prevent execution of a mandatory expert subgroup or dirty-state writeback.

The source model's maximum position count is a semantic ceiling; it is not the default allocation target. Set an honest admitted context and explain the limiting tensor/state/host/device budget when a request is too large. Do not allocate million-token structures merely because the checkpoint advertises that limit.

### 15.4 Physical resources and readiness

Probe GPU SM, VRAM, driver/toolkit, compute capability, negotiated PCIe generation/width, CPU model/ISA, NUMA topology, per-node usable RAM, DIMM population, NIC routing, process/container limits, and pinning limits. Record measurements without guessing unspecified hardware.

Startup establishes artifact integrity, kernel availability, model/profile agreement, actual memory placement, and required page residency before readiness. A bare loaded process is not ready. A GPU cache can be cold at readiness if mandatory streaming slots fit; cold-cache costs must be reported.

Use process/service-scoped controls. Do not globally disable swap, change NUMA balancing, disable security, or reserve all huge pages as an installation side effect. An administrator may explicitly provision resources, but the runtime checks the resulting state and fails clearly if its contract is unavailable.

### 15.5 Operator workflow to deliver

Provide native build targets and small tools for:

1. Inspecting hardware and checkpoint tensors without starting full inference.
2. Preparing/pinning artifacts and producing a bounded-memory plan.
3. Testing one target operator/expert and calibrating CPU, PCIe, network, and client placement.
4. Starting the CPU-only or CUDA expert service and confirming readiness.
5. Starting the CUDA client with the saved plan, using CLI prompts and the project's existing serving entry point where applicable.
6. Running regression, fidelity, capacity, and benchmark suites and exporting machine-readable reports.

Reuse existing user-facing DwarfStar entry points where practical. Do not invent a broad new serving API solely for this handoff. Native reasoning/tool encoding and separate declared image support must remain visible. Development scripts must check errors, avoid secret leakage, and produce actual exit codes rather than success-shaped logs after failures.

## 16. Failure, cancellation, and security

### 16.1 Fail closed on incomplete inference

A server disconnect/restart, timeout, invalid frame, wrong model/profile, missing contribution, corrupt row, worker failure, kernel error, memory allocation failure, or invalid state writeback aborts the affected generation. No output token may be computed from zero-filled missing work, fewer experts, stale pages, or a different precision/backend.

The client invalidates uncertain state for that generation and waits for outstanding readers/DMA before reclaiming resources. A canceled copy cannot publish a valid host page. A server CUDA cache entry is not valid until its weight/scale upload completes successfully. CPU replicas must remain immutable and hash-consistent.

### 16.2 Token and state commitment

Track separately the committed input prefix, the positions already incorporated into model state, the next sampled token, emitted output, and sampling/RNG state. The newest sampled token is not necessarily already represented in KV/Engram state. This distinction must be explicit in recovery and continuation tests.

Initial recovery uses a fresh session and deterministic replay of the committed token/image history, not reuse of uncertain partially advanced state. Do not regenerate and silently replace already emitted tokens. Durable resume is not a required initial feature; in-memory host state is capacity, not persistence.

Token streaming waits for a complete valid generation step. Text/tool parsing may buffer incomplete UTF-8 or structured output, but it must not label partial data as a completed tool invocation. Model reload requires a fresh binding and no reuse of old state/cache identities.

### 16.3 Mode guarantees

A CPU-only server must be tested without a working CUDA installation. A CUDA-only server must be instrumented to show zero CPU expert math during cache misses, prefill, low-memory handling, and errors. The client must show zero CPU neural operations while staging weights, reading host-backed state, and processing host expert hits in either server mode.

Host cryptography, parsing, tokenization, table gathering, and memory copies are allowed. They still consume CPU time and belong in profiling. Their presence is not evidence of forbidden expert kernels; their absence is not the requirement.

### 16.4 Security and data handling

Use standard authenticated encryption, least-privilege service accounts, read-only prepared artifacts, explicit peer authorization, finite queues, bounded parsing, and restricted listeners. No raw address, arbitrary path, executable graph, pickle, or user-supplied kernel is accepted remotely.

Treat activations, token history, images, Engram accesses, and state as potentially sensitive. Logs and traces omit raw contents by default. Golden fixtures containing user data require explicit opt-in and controlled storage; prefer redistributable synthetic/public fixtures. Certificates and secrets stay out of release artifacts and reports.

GPU/host pools cannot leak prior-session data through uninitialized padding or oversized replies. Initialize valid payload ranges and any transmitted padding. Revoked sessions cannot access retained private objects. Immutable model caches may remain only under their authorized identity.

### Crash confidentiality and fatal executor errors

Set the service's core-dump resource limit to zero and set `PR_SET_DUMPABLE=0` after final UID/credential setup and before loading secrets, model state or requests; verify it in the actual release container. Mark large sensitive arenas `MADV_DONTDUMP` where supported as defense in depth. A zero `RLIMIT_CORE` alone is insufficient on hosts that pipe dumps to a collector. This is service-scoped, not a change to the host's global core policy. Never upload a core dump containing credentials, prompts, state or model banks as an ordinary diagnostic artifact. [R82][R83]

A sticky CUDA error, illegal memory access, lost device, or an executor that cannot finish within the tested drain deadline takes the service out of readiness and terminates the affected process nonzero. Do not spin forever waiting on a failed event, recycle its device/pinned buffers into a new session, or invoke another backend. Best-effort safe drainage is bounded; process/context teardown, not reuse of suspect allocations, is the terminal path. Do not reset a host GPU that may serve other processes. If the OS or driver cannot complete teardown, leave the workload unavailable and require operator recovery; do not claim clean resource release or restart blindly onto the suspect device. CPU executor corruption is likewise not recovered by continuing the same process with partially updated state.

Initialize transmitted/pool padding and invalidate private generations before safe pool reuse. Readiness cannot be restored by catching an exception while a background DMA or worker can still write old addresses. A full report volume must produce a bounded visible diagnostic and stop new required-audit admission according to the configured policy; it must not cause unbounded log buffering, overwrite model data, or silently fabricate a successful report.

### 16.5 Resource and protocol faults

Test integer overflow, negative/out-of-range IDs, repeated selected slots, NaN/infinite coefficients, truncated frames, oversized declarations, unsupported dtypes/layouts, wrong digests, stale epochs, duplicate replies, stalled readers, and connection loss in every transfer phase. A parser failure must not allocate from attacker-controlled sizes before validation.

Do not assume mathematical expert statelessness makes retries safe. Client state and local work may already have advanced. The initial failure behavior is a clear error and safe invalidation, not an unimplemented exactly-once distributed transaction service.

## 17. Performance model and measurement

### 17.1 Use the actual CED schedule

For each invoked layer/phase, a planning model is:

```text
client attention/routing/state work
  + max(client-local expert branch, remote expert branch)
  + canonical merge/post-operation
  + unhidden required transfers
```

The formula describes dependencies, not guaranteed overlap. Account for shared GPU engines, host bandwidth, copy engines, sockets, NICs, NUMA links, and queues. Sum over the real executed schedule; do not assume a headline layer count multiplied by every token or claim all layers run concurrently.

A server GPU may save CPU arithmetic but wait for PCIe. CPU serving may avoid PCIe but be limited by unpacking, arithmetic, or local/remote DRAM reads. The client may reduce server work yet delay attention by using its GPU for too many host-cached experts. Benchmark these cases rather than declaring one resource universally faster.

### 17.2 Bandwidth accounting

For CUDA expert serving:

```text
server weight-transfer work >= actual server-cache miss bytes / measured H2D bandwidth
```

For CPU serving, bound each worker's local/remote bytes and arithmetic separately. Layer completion is constrained by the slowest assigned worker plus necessary synchronization; summing nominal socket bandwidth is not a timing result.

For the client, measure H2D and D2H for attention weights, shared/head weights, expert cache misses, state pages, Engram rows, metadata, and network input/output staging separately. Model dual-direction contention rather than adding full-duplex link rates and calling the sum one-way bandwidth.

For one row with FP32 transport, input width `d_in`, output width `d_out`, and `k_remote` requested expert contributions:

```text
expert payload ~= 4*d_in + 4*k_remote*d_out + metadata/quantizer context
```

At the inspected 5,120-wide expert input/output and six remote experts, the unframed payload is approximately **140 KiB per row per invoked MoE layer**. This is arithmetic for the per-expert reply contract, not a complete per-token or network benchmark. Add shared-operation replies if remote, Engram misses, framing, and TLS; count bootstrap separately. Client host paging is local PCIe traffic, not inter-host state traffic.

### 17.3 Measured calibration inputs

Record real expert-sized reads/copies, not only large contiguous bandwidth tests. Calibration includes:

| Area | Measurements |
|---|---|
| Client | RAM copy/gather, H2D/D2H by size and direction, simultaneous copies, expert and attention kernels, state-page access, copy-engine/GPU contention |
| CPU server | Per-ISA quantizer/unpack/GEMV/GEMM, per-node bandwidth, remote-node penalties, job distribution and synchronization |
| CUDA server | Cold and warm cache expert latency, staging copies, H2D by expert/tile size, subgroup/prefill kernels, cache reuse |
| Network | Actual authenticated application RTT, request/reply size curves, concurrent Engram/bulk traffic, CPU TLS/framing overhead |
| End to end | Startup, cache preparation, time to first token, prefill rate, median/p95/p99 inter-token latency, committed-token rate |

Record device/link topology and power/thermal state. A Gen5-capable 5090 in a lower-generation or reduced-lane slot is measured as that negotiated link. DDR5 capability alone does not establish GPU input bandwidth. Do not require expensive counters that the platform cannot expose; mark missing metrics and use explicit byte/latency instrumentation instead of inventing values.

### 17.4 Controlled experiment matrix

Hold model, numerical realization, prompts, context, seed, output limit, encoding, and client placement constant when comparing server modes. Required comparisons are:

- CPU single node; CPU multi-node sharded; replicated experts; replicated server set where capacity permits.
- CUDA server with cold/limited caches and realistic protected caches.
- Client resident reference; required host-backed weights/state without a host expert cache; the same backing with host expert caching enabled.
- Expert VRAM hits, host-only hits chosen locally, host-only hits sent remotely, double misses, and mixed rows/layers.

Separate boot/preparation from warm serving, but report both. Include repeated sessions/domains, context growth, continued prompts, and broad routing traces larger than the caches under test. Do not report only a single favorable prompt or one hot expert. Compare equal workloads to claim a speed improvement; use a distinct experiment to show increased supported capacity.

### 17.5 Cache planning inputs and outcomes

Record client VRAM expert hit rate, client RAM membership and chosen local execution, server GPU cache hits, CPU locality, Engram tier hits, weight/state bytes, fully local layers, and critical-path wait times. A high hit rate can conceal large remaining expert sizes or an unavoidable remote request.

The saved plan includes the calibration hardware/build/profile identity and cache-fill recipe. It must be rejected or recalibrated when an incompatible configuration changes. Do not continuously mutate policy to improve one benchmark without recording it.

There is no invented minimum tokens/s target. The release report declares measured usability for the chosen context/hardware and any agreed deployment target. A CPU route can be functionally correct yet too slow for a particular interactive workload; that is a reportable result, not permission to remove a required mode.

## 18. Validation and acceptance matrix

### 18.1 Test hierarchy

Use exact byte/identity tests, operator comparisons, complete layer/state tests, full-model teacher-forced evaluations, and generation tests. A readable short completion is not model fidelity evidence. Unit tests for a donor project do not replace these tests.

The test suite has an independent reference boundary. It must not generate expected outputs by invoking the same faulty optimized function being tested. Golden fixtures record their source, profile, reference implementation, input dtype, quantizer settings, state layout, and tolerance policy.

Numerical tolerances and quality criteria are committed before evaluating an optimization. Define appropriate absolute/relative error, cosine/error norms, logit and teacher-forced loss criteria by operator/profile. Do not widen thresholds after a failure merely to pass. Exact equality applies to identities, serialized values, slot associations, retained packed state, and deterministic pure bookkeeping; heterogeneous floating arithmetic uses justified limits.

### 18.2 Mandatory operator fixtures

| Category | Coverage |
|---|---|
| Tensor/quantization | Every used value code, scale orientation and extrema, nibble ordering, zeros, saturation, rounding, padding, odd/tile boundaries, block edges, quantizer scope across split batches |
| FP8/NVFP4/BF16 | Native source decode; strict W4A4 versus deliberately different W4A16 negative test; W8A8 groups; sensitive FP32 operations; output dtype and casts |
| Expert | Both gate/up branches, asymmetric clamps, route coefficient before down quantization, zero/extreme valid coefficients, shared branch once, ordered reduction |
| Routing | Source score function, normalized raw weights versus biased selection, text/image bias, close/tied scores, original selected slots preserved |
| Attention/indexing | Causality, sparse validity, candidate blocks, latest partial block, source-layer aliases, RoPE/sinks, quantized historical cache bytes, tile normalization |
| mHC/Engram | Mix ownership across sublayers, flattened-stream norms, layer injection, hash history/padding/special tokens, exact rows/scales, CUDA fusion |
| State | Full/partial compression groups, dirty writeback, immutable-page reuse, ring wrap, aliases, cancellation, continuation and exact reconstruction |

Test the actual released target dimensions as well as smaller fixtures. Small fixtures expose indexing and arithmetic defects but do not prove target kernel launch limits, scale layouts, or memory capacity.

### 18.3 Required execution combinations

The core matrix contains **three profiles × two server modes × two client residency conditions**, with target-model end-to-end coverage in each supported combination:

| Dimension | Required cases |
|---|---|
| Profile | `nvfp4`, `fp8`, `bf16` |
| Server executor | CPU-only, Blackwell CUDA |
| Client residency | Resident reference where feasible; explicit client-RAM-backed weights and state |
| Placement | All remote; available client VRAM; client host-only; mixed selected slots |
| Phase | Initial prefill, ordinary decode, continued prefill/multi-turn |

For fixtures or reference runs needing more memory than the target host provides, use a qualified larger reference environment or bounded independent reference evaluation and record it. Do not claim full-model parity from an unexecuted reference or a toy model. A reduced context is a declared test condition, not proof of all configured contexts.

All required CPU NUMA policies receive correctness/capacity tests. Full-replica end-to-end performance is qualified only on a machine with enough per-node RAM; emulated metadata tests are labeled as such. A mode that has not passed its target-model gate remains incomplete, not quietly dropped from the matrix.

### 18.4 Residency and capacity tests

Force a lower VRAM budget to exercise migrations even when a short prompt fits. Also execute a real target workload whose admitted client footprint exceeds physical/configured VRAM. Confirm actual H2D/D2H, valid host state, and absence of client CPU neural operations.

Required adversarial allocations include minimum executable slots; cache budgets of zero and very small sizes; a server VRAM cache smaller than the current selected-expert union; insufficient host budget; insufficient per-node replica memory; registration failure; and prefill with a broad expert union. Admission must identify the real limiting resource.

Inspect physical pages, resident/proportional shared accounting, major faults, disk reads, swap, pinned pools, CUDA allocations, conversion peaks, source mappings, and intentional replicas. Exercise checked offsets beyond 4 GiB and 1 TiB with synthetic sparse fixtures, then separately qualify real high-capacity runs. Synthetic addressing success is not multi-terabyte performance evidence.

Prove that neither client initialization nor server expert-only initialization loads an unintended complete second model. Verify no duplicate server attention/KV stack and no unnecessary replica of each unused model component.

### 18.5 State and feature tests

Compare resident and host-backed continuation over long prompts, short appended turns, tool results, image spans when enabled, cache-source boundaries, candidate-block boundaries, and sliding-window wrap. Test reused shared state across all readers and compressor partial groups at chunk boundaries.

Test exact retained-state versus exact reconstruction, and include a negative test demonstrating that approximate bounded replay cannot be labeled the exact baseline. Quantization-profile changes invalidate state/cache identity. Memory migration alone must not requantize historical state.

Native encoding tests include system/user/assistant/tool messages, reasoning mode, special tokens, BOS/EOS, empty input validation, Unicode/multibyte streaming, and incomplete structured output. Image support needs its own vision fixtures and whole-model tests; text success cannot be reported as multimodal acceptance.

### 18.6 Failure injection

Inject failures before dispatch, during partial transmission, after server execution but before reply, during a contribution copy, during a state writeback, and after a token is sampled but before its subsequent state update. Include CPU worker loss, server restart, CUDA errors, OOM, cancellation, stale responses, and wrong artifact bindings.

The expected result is a typed failure, safe resource quiescence, invalidated uncertain session state, and no substituted output. Confirm stale DMA cannot overwrite a recycled new-session buffer and that an abandoned prefetch cannot publish validity after cancellation.

### 18.7 Performance and soak qualification

Run repeated representative workloads with all instrumentation necessary to explain the bottleneck. Include at least a sustained mixed-workload soak sufficient to expose state/cache leaks and thermal effects; use a minimum one-hour engineering soak for the initial release report, with exact duration/hardware recorded. This is an acceptance test requirement, not a claim that it has been run.

Report distributions, prompt/output lengths, cold/warm state, visible errors, peak memory, relevant counters, and reproduction instructions. Keep benchmark scripts and raw machine-readable data with the source release. Do not summarize aggregate concurrent throughput as single-session token speed.

### Additional mandatory contract tests

Add source-independent byte fixtures for zero-to-bound session transition, different approved physical layouts, u64 decimal JSON values above `2^53`, JCS Unicode ordering/escaping, manifest self-digest exclusion, metadata-index trust and chunk-boundary permissions. For requests, test exact length equations, CSR/row/expert uniqueness, out-of-order responses, type-specific limits, no-credit/no-buffer progress, partial TLS frames and operation-versus-frame deadlines.

Exercise cancellation before dispatch, in a CPU/GPU worker, after result completion, and with an acknowledgement racing the terminal reply. Assert one terminal target response and one credit release, never one per cancellation message. Test control and bulk request-ID namespaces, one-use capabilities, failed bulk reconnect, and fresh binding after invalidation. Include a dirty page or output buffer still leased when cancellation arrives.

Container tests include bootstrap without a final memory plan, finalize refusing incomplete calibration/capability evidence, a changed final cap forcing a re-probe, inherited shell variables attempting to override an approved GPU/image/memory setting, effective Compose versus actual-container comparison, and a core-dump attempt producing no secret/state dump. Re-encoded thinking/tool/image histories that change old tokens must reset and re-prefill rather than reuse an unsafe prefix. Report a failed/sticky GPU event through bounded process termination, not a hanging health check.

### 18.8 Release gates

| Gate | Passing evidence |
|---|---|
| Identity | Verified target artifacts, source lock, full tensor/operator inventory, native encoding fixtures |
| SM120 graph | Correct target client prefill/decode/state on RTX 5090, no CPU neural operations |
| Precision | All required profile contracts and both fidelity comparisons, not just dtype loading |
| CPU server | Actual-ISA target expert execution without CUDA, performance and NUMA reports |
| CUDA server | Bounded RAM-to-GPU expert execution on qualified Blackwell, no CPU expert fallback |
| Client RAM | Correct backing of weights and exact state, meaningful host expert/row caches, bounded peaks |
| Distribution | Physical two-host target runs, unique contribution ownership, correct failure handling |
| NUMA | Real allocation/replica identity, per-node capacity, distinct jobs, qualified policy measurements |
| Engram | Exact required tables/rows/history, separate caching, CUDA fusion |
| Features | Text/reasoning/tools/continuation; separately qualified native images |
| Operations | Authentication, resource bounds, readiness, cancellation, logging and release reproduction |
| Docker | Digest-pinned CPU/CUDA images, two-host Compose deployment, resource/NUMA/GPU isolation, driver-free CPU tests, containerized fidelity/capacity/failure evidence |
| Performance | Honest hardware/profile/context results with latency, capacity, and bottleneck attribution |

The Docker acceptance matrix in Section 22 applies to these same gates, not to an alternate container-only subset. A staged milestone can be useful while gates remain incomplete. The release must state that incompleteness and cannot advertise the entire contract as working until the corresponding matrix passes.

## 19. Implementation sequence and deliverables

### 19.1 Start with evidence and independent operator boundaries

Do not begin by wiring two full-model servers together. The first engineering work is the source/artifact lock, target tensor inventory, memory estimator, and independent mathematical fixtures. Hardware/transfer probing can run in parallel with that work.

| Milestone | Work | Exit evidence |
|---|---|---|
| Source and inventory | Pin upstreams, inspect target shards/scales, create manifest and native encoding fixtures, inventory actual hardware | Reproducible source lock; no unknown required tensors or fabricated pins |
| Container foundation | Build locked CPU/CUDA toolchain and runtime targets; implement health/probe/configuration interfaces and resource policy | Non-root smoke tests, CPU image without NVIDIA dependencies, SM120 probe on a GPU host, verified NUMA/memlock/cgroup permissions |
| Profile and expert core | Strict NVFP4 W4A4, source-compatible FP8 W8A8, BF16; extract selected-expert evaluator | Independent operator parity; bounded tiles; CPU/CUDA kernel capability report |
| Client V4.1 graph | Port required CED/CSA2/mHC/Engram graph into one native runtime, eager first | SM120 target graph/state fixtures and resident reference execution |
| Expert service | One protocol; CPU-only and CUDA builds; read-only tensor/row service; loopback then physical network | Exact associations, integrity/authentication, resource and failure tests |
| CPU locality | NUMA worker ownership, sharding, replicas, per-node admission | Verified physical locality and representative CPU results |
| Client memory tiers | Host weight/state backing, migration/leases, exact continuation, double client-cache misses | Real constrained-VRAM target execution, no CPU math or remote private-state spill |
| Bounded prefill and caches | Prompt-row/expert groups, protected/transient slots, client phase plans, server residual-workload cache | Target prefill/decode under hard budgets; measured traffic by class |
| Full target qualification | Both server modes and profiles, continued prompts, required feature tests, failure injection, soak | Completed acceptance matrix and evidence report |
| Optimization | Verified copy/compute overlap, selected fused kernels, optional graph segments, cache tuning | Equal-workload end-to-end improvement without fidelity or capacity regression |

These are dependency-aware milestones, not a temporary alternate architecture or a promise of implementation duration. CPU and CUDA kernel work can proceed independently behind the same descriptor contract. State paging depends on the correct graph, not merely on an allocator.

### 19.2 Native code refactor boundaries

| Existing area / new focused module | Required change |
|---|---|
| `ds4.c` and model graph | Add the actual V4.1 schedule/semantics, explicit tensor handles, normalized-input/selected-expert boundary, source-defined merge |
| `ds4_gpu.h` | Express bounded operator/subset execution and completion requirements without pretending every weight has a local map address |
| `ds4_cuda.cu` / targeted CUDA modules | Qualify SM120 operations; remove relevant V4/fixed-capacity assumptions; expose expert subgroups and state/weight tile inputs |
| Existing distributed source | Reuse reviewed utilities only; whole-layer workers remain distinct from the new expert service |
| Expert evaluator module | Shared descriptor and CPU/CUDA implementations; no complete transformer initialization |
| Model preparation/store | Native mixed checkpoint ingestion, verified tensor banks/ranges, reproducible profile preparation |
| Client residency/transfer modules | Host/device budgets, state validity, leases/events, progress pools, safe staging |
| NUMA allocator and workers | Explicit sharding/replication, physical admission and unique job ownership |
| Engram service/client cache | Exact row retrieval, history/identity, client host/device row caching and CUDA fusion |
| Protocol and service | Narrow authenticated request handling, credits, typed errors, immutable binding |
| Tests/metrics | Golden fixtures, capability checks, memory counters, source/profile/topology reports |

These are responsibilities and likely source boundaries; avoid one new abstraction per table row when a small direct function suffices. Do not add version-suffixed parallel implementations, legacy migration windows, or fake pointer-compatible wrappers. Preserve unrelated existing functionality without redesigning the entire project.

### 19.3 Required handoff outputs from implementation

The implementation release must include the integrated source; Dockerfiles, image/base digests, SBOM and build provenance, host-local Compose manifests, generated deployment configuration, a reviewed NUMA seccomp profile, container preflight/runbooks and tests; source/build/model locks; preparation/inspection tools; CPU and CUDA server builds; the CUDA client; exact configuration schema; reproducible deployment instructions; numerical and state fixtures; protocol tests; a target-profile support matrix; capacity/placement/calibration reports; and benchmark/failure/soak results.

Store raw reports and final artifacts durably in the project's repository/artifact store. Do not rely on an ephemeral ZIP, a terminal scrollback, or a mutable upstream branch. Source-controlled checksums and deterministic filenames identify artifacts without semantic version prefixes on ordinary functions or modules.

Document which experiments require larger reference hardware, model-access permission, or more RAM than the initial server. Do not implement lossy shortcuts to hide a hardware shortfall. A missing required optimized kernel is work to complete, not evidence that changing the target model satisfies the project.

### 19.4 First implementation checklist

The engineer beginning from this file should first resolve the named repository identity and exact commits, inspect the official/NVIDIA tensor inventory, generate the memory and source lock, and construct one real target expert fixture preserving scales/clamps/weighting. Then execute it locally on the intended CPU and Blackwell devices before introducing the network.

In parallel, establish a minimal SM120 V4.1 attention/index/cache fixture and verify its packed state against the source. Those two independent boundaries expose the highest-risk kernel gaps early. No unrelated small-model success should defer resolving them.

## 20. Risk register and decisions to measure

Architecture choices above are settled requirements. The following are empirical qualification tasks, not reasons to reopen passive storage-only streaming as the default.

| Risk / unknown | Required resolution | Not an acceptable substitute |
|---|---|---|
| “BlackStar” lacks a verified upstream identity | Retain the known DwarfStar base; record any later explicit repository selection separately | Invent a repository or silently switch projects |
| V4.1 source and dependencies move rapidly | Pin exact coherent commits and retain patches/tests | Treat moving documentation as a tested release |
| SM120 attention layout/kernel gaps | Port/implement and verify actual operators/cache formats | Run SM100-only code or substitute generic attention |
| Required W4A4 path lacks correct clamping/scales | Implement exact operation or identify a separately tested numerical realization | Report W4A16 as W4A4 |
| FP8 CPU path missing/slow | Add ISA-qualified bounded arithmetic and benchmark target workloads | Silent CUDA requirement in CPU-only mode |
| Old CPU cannot exploit DRAM bandwidth | Optimize measured unpack/compute/NUMA bottleneck and report speed honestly | Multiply nominal socket bandwidth into a tokens/s claim |
| CUDA server is PCIe-limited | Measure misses, cache/group reuse, staging and compute | Assume installing a GPU guarantees faster execution |
| Client RAM staging competes with attention | Use measured phase plan and priority/progress reservations | Always execute every host-cached expert locally |
| 512 GB server cannot hold selected population | Produce actual tensor-bytes admission; use appropriately sized test hardware | Prune active experts or substitute lossy tables |
| Replication exceeds one node's capacity | Shard or use an admitted replica set and report locality | Claim local replicas backed remotely/shared physically |
| State pager loses aliases/dirty data | Versioned validity, leases, exact fixtures and failure tests | Approximate bounded replay or silent context truncation |
| Fused prefill needs all experts resident | Bounded expert/row groups or validated tiles | Increase unreported peaks or fall back to CPU on client |
| Remote latency dominates | Measure full layer path, reduce protocol/copy overhead, exploit exact cache locality | Pretend dependent layers batch into one exact request |
| Default container policy blocks local NUMA allocation | Deliver reviewed seccomp, finite memlock, effective cpuset and physical page tests | Privileged/unconfined deployment or silently nonlocal replicas |
| Container image lacks accelerated SM120 artifacts or imports an incompatible host ISA | Probe `sm_120a` low-bit code and all linked host dependencies on each real target | Treat plain SM120 selection, nvidia-smi or a modern build host as qualification |
| Docker cgroup, tmpfs or startup peak exceeds reserved RAM | Check effective limits and full container peak, with no swap | Count only model tensor bytes or disable the OOM killer |
| No deployment speed target supplied | Report measured distributions by workload and agree any operational threshold before purchase | Promise a fixed interactive rate from capacity alone |

## 21. Traceability and definition of completion

### 21.1 Requirement coverage

| Locked requirement | Specification | Acceptance evidence |
|---|---|---|
| Exact V4.1 target and native behavior | Sections 3–6, 9–10 | Identity, graph, encoding, profile, state fixtures |
| Blackwell/5090 client | Sections 1, 5, 9 | Actual SM120 build/operator/end-to-end tests |
| NVFP4, FP8, BF16 | Section 4 | Both fidelity axes, both server modes, all used tensor classes |
| No client CPU neural work | Sections 1, 7–9, 16 | Instrumented host-backed and local-expert tests |
| Client DDR5 use | Sections 8–9, 13, 15 | Real constrained-VRAM weight/state/cache execution |
| CPU-only server | Section 11 | Driver-free serving and actual-ISA target benchmarks |
| CUDA server with small VRAM | Section 12 | Bounded expert groups, misses, prefill, no attention/KV duplicate |
| NUMA copies and concurrency | Section 11 | Physical replicas, unique jobs, per-node admission and comparisons |
| Expert disaggregation | Sections 7, 14 | Physical-host inputs/results path and double-miss tests |
| No routine network expert-weight misses | Sections 6–7, 13–14 | Traffic-class accounting and cold client-cache tests |
| Engram correctness and caches | Section 10 | Source row/history/fusion fixtures and bounded tiers |
| Preserve exact state/model behavior | Sections 4, 9, 16 | Resident/paged parity and approximation-negative tests |
| Maximum useful resource utilization | Sections 13, 15, 17 | End-to-end latency/capacity optimization under hard budgets |
| Operational safety/durability | Sections 14–16, 19, 22 | Auth/failure tests, reproducible release and durable evidence |
| Docker deployment | Section 22 | Native/contained parity, CPU-only image independence, effective GPU/NUMA/memory controls, two-host startup and failure tests |

### 21.2 Completion statement

The project is complete for a declared Docker release configuration when the target model/profile is verified, both required server modes are implemented and qualified, the Blackwell client uses its host/device tiers correctly, the actual working set fits without hidden disk/CPU fallback, and the release evidence establishes numerical behavior, capacity, failure safety, and performance for the declared scope.

A capability can be reported as functionally supported while a specific hardware combination misses its desired latency target. That distinction must remain explicit. The design does not promise arbitrary model architectures, unlimited context, universal replica feasibility, an interchangeable Blackwell binary, or a particular generation rate.

**Final architecture:** a DwarfStar-derived CUDA attention client with explicitly managed VRAM and system RAM, connected to a RAM-resident expert/Engram service using either NUMA-aware CPU execution or Blackwell CUDA execution. FreeToken's useful memory/expert mechanisms are integrated into this one runtime with corrected target-model semantics and bounded resource use. Every selected operation is performed once at an admitted owner; normal expert traffic is inputs and contributions. Performance is optimized by measured critical-path savings, not by hiding transfers, modifying routing, or allocating every available byte.

## 22. Docker deployment and operating contract

### 22.1 Deployment boundary and supported hosts

Docker is a release requirement, not an optional wrapper around an undocumented native launch. Deploy **one attention-client container on the client host and one expert-server container on the server host**. A container does not span both machines. Use two host-local Docker Compose projects connected by the authenticated expert protocol; do not assume a Compose bridge, service name, or `depends_on` relationship spans two independent Docker Engines. [R56]

The initial deployment baseline is native **Linux x86-64, rootful Docker Engine, Docker Compose v2, and cgroup v2**. The service processes themselves run as an unprivileged numeric UID/GID. Pin the tested Engine, Compose, runc/containerd, Linux distribution/kernel, NVIDIA driver and Container Toolkit in the deployment lock. “Latest” is not a reproducible version. Docker Desktop, WSL2, rootless Docker, user-namespace remapping, Swarm, Kubernetes, and non-x86 CPU servers need separate qualification and are not implied by this baseline. No Windows-container image is required.

A CPU-only server host needs no NVIDIA GPU, kernel driver, Container Toolkit, or CUDA installation. Each GPU-equipped host needs a compatible NVIDIA **host driver** and NVIDIA Container Toolkit configured for its Docker Engine. The container supplies the tested user-space CUDA libraries; it does not install a kernel driver or replace the host's driver. Configure the runtime as an explicit administrator task and revalidate existing workloads before restarting Docker. The standard configuration operation is `nvidia-ctk runtime configure --runtime=docker`; the project startup script must not run it or restart a host daemon silently. [R57]

NVIDIA's CUDA-family compatibility table lists driver branch 580 or later for CUDA 13.x minor-version compatibility, but features, PTX JIT, GPU support and specific libraries may require a newer driver. Record the actual tested driver and kernel set; do not interpret that family-level floor as qualification of every kernel. The `CUDA Version` displayed by `nvidia-smi` is a driver capability indicator, not the toolkit version installed in the image. [R58]

The original DwarfStar/FreeToken host-memory architecture is unchanged by containerization. Docker cannot turn client RAM into VRAM, repair an unsupported CPU ISA, expose an uninstantiated SM120 kernel, create local NUMA replicas automatically, or make a too-large model fit its memory cgroup.

### 22.2 Required image artifacts

Deliver two production runtime images from the same integrated source revision and shared logical operator contract:

| Image artifact | Contents | Required independence |
|---|---|---|
| CPU expert runtime | `ds4-expert-server`, `ds4ctl`, required CPU/NUMA/TLS/standard libraries, licenses and build metadata | No GPU request, CUDA initialization, `libcuda`/`libcudart` linkage, Torch/Triton import, or NVIDIA runtime dependency |
| CUDA runtime | CUDA client CLI `ds4`, HTTP client service `ds4-server`, CUDA-mode `ds4-expert-server`, `ds4ctl`, qualified CUDA libraries/kernels and licenses | One selected GPU; no full-model Python engine; no CPU expert fallback in CUDA server or neural fallback in client |

A separate build/preparation/reference image may carry compilers, Python, reviewed conversion tools and independent reference dependencies. It is not a production inference tier and must not be started as an additional full-model service. CPU tests must include dependency inspection and a launch on a host without usable NVIDIA libraries, not merely selecting `--executor cpu` inside a CUDA image.

Images contain code, configuration schemas, test-sized non-sensitive fixtures and kernel artifacts—not model weights, HF credentials, private certificates, user prompts, downloaded caches, or prepared multi-terabyte banks. Model acquisition and conversion happen before production serving through explicit preparation tasks. Runtime containers must work without internet access; their only required network peer is the configured expert/client relationship and permitted API caller.

Use multi-stage builds, digest-pinned base images, locked dependency sources/package inputs and retained build provenance. Build outputs record target architecture, compiler flags, linked library versions, source commits and profile/kernel capabilities. A base digest freezes that base, not subsequently fetched unpinned packages; any dependency installation must also use locked inputs or a recorded package snapshot. Preserve an SBOM and redistribution notices. [R59][R60]

### 22.3 Build contract and image layout

The following new packaging targets and paths are **required outputs to implement in the fork**. They are not existing upstream Makefile targets. Keep the existing application names where applicable; add only the expert entry point and control/probe tool needed by this architecture.

| Build target / output | Contract |
|---|---|
| `package-cpu` | Install the CPU expert runtime under `DESTDIR`, using the specified `PREFIX`; no CUDA discovery or build requirement |
| `package-cuda` | Install the CUDA client, HTTP client service, CUDA expert service, control tool, native kernels and licenses |
| `CPU_BASELINE=x86-64` | Safe host-code baseline for the intended x86-64 deployment; optional optimized functions use capability-checked dispatch |
| `CUDA_ARCH=sm_120a` | Explicit accelerated target for the required RTX 5090 native block-scaled kernels; physical device SM120 is checked separately. No build-time device auto-detection |
| `/usr/local/share/dwarfstar/build-info.json` | Source/build lock, profile/kernel support, CPU ISA baseline, exact CUDA targets and dependency hashes |
| `/usr/local/share/dwarfstar/schema/` | Strict service, memory-plan, manifest and protocol schemas |
| `/usr/local/share/licenses/dwarfstar/` | Complete notices for incorporated source and runtime components |

The project must test **all linked host libraries** on the supported CPU baseline. Avoid `-march=native` in a redistributable image. A baseline executable that loads a library requiring AVX2 on an AVX-only DDR3 server does not meet CPU portability. Building on a modern workstation does not change the older host's capabilities.

Compile required SM120 SASS/AOT kernels, including `sm_120a` accelerated code where required by Section 5, without needing a GPU on the build worker. Other Blackwell architecture artifacts are explicit separate build outputs or tested additional kernel sets. Never copy an SM100-only binary into an SM120 image and rely on a family name. A Python/Triton/CuTe donor needs a supported native/AOT integration or a reviewed port; a `.py` entry point is not a native kernel package.

Ahead-of-time compilation is the production baseline. Where a tested backend unavoidably performs JIT, compilation must finish during a bounded startup/preparation phase before readiness, use a dedicated size-limited kernel cache, and have its source/compiler/device key recorded. Runtime internet downloads and first-user-request compilation are forbidden. Kernel cache files are code cache, not model/state paging; the model residency contract still applies.

**CPU Dockerfile template** (`deploy/Dockerfile.cpu`):

```dockerfile
ARG BUILD_IMAGE
ARG RUNTIME_IMAGE
FROM ${BUILD_IMAGE} AS build
WORKDIR /src
COPY . .
RUN --network=none make package-cpu \
    DESTDIR=/out PREFIX=/usr/local CPU_BASELINE=x86-64

FROM ${RUNTIME_IMAGE} AS runtime
COPY --from=build /out/ /
ENV HOME=/run/dwarfstar
USER 10001:10001
WORKDIR /run/dwarfstar
ENTRYPOINT ["/usr/local/bin/ds4-expert-server"]
CMD ["--config", "/etc/dwarfstar/service.json"]
```

**CUDA Dockerfile template** (`deploy/Dockerfile.cuda`):

```dockerfile
ARG BUILD_IMAGE
ARG RUNTIME_IMAGE
FROM ${BUILD_IMAGE} AS build
WORKDIR /src
COPY . .
RUN --network=none make package-cuda \
    DESTDIR=/out PREFIX=/usr/local CPU_BASELINE=x86-64 CUDA_ARCH=sm_120a

FROM ${RUNTIME_IMAGE} AS runtime
COPY --from=build /out/ /
ENV HOME=/run/dwarfstar
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
USER 10001:10001
WORKDIR /run/dwarfstar
ENTRYPOINT ["/usr/local/bin/ds4-server"]
CMD ["--config", "/etc/dwarfstar/service.json"]
```

`BUILD_IMAGE` and `RUNTIME_IMAGE` must be digest-qualified values from the source/build lock. Their exact contents must include the already resolved build or runtime dependencies: a bare CUDA base does not necessarily provide TLS, NUMA or every required math library. These templates intentionally do not invent tested image digests or package versions. The release builder validates the lock, supplies the images and stages reviewed offline dependencies before the network-disabled packaging step. The source/build lock must be copied explicitly into the context even when `.git` is excluded. `/out` must contain a complete filesystem installation rooted at `/`, including its runtime directory and metadata. The runtime stage must resolve every library without using build-only absolute paths or a copied host driver. Use a matching, tested libc/toolchain family rather than transferring arbitrary glibc binaries into an Alpine/musl image. Exec-form entry points and explicit numeric users are deliberate image contracts. [R77]

A `.dockerignore` must exclude at least model/input/output roots, `*.safetensors`, `*.gguf`, prepared weight banks, private keys/certificates, credentials, `.env` secrets, large caches, benchmark data containing prompts, and VCS/build junk not consumed by the source lock. Do not indiscriminately exclude small tracked fixtures or required source-lock metadata. Inspect the final build context and image history; a later `RUN rm` does not remove secret/model bytes from an earlier image layer. Build secrets, when acquisition needs them, use BuildKit secret mounts rather than `ARG`, `ENV`, or committed configuration. [R61]

### 22.4 Native entry points and administrative interface

Implement these interfaces consistently across packaging, configuration, health checks and tests:

| Entry point | Required behavior |
|---|---|
| `ds4 --config PATH` | Interactive/one-shot client using the same model engine and admission plan; not a second architecture |
| `ds4-server --config PATH` | Client-side HTTP/SSE service; performs inference on the client GPU and talks to the remote expert service |
| `ds4-expert-server --config PATH` | CPU or CUDA expert service according to its compiled capabilities and explicit configuration |
| `ds4ctl validate --config PATH` | Validate schema, file/secret accessibility, lock/profile coherence and configured bounds without loading the whole model |
| `ds4ctl probe --probe-config PATH --output PATH` | Bounded container-aware hardware/permission/kernel probe using its own minimal schema; requires no final memory plan, deployment lock or loaded model |
| `ds4ctl plan --request PATH --inventory PATH --probe PATH --calibration PATH --output PATH` | Generate admission from independent planning inputs; no already-completed service memory plan is required |
| `ds4ctl health --socket PATH --ready` | Read the local process's management socket with a bounded timeout; exit zero only when its ready contract holds |
| `ds4ctl drain --socket PATH` | Stop new session admission and initiate bounded orderly cancellation/drain |

A control tool operation must have a finite deadline and return a real nonzero exit code on failure. Probe GPU memory with a small explicitly capped allocation, run a representative compiled kernel for the selected SM, and free it. `nvidia-smi` success alone is not a kernel or model-readiness test. CPU-mode probes do not initialize CUDA. A probe reports unavailable optional telemetry as unavailable rather than requiring broad privilege.

The management socket is private at `/run/dwarfstar/admin.sock`, mode `0600`, owned by the container's service UID. Use a small bounded local protocol and authenticate local access using filesystem permissions and peer credentials where available. It exposes readiness, current progress, safe metrics and drain—not arbitrary code, paths or memory reads. Liveness does not perform a full model scan. A fully initialized process remains healthy while its admitted inference slot is busy; busy/queue capacity is separate from health. A client process can be alive while waiting for server readiness; it must not advertise inference readiness until the successful selected-policy binding and client memory/kernel plan are ready.

### 22.5 Service configuration and resolved plans

Implement strict JSON schemas with duplicate/unknown-field rejection and explicit types. Production JSON has no environment-variable interpolation, implicit unit suffixes, hidden path fallback, or guessed essential defaults. A deployment renderer resolves variables into the file and checks it before launch. The fixed paths below are container paths; host paths are only in the deployment's bind definitions.

The common service document requires:

| Field | Type and validation |
|---|---|
| `schema_revision` | Integer `1`; reject unknown revision |
| `role` | Exactly `client` or `expert` |
| `executor` | `cuda` for client; exactly `cpu` or `cuda` for expert |
| `model_manifest` | Absolute read-only path to the admitted compact client or server manifest |
| `memory_plan` | Absolute path to the resolved plan, including profile, context and phase placement |
| `deployment_lock` | Absolute path to the immutable build/source/image/environment lock |
| `device_index` | `0` when the one selected GPU is exposed; `null` in CPU mode; verify UUID against deployment lock |
| `management_socket` | `/run/dwarfstar/admin.sock` |
| `artifacts_dir` | Explicit writable report directory; no model/state spill |
| `tls` | Default mutual TLS: CA/certificate/key paths, peer SAN identity, TLS 1.3, no early data, optional `enabled: true`. Trusted-network TCP: exactly `{ "enabled": false }`, without certificates/encryption/cryptographic peer authentication |
| `network` | Control/bulk addresses, explicit connect/handshake/operation deadlines, frame/credit limits |
| `api` | Client-only HTTP policy described below; forbidden in expert role |

**Illustrative client service document.** Paths and structural choices are fixed; the deployment renderer substitutes the actual verified peer name and writes the associated memory plan. The timeouts shown are starting configuration values, not measured latency targets or universal limits.

```json
{
  "schema_revision": 1,
  "role": "client",
  "executor": "cuda",
  "model_manifest": "/model/manifest.json",
  "memory_plan": "/etc/dwarfstar/memory-plan.json",
  "deployment_lock": "/etc/dwarfstar/deployment-lock.json",
  "device_index": 0,
  "management_socket": "/run/dwarfstar/admin.sock",
  "artifacts_dir": "/artifacts",
  "tls": {
    "ca_file": "/run/secrets/ca.pem",
    "certificate_file": "/run/secrets/peer.pem",
    "private_key_file": "/run/secrets/peer.key",
    "expected_peer_name": "expert.example.internal",
    "minimum_version": "TLS1.3",
    "early_data": false
  },
  "network": {
    "control_address": "expert.example.internal:7443",
    "bulk_address": "expert.example.internal:7444",
    "connect_timeout_ms": 10000,
    "handshake_timeout_ms": 10000,
    "operation_timeout_ms": 120000,
    "frame_io_timeout_ms": 30000,
    "write_timeout_ms": 30000,
    "max_row_lookup_rows": 256,
    "max_inflight_payload_bytes": 67108864,
    "max_frame_payload_bytes": 16777216,
    "max_bulk_data_bytes": 4194304,
    "max_inflight_expert_requests": 2
  },
  "api": {
    "bind_address": "0.0.0.0:8000",
    "bearer_token_file": "/run/secrets/api.token",
    "max_body_bytes": 67108864,
    "header_timeout_ms": 10000,
    "body_timeout_ms": 30000,
    "stream_write_timeout_ms": 30000,
    "max_active_generations": 1,
    "max_queued_generations": 0,
    "allow_remote_image_urls": false,
    "cors_allowed_origins": []
  }
}
```

For the expert document, set `role=expert`, choose `executor`, set `device_index=null` for CPU or `0` for CUDA, omit `api`, and set control/bulk to `0.0.0.0:7443` and `0.0.0.0:7444`. In TLS mode set `tls.expected_peer_name` to the authorized client certificate's SAN identity. For trusted-network TCP use exactly `tls: {"enabled": false}` on both hosts and `expected_peer_name: null` in applicable expert grants; no CA/certificate/key provisioning is required. The client API token is still required. The same JSON field names express listener addresses on the expert and destination addresses on the client; schema/role validation makes the distinction explicit. In TLS mode the client peer name must match server certificate identity, not an arbitrary Compose service name.

The separately resolved memory plan MUST contain the profile and operator-contract digests; exact model/tensor/layout identities; maximum semantic token positions; phase-specific prefill microbatch; image feature limits; client host/VRAM/pinned/state/cache/progress pools; server global/per-node budgets; CPU worker affinity and placement or CUDA slots/cache; worst-case peaks; strict residency requirements; quantizer-scoped partition rules; cache membership/destination plan; and the calibration identity. Configured memory amounts are nonnegative safe-integer JSON byte counts no greater than `9007199254740991`; essential minimums must be positive. Full-width identity/offset fields use the decimal-string convention in Section 6. Internal allocation arithmetic is checked against the platform and allocator limits. Pinned/copy overlap is accounted without double charging. Each budget has a peak allocation equation and an enforcing allocator/counter. Exceeding a negotiated per-request limit is rejected before work is accepted.

The deployment lock ties image digest and native build digest to the source/model/profile, selected GPU UUID/SM, CPU ISA, tested driver/runtime, Docker/Compose/kernel, cgroup/NUMA requirements, seccomp digest, schema revision, selected transport policy and TLS peer names (null for trusted-network TCP). It contains **no private keys or bearer-token contents**. Hardware-dependent values remain unresolved until probed; a file of placeholder hashes cannot pass admission. Changes to these dependencies invalidate the affected plan and require explicit regeneration/requalification, not a live silent fallback.

### Probe schema and noncircular deployment rendering

Hardware discovery precedes final admission. `ds4ctl probe` must therefore **not** require the final service document's `memory_plan`, deployment digest or model readiness. Its distinct strict JSON schema contains `schema_revision=1`, intended `role`, intended `executor`, `device_index` (`null` for CPU), `expected_gpu_uuid` (`null` for CPU), `numa_nodes` (an explicit integer list), `max_host_test_bytes`, `max_device_test_bytes`, `max_pinned_test_bytes`, `deadline_ms`, and `disable_core_dumps=true`. It contains no model handles or neural input data. CPU mode requires zero device/pinned GPU-test budgets and never initializes a CUDA library. Memory allocation probes remain bounded independently of the eventual multi-terabyte model plan.

Produce a **bootstrap deployment** from operator-selected host caps, image/build identity, peer names and paths. This writes `probe.json` and enough Compose environment/mount/security settings to run the bounded probe, not a forged passing memory plan. `tools/render_deployment.py bootstrap --request PATH --output-dir DIR` and `tools/render_deployment.py finalize --request PATH --probe PATH --inventory PATH --calibration PATH --output-dir DIR` are required renderer interfaces. Both validate inputs and publish outputs atomically. Bootstrap reports `not_admitted`; production service startup must reject it.

The probe records actual container CPU/memory-node masks, cgroup ceilings, finite lock allowance, process dumpability, seccomp/NUMA self-allocation, GPU UUID/SM and required small kernels. An external host preflight records parent-cgroup and physical topology constraints not visible inside. Finalize combines this evidence with immutable tensor inventory, profile/operator capabilities and placement calibration to generate `service.json`, `memory-plan.json`, `deployment-lock.json`, and the exact Compose environment. The renderer and `ds4ctl plan` use the same validated admission calculations; do not maintain independent Python and C++ versions of the memory equations that can disagree. The planning request supplies intended role/mode, source/profile identities, feature/context limits and hard resource caps, not an already passing plan. Final `ds4ctl validate` checks that package, and startup enforces it again.

Any probe-relevant resource/security/image/device change requires rerunning the probe under the resulting settings. Calibration may use a separately admitted operator-fixture configuration before a full generation plan exists; it cannot require a successfully running full model to decide whether that model can be admitted. Probe, calibration, final admission and readiness are different outputs. The renderer must never fill unknown measurements with optimistic constants.

The external deployment lock references the completed image digest, source/build identity, verified config/plan digests, resolved Compose configuration and host report. The image's embedded build metadata does not include this external deployment lock or its own final image digest. Secrets are not hashed into public reports. Recreate a container when the external lock or its read-only config changes; do not pretend an already admitted process observes a new bind-mounted file atomically.

### 22.6 Docker memory, pinning and NUMA permissions

Application budgets, Docker limits and physical host headroom must be distinct. Generate Docker `mem_limit` from the complete process/cgroup peak, including application host pools, live file-cache charges, tmpfs, thread stacks, TLS/socket buffers and tested runtime margin. Keep it below the host's reservable physical capacity. The application's controlled pools must fit below that limit with room for non-pool overhead. A memory reservation is not a guarantee of available physical RAM. [R62]

For the strict no-swap deployment, set Compose `memswap_limit` to the **same positive byte value** as `mem_limit`; zero is not the no-swap setting. Validate effective cgroup v2 `memory.max`, `memory.swap.max`, `memory.current`, `memory.events` and pressure inside the container. Resolve the process's actual cgroup mount/relative path rather than hard-code the host path or trust `free`. Parent service/slice limits can be tighter and must be checked by host preflight. Do not disable the OOM killer or lower its protection to hide an under-sized plan. [R62][R30]

No-swap alone does not ensure that file-backed model pages remain resident. The required loaded arenas are prefaulted and locked with checked return values. Use a finite Docker `ulimits.memlock` soft/hard limit sufficient for the actual locked population and registration accounting, **including hundreds of GiB or multiple replicas when required**, not just a small transfer pool. `RLIMIT_MEMLOCK` and the application's smaller CUDA-pinned limit are different controls. Linux permits unprivileged `mlock` within its limit; the baseline does not require `CAP_IPC_LOCK` merely to bypass accounting. Test actual OS and CUDA registration behavior, alignment/rounding and unique-page accounting; do not assume both APIs charge identically. [R63]

Keep `cap_drop: [ALL]` and `no-new-privileges`. Docker's default seccomp profile blocks NUMA management syscalls that this implementation needs. Deliver `deploy/seccomp-numa.json` derived from the **pinned Engine default profile**, retaining its default deny policy and all unrelated restrictions. Add only reviewed calls used by the allocator/probe: `get_mempolicy`, `set_mempolicy`, and `mbind` for this process's own new arenas. The baseline binds before first touch, uses `mbind` with no page-migration flags, and needs no `MPOL_MF_MOVE_ALL`. Do not add broad `SYS_ADMIN`, `SYS_NICE`, `privileged`, or an unconfined profile to solve this issue. [R64][R65]

If page-location querying uses `move_pages`, allow only the self-query form: `pid=0`, `nodes=NULL`, and `flags=0`; constrain those scalar/pointer arguments in seccomp. It must not become cross-process inspection or a page migration service. [R78] Prefer `/proc/self/numa_maps` plus sampled self-query over copying model-sized metadata. The profile generator removes conflicting syscall rules for the explicitly replaced names, adds reviewed argument-constrained entries, validates the profile against the locked runtime, and retains a machine-readable diff from the default. Do not ship a three-line allow-all JSON profile and call it hardened.

Bind worker threads within the container's allowed CPU set and model arenas within its actual `Mems_allowed_list`. Compose `cpuset` restricts CPU IDs; it is not a memory-node allocator. Docker `run` supports `--cpuset-mems`, but do not invent a Compose `cpuset_mems` key. The baseline Compose deployment uses the application's verified NUMA binding inside the allowed set; an administrator may additionally constrain allowed memory nodes through the host runtime and must record that in the plan. A node restriction without enough per-node memory is a startup failure, not a reason to scatter a promised replica. [R66]

Preserve AppArmor/SELinux isolation. Required changes must be narrow and documented for the actual host; disabling the security module is not an installation step. File permissions/labels on model and secret binds must permit the service UID while remaining least-privilege. Profiling access such as unrestricted `perf_event_open`, host PID namespaces or `SYS_PTRACE` is not a production default; run optional privileged profiling only as a separately approved diagnostic and retain unprivileged timing/byte counters.

Use bounded tmpfs for sockets and small temporary runtime files. tmpfs and `/dev/shm` consume real memory and are included in cgroup accounting; they are not free space outside `mem_limit`. A tmpfs can otherwise be swapped, so it does not replace the no-swap policy. Do not put model banks or large private-state files in `/tmp`, `/dev/shm`, or writable image layers. The native single-process baseline uses a small private shared-memory allowance and does not need `ipc:host`. [R67]

### 22.7 Filesystems, credentials and persistence

Use explicit read-only binds for the server prepared model and each host's configuration and credential directory. The client binds only its verified compact package and intentionally populated caches—not the complete server checkpoint. Both may mount that different host-local directory as `/model`; the directory content/manifest role is what differs.

Model acquisition paths and bind sources live on the Docker **daemon host**, which may differ from the shell executing a remote Docker context. Missing bind sources must fail deployment (`create_host_path: false`), not become newly created empty directories. Use resolved absolute host paths and verify their expected object type. The runtime neither downloads weights nor repairs directories/permissions automatically. [R68]

Keep generated reports and permitted immutable preparation/cache artifacts in dedicated host volumes, outside the container writable layer. Live private state remains in admitted memory, not durable filesystem state. Runtime reports have size/retention limits and contain no raw prompts/activations by default. Treat the report volume as untrusted on subsequent reads; do not load executable code from it. Container deletion must not delete the source checkpoint or committed benchmark evidence.

Credentials are files, not Docker environment values or command arguments. Use a private directory with per-host CA/certificate/key in TLS mode and, on the client in either mode, an independently generated API token. The expert credential directory may be empty in trusted-network mode. Switching transport policy invalidates affected environment, registration and deployment evidence; plaintext reports record a null peer certificate digest and cannot claim mTLS verification. Service UID `10001` must be able to read the key, while other users cannot. Compose file-backed secrets do not reliably apply the declared `uid`, `gid` and `mode` attributes as remapped file ownership; provision actual host file ownership/modes and test access inside the non-root container. Never solve a failed key read with world-readable permissions. Local file-backed secrets are not an encrypted distributed secret store. [R69]

The service root filesystem is read-only. Supply only `/run/dwarfstar`, a bounded temporary directory and `/artifacts` as required writable locations. Do not mount the Docker socket, host `/`, entire home directory, host `/dev`, or host `/sys` read-write. GPU access is requested through the supported container runtime, not a wildcard privileged device mount. Pin the image digest, verify the release's trusted digest/signature when applicable, and preserve runtime notices.

### 22.8 Two-host Compose templates

These are **deployment templates for the implementation defined here**, not a claim that a runnable image already exists. The release includes these files plus a renderer that produces a validated `.env`, service configuration, memory plan, deployment lock and seccomp profile for each host. The template requires every deployment-specific value; it intentionally supplies no fake image digest, GPU UUID, model path, NUMA mask or multi-terabyte memory allowance.

All `_BYTES` environment values are decimal integer byte counts with no unit suffix. A `*_IMAGE` value is an immutable `registry/repository@sha256:<digest>` reference. `*_CPUSET` is an allowed CPU list/range string. `*_GPU_UUID` identifies exactly one qualified GPU; `device_index=0` is its ordinal **inside** the container. `*_START_PERIOD` and `*_STOP_GRACE` are explicit Compose durations generated from the startup/drain plan. Host source directories must already exist with correct ownership. `SECCOMP_PROFILE` is the profile path visible to the local deployment client.

**Expert host — `deploy/compose.expert.yaml`:**

```yaml
name: dwarfstar-expert
services:
  expert:
    image: "${EXPERT_IMAGE:?Set the digest-pinned CPU or CUDA image}"
    platform: linux/amd64
    runtime: runc
    entrypoint: ["/usr/local/bin/ds4-expert-server"]
    command: ["--config", "/etc/dwarfstar/service.json"]
    user: "10001:10001"
    init: true
    read_only: true
    restart: "no"
    stop_signal: SIGTERM
    stop_grace_period: "${EXPERT_STOP_GRACE:?Set the tested drain grace}"
    cap_drop: [ALL]
    security_opt:
      - no-new-privileges:true
      - "seccomp:${SECCOMP_PROFILE:?Set the reviewed NUMA seccomp profile}"
    cpuset: "${EXPERT_CPUSET:?Set allowed service and worker CPUs}"
    pids_limit: ${EXPERT_PIDS_LIMIT:?Set the tested thread and process bound}
    mem_limit: "${EXPERT_CGROUP_BYTES:?Set the complete cgroup memory limit}"
    memswap_limit: "${EXPERT_CGROUP_BYTES:?Set the same limit for no swap}"
    shm_size: "67108864"
    ulimits:
      memlock:
        soft: ${EXPERT_MEMLOCK_BYTES:?Set the finite locked-memory allowance}
        hard: ${EXPERT_MEMLOCK_BYTES:?Set the finite locked-memory allowance}
      core:
        soft: 0
        hard: 0
      nofile:
        soft: 4096
        hard: 4096
    tmpfs:
      - /run/dwarfstar:rw,nosuid,nodev,noexec,size=16777216,mode=0700,uid=10001,gid=10001
      - /tmp:rw,nosuid,nodev,noexec,size=67108864,mode=1777
    volumes:
      - type: bind
        source: "${EXPERT_MODEL_DIR:?Set the prepared model directory}"
        target: /model
        read_only: true
        bind:
          create_host_path: false
      - type: bind
        source: "${EXPERT_CONFIG_DIR:?Set the resolved config directory}"
        target: /etc/dwarfstar
        read_only: true
        bind:
          create_host_path: false
      - type: bind
        source: "${EXPERT_SECRET_DIR:?Set the private credential directory}"
        target: /run/secrets
        read_only: true
        bind:
          create_host_path: false
      - type: bind
        source: "${EXPERT_REPORT_DIR:?Set the report directory}"
        target: /artifacts
        bind:
          create_host_path: false
    ports:
      - "${EXPERT_BIND_IP:?Set the server private IPv4 address}:7443:7443/tcp"
      - "${EXPERT_BIND_IP:?Set the server private IPv4 address}:7444:7444/tcp"
    healthcheck:
      test: ["CMD", "/usr/local/bin/ds4ctl", "health", "--socket", "/run/dwarfstar/admin.sock", "--ready"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: "${EXPERT_START_PERIOD:?Set the measured model startup allowance}"
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"
```

The base expert file is the CPU deployment: CPU image, `executor=cpu`, no GPU request, and the normal `runc` runtime. For CUDA serving, select the CUDA image and CUDA configuration, then apply only this overlay; it changes the runtime/device request, not the mathematical service interface.

**Expert CUDA overlay — `deploy/compose.expert.cuda.yaml`:**

```yaml
services:
  expert:
    runtime: nvidia
    environment:
      NVIDIA_DRIVER_CAPABILITIES: compute,utility
      NVIDIA_VISIBLE_DEVICES: "${EXPERT_GPU_UUID:?Select exactly one server GPU}"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["${EXPERT_GPU_UUID:?Select exactly one server GPU}"]
              capabilities: [gpu]
```

**Client host — `deploy/compose.client.yaml`:**

```yaml
name: dwarfstar-client
services:
  client:
    image: "${CLIENT_IMAGE:?Set the digest-pinned CUDA image}"
    platform: linux/amd64
    runtime: nvidia
    entrypoint: ["/usr/local/bin/ds4-server"]
    command: ["--config", "/etc/dwarfstar/service.json"]
    user: "10001:10001"
    init: true
    read_only: true
    restart: "no"
    stop_signal: SIGTERM
    stop_grace_period: "${CLIENT_STOP_GRACE:?Set the tested drain grace}"
    cap_drop: [ALL]
    security_opt:
      - no-new-privileges:true
      - "seccomp:${SECCOMP_PROFILE:?Set the reviewed process seccomp profile}"
    environment:
      NVIDIA_DRIVER_CAPABILITIES: compute,utility
      NVIDIA_VISIBLE_DEVICES: "${CLIENT_GPU_UUID:?Select the RTX 5090 UUID}"
    cpuset: "${CLIENT_CPUSET:?Set allowed client service CPUs}"
    pids_limit: ${CLIENT_PIDS_LIMIT:?Set the tested thread and process bound}
    mem_limit: "${CLIENT_CGROUP_BYTES:?Set the complete cgroup memory limit}"
    memswap_limit: "${CLIENT_CGROUP_BYTES:?Set the same limit for no swap}"
    shm_size: "67108864"
    ulimits:
      memlock:
        soft: ${CLIENT_MEMLOCK_BYTES:?Set the finite locked-memory allowance}
        hard: ${CLIENT_MEMLOCK_BYTES:?Set the finite locked-memory allowance}
      core:
        soft: 0
        hard: 0
      nofile:
        soft: 4096
        hard: 4096
    tmpfs:
      - /run/dwarfstar:rw,nosuid,nodev,noexec,size=16777216,mode=0700,uid=10001,gid=10001
      - /tmp:rw,nosuid,nodev,noexec,size=67108864,mode=1777
    volumes:
      - type: bind
        source: "${CLIENT_MODEL_DIR:?Set the compact client package directory}"
        target: /model
        read_only: true
        bind:
          create_host_path: false
      - type: bind
        source: "${CLIENT_CONFIG_DIR:?Set the resolved config directory}"
        target: /etc/dwarfstar
        read_only: true
        bind:
          create_host_path: false
      - type: bind
        source: "${CLIENT_SECRET_DIR:?Set the private credential directory}"
        target: /run/secrets
        read_only: true
        bind:
          create_host_path: false
      - type: bind
        source: "${CLIENT_REPORT_DIR:?Set the report directory}"
        target: /artifacts
        bind:
          create_host_path: false
    ports:
      - "127.0.0.1:${CLIENT_API_PORT:?Set the host API port}:8000/tcp"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["${CLIENT_GPU_UUID:?Select the RTX 5090 UUID}"]
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "/usr/local/bin/ds4ctl", "health", "--socket", "/run/dwarfstar/admin.sock", "--ready"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: "${CLIENT_START_PERIOD:?Set the measured client startup allowance}"
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"
```

GPU reservation requires `capabilities: [gpu]`; do not combine `count` with `device_ids`. Exact UUID assignment and a matching `NVIDIA_VISIBLE_DEVICES` avoid inheriting a base-image “all GPUs” setting in the explicit NVIDIA-runtime path. Retain `compute,utility`, not graphics/video/display capabilities. Never set `NVIDIA_DISABLE_REQUIRE` to bypass driver checks. Container device exposure is not exclusive ownership or a VRAM quota: the administrator and application must prevent conflicting GPU workloads and recheck free VRAM before admission. [R70][R71]

These templates intentionally use no Swarm-only scheduling, host network/PID/IPC namespace, privileged device access, Docker socket, or kernel sysctl changes. Their CPU sets include both compute and progress threads. PIDs limits must count threads, health-check processes and startup helpers; too-small values can fail runtime initialization. Filesystem execute restrictions assume AOT native binaries; a declared JIT backend needs a separate controlled code-cache location and tests, not blanket writable/executable rootfs.

`restart: "no"` is the safe initial deployment policy: an incompatible model, unfit memory plan or permanent executor fault must not repeatedly reload a multi-terabyte bank. An operator may select a separately tested restart policy after readiness/failure handling is qualified. A Docker health failure by itself does not run the application's recovery logic or provide durable session resume; restart policies act on process exit. Keep health state, process restart and client session invalidation separate. [R72]

### Effective Compose configuration is the launched contract

Compose interpolation gives exported shell variables precedence over values in `--env-file`. Consequently, supplying the reviewed file alone does not prove the reviewed image, UUID, mount or memory limit was used. The release launcher must run Compose with a controlled allowlist of environment variables, explicitly unset deployment-variable names inherited from the shell, disable unintended `.env` discovery, and select its reviewed files in a fixed order. Do not source the environment file as executable shell code. [R86]

Render `docker compose ... config --format json` under that same controlled environment and compare its normalized resource/device/mount/port/security/entrypoint values with the finalized deployment contract before `up`. Use the same environment, Compose version and file order for launch; preserve the effective-config digest and verify actual `docker inspect` values after creation. Reject unreviewed override files and unresolved substitutions. `config --quiet` alone validates syntax, not equivalence to the approved security/resource policy. The renderer writes escaped literal values correctly for Compose's dollar/interpolation rules and tests paths containing spaces and dollar signs. [R86][R87]

`docker compose run` intentionally does not publish service ports by default, which is appropriate for probes; it must still receive the same tested GPU, user, cgroup, cpuset, mounts and security settings. No hardware probe needs an externally reachable inference listener. Validate effective container settings, not just the source YAML.

### 22.9 Host networking and API exposure

The client connects to a real routable expert numeric address. TLS verifies the separately configured certificate SAN; trusted-network mode requires no certificate or DNS identity. `localhost` inside the client container refers to that container, not the expert host; a server Compose service name is local to that Engine's network. Use private LAN routing and explicit TCP 7443/7444 publication on the expert host, with only the client host authorized. The templates illustrate IPv4 binds; dual-stack/IPv6 exposure needs explicit equivalent policy and tests rather than accidental additional listeners. [R56]

The runtime must not inherit the donor's optional disk KV cache, autonomous agent launcher, ignored sampling settings, or V4 model aliases into this target path. Requested unsupported sampling/reasoning settings must be rejected or explicitly normalized by the pinned native encoding contract and reported; silently ignoring them is not reproducible testing. Default client API publication is loopback only, for local use or an authenticated SSH tunnel. Broader exposure requires explicit authenticated HTTPS termination and network restrictions. The internal expert protocol, including mutual TLS when enabled is not a substitute for securing the public inference API. In TLS mode certificates are separate per peer and have correct server/client usage; production must not use sample self-signed trust-all settings or disable hostname verification.

Validate published-port restrictions in Docker's actual forwarding/firewall path. With Docker's iptables backend, an ordinary host INPUT/ufw rule may not govern the published-port path; use the appropriate Docker forwarding policy, including `DOCKER-USER` where applicable. For a qualified nftables setup use that backend's documented equivalent, not an assumed identical chain. TLS authenticates RPC peers; trusted-network TCP relies on the restricted network. Both modes enforce exact model grants, and the HTTP API still requires bearer authentication. Test from an unauthorized host. [R73]

Neither readiness nor startup depends on internet DNS/model repositories; only the configured numeric local peer addresses and, in TLS mode, provisioned PKI are needed. Runtime endpoints are numeric; a configured DNS SAN is compared as a certificate identity without performing name resolution. Runtime code must not discover peers by broadcasting, fetch remote image URLs, execute model-emitted tools, or launch downloaded shell commands.

The fork reuses `ds4-server`'s HTTP/SSE layer rather than importing a second model runtime. Initially qualify `GET /v1/models` and `POST /v1/chat/completions`; preserve/qualify `/v1/responses`, completions and messages adapters only when their native encoding and streaming fixtures pass. An unqualified endpoint returns a clear unsupported response; it must not silently translate a different model template. Report only `DeepSeek-V4.1-Flash` and declared profile identities, not upstream V4 Flash/Pro aliases that misleadingly identify another architecture. [R74]

The client API requires the file-provisioned bearer token even on loopback, bounds request bodies and queued work, rejects unknown/unsupported model or feature settings, validates token/context limits, and streams only committed step results. One generation is active; a second concurrent generation is rejected as busy by the initial zero-queue default. Stateless resent histories may reuse a proven identical live prefix, including image identity and reasoning/tool encoding, but may not attach one caller to another caller's mutable state. No durable cross-restart response IDs are promised.

Use input limits that account for encoded bytes, decoded image size, decompression ratio, final patch/token expansion, JSON nesting, message count and tool-schema size. Remote image-URL fetching is disabled to avoid a server-side request-forgery surface; data uploads are validated before GPU allocation. Endpoint body allowance is not the same as admitted semantic context. No CORS wildcard or tool-execution agent is enabled by default. Public protocol errors use clear 4xx/5xx responses and valid terminal SSE events; never emit a successful completed response after remote inference fails.

### User-facing API contract and bounded streaming

Deliver a strict supported subset of the inherited API rather than promising compatibility with every request accepted by another engine. `GET /v1/models` lists the loaded target/profile identity. For `POST /v1/chat/completions`, require a known model ID and a validated nonempty message list; permit one completion (`n=1`), explicit Boolean `stream`, bounded output-token limit, and only the sampling/reasoning/tool/image fields implemented by the pinned native adapter. Reject conflicting token-limit aliases, unsupported logit processors, requested probabilities or structured-output modes instead of ignoring them. Record defaults, seed/RNG algorithm, temperature/greedy behavior, top-p/tie handling and all admitted settings in the request's safe execution metadata.

Set explicit header/body parse deadlines, maximum header bytes and JSON depth, total encoded/decoded image budgets, and a bounded SSE write deadline. Verify bearer authorization before expensive decoding or allocation. A disconnected or persistently stalled API consumer abandons its generation and initiates the same safe cancellation path as a local request; it must not leave a hidden generation running indefinitely. The one active-generation slot is not released until its abandoned work can no longer access mutable buffers.

On success, emit the adapter's declared terminal finish reason and usage once, followed by its normal end marker. Count source-native prompt positions, generated tokens and any image expansion according to a documented usage schema; do not equate visible characters with tokens or count an emitted but unincorporated final token incorrectly. On a failure before streaming, return a typed HTTP error. After streaming begins, emit a bounded error event where the protocol permits and close **without a successful completion event**; already emitted tokens are not silently regenerated. Define fixtures for UTF-8 splits, tool argument fragments, EOS/stop strings, `length` termination, cancellation and remote faults.

`0.0.0.0` inside the container is needed for the declared bridge publication; host loopback publication restricts the normal host-facing endpoint, not all possible container-to-container paths. API bearer verification therefore remains mandatory, and a broader deployment uses explicit HTTPS and forwarding restrictions. No remote image fetch, model-emitted tool execution or autonomous agent mode is enabled.

### 22.10 Startup, preflight and orderly shutdown

The release delivers the two-stage `tools/render_deployment.py` contract in Section 22.5, or an equivalently small native implementation. Bootstrap needs intended restrictions, not a completed model-admission plan. Finalize requires the probe, immutable inventory and measured calibration before writing the final configuration/plan/deployment lock. It must reject unresolved placeholders, unreadable secrets, invalid GPU UUIDs, insufficient memory, unknown kernel/profile combinations and CPU/CUDA image mismatches. It is an operator tool, not an online adaptive resource broker.

The operational sequence is:

1. Provision the host, model artifacts, directories, finite memory/pinning allowances, seccomp profile, credentials and any runtime prerequisites as administrator-controlled deployment inputs. Verify artifact/image hashes before serving; do not trust a basename or tag.
2. Render bootstrap files, then probe from a one-shot container with the intended production **user, cpuset, memory, seccomp, GPU and mounts**, using `probe.json` and no final memory-plan prerequisite. Record limits, effective NUMA calls, self page placement, lock/registration probes, required small compiled kernels and transfer results. A privileged host-only benchmark cannot substitute for this check.
3. Finalize planning from the probe/inventory/calibration, validate the final service package, and use `docker compose config --quiet` plus normalized effective-config comparison on each host and selected override. Rerun the probe if finalize changed its environment. Start the locally ready expert, then bind/bootstrap the client. Expert readiness does not depend on client readiness; there is no cross-host `depends_on`.
4. Bind identities/mode, run operator and target-model smoke tests, then the applicable acceptance suite. Save configuration hashes and container image IDs with every report.
5. Admit production requests only after the model/profile memory plan, peer binding and GPU/CPU executor tests are ready. A cold optional cache is acceptable; unavailable required weights/state or a deferred failing kernel is not.

Example launch commands, **after the implementation, bootstrap probe, finalize step and host-specific files exist**, run from the corresponding host's deployment directory. The probe repeated here verifies the final environment; it is not the first discovery step. In `compose run`, `expert` or `client` is the Compose service name, while `validate`/`probe` is the command passed to the overridden `ds4ctl` entry point. These command expansions are executed by the release launcher under the controlled environment/effective-config checks above, not an arbitrary inherited shell:

```bash
# CPU expert host. expert.env selects the CPU image and CPU service configuration.
docker compose --env-file expert.env -f compose.expert.yaml config --quiet
docker compose --env-file expert.env -f compose.expert.yaml run --rm --no-deps \
  --entrypoint /usr/local/bin/ds4ctl expert validate --config /etc/dwarfstar/service.json
docker compose --env-file expert.env -f compose.expert.yaml run --rm --no-deps \
  --entrypoint /usr/local/bin/ds4ctl expert probe --probe-config /etc/dwarfstar/probe.json \
  --output /artifacts/container-probe.json
docker compose --env-file expert.env -f compose.expert.yaml up -d

# CUDA expert alternative. Use an env/config selecting the CUDA image and executor.
docker compose --env-file expert-cuda.env -f compose.expert.yaml \
  -f compose.expert.cuda.yaml config --quiet
docker compose --env-file expert-cuda.env -f compose.expert.yaml \
  -f compose.expert.cuda.yaml run --rm --no-deps --entrypoint /usr/local/bin/ds4ctl \
  expert probe --probe-config /etc/dwarfstar/probe.json --output /artifacts/container-probe.json
docker compose --env-file expert-cuda.env -f compose.expert.yaml \
  -f compose.expert.cuda.yaml up -d

# Client host, after the selected expert service is ready.
docker compose --env-file client.env -f compose.client.yaml config --quiet
docker compose --env-file client.env -f compose.client.yaml run --rm --no-deps \
  --entrypoint /usr/local/bin/ds4ctl client validate --config /etc/dwarfstar/service.json
docker compose --env-file client.env -f compose.client.yaml run --rm --no-deps \
  --entrypoint /usr/local/bin/ds4ctl client probe --probe-config /etc/dwarfstar/probe.json \
  --output /artifacts/container-probe.json
docker compose --env-file client.env -f compose.client.yaml up -d
```

Do not run both expert alternatives concurrently or run the probe alongside an active memory-saturating model unless its separate resources have been reserved. The renderer emits only the selected server deployment. Before changing server mode/image, drain the client and expert, stop the old service, verify RAM/VRAM release, then start and bind the new service. No in-place mode mutation or speculative automatic rollback during a generation is implemented.

Startup exposes progress stages: verify/configure, inventory, read/hash, convert/repack if explicitly pre-authorized, allocate/place, prefault/lock, initialize kernels, bind peer, ready. Production should normally use already prepared artifacts; unbounded conversion at startup is not allowed. A large model can take much longer to load than a small default health period. Configure `start_period` and deployment health wait from measured startup, and distinguish slow progress from a deadlock. A health command is small and does not repeatedly lock or scan all weights.

On `SIGTERM`, stop new admission, mark readiness false, cancel or finish the current bounded unit according to its committed-state contract, stop application of abandoned replies, quiesce GPU DMA/CPU workers, close peer connections, flush safe reports, and exit. `init: true` handles signal forwarding/reaping; application entry points use exec-form commands, not a shell that swallows signals. Configure a finite stop grace that covers the tested drainage path; Docker may send SIGKILL when it expires. No correctness claim depends on shutdown completing after SIGKILL. [R75]

On OOM, GPU reset/access loss, executor fault or peer restart, fail the affected generation and discard uncertain client state. Do not resume an old epoch merely because Compose starts a process with the same name. NVIDIA documents a GPU-access issue associated with systemd daemon reload on some Container Toolkit configurations; include host maintenance/recreation checks and fail closed if GPU access disappears. This is not a reason to disable cgroups. [R57]

### 22.11 Containerized acceptance and release evidence

All required model/profile/server-mode/client-memory combinations must pass **inside the release images**. Native test results are useful references, not substitutes for the actual deployment. Every run records image digest, service config/plan digests, GPU UUID/SM, driver/toolkit, Docker/Compose/kernel, allowed CPUs/memory nodes, seccomp digest, effective cgroup/pinning limits, and actual artifact identity.

| Area | Mandatory container test |
|---|---|
| Build/reproducibility | Resolve only locked sources; build CPU and CUDA targets without a GPU build dependency; record reproducible inputs and compare outputs/provenance. Explain unavoidable nondeterministic metadata rather than claim bit-reproducibility without testing. |
| CPU independence | CPU runtime starts and serves target expert fixtures on a GPU-free host with the actual CPU ISA; dependency scan shows no CUDA/GPU framework requirement. |
| GPU identity | Expose exactly the requested UUID, verify SM120 client kernels and separately qualified server SM, deny unsupported devices/profiles before model loading. |
| Non-root/security | Production UID can read only authorized artifacts/secrets and write only declared paths; no privileged mode, unrestricted syscalls, Docker socket or unintended host mounts. |
| NUMA | Allocation and sampled locality probes succeed with reviewed seccomp; default-profile denial is a negative test; replicas are real local pages within effective node limits. |
| Memory | Finite memlock and cgroup limits match the resolved plan; enforce no swap, verify locked model/state pages, account tmpfs/pinned/driver overhead and reject under-sized limits. |
| Client tiers | Force actual host-backed weight/state migrations and expert host hits in the release container; zero client CPU neural kernels and no disk/remote-state overflow. |
| Network | Test two physical hosts, not localhost-only; verify both supported transport policies. TLS tests SAN/client authorization and wrong/expired credential rejection; trusted-network tests need no credential files and must reject mixed-mode peers and mismatched control/bulk IPs. Both test exact model grants, one-use bulk binding, unauthorized-host firewall denial and bounded timeout/failure behavior. |
| Modes | CPU/CUDA server results match the declared realization tolerance; no automatic device substitution; mode changes require a new bound session. |
| Readiness | Process existence or `nvidia-smi` is insufficient; failed model hash, unavailable peer, unsupported kernel, residency failure and incomplete startup never become ready. |
| Pressure/faults | Exercise cgroup OOM, pin failure, denied NUMA call, GPU access loss, broken pipe, stalled receiver, SIGTERM/SIGKILL and canceled dirty writeback; no committed output from incomplete math. |
| Durability | Recreate/remove containers and verify checkpoint, prepared manifest and reports remain intact; private session state is not falsely resumed. |
| Performance | Compare native and containerized matched workloads; attribute any difference to resource/NUMA/network settings. Repeat the required soak under the production memory/security policy. |
| Distribution hygiene | Inspect image layers and logs for weights, secrets, prompts and unwanted build tools; retain licenses, SBOM, trusted image references, schema/tests and operating runbook. |

Automated structural checks must parse YAML/JSON, reject duplicate keys, validate schemas, render Compose with all three deployment combinations, lint Dockerfiles, run shell syntax checks, verify Markdown/reference anchors, and assert dangerous settings are absent. Actual `docker compose config`, image builds and container execution are separate checks and cannot be replaced by a YAML parser. Performance thresholds, exact device/library pins and physical memory feasibility are recorded acceptance inputs/results, not facts manufactured in this handoff.

The effective-config, two-stage probe, crash-confidentiality and protocol-lifecycle cases above belong to the release image tests. Host administrators supply the real physical resource and secret inputs; the renderer validates them rather than inventing them. Static schema parsing and pure protocol fixtures can run without Docker or a GPU, but cannot attest that Docker accepted a seccomp profile, that a driver accepted registration, or that the target kernels executed.

### 22.12 Implementation-completion checklist for Docker

The Docker deliverable is complete only when the integrated repository includes both image recipes and reproducible image/base digests; packaging targets and native entry points; strict config/plan/protocol schemas; the renderer and hardware probe; reviewed seccomp source/diff; per-host Compose examples and mode selection; secret/mount provisioning instructions; readiness/drain/error behavior; container-aware capacity accounting; CPU/GPU/NUMA/security/fault tests; qualified model/profile matrix; and retained evidence from the physical two-host deployment.

Remaining inputs that **must be supplied by implementation or hardware discovery** are explicit: coherent dependency commits and tested base digests; actual CPU ISA and NUMA capacities; negotiated GPU links and usable VRAM; network addresses and, when TLS is selected, certificates; measured context/resource/timeout budgets; independent operator/profile error thresholds; and target benchmark results. None is replaced with an invented pin, a performance promise, or a lossy smaller model. The model/architecture/mode/memory-policy requirements themselves are settled and do not need to be reopened to implement these inputs.


### 22.13 Required schema and conformance artifacts

The repository must contain strict machine-readable schemas for the compact root/manifest, operator contract, physical layout, service configuration, probe configuration, deployment request/lock, resolved memory plan, calibration report and protocol messages. Cross-field validators enforce the equations and relationships in this document; JSON type validation alone is not sufficient. Unknown required tensor/operator variants are errors, not arbitrary executable extensions.

The conformance suite must generate CPU/CUDA service packages from the same logical fixture and extract every embedded template into the actual deployment files without manual transcription. Validate both expert modes and the client; snapshot the normalized configurations and compare them with container inspection. Pure fixtures cover framing, canonical identities, binding/cancellation, resource-accounting equations and exact deterministic bookkeeping. Native CPU reference tests, compiled CUDA tests, Docker execution and physical-host model tests remain separate named test targets with separate results.

Acceptance thresholds for floating operators and full-model quality must be specified in a reviewed `validation-policy.json` **before** optimized-result inspection. It identifies the oracle, actual corpus/positions, profile, absolute/relative error rules including near-zero handling, loss/quality criteria, determinism settings and a justified threshold for each measured quantity. There are no fabricated universal tolerances for every quantized operator. A missing required policy or unexecuted target test prevents a passing release report; an engineer cannot substitute an arbitrary tolerance or a favorable sample to complete the checklist.

Outstanding hardware-derived inputs are intentionally explicit: actual CPU ISA/NUMA capacities, coherent tested GPU/toolchain libraries and image digests, real host-memory and PCIe limits, peer credentials/addresses, admitted context/resources/deadlines, and measured latency/quality results. The specification fully fixes their **required evidence and validation path**, not their unknowable values before implementation. No final-review statement can replace those measurements.

### 22.14 Document-review validation record

The final review ran **155 document/static-contract assertions**, including reference and anchor resolution, unique headings, fenced-block structure, duplicate-aware JSON/YAML parsing, shell syntax, the three rendered deployment-template combinations, security/resource-policy invariants, fixed binary field sizes/offsets, exact payload equations, wide-integer arithmetic, canonical-JSON reference fixtures, and cancellation/credit state fixtures. It additionally compared the specified stable attention-block merge with a direct numerical reference in **200 deterministic cases**, including empty blocks and denominator-only sinks.

These are **specification/conformance-fixture checks**, not the shipped runtime. The YAML fixture renderer/merger is not `docker compose config`; the small canonicalization reference is not certification of an as-yet-unselected production JSON library; the numerical recurrence checks are not DeepSeek model, CUDA or quantization validation. Docker, `nvcc`, a target GPU and the complete checkpoint were unavailable for executable deployment/model tests in this review environment. No image build, container startup, TLS integration, GPU operation, physical NUMA test or inference benchmark is claimed.

Implementation must turn these contracts into the actual repository's tests and run the release matrix above. Every observed result must identify its exact source/artifact/build, input and environment. Claims of correctness, capacity and performance remain distinct; documentation completeness does not substitute for those execution results.

## 23. Source register

**Audit context:** October 3, 2026 requirements; final source/contract review. Rechecked evidence includes target configuration and numerical references, publisher conversion and cited runtime integration, FreeToken memory policies, accelerated SM120 requirements, rendered report state semantics, CPU/NUMA principles, canonical identity and Docker deployment authorities. The video entry records the earlier transcript/visual audit; it was not replayed for this document pass. These links identify primary evidence and implementation leads. Branch links are not immutable release pins. The source-lock milestone must replace consumed moving references with full commit/artifact identities and preserve the exact imported files. Video observations derive from the earlier retrieved transcript and timestamped visual audit; no benchmark reproduction is claimed.

### Model, graph, numerical, and hardware authorities

| Reference | Source and use |
|---|---|
| [R10] | DeepSeek-V4.1-Flash model card, artifact identity, encoding and feature guidance. |
| [R11] | DeepSeek released inference configuration; target dimensions, layer-source maps, expert and Engram parameters. |
| [R12] | DeepSeek inference `model.py`; linear/quantization dispatch, router, clamped expert, reduction, attention and mHC semantics. |
| [R13] | DeepSeek generation reference; native encoding imports, prompt processing and generation integration. |
| [R14] | NVIDIA DeepSeek-V4.1-Flash-NVFP4 model card; mixed conversion, W4A4, calibration, artifact footprint, publisher qualification. |
| [R17] | DeepSeek FlashMLA README; V4.1 sparse attention, packed layouts, supported GPU/toolkit requirements and API changes. |
| [R25] | NVIDIA Transformer Engine NVFP4 documentation; numeric values/scales and representation context. |
| [R26] | NVIDIA TensorRT quantization schemes; distinguish precision and scaling contracts. |
| [R27] | DeepSeek-V4.1-Flash technical report, especially architecture and Section 3.2.2; bounded replay's explicit approximation. Architecture and replay pages were inspected as rendered PDF pages. |
| [R28] | NVIDIA CUDA GPU compute-capability catalogue; exact GPU family/SM qualification. |
| [R29] | Linux NUMA memory-policy documentation; task/VMA/shared-page allocation and locality caveats. |
| [R30] | Linux cgroup v2 documentation; scoped memory, swap, pressure and resource controls. |
| [R31] | NVIDIA RTX 5090 specification; device memory and PCIe capability, not host-link measured throughput. |
| [R32] | NVIDIA CUDA Best Practices Guide; explicit memory transfers, pinned resources, overlap, numerical testing and profiling. |
| [R42] | DeepSeek inference `engram.py`; table layout and token-history/hash state implementation. |

### Runtime and kernel implementation sources

| Reference | Source and use |
|---|---|
| [R01] | User-selected `stefandsl/DwarfStar` repository. |
| [R02] | Related `antirez/ds4` README; DwarfStar evolution and documented V4.1 execution scope. |
| [R03] | FreeToken repository; project scope and reusable execution/memory mechanisms. |
| [R04] | Selected DwarfStar CUDA source; existing packed operations, expert cache, model-map and allocation assumptions to audit. |
| [R05] | Selected DwarfStar GPU interface; model-offset and operator boundary material. |
| [R06] | FreeToken CLI; explicit execution strategies, CPU-format restrictions, expert backends and bandwidth calibration controls. |
| [R07] | FreeToken model documentation; supported configurations and strategy semantics. |
| [R08] | FreeToken PR #460; experimental V4.1 status and author-reported qualification limits. |
| [R09] | PR branch V4.1 document; eager execution, clamping/backend restrictions, community artifact, packed state and prefill assumptions. |
| [R18] | FlashInfer SM120 sparse-MLA support table; instantiated configurations to verify against V4.1. |
| [R19] | FlashInfer consumer-Blackwell fused-MoE interface; candidate expert kernel, not automatic strict-profile compatibility. |
| [R20] | FlashInfer MXFP8 GEMM interface; backend/shape/scaling qualification. |
| [R21] | FreeToken host-bank code; file-backed filling, residency, CUDA registration and CPU-route assumptions. |
| [R22] | FreeToken offload-cache code; schemas, expert slots, staging, cache mechanics. |
| [R23] | FreeToken CPU executor; implementation lead, not proof of required target formats/ISA. |
| [R24] | FreeToken bandwidth calibration implementation; representative CPU-versus-transfer measurements. |
| [R40] | FreeToken license; verify notices and obligations at the consumed commit. |
| [R41] | Related DwarfStar donor license; selected base and all additional files still require their own license verification. |
| [R43] | vLLM Engram documentation; retrieval/fusion separation and host placement precedent. |
| [R44] | Selected DwarfStar whole-layer distribution source; networking material but not the required expert-service contract. |
| [R45] | NVIDIA CUTLASS; supported numerical types, architecture-specific implementations and candidate kernels. |
| [R46] | FreeToken NVFP4 implementation; inspect exact activation/scaling semantics before reuse. |
| [R47] | FreeToken expert-bank schema and registry implementation. |
| [R48] | FreeToken offload kernels; bounded transfer/slot execution material. |
| [R49] | FreeToken package configuration; dependency snapshot requiring a coherent source/build lock, not blanket adoption. |
| [R50] | Selected DwarfStar license page; exact imported content must be retained with its notices. |

### Disaggregation evidence and supporting research

| Reference | Source and relevance |
|---|---|
| [R15] | User-supplied Chris Hay video. Localhost versus WAN findings establish a service split, not target hardware speed. |
| [R16] | LARQL CLI; sequential versus layer predispatch/refinement and its accuracy distinction. |
| [R33] | Fiddler; CPU/GPU execution placement and transfer-cost reasoning for MoE. |
| [R34] | FreeToken research paper; bandwidth-aware edge serving concepts, not a mandate for automatic hybrid policy here. |
| [R35] | KT-Kernel documentation; CPU/NUMA kernels and their ISA/format requirements. |
| [R36] | FlexGen; memory-hierarchy/offload precedent, not a single-session latency guarantee. |
| [R37] | FlashAttention; exact I/O-aware attention tiling precedent, not a ready-made V4.1 backend. |
| [R38] | PagedAttention; block-based state-management precedent, not proof of V4.1 host paging. |
| [R39] | MegaScale-Infer; attention/expert disaggregation at a different hardware/scale regime. |
| [R51] | LARQL distributed FFN documentation; actual inputs/results boundary and dependency cautions. |
| [R52] | LARQL development plan; CPU portability, numerical and experiment distinctions. |

### Additional audit and Docker implementation authorities

| Reference | Source and use |
|---|---|
| [R53] | Native reference quantization kernels, including activation grouping/rounding and mixed FP4/FP8 arithmetic. |
| [R54] | Safetensors format and parser constraints; header-relative data offsets and defensive loading. |
| [R55] | DeepSeek recipe native conversation/encoding and response conversion; no inference or tool executor. |
| [R56] | Docker Compose networking; host-local service discovery and cross-host routing distinction. |
| [R57] | NVIDIA Container Toolkit installation and Docker runtime configuration; host driver and GPU-access caveat. |
| [R58] | CUDA minor-version compatibility and its feature/PTX limitations; not full application qualification. |
| [R59] | Docker build practices, immutable image inputs and context hygiene. |
| [R60] | Docker multi-stage builds and separation of build/runtime contents. |
| [R61] | BuildKit secret mounts; avoid secret build arguments and image layers. |
| [R62] | Docker memory, swap, CPU and OOM controls; effective limits and no-swap semantics. |
| [R63] | Linux mlock, unprivileged resource limits and page-residency behavior. |
| [R64] | Docker seccomp allowlist and blocked NUMA operations. |
| [R65] | Linux mbind policy and migration permissions; own pre-fault allocation versus privileged migration. |
| [R66] | Docker run CPU/memory-node affinity and cpuset-mems semantics. |
| [R67] | Docker tmpfs accounting, lifetime and potential swap behavior. |
| [R68] | Docker bind mount locality and read-only host data. |
| [R69] | Compose file-backed secrets and ownership/mode qualifications. |
| [R70] | Docker Compose GPU device requests, capabilities and device selection. |
| [R71] | NVIDIA container visible devices, driver capabilities and requirement checks. |
| [R72] | Docker restart-policy behavior; process exit is distinct from application recovery. |
| [R73] | Docker iptables forwarding and DOCKER-USER controls for published ports. |
| [R74] | Selected DwarfStar HTTP endpoint/streaming implementation scope; target adapters require requalification. |
| [R75] | Compose service schema, health checks, init, user, resource settings and termination grace. |
| [R76] | TLS 1.3 handshake, authentication and early-data replay considerations. |
| [R77] | Dockerfile exec-form entry points, runtime user and build instruction semantics. |
| [R78] | Linux move_pages self-query and migration distinction. |
| [R79] | Publisher model API: observed source revision, artifact inventory and chat-template presence; metadata evidence, not a verified checkpoint download. |

### Final contract-verification authorities

| Reference | Source and use |
|---|---|
| [R80] | NVIDIA PTX ISA: accelerated SM120 block-scaled MMA instruction targets, distinct from device compute capability. |
| [R81] | RFC 8785 JSON Canonicalization Scheme: deterministic UTF-8 identity, Unicode/number constraints, and string encoding for larger integers. |
| [R82] | Linux PR_SET_DUMPABLE: service-scoped prevention of process memory dumps. |
| [R83] | Linux core dump handling: piped core collectors may ignore RLIMIT_CORE; confidentiality requires more than a zero file-size limit. |
| [R84] | NVIDIA target mixed-quantization configuration: per-module NVFP4 descriptors coexist with source FP8 metadata. |
| [R85] | Publisher-cited SGLang ModelOpt integration at immutable commit: activation-scale reduction domains, separate gate/up factors and backend differences; inspected code, not a reproduced benchmark. |
| [R86] | Docker Compose variable interpolation and shell/environment-file precedence. |
| [R87] | Docker Compose config: resolved application model, JSON rendering and validation commands. |

[R01]: https://github.com/stefandsl/DwarfStar
[R02]: https://raw.githubusercontent.com/antirez/ds4/main/README.md
[R03]: https://github.com/FlashML-org/FreeToken
[R04]: https://raw.githubusercontent.com/stefandsl/DwarfStar/main/ds4_cuda.cu
[R05]: https://raw.githubusercontent.com/stefandsl/DwarfStar/main/ds4_gpu.h
[R06]: https://raw.githubusercontent.com/FlashML-org/FreeToken/main/docs/cli.md
[R07]: https://raw.githubusercontent.com/FlashML-org/FreeToken/main/docs/models.md
[R08]: https://github.com/FlashML-org/FreeToken/pull/460
[R09]: https://raw.githubusercontent.com/ArqAlice/FreeToken/feat/deepseek-v4_1-flash/docs/deepseek-v41.md
[R10]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
[R11]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/inference/config.json
[R12]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/inference/model.py
[R13]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/inference/generate.py
[R14]: https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-NVFP4
[R15]: https://www.youtube.com/watch?v=1jGR4zqpyKA
[R16]: https://raw.githubusercontent.com/chrishayuk/larql/main/docs/cli.md
[R17]: https://raw.githubusercontent.com/deepseek-ai/FlashMLA/main/README.md
[R18]: https://docs.flashinfer.ai/generated/flashinfer.mla.supported_sparse_mla_sm120_configs.html
[R19]: https://docs.flashinfer.ai/generated/flashinfer.fused_moe.b12x_fused_moe.html
[R20]: https://docs.flashinfer.ai/generated/flashinfer.gemm.mm_mxfp8.html
[R21]: https://raw.githubusercontent.com/FlashML-org/FreeToken/main/python/freetoken/moe/host_banks.py
[R22]: https://raw.githubusercontent.com/FlashML-org/FreeToken/main/python/freetoken/moe/offload_cache.py
[R23]: https://raw.githubusercontent.com/FlashML-org/FreeToken/main/python/freetoken/moe/cpu_executor.py
[R24]: https://raw.githubusercontent.com/FlashML-org/FreeToken/main/python/freetoken/moe/benchbw.py
[R25]: https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html
[R26]: https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html
[R27]: https://arxiv.org/pdf/2609.19969
[R28]: https://developer.nvidia.com/cuda/gpus
[R29]: https://www.kernel.org/doc/html/latest/admin-guide/mm/numa_memory_policy.html
[R30]: https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html
[R31]: https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/
[R32]: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
[R33]: https://arxiv.org/abs/2402.07033
[R34]: https://arxiv.org/abs/2608.16157
[R35]: https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md
[R36]: https://arxiv.org/abs/2303.06865
[R37]: https://arxiv.org/abs/2205.14135
[R38]: https://arxiv.org/abs/2309.06180
[R39]: https://arxiv.org/abs/2504.02263
[R40]: https://raw.githubusercontent.com/FlashML-org/FreeToken/main/LICENSE
[R41]: https://raw.githubusercontent.com/antirez/ds4/main/LICENSE
[R42]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/inference/engram.py
[R43]: https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/engram.md
[R44]: https://raw.githubusercontent.com/stefandsl/DwarfStar/main/ds4_distributed.c
[R45]: https://github.com/NVIDIA/cutlass
[R46]: https://raw.githubusercontent.com/FlashML-org/FreeToken/main/python/freetoken/moe/fused_nvfp4.py
[R47]: https://raw.githubusercontent.com/FlashML-org/FreeToken/main/python/freetoken/moe/expert_banks.py
[R48]: https://raw.githubusercontent.com/FlashML-org/FreeToken/main/python/freetoken/moe/offload_kernels.py
[R49]: https://raw.githubusercontent.com/FlashML-org/FreeToken/main/pyproject.toml
[R50]: https://github.com/stefandsl/DwarfStar/blob/main/LICENSE
[R51]: https://github.com/chrishayuk/larql/blob/main/docs/ffn/distributed.md
[R52]: https://raw.githubusercontent.com/chrishayuk/larql/main/docs/dec-funnel.md

[R53]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/inference/kernel.py
[R54]: https://github.com/safetensors/safetensors#format
[R55]: https://github.com/deepseek-ai/deepseek-recipe/blob/main/README.md
[R56]: https://docs.docker.com/compose/how-tos/networking/
[R57]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
[R58]: https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html
[R59]: https://docs.docker.com/build/building/best-practices/
[R60]: https://docs.docker.com/build/building/multi-stage/
[R61]: https://docs.docker.com/build/building/secrets/
[R62]: https://docs.docker.com/engine/containers/resource_constraints/
[R63]: https://man7.org/linux/man-pages/man2/mlock.2.html
[R64]: https://docs.docker.com/engine/security/seccomp/
[R65]: https://man7.org/linux/man-pages/man2/mbind.2.html
[R66]: https://docs.docker.com/engine/containers/run/
[R67]: https://docs.docker.com/engine/storage/tmpfs/
[R68]: https://docs.docker.com/engine/storage/bind-mounts/
[R69]: https://docs.docker.com/reference/compose-file/services/#secrets
[R70]: https://docs.docker.com/compose/how-tos/gpu-support/
[R71]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html
[R72]: https://docs.docker.com/engine/containers/start-containers-automatically/
[R73]: https://docs.docker.com/engine/network/firewall-iptables/
[R74]: https://raw.githubusercontent.com/stefandsl/DwarfStar/main/README.md
[R75]: https://docs.docker.com/reference/compose-file/services/
[R76]: https://www.rfc-editor.org/rfc/rfc8446.html
[R77]: https://docs.docker.com/reference/dockerfile/
[R78]: https://man7.org/linux/man-pages/man2/move_pages.2.html

[R79]: https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4.1-Flash

[R80]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
[R81]: https://www.rfc-editor.org/rfc/rfc8785.html
[R82]: https://man7.org/linux/man-pages/man2/PR_SET_DUMPABLE.2const.html
[R83]: https://man7.org/linux/man-pages/man5/core.5.html
[R84]: https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-NVFP4/raw/main/config.json
[R85]: https://raw.githubusercontent.com/sgl-project/sglang/da64c5cbb8cf6bfd39be19da43573fdfd484c43a/python/sglang/srt/layers/quantization/modelopt_quant.py
[R86]: https://docs.docker.com/compose/how-tos/environment-variables/variable-interpolation/
[R87]: https://docs.docker.com/reference/cli/docker/compose/config/
