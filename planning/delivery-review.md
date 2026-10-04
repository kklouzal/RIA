# Delivery, operations, and release review

This is a planning handoff. No runtime, deployment image, schema implementation, or model qualification is claimed.

## Review provenance and governing decisions

- Governing local instructions: `/home/kklouzal/AGENTS.md`, read completely. No deeper `AGENTS.md` existed under `RIA` at review time.
- Primary specification: `/home/kklouzal/DwarfStar-Remote-Inference-Architecture.md`, SHA-256 `14224cdb33476944111e14f69a5679f0597c192a44d67b48f048910f326f3f6e`.
- Assigned range consumed completely: lines **1235–2133**, covering §18.8, §§19–22, and the complete source register.
- Additional ranges consumed for the referenced acceptance/lifecycle contracts: **1033–1099** and **1100–1234**. Heading/reference searches were navigation aids, not substitutes for those reads.
- The latest user correction makes **`antirez/ds4` the canonical base**. That supersedes the handoff's `stefandsl/DwarfStar` selection, including affected §19/§20 and R01/R04/R05/R44/R50 references. Separately imported code needs its own identity/license lock. The parent materialized `kklouzal/RIA` with `origin=https://github.com/kklouzal/RIA.git`, `upstream=https://github.com/antirez/ds4.git`, and initial checked-out commit **`0aaea5a238fb41a35106a551e73c8409dfb751ac`**, verified locally by this reviewer. Final coherent dependency/build/model release pins remain D0 work.
- The primary planner verified these current `antirez/ds4` paths: `ds4.c`, `ds4.h`, `ds4_cuda.cu`, `ds4_gpu.h`, `ds4_deepseek41_cuda.cuh`, `ds4_deepseek41_gpu.h`, `ds4_engram.c/h`, `ds4_image.c/h`, `ds4_prompt_prefix.c/h`, `ds4_linux_memory.h`, `ds4_server.c`, `ds4_cli.c`, `ds4_eval.c`, `Makefile`, `AGENT.md`, `CONTRIBUTING.md`, and `QA_BEFORE_RELEASES.md`. This reviewer read `AGENT.md` and `CONTRIBUTING.md` completely after checkout. Existing text CUDA, GGUF, or disk-backed Engram behavior is a reuse lead, not proof of this contract.
- `RIA` initially contained no implementation files and was not a Git repository; the base checkout arrived during the review. The named packaging targets and control interfaces in the specification are required future outputs, not commands that have passed here. No implementation code was changed by this review.

The base's native C requirement remains applicable: keep one native C admission engine and do not introduce C++. User requirements for production CPU expert serving and strict admitted RAM/state override the donor's reference-only CPU and disk-backed target defaults. Preserve unrelated donor backends and run their relevant documented regression checks when actual changes affect them. `make`, `make test`, `make cpu`, `make cuda-regression`, the `ds4_test` targeted checks, and matched `ds4-bench` CSVs are discovered donor commands; their platform/model prerequisites must be honored and their V4-specific fixtures cannot certify V4.1. No code change here requires running those inference builds/tests.

The settled architecture remains one CUDA attention client and one expert/Engram server on separate physical hosts; both CPU-only and Blackwell CUDA server modes; three numerical profiles; exact model/state behavior; admitted RAM/VRAM/pinning; no client CPU neural fallback, CUDA-server CPU expert fallback, routine network expert-weight miss service, or private-state disk/remote overflow. Unknown measurements, coherent pins, quality thresholds, and capacity are implementation inputs. They do not reopen those requirements.

## Dependency-aware implementation plan

| ID | Deliverable and work boundary | Required predecessors | Exit evidence |
|---|---|---|---|
| D0 | Canonical source/build/model lock; actual checkpoint/scales/operator inventory; physical layout descriptors; licenses; reviewed oracle and validation policy | User-selected base and permitted artifact access | Exact coherent commits/artifact digests; complete required inventory; no fabricated pins; oracle/corpus/error rules committed before optimized-result inspection |
| D1 | Authoritative schemas, cross-field validation, identity rules, one native admission engine; pure accounting and lifecycle fixtures | D0 logical identities; declared workload/feature caps | Strict duplicate/unknown-field rejection; safe-integer/u64 tests; all memory equations share one implementation; failure causes identify the limiting resource |
| D2 | Locked CPU/CUDA packaging and native entry points; small admin/probe interfaces; reviewed seccomp generator; host preflight/launcher foundation | D0 source/toolchain choices; D1 schemas | CPU package builds without CUDA discovery; `DESTDIR` installation resolves runtime libraries; non-root smoke; metadata/SBOM/notices; GPU-free build worker can create SM120 AOT artifacts |
| D3 | Container-aware bootstrap probe and external host preflight | D2 real images; administrator-provided paths, caps, credentials and masks | Actual user/security/cgroup/cpuset/GPU/mount settings recorded; bounded NUMA/lock/registration/kernel probes; parent limits and physical topology recorded; bootstrap marked `not_admitted` |
| D4 | One real target expert fixture and independent attention/index/packed-state fixture; CPU/CUDA implementations behind the same semantic descriptor | D0 mathematical/native encoding fixtures; D2 qualified toolchains; actual target devices | Strict NVFP4 W4A4, source FP8 W8A8, BF16 and sensitive operations meet preregistered policies; actual SM120 and actual CPU ISA execute the target dimensions; bounded tile/group capabilities known |
| D5 | Placement calibration from separately admitted operator fixtures | D3 environment; D4 kernels; D0 inventory; D1 fixture admission | Representative expert/row/tile transfers and arithmetic, local/remote NUMA, network TLS/framing and client contention measured; finite budgets; immutable calibration identity |
| D6 | Finalize renderer, resolved service/plan/lock/Compose environment, effective-config comparison | D1 single admission engine; D3 probe/host report; D5 calibration; D0 inventory; completed image digest | No unresolved inputs; package validates; probe repeated after any relevant final environment change; final Compose matches declared settings and actual inspect; atomic publication leaves a coherent package |
| D7 | Source-correct V4.1 eager client graph, state commitment and native adapters | D0 and D4 | Resident graph/layer/state fixtures; exact encoding/continuation; target SM120 execution; no substituted attention/graph or CPU neural operations |
| D8 | Narrow expert/row protocol and expert-only CPU/CUDA services; unique job ownership; NUMA policies | D0/D1 identities and framing; D4 evaluator; D6 admitted executor plan | Loopback then physical-host integrity/auth/credit/cancellation tests; local expert readiness independent of client; real per-node replicas/shards; no attention/KV stack or unintended full second model |
| D9 | Client host/device tiers, leases/events and exact migration; bounded prefill and expert/row cache plans | D7 correct graph/state; D8 real service; D5 calibration; D6 final plan | Actual target workload exceeds client VRAM yet executes with exact host-backed weights/state; double misses are inputs/results; broad expert unions and tiny/zero caches fit; bounded peaks and traffic-class evidence |
| D10 | Full release matrix, operating runbook, API/fault/durability tests and sustained soak | D0–D9 | All required image/profile/mode/residency/feature cases have named results and durable evidence; unavailable hardware is an explicit incomplete gate |
| D11 | Scoped optimization and final bottleneck review | Correct integrated baseline plus D5/D10 representative evidence | Candidate outcomes retained; equal-workload end-to-end improvements preserve P0–P2 and every hard budget; dominated paths removed; all required post-change tests rerun |

Source inventory and bounded host/transfer probing can proceed independently. CPU and CUDA evaluator work can proceed in parallel behind one descriptor contract. The attention fixture and expert fixture should expose kernel gaps before networking/full-model integration. State paging follows the correct graph and commitment model; an allocator alone cannot justify it. Packaging/lifecycle scaffolding may precede real kernels, but cannot pass the kernel/readiness gate with stubs.

For each measured REGION, declare the ordered objective and hard budgets before selecting policies. Prefill regions need TTFT/prefill responsiveness; decode regions need single-session inter-token distributions and committed-token rate. Record which objective is primary for each workload. Host/VRAM/locked/pinned/startup/cache/scratch/cgroup limits, progress capacity, context/output/image expansion, queue/credit/frame bounds, startup/drain deadlines, and report retention are hard acceptance inputs. No universal token rate or tolerance is supplied by the specification. Meeting one budget does not end candidate exploration or remove another required mode.

## Required schemas and dependency order

All project JSON parsers reject duplicate keys before ordinary object construction, malformed Unicode/lone surrogates, nonfinite values, unknown fields/revisions, invalid enum combinations, and unchecked size/range expansion. Ordinary byte/count numbers are exact safe integers no larger than `9007199254740991`; essential minimums are positive. Schema-declared unrestricted u64 IDs/epochs/offsets are canonical decimal strings parsed without binary64. Identity objects use RFC 8785 JCS and lowercase SHA-256 text. Cross-field checks are part of admission, not optional JSON type validation.

| Artifact | Required content and cross-field obligations |
|---|---|
| Compact trusted root/manifest | Source/model/profile/operator/encoding identity, role-specific tensor population, authenticated metadata graph/chunk indexes, authorized ranges and expansion bounds; manifest digest excludes only top-level `digest` and `signatures` |
| Operator contract | Exact quantizer scope/scales/clamps/reduction/dtypes/numerical realization; required capability per used tensor/operator; reject unknown required variants |
| Physical layout | Actual CPU/CUDA packed representation, offsets/ranges/alignment/scale layout, conversion provenance and mappings to logical identity; different approved layouts may bind to one logical contract |
| Service | Revision 1; exact role/executor combinations; absolute manifest/plan/lock paths; GPU ordinal 0 with UUID verification or CPU `null`; fixed admin socket; explicit artifacts path; TLS 1.3/no early data/peer SAN; network deadlines/credits/frames; client API policy forbidden on experts |
| Probe | Independent revision-1 schema: role/executor, device/expected UUID or CPU nulls, explicit NUMA node list, host/device/pinned test caps, deadline, `disable_core_dumps=true`; CPU GPU-test caps zero; no model handles, neural inputs, final plan or lock prerequisite |
| Deployment request | Intended role/mode/source/profile/features/context, hard host/device/resource caps, peer names, reviewed paths and selected build/image identity; requested requirements rather than a presupposed passing plan |
| Calibration report | Exact hardware/build/profile/layout/kernel/environment identity, fixture/workload/phase/size, cache state and warmup, concurrency/topology/power/thermal state, run count/order, timing/byte distributions and uncertainty; unavailable telemetry explicitly unavailable |
| Memory plan | Profile/operator/model/layout digests, maximum semantic positions, phase microbatch/image caps, every client host/VRAM/pinned/state/cache/progress pool, server global/per-node memory and CPU affinity or CUDA slots/cache, quantizer partitions, cache membership/destinations, worst-case/startup peaks, strict residency and calibration identity; an enforcing allocator/counter for every equation; no double-charged pinned/copy population |
| Deployment lock | Completed image and native build/source identity, model/profile, selected GPU UUID/SM or CPU ISA, tested driver/library/runtime/Engine/Compose/kernel, cgroup/NUMA policy, seccomp/schema/peer identities, config/plan/effective-Compose/host-report digests; no secrets or optimistic placeholders |
| Protocol | Framing, canonical identities, immutable logical/physical binding, exact payload equations, bounded credit ownership, request-ID namespaces, deadlines, one-use bulk grants and cancellation/terminal state; generate bindings/fixtures from the authoritative schema |
| Validation policy | Reviewed oracle, actual corpus/positions, profile, determinism/RNG settings, absolute/relative/near-zero rules, loss/quality criteria and justified thresholds per measured quantity; frozen before inspecting optimized results |
| Release evidence (proposed report schema) | Gate ID/status, source/build/model/fixture/environment identities, command/exit result, raw artifact paths/hashes, numerical/capacity/performance findings, incomplete/unrun cases and reasons; distinguish document assertions, emulated fixtures, native tests, container tests and physical-host tests |

`tools/render_deployment.py bootstrap --request PATH --output-dir DIR` creates probe input and enough reviewed Compose settings to run it. `finalize --request PATH --probe PATH --inventory PATH --calibration PATH --output-dir DIR` calls the same native admission calculations as `ds4ctl plan --request PATH --inventory PATH --probe PATH --calibration PATH --output PATH`; Python does not carry a duplicate memory model. Both validate and publish atomically. Production rejects bootstrap `not_admitted` output. A separately admitted operator fixture breaks the apparent calibration/full-model circularity.

The valid hash order is:

```text
source/model/operator/layout → native build → completed image digest
probe + host report + inventory + calibration → memory plan
memory plan + lock pathname → service.json
reviewed templates + generated environment → normalized effective Compose
all finalized identities/config/plan/Compose/host hashes → external deployment lock
final validation + startup enforcement → admitted process
```

There is **no required self-hash cycle**: `service.json` names the lock's absolute path, not its digest; the lock may then cover the service digest. Do not add the lock's content digest to service/plan, embed the final image digest in image build metadata, or put an external deployment lock inside its own image. If a lock self-digest is added, declare a detached digest or a reviewed exclusion scheme; the specification's explicit manifest exclusion rule does not automatically define every other artifact's scheme.

A changed image/device/security/resource setting invalidates its probe and dependent calibration/plan. Finalize must not silently reuse bootstrap measurements under different caps. Define a finite rerender/reprobe convergence policy that either publishes a configuration matching its evidence or reports an unresolved input; it must not fill measurements with constants. Recreate containers for config/lock changes. A live process cannot be assumed to observe multi-file bind updates atomically.

## Readiness, admission, health, and commitment semantics

| State/operation | Required observable semantics |
|---|---|
| Probe success | Bounded actual-container capability/permission results; not model admission or model readiness |
| Calibration success | A measured valid fixture under declared caps; not proof that the full checkpoint fits |
| Plan admitted | All required identities/capabilities/resources known and equations satisfied; no bootstrap marker; startup rechecks dynamic availability |
| Alive | Local process/admin protocol progresses with bounded health timeout; no model-wide hash/lock scan per health check |
| Expert ready | Required expert/row banks loaded, validated, placed/prefaulted/locked; actual CPU ISA or server GPU/kernel qualified; local resources/progress ready; listener/auth policy initialized; does **not** require a client already ready/bound |
| Client ready | Required compact package and graph/state/residency/kernel plans ready, startup/JIT complete, current expert authenticated and immutably bound; absent peer or incomplete binding keeps ready false |
| Busy | Fully initialized ready process may be executing its admitted generation. Health stays good; the initial one-active/zero-queue API returns typed busy for a second generation |
| Draining | Stop new admission and mark ready false; finish/cancel only according to committed-state boundaries; abandoned replies cannot publish state or output; hold the generation slot/buffers until workers/DMA cannot touch them |
| Fatal fault | Remove readiness, fail affected generation, invalidate uncertain private state, bounded best-effort drainage and nonzero process termination; no catch-and-continue, backend substitution or indefinite failed-event wait |
| Restart/mode change | Fresh peer binding/epoch and no reuse of uncertain old state; drain both roles, stop old mode, verify resource release, then launch the selected replacement; no in-place mode mutation or generation rollback |

`ds4ctl validate` validates schemas, accessible files/secrets, identity/profile coherence and bounds without loading the whole model. Probe/plan/validate/health/drain have finite deadlines and meaningful nonzero failure exits. `health --socket /run/dwarfstar/admin.sock --ready` is role-sensitive. The socket is UID-owned mode 0600 with peer credentials when available; its bounded local protocol exposes progress/readiness/safe metrics/drain only.

Track committed input prefix, positions incorporated into state, next sampled token, emitted output and RNG state separately. An emitted newest token need not yet be in KV/Engram state. Stream only complete valid steps. Failure never produces a success terminal event or commits output from missing expert math. Initial recovery uses a fresh session and exact replay of the committed history; it never silently replaces previously emitted tokens or promises durable cross-restart response IDs.

Startup progress is verify/configure → inventory → read/hash → explicitly authorized bounded convert/repack → allocate/place → prefault/lock → initialize kernels → bind peer → ready. Normally serving starts from prepared artifacts. Measure startup and derive health `start_period`/wait deadlines. Cold optional caches can be ready; unavailable required banks/state and first-request kernel compilation cannot.

SIGTERM: stop admission/ready; stop abandoned reply application; cancel/finish the bounded committed unit; quiesce DMA and workers; close peers; flush permitted reports; exit within tested finite grace. `init: true` and exec-form entry points support forwarding/reaping. SIGKILL provides no orderly-shutdown guarantee. Sticky CUDA/CPU corruption or teardown failure leaves the workload unavailable for operator recovery, not blind reload onto a suspect device.

Set zero core limits **and** `PR_SET_DUMPABLE=0` after final credentials and before sensitive loading; verify in release containers. Mark sensitive arenas `MADV_DONTDUMP` when available. Host piped core collectors can ignore the size limit. A full report volume has a bounded diagnostic and configured required-audit admission failure policy, not unbounded buffers or fabricated evidence.

## Packaging, launcher, and operational deliverables

| Deliverable | Required acceptance boundary |
|---|---|
| `package-cpu` | CPU expert server/control tool/CPU-NUMA-TLS libs and metadata under `DESTDIR`/`PREFIX`; no CUDA discovery, GPU framework imports, CUDA linkage/initialization or NVIDIA runtime requirement |
| `package-cuda` | `ds4`, `ds4-server`, CUDA expert server/control tool, exact qualified native kernels/libraries/metadata/notices; client CUDA and server CUDA policies remain explicit |
| `deploy/Dockerfile.cpu`, `deploy/Dockerfile.cuda` | Multi-stage, digest-pinned build/runtime images, locked offline package inputs, x86-64 host baseline across all linked libraries, explicit `sm_120a` client accelerated artifacts, no device autodetection on build worker, numeric unprivileged user and exec form |
| `.dockerignore` and image hygiene tests | Exclude model/prepared banks, private credentials/certificates, `.env` secrets, prompt data, large caches, VCS/build junk; preserve required lock metadata/small public fixtures; inspect context/history/layers; no secret in ARG/ENV/layers |
| Embedded build metadata/schema/license directories | `/usr/local/share/dwarfstar/build-info.json`, `/usr/local/share/dwarfstar/schema/`, `/usr/local/share/licenses/dwarfstar/`; source/compiler/architecture/dependency hashes and kernel/profile capabilities; not the external final deployment lock |
| `deploy/seccomp-numa.json` plus source/diff generator | Derive from the pinned Engine default deny profile; review `get_mempolicy`, `set_mempolicy`, `mbind`; bind before first touch without migration; optional `move_pages` only pid=0/nodes=NULL/flags=0; remove conflicting entries; retain unrelated restrictions |
| `deploy/compose.expert.yaml`, `deploy/compose.expert.cuda.yaml`, `deploy/compose.client.yaml` | One selected expert mode and one separate-host client; actual digest/UUID/paths/cpuset/finite caps; no cross-host `depends_on`; no fake paths/memory/digests; fixed file order; no privileged/host PID/network/IPC/Docker socket |
| Template extraction/conformance source | Extract every embedded template into actual files from the consumed source blocks with source digest/range provenance; tests detect drift instead of manual transcription |
| Renderer/launcher/host preflight | Typed request/bootstrap/finalize; controlled environment, no shell `source`; explicit local/remote daemon identity and absolute existing bind paths; no automatic host daemon/runtime/permission repair |
| Release operator tools/runbook | Provisioning, artifact access/preparation, CPU vs CUDA mode selection, secrets/labels, probe/admission, startup/progress, authenticated networking/API, health/drain/mode change, maintenance/recreation and fault recovery |
| Durable release bundle | Integrated source and maintained patches, exact locks/image/base digests, SBOM/provenance/licenses, schemas/native packages/test targets, oracle/golden policies, calibration/placement/capacity reports, raw benchmarks/fault/soak reports and supported/incomplete matrix |

CPU runtime testing includes a host without usable NVIDIA libraries and all linked libraries on the **actual supported CPU ISA**. Do not use `-march=native` or import a modern math library that silently requires AVX2 on an AVX-only target. Cross-building x86-64 on ARM, or an emulated CPU test, cannot replace native target execution.

The launcher supplies an allowlisted environment, clears inherited deployment/Compose override variables, suppresses unintended `.env` discovery, fixes the Compose version/file order, and never sources the generated environment as shell code. Render `docker compose ... config --format json` under that same environment, compare normalized image/device/resource/mount/port/security/entrypoint values with the final contract, retain its digest, then inspect the created container. `config --quiet` and a YAML parser are complementary checks, not equivalence proof. Test spaces, Unicode and dollar signs in legitimate paths. Probe `compose run` must retain production resources/security/device/mounts while intentionally publishing no inference listener.

Application pools, cgroup peak and physical headroom are separate. Budget live file-cache charges, runtime/driver overhead, TLS/socket buffers, thread stacks, tmpfs and `/dev/shm`; finite memlock covers **all** locked model/state/replica population and tested registration accounting. `memswap_limit` equals the same positive `mem_limit`; validate actual cgroup v2 limits/events/pressure and parent limits. No-swap alone does not lock file pages. Prefault and checked lock/placement are required. `cpuset` selects CPUs, not memory nodes; application NUMA binding must respect actual allowed nodes and per-node capacity. Retain cap-drop-all/no-new-privileges, AppArmor/SELinux and narrow read-only binds.

Only required small writable runtime/tmpfs/report/code-cache locations exist. JIT is exceptional, bounded before readiness with a separate size-limited code cache and source/compiler/device identity; it cannot use blanket writable/executable rootfs. Model/private-state spill into tmpfs, image layers or reports is forbidden. The server mounts its prepared banks; the client mounts only its compact package and intentional caches. Missing daemon-host sources fail with `create_host_path: false`. Reports/prepared artifacts survive container removal; private live sessions do not.

Credentials are per-peer private files with actual host ownership/modes permitting service UID 10001, not environment values/arguments. File-backed Compose secret ownership declarations cannot repair them. Every image runs offline except the declared local peer/API relationship; no weight downloads, remote-image fetching, broadcasts, downloaded tool execution or host Docker daemon reconfiguration at startup.

Exactly one selected GPU is exposed by UUID/device request and runtime visibility; ordinal 0 is container-relative. GPU exposure is not exclusive ownership or a VRAM quota. Administrator/workload policy prevents conflicts, and dynamic free VRAM is checked before admission. GPU-equipped hosts need the tested host driver/toolkit runtime configuration; CPU hosts do not. Do not silently run `nvidia-ctk`, restart Docker, bypass driver checks or reset a GPU used by other processes.

## API and network conformance

The reused native `ds4-server` adapter initially qualifies `GET /v1/models` and `POST /v1/chat/completions`. Other adapters are unsupported until their native encoding/streaming tests pass. Model listing exposes only the loaded `DeepSeek-V4.1-Flash` and declared profile identity. No donor V4 aliases, disk KV mode, autonomous agent launcher or ignored sampling parameters enter the target path.

API tests cover bearer authorization even on loopback and **before** expensive parsing/image decoding/GPU allocation; known model/nonempty message list; `n=1`; explicit Boolean stream; bounded output/context reservation; conflicting token aliases; unsupported logit/probability/structured-output/reasoning/tool/image settings; exact admitted defaults and RNG metadata. Add schema fields for required header-byte/JSON-depth/message/tool-schema/encoded-image/decoded-image/decompression/patch-expansion limits and relevant deadlines, even though the illustrative JSON lists only a subset. Bounds are explicit and derived; body allowance is not semantic context.

One active generation and zero queued generations is the initial policy. Prefix reuse proves identical native tokens, image identity and reasoning/tool encoding, and cannot attach a caller to another caller's mutable state. Histories that alter an incorporated token require full re-prefill/fresh epoch. Define usage in native positions/tokens/image expansion. A stalled/disconnected consumer enters safe cancellation, retains its slot while work is leased, and terminates without success on failure. Fixtures cover UTF-8 splits, tool fragments, EOS/stop strings, length termination, terminal finish reason/usage/end marker exactly once, and faults before/after SSE begins.

Expert networking uses real cross-host routing and certificate SAN/EKU/client authorization; service names/localhost do not span Engines. Control and bulk authenticate/bind independently to the declared epoch. Ports 7443/7444 bind the private expert IP and allow only the client host; client API publishes host loopback by default but retains bearer auth on container paths. Broader publication needs explicit HTTPS and forwarding restrictions. Test unauthorized hosts through Docker's real forwarding/firewall backend; INPUT/ufw alone may not protect published traffic. IPv6/dual-stack is a separate declared equivalent policy. Offline readiness uses configured peer resolution/PKI and deliberate host mapping if necessary, not internet repositories.

## Conformance and release gate inventory

Each row below is a named gate family with raw evidence and separate native/container/physical results. Pure fixtures, document assertions, a small-model completion, or process existence cannot mark target execution passed. The core numerical/mode/residency matrix has **3 profiles × 2 server modes × 2 client residency conditions = 12 cells**, each covering initial prefill, decode and continued prefill; all-remote, client VRAM, client host-only and mixed slots; required text/reasoning/tools/continuation; separately qualified images. Resident reference requires suitable hardware or a documented independent bounded reference strategy; an unexecuted full model is not parity evidence. Add required NUMA policies and server/client cache regions where capacity permits.

| Gate | Required positive and adversarial coverage |
|---|---|
| G01 Source identity/inventory | Reviewed trust anchor, exact commits/artifact/profile/encoding/layout/operator identities, complete target shards/scales; modified hash/wrong artifact/unknown tensor rejected; logical identity permits only approved different physical layouts |
| G02 Schema/canonical identity | Duplicate-aware JSON/YAML, revision/type/enum/unknown fields, malformed Unicode/nonfinite, safe-number/u64 >2^53, JCS ordering/escaping/number fixtures, manifest self-digest exclusion; checked offset/count arithmetic beyond 4 GiB and 1 TiB synthetic fixtures separately labeled |
| G03 Preparation integrity | Bounded parsing/truncation/overlap/layout errors, authenticated metadata graph and chunk boundary permissions, authorized overfetch/scratch, verified scales before use, no required simultaneous full source+repack population |
| G04 Quantization/operator parity | Every value code, scale orientation/extrema, zeros/saturation/rounding/nibbles/padding/odd/tile/block boundaries; strict W4A4 vs W4A16 negative fixture; W8A8/BF16 and sensitive FP32/cast rules; quantizer partition scope; both required fidelity axes |
| G05 Expert/router parity | Actual target dimensions, both gate/up branches and asymmetric clamps, coefficient before down quantization, zero/extreme valid coefficients, shared branch once, original slots and canonical reduction; biased selection vs raw normalized weights and tie/image bias |
| G06 Attention/graph/state | Actual SM120 V4.1 graph; causality/sparse validity/candidate/latest partial blocks/RoPE/sinks/source aliases/tile normalization/CED/mHC/Engram history; exact packed history/state bytes; no generic graph replacement |
| G07 CPU mode/ISA | Driver-free actual x86 ISA host, full linked dependency inspection, no CUDA initialization/framework requirement; CPU FP8/NVFP4/BF16 fixture and target-model cases; no single-hot-expert speed claim for broad bank |
| G08 GPU mode/architecture | GPU-free reproducible AOT builds, exactly requested UUID, actual client SM120 `sm_120a` accelerated kernels and separately qualified server SM; unsupported GPU/profile/library rejected before bank load; no SM100-only substitution or first-request compilation |
| G09 Client tiers | Forced small VRAM and real target footprint greater than VRAM; host-backed attention/shared/head weights and state, expert host hits, separate Engram caches; actual H2D/D2H and valid leases; zero client CPU neural operations; no remote private-state/disk spill |
| G10 Server bounds | Expert-only initialization, no duplicate attention/KV/full transformer, cache smaller than selected union, cold/double misses, broad prefill groups, progress reserves, bounded weight/state/cache/scratch/transfer pools; zero CUDA-server CPU expert fallback |
| G11 NUMA | All required sharding/replication policies correctness/capacity; real physical local copies and unique job ownership, per-node admission and local/remote penalties; default seccomp denial negative test; small node limit rejects rather than scattering promised replicas |
| G12 Memory/capacity | All process/cgroup/startup/conversion/tmpfs/driver peaks, pinned and locked unique-page accounting, major faults/disk reads/swap/local pages; tiny/zero caches/minimum slots/insufficient host/node budgets/registration failure/broad unions; actual high-capacity witness distinct from sparse synthetic offset test |
| G13 Exact continuation/features | Long prompts/appended turns/tool results/images when enabled, chunk/source/candidate boundaries/sliding wrap/partial compressor groups/aliases; exact retained vs exact reconstruction; bounded approximate replay negative; profile change invalidation and no migration requantization |
| G14 Framing/request association | Exact byte equations, CSR/rows/expert/slot uniqueness and ranges, wrong dtype/layout/digest/epoch/NaN coefficient, duplicate/missing/out-of-order replies; no attacker-size allocation; no-credit/no-buffer progress; partial TLS frames and operation/frame/write deadline separation |
| G15 Binding/bulk/cancellation | Zero-to-bound transition, fresh binding after invalidation, control/bulk namespaces and one-use grants, rejected reconnect; cancellation before dispatch/in workers/after result/racing terminal; exactly one terminal target result and one credit release; dirty page/output still leased; late DMA/prefetch never publishes/reuses |
| G16 Two-host authentication/network | Actual physical hosts, SAN/client authorization/wrong or expired credentials, bulk channel binding, unauthorized host/firewall denial, broken pipe/stalled receiver/timeouts; no localhost-only qualification or raw weight miss traffic |
| G17 Admission/rendering | Bootstrap without completed plan, finalize rejects incomplete capability/calibration/placeholders/secret/UUID/cap failures, changed final resources force re-probe, identical native plan calculations, atomic output/restart path; service rejects `not_admitted` |
| G18 Effective Compose | Three combinations CPU expert/CUDA expert/client; inherited shell/image/GPU/memory overrides rejected; fixed files/env/version; normalized `config --format json` equals approved policy and actual inspect; path spaces/dollars; probe run same production settings |
| G19 Readiness/health | Hash failure/absent client peer/unsupported kernel/residency failure/unfinished startup remain unready; expert local ready before client; client awaits binding; busy stays healthy; finite health/progress timeout and no model scanning |
| G20 HTTP/SSE | Token auth, header/body/image/context/tool/nesting limits, sampling/model/feature rejection, one active/zero queue, caller/prefix isolation; stream fragments/stops/length/usage/terminal markers; stalled consumer cancellation; remote faults cannot produce success completion |
| G21 Security/confidentiality | UID authorized read/write only, fixed private admin protocol, cap-drop-all/default-derived seccomp/LSM/read-only binds, no dangerous host mounts/sockets; core-dump attempt emits no secret/state dump; initialized transmitted padding/private-generation isolation; report volume untrusted |
| G22 Fault/quiescence | Fail before/during dispatch/partial TX/after server execution/before reply/contribution copy/state writeback/after sample before update; CPU worker loss/restart/CUDA errors/OOM/wrong binding/cancel; typed failure, invalid uncertain state, safe bounded reclamation, no substituted math/output |
| G23 Container pressure/shutdown | Cgroup OOM, finite memlock pin failure, NUMA denial, GPU access loss/systemd-maintenance case, broken/stalled peers, SIGTERM/SIGKILL/canceled dirty writeback; no indefinite failed event, unsafe recycle or blind same-epoch restart; tested stop grace |
| G24 Durability/mode change | Container recreate/remove preserves checkpoint/manifest/reports, no private-state durable-resume fiction, source banks never cleanup targets; drain/stop/release/rebind mode sequence; failed teardown remains unavailable |
| G25 Build/reproducibility | Locked sources/offline packages/digest bases/no GPU worker prerequisite; native/library/source architecture metadata and complete runtime linkage; compare outputs/provenance and explain nondeterministic metadata rather than asserting bit identity |
| G26 Image/distribution hygiene | Context/history/layers/logs exclude weights/secrets/prompts/unwanted build tools; BuildKit secret use if acquisition authorized; SBOM/notices/trusted release digests/schema/tests/runbook retained; CPU image independent of CUDA |
| G27 Performance/soak | Matched native/container workloads; cold/warm/startup/prefill/TTFT/median-p95-p99 ITL/committed rate/memory/failures; equal workload mode/locality/cache comparisons; broad routing across sessions/domains/context growth; physical link/power/thermal/concurrency and uncertainty; minimum one-hour initial engineering soak under production security/memory policy |
| G28 Static/test-target completeness | Actual template extraction, duplicate-aware schemas, YAML/JSON render, Dockerfile lint/shell syntax/Markdown/reference anchors/dangerous-setting assertions; separate pure protocol/CPU/CUDA/Docker/physical test targets; every required result present, failures triaged, optimized candidates resolved and post-change bottlenecks reviewed |

Every run retains exact source/build/model/profile/layout/fixture/policy/image/config/plan identity; GPU UUID/SM/driver/toolkit; CPU ISA/allowed CPU/node masks; Engine/Compose/kernel/seccomp/effective cgroup/lock/pin limits; workload/cache state/warmup/run count/order; raw commands/status/logs/counters and uncertainty. A public release matrix must distinguish functional support, numerically validated support, fitted capacity, achieved performance and incomplete/unrun tests. No aggregate result hides a required-path regression.

## Requirement traceability to the release matrix

| Locked requirement (§21.1) | Plan ownership | Release gates |
|---|---|---|
| Exact V4.1 identity/native behavior | D0/D4/D7 | G01–G06/G13 |
| Blackwell/5090 client | D2/D3/D4/D7 | G08/G09/G19/G27 |
| NVFP4/FP8/BF16 | D0/D4/D10 | G04/G05/G07/G08 across all 12 core cells |
| No client CPU neural work | D7/D9 | G09/G22/G27 instrumented in both modes |
| Client DDR5/host backing | D3/D5/D9 | G09/G12/G13 with real constrained VRAM |
| CPU-only server | D2/D4/D8 | G07/G11/G16/G27 |
| Small-VRAM CUDA server | D4/D8/D9 | G08/G10/G12/G27 |
| NUMA copies/concurrency | D3/D5/D8 | G11/G12/G15/G27 |
| Expert disaggregation | D8/D9 | G10/G14–G16/G22 |
| No routine network expert-weight misses | D8/D9 | G03/G09/G10/G16/G27 traffic-class accounting |
| Engram correctness/cache tiers | D4/D7/D8/D9 | G03/G06/G09/G13 |
| Exact state/model behavior | D7/D9 | G04–G06/G13/G15/G22 |
| Useful resource utilization under budgets | D5/D6/D9/D11 | G10–G12/G17/G27/G28 |
| Operational safety/durable evidence | D2/D6/D8/D10 | G14–G28 |
| Docker deployment | D2/D3/D6/D10 | G07–G12/G16–G28 inside actual release images |

## Development host versus release test hardware

Read-only observations on 2026-10-03 (all inventory subprocesses exited 0 except intentionally querying Git status in the empty non-repository):

| Observation | Current development host |
|---|---|
| OS/architecture | Linux `7.0.0-1019-nvidia`, `aarch64`, glibc 2.39 |
| CPU/NUMA | Cortex-X925/Cortex-A725, 20 online CPUs, one NUMA node; allowed CPUs `0-19`, memory nodes `0` |
| RAM/swap | `/proc/meminfo` total `127533268 kB` (about 121.6 GiB); host swap `33554428 kB`; neither is a release-container admission result |
| GPU | NVIDIA GB10, compute capability 12.1, driver `580.178.04`; `nvidia-smi` memory total reported `[N/A]`; no discrete 5090 VRAM capacity inferred |
| Docker runtime | Engine/client 29.6.2 arm64, containerd 2.2.5, runc 1.3.6, Compose `v5.2.0`; accessible current daemon, not the pinned x86 release environment |
| Compiler availability | GCC/G++/make/Python/Git/numactl present; `nvcc` and clang absent from PATH; absence from PATH does not prove no separately staged toolchain exists |

This host supports planning, source inspection, pure schema/JCS/accounting/protocol fixtures, template extraction, static checks, and limited native host-independent work. Existing Docker can exercise host-local tooling under its declared ARM/current-Compose environment. That is not qualification of the required Linux x86-64/Compose-v2 images, AVX-era server ISA, multi-node replicas, RTX 5090 SM120 kernels, discrete PCIe/DDR5 staging, or full checkpoint capacity. GB10/SM121 is a separately qualified development variant if used; it cannot replace the target.

Required release access includes (1) an actual Linux x86-64 RTX 5090 client with measured usable RAM/VRAM/negotiated link, (2) a GPU-free actual-ISA x86-64 CPU server with measured global/per-node capacities and enough NUMA nodes for required locality policies, (3) a separately qualified Blackwell CUDA server with bounded VRAM and sufficient admitted host RAM, (4) two physical hosts/private authenticated routing/PKI, and (5) sufficiently large reference/high-capacity hardware and authorized full artifact access for required resident/profile/replica cells. No numerical RAM floor is invented here: inventory, temporary peaks and replica equations determine it. A 512-GB concept and this 121.6-GiB host are not feasibility proofs.

## Decisions, tensions, and unresolved inputs

| Item | Resolution or next evidence; release implication |
|---|---|
| Canonical base correction | Replace mandatory `stefandsl` identity with `antirez/ds4`; pin coherent commit and audit current entry points/API/help/build/release guidance. Keep separately reviewed donors explicit. Source document's stale selection cannot silently override the user. |
| Compose major version | Specification declares Compose v2; installed development CLI is v5.2.0. Use a specifically pinned qualified v2 release for the stated baseline, or obtain an explicit baseline update with complete affected qualification. Do not infer equivalence from the common `docker compose` command. |
| No final deployment/build pins | Resolve source/toolchain/library/base/image/platform versions together; digest-qualified templates deliberately have no runnable fake values. This blocks reproducible release, not planning. |
| Missing budgets/tolerances | Actual context/image/output limits, finite deadlines, physical capacities, primary performance objectives and reviewed per-operator/full-model thresholds must be supplied. Freeze quality policy before optimized results; measure, do not fabricate a token rate. |
| Bootstrap/final environment drift | Shared admission engine and hash order are settled. Define bounded regenerate/reprobe behavior and dependency invalidation; only evidence under the final settings is admission evidence. |
| Image/lock hash cycle | Keep image build metadata upstream of final image digest; service carries lock path only; external lock covers config/plan/effective Compose. Any optional self-digest needs an explicit scheme. |
| API schema example incomplete | Add explicit header/depth/decoded-image/tool/context/usage/RNG/stream limits and admitted-setting metadata from the actual native adapter. Example timeout/body numbers are illustrative, not performance targets. |
| Expert readiness wording | Interpret local expert readiness independently of peer readiness; client readiness requires bound expert. Busy/queue capacity is separate. Tests must make the distinction explicit. |
| Reference feasibility | Identify hardware/oracle strategy for resident FP8/BF16 and full NUMA replicas. Record unexecuted cells as incomplete; bounded fixtures do not become full-model or multi-terabyte evidence. |
| Seccomp version dependence | Vendor the selected Engine default as a project-owned generated input/digest/diff, add only reviewed calls/arguments, validate with its runtime and actual UID. Never edit installed Docker policies in place. |
| Startup conversion vs offline model | Normally use prepared artifacts; any conversion/repack at startup is explicit, pre-authorized, bounded, in measured peaks and complete before ready. No runtime downloads or unbounded conversion. |
| Fatal teardown/report policy | Configure tested finite drain/grace and required-audit storage exhaustion policy. Do not claim release after failed GPU/CPU quiescence or silently drop required evidence. |
| Donor serving/storage behavior | Current antirez disk-backed Engram/GGUF/API behavior requires a bounded audited adaptation to the immutable prepared-bank/native target contract, not inherited disk paging, lossy replay, V4 aliases or ignored settings. |

No architectural contradiction was found in the assigned range after separating the intended milestones, role-specific readiness, and external lock/hash ordering. The baseline/installed-host mismatch and source-repository correction are real evidence/identity differences requiring explicit handling. The other listed items are specified empirical inputs or implementation decisions, not permission to substitute a smaller model or drop a required mode.

## Bounded primary-source verification

Official pages were accessed during this review on 2026-10-03. These checks verify technology contracts, not application execution:

- Shell variables override Compose environment-file values; literal quoting/interpolation and fixed file order matter. This supports controlled launcher rendering and override-negative tests. [Docker variable interpolation](https://docs.docker.com/compose/how-tos/environment-variables/variable-interpolation/)
- `config` merges files, resolves variables and renders the applied model; JSON format is available, while quiet only validates. This supports normalized comparison followed by actual container inspection. [Docker Compose config](https://docs.docker.com/reference/cli/docker/compose/config/)
- Equal positive memory and memory-swap limits prohibit swap; zero is ignored, and `free` inside containers is not limit evidence. This confirms the required no-swap/effective-cgroup check. [Docker resource constraints](https://docs.docker.com/engine/containers/resource_constraints/)
- Default seccomp restricts NUMA management calls. This confirms the need for an Engine-pinned reviewed profile and an actual-container positive/negative probe. [Docker seccomp](https://docs.docker.com/engine/security/seccomp/)
- File-backed secret `uid`/`gid`/`mode` attributes are ignored because of bind mounting, and `cpuset` selects CPU IDs. Actual host permissions and application NUMA binding remain necessary. [Compose services](https://docs.docker.com/reference/compose-file/services/)
- CUDA 13.x minor-version compatibility lists driver >=580 and documents feature/PTX/target/library caveats; this is a family floor, not target kernel qualification. [CUDA minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
- The catalogue distinguishes RTX 5090 CC12.0 from GB10 CC12.1. [NVIDIA GPU compute capabilities](https://developer.nvidia.com/cuda/gpus)
- The PTX target notes for the relevant block-scaled `mma` NVFP4 kinds enumerate architecture-specific `sm_120a`/`sm_121a`, not interchangeability of plain SM targets. Actual compiled target execution remains required. [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)

The specification's reported 155 document/static assertions and 200 recurrence cases are prior document-review claims, not tests executed by this reviewer or this repository. No image build, production schema/library certification, TLS integration, CUDA execution, physical NUMA replica, complete checkpoint load, or inference benchmark was run here. Planning completion leaves those release gates open until the implementation supplies their evidence.
