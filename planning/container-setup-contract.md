# Container-managed setup contract

Baseline: `5892b590735bec3b8b25ff615aaf2f887b02c745`. This work automates the existing initial-fixture admission path. It does not turn that admission into full-model release qualification.

## Ownership and target

One explicitly administrative `ria-setup` container runs per native Linux amd64/rootful Docker/cgroup-v2 host. Its local root controller owns the Docker socket, persistent controller locks, host PID/cgroup observation and its private workspace. The inference containers retain their existing non-root, read-only, restricted configuration. A fresh setup network worker drops to UID/GID10001 with no inherited Docker/CA-key descriptors; it submits bounded, fixed operations to the parent. It cannot choose host paths, commands, mounts, resources or images.

The CPU setup image includes its own CPU metadata/planning tool, Python preparation/reference dependencies, locked metadata/schema/source inputs and pinned Docker/Compose clients. It never requires a GPU on a CPU expert host. CUDA host utility discovery is explicit and separate from measured native probes.

The operator provides immutable service image references, physical host prerequisites, role/executor/profile, source or trusted prepared server, workload/context, CPU/NUMA membership, resource reserves/caps, timing and transport bounds, accepted numerical thresholds and fixture hard limits. Setup derives model/build/environment identities, tokenizer paths, placement/grants, credentials, request documents, registration, evidence routing and launch order. No unmeasured resource fit, tolerance, hardware capability or performance winner is invented.

## Pairing

Expert creates a private invitation containing a random job identity and independent pairing secret. The invitation is the one trusted out-of-band transfer. TLS is the default: setup creates a local CA/server key, invitation carries public trust, and client creates its own private key and sends only a CSR. The parent signs the fixed job/client identity. Private keys never cross hosts. Explicit trusted-network mode creates no certificate material; setup messages still have request/response MACs and replay protection, without claiming confidentiality. There is no downgrade fallback.

Both endpoints bind numeric private IPv4 addresses. Setup port defaults to9010 and is configurable. Native service ports remain the canonical Compose ports. API bearer credentials are generated privately or provisioned explicitly and never put in peer facts/logs.

## Ordered workflow

1. Validate local settings and exclusive workspace ownership; freeze settings identity.
2. Expert discovers actual local host/container/build facts and opens its paired setup listener. Client authenticates the invitation, obtains its certificate if enabled and commits its explicit local expert selection. Expert prepares/verifies its server model and extracts only that compact client package.
3. Client validates expert model/build facts and discovers its actual local host/container/build facts through canonical tools, then fetches only the allowlisted compact client export. Verify the complete package and extraction provenance.
4. Client derives its declared placement and API token and freezes its bootstrap environment. Expert accepts these fixed client facts, derives exact grants and freezes its own environment.
5. Expert freezes the explicitly accepted policy and constructs a single four-run registration from actual role images/builds/environments. Each host validates the same registration and installs its own derived inputs.
6. Start both exact owned fixture containers. Probe each locally, execute both native fixtures, start expert transport fixture then client transport fixture. Native connection retries provide readiness; do not consume a fixture connection with a dummy probe. Existing1800-second fixture lifetime remains unchanged.
7. Transfer complete four-run raw reports, wait4 supervision and preflight/proof graphs. Each role derives its own calibration using the canonical producer, generates native inventory and finalizes through existing validators/planner.
8. Stop both fixtures before loading model services. Launch expert, wait for actual native health, then launch client and wait for health. Persist exact container ownership and evidence. Close setup channel after completion; inference continues without either setup process.

Stages record durable intent before effects and complete only after their owned outputs validate. Lost responses query the same operation identity in the running controller. An interrupted measurement is failed, never reconstructed from stdout or retried until it passes. Unfinished controller restarts stop exact owned resources and require a new job/workspace; standalone acquisition/preparation tools can reuse their own verified resumable checkpoints, and a new job can accept a completed trusted prepared bank without copying it. Completed-job reruns verify current authority, host and actual native health rather than trusting old readiness receipts. Failed admission stops owned fixtures/services; cleanup failures remain explicit. A settings/policy/image change requires a new job/workspace and fresh affected evidence.

## Interfaces and verification

`setup_config.validate_settings` validates declarative settings; `setup_host.discover_local` publishes actual host/build facts and `build_request` derives canonical deployment requests. `setup_artifacts` owns preparation, compact import verification, placement/grants and private credential provisioning. `setup_security`, `setup_peer` and `setup_journal` own certificate/MAC/pairing, restricted peer operations, bounded allowlisted transfer and durable operation state. Root-owned `setup` coordinates these through existing deployment/fixture/qualification/native inventory APIs.

All local verification uses synthetic packages, mocked Engine/host boundaries, isolated software sockets/TLS/fork fixtures and static compilation. No checkpoint acquisition, physical GPU/probe/container/model/soak execution is authorized here. Hosted model-free builds publish all images only after offline gates; target hardware execution and the existing complete-model release gap remain explicitly unverified.

Runtime report trees cross the UID10001/root boundary: take descriptor-checked no-link/no-hardlink/special-file-free bounded snapshots before export, with expected-size hashes and absolute cancellation/deadline checks. Effective controller mount ancestry for external source/prepared/token inputs must match their exact daemon-host paths; host-side symlinks hidden by a bind remain an explicit trusted host administrator prerequisite. The Docker socket must be root-owned and inaccessible to the dropped worker through mode or access ACL; the worker additionally checks its effective socket access and all process capability/no_new_privs state.

Cleanup cancellation applies to every ordinary production subprocess and bounded file/hash loop. Only the actual cleanup thread may issue contained stop/remove commands after cancellation. Peer response drain must acknowledge the exact flushed final reply and propagate lost/deadline ambiguity. Never release workspace/role leases with live broker/tasks/network workers or ambiguous ownership: if bounded joins and the locked ownership snapshot cannot prove quiescence, persist fatal/ambiguous state and terminate the whole controller process with125. Termination encloses diagnostics in an unconditional finally, so failed marker writes or broken stderr cannot unwind leases with surviving work. A new controller reconciles durable unfinished intent and only exact owned resources. No completion receipt is published after such containment.

Publication uses the existing public GHCR CPU package for both its ordinary service image and the separate `setup-sha-COMMIT` controller image. Distinct immutable digests, entrypoints, users and role metadata identify them; the registry name alone is never a role contract. The CUDA image remains in its existing public package. Offline, actual amd64 builds and bounded model-free transfer comparisons precede publication; anonymous access and exact source/runtime metadata are verified afterward.
