# Prompt prefill admission checkpoint

The selected prompt microbatch is an explicit `prefill_rows` field in the
planning request, generated memory plan, and placement plan's `runtime` object.
The supported bound is 1–64 rows and cannot exceed the admitted semantic
context. Placement and planning must agree. This is a predeployment schema
change: earlier documents missing the field must be regenerated; no implicit
default silently admits a one-row path. Decode remains a single-row operation.

`ria_service.prefill_rows` records the authenticated memory-plan value. The
native admission result preserves it, and generated inventory provenance binds
it along with context, manifest, runtime policy, and request identity. The
renderer checks the native result preserves both workload bounds.

Metadata-only native inventory uses the graph-owned
`ria_graph_prefill_required_bytes` API and endpoint-owned
`ria_remote_host_required_bytes` API. It does not implement a second copy of
either allocation equation. New raw request fields authenticate graph host
state, projection tile rows, endpoint frontend host reservation, and transport
limits. Snapshot/group staging must fit the graph host owner, the added
projection workspace must fit the device owner, the reusable pinned pool must
fit the pinned cap, and endpoint batch/wire storage must fit its frontend owner.
Distinct client host owners cannot borrow the same reserved bytes. Serving
additionally checks complete graph, vision, tokenizer, frontend, and population
allocations before readiness. The generated inventory conservatively reserves
owners across every phase because those allocations live with the session.

`ria_service_network_parse` is the shared strict transport configuration parser
for serving and metadata-only inventory; all strings borrow the input JSON
document. CPU-only inventory calls no CUDA runtime and never populates model
payloads.

The second memory review reproduced an endpoint control-buffer undercount.
A maximum 262,144-byte response can retain 786,433 bytes in the owned JSON
document alongside its wire payload, totaling at least 1,048,577 bytes before
semantic rejection. The old two-control-buffer allowance was 524,288 bytes.
The model-free allocation observation is retained in
`build/ria/evidence/prefill-memory-observation.json`. JSON parsing, canonical
output, and sorted-key allocation now use shared checked sizing functions.
The endpoint budgets both concurrently owned documents, both control payloads,
canonical output, and the nested sorted-key peak. Native fixtures compare the
reported owned bound against actual parser allocations through the maximum
control size, and reject overflow and invalid limits. The endpoint also accounts
for its full public multirow six-expert CSR shape and frame-limited row lookup;
its production callback's smaller shape is not used as an allocation bound.

`ria_numa_worker_rows` derives the server batch capacity from its actual fixed
2 MiB worker arena, shared 1 MiB stack reservation, CPU expert scratch, and
paired input/output row buffers. CUDA workers therefore admit at most 25 rows
per worker even when the client microbatch is 64. Service and metadata-only
inventory use that same geometry and the authoritative pinned projection-pool
equation, rejecting an undersized expert CUDA pinned reservation before startup.
The complete CUDA device allocation is still checked by actual pooled context
creation before listeners/readiness; no guessed private-owner size is described
as an exact static device minimum. No GPU or model payload is needed by these
inventory checks.

Final delegated verification passed: the canonical strict C99 build of
`ds4ctl` and the native contract fixture, the native fixture execution, all 59
inventory/deployment Python cases, Ruff, regenerated schema consistency, and
diff whitespace checks. Logs are
`build/ria/evidence/prefill-admission-budget-build.log`,
`prefill-admission-budget-contracts.log`, and
`prefill-admission-budget-tests.log`. Fixtures cover missing legacy fields, hard
row/context bounds, mismatched placement rows, underfunded host/device/pinned/
endpoint owners, altered or missing inventory provenance, accepted maximum
64-row inventories, CUDA pinned worker pools without CUDA initialization, and
preserving an existing destination on failure. Frozen-environment and native
plan fixtures now declare the same explicit row bound. Root owns the subsequent
frozen integrated gates for graph/engine/server completion; these delegated
checks do not claim live hardware correctness or measured throughput gains.
