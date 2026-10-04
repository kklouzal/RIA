# Preparation, admission and qualification review

This bounded independent review starts from source commit
`bd76be469c45c6e672481125023090577b08cdf3` and specification SHA256
`14224cdb33476944111e14f69a5679f0597c192a44d67b48f048910f326f3f6e`.
It follows the artifact/provenance, resource admission, deployment and release
requirements through their source, callers, schemas, tests and operator commands.
The earlier planning review consumed specification lines 1033–2133 completely;
this implementation review additionally traced the artifact, quantization,
prefill, NUMA and state requirements in sections 3–16. The implementation branch
is a maintained project fork of `antirez/ds4`; donor disk-backed defaults do not
replace RIA's strict RAM residency contract.

## Confirmed preparation defects and corrections

The preparer originally authenticated a source file, then reopened it during
calibration, conversion and copying. A same-size source change between those
operations could publish an output whose `source_sha256` described the previous
bytes. A synthetic BF16 reproduction produced a verified package while its
source hash no longer matched the recipe. This violated specification sections
3, 6.3–6.5 and 19.4.

`tools/ria/safetensors.py:60` now authenticates the entire trusted source hash and
parses its header on one opened regular-file snapshot. Subsequent tensor/scalar
reads check device, inode, size, mtime and ctime before and after use. Preparation
rechecks every source before returning an already committed package and before
publishing a new manifest (`tools/ria/preparation.py:532`, `:691`); target recipe
calibration uses the same checked reads. This catches source replacement,
same-size mutation and mutate/restore without an extra full-bank hash pass. The
contract trusts local filesystem/admin metadata; it does not claim to defeat a
privileged adversary forging that metadata.

An adjacent compact-metadata path checked a file hash and then reopened the
file for copying. Config/index/tokenizer reads could similarly use bytes other
than those named by later provenance hashes. `read_verified_bytes` authenticates
the exact bounded bytes parsed or published (`tools/ria/identity.py:123`). The
preparer uses it for source indexes and metadata; target recipe generation
retains pinned config/index/tokenizer identities; Engram derives its tokenizer
from authenticated bytes and hashes the exact bytes consumed and emitted
(`tools/ria/target.py:59`, `tools/ria/engram.py:83`). A source changed after a
successful compact read cannot change the already authenticated output bytes.

The mode documentation incorrectly said all reports were private. General
atomic host-tool output defaults to 0644, while deployment staging directories
are 0700 and explicit private checkpoints/controller state use stricter modes.
The artifact/deployment and handoff documents now describe these actual modes
and the required owner/group provisioning.

## Native inventory and admission review

The original software required the operator to supply internal allocation and
phase lists manually. A selfsealed inventory could omit a mandatory allocation
or phase. The new metadata-only `ds4ctl inventory` path closes that software gap:

* `ria/tensor.c:929` validates authenticated TensorStore metadata without opening
  the payload or initializing CUDA. `ria_tensor_owned_bytes` shares the serving
  representation's exact capacities, page-rounded shard storage, metadata and
  chunk-hash accounting.
* `ria/inventory.c:34` uses that accountant and the same expert configuration and
  NUMA population/replica accountant as startup. Explicit runtime, device and
  pinned reserves remain configured caps; these are not fabricated measurements.
  The inventory conservatively reserves the resulting population and pools in
  every required phase. Pinned bytes are counted once in host memory and also
  against the separate pinned limit.
* `tools/ria/inventory.py:10` derives the request and runtime policy from the
  actual authenticated manifest and role-specific options. `:56` checks the
  native result's complete derivation identity before publishing it.
* `tools/ria/deployment.py:599` rederives the entire inventory during finalize and
  requires exact equality. Changing allocations, aliases, replicas, phases,
  context, placement or runtime caps requires regeneration; resealing a modified
  list does not authorize it.
* Actual client/server startup still checks configured pool lower bounds before
  model first-touch. Expert startup explicitly sums runtime, TLS, worker,
  transport, pinning and metadata reservations (`ria/server.c:621`). Admission
  is distinct from full-bank residency/readiness and numerical qualification.

The independent review found no new confirmed unsafe accounting defect in this
path. Metadata-only success does not authenticate payload residency, establish
actual physical locality, or qualify a target model. Those remain separate
startup and hardware tests. The native inventory tests exercise tiny synthetic
metadata, all three NUMA policies, absent payloads, cap failures and resealed
omitted allocations/phases; they do not establish production capacity.

## Qualification software still incomplete

The former physical-contract validator could assign an unrelated scalar and
operator-chosen bounds to all G01–G28 and all candidate cells. Hashing those
reports proved byte integrity, not execution of the specified cases. The revised
schema revision 2 quarantines numeric reports as `unqualified_diagnostic` and
rejects every gate/cell or final-release claim
(`tools/ria/physical_contract.py:205`). Preregistered bounds, raw identity,
domains, timestamps and monotonic duration remain validated for diagnostics.
This is a safe containment fix, not an implementation of missing producers.

`tools/ria/qualification_readiness.py` fixes the G01–G28 obligations and five
matrix observation requirements to the specification identity. Its report
separates available, partial and missing software for each obligation, and
separately identifies the missing semantic release adapters. Existing schema,
admission, build, operator, loopback TLS, HTTP replay, logit comparison and soak
software can supply useful evidence. The review does not label every gate as
missing software: G02/G17 have supplied obligation software; G25 has supplied
build paths plus narrower provenance/reproducibility gaps.

A friend cannot execute every promised qualification path with the supplied
software. The full 540-cell producer lacks actual route/cache/residency/state
migration/NUMA observations and independent full-target execution; complete
semantic fault/container producers and receipts are also missing. Required
external reference/checkpoint authorization and administrator-controlled fault,
firewall, cgroup and device operations are separate inputs, not substitutes for
missing instrumentation. HTTP replay deliberately publishes zero physical
`observed_cells` (`tools/ria/release_runner.py:891`).

The model review separately confirmed that specification section 7.3 grouped
prefill is not implemented: `ria_graph_prefill` (`ria/graph.c:593`) and engine
prompt synchronization (`ria/engine.c:659`) iterate one-token work. Grouped row
state, multirow callbacks, bounded workspaces, expert grouping/scattering and
independent sequential-versus-grouped checks must be implemented before a
complete software claim. Neither missing software gap is resolved by buying or
provisioning the required hardware.

## Executed and unexecuted checks

On the development host, the owned artifact/native-artifact/physical-contract/
readiness/qualification suite passed **186 tests** using
`/tmp/ria-validation-venv/bin/python -m pytest -q tests/ria/test_artifacts.py
tests/ria/test_native_artifacts.py tests/ria/test_physical_contract.py
tests/ria/test_qualification_readiness.py tests/ria/test_qualification.py`.
Targeted Ruff and `git diff --check` passed.
Pinned tokenizer construction from authenticated `from_str` bytes also matched
`from_file` for the 129280-entry vocabulary and a bounded tokenization case.
This checks the changed input API, not full Engram normalization qualification.
A concurrently updated combined
inventory run produced 201 passes and one test regex failure: corrupted placement
correctly raised `JSON identity mismatch`; the assertion expected `digest`.
The primary owns that test and final integrated verification.

No checkpoint weights were loaded, GPU queried/initialized, CUDA kernel run,
target inference executed, or hardware qualification manufactured. The actual
development host is aarch64 GB10, not the required x86-64 GPU-free CPU host and
RTX 5090 sm120 client. Full-model fidelity on both axes, physical capacity and
NUMA/security/fault tests, controlled performance and the one-hour production
soak remain unexecuted. Initial admission must remain distinct from final release;
`tools/qualify_ria.py readiness` reports `release_ready=false` and exits 1 while
mandatory producer software is missing.
