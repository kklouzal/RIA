# Independent physical matrix and gate reports

`tools/ria/physical_contract.py` validates independent instrumented reports for
release matrix cells and G01–G28 gates. It executes no workload, supplies no
passing thresholds, and cannot generate physical qualification from synthetic
fixtures. Its tests are synthetic parser and contract fixtures only.

The independent producer must run the declared physical workloads and retain
their evidence. A selfhash authenticates bytes and references; **it is not
hardware attestation and cannot prove that execution actually happened**. The
operator must separately establish producer trust and genuine hardware execution.

The four strict schemas exported as `PHYSICAL_CONTRACT_SCHEMAS` are
`physical-runtime-identity`, `physical-contract-plan`,
`physical-contract-measurements`, and `physical-contract-evidence`. Every document
is revision 1 and selfsealed with the repository RFC 8785 digest convention.
Unknown fields, unknown kinds, duplicate JSON keys, and nonfinite numbers fail.

Plan, runtime identity witness, raw measurements, and final evidence bind the
same `policy_digest`, `logical_model_digest`, `source_lock_digest`,
`environment_digest`, `build_digest`, `operator_contract_digest`,
`runtime_config_digest`, `profile`, and `server_executor`. The policy/model/source
identities must match the frozen qualification policy. The identity witness
contains digests only; do not put credentials or configuration contents in it.
A proof covers one profile/executor realization. The 540-cell matrix can combine
different candidate proofs; every covered cell must match its proof's candidate.

Register the sealed plan after the policy and before execution, using exact UTC
timestamps (`YYYY-MM-DDTHH:MM:SSZ`). The plan freezes the child binary SHA256,
positive monotonic elapsed limit, required cells/gates, and named checks. Each
check declares a unit, numeric domain, inclusive minimum and/or maximum, and
assigned cells/gates. At least one bound is mandatory. Every required obligation
needs a planned check; assignments cannot expand coverage beyond the plan.
Units must match exactly; the validator performs no conversion. Domain `number`
requires a finite JSON number within the repository safe-integer convention.
Domain `u64` requires a canonical decimal string in `[0, 18446744073709551615]`
and compares integers exactly, including values above `2^53`.

Raw reports must identify the plan, a distinct run ID, child PID, actual binary
SHA256, UTC start, `CLOCK_MONOTONIC` elapsed nanoseconds, complete child lifetime,
successful exit code, and no terminating signal. Elapsed time must be positive
and within the declared limit. Every planned check has exactly one measurement
with a unique measurement ID, matching units/domain, and exact assigned coverage.
The final proof references the sealed plan and raw JSON files and associates each
result with its measurement ID and raw digest. `passed` is recomputed from the
plan's bounds for every check and for the entire proof. A consistent measured
failure is valid evidence with `passed:false`; it provides no passing coverage.

`validate_physical_contract(proof, evidence_dir, frozen_policy)` returns sorted,
deduplicated relative `{path,digest}` dependencies for staging. References are
canonical relative paths and every directory component and file is opened
without following symlinks. There are at most 4096 checks, measurements, and
total referenced files, 540 cells, and 28 gates. Parsing is bounded to 8 MiB and
250000 nodes per document, depth 16, and 64 MiB total referenced input plus the
proof and policy. No external contents are included in schema error messages.
The release aggregator additionally authenticates the top-level proof reference
and requires `passed:true` before crediting any claimed cell or gate.
