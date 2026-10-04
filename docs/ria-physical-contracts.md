# Physical qualification software and diagnostic contracts

**Full physical qualification software is incomplete.** The supplied native
fixtures, paired selected-transport tests and HTTP replay are runnable partial producers. No
supplied producer and semantic release adapter covers the complete 540-cell
route/cache/residency/NUMA matrix and G01–G28. Some offline gate obligations have
complete runnable tests; others have partial or missing producers. Adequate
hardware alone does not resolve the remaining missing software.
An operator is not expected to write the missing internal instrumentation or
fault harness as a provisioning step.

Run the bounded offline readiness check before arranging hardware qualification:

```sh
python tools/qualify_ria.py readiness --output /artifacts/qualification-readiness.json
```

It executes no workload, probes no GPU and loads no checkpoint. It writes a
selfsealed report and exits **1** while required producer software is missing.
The report lists the fixed G01–G28 obligations, their specification sections,
per-obligation available/partial/missing software with exact paths, all matrix
axes, supported partial producer commands and distinct software and external-input
blockers. For example, G02 has runnable schema/JCS/wide-offset tests; G25 has
locked builds and provenance tooling but lacks automated repeated-build output
comparison. Neither is falsely labeled as having no test software. A complete
authenticated gate receipt/semantic release adapter is still needed to integrate
those results into final qualification. The authoritative catalog is
`tools/ria/qualification_readiness.py`; it binds specification SHA256
`1720ef37b3bf1f561501293823973ef1475f65979d351804f45562fff5ce5767`.
Its digest is frozen into each diagnostic plan. The gate names are the plan's
traceability identifiers for the cumulative specification acceptance contract.

The existing execution paths remain useful:

- `tools/qualify_ria.py run-fixture` supervises the registered native expert,
  graph/state/transfer or paired transport fixture, checks its binary and
  environment identity, and bounds complete-child RSS, output and elapsed time.
  Follow [native fixtures](ria-native-fixtures.md) and the
  [bootstrap guide](ria-artifacts-and-deployment.md).
- `tools/run_ria_release.py --execute` runs registered authenticated HTTP
  workloads and measures actual native token timing, features, continuation
  and monotonic soak. Follow [replay and soak](ria-release-replay.md).
- `tools/qualify_ria.py compare-logits` independently compares saved candidate
  and reference logits on the frozen corpus and fidelity axis. It consumes
  logits; it does not execute a trusted native-source reference itself.

Those paths cannot certify a complete gate merely because they exercise part
of it. Missing software includes full-target reference/matrix execution with
actual routing, ownership, cache/migration and physical NUMA observations;
independent packed-state/oracle comparisons; all prescribed protocol race and
fault injection points; container pressure, GPU access loss and shutdown tests;
and matched native/container broad-routing performance orchestration with raw
resource, locality, power/thermal and uncertainty records. Their complete
positive and adversarial obligations are available in the readiness report.

External inputs remain necessary after implementing those producers: authorized
checkpoint and independent reference access; reviewed preregistered policies
and corpus; real Linux x86-64 CPU and RTX 5090/SM120 hosts; enough actual
per-node capacity for each required replica policy; a separately qualified
Blackwell expert; and administrator-controlled cgroups, firewall, GPU access
and owned container recreation/fault controls. Synthetic offsets, simulated
NUMA metadata, localhost transport or HTTP-only observations cannot supply them.

## Numeric diagnostic reports grant no release coverage

`tools/ria/physical_contract.py` retains strict validation of preregistered,
bounded numeric diagnostics. **Revision 2 is deliberately unqualified.**
All cell and gate coverage arrays are required to be empty; plan/evidence
`qualification_scope` is `unqualified_diagnostic`, and raw/evidence
`classification` is `unqualified_measurements`. Any claimed cell, gate or
`final_release` scope is rejected. Revision-1 generic physical proofs are not
accepted: assigning one unrelated numeric metric to every cell/gate was a
qualification loophole, and renaming the metric cannot repair its semantics.

The exported strict schemas remain `physical-runtime-identity`,
`physical-contract-plan`, `physical-contract-measurements` and
`physical-contract-evidence`. Every document is revision 2 and selfsealed with
the repository RFC 8785 convention. Unknown fields/kinds/revisions, duplicate
JSON keys and nonfinite numbers fail. Plan, runtime identity witness, raw and
evidence share policy/model/source/environment/build/operator/runtime/profile
and executor identities. The witness contains digests only; it is not proof
that the running configuration matched those digests.

Register the plan after the frozen policy and before execution, with exact UTC
timestamps. Freeze the child binary SHA256, a positive monotonic elapsed limit,
the fixed catalog digest, units/domains and inclusive minimum/maximum bounds.
Each check needs at least one consistent bound. Domain `number` requires a
finite JSON number under the repository safe-integer rules. Domain `u64`
requires a canonical decimal string in `[0,18446744073709551615]` and uses exact
integer comparison above `2^53`; no unit conversion occurs.

Raw records identify their plan, distinct run ID, PID, binary, UTC start,
`CLOCK_MONOTONIC` elapsed time, complete child lifetime, successful exit and no
signal. Each planned diagnostic has exactly one unique measurement. The final
report binds each result to its measurement ID and raw digest. Numeric pass
bits are recomputed from the frozen bounds; a consistent measured failure is
valid evidence with `passed:false`. A numeric `passed:true` is not model,
hardware, capacity, security, performance or release qualification.

`validate_physical_contract` returns authenticated relative dependencies for
staging diagnostics. Paths are canonical and every component is opened without
following symlinks. Parsing is bounded to 4096 checks/references, 8 MiB and
250000 nodes per document, depth 16 and 64 MiB total input. Error messages do
not include externally supplied contents. Selfhashes authenticate bytes; they
are not execution attestation or independent producer trust.

Full-release aggregation requires complete semantic coverage and therefore
fails closed with current diagnostic reports. Enabling a gate or cell requires
implementing its real producer and a validator that derives the fixed
obligations from the specific raw observations and independent oracles. Adding
generic numeric/Boolean assertions, trusting a selfsealed external pass flag,
or widening the coverage schema is insufficient.
