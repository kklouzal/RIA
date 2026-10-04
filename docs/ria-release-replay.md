# Preregistered native HTTP replay and soak

`tools/run_ria_release.py` operates on exact, sealed request files. It uses literal
loopback HTTP only, sends sequential `/v1/chat/completions` requests, and never
retries. Workload repetition is a new scheduled cycle, not retry or recovery of
an ambiguous request. This host tool needs the pinned `requirements-ria.txt`
Python environment and Linux resource accounting. It does not load model
weights or invoke CUDA itself.

The authoritative contracts are `RELEASE_RUN_SCHEMAS` in
`tools/ria/release_runner.py`: `release-replay-plan-request`,
`release-replay-plan`, `release-workload-request`, and
`release-replay-evidence`. Register the policy first using `tools/qualify_ria.py
freeze`. A workload file is a sealed `{schema_revision:1,
kind:"release_workload_request",body:EXACT_REQUEST,digest:SHA256}` document.
Its body must select `DeepSeek-V4.1-Flash`, `stream:true`,
`stream_options:{include_usage:true}`, exact messages, and one positive finite
`max_tokens` or `max_completion_tokens`. The runner sends the RFC 8785 canonical
body without edits. Images must be inline PNG/JPEG data URLs; remote image URLs
and WebP are rejected.

The plan request binds policy/model/source, environment/build/operator/runtime
configuration identities, a relative `runtime_config_path` whose canonical JSON
identity must match `runtime_config_digest`, one deployed realization, each request path/digest,
ordered steps, actual expected feature/phase coverage and every hard limit.
There are no default performance thresholds. The operator must choose the
required request and overall deadlines, idle/connect limits, body/header/event
and population bounds, runner RSS cap, error allowance, completion/token floors,
throughput floor, and TTFT/p95/p99/worst timing limits before registration.
`retry_limit` is zero. Parser implementation ceilings supplement these declared
bounds. A fixture plan has `minimum_soak_seconds:0` and runs one ordered cycle.
A release plan requires all five features and all three phases, token timing
observation, and `minimum_soak_seconds >= policy.minimum_soak_seconds >= 3600`;
its global deadline must exceed that duration. Repeated cycles stop only after
the measured soak floor, failure, population exhaustion, or deadline.

```sh
python tools/run_ria_release.py --register-plan \
  --plan /artifacts/workload/plan-request.json --policy /artifacts/policy.json \
  --workload-root /artifacts/workload --output /artifacts/workload/plan.json
python tools/run_ria_release.py --validate-plan \
  --plan /artifacts/workload/plan.json --policy /artifacts/policy.json \
  --workload-root /artifacts/workload
```

Validation reads only bounded immutable plan/request files; it opens no socket
and does not read credentials. The registered plan must remain inside the
workload root. All request references reject symlinks and path traversal. Actual
HTTP execution requires the separate explicit command:

```sh
python tools/run_ria_release.py --execute \
  --plan /artifacts/workload/plan.json --policy /artifacts/policy.json \
  --workload-root /artifacts/workload \
  --bearer-token-file /run/secrets/client-api-token \
  --output /artifacts/replay-evidence.json
```

The bearer file must be a non-symlink regular file owned by the effective UID,
mode 0400/0600, one link, and contain at most 4096 printable ASCII bytes with an
optional final newline. The token never appears in logs, hashes, evidence, or
command arguments. Request/response text, reasoning and tool arguments also
remain private; evidence contains hashes, counters, static error codes and
timings. Exit 0 means the explicit operational checks passed; exit 1 means
measured failure; exit 2 means invalid input or setup failure. None of these
status codes declares model or full release qualification.

Set `ria_measurements:true` in each exact body and
`require_token_measurements:true` in the plan for token timing and prefix reuse.
The authenticated native frontend emits `event: ria_measurement` with canonical
decimal `index`, `elapsed_ns`, `previous_write_ns`, followed after final usage by
`event: ria_diagnostics` containing `phase`, `reused_prefix_tokens`,
`sampled_tokens`, `completion_tokens`, `write_ns`. The runner verifies event
ordering, count/usage association and terminal EOS accounting. Worker intervals
measure sampled-token boundaries and include previous network backpressure;
terminal EOS is reported separately from incorporated completion-token usage.
TTFT to first content delta and TTFT to first token measurement use client
monotonic observation after SSE event framing; this includes parser overhead
and is not a NIC arrival timestamp. p95/p99 use exact empirical nearest ranks. Ordinary buffered
UTF8/source-boundary deltas are reported as inter-delta gaps, never mislabeled as
token gaps. Instrumented requests form a separate observer region; their
numbers do not claim uninstrumented performance.

An exact-continuation step must immediately follow its named predecessor,
preserve all prior request messages, and append an assistant message whose
canonical normalized hash matches the predecessor's frozen expected response
hash. Normalization is `{role:"assistant",content:"",reasoning_content:"",
tool_calls:[]}` with actual fields filled. Actual continuation coverage also
requires native `phase:"continuation"` and positive observed reused-prefix
count. Declared prefixes alone never pass. Reasoning/tools require actual output
fields; images count only a successfully admitted inline image request, not
vision numerical quality.

Published evidence is sealed and binds its plan file reference. Only actual
release-mode responses observed over the complete declared monotonic soak can
produce `physical_replay_evidence`/`classification:"physical"`. Short or
incomplete runs produce unqualified `replay_measurements`. `passed` is derived
from every frozen bound; `qualified` and `hardware_qualified` stay false.
`validate_evidence` revalidates the plan/workloads, sequence, counters, durations,
coverage and bound judgments. `observed_workload_cells` records real phases
under the configured realization; `observed_cells` stays empty because current
HTTP diagnostics do not observe route/cache placement hits or actual
NUMA/residency execution. The complete instrumented matrix/gate producer and
semantic validator software is currently missing; see
[qualification software readiness](ria-physical-contracts.md). Generic numeric
diagnostics cannot credit those cells. No 540-cell or G01–G28 blanket pass is made.
Source/native model parity, model quality, all other release gates and physical
hardware/resource qualification remain separate required proofs.

Runner RSS is Linux `ru_maxrss` for its process lifetime. Optional `server_pid`
sampling records bounded RSS observations at completed request boundaries and
checks PID start-time identity; those samples are not a server peak bound.
`gpu_peak_bytes` is null. Full process/device/pinned/cgroup resource gates need
the independently supervised native evidence. Synthetic HTTP tests in
`tests/ria/test_release_runner.py` exercise the protocol and failure contracts
without a model or GPU; they cannot generate release qualification.
