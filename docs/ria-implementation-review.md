# Independent implementation review

The second review found correctness defects and mandatory software gaps despite
the earlier successful offline checks. The prior claim of complete software
implementation is withdrawn. Hardware qualification remains unexecuted, and
missing implementation must not be described as merely awaiting hardware.

The audit starts from commit `bd76be469c45c6e672481125023090577b08cdf3`
and the original specification with SHA256
`14224cdb33476944111e14f69a5679f0597c192a44d67b48f048910f326f3f6e`.
Independent reviewers covered the model/operator/state graph, protocol and
concurrent ownership, and artifact/deployment/qualification chain. Integration
also checked the native allocation equations and operational handoff.

| Finding | Contract and correction | Regression evidence |
| --- | --- | --- |
| mHC premix crossed token boundaries | Source forward starts with the identity premix at every token; preserve cached attention/history while resetting that premix | Actual two-token graph command recorder, poisoned previous premix, independent pinned-source oracle |
| Sparse attention changed BF16 tile boundaries | Preserve source masked window slots and64-slot exponent rounding groups | Independent packed-value mathematical case and actual graph schedule recorder |
| Authenticated pair setup could stall indefinitely | Bound idle pre-Bind and missing bulk association | Real loopback mTLS fixtures without a model |
| Disconnected work could drain forever | Keep a finite retirement deadline until executor completion; terminate without releasing live ownership on expiry | Actual admitted synthetic work, both-channel loss and hung-worker child fixtures |
| Completed work could outlive a binding | Fence retained requests and validate original epoch/session before publication | Retained-completion/new-binding and credit fixtures |
| Bind/admission errors lacked typed terminal replies | Validate headers before replying; failed initial Bind retains zero identity; refusals return no unearned credit | Real socket Bind failures, coalesced Health refusals and direct credit accounting |
| Direct OpenSSL operations violated retry/error rules | Clear the error queue; preserve retry arguments; do not interleave reads with a pending write; close fatal transports without SSL shutdown | Stale-error and blocked-write socket fixtures |
| Failed CUDA drain released borrowed model backing | Retain unusable graph/expert/vision owners and source/pinned backing; serving owner exits before other teardown | Driver-free exact cleanup bodies, actual graph/engine callers and child fail-stop recorder |
| Responses could allocate the full negotiated maximum before association | Bound receiving allocation by the known response/error envelope and account the error envelope in shared credits | Tiny-result/error and credit boundary fixtures |
| Accepted smaller limits could fail on mandatory graph callbacks | Resolve registered operation and whole-chunk minima before readiness; split expert/Engram callbacks by negotiated frame/row/credit limits while preserving associations and atomic output | Six selected slots under64KiB frames,24 Engram rows under a one-row limit, late-group failure and inadequate Bind/granule fixtures |
| Source mutation could escape preparation verification | Bind source hashing, header validation and every conversion/calibration read to the same regular-file snapshot; recheck before publication | Same-size mutation, replacement and resumed-package adversarial fixtures |
| Admission depended on hand-authored allocation lists | Derive native metadata/NUMA populations and conservative explicit runtime reservations; regenerate before finalization | Actual metadata-only CLI, all NUMA policies, absent payloads, resealed omitted allocation/phase and cap failures |
| Generic numeric reports could claim arbitrary gates/cells | Restrict them to unqualified diagnostics; reject full coverage claims; publish explicit producer readiness | Synthetic false-credit reproduction and fail-closed report/schema tests |

No fixture in this review loads checkpoint weights, initializes a physical GPU,
launches a CUDA kernel, runs live model inference or probes target hardware.
CUDA compilation and instruction inspection are separate from numerical
execution. Final check results are recorded in the
[offline verification report](ria-offline-verification.md).

Two remaining implementation gaps prevent a complete-software claim:

1. Specification section7.3 requires bounded prompt chunks with independent
   rows grouped by expert and contributions scattered into original slots.
   `ria_graph_prefill` and engine prompt synchronization currently call the
   one-token graph repeatedly; the client callback is single-row. Implementing
   grouped prefill requires per-row CED/mHC/Engram state, query-specific
   source-candidate snapshots, a multirow client executor/callback, admitted
   microbatch workspaces, and independent sequential-versus-grouped checks.
2. The full540-cell matrix and complete semantic fault/gate producers and
   validators are missing. Existing numerical, TLS, HTTP replay, soak and
   host/deployment tools cover parts of these obligations. They cannot establish
   every residency, route, NUMA, full-bank, failure and source-model contract.
   [Producer readiness](../planning/qualification-software-readiness.json)
   distinguishes supplied software, partial coverage and missing software for
   every gate family. `tools/qualify_ria.py readiness` exits1 while blocked.

Physical GPU numerical comparisons, both full-model fidelity axes, target
capacity/NUMA/security/fault tests, controlled performance comparisons and the
minimum one-hour soak also remain unrun, as directed. Build-verified images
must remain explicitly unqualified for release.
