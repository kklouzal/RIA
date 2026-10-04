# Protocol and lifecycle implementation review

Reviewed base: `bd76be469c45c6e672481125023090577b08cdf3` in the working tree based on `antirez/ds4`. Specification: `docs/ria-specification.md`, SHA256 `14224cdb33476944111e14f69a5679f0597c192a44d67b48f048910f326f3f6e`; unchanged. Implementation frozen after the checks below, 2026-10-04. No commit made by this reviewer.

Scope: production protocol/binding and TLS implementations; remote callbacks; complete expert-server network/worker ownership; engine claim/reconnect/invalidation and relevant frontend generation, streaming disconnect, cancellation and drain callers. Consumed specification Sections 14–16 and related capacity, validation and frontend contracts; the prior planning review consumed original source ranges 611–1234 and 1928–2133 completely. No weights, inference, GPU probes or live deployment were used.

## Confirmed findings and fixes

| Finding before the changes | Severity and contract | Implemented containment/behavior |
|---|---|---|
| An authenticated peer could remain silent before control Bind, or finish control Bind without establishing bulk, and retain the sole client slot indefinitely. | P1: finite initial handshake/operation lifecycle; §14.9–14.10. | One finite initial pair deadline includes handshake, first Bind byte and missing bulk setup. It ends only when successful bulk Bind acknowledgment is fully published. Established idle bindings remain usable. |
| Ordinary channel loss canceled requests but did not start the executor drain deadline. A live executor could retain its buffers and service slot indefinitely while process-level drain was false. | P1: bounded cancellation and ownership; §14.7, §16 fatal executor policy. | Binding retirement has an independent drain deadline and takes readiness down while live work remains. Failure to quiesce terminates nonzero with borrowed arenas intact. Either control or established bulk loss follows this path. |
| Admission tested only unfinished work. A completed retained record could coexist with a replacement binding; its old canceled terminal response could then be stamped with the new binding identity. | P1: stale-session isolation and lifetime; §14.8–14.9, §16.4. | New control acceptance waits until every retained work record is disposed. Completion publication additionally checks original session and epoch. The demonstrated defect was a stale canceled response, not a demonstrated old activation disclosure. |
| Direct server SSL calls did not clear the thread error queue, mark fatal I/O unusable, or fence reads while a write required retry. | P1: OpenSSL native runtime semantics. | Clear errors immediately before SSL I/O; keep retry buffer/arguments stable and retry pending writes before another channel operation; fatal/abandoned partial I/O skips `SSL_shutdown`. Queued publication deadlines also apply when the socket never becomes writable. |
| Valid failed initial Bind and preadmission resource refusals closed without their required typed errors. | P0: required observable error/credit contract; §14.9–14.10. | Failed initial Bind sends a bounded typed error with zero session/epoch then closes. A preadmission refusal has no server pending record or terminal credit bookkeeping. Lightweight credit rejection can preserve the binding; required neural/row failures retire the pair after bounded error publication. Client Bind handling preserves the verified typed status. Corrupt identity/direction/order still closes without admission. |
| Remote replies were allocated against the full negotiated frame ceiling before association/shape validation, even for tiny expected replies. Small-success charge arithmetic also omitted the larger legal error alternative. | P1: allocation-before-validation and byte budgets; §14.8, §14.10–14.11, §15.3. | Receive cap is `min(agreed_frame, max(expected_success, 16384))`. Both endpoints use `request + max(success_reply, 16384) + 128 + workspace`; workspace remains 8 MiB for Expert/Shared and zero otherwise. Public table is in `protocol/schema.json` and `ria/remote.h`; independent tiny-operation numerical oracles were added. |
| A lower accepted frame/row limit could not execute the native six-slot or 24-row callback: each callback sent its entire selection in one request. | P0: lower negotiated limits and original associations; §14.5–14.6, §14.9. | Bind preselects group capacities from exact frame/count/credit bounds. Single-row expert callbacks preserve original IDs, slots, coefficients and input rows, assign a fresh invocation ID per group, and publish only after every group succeeds. Engram callbacks preserve original row associations across groups, likewise atomically. Multirow callers continue to provide already bounded units. |
| Bind could accept limits incapable of one registered operation, error envelope, authorized immutable chunk, or mandatory expert after protected progress. | P0: minimum viable admission; §14.9, §15.3. | Static registered expert/table and authorized whole-chunk minima are computed before listener readiness. Local startup and peer Bind validate them. Native one-entry 5120-wide request requires 20,544 bytes. Whole verification chunks retain the manifest hash/granule and are never arbitrarily split. The mandatory error floor is 16 KiB. |
| Failed CUDA destructor synchronization could still be followed by release of source host arenas, request buffers or Bank/NUMA storage. | P1: unproven DMA quiescence; §16 fatal executor policy. | The server exits before those releases if the CUDA destructor cannot prove quiescence, preserving the primary error code. Lower-level CUDA destructor corrections are owned by the model reviewer. |

Primary OpenSSL behavior was verified against [SSL_get_error documentation](https://docs.openssl.org/3.0/man3/SSL_get_error/): the error queue must be empty before I/O, WANT_WRITE retries prohibit intervening I/O, and fatal SSL errors prohibit shutdown I/O. The socket fixtures separately exercise those conditions.

## Changed surfaces and invariants

- `ria/server.c`: registered minima, finite pair setup, retirement ownership/deadlines, completion identity fences, typed rejection/retirement, TLS retry/fatal state, bounded publication, CUDA cleanup containment; worker arenas use shared `RIA_NUMA_WORKER_BYTES`.
- `ria/remote.c`, `ria/remote.h`: error-aware shared charges, bounded receive allocation, typed Bind errors, startup group selection and bounded publication staging, exact expert/row bounds before allocation, single-row slot/row splitting and invocation exhaustion retirement.
- `protocol/schema.json`: authoritative revision-1 credit formula and receive cap annotation.
- `tests/ria/test_server_lifecycle.c`: actual production server implementation callbacks and event loop with isolated mutually authenticated loopback sockets; controlled fault peers exercise real client RPCs. No model/executor startup or substituted neural path.
- `tests/ria/test_server_contracts.c`: independent charge constants for tiny Rows, Chunk, Cancel, Close and Expert, plus overflow and maximum-chunk arithmetic.

The graph thread owns SSL/binding/staging. Abort only shuts down stable descriptors; cleanup follows that owner's completion. Split publication pools are allocated once only when negotiated limits require them: at most 122,880 expert bytes and 6,336 row bytes. Single-group callbacks retain their direct validated publication path. Those endpoint allocations remain part of separate memory-plan admission, distinct from agreed protocol charges; target memory/performance evidence remains required.

## Completed offline checks

All recorded runs exited 0. The production lifecycle binary covers **34 cases**:

1. Silence before Bind; successful control Bind with missing bulk; established idle reuse; started-header deadline.
2. Stale completion isolation; retained-work acceptance fence; live-owner drain/readiness; actual admitted-request loss on either channel.
3. Poisoned OpenSSL error queue; fatal I/O unusable marking; a blocked write with incoming bytes, retaining the write retry and honoring its deadline.
4. Failed initial Bind for authorization and malformed schema; five coalesced Health requests against four progress slots; refusal bookkeeping; client preservation of authorization/resource Bind statuses.
5. Small Rows and tiny Chunk: successful response, legal full 16 KiB typed error, a four-byte UTF-8 codepoint crossing the error-message clipping boundary, and a 16,385-byte declaration rejected immediately before body read/application. Failed operations leave output unchanged; error clipping preserves valid UTF-8.
6. Six expert slots at a 64 KiB frame cap, preserving shuffled IDs/slots/coefficients across two generations and unique invocation groups; failure in a later group leaves output unchanged.
7. Twenty-four Engram rows with negotiated row limit one, including repeated row IDs and original associations; failure in a later group leaves output unchanged.
8. Bind refusal below the error floor, below one native expert unit, below an authorized whole chunk, and below protected-progress-plus-one-expert credit.

Evidence:

- Strict native lifecycle build and run: `build/ria/evidence/server-lifecycle.log`.
- ASan/UBSan lifecycle build and run, `detect_leaks=1:halt_on_error=1`, `UBSAN_OPTIONS=halt_on_error=1`: `build/ria/evidence/sanitize-server-lifecycle.log`.
- Existing production server contracts including bounded concurrent queue stress and new independent credit oracles: `build/ria/evidence/server-contracts-review.log`.
- Native JSON/wire/binding/admission/TLS regression binary: `build/ria/evidence/contracts-protocol-review.log`.
- Strict `RIA_WITH_CUDA` host syntax checks for server/remote passed; these execute no CUDA.
- Protocol JSON schema validation passed with `/tmp/ria-validation-venv/bin/python` and the pinned schema validator. `git diff --check` passed.

Expected fatal-drain fixtures terminate their isolated child nonzero while retaining simulated live ownership; they do not claim safe reclamation of a hung executor. No performance gain, physical network latency, GPU correctness, actual target NUMA placement or full model qualification is claimed from these fixtures.

## Remaining verification scope

No further confirmed software defect remains in this review's owned protocol/lifecycle scope after these changes. Root owns the integrated static/offline/sanitizer/race/build and hosted checks against the frozen tree. Hardware acceptance remains open: real CPU/CUDA executor drain and driver failure, production memory peaks (including selected publication pools), NUMA placement, numerical fidelity, target-scale channel behavior, tails and soak. Grouped prefill scheduling was deliberately not broadened by this correction; the existing multirow API still requires its caller to construct bounded independent units.
