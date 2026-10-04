# RIA hardware handoff

RIA is being implemented on the pinned `antirez/ds4` source in this fork. The native
serving path is C/CUDA; Python handles artifact preparation, deployment and
supervised qualification. Select RIA with `--config service.json`. Existing
donor execution remains available when no RIA configuration is supplied.

No checkpoint weights were downloaded for this implementation. No model
inference, CUDA kernel execution, physical target probe, performance test or
engineering soak was run on the development host. Its ARM64/CC12.1 environment
is outside the declared Linux x86-64 target. Offline fixtures, static analysis,
CUDA compilation/disassembly and container builds establish separate evidence.
The [tracked verification report](ria-offline-verification.md) records their actual results; compilation
does not establish GPU numerical correctness.

## Read and provision

Use source branch `codex/ria-implementation` and read the
[independent review](ria-implementation-review.md) before hardware handoff.
The updated sources passed local and hosted frozen offline gates.
[Expert-grouped prefill](ria-prompt-prefill.md) is implemented; full semantic
physical qualification producers remain missing.

The preceding published implementation commit is
`3614e21aad6465c1af41a2fbeccad2991a6243d2`; later documentation-only commits leave its
runtime sources unchanged. Read the [full implementation once-over](ria-once-over.md).
The [hosted build](https://github.com/kklouzal/RIA/actions/runs/37215164710) passed all
three jobs and published these Linux amd64 images:

```text
ghcr.io/kklouzal/ria-cpu@sha256:54d515c2c0249b81b6f575317bd20287b3b34a888bea8d8082cead0f31c8c4be
ghcr.io/kklouzal/ria-cuda@sha256:7a6f2185e70315d2c740916d153490d591c60ebd55b96bbbd752a1b7e6a4e45a
```

Their immutable index/manifest/config hashes, source revision, UID 10001, SPDX/SLSA
descriptors and anonymous manifest access were verified without downloading filesystem
or attestation layers. The [publication
record](../locks/verification/once-over-publication.json) binds the source, jobs, images
and downloaded verification logs. Attestation contents/signatures were not verified.
These preceding images contain grouped prompt prefill and artifact/startup/lifecycle
corrections but support mutual TLS only. Optional trusted-network TCP requires the
new source build; verified replacement pins follow its successful publication. They are build-verified and remain unqualified for model/physical release.

Read the [updated specification](ria-specification.md) and
[implementation plan](../IMPLEMENTATION_PLAN.md). The user's canonical-base
correction selects `antirez/ds4`, recorded in [the source lock](../locks/source-lock.json).
Use these operational guides:

| Work | Guide |
| --- | --- |
| Source preparation, bootstrap, fixture provenance, admission, launch | [Artifacts and deployment](ria-artifacts-and-deployment.md) |
| Independent expert, graph and transfer fixtures | [Native fixtures](ria-native-fixtures.md) |
| Grouped prompt chunks, continuation, cancellation and memory admission | [Prompt prefill](ria-prompt-prefill.md) |
| CLI, authenticated HTTP, images, tools, continuation, saved logits | [Frontend](ria-frontend.md) |
| Preregistered HTTP replay, token timing and measured soak | [Replay and soak](ria-release-replay.md) |
| Missing physical qualification software and bounded diagnostics | [Physical contracts](ria-physical-contracts.md) |

Provision Linux x86-64 hosts, rootful Docker with Compose v2 and cgroup v2,
the pinned image digests, CPU/NUMA masks and measured memory/lock limits. The
client requires exactly one physical RTX5090, CC12.0 and the frozen UUID.
A CUDA expert server requires its frozen CC12.0 device/UUID; a CPU expert
server requires no CUDA device or driver. Native CUDA executables contain only
SM120a cubins, and serving disables PTX JIT.

Choose actual host/device/pinned caps, context capacity, finite operation and
shutdown deadlines, concurrency and CPU/NUMA ownership before qualification.
Do not substitute development-host capacities. Routed weight elements alone
require at least253.125GiB in NVFP4,506.25GiB in FP8 or1012.5GiB in BF16,
before scales, shared/nonrouted tensors, the full Engram population, replicas,
metadata, stacks and working memory. The native admission planner accounts
the complete declared population and reservations; these lower bounds cannot
serve as a deployment memory plan.

Select transport policy explicitly. For trusted-network TCP set `tls: {"enabled": false}` on both hosts/transport bootstrap inputs and use `expected_peer_name: null` in applicable model grants; no certificates are needed. For mTLS provision CA/certificates/keys, expected peer SAN names and matching model grants; transport fixture bootstrap also pins the opposite leaf DER fingerprint. Both modes require a client API bearer file. Mount credentials read-only at the documented
paths. Expert control/bulk listeners are private; the HTTP export is loopback.
The project never generates a shared default credential or embeds a secret
in an image, command argument or evidence record.

## Bootstrap before model admission

1. Prepare the pinned local source into a server population and compact client
   population, using the artifact guide. Full source hashes and original-domain
   calibration precede pruning. All three profiles are supported; BF16/FP8
   preparation from the pinned NVFP4 publication preserves that publication's
   information and does not recover unavailable original master weights.
2. Preregister accepted numerical tolerances, a positive relative-error floor,
   ordered objectives and real hard resource/timing budgets. The policy has
   no invented quality or latency defaults. Keep the two fidelity axes separate.
3. Capture host preflight and freeze each host-local bootstrap package against
   exact image/build/environment identities. Start only the bounded inspected
   qualification containers. Run bounded native probes inside those same
   containers; verify their hidden host cgroup ancestry from the host.
4. Register the joint four-run fixture population, derive requests, run the
   numerical/transfer and selected-mode paired transport qualifiers under supervision, and derive
   all five component proofs from their authenticated raw measurements.
   Qualification Python resides at `/opt/ria-qualification/bin/python`, with
   tools at `/opt/ria-tools`; it is not the learned model executor.
5. Stop the owned qualification containers. Finalize **initial fixture**
   admission with complete inventory, compact/raw probes, policy and raw-backed
   calibration evidence. Only then launch the full model/bank. There is no
   full-model or one-hour-soak dependency before initial admission.

Provision mount ownership explicitly: the runtime UID 10001 must be able to
read the immutable model/config/credential files and write only its report
directory. General atomic host-tool files default to 0644; deployment staging
directories are 0700 and explicitly private checkpoints/credentials have stricter
modes. Set the intended owner/group and private report-directory permissions
before launch; a readable package does not grant runtime write access.

Missing proofs, source/hash changes, cgroup or mask changes, changed images,
nonfinite results, insufficient resource headroom and failed quiescence stop
the affected operation. Fix the cause and obtain fresh evidence. A transport
failure retires the entire binding; continuation then requires fresh binding
and complete prompt replay.

## Qualify the implementation on the target

Run actual native GPU fixtures before full-model comparisons. Save teacher-forced
FP32 logits using the evaluator, and compare both the same-realization and
native-source axes against the frozen tolerances and shifted-label NLL.
Check causal boundaries, private-state restoration, failure/cancellation,
healthy continuation and text/reasoning/tools/PNG-JPEG features independently.

Create the explicit540-cell release matrix: three profiles, two server
executors, two client residencies, five expert placements, three phases and
three NUMA policies. Complete G01–G28 under their specified contracts and
retain the measurements. The replay harness observes HTTP features, prefix
reuse and exact token boundaries. It leaves placement/residency/NUMA cells
uncredited because HTTP observations alone do not prove those paths.
The complete matrix/gate producer software is currently missing. This is an
implementation blocker in addition to unavailable hardware, not an operator
provisioning task. Run `tools/qualify_ria.py readiness --output
/artifacts/qualification-readiness.json` for the fixed obligations, partial
producer commands and precise blockers; exit1 prevents full qualification.
Generic numeric reports are revision2 unqualified diagnostics and grant no
cell/gate coverage. Their validator authenticates bytes and recomputes numeric
bounds but cannot prove routing, residency, NUMA execution, fault safety or
the full gate semantics. See [physical qualification](ria-physical-contracts.md).

Prompt prefill now groups independent FFN rows by expert while attention and
private source history advance causally. Set mandatory `prefill_rows` between 1
and 64, no larger than context capacity. Regenerate the planning request, native
inventory, memory/placement plans, bootstrap/probe/calibration evidence and
deployment lock together; older packages lack the new workspace identity and
fail closed. The remaining semantic qualification software gap prevents a
complete-implementation claim.

Register and execute the release replay plan for at least the accepted soak
duration, minimum one hour. Use actual monotonic durations, failure counts,
throughput and TTFT/p95/p99/worst timing under the declared bounds. Instrumented
and uninstrumented performance are separate regions. The soak report does
not certify model fidelity or all release gates.

Final release admission requires the raw-backed initial components, both
fidelity comparisons, every physical matrix/gate proof, and authenticated
feature/soak executions with their frozen plans and workloads. Until these
pass, images and deployment records remain explicitly unqualified for release.
No hardware pass is inferred from an offline fixture or a successful build.

## Reproduce offline checks

Use CPython3.12 and the hash-locked dependencies, then install the separately
pinned static check tools:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install --no-deps --require-hashes -r requirements-ria-dev.txt
.venv/bin/python deploy/install_checks.py --output-dir build/ria/check-tools
PATH="$PWD/.venv/bin:$PATH" make -j2 ria-offline RIA_PYTHON="$PWD/.venv/bin/python"
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python deploy/offline_checks.py --output build/ria/evidence/offline.json
PATH="$PWD/.venv/bin:$PATH" make ria-sanitize RIA_PYTHON="$PWD/.venv/bin/python"
make ria-thread-sanitize
PYTHONPATH=tools .venv/bin/python -m ria.schemas
make -j2 ria-cuda
```

The thread sanitizer needs Clang 18 and its runtime package; its ASLR policy
applies only to the diagnostic process. The CUDA build needs the pinned
compiler/toolkit but no GPU. The workflow
`.github/workflows/ria.yml` runs offline checks and builds Linux amd64 CPU/CUDA
containers. Authorized pushes publish commit-tagged GHCR images with SBOM and
provenance; use the resulting immutable digest in a deployment request.
Standard GitHub-hosted runners are
[free for public repositories](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).
GHCR container storage and bandwidth are
[currently free](https://docs.github.com/en/billing/concepts/product-billing/github-packages).
The workflow selects standard runners and uses no paid external service.
