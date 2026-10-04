# RIA hardware handoff

RIA is being implemented on the pinned `antirez/ds4` source in this fork. The native
serving path is C/CUDA; Python handles artifact preparation, deployment and
supervised qualification. Select RIA with `--config service.json`. Existing
donor execution remains available when no RIA configuration is supplied.

No checkpoint weights were downloaded for this implementation. No model
inference, CUDA kernel execution, physical target probe, physical/model
performance test or engineering soak was run on the development host. Its
ARM64/CC12.1 environment is outside the declared Linux x86-64 target. Bounded
model-free setup-transfer comparisons establish local software mechanism
evidence, separately from target LAN/model performance. Offline fixtures,
static analysis, CUDA compilation/disassembly and hosted container builds
establish separate evidence.
The [tracked verification report](ria-offline-verification.md) records their actual results; compilation
does not establish GPU numerical correctness.

## Read and provision

Use source branch `codex/ria-implementation` and read the
[independent review](ria-implementation-review.md) before hardware handoff.
The updated sources passed local and hosted frozen offline gates.
[Expert-grouped prefill](ria-prompt-prefill.md) is implemented; full semantic
physical qualification producers remain missing.

The latest corrected implementation commit is
`d0288edbaa35c55d3fae43988bc69568cbc5615e`; later documentation-only commits leave its
runtime sources unchanged. Read the [full implementation once-over](ria-once-over.md).
The [hosted build](https://github.com/kklouzal/RIA/actions/runs/37240824381) passed all
four jobs and published these matching Linux amd64 images:

```text
# Administrative setup controller:
ghcr.io/kklouzal/ria-cpu@sha256:d56c5bdf818ae76df742fd9329c77883e78072f35b1a5bef13fd19b60a6bf08a
# Native CPU expert:
ghcr.io/kklouzal/ria-cpu@sha256:bdffdedc410b434284bd51cbc1ddc10f4d1a952dd650f9e578b3cf685b313bb7
# Native CUDA client/expert:
ghcr.io/kklouzal/ria-cuda@sha256:331467089483f84cded050fc95e2f6671f3229512443cc795fcc3c621891434c
```

Their immutable index/manifest/config hashes, source revision, role/entrypoint,
UID10001 for native serving versus root for the administrative setup controller,
SPDX/SLSA descriptors and anonymous manifest access were verified without
downloading filesystem or attestation layers. The separate setup image uses
`ria-cpu:setup-sha-COMMIT` in the existing public package. The [publication
record](../locks/verification/container-setup-publication.json) binds the source, jobs, images
and downloaded verification logs. Attestation contents/signatures were not verified.
These images contain container-managed local/peer preparation, evidence exchange,
initial admission and ordered startup, grouped prompt prefill,
artifact/startup/lifecycle corrections and optional certificate-free
trusted-network TCP. CUDA roles use matching
NVIDIA NGC CUDA 13.4.2 builder/final bases, selected compiler/header/static-cudart
provenance and verified final-image ELF library closure. The
[toolchain guide](ria-cuda-toolchain.md) explains version pins and fresh admission
requirements. Automated mutual TLS is the default setup mode; an explicit
trusted-network setup needs no certificates. They are build-verified and remain
unqualified for model/physical release.

Read the [updated specification](ria-specification.md) and
[implementation plan](../IMPLEMENTATION_PLAN.md). The user's canonical-base
correction selects `antirez/ds4`, recorded in [the source lock](../locks/source-lock.json).
Use these operational guides:

| Work | Guide |
| --- | --- |
| Recommended two-host container-managed setup, configuration and recovery | [Automated setup](ria-container-setup.md) |
| Advanced manual preparation, bootstrap, fixture provenance, admission, launch | [Artifacts and deployment](ria-artifacts-and-deployment.md) |
| Independent expert, graph and transfer fixtures | [Native fixtures](ria-native-fixtures.md) |
| Grouped prompt chunks, continuation, cancellation and memory admission | [Prompt prefill](ria-prompt-prefill.md) |
| CLI, authenticated HTTP, images, tools, continuation, saved logits | [Frontend](ria-frontend.md) |
| Preregistered HTTP replay, token timing and measured soak | [Replay and soak](ria-release-replay.md) |
| Missing physical qualification software and bounded diagnostics | [Physical contracts](ria-physical-contracts.md) |

Provision Linux x86-64 hosts, rootful Docker and cgroup v2, the pinned image
digests, CPU/NUMA masks and accepted memory/lock limits. Automated setup bundles
Docker29.8.2, Compose5.5.1 and its Python/preparation/native inventory tools;
the advanced manual workflow needs the corresponding host tools. The
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

Automated setup defaults to automatic mutual TLS with separate role keys and
supports explicit `security.mode: "trusted_network"` without certificates.
It derives the native transport configuration, model grants, peer fingerprints
and private API bearer token. Its invitation is transferred once through a
trusted channel; it is not a public configuration file.

For the advanced manual workflow, select transport policy explicitly. For trusted-network TCP set `tls: {"enabled": false}` on both hosts/transport bootstrap inputs and use `expected_peer_name: null` in applicable model grants; no certificates are needed. For mTLS provision CA/certificates/keys, expected peer SAN names and matching model grants; transport fixture bootstrap also pins the opposite leaf DER fingerprint. Both modes require a client API bearer file. Mount credentials read-only at the documented
paths. Expert control/bulk listeners are private; the HTTP export is loopback.
The project never generates a shared default credential or embeds a secret
in an image, command argument or evidence record.

## Automated initial admission

Use the [complete setup guide](ria-container-setup.md). Fill the generated role
settings and accepted qualification limits, then start one administrative
setup container per physical host with the declared same-path workspace/input
binds and local Engine socket. Transfer the expert's private invitation to the
client once. The pair prepares or explicitly acquires the expert package,
transfers the compact client population, discovers actual host/image facts,
derives placement/configuration/grants, registers and supervises the existing
four fixtures, exchanges complete evidence and admits both hosts. It stops
fixtures before loading the model, starts the expert, verifies health and then
starts the client. Both setup containers exit after native service health.

The administrative controller owns root management authority; its network
worker drops to UID/GID10001 without Docker/CA authority, and native serving
containers retain their restricted UID10001 deployment with no Docker socket.
Status, exact-owned shutdown, failures and fresh-job restart are documented in
the guide. Resource/numerical/workload policy, host Engine/driver installation
and trusted invitation transfer remain operator responsibilities. This path
provides `initial_fixture` admission; it does not fill the remaining final
release qualification software gaps.

## Advanced manual bootstrap before model admission

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
sudo -n env "PATH=$PWD/.venv/bin:$PATH" make -j2 ria-offline RIA_PYTHON="$PWD/.venv/bin/python"
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python deploy/offline_checks.py --output build/ria/evidence/offline.json
PATH="$PWD/.venv/bin:$PATH" make ria-sanitize RIA_PYTHON="$PWD/.venv/bin/python"
make ria-thread-sanitize RIA_PYTHON="$PWD/.venv/bin/python"
PYTHONPATH=tools .venv/bin/python -m ria.schemas
sudo -n "$PWD/.venv/bin/python" tests/ria/setup_peer_benchmark.py --output build/ria/evidence/setup-peer-benchmark.json
sudo -n env "PATH=$PWD/.venv/bin:$PATH" make -j2 ria-cpu ria-cuda RIA_PYTHON="$PWD/.venv/bin/python"
```

The privileged offline invocation verifies actual administrative UID/GID and
capability drops; an unprivileged invocation skips those cases. The syntax
checker generates its caches in a private temporary evidence directory, so
prior root-owned source caches do not affect the later static gate. Native
CPU/CUDA builds use the same authority as the offline fixture build because
they share native object/output directories. The
transfer comparison uses isolated synthetic data and actual local worker/TLS
processes, without a checkpoint, Docker service or GPU operation.

The thread sanitizer needs Clang 18 and its runtime package; its ASLR policy
applies only to the diagnostic process. The CUDA build needs the pinned
compiler/toolkit but no GPU. The workflow
`.github/workflows/ria.yml` runs offline checks and a bounded amd64 setup-transfer
comparison, then builds Linux amd64 CPU/CUDA/setup containers. Authorized pushes
publish commit-tagged GHCR images with SBOM and
provenance; use the resulting immutable digest in a deployment request.
Standard GitHub-hosted runners are
[free for public repositories](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).
GHCR container storage and bandwidth are
[currently free](https://docs.github.com/en/billing/concepts/product-billing/github-packages).
The workflow selects standard runners and uses no paid external service.
