# Local artifacts and scoped deployment

These tools prepare provisioned local checkpoint files and consume measured
evidence. They never acquire weights. Offline builds and synthetic contract
tests do not qualify a GPU, NUMA host, model, release matrix or soak.

## Provisioned input and preparation

Install the hash-locked Python dependencies in an isolated environment:

```sh
python3 -m venv /absolute/path/ria-tools
/absolute/path/ria-tools/bin/pip install --require-hashes -r requirements-ria.txt
/absolute/path/ria-tools/bin/pip install --no-deps --require-hashes -r requirements-ria-reference.txt
```

The automatic recipe accepts the complete local
`nvidia/DeepSeek-V4.1-Flash-NVFP4` checkpoint at commit
`3431dde3247c13b5957f682b1e3c6fcae2566079`. Its directory must contain the exact
`config.json`, `model.safetensors.index.json`, and every indexed safetensors
shard, plus the pinned original `tokenizer.json`, `tokenizer_config.json` and
`chat_template.jinja`. The tool checks publisher LFS byte lengths and SHA256 identities,
bounded headers, index/header ownership and all 188245 tensor names. That pinned
index includes all 259 vision tensors; a hybrid checkpoint is unnecessary and
is not accepted by the automatic path. Compact original reference and tokenizer
metadata are pinned separately under `locks/metadata`, including the DeepSeek
commit `2cba9e42aa026125f3ed06c6d98c1db82f7ca027`.
The recipe, prepared root and operator contract use that native graph revision
as `source_revision`. The independent NVIDIA artifact revision remains bound
in calibration provenance and the logical model digest; it is not substituted
for the graph revision. Preparation rejects a mismatching operator/root revision.

Choose `nvfp4`, `fp8` or `bf16` explicitly. FP8 and BF16 conversion decode the
publisher's NVFP4 routed weights; widening does not recover unquantized master
weights. Plain BF16/FP32 client, shared, vision and learned Engram tensors retain
their declared source precision. Engram tables become authenticated row264
values plus exact prepared token-map/hash metadata. NVFP4 global weight and
activation factors come from the published positive FP32 scalar tensors. The
gate/up activation reduction spans the full original 384-expert population in
each layer before any client selection; its immutable calibration identity
binds the operator contract.

```sh
/absolute/path/ria-tools/bin/python tools/prepare_ria.py recipe \
  --source-dir /absolute/path/publisher-checkpoint --profile nvfp4 \
  --output /absolute/path/recipe.json
/absolute/path/ria-tools/bin/python tools/prepare_ria.py estimate \
  --source-dir /absolute/path/publisher-checkpoint --recipe /absolute/path/recipe.json
/absolute/path/ria-tools/bin/python tools/prepare_ria.py prepare \
  --source-dir /absolute/path/publisher-checkpoint --recipe /absolute/path/recipe.json \
  --role server --output-dir /absolute/path/prepared-server
/absolute/path/ria-tools/bin/python tools/prepare_ria.py verify /absolute/path/prepared-server
```

Use the estimate's disk, bounded scratch, physical padding and publication
requirements before preparing. The default 128 GiB shard cap permits the large
Engram table tensors; smaller tensor bundles stay coarse, while routed tensors
and scales occupy independently aligned 4 KiB physical spans for pre-touch NUMA
placement. Metadata is paged and independently authenticated. Every safetensors
offset is data-relative; the header determines the authenticated absolute
`data_start`. Full-file hashes and finite 4 MiB chunk hashes bind byte access.
Unrecognized required tensors, incomplete scales or a changed publisher shard
stop preparation. Whole source SHA256 and the header are verified on one
opened regular-file snapshot. Every subsequent calibration, conversion and
copy read checks device/inode/size/mtime/ctime before and after use; final
publication rechecks all source snapshots. Sources must remain immutable on
a trusted local filesystem throughout preparation. This detects same-size
changes, pathname replacement and mutate/restore without another full-bank
hash pass; privileged filesystem metadata forgery is outside that trust boundary.
Compact metadata is authenticated from the exact bounded bytes parsed or
published. Recipe and Engram provenance retain those byte identities rather
than rehashing potentially changed paths after using their contents.
Resume verifies completed outputs and publishes the root
manifest last. Reusing a completed output verifies its exact identity.
Existing output subdirectories and every ancestor must be real directories.
Preparation rejects directory symlinks before emitting shards or metadata;
publication holds directory descriptors so a later path swap cannot redirect
writes. New directory entries, completed file bytes and parent publication are
synchronized. Immutable shard publication never replaces a different existing
file, and comparison checks its bounded size and unchanged descriptor identity.

Extract a compact client package using the independently provisioned server
manifest digest, not an identity obtained from an unauthenticated peer:

```sh
/absolute/path/ria-tools/bin/python tools/prepare_ria.py client-package \
  --server-package /absolute/path/prepared-server \
  --trusted-manifest-digest REVIEWED_SERVER_MANIFEST_SHA256 \
  --output-dir /absolute/path/prepared-client
```

Optional `--selected-tensors /absolute/path/selected-tensors.json` supplies an
explicit JSON array of original tensor names for complete local experts.
Incomplete expert/cache closures fail. The package records excluded/inactive
inventory and aliases explicitly. Server-only Engram ranges never receive
client bootstrap grants. Selection does not change logical, tokenizer, encoding
or operator identity. Placement membership is separately frozen as sorted
`{layer,expert,tier,local_phases}` entries in the reviewed placement plan.

## Host baseline, bootstrap and probe

The deployment request conforms to `schema/deployment-request.json`. Supply
actual image digest/build information, private addresses, TLS credentials,
source-lock/model/operator identities, explicit caps/context, CPU mask,
finite cgroup/memlock limits and the reviewed seccomp profile. A CPU expert
image has no NVIDIA dependency; a CUDA role locks one physical GPU UUID.
Client arithmetic is CUDA even when its expert server uses CPU. The expert
policy selects actual sharded, replicated-experts or replicated-server-model
NUMA ownership, workers, per-node capacity and startup/runtime reservations.
The client placement plan supplies explicit runtime reservations and trusted
tokenizer paths/identities. Provision reviewed placement and peer-grants files
before probing: their sealed identities bind the physical environment. Adding
or changing either policy requires fresh evidence. No cache membership or
tolerance is inferred. The initial/final qualification scope is an audit label
on request, calibration and deployment lock; it does not change runtime
environment identity when the physical settings and policies are identical.

The planning request must declare `prefill_rows` from 1 through 64, no greater
than `context_positions`. Client placement must declare the identical
`runtime.prefill_rows`; the native memory plan preserves that bound. The
protected snapshots, expert groups, projection workspace, pinned staging and
endpoint result/wire storage are sized before admission with the same native
allocation equations used by serving. These persistent pools count throughout
the session; decode consumes one row. An earlier package without an explicit
microbatch is rejected. Regenerate the placement/planning documents, inventory,
bootstrap/probe/calibration evidence, memory plan and deployment lock whenever
the selected microbatch changes, then recreate the containers.
Native metadata loading also reserves separate input/DOM/duplicate-key and
retained-DOM/canonical-hash peaks. A sufficient final metadata size alone does
not admit a larger temporary parsing or hashing allocation.

On each actual native Linux x86-64 host, first record the real rootful Engine,
cgroup-v2 topology and intended host parent hierarchy:

```sh
/absolute/path/ria-tools/bin/python tools/host_preflight.py observe \
  --cgroup-parent /sys/fs/cgroup/ACTUAL_PARENT --output /absolute/path/host-report.json
```

For CUDA add `--gpu-uuid GPU-ACTUAL_CANONICAL_UUID`; for the client also add
`--client`. The report's parent chain excludes transient container usage.
Freeze separate client and expert deployment requests referencing their actual
host reports. Render each bootstrap package:

```sh
/absolute/path/ria-tools/bin/python tools/render_deployment.py bootstrap \
  --request /absolute/path/expert-request.json --output-dir /absolute/path/expert-bootstrap
/absolute/path/ria-tools/bin/python tools/render_deployment.py bootstrap \
  --request /absolute/path/client-request.json --output-dir /absolute/path/client-bootstrap
```

Provision the prepared models, bootstrap configuration and public certificate
files readable by runtime UID 10001, private keys readable only by their
authorized owner/group, and reports writable by UID 10001. General atomic
host-tool files default to mode 0644; staged deployment directories are 0700
and explicitly private checkpoints use 0600. Set private report directories and
the intended runtime ownership/group access before launch. The
authorized host controller must retain access to inspect and render them.
One host controller owner serializes each role through its persistent mode 0600
regular one-link `/run/lock/ria-ROLE.controller.lock` inode. Startup, execution,
stop and normal launch share this lock even across different configuration
directories. Acquisition uses a finite nonblocking wait, bounded by the request
management deadline (at most one hour); the inode is never unlinked.

Bootstrap emits `probe.json`, sealed `environment.json`, a controlled env file,
the validated effective Compose configuration and `admitted=false`. It does not
publish a service or memory plan. Start the managed fixed 1800-second idle
container and run the probe, separately on each host:

```sh
/absolute/path/ria-tools/bin/python tools/render_deployment.py fixture-start \
  --request /absolute/path/expert-request.json --output-dir /absolute/path/expert-bootstrap
/absolute/path/ria-tools/bin/python tools/render_deployment.py fixture-exec \
  --request /absolute/path/expert-request.json --output-dir /absolute/path/expert-bootstrap --run probe
```

Use the client request/bootstrap paths on the client host. The managed command
executes this exact native invocation inside the inspected idle container:

```sh
/usr/local/bin/ds4ctl probe --probe-config /etc/dwarfstar/probe.json \
  --output /artifacts/probe.json
```

The native probe verifies the image's actual build-info file and produces the
compact `/artifacts/probe.json` and sealed `/artifacts/probe.json.details.json`.
Keep both. Before any
qualification work, the owner must inspect the actual container image,
entrypoint, UID, restrictions, mount sources, listeners and physical GPU, and
verify its PID's hidden host ancestors:

```sh
/absolute/path/ria-tools/bin/python tools/host_preflight.py verify-container \
  --pid ACTUAL_CONTAINER_HOST_PID --host-report /absolute/path/host-report.json \
  --output /absolute/path/container-parent-proof.json
```

`fixture-start` and every `fixture-exec` perform those inspections automatically.
They also re-observe the complete stable host baseline, including current
kernel, Engine, Compose, GPU driver/device and NUMA topology. Any change requires
a fresh baseline and probe. Owned cleanup remains available after host drift.
The launcher freezes `/usr/bin/sleep 1800` with disabled health checks, publishes
only the existing reviewed service ports and refuses to replace an existing
container. The wrapper runs
inside the same inspected container. Its inherited-state preflight checks real
UID 10001, disabled dumps, zero core limits, no-new-privileges, seccomp mode 2,
finite exact memlock/cgroup bounds, zero effective swap, CPU/NUMA masks and
current headroom. Host ancestor inspection remains necessary because a private
cgroup namespace can hide tighter parents.

## Preregistered initial fixture evidence

Choose and review `schema/qualification-policy.json` tolerances, relative floors,
ordered objectives and minimum one-hour final soak before measurements:

```sh
/absolute/path/ria-tools/bin/python tools/qualify_ria.py freeze \
  --policy /absolute/path/policy-input.json --output /absolute/path/policy.json
```

Prepare `schema/fixture-registration-request.json` with exactly four runs:
`native_server`, `native_client`, `transport_server`, `transport_client`.
Their hard limits explicitly set owner host/device/pinned bytes, full child RSS,
startup, complete elapsed and per-case latency. Use target-shaped experts,
at least two measured repetitions, explicit warmup and seed, and the policy's
exact relative floor. Register each role's actual environment/build digest.
The transport bootstrap bodies freeze numeric addresses, deadlines, credential
paths and the actual peer certificate DER SHA256. Their credits reserve at
least 131328 control, 33160 expert, 432 row and 1024 bulk bytes, total at most 64 MiB.
`max_frame_bytes` and credits are canonical decimal strings.

```sh
/absolute/path/ria-tools/bin/python tools/qualify_ria.py register-fixtures \
  --input /absolute/path/registration-input.json --policy /absolute/path/policy.json \
  --server-build-info /absolute/path/server-build-info.json \
  --client-build-info /absolute/path/client-build-info.json \
  --server-environment /absolute/path/expert-bootstrap/environment.json \
  --client-environment /absolute/path/client-bootstrap/environment.json \
  --output /absolute/path/registration.json
/absolute/path/ria-tools/bin/python tools/qualify_ria.py fixture-requests \
  --registration /absolute/path/registration.json --policy /absolute/path/policy.json \
  --output-dir /absolute/path/fixture-requests
```

Registration seals the four request bodies without a preregistration reference,
then derives executable requests by adding the sealed registration digest.
Native request identity hashes its full JCS document; transport requests carry
a self-excluding JCS digest. This graph is acyclic. Build-info pins the actual
`ds4-ria-qualify` and `ds4ctl` binary hashes.

On each host place the sealed policy, registration and its role's derived
transport bootstrap in the report directory's `inputs/` subdirectory. They
are referenced as `/artifacts/inputs/policy.json`, `registration.json`, and
`transport_server-bootstrap.json` or `transport_client-bootstrap.json`.
The wrapper authenticates these bytes before use. Run the managed operations:

```sh
/absolute/path/ria-tools/bin/python tools/render_deployment.py fixture-exec \
  --request /absolute/path/expert-request.json --output-dir /absolute/path/expert-bootstrap --run native_server
/absolute/path/ria-tools/bin/python tools/render_deployment.py fixture-exec \
  --request /absolute/path/client-request.json --output-dir /absolute/path/client-bootstrap --run native_client
```

The fixed expert command inside the inspected container is equivalent to:

```sh
/opt/ria-qualification/bin/python /opt/ria-tools/qualify_ria.py run-fixture \
  --registration /artifacts/inputs/registration.json --policy /artifacts/inputs/policy.json \
  --run native_server --executable /usr/local/bin/ds4-ria-qualify \
  --build-info /usr/share/dwarfstar/build-info.json --environment /etc/dwarfstar/environment.json \
  --probe-config /etc/dwarfstar/probe.json --probe /artifacts/probe.json \
  --probe-evidence /artifacts/probe.json.details.json \
  --output-dir /artifacts/native-server
```

Run the same command on the inspected client container with `--run native_client`
and `--output-dir /artifacts/native-client`. The client binary executes the 29
CUDA graph/state/transfer/expert cases; server CPU executes two expert cases,
server CUDA five expert/transfer cases. No full checkpoint is needed.

Start the paired transport server command on the expert host, then its client
command on the client host within their registered deadlines:

```sh
/absolute/path/ria-tools/bin/python tools/render_deployment.py fixture-exec \
  --request /absolute/path/expert-request.json --output-dir /absolute/path/expert-bootstrap --run transport_server
/absolute/path/ria-tools/bin/python tools/render_deployment.py fixture-exec \
  --request /absolute/path/client-request.json --output-dir /absolute/path/client-bootstrap --run transport_client
```

The expert command inside the inspected container is equivalent to:

```sh
/opt/ria-qualification/bin/python /opt/ria-tools/qualify_ria.py run-fixture \
  --registration /artifacts/inputs/registration.json --policy /artifacts/inputs/policy.json \
  --run transport_server --executable /usr/local/bin/ds4ctl \
  --build-info /usr/share/dwarfstar/build-info.json --environment /etc/dwarfstar/environment.json \
  --probe-config /etc/dwarfstar/probe.json --probe /artifacts/probe.json \
  --probe-evidence /artifacts/probe.json.details.json \
  --transport-config /artifacts/inputs/transport_server-bootstrap.json \
  --output-dir /artifacts/transport-server
```

On the client use `--run transport_client`, its derived
`transport_client-bootstrap.json` and `/artifacts/transport-client`.
Each native child gets a finite process-group deadline, at most 2 MiB stdout and
64 KiB diagnostics, and an explicit RSS cap. `wait4` measures its complete
lifetime including report emission. The container cgroup enforces aggregate
wrapper/child resources; child RSS is not claimed to measure supervisor memory.
Nonzero exit, invalid raw output, exhausted bounds or deadline failure publishes
an owned `.failed` directory with bounded diagnostics and no qualified record.
The managed execution failure stops the whole owned container, including an
exec child whose client connection became uncertain. The idle lifetime bounds
abandoned containers. Stop and remove each completed owned fixture container:

```sh
/absolute/path/ria-tools/bin/python tools/render_deployment.py fixture-stop \
  --request /absolute/path/expert-request.json --output-dir /absolute/path/expert-bootstrap
```

Repeat on the client. The sealed ownership checkpoint permits reinspection and
resuming a still-running idle container; a stopped session needs a fresh
bootstrap directory. No learned model service starts during this sequence.

Copy all four completed run directories into one reviewed local evidence
directory, preserving immutable bytes. Produce five source-authenticated proofs
and the compact initial calibration separately for each admission role:

```sh
/absolute/path/ria-tools/bin/python tools/qualify_ria.py components \
  --registration /absolute/path/registration.json --policy /absolute/path/policy.json \
  --admission-role expert --native-server /absolute/path/native-server \
  --native-client /absolute/path/native-client --transport-server /absolute/path/transport-server \
  --transport-client /absolute/path/transport-client --output-dir /absolute/path/expert-calibration
```

Repeat with `--admission-role client` and a fresh output directory. The package
contains registration, all four raw reports and supervisor observations,
experts/graph-state/transfers/transport/resource-bounds proofs, aggregate
`calibration-evidence.json`, compact `calibration.json` and policy. Every case,
dimension, sample, required exact comparison, repeat/placement result, raw cap,
latency, authenticated paired payload and actual inherited restriction is
revalidated when consuming these proofs. A caller's `qualified=true` or freshly
recomputed SHA256 cannot replace that population. Initial evidence explicitly
has no full-model comparison, release matrix or soak claim.

## Admission and controlled launch

Generate `schema/inventory.json` from the authenticated role manifest and the
explicit deployment reservations:

```sh
/absolute/path/ria-tools/bin/python tools/build_inventory.py \
  --request /absolute/path/expert-request.json \
  --manifest /absolute/path/server-population/manifest.json \
  --output /absolute/path/expert-inventory.json
```

Repeat for the client manifest and request. This command reads authenticated
metadata only, without loading weight payloads, probing NUMA or initializing
CUDA. The serving loader and NUMA accountant derive physical tensor pages,
aliases, metadata, worker reservations and replicas. Explicit runtime budgets
cover the graph/frontend, library/admin stacks, pinned pools and workspaces;
choose and qualify these budgets before deployment. The inventory reserves
them conservatively across every phase and binds the context and policies.
It does not certify that the requested capacity is physically available.

Finalization regenerates this inventory and rejects edited allocations or
omitted phases, even when the edited document has a valid self-digest. The
Python renderer invokes the actual native `ds4ctl plan`; it has no second
tensor or NUMA memory equation implementation.
Finalize each role with its own evidence and an explicit reviewed placement
plan or peer-grants document where required:

```sh
/absolute/path/ria-tools/bin/python tools/render_deployment.py finalize \
  --request /absolute/path/expert-request.json --output-dir /absolute/path/expert-final \
  --probe /absolute/path/expert-probe.json --probe-evidence /absolute/path/expert-probe-details.json \
  --inventory /absolute/path/expert-inventory.json \
  --calibration /absolute/path/expert-calibration/calibration.json \
  --calibration-evidence /absolute/path/expert-calibration/calibration-evidence.json \
  --policy /absolute/path/policy.json
/absolute/path/ria-tools/bin/python tools/render_deployment.py launch \
  --request /absolute/path/expert-request.json --output-dir /absolute/path/expert-final
```

The launcher reauthenticates frozen settings, full calibration source graph,
actual effective Compose configuration, root manifest and image; validates
native configuration and the current stable host baseline before starting;
then checks actual container restrictions
and hidden host ancestors. Failed inspection stops the owned service within a
finite grace period. A changed image, source, build, caps, context, TLS/network,
cache policy or security settings invalidates the relevant evidence.

Bind may agree to smaller limits than the configured ceilings. The server
rejects a ceiling that cannot hold a mandatory registered operation, protected
progress or an authorized whole verification chunk. Expert and Engram callbacks
split within accepted limits, preserving original slots, coefficients and row
associations. A split failure publishes no partial callback result and retires
the pair. Credit accounting includes the larger of the success reply and the
16KiB typed-error envelope; the reviewed expert workspace allowance is separate.

Mount validation uses the frozen actual Compose version. Released Compose 2
through 2.40.3 omits a false `create_host_path` value from a present `bind`
object; that empty object is accepted for those versions. Missing/null `bind`
settings remain rejected. Compose 5 and unknown versions must emit an explicit
false value because their omission semantics differ. Source paths, access
flags and the exact mount population remain mandatory in either representation.

Initial admission records `qualification_scope=initial_fixture` and
`final_release_qualified=false`, permitting bounded full-bank/model qualification
after admission. Final release separately requires actual full-model fidelity
on both axes, all 540 required matrix cells, all 28 gate families and the policy's
physical soak. Initial operator fixtures never supply those claims.
