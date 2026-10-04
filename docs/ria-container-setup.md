# Container-managed RIA setup

Run one administrative `ria-setup` container on each physical host. The expert creates a private invitation; transfer that file to the client and start its setup container. The pair prepares and transfers the necessary model package, discovers actual host/image facts, generates configuration and credentials, executes the existing supervised initial-admission fixtures, exchanges their complete evidence, admits both deployments and starts the native services in order. The setup containers then exit; the services continue running.

This workflow replaces the file creation, identity transcription and evidence shuffling in the [advanced deployment runbook](../README.md#host-tools-and-directory-layout). It provides **initial fixture admission for supervised hardware testing**, not final release qualification. The [existing full-model/reference, semantic matrix and fault/soak software gaps](../README.md#qualification-limits) remain. No weights, GPU kernels or physical qualification were run while developing this automation.

## What remains an operator choice

Install rootful Docker on Linux x86-64 with cgroup v2. Install a supported NVIDIA driver and Container Toolkit on CUDA hosts; the client requires one physical RTX 5090/compute capability 12.0, and a CUDA expert requires the declared compute capability 12.0 target. A CPU expert needs no NVIDIA software. Follow the [host and driver prerequisites](../README.md#machines-images-and-prerequisites); no host Python, compiler, CUDA SDK or separate preparation dependency installation is needed for this workflow. The setup image includes Docker29.8.2, Compose5.5.1, CPython3.12, canonical CPU inventory tools and hash-locked preparation dependencies. It connects only to the local rootful Engine socket.

Choose the profile, context/prefill bounds, CPU/NUMA membership, RAM/VRAM/pinned-memory caps, runtime reservations, deadlines, API bounds and accepted numerical/fixture limits. Discovery validates these choices against actual resources; it does not guess spare memory, relax tolerances or increase failed limits. Images and settings are frozen for a job. Use the three matching immutable image references listed in the [README](../README.md#automated-container-setup).

The administrative image is named `ria-setup` in this guide and is published as `ria-cpu:setup-sha-COMMIT` in the existing public GHCR package. Its digest, root/controller entrypoint and role identity are distinct from the CPU service image. This avoids creating a new private-by-default registry package; no registry login is required for the verified public publications. Always select the exact setup digest from the README.

The expert needs either a complete pinned source checkpoint, a trusted prepared server package, or explicit permission to acquire the pinned source. The client receives only its required graph/metadata and explicitly selected local experts. With `local_experts: []`, routed experts stay remote. No manual model transfer, tokenizer-path derivation, certificate signing, API-token generation, image metadata extraction, deployment-request generation or registration-digest editing is needed.

## Generate and fill the settings

The following Bash commands are for the eventual hosts. Use a root administrative shell so private configuration, invitations, locks and workspaces have one owner. Set `SETUP_IMAGE` to the exact reviewed README digest, then use `set -euo pipefail` and `umask 077`. The examples use `/srv/ria-input` for reviewed inputs and distinct fresh job workspaces. Do not create renderer-owned subdirectories within a workspace.

```bash
install -d -m 0700 /srv/ria-input /srv/ria-expert-job
docker run --rm --network none "$SETUP_IMAGE" template \
  --role expert --executor cpu > /srv/ria-input/expert.json
docker run --rm --network none "$SETUP_IMAGE" qualification-template \
  > /srv/ria-input/qualification.json
```

For a CUDA expert, generate with `--executor cuda` instead. On the client host:

```bash
install -d -m 0700 /srv/ria-input /srv/ria-client-job
docker run --rm --network none "$SETUP_IMAGE" template \
  --role client --executor cuda > /srv/ria-input/client.json
```

Edit the JSON files and replace every required `null` with an accepted value. `gpu_uuid: null` means discover the unique supported physical device; `security.api_token_file: null` means generate a private client API token. Those two nulls are valid. The templates are intentionally invalid until policy choices are filled; they contain every required field from the authoritative [setup settings schema](../schema/setup-settings.json) and [setup qualification schema](../schema/setup-qualification.json). Unknown keys are rejected. JSON integer byte/ms fields and unsigned decimal-string reservation/ns fields are different contracts; the schema and tables below identify them.

| Settings | Meaning and accepted choices |
| --- | --- |
| `schema_revision`, `role`, `executor` | `1`; `expert` or `client`; `cpu` or `cuda`. Client must use CUDA. |
| `workspace` | Fresh absolute host job path, matching its container bind exactly; root-owned `0700`, without symlink components. |
| `service_image` | Matching immutable CPU expert or CUDA role image from the same publication as the setup image. Discovery pulls and validates its real build/source/entrypoint metadata. |
| `profile` | Explicit `nvfp4`, `fp8` or `bf16`; both hosts agree. Wider profiles retain the published NVFP4 source realization, not unpublished master weights. |
| `bind_address` | This host's reachable private numeric IPv4, for example `10.20.0.2` on expert and `10.20.0.3` on client. Loopback/public/multicast/link-local addresses are rejected. |
| `server_executor` | `cpu` or `cuda`, matching the expert. |
| `peer_address` | Optional client assertion of the invitation's expert IPv4, or expert assertion of the accepted client's IPv4; normally omit it and let pairing derive it. |
| `gpu_uuid` | Null for unique compatible-device discovery, or explicit canonical `GPU-…` physical UUID. Multiple compatible devices require an explicit UUID. CPU requires null. Production always binds the selected physical UUID. |
| `planning.context_positions`, `planning.prefill_rows` | Positive JSON integers; at most 1048576 positions and 64 grouped prefill rows, within admitted capacities. |
| `planning.caps` | JSON integer hard `host_bytes`, `device_bytes`, `pinned_bytes`, and `numa: [{"node": ID, "bytes": BYTES}]`. Selected NUMA nodes have positive byte caps; CPU device/pinned caps are zero. These are admission bounds, not automatic allocations or total-memory estimates. |
| `environment.cpuset` | Explicit CPU list/ranges, e.g. `0-15`; expert node CPU lists cover exactly this set. |
| `environment.cgroup_bytes`, `memlock_bytes`, `pids_limit` | Positive JSON integers for production memory, locked memory and process limits. Cover admitted memory/reservations and leave reviewed overhead headroom; pinned cap cannot exceed memlock or host cap. |
| `environment.start_period_seconds`, `stop_grace_seconds` | Positive startup and shutdown limits. Allow actual full-bank population before readiness. |
| `environment.api_port` | Host client API port, default template8000. The native container listens on8000; deployment publishes it only on host127.0.0.1 at this port. Expert does not expose an API. |
| `probe` | Explicit JSON integer `max_host_test_bytes`, `max_device_test_bytes`, `max_pinned_test_bytes` and `deadline_ms`; bounds do not exceed caps. CPU device/pinned tests are zero; CUDA tests are positive. These govern real deployed physical probes, not synthetic reports. |
| `network` timeouts | JSON integer `connect_timeout_ms`, `handshake_timeout_ms`, `operation_timeout_ms`, `frame_io_timeout_ms`, `write_timeout_ms`, each positive and at most3600000. Frozen into native transport. |
| `network` bounds | JSON integer `max_row_lookup_rows`, `max_inflight_payload_bytes`, `max_frame_payload_bytes`, `max_bulk_data_bytes`, `max_inflight_expert_requests`. Frame payload is at most16MiB; bulk data at most4MiB; bulk plus64 bytes fits frame, frame fits inflight credit; expert requests are1 or2. See the strict schema for all domains. |
| `deadline_ms` | Per administrative subprocess/ordinary peer-operation limit, positive integer ms, at most3600000. Preparation and transfers have their own bounds within the overall budget. |
| `max_transfer_bytes` | Positive integer total allowance for each exported/imported tree, at most5TiB. Set enough for the selected compact package and evidence, rather than the entire source bank. Transfers also cap file population16384, chunk4MiB and metadata8MiB. |
| `transfer_deadline_ms`, `setup_deadline_ms` | Positive integer total transfer/setup wall budgets; maximum24h per transfer and7days per setup job. Overall elapsed time includes initialization, preparation, waiting and launch. |
| `setup_port` | Expert setup listener, default9010, permitted1024–65535 except native ports7443/7444. Client follows the invitation endpoint. |
| `security.mode` | `tls` (default): automatic private CA and separate role keys/certificates; or explicit `trusted_network`: no PEM files, no encryption. Both hosts must select the same mode. There is no fallback. |
| `security.api_token_file` | Client only: null generates the API token; otherwise a private existing token file copied into the job. API bearer authentication remains required in either transport mode. |

On the expert, fill `expert` using the [expert resource contract](../README.md#create-each-hosts-deployment-request): `host_runtime_bytes`, `startup_host_bytes`, `device_workspace_bytes`, `pinned_workspace_bytes` are **unsigned decimal strings**; `projection_tile_rows` is an integer1–64. `nodes` contains integer `node`, integer-array `cpus`, positive integer `workers` and decimal-string `local_bytes`. Worker counts fit CPU membership, selected node IDs match NUMA caps, and reservations fit accepted startup/runtime/capacity limits. CPU device/pinned workspaces are `"0"`. CUDA uses the existing one-worker/device-workspace contract. Do not estimate CUDA transient storage with a new Python formula: the canonical native accountant checks the actual package and chosen configuration.

On the client, `client_runtime.runtime` contains decimal-string `tokenizer_memory_bytes`, `host_state_bytes`, `device_state_bytes`, `frontend_host_bytes`, and integer `projection_tile_rows`, `state_tile_rows`, `max_image_patches`. The first two tile bounds are1–4096 and image patches1–9216. `host_expert_cache_bytes`, `device_expert_cache_bytes` and `engram_cache_bytes` are JSON integer pool sizes. `local_experts` is an array sorted uniquely by `(layer, expert)`, with each entry containing integer `layer`0–39, integer `expert`0–383, `tier: "host"|"vram"`, and a unique nonempty `local_phases` array from `prefill`, `decode`, `continuation`. Use zero expert-cache pools and `[]` for an all-remote plan; reserve any Engram cache and runtime state explicitly. Placement covers complete selected expert tensor closures and keeps the full-reference schedule.

Client `api` contains integer `max_body_bytes`, `max_header_bytes`, `max_json_depth`, `max_json_nodes`, `max_messages`, `max_tools`, `max_images`, `max_encoded_image_bytes`, `max_decoded_image_bytes`, `max_http_connections`, `header_timeout_ms`, `body_timeout_ms`, `stream_write_timeout_ms`. These are the existing [frontend admission limits](../README.md#create-each-hosts-deployment-request), with native range checks. The controller derives the tokenizer file/hash, API bind/token path, one active generation, zero queued generations, empty CORS and disabled remote image URLs.

### Choose expert model provisioning

Replace the expert's `model` object with exactly one of these forms; paths and amounts are illustrative operator inputs:

```json
{"mode":"source","source_dir":"/srv/ria-source"}
```

Source mode verifies the complete publisher checkpoint at the locked revision and prepares the server bank. It may provision missing locked native tokenizer/chat originals and writes only those fixed files and its owned `.ria-recipes` subtree in the explicitly writable source root. Existing conflicting metadata is rejected without overwriting it. Optional integer preparation controls are `chunk_size` (default4MiB, maximum4MiB), `max_shard_bytes` (default128GiB) and `scratch_bytes` (default64MiB). These bound preparation; they do not choose runtime capacity.

```json
{"mode":"prepared","package_dir":"/srv/ria-prepared","trusted_manifest_digest":"REPLACE_WITH_REVIEWED_64_HEX_MANIFEST_DIGEST"}
```

Prepared mode verifies the whole existing server package against its trusted manifest identity, then derives the compact client package. It does not copy the large server bank or change its ownership. Make the package readable by runtime UID/GID10001 before use; the runtime bind root hides external ancestors. The digest is the manifest's canonical `digest`, not its file SHA256. See [preparation](../README.md#prepare-and-transfer-the-model).

```json
{"mode":"download","max_download_bytes":528000000000,"download_deadline_ms":345600000,"hf_token_file":null}
```

Download mode is explicit opt-in on the eventual expert host. It acquires only reviewed `nvidia/DeepSeek-V4.1-Flash-NVFP4` revision `3431dde3247c13b5957f682b1e3c6fcae2566079`:48 indexed shards totaling527293384576 bytes, locked configuration/index and native tokenizer/chat originals. Set the byte budget large enough for **all** source files, including metadata beyond that shard total. It hashes every source against its pinned identity, uses bounded HTTPS redirects and wall/storage limits, and never executes downloaded code. A private optional HF token is used only for authorized origin requests. Reserve disk for source, the selected prepared profile, compact export and transfer staging; profile expansion can exceed source size. Settings do not permit an arbitrary model URL/revision. The [acquisition review](../planning/setup-acquisition-review.md) documents exact pins and resume semantics.

### Fill the accepted qualification input

Only the expert takes `/srv/ria-input/qualification.json`; it preregisters one policy and four runs before measurements. Under `policy.thresholds`, fill **JSON numbers** for `same_realization` and `native_source`: `max_abs_error`, `max_relative_error`, `max_rms_error`, `max_loss_delta` are nonnegative; `relative_floor` is positive. Fill integer `minimum_soak_seconds` (at least3600) and a nonempty unique string-array `ordered_objectives`. Recording a soak minimum does not run or establish a soak.

`native` and `transport` require integer `deadline_ms`, `warmup`, `repeats` (at least2), `fixture_seed`. Transport also requires unsigned decimal strings for `max_frame_bytes`, `control_credit`, `expert_credit`, `row_credit`, `bulk_credit`, consistent with both settings' native bounds. Each of `hard_limits.native_server`, `native_client`, `transport_server`, `transport_client` requires positive **decimal-string nanoseconds** `max_elapsed_ns`, `max_startup_ns`, `max_case_latency_ns`. Optional stricter decimal-string `host_bytes`, `device_bytes`, `pinned_bytes`, `max_rss_bytes` override automatically derived bounds from each frozen role's accepted caps/cgroup; transport device/pinned bounds are zero. Overrides cannot exceed those caps. Numerical tolerances, repetitions, timing thresholds and objectives are never filled from a measured result after the fact. The [registration schema](../schema/fixture-registration-request.json) defines the complete resulting contract.

Validate settings without weights, Docker authority or hardware:

```bash
docker run --rm --network none \
  --mount type=bind,src=/srv/ria-input,dst=/srv/ria-input,readonly \
  "$SETUP_IMAGE" validate --settings /srv/ria-input/expert.json \
  --qualification /srv/ria-input/qualification.json
```

On the client, use `validate --settings /srv/ria-input/client.json`; an invitation is optional for validation and required for running. Repeated `--set 'existing.dotted.key=JSON-value'` overrides are accepted by `validate` and `run`, for example `--set 'security.mode="trusted_network"'` or `--set 'planning.prefill_rows=8'`. Values are JSON data, never shell/code evaluation; new unknown keys and list-index paths are rejected. Use a fresh workspace when changing effective settings.

## Start the paired setup

Authorize host firewalls for the expert's setup TCP9010 (or selected port) and native control/bulk TCP7443/7444 from the client. The client API is published only on127.0.0.1 at its selected host port; remote callers need an explicitly configured authenticated proxy or SSH tunnel, as described in the API runbook. The temporary transport fixtures use the registered native addresses. Containers do not change the host firewall or install host drivers. Transfer uses an authenticated pairing secret in both modes; trusted-network mode provides integrity/replay checks but **no confidentiality**. Possession of the invitation authorizes this one fixed setup job, so keep it private and move it through a trusted channel.

The following CPU-expert launch is complete for source mode. All external model/token inputs must use their **same absolute host/container paths**. Do not mount a source elsewhere inside the controller: later service binds refer to host paths. Parent mounts are permitted only with the same canonical ancestry. Source mode needs write access; prepared/token inputs may be read-only.

```bash
docker run --rm --init --name ria-setup-expert \
  --network host --pid host --cgroupns host \
  --mount type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock \
  --mount type=bind,src=/run/lock,dst=/run/lock \
  --mount type=bind,src=/sys/fs/cgroup,dst=/sys/fs/cgroup,readonly \
  --mount type=bind,src=/sys/devices/system/node,dst=/sys/devices/system/node,readonly \
  --mount type=bind,src=/srv/ria-expert-job,dst=/srv/ria-expert-job \
  --mount type=bind,src=/srv/ria-input,dst=/srv/ria-input,readonly \
  --mount type=bind,src=/srv/ria-source,dst=/srv/ria-source \
  "$SETUP_IMAGE" run --settings /srv/ria-input/expert.json \
  --qualification /srv/ria-input/qualification.json
```

Set `workspace` to `/srv/ria-expert-job`. For prepared mode replace the source mount with `src=/srv/ria-prepared,dst=/srv/ria-prepared,readonly`. Download mode needs neither source nor prepared mount; acquired files stay in its workspace. If using an external HF/API token, mount its exact absolute file path read-only (or keep it beneath the same-path private input bind).

For a CUDA expert, add `--gpus all -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=utility` before the image. These expose management inventory only; discovery picks the unique compatible device or validates the explicit `gpu_uuid`. A supported explicit physical UUID selector, index or single-device count may narrow the management view; production uses the actual selected UUID exclusively. CPU expert launch must have no `--gpus` or NVIDIA/CUDA environment variables. Keep host NVIDIA runtime constraints enabled.

The expert writes `/srv/ria-expert-job/invitation.json` (root-owned0600) and prints its **path**, never its secret. It opens the listener after local discovery; let it continue running. Copy that file through your trusted administrative channel to `/srv/ria-input/invitation.json` on the client and preserve root ownership/mode0600. Do not paste it into public logs, command arguments or Git. With TLS, only public CA/CSR/certificates travel between hosts; client private keys are created locally and the CA private key stays private on the expert.

```bash
docker run --rm --init --name ria-setup-client \
  --network host --pid host --cgroupns host \
  --gpus all -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=utility \
  --mount type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock \
  --mount type=bind,src=/run/lock,dst=/run/lock \
  --mount type=bind,src=/sys/fs/cgroup,dst=/sys/fs/cgroup,readonly \
  --mount type=bind,src=/sys/devices/system/node,dst=/sys/devices/system/node,readonly \
  --mount type=bind,src=/srv/ria-client-job,dst=/srv/ria-client-job \
  --mount type=bind,src=/srv/ria-input,dst=/srv/ria-input,readonly \
  "$SETUP_IMAGE" run --settings /srv/ria-input/client.json \
  --invitation /srv/ria-input/invitation.json
```

Set client `workspace` to `/srv/ria-client-job`. Use separate foreground terminals or your usual supervised background container launcher. Both controllers must remain alive through completion. They print stage names to stderr and one machine-readable result to stdout. Long preparation/transfer fits the declared total budgets; fixtures remain deliberately finite with the existing1800-second managed fixture lifetime. On any failed check, the operation fails and attempts to stop only its owned resources. It never fabricates a report, retries an unsuccessful measurement until it passes, silently changes capacity or promotes fixture admission to a final release.

The controller runs as root with explicit Docker socket/host PID/cgroup/network observation and private writable job/lock mounts. This is administrative host authority; run it only with reviewed settings/images. It does not require `--privileged`. The network worker drops to UID/GID10001 without management/CA-key descriptors and can request only fixed, locally authorized stages and allowlisted exports. The native model containers retain their existing non-root/read-only/capability/seccomp/mount/GPU restrictions and never receive the Docker socket. Setup authority disappears when the controllers exit.

## Use, inspect and stop

A successful result says `healthy: true`, `qualification_scope: "initial_fixture"`, `final_release_qualified: false`. Both hosts retain actual measurements, supervision, registration, calibration, inventory and final deployment locks under their job directories. The client API token is `/srv/ria-client-job/secrets/api.token`, private UID10001/mode0600; root can read it locally. Use the [authenticated prompt examples](../README.md#start-services-and-use-the-api) with that token and `127.0.0.1` at the selected host port on the client machine (or through your reviewed proxy/tunnel). The token is not printed by setup/status and is not transferred to the expert.

Inspect journal stage/status records without secrets:

```bash
docker run --rm --network none \
  --mount type=bind,src=/srv/ria-client-job,dst=/srv/ria-client-job,readonly \
  "$SETUP_IMAGE" status --workspace /srv/ria-client-job
```

Use `docker logs ria-setup-client` or `ria-setup-expert` while those containers exist; `--rm` removes their container logs afterward, so redirect stdout/stderr to private files if retention is needed. Persistent stage/evidence files remain in the workspace. SIGINT/SIGTERM request bounded owned cleanup, with exit130/143 respectively. Loss of one host's network can prevent acknowledgement of remote abort; the failure reports that ambiguity, and each host can stop its exact owned resources locally.

If bounded shutdown cannot quiesce owned controller work or prove its ownership state, the controller terminates with125 before releasing its leases to surviving work. Its private `fatal-containment.json` records ambiguous effects and failure context when publication is possible; no completion is published. Failed diagnostics cannot prevent termination. Reconcile the old workspace with its exact-owned `stop` operation before starting a replacement job. This is an explicit failure, not readiness.

To stop/remove the completed client's exact owned service, drain callers first, then run a controller with the same management mounts/namespaces/device settings and replace its `run ...` arguments with `stop --workspace /srv/ria-client-job`. Repeat on the expert using its recorded workspace. `stop` validates the original settings/current controller authority and recorded ownership; it does not delete model/evidence data or stop an unrelated replacement container. The API/admin drain details remain in the [operations runbook](../README.md#operate-drain-and-reconfigure). Stop an old owning workspace before launching another job for that role; role leases and existing-container checks prevent overlaps.

A completed-job rerun revalidates the current setup/source, admitted files, actual host/cgroup ancestry, exact original container and native health. It does not treat a historical completion receipt as current readiness. An **unfinished** controller restart contains its exact owned resources and requires a new workspace/invitation/job, preserving evidence and prepared data; reusing a failed measurement would invalidate preregistration. Profile, image, placement, bounds, credentials or host changes likewise require a fresh affected admission.

Standalone acquisition can resume verified partial source chunks without rerunning admission. After the old setup controller has stopped, choose accepted acquisition byte/time limits and run:

```bash
docker run --rm --init --entrypoint /opt/ria-setup/bin/python \
  -e PYTHONPATH=/opt/RIA/tools \
  --mount type=bind,src=/srv/ria-expert-job,dst=/srv/ria-expert-job \
  "$SETUP_IMAGE" -m ria.setup_acquisition --workspace /srv/ria-expert-job \
  --max-bytes "$ACQUISITION_BYTES" --deadline-ms "$ACQUISITION_DEADLINE_MS"
```

For a token, add its same-path read-only mount and `--token-file /absolute/private/token`. This tool needs HTTPS access but no Docker socket, GPU or host namespaces, and does not produce admission. Its result gives the verified `source_dir`; in a fresh setup job use `mode: "source"` with that exact root and add its same-path writable source bind, retaining the old source workspace. Only an **already completed prepared server package** can use `mode: "prepared"` with its trusted manifest digest. Keep each original owner workspace/data until no controller or service uses it.

For troubleshooting, inspect the named failed stage and its bounded diagnostics, then correct the root cause. Missing/mismatched mounts fail before effects; mismatched source/image/model/manifest/tokenizer/selection/security identities fail closed; physical allocation/NUMA/driver/probe failures require changed accepted hardware/settings and fresh evidence. The [strict schemas](../schema/setup-settings.json), [architecture contract](../planning/container-setup-contract.md), [acquisition review](../planning/setup-acquisition-review.md) and [advanced runbook](../README.md#troubleshooting) are the exhaustive references.
