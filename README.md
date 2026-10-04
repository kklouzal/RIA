# RIA — Remote Inference Architecture

This fork of [antirez/ds4](https://github.com/antirez/ds4) implements the native RIA path for **DeepSeek-V4.1-Flash**: a CUDA client runs the graph, attention/state, tokenizer and HTTP frontend; a separate CPU or CUDA expert service holds the prepared server bank in RAM and executes remote expert/Engram requests. The two machines communicate over control and bulk connections using either explicitly selected trusted-network TCP or mutual TLS. Python tools prepare artifacts and admit deployments; Python is not the model serving runtime.

**Current status:** candidate inference, grouped prompt prefill, container builds, physical probe/initial fixture admission, and a teacher-forced logit exporter are implemented. These images have passed offline build/static/sanitizer checks, but have not been qualified on the intended hardware. Complete full-model reference/corpus registration and execution, the semantic release matrix, and fault/soak gate producers/validators remain software gaps. Use `qualification_scope: "initial_fixture"` for supervised hardware testing; this is not a final release approval. See [qualification limits](#qualification-limits) before handoff.

This is the deployment runbook for **RIA safetensors containers**. The [original DwarfStar documentation](#original-dwarfstar-documentation) below describes separate donor GGUF/SSD paths.

- [Machines, images and prerequisites](#machines-images-and-prerequisites)
- [Host tools and directory layout](#host-tools-and-directory-layout)
- [Prepare and transfer the model](#prepare-and-transfer-the-model)
- [Network and credentials](#network-and-credentials)
- [Configure placement and peer authorization](#configure-placement-and-peer-authorization)
- [Create each host's deployment request](#create-each-hosts-deployment-request)
- [Bootstrap, probe and register calibration](#bootstrap-probe-and-register-calibration)
- [Run fixtures and finalize admission](#run-fixtures-and-finalize-admission)
- [Start services and use the API](#start-services-and-use-the-api)
- [Operate, drain and reconfigure](#operate-drain-and-reconfigure)
- [Troubleshooting](#troubleshooting)
- [Qualification limits](#qualification-limits)

## Machines, images and prerequisites

| Machine | Required execution environment | Container/Compose files |
| --- | --- | --- |
| Client | Linux **x86-64**, one visible physical **NVIDIA GeForce RTX 5090**, compute capability **12.0** | CUDA image; [compose.client.yml](deploy/compose.client.yml) |
| CPU expert | Linux **x86-64**, sufficient physical RAM and explicitly selected CPU/NUMA resources | CPU image; [compose.expert.yml](deploy/compose.expert.yml) |
| CUDA expert | Linux **x86-64**, one visible physical GPU with compute capability **12.0**, sufficient host RAM and device memory | CUDA image; expert base **then** [compose.expert-cuda.yml](deploy/compose.expert-cuda.yml) |

The client always uses CUDA, including when the expert uses CPU. CUDA binaries contain `sm_120a` AOT code, with no PTX fallback. Do not substitute ARM64, another GPU architecture, MIG identities or an arbitrary device ordinal. The selected physical UUID becomes the sole visible device, addressed as container device index `0`. Use the `GPU-` prefix and lowercase hexadecimal UUID consistently in preflight, requests and registration; the native fixture schema requires that spelling.

Use these immutable, published `linux/amd64` images with optional certificates and NVIDIA NGC CUDA **13.4.2** for GPU roles. Their runtime source is `d9d0b6e1692c005772358abdd5ed01e1fc03cb4a`:

```bash
CPU_IMAGE='ghcr.io/kklouzal/ria-cpu@sha256:11e89db1681fadb9aa092ec94078d42e850b4b5c5b3c5e879d89f1059c7160a4'
CUDA_IMAGE='ghcr.io/kklouzal/ria-cuda@sha256:dceb8971fa01fcdf474624cbafe87e3eec617c4ef78b35dc1e73f71445e47a09'
```

Both are publicly pullable. [The successful build/check run](https://github.com/kklouzal/RIA/actions/runs/37227441606), [publication identities](locks/verification/ngc-cuda-publication.json) and [handoff](docs/ria-handoff.md) record the verification scope. GHCR images include build metadata and published SBOM/provenance descriptors; descriptor presence does not establish verified attestation contents or signatures. Tags such as `latest` are not accepted deployment identities. Future images need their own build identities and fresh admission evidence.

Install a **rootful Docker Engine**, its **Compose v2 plugin**, and a host with **cgroup v2** using the [official Docker installation instructions](https://docs.docker.com/engine/install/). This orchestration uses the local `/var/run/docker.sock`; rootless, remote Docker contexts and Swarm are outside this path. Record actual Engine/Compose/kernel versions, rather than supplying example version strings.

On each CUDA host, install an NVIDIA driver that supports the selected image's CUDA runtime, then NVIDIA Container Toolkit using its [official installation instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Configure the rootful Docker runtime before recording the host baseline:

The current build recipes use **NVIDIA NGC CUDA 13.4.2**, the latest matching Ubuntu 24.04 release verified against the [NGC catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags) on 2026-10-04. The builder uses NGC `devel`; the final image uses the matching smaller NGC `base` plus its required host libraries. Both are digest-pinned in [container-lock.json](deploy/container-lock.json). Compilation and static CUDA runtime linking use that builder's toolkit, with compiler/header/archive hashes, exact SDK version checks and final-image dependency checks. Installing a newer host toolkit does not upgrade an existing image; no host CUDA SDK is needed to run the published container. The CPU image retains its CUDA-free Ubuntu base.

For CUDA 13.4, NVIDIA lists the **R615** corresponding driver branch in its [release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/#cuda-driver). [CUDA minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html) permits CUDA 13.x on R580 or newer with feature limitations; those combinations are not a qualified project floor. Choose a supported driver for the exact GPU/image and validate it through the physical probe. Keep the inherited NVIDIA driver constraints enabled. Toolkit or image changes require fresh numerical, resource and performance admission evidence, including vision GELU: NVIDIA changed `erff` behavior in CUDA 13.2. See [toolchain and image verification](docs/ria-cuda-toolchain.md).

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

The restart affects other containers on that daemon. On the physical hosts, inspect the GPU UUID, memory and topology, for example with `nvidia-smi --query-gpu=name,uuid,memory.total --format=csv`, and confirm the native probe accepts the exact device. A successful image build is not a driver/device compatibility test.

Choose `nvfp4`, `fp8` or `bf16` explicitly. With the pinned NVFP4 publisher checkpoint, FP8/BF16 routed weights are **widened from the published NVFP4 realization**; they do not recover original full-precision master weights. Routed weights alone are approximately **253.125 GiB / 506.25 GiB / 1012.5 GiB**, respectively. These are not total RAM requirements. Add Engram/non-routed tensors, scales, replicas, worker buffers, state, frontend/TLS overhead and startup transients. Use the estimator, native inventory, physical probes and admitted plan to choose limits. No unmeasured universal RAM, VRAM, latency or numerical-tolerance default is supplied here.

## Host tools and directory layout

The commands below are for the **eventual target machines**, not a requirement to download or execute a model on the development machine. Use Bash with `set -euo pipefail`; review host-specific values before running commands. Paths below are a consistent example. Every mounted/input path must resolve to an existing real local directory/file, without symlink substitutions.

Use one controller identity for the entire workflow on each host. This runbook uses a root administrative shell (`sudo -i`) for cgroup/proc inspection, Docker and protected credentials. Do not switch between root and a regular user during management: `/run/lock/ria-{expert,client}.controller.lock` is a persistent, single-link, mode `0600` file owned by that controller. Do not delete that lock to bypass a conflict.

Install Git, CPython **3.12**, venv/pip, a C compiler, GNU Make, OpenSSL development headers, libnuma development headers, curl, and OpenSSL/SSH/file-transfer utilities (rsync if selected). On Debian/Ubuntu the native build packages include `build-essential`, `libssl-dev` and `libnuma-dev`; install Python 3.12 using your distribution's supported method. Then, on both machines:

```bash
set -euo pipefail
umask 077
git clone --branch codex/ria-implementation https://github.com/kklouzal/RIA.git /opt/RIA
cd /opt/RIA
git checkout d9d0b6e1692c005772358abdd5ed01e1fc03cb4a
python3.12 -m venv /opt/ria-venv
RIA_PYTHON=/opt/ria-venv/bin/python
"$RIA_PYTHON" -m pip install --require-hashes -r requirements-ria.txt
"$RIA_PYTHON" -m pip install --no-deps --require-hashes -r requirements-ria-reference.txt
make -j2 ria-cpu
install -d -m 0750 -o root -g 10001 /srv/ria /srv/ria/secrets
install -d -m 0700 /srv/ria/operator /srv/ria/operator/evidence
install -d -m 0700 -o 10001 -g 10001 /srv/ria/reports
```

`/opt/RIA/bin/ds4ctl` is the **host CPU-built** inventory/planning tool, including on CUDA hosts. The container's CUDA binary is not a substitute for this host executable. Hash-pinned reference dependencies support local preparation/Engram derivation; these tools do not acquire checkpoint weights automatically.

| Host location | Meaning / required access |
| --- | --- |
| `/srv/ria/source` | Complete publisher checkpoint plus pinned native tokenizer/chat metadata; preparation can write its owned `.ria-recipes` subtree |
| `/srv/ria/model` | This host's verified prepared server or client package; mounted read-only at `/model` |
| `/srv/ria/secrets` | TLS mode: `ca.pem`, this role's `peer.pem`, `peer.key`; client `api.token` is required in both modes. The expert directory may be empty in trusted-network mode; mounted read-only at `/run/secrets` |
| `/srv/ria/reports` | Writable by UID/GID `10001:10001`; mounted at `/artifacts`; retain probe/raw fixture outputs |
| `/srv/ria/operator` | Reviewed requests, manifests/identities, build info, policies, transferred evidence; host controller only |
| `/srv/ria/bootstrap` | Renderer-created, temporary no-model fixture configuration |
| `/srv/ria/final` | Renderer-created admitted configuration; mounted read-only at `/etc/dwarfstar` |

Do not precreate renderer output directories; they are atomically published to new paths. All ancestors of model/config/secrets/reports must permit UID 10001 traversal. Secrets and generated config can be `root:10001`, directories `0750`, readable files `0640`; reports are `10001:10001`, directory `0700`. Prepared package directories/files must be readable by UID 10001. Grant only necessary group access; do not make keys/tokens world-readable or recursively change unrelated paths. The renderer initially creates private output directories; explicitly prepare their runtime permissions before container creation as shown later.

Pull the selected image on each physical host and extract its actual metadata. For a CPU expert use `CPU_IMAGE`; for client/CUDA expert use `CUDA_IMAGE`:

```bash
# Define CPU_IMAGE and CUDA_IMAGE exactly as above in this shell.
IMAGE="$CPU_IMAGE"  # Change to "$CUDA_IMAGE" on CUDA hosts.
docker --host unix:///var/run/docker.sock pull --platform linux/amd64 "$IMAGE"
docker --host unix:///var/run/docker.sock run --rm --runtime=runc \
  --network=none --read-only --entrypoint /bin/cat "$IMAGE" \
  /usr/share/dwarfstar/build-info.json > /srv/ria/operator/build-info.json
```

The metadata command runs `cat` with `runc`; it does not initialize CUDA. Keep this exact file. Its `digest` is `environment.build_digest`, and its `binaries` hashes are used for fixture registration. An OCI image digest, source commit, file hash or self-written build-info file is not interchangeable with that build digest.

## Prepare and transfer the model

Provision the **complete** `nvidia/DeepSeek-V4.1-Flash-NVFP4` safetensors checkpoint at revision **`3431dde3247c13b5957f682b1e3c6fcae2566079`** using an authorized checkpoint acquisition method on a machine with sufficient storage. It must contain `config.json`, `model.safetensors.index.json` and **every indexed shard**, including vision tensors. Do not combine arbitrary model revisions.

Use the pinned native DeepSeek-V4.1-Flash tokenizer/chat metadata at revision **`2cba9e42aa026125f3ed06c6d98c1db82f7ca027`**. The required files are already retained under [locks/metadata/deepseek-ai/DeepSeek-V4.1-Flash](locks/metadata/deepseek-ai/DeepSeek-V4.1-Flash). Copy these original files explicitly into the source root; converted wrappers are not originals:

```bash
cd /opt/RIA
NATIVE_METADATA='locks/metadata/deepseek-ai/DeepSeek-V4.1-Flash/2cba9e42aa026125f3ed06c6d98c1db82f7ca027'
for name in tokenizer.json tokenizer_config.json chat_template.jinja; do
  install -m 0600 "$NATIVE_METADATA/$name" "/srv/ria/source/$name"
done
"$RIA_PYTHON" tools/verify_ria.py --source-lock locks/source-lock.json
"$RIA_PYTHON" tools/prepare_ria.py recipe --source-dir /srv/ria/source \
  --profile nvfp4 --output /srv/ria/operator/recipe.json
"$RIA_PYTHON" tools/prepare_ria.py estimate --source-dir /srv/ria/source \
  --recipe /srv/ria/operator/recipe.json
"$RIA_PYTHON" tools/prepare_ria.py prepare --source-dir /srv/ria/source \
  --recipe /srv/ria/operator/recipe.json --role server \
  --output-dir /srv/ria/prepared-server
"$RIA_PYTHON" tools/prepare_ria.py verify /srv/ria/prepared-server
```

Change the profile at recipe creation if required; preserve it everywhere thereafter. Recipe construction hashes the complete source bank and produces derived calibration files under `SOURCE/.ria-recipes`; leave source tensor bytes unchanged. The source provenance distinguishes the NVIDIA publisher from the native model revision. Preparation defaults are 4 MiB chunks, 64 MiB scratch and a **128 GiB maximum shard** to accommodate large Engram tensors. Review the estimate before publication and budget source + prepared server + prepared client + temporary output storage. Resume verifies existing artifacts against the same identities; publication completes with the root manifest last.

Record the verified server manifest's **`digest`** through your trusted operator channel, then create the client package:

```bash
# Set this to the reviewed server manifest's digest, not its file sha256sum.
SERVER_MANIFEST_DIGEST='REPLACE_WITH_VERIFIED_64_HEX_DIGEST'
"$RIA_PYTHON" tools/prepare_ria.py client-package \
  --server-package /srv/ria/prepared-server \
  --trusted-manifest-digest "$SERVER_MANIFEST_DIGEST" \
  --output-dir /srv/ria/prepared-client
"$RIA_PYTHON" tools/prepare_ria.py verify /srv/ria/prepared-client
```

Omitting `--selected-tensors` is appropriate for an all-remote routed-expert setup. To place selected routed experts on the client, provide `--selected-tensors PATH` containing a JSON array of original tensor names; every selected expert must have its entire supported operator closure. Placement must match that client package. An all-remote configuration is a simple starting configuration, not a measured performance recommendation.

Transfer the **entire** server package to the expert's `/srv/ria/model` and the entire client package to the client's `/srv/ria/model`, preserving relative paths and bytes. Use SSH/rsync or another trusted transport; distribute each reviewed root manifest digest independently. On each receiving host run:

```bash
cd /opt/RIA
"$RIA_PYTHON" tools/prepare_ria.py verify /srv/ria/model
chown -R root:10001 /srv/ria/model
find /srv/ria/model -type d -exec chmod 0750 {} +
find /srv/ria/model -type f -exec chmod 0640 {} +
```

Compare the resulting manifest identity with the trusted digest before use. Keep both manifests available to the operator configuring placement/grants; they contain `logical_model_digest`, `operator_contract_digest`, `encoding_digest`, `layout_digest`, `tokenizer_digest` and provenance. Do not edit sealed manifests, metadata wrappers, indexes or shards.

The runtime tokenizer is **not** `/model/tokenizer.json`: metadata wrappers point to an opaque `metadata/<index>.bin`. After verification, derive its exact path on the client:

```bash
cd /opt/RIA
PYTHONPATH=tools "$RIA_PYTHON" - <<'PY'
from pathlib import Path
from ria.identity import read_json, verify_identity, within
root = Path('/srv/ria/model')
manifest = read_json(root / 'manifest.json')
verify_identity(manifest)
for reference in manifest['metadata']:
    wrapper = read_json(within(root, reference['path']))
    verify_identity(wrapper, reference['digest'])
    if wrapper.get('source_path') == 'tokenizer.json':
        if wrapper['sha256'] != manifest['tokenizer_digest']:
            raise ValueError('tokenizer identity mismatch')
        print('tokenizer_file=/model/' + wrapper['path'])
        print('tokenizer_sha256=' + manifest['tokenizer_digest'])
        break
else:
    raise ValueError('tokenizer metadata missing')
PY
```

## Network and credentials

Use a private routed network between machines. Permit expert TCP **7443** (control) and **7444** (bulk) from the authorized client host. Restrict those ports with the host/network firewall. The expert Compose binding is an explicit private **host IPv4 address**. The client API is published on **host loopback only**.

| Setting | Expert container | Client container |
| --- | --- | --- |
| `network.control_address` | `0.0.0.0:7443` | Actual expert private numeric address, e.g. `192.168.50.10:7443` |
| `network.bulk_address` | `0.0.0.0:7444` | Same expert address, port `7444` |
| `tls.expected_peer_name` (TLS only) | Client leaf SAN, e.g. `ria-client.internal` | Expert leaf SAN, e.g. `ria-expert.internal` |
| `network.server_executor` | Selected `cpu` or `cuda` | The **same expert executor**, even though the client itself is CUDA |
| `api.bind_address` | Not applicable | `0.0.0.0:8000` inside the container |
| Published API | Not applicable | `127.0.0.1:API_PORT:8000` on the client host |

The addresses/names above are illustrative; substitute your actual network and certificate identities. Network endpoints use numeric `IPv4:port` or `[IPv6]:port`; SAN identity is a separate TLS field. Do not bind inside the expert container to the host's private IP. Select the same transport mode for both roles/channels. TLS mode uses TLS 1.3, mutual certificates and disabled early data.

For a **trusted private network**, put this entire `tls` object in both deployment requests and both transport fixture bootstrap inputs:

```json
{"enabled": false}
```

That is all the transport credential configuration needed. Skip the CA/certificate/PEM steps below, leave the expert secrets directory empty, and set every applicable expert grant's `expected_peer_name` to **`null`**. The client still needs `api.token`. Do not leave unused certificate/SAN/fingerprint fields in the disabled object: the schema rejects them. Control and bulk must reach the same expert IP and originate from the same client IP; numeric peer pairing and the one-use bulk capability remain required.

This mode is ordinary TCP without encryption or cryptographic peer authentication; it relies on the network being trusted. Model/layout/placement grants, data validation, session/epoch binding, bounded credits, deadlines and HTTP API authentication still apply. The mode is frozen into admission evidence; mixed-mode peers fail and TLS errors never trigger a downgrade.

For **mutual TLS**, retain the certificate configuration below (existing configs default to TLS), optionally adding `"enabled": true`. Provision a trusted CA and **different client/expert private keys and leaves**. Leaves need the appropriate server/client authentication EKUs and DNS/IP SANs that match `expected_peer_name`; common-name-only identities are insufficient. Either use your existing issuer or, on a protected issuing machine, create a dedicated test CA and short-lived leaves. This example uses a 365-day CA and 30-day leaves; select validity/renewal appropriate to your testing window:

```bash
set -euo pipefail
umask 077
mkdir ria-issuer
cd ria-issuer
openssl req -x509 -newkey rsa:3072 -sha256 -nodes -days 365 \
  -keyout ca.key -out ca.pem -subj '/CN=RIA test CA' \
  -addext 'basicConstraints=critical,CA:TRUE' \
  -addext 'keyUsage=critical,keyCertSign,cRLSign'
for role in expert client; do
  openssl req -new -newkey rsa:3072 -nodes \
    -keyout "$role.key" -out "$role.csr" -subj "/CN=ria-$role.internal"
  cat > "$role.ext" <<EOF
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth,clientAuth
subjectAltName=DNS:ria-$role.internal
EOF
  openssl x509 -req -in "$role.csr" -CA ca.pem -CAkey ca.key \
    -CAcreateserial -out "$role.pem" -days 30 -sha256 -extfile "$role.ext"
  openssl verify -CAfile ca.pem "$role.pem"
  openssl x509 -in "$role.pem" -outform DER | openssl dgst -sha256
done
```

Record each leaf's **DER SHA-256** for the transport fixture's `authorized_peer_sha256`. It is not the hash of the PEM file. Deliver `ca.pem` and **only that host's** leaf/key to each host as `/srv/ria/secrets/{ca.pem,peer.pem,peer.key}` using a protected channel. Keep the CA private key on the issuing machine. Native TLS loading requires nonempty regular PEM files bounded to 256 KiB, a matching leaf/key and a valid trusted chain; use a noninteractive key rather than an encrypted-key password prompt.

On the client generate its HTTP bearer token directly into a protected file:

```bash
umask 077
openssl rand -hex 32 > /srv/ria/secrets/api.token
chown root:10001 /srv/ria/secrets/api.token
chmod 0640 /srv/ria/secrets/api.token
```

In TLS mode only, apply PEM permissions on both hosts:

```bash
chown root:10001 /srv/ria/secrets/ca.pem /srv/ria/secrets/peer.pem /srv/ria/secrets/peer.key
chmod 0640 /srv/ria/secrets/ca.pem /srv/ria/secrets/peer.pem /srv/ria/secrets/peer.key
```
 Token content must be 1–4096 printable ASCII bytes, optionally followed by LF/CRLF. Never put token/key contents in a deployment JSON, Compose environment, command argument, Git commit or log. The HTTP API uses bearer authentication over plaintext loopback HTTP; RPC mTLS does not secure that API. For remote access use an SSH tunnel or a separately reviewed authenticated TLS proxy.

## Configure placement and peer authorization

Create reviewed placement and grants before freezing environments. These are operator inputs, not discovered from an unauthenticated peer. Every field below is required. Byte counts are **JSON integers** unless marked **decimal string** (unsigned base-10, no unit suffix or leading zeros). Schemas reject unknown fields.

Client [placement-plan schema](schema/placement-plan.json):

| Field | Value/meaning |
| --- | --- |
| `schema_revision` | `1` |
| `logical_model_digest`, `operator_contract_digest` | Identities from the prepared manifests |
| `server_layout_digest`, `server_executor` | Expert manifest's layout; actual expert mode |
| `schedule`, `shared_placement` | `"full_reference"`, `"client"` |
| `expert_policy` | `"remote"` for all routed experts remote; `"explicit"` for reviewed local membership |
| `host_expert_cache_bytes`, `device_expert_cache_bytes`, `engram_cache_bytes` | Explicit bounded cache capacities; zero means no capacity, subject to native required-operation checks |
| `local_experts` | `[]` for all remote; otherwise sorted unique entries `{layer, expert, tier, local_phases}`, with `tier: "host"` or `"vram"`, layer 0–39, expert 0–383, and nonempty phase membership from `prefill`, `decode`, `continuation` |
| `runtime.tokenizer_file`, `runtime.tokenizer_sha256` | Exact derived `/model/metadata/...bin` path and tokenizer digest |
| `runtime.tokenizer_memory_bytes` | **Decimal string**, budget for tokenizer initialization/retention |
| `runtime.host_state_bytes`, `runtime.device_state_bytes`, `runtime.frontend_host_bytes` | **Decimal strings**, explicit state/frontend budgets |
| `runtime.projection_tile_rows`, `runtime.state_tile_rows` | Integer tile sizes, 1–4096; choose for the declared workload |
| `runtime.prefill_rows` | Integer 1–64, same as the planning request; prompt microbatch bound |
| `runtime.max_image_patches` | Positive integer, within native supported bound (9216) |
| `digest` | Generated JCS identity, as below |

Expert [peer-grants schema](schema/peer-grants.json): `schema_revision: 1`, `grants: [...]`, generated `digest`. Each grant requires:

| Grant field | Value/meaning |
| --- | --- |
| `expected_peer_name` | TLS: authorized client's SAN identity. Trusted-network TCP: **`null`**; that model grant applies to clients on the trusted network, without certificate identity |
| `logical_model_digest`, `operator_contract_digest`, `encoding_digest` | Exact shared prepared-model/operator/encoding identities |
| `client_layout_digest`, `placement_plan_digest` | Authorized client manifest layout and sealed placement identity |
| `profile`, `server_executor`, `server_layout_digest` | Selected profile, expert mode and expert manifest layout |

For the initial single-client setup use one grant. Generate the placement first, then use its digest when writing grants. To seal a reviewed **unsealed** placement/grants JSON (without `digest`), use the project's RFC 8785 canonical identity code:

```bash
cd /opt/RIA
PYTHONPATH=tools "$RIA_PYTHON" - placement-plan \
  /srv/ria/operator/placement-input.json /srv/ria/operator/placement.json <<'PY'
import sys
from ria.identity import atomic_json, read_json, seal
from ria.schemas import validate
kind, source, destination = sys.argv[1:]
body = read_json(source)
if 'digest' in body:
    raise ValueError('expected a reviewed unsealed input')
document = seal(body)
validate(kind, document)
atomic_json(destination, document)
PY
```

Repeat with kind `peer-grants` and your grants input/output. **Do not use `sha256sum file.json` as a document digest**: the identity hashes canonical content excluding its own `digest`. File-byte hashes, tensor hashes and DER certificate fingerprints have different contracts. Do not use this sealing helper to manufacture probe/fixture/qualification reports.

## Create each host's deployment request

Write `/srv/ria/operator/request-input.json` on each host using the [deployment-request schema](schema/deployment-request.json). The tables in this section cover every required input. Bounds in the linked schemas and stricter native checks remain authoritative. Memory/resource limits and numerical thresholds must be selected against your actual hardware, workload and independently accepted contracts; a string such as `REPLACE_...` is not a usable configuration.

First record the real host baseline. Determine the actual parent cgroup Docker uses from the daemon's cgroup driver/configuration and a container PID's `/proc/PID/cgroup`, then its corresponding `/sys/fs/cgroup/...` path. On systemd Docker this is commonly `/sys/fs/cgroup/system.slice`; **do not guess** that value or use the root cgroup instead. Compose does not override `cgroup_parent`. The managed launcher will compare the running container's ancestry and ancestor limits with this baseline.

On a fresh daemon, the following bounded, metadata-only container lets you inspect its actual default parent. It uses the already pulled role image, `runc` and `sleep`, with no GPU/model execution. Run in the root administrative shell; `--rm` also removes it after its 60-second lifetime if inspection fails:

```bash
set -euo pipefail
CGROUP_CONTAINER=$(docker --host unix:///var/run/docker.sock run --detach --rm --pull=never \
  --runtime=runc --network=none --read-only --no-healthcheck --user 10001:10001 \
  --cap-drop ALL --security-opt no-new-privileges:true --pids-limit 16 \
  --memory 64m --memory-swap 64m --entrypoint /bin/sleep "$IMAGE" 60)
CGROUP_PID=$(docker --host unix:///var/run/docker.sock inspect \
  --format '{{.State.Pid}}' "$CGROUP_CONTAINER")
CGROUP_PARENT=$("$RIA_PYTHON" - "$CGROUP_PID" <<'PY'
from pathlib import Path, PurePosixPath
import sys
pid = int(sys.argv[1])
if pid <= 0:
    raise ValueError('inspection container is not running')
rows = Path(f'/proc/{pid}/cgroup').read_text().splitlines()
paths = [row[3:] for row in rows if row.startswith('0::')]
if len(paths) != 1 or not paths[0].startswith('/') or '..' in PurePosixPath(paths[0]).parts:
    raise ValueError('expected one unified absolute cgroup path')
parent = Path('/sys/fs/cgroup') / str(PurePosixPath(paths[0]).parent).lstrip('/')
if not parent.is_dir():
    raise ValueError('actual parent cgroup is absent')
print(parent)
PY
)
docker --host unix:///var/run/docker.sock rm --force "$CGROUP_CONTAINER"
printf 'Actual Docker parent: %s\n' "$CGROUP_PARENT"
```

```bash
cd /opt/RIA
# Use CGROUP_PARENT from the inspection above, or independently verify it.
"$RIA_PYTHON" tools/host_preflight.py observe \
  --cgroup-parent "$CGROUP_PARENT" --output /srv/ria/operator/host-report.json
```

That command is for a CPU expert. A CUDA expert adds `--gpu-uuid GPU-...`; the client adds both `--gpu-uuid GPU-...` and `--client`. Use the exact physical UUID from that host. CPU/NUMA choices come from this report, not an example CPU range. A changed kernel, Docker, Compose, device or baseline requires refreshed evidence.

| Top-level field | Required content |
| --- | --- |
| `schema_revision`, `qualification_scope` | `1`, `"initial_fixture"`; final release is currently blocked |
| `host_report` | Absolute path to this host's actual report |
| `planning_request`, `probe_config`, `environment`, `tls`, `network` | Objects specified below |
| `native_ctl` | `/opt/RIA/bin/ds4ctl`, the host CPU executable |
| `compose_files` | Absolute ordered file list: client file; CPU expert base; CUDA expert base then CUDA override |
| `deadline_ms` | Controller operation deadline, integer 1–3,600,000 ms; native sub-operation deadlines must also remain within their bounds |
| `api`, `placement_plan` | **Client only:** API object and absolute path to sealed placement |
| `expert`, `peer_grants` | **Expert only:** expert object and absolute path to sealed grants |

`planning_request` ([standalone schema](schema/planning-request.json)):

| Field | Required content |
| --- | --- |
| `schema_revision`, `role` | `1`; `"client"` or `"expert"` |
| `executor`, `profile` | Client `"cuda"`; expert selected mode; one consistent `nvfp4/fp8/bf16` |
| `logical_model_digest`, `operator_contract_digest` | This role's verified manifest identities |
| `context_positions`, `prefill_rows` | Positive admitted context positions; prefill 1–64 and no greater than context |
| `caps.host_bytes`, `caps.device_bytes`, `caps.pinned_bytes` | JSON integer hard capacities; CPU expert device/pinned capacities are `0` |
| `caps.numa` | Array of `{node, bytes}` using actual NUMA IDs and integer per-node budgets |

The optional planning `digest` may be omitted. Budget host state, full bank/population, replicas and transients; leave cgroup headroom beyond admitted inventory for TLS, threads, tmpfs and runtime overhead. `memlock_bytes` must cover the locked population, separately from pinned-allocation caps. Swap is prohibited; setting a large number in JSON cannot create physical capacity.

`environment`:

| Field | Required content |
| --- | --- |
| `image`, `image_kind` | Exact immutable GHCR reference; `"cpu"` or `"cuda"` matching role/executor |
| `build_digest` | `digest` from that image's extracted `build-info.json` |
| `cpuset` | Canonical explicit host CPU mask, e.g. `0-7,16-23` **only if those CPUs are appropriate on your host** |
| `cgroup_bytes`, `memlock_bytes`, `pids_limit` | Positive finite integer bytes/bytes/process-thread limit |
| `gpu_uuid` | Actual physical UUID for CUDA; `null` for CPU expert |
| `model_dir`, `secret_dir`, `report_dir` | Existing absolute host paths to the directories described above |
| `seccomp_profile` | `/opt/RIA/deploy/seccomp-numa.json`; reviewed generated NUMA-aware profile |
| `bind_ip` | Actual private host IPv4; required in both requests, used to publish expert ports |
| `api_port` | Valid host TCP port; required in both requests, used only for client loopback publication |
| `start_period_seconds`, `stop_grace_seconds` | Positive finite health startup and stop/drain allowances |
| `docker_version`, `compose_version`, `kernel_version` | Exact strings from the actual host report |
| `host_report_digest`, `source_lock_digest` | Actual host report's `digest`; repository `locks/source-lock.json`'s `digest` |

`probe_config` ([schema](schema/probe.json)):

| Field | Required content |
| --- | --- |
| `schema_revision`, `role`, `executor` | `1`, actual role and its executor |
| `device_index`, `expected_gpu_uuid` | CUDA `0` and exact physical UUID; CPU `null` and `null` |
| `numa_nodes` | Nonempty list of actual selected NUMA node IDs |
| `max_host_test_bytes`, `max_device_test_bytes`, `max_pinned_test_bytes` | Explicit integer allocation-test bounds within selected capacities; CPU device/pinned bounds `0` |
| `deadline_ms`, `disable_core_dumps` | Positive bounded probe deadline; `true` |
| `environment_digest`, `build_digest` | Derived frozen environment identity and actual image build identity; generate below |
| `build_info_file` | `/usr/share/dwarfstar/build-info.json` inside the container |

`tls` is exactly `{ "enabled": false }` for trusted-network TCP. For TLS it is this object with the other role's actual SAN substituted; `"enabled": true` is optional:

```json
{
  "ca_file": "/run/secrets/ca.pem",
  "certificate_file": "/run/secrets/peer.pem",
  "private_key_file": "/run/secrets/peer.key",
  "expected_peer_name": "ria-expert.internal",
  "minimum_version": "TLS1.3",
  "early_data": false
}
```

`network`:

| Field | Required content |
| --- | --- |
| `control_address`, `bulk_address`, `server_executor` | Endpoints and expert mode from the network table |
| `connect_timeout_ms`, `handshake_timeout_ms`, `operation_timeout_ms`, `frame_io_timeout_ms`, `write_timeout_ms` | Explicit positive deadlines, at most 3,600,000 ms where enforced by native transport |
| `max_row_lookup_rows` | Bound on a single row lookup, integer 1–4294967295 |
| `max_inflight_payload_bytes`, `max_frame_payload_bytes`, `max_bulk_data_bytes` | Bounded payload/frame/chunk capacities; frame at most 16 MiB, bulk at most 4 MiB and no greater than frame minus 64 bytes; frame must fit inflight payload capacity |
| `max_inflight_expert_requests` | Integer **1 or 2** |

Negotiation can lower peer limits. With preparation's default 4 MiB chunks, bulk capacity must allow 4 MiB and frame capacity must allow at least 4 MiB + 64 bytes: authenticated chunks cannot be arbitrarily split smaller. Size framing, row splits and credits consistently; native validation rejects unsupported relationships rather than silently increasing capacities.

Expert-only `expert`:

| Field | Required content |
| --- | --- |
| `numa_policy` | `"sharded"`, `"replicated_experts"` or `"replicated_server_model"`; account for the full selected replication cost |
| `nodes` | Nonempty array of `{node, cpus, workers, local_bytes}`; actual node ID, CPU integer array, 1–64 workers, **decimal-string** local byte budget |
| `projection_tile_rows` | Integer 1–64 |
| `host_runtime_bytes`, `startup_host_bytes`, `device_workspace_bytes`, `pinned_workspace_bytes` | **Decimal strings**; CPU expert device/pinned values `"0"` |
| `drain_timeout_ms` | Positive finite runtime drain deadline |

Node/CPU memberships, per-node caps, probe nodes and container cpuset must agree. CPU IDs must be distinct across nodes, each node's workers cannot exceed its CPU count, and total workers cannot exceed 128. A **CUDA expert requires exactly one total worker** and positive device/pinned workspace reservations. Host runtime and per-node reservations must be positive; startup host reservation must cover runtime reservation, and all reservations must fit the planning caps. NUMA policy is process-specific; do not alter the host's global VM policy as a workaround.

Client-only `api` ([runtime semantics](docs/ria-frontend.md)):

| Field | Required content |
| --- | --- |
| `bind_address`, `bearer_token_file` | `"0.0.0.0:8000"`, `"/run/secrets/api.token"` |
| `max_body_bytes`, `max_header_bytes` | Positive integer HTTP byte bounds |
| `max_json_depth`, `max_json_nodes` | Positive parser nesting/node bounds |
| `max_http_connections` | Integer 1–64 |
| `max_messages`, `max_tools`, `max_images` | Explicit request-count bounds; messages/tools positive, images may be zero |
| `max_encoded_image_bytes`, `max_decoded_image_bytes` | Positive encoded/decoded image byte bounds |
| `header_timeout_ms`, `body_timeout_ms`, `stream_write_timeout_ms` | Explicit positive finite deadlines |
| `max_active_generations`, `max_queued_generations` | **`1` and `0`**; busy admission returns HTTP 429 |
| `allow_remote_image_urls` | **`false`**; only supported inline image data is accepted |
| `cors_allowed_origins` | **`[]`**; nonempty CORS configuration is unsupported in this baseline |

After all reviewed fields and sealed placement/grants exist, derive `probe_config` identities instead of typing a dummy hash. In `request-input.json`, the two derived probe fields may initially be omitted; this draft is not a valid deployment request until the following publication step:

```bash
cd /opt/RIA
PYTHONPATH=tools "$RIA_PYTHON" - <<'PY'
from ria.deployment import probe_configuration
from ria.identity import atomic_json, read_json
from ria.schemas import validate
request = read_json('/srv/ria/operator/request-input.json')
request['probe_config'] = probe_configuration(request)
validate('deployment-request', request)
atomic_json('/srv/ria/operator/request.json', request)
PY
"$RIA_PYTHON" tools/verify_ria.py --schema deployment-request \
  --document /srv/ria/operator/request.json
```

Before bootstrap, review both complete requests together: shared profile/model/operator/encoding, granted layouts, expert executor, network, peer identities, prefill/context and resource contracts must agree. Do not hand-author `service.json` or `memory-plan.json`; the finalizer generates them from verified evidence.

## Bootstrap, probe and register calibration

On **each** host, render temporary bootstrap configuration and make it readable by UID 10001. Prepare registration and distribute inputs before starting the finite-lifetime fixture container:

```bash
cd /opt/RIA
"$RIA_PYTHON" tools/render_deployment.py bootstrap \
  --request /srv/ria/operator/request.json --output-dir /srv/ria/bootstrap
chown -R root:10001 /srv/ria/bootstrap
find /srv/ria/bootstrap -type d -exec chmod 0750 {} +
find /srv/ria/bootstrap -type f -exec chmod 0640 {} +
```

Bootstrap emits `probe.json`, `environment.json`, role `.env`, effective Compose configuration and an **unadmitted** publication marker. It has no `service.json`/memory plan and cannot be used for ordinary model startup. Later, `fixture-start` creates an idle, no-model container with health disabled. The idle process has a **1800-second lifetime**; every registered run plus its supervision margin must fit the remaining lifetime. If it expires, stop it and create a fresh bootstrap output/evidence set; do not extend an old run's deadline after seeing results.

The controller checks actual mounts, UID, restrictions, image, host versions/GPU and cgroup ancestry. Probe results are `/srv/ria/reports/probe.json` and **`probe.json.details.json`**; retain both. Never replace an allocation measurement with a claimed amount of memory. Optional independent PID inspection is available through `host_preflight.py verify-container --pid PID --host-report REPORT --output OUTPUT`.

On a trusted coordinator freeze an independently reviewed qualification policy **before fixtures or model output comparisons**. Unsealed policy input ([schema](schema/qualification-policy.json)) contains:

| Field | Required content |
| --- | --- |
| `schema_revision` | `1` |
| `logical_model_digest`, `source_lock_digest` | Actual reviewed identities |
| `thresholds.same_realization`, `thresholds.native_source` | Each contains `max_abs_error`, `max_relative_error`, `max_rms_error`, `max_loss_delta` (nonnegative finite numbers) and `relative_floor` (positive finite number) |
| `minimum_soak_seconds` | At least `3600`; this declares a later gate, not a claim that a soak occurred |
| `ordered_objectives` | Nonempty ordered list of workload-specific objective strings |

Do not copy a test fixture's tolerance or tune thresholds after viewing results. `freeze` adds the policy digest:

```bash
cd /opt/RIA
"$RIA_PYTHON" tools/qualify_ria.py freeze \
  --policy /srv/ria/operator/policy-input.json --output /srv/ria/operator/policy.json
```

Exchange each host's exact `build-info.json`, bootstrap `environment.json`, and reviewed request through the trusted operator channel. Create one [fixture-registration-request](schema/fixture-registration-request.json), `/srv/ria/operator/registration-input.json`, for the pair. All fields below are required; use the selected profile/expert executor throughout.

| Registration field | Required content |
| --- | --- |
| `schema_revision`, `kind`, `qualification_scope` | `1`, `"fixture_registration_request"`, `"initial_fixture"` |
| `profile`, `server_executor` | Selected profile and expert mode |
| `policy_digest`, `logical_model_digest`, `source_lock_digest`, `operator_contract_digest` | Exact frozen policy and prepared/locked identities |
| `realizations.server`, `realizations.client` | Each `{environment_digest, build_digest}` from that role's sealed bootstrap environment/build info |
| `runs` | Exactly `native_server`, `native_client`, `transport_server`, `transport_client` |

Each native run contains `request_body` and `hard_limits`. The request body has every [native-fixture-request](schema/native-fixture-request.json) field **except** `preregistration_digest`, which the freezer derives:

| Native request field | Required content |
| --- | --- |
| `schema_revision`, `kind` | `1`, `"native_fixture_request"` |
| `profile`, `executor` | Selected profile and **expert executor**, including for the client runner |
| `role`, `runner_executor` | Server run `"server"` and actual expert executor; client run `"client"` and `"cuda"` |
| `gpu_uuid`, `expert_shape` | Actual runner UUID or CPU `null`; `"target"` for admission |
| `logical_model_digest`, `source_lock_digest`, `operator_contract_digest`, `policy_digest` | Shared reviewed identities |
| `environment_digest`, `build_digest` | This runner's actual realization identities |
| `host_budget`, `device_budget`, `pinned_budget` | Unsigned **decimal strings** (schema also permits safe integers); CPU device/pinned `"0"` |
| `deadline_ms`, `warmup`, `repeats`, `fixture_seed` | Deadline at most 600,000 ms; warmup 0–8; admission repeats **2–32**; explicit deterministic seed |
| `relative_floor` | Frozen policy's `thresholds.same_realization.relative_floor` |

Each transport run contains `request_body`, `hard_limits`, and `bootstrap_body`. The [transport-request](schema/transport-request.json) body omits both `preregistration_digest` and `digest`, which are derived after registration. Its input fields are:

| Transport request field | Required content |
| --- | --- |
| `schema_revision`, `kind` | `1`, `"transport_request"` |
| `policy_digest`, `logical_model_digest`, `source_lock_digest`, `operator_contract_digest` | Shared identities |
| `environment_digest`, `build_digest` | This runner's realization identities |
| `deadline_ms`, `warmup`, `repeats`, `fixture_seed` | Positive bounded deadline; warmup 0–100; admission repeats 2–1000; safe-integer deterministic seed |
| `max_frame_bytes` | **Decimal string**, 4096–16777216 |
| `control_credit`, `expert_credit`, `row_credit`, `bulk_credit` | **Decimal strings**, minimum 131328 / 33160 / 432 / 1024 bytes respectively; combined at most 67108864 |

Paired transport populations/seeds/frame/credit/warmup/repeat settings must agree. These admission fixtures need real simultaneous peers; local synthetic socket tests are not substitutes.

Every run's `hard_limits` contains **decimal strings**: `host_bytes`, `device_bytes`, `pinned_bytes`, `max_rss_bytes`, `max_elapsed_ns`, `max_startup_ns`, `max_case_latency_ns`. Match native byte budgets exactly and remain inside the deployment/probed caps; transport device/pinned values are `"0"`. Bound RSS within cgroup memory. Time caps must be positive, startup/case caps no greater than elapsed, and elapsed no greater than the declared deadline in nanoseconds. Choose explicit measured-workload objectives rather than silently treating every deadline as a performance target.

Each transport `bootstrap_body` has every [transport-bootstrap](schema/transport-bootstrap.json) field **except** `request_digest`, which is derived:

| Bootstrap field | Required content |
| --- | --- |
| `schema_revision`, `role` | `1`; server run `"expert"`, client run `"client"` |
| `environment_digest`, `build_digest`, `build_info_file` | Runner identities; `/usr/share/dwarfstar/build-info.json` |
| `network` | Exact deployment control/bulk addresses and five timeout fields; **no** deployment payload-cap/executor fields in this smaller object |
| `tls` | Trusted-network TCP: exactly `{ "enabled": false }`. TLS: `ca_file`, `certificate_file`, `private_key_file`, `expected_peer_name`, `authorized_peer_sha256`, optionally `enabled: true`; fingerprint is the **opposite peer's DER certificate hash** |

Transport bootstrap TLS does not take `minimum_version` or `early_data` keys. Both registered roles must select the same mode. Raw reports record `tls_enabled`; trusted-network reports have a `null` certificate digest and a numeric-peer pairing check, never a claimed mTLS check. Registration input does not take a guessed `binary_sha256`; the freezer authenticates it from the supplied actual build-info. Freeze the four-run plan and derive its requests:

```bash
cd /opt/RIA
"$RIA_PYTHON" tools/qualify_ria.py register-fixtures \
  --input /srv/ria/operator/registration-input.json --policy /srv/ria/operator/policy.json \
  --server-build-info /srv/ria/operator/expert-build-info.json \
  --client-build-info /srv/ria/operator/client-build-info.json \
  --server-environment /srv/ria/operator/expert-environment.json \
  --client-environment /srv/ria/operator/client-environment.json \
  --output /srv/ria/operator/registration.json
"$RIA_PYTHON" tools/qualify_ria.py fixture-requests \
  --registration /srv/ria/operator/registration.json --policy /srv/ria/operator/policy.json \
  --output-dir /srv/ria/operator/fixture-requests
```

Distribute the same sealed `policy.json` and `registration.json` to both hosts. Place those files under each `/srv/ria/reports/inputs`, along with that role's derived `transport_server-bootstrap.json` or `transport_client-bootstrap.json` from `fixture-requests`. Preserve those exact filenames. Give UID 10001 read/traverse access to inputs; keep report output writable. Do not copy another host's private key. The managed fixture executor derives the registered native/transport requests itself.

After the coordinator's files have been securely transferred to the same operator paths on each host:

```bash
install -d -m 0750 -o root -g 10001 /srv/ria/reports/inputs
install -m 0640 -o root -g 10001 /srv/ria/operator/policy.json /srv/ria/reports/inputs/policy.json
install -m 0640 -o root -g 10001 /srv/ria/operator/registration.json /srv/ria/reports/inputs/registration.json
# Expert host; on the client use transport_client-bootstrap.json instead:
install -m 0640 -o root -g 10001 \
  /srv/ria/operator/fixture-requests/transport_server-bootstrap.json \
  /srv/ria/reports/inputs/transport_server-bootstrap.json
```

## Run fixtures and finalize admission

With policy/registration/transport inputs prepared on both hosts, start the managed fixture container and execute the actual allocation/topology probe on **each host**:

```bash
cd /opt/RIA
"$RIA_PYTHON" tools/render_deployment.py fixture-start \
  --request /srv/ria/operator/request.json --output-dir /srv/ria/bootstrap
"$RIA_PYTHON" tools/render_deployment.py fixture-exec \
  --request /srv/ria/operator/request.json --output-dir /srv/ria/bootstrap --run probe
```

Successful registration does not replace the probe: fixture execution still verifies its actual measurement/evidence and selected allocation bounds. Allow all subsequent runs to finish inside each container's remaining 1800-second lifetime.

On the expert:

```bash
cd /opt/RIA
"$RIA_PYTHON" tools/render_deployment.py fixture-exec \
  --request /srv/ria/operator/request.json --output-dir /srv/ria/bootstrap --run native_server
```

On the client, run the same command with `--run native_client`. Then start `--run transport_server` on the expert **first**, and promptly run `--run transport_client` on the client in a separate terminal, within the registered connection/deadline bounds. Both commands are synchronous and must complete successfully. Raw results/supervision live under `/srv/ria/reports/native-server`, `native-client`, `transport-server`, `transport-client` on their respective hosts. Do not launch model serving alongside the fixture container.

Stop both temporary containers through their controllers:

```bash
"$RIA_PYTHON" tools/render_deployment.py fixture-stop \
  --request /srv/ria/operator/request.json --output-dir /srv/ria/bootstrap
```

Consolidate **all four complete run directories**, including raw reports, supervision, registration/environment/proof files, into a trusted coordinator's `/srv/ria/operator/evidence/{native-server,native-client,transport-server,transport-client}`. Copying only a compact `passed` summary is insufficient. Preserve original bytes and directory structure. Produce separate role calibration bundles:

```bash
cd /opt/RIA
for role in expert client; do
  "$RIA_PYTHON" tools/qualify_ria.py components \
    --registration /srv/ria/operator/registration.json --policy /srv/ria/operator/policy.json \
    --admission-role "$role" \
    --native-server /srv/ria/operator/evidence/native-server \
    --native-client /srv/ria/operator/evidence/native-client \
    --transport-server /srv/ria/operator/evidence/transport-server \
    --transport-client /srv/ria/operator/evidence/transport-client \
    --output-dir "/srv/ria/operator/$role-calibration"
done
```

Transfer the **entire role-specific calibration directory** to that host, e.g. `/srv/ria/operator/calibration`. It includes `calibration.json`, `calibration-evidence.json`, policy/registration/raw reports and independently checked component proofs. Finalization verifies the full local evidence graph. Keep the local probe and its details from the actual role.

On each host generate metadata-only native inventory and finalize:

```bash
cd /opt/RIA
"$RIA_PYTHON" tools/build_inventory.py --request /srv/ria/operator/request.json \
  --manifest /srv/ria/model/manifest.json --output /srv/ria/operator/inventory.json
"$RIA_PYTHON" tools/render_deployment.py finalize \
  --request /srv/ria/operator/request.json --output-dir /srv/ria/final \
  --probe /srv/ria/reports/probe.json \
  --probe-evidence /srv/ria/reports/probe.json.details.json \
  --inventory /srv/ria/operator/inventory.json \
  --calibration /srv/ria/operator/calibration/calibration.json \
  --calibration-evidence /srv/ria/operator/calibration/calibration-evidence.json \
  --policy /srv/ria/operator/policy.json
chown -R root:10001 /srv/ria/final
find /srv/ria/final -type d -exec chmod 0750 {} +
find /srv/ria/final -type f -exec chmod 0640 {} +
```

The final directory contains admitted `service.json`, `memory-plan.json`, `deployment-lock.json`, `environment.json`, `host-report.json`, role `.env`, `compose-effective.json`, role placement/grants, and copied qualification evidence/publication state. **Do not modify generated bytes**. Permission preparation is allowed; changing authenticated content invalidates admission. If any allocation, schema, identity, hard limit or comparison fails, fix the underlying configuration and regenerate affected evidence. There is no documented bypass.

## Start services and use the API

Launch the expert first through the managed launcher, wait for it to become healthy, then launch the client:

```bash
cd /opt/RIA
"$RIA_PYTHON" tools/render_deployment.py launch \
  --request /srv/ria/operator/request.json --output-dir /srv/ria/final
```

Run this command on each corresponding host. It repeats host/container inspection and checks the deployment lock. A successful `started` response means the container was created; it is not proof that bank population/model initialization has finished. Health becomes ready only after initialization. Choose the finite startup allowance to accommodate the actual bank/IO workload; that Compose allowance is not a certified startup-performance deadline.

Use the exact final configuration for all Compose management. This shell function on each host prevents ambient environment variables from overriding the sealed `.env` values. Set **one** role/file selection:

```bash
# Client host:
ROLE=client
COMPOSE_FILES=(-f /opt/RIA/deploy/compose.client.yml)
# CPU expert host instead:
# ROLE=expert
# COMPOSE_FILES=(-f /opt/RIA/deploy/compose.expert.yml)
# CUDA expert host instead:
# ROLE=expert
# COMPOSE_FILES=(-f /opt/RIA/deploy/compose.expert.yml -f /opt/RIA/deploy/compose.expert-cuda.yml)
ria_compose() {
  env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin LANG=C LC_ALL=C \
    docker --host unix:///var/run/docker.sock compose \
    --project-directory /srv/ria/final --env-file "/srv/ria/final/$ROLE.env" \
    "${COMPOSE_FILES[@]}" "$@"
}
ria_compose ps
ria_compose logs --tail 100 "$ROLE"
ria_compose exec -T --user 10001:10001 "$ROLE" \
  /usr/local/bin/ds4ctl health --socket /run/dwarfstar/admin.sock --timeout-ms 5000
```

Exit status is meaningful: an unready/failed health check is not readiness. The admin socket is mode `0600` inside the runtime tmpfs and owned by UID 10001; use that user. `ds4ctl validate --config /etc/dwarfstar/service.json` checks configuration, not model readiness.

The renderer writes these variables; they are **outputs**, not an invitation to override the plan:

| Generated setting | Source |
| --- | --- |
| `ROLE_IMAGE`, `ROLE_CPUSET`, `ROLE_CGROUP_BYTES`, `ROLE_MEMLOCK_BYTES`, `ROLE_PIDS_LIMIT` | Frozen image/resource settings; `ROLE` is `CLIENT` or `EXPERT` |
| `ROLE_MODEL_DIR`, `ROLE_CONFIG_DIR`, `ROLE_SECRET_DIR`, `ROLE_REPORT_DIR` | Exact host mount roots |
| `ROLE_START_PERIOD`, `ROLE_STOP_GRACE` | Finite configured allowances, with seconds suffix |
| `ROLE_GPU_UUID` | CUDA roles only |
| `EXPERT_BIND_IP`, `CLIENT_API_PORT` | Respective expert private publish/client loopback port |
| `SECCOMP_PROFILE` | Frozen NUMA-aware seccomp file |

The expert command is `ds4-expert-server --config /etc/dwarfstar/service.json`; client is `ds4-server --config /etc/dwarfstar/service.json`. Compose supplies them and the UID, read-only root, dropped capabilities, no-new-privileges, finite memory/memlock/PID limits, disabled cores/swap, bounded tmpfs and required binds. CUDA expert needs **both** Compose files; do not start CPU and CUDA alternatives concurrently. Use `render_deployment.py launch` for startup rather than a direct `compose up` that skips admission/host inspection.

The client exposes authenticated `GET /v1/models` and `POST /v1/chat/completions`. Model ID is **`DeepSeek-V4.1-Flash`**. It supports native prompt rendering, streaming SSE, admitted reasoning/tools/inline-image request forms and one active generation. Tools are returned for your application to execute; the server does not execute them. Remote image URL fetching is disabled. Accepted fields, limits, response/error/streaming semantics are documented in [ria-frontend.md](docs/ria-frontend.md); this is a defined subset of the OpenAI-style API, not every endpoint/parameter.

On the client host, with the published API port (example `8000`), keep the token out of process arguments by supplying headers on stdin:

```bash
set -euo pipefail
API_URL=http://127.0.0.1:8000
{ printf 'Authorization: Bearer '; tr -d '\r\n' < /srv/ria/secrets/api.token; printf '\n'; } | \
  curl --fail-with-body --header @- "$API_URL/v1/models"
cat > /srv/ria/operator/chat.json <<'JSON'
{"model":"DeepSeek-V4.1-Flash","messages":[{"role":"user","content":"Hello"}],"temperature":0,"max_tokens":32,"stream":true}
JSON
{ printf 'Authorization: Bearer '; tr -d '\r\n' < /srv/ria/secrets/api.token; printf '\n'; } | \
  curl --fail-with-body --no-buffer --header @- --header 'Content-Type: application/json' \
  --data-binary @/srv/ria/operator/chat.json "$API_URL/v1/chat/completions"
```

These are **live-model checks for the physical testing host**. For a remote workstation, run `ssh -N -L 18000:127.0.0.1:8000 YOUR_CLIENT_HOST`, use `http://127.0.0.1:18000`, and securely provision the bearer token to that authorized caller. Adjust the tunnel's destination if `api_port` differs. Do not publish unauthenticated/plaintext API access to the LAN or Internet.

## Operate, drain and reconfigure

Keep requests, root manifests/digests, build info, policy/registration, host/probe reports, complete raw evidence and final locks together as a reproducible operator record. Logs are diagnostics; machine JSON/exit statuses and sealed evidence establish results. Bound log retention on your Docker host and protect artifacts that contain prompts or output.

Drain **client first, then expert**, through the admin socket. Replace the timeout/stop grace with reviewed values suitable for your admitted operations:

```bash
DRAIN_TIMEOUT_MS=REPLACE_WITH_REVIEWED_INTEGER
STOP_GRACE_SECONDS=REPLACE_WITH_REVIEWED_INTEGER
ria_compose exec -T --user 10001:10001 "$ROLE" \
  /usr/local/bin/ds4ctl drain --socket /run/dwarfstar/admin.sock --timeout-ms "$DRAIN_TIMEOUT_MS"
ria_compose stop --timeout "$STOP_GRACE_SECONDS" "$ROLE"
```

`ds4ctl --timeout-ms` bounds the caller's wait; it does not extend the HTTP client's `network.operation_timeout_ms` admin deadline or the expert's `expert.drain_timeout_ms` admin/drain deadline. Docker's stop grace is another independent deadline and can escalate to forced termination. Choose these bounds coherently, then check completion and exit status. Phase/block cancellation checks and drain timers do not establish a full physical deadline for blocking filesystem/foreign-runtime operations or cleanup; that still requires target-host qualification.

SIGTERM removes readiness and begins bounded draining. The HTTP client forces exit on a second signal; the expert retains its drain/termination state. `ria_compose down` removes the stopped project's containers/network while leaving bind-mounted model/config/evidence files; do not use broad volume/file deletion commands. The templates intentionally have `restart: 'no'`. After a failure, fix it and relaunch through the controller. Session/KV state is not persisted across process restarts; recovery creates a new binding and replays the full prompt. A lost response does not authorize an automatic repeat of a potentially completed generation.

An expert serves one admitted client binding. Do not run `ds4` or `ds4-eval` through `docker exec` alongside the HTTP client: that attempts another binding. For an optional one-shot CLI, drain/stop the HTTP client first, keep the expert healthy, use the same reviewed client mounts/caps/identity, and then restore HTTP with the managed launcher:

```bash
# On the client, only after the HTTP client has been drained/stopped:
ria_compose run --rm --no-deps --entrypoint /usr/local/bin/ds4 client \
  --config /etc/dwarfstar/service.json --prompt 'Hello' --tokens 32
```

A one-shot Compose run is an explicitly reviewed diagnostic operation and does not replace managed deployment inspection/admission. Likewise, `ds4-eval --config /etc/dwarfstar/service.json --teacher-forced /artifacts/inputs/tokens.json --logits-output /artifacts/logits.bin` exports candidate logits when it owns the client binding. Input is exactly `{schema_revision: 1, tokens: [...]}` with optional `label_mask`: 2–1048576 integer IDs in 0–129279, fitting the admitted context; the Boolean mask has `len(tokens)-1` entries and at least one scored position. Keep the JSON within the 16 MiB/1,000,000-node parser limits. It does not generate an independent full-model reference or complete final qualification; see below.

Changing profile, executor, image, placement/cache membership, context/prefill, budgets, CPU/NUMA selection, transport mode, relevant driver/kernel/runtime/security policy, or credentials requires review of affected identities and **fresh applicable baseline/probe/registration/calibration/inventory/finalization evidence**. New grants are required for new authorized identities/layouts. Prepare new output directories and publish only after their checks pass. Do not patch a sealed plan or reuse stale evidence to force a restart.

## Troubleshooting

| Symptom | Check / correction |
| --- | --- |
| Unsupported architecture/GPU or zero/multiple visible devices | Confirm Linux amd64, exact physical UUID and CC 12.0; client must be an RTX 5090. Check driver/toolkit and both CUDA expert Compose files. Never replace UUID with `all`. |
| Controller owner/lock conflict | Use the same administrative EUID as earlier commands; investigate the active controller. Preserve its persistent lock file. |
| Missing mount/config or UID 10001 permission error | Check every absolute source/ancestor, `create_host_path: false`, final `service.json`, group traversal and file modes. Bootstrap is not a serving config. |
| Host/cgroup/Compose identity mismatch | Record actual local daemon versions and container parent ancestry; remove ambient overrides; rerun affected baseline/evidence after a real change. |
| Seccomp/NUMA or locked-memory failure | Use the reviewed `seccomp-numa.json`; check selected CPU/node memberships and finite memlock/population budgets. Do not use privileged/unconfined containers or enable swap as a workaround. |
| OOM/allocation/plan rejection | Compare estimator/inventory/probe details, cgroup ancestor limits, actual RAM/VRAM, replication/workspace/state/startup accounting. Reduce workload or choose adequate resources and regenerate evidence. |
| Transport timeout or identity rejection | Check both roles select the same mode, private firewall/both ports and grant/layout/profile identity. TLS additionally checks clock/expiry, CA/leaf/key, SAN and opposite DER fingerprint. Trusted-network TCP requires same control/bulk peer IP and null grant peer name. Start transport server before client. |
| Fixture container vanished or run exceeds idle deadline | Its lifetime is 1800 seconds. Stop/reset through the controller; create a new bootstrap/evidence set with preregistered feasible bounds. |
| Finalizer rejects calibration/proof | Retain and transfer all four full run directories and the complete role bundle; match policy/registration/environment/build/probe identities. Do not write `passed: true` or raise tolerance after observing output. |
| Container started but health unready | Read role logs and initialization progress; `started` is not `ready`. Bank loading/population must finish within the reviewed operational allowance. |
| HTTP 401/429/413/invalid request | Use the exact protected bearer token, one active generation, admitted body/image/context bounds and supported frontend fields. |
| A second client cannot bind | Stop/drain the current HTTP/CLI/evaluation client first; inspect expert teardown/readiness before replacing it. |
| GPU access fails after daemon/systemd changes | Recheck Container Toolkit's documented runtime issues and actual device/cgroup state; refresh affected evidence after correcting the host. |

Tools fail closed and report nonzero status/actionable diagnostics. Keep the primary error and its evidence; do not treat an error message printed with some JSON as a successful operation. `--help` is available on each host tool/subcommand. Detailed contracts and schemas are in [artifact/deployment workflow](docs/ria-artifacts-and-deployment.md), [frontend](docs/ria-frontend.md), [prompt prefill](docs/ria-prompt-prefill.md) and [specification](docs/ria-specification.md).

## Qualification limits

Initial admission verifies real host/resource conditions and preregistered native/transport fixtures. It enables candidate serving for hardware testing. It **does not** establish complete-model numerical fidelity, all required configuration cells, fault recovery or a completed one-hour soak.

The implemented `ds4-eval` teacher-forced exporter and `qualify_ria.py compare-logits` can compare supplied saved reference/candidate logit files under a frozen policy. The missing full-model workflow must still preregister the independent reference realization, corpus/token identities, scored positions/label masks, profiles and reproducibility scope, then independently produce/run/validate all required evidence. Candidate export or a passing comparison on arbitrarily supplied files is not that workflow. Complete semantic **540-cell** release matrix/fault/gate producers and validators also remain incomplete.

Read the current blockers without executing any workload:

```bash
cd /opt/RIA
"$RIA_PYTHON" tools/qualify_ria.py readiness --output /srv/ria/operator/readiness.json
```

This currently returns **exit 1** because `release_ready` is false; the report is useful output, not a successful release gate. `matrix --output PATH` enumerates required cells but does not execute them. No full checkpoint, physical GPU/live model, hardware performance or soak validation was performed on the development host. Offline checks and publication evidence are recorded in [the once-over report](docs/ria-once-over.md) and [handoff guide](docs/ria-handoff.md). Free GitHub Actions currently run offline checks, amd64 CPU/CUDA builds and GHCR publication; there is no automated GPU/full-model physical qualification runner.

## Original DwarfStar documentation

The preserved upstream documentation below applies to donor execution paths. For RIA deployment, use the runbook above and its prepared safetensors, native configuration and admission tools.

<p align="center">
  <img src="logo.svg" alt="DwarfStar logo" width="220">
</p>

**DwarfStar** aims to be the best way to run a few excellent large
language models on consumer hardware (that is, hardware that people
can actually own). To reach this goal, we are building
a small native inference engine optimized first for
**DeepSeek V4 Flash** (including the experimental vision model),
**DeepSeek V4.1 Flash** (Metal, and text inference on CUDA),
and additionally **GLM 5.2 and 5.3**, **GLM 5.3 Flash** and
**DeepSeek V4 PRO**, and **Qwen3.8 Flash Next** (Metal and CUDA). The code is self-contained and
deliberately narrow, not a general GGUF runner: you need to use the
GGUF files the project produces, that are part of the project
itself.

We test things in integration: model loading, prompt rendering,
tool calls, KV state, the HTTP server, and the coding agent are built and tested together.
The repository also includes tools and data for GGUF, imatrix, quality, and speed.

## Supported hardware

* **Metal**, the primary target, on Macs with 96 GB or more. Smaller machines
  can use SSD streaming. SSD streaming is also needed in order to run very
  large models such as full GLM 5.x (not Flash) on 128GB systems.
* **NVIDIA CUDA**, the DGX Spark is our main gaol. DwarfStar also supports multi-GPU systems that are not supported by other backends, for instance it can run DeepSeek v4 Flash on Ada Lovelace cards.
* **ROCm** on Strix Halo systems such as the Framework Desktop.

This project would not exist without **llama.cpp and GGML**, make sure to read
the acknowledgements section, a big thank you to Georgi Gerganov and all the
other contributors.

**Model support is intentionally opportunistic**. The project follows the best open
weights for useful local machine sizes, especially 128 GB laptops and 256/512 GB
workstations. A model may be removed when a better replacement arrives.

# So, what can I do with this software?

* You can run a very capable models in your consumer hardware, a MacBook, a DGX Spark, or a Strix Halo for example. Even if you have not enough RAM, with SSD streaming, you can run it at a decent speed.
* You can use multiple CUDA cards as a multi-user LLM server. Ada Lovelace, including L40S, is supported: newer models can run here even when their other inference implementations require newer GPUs. Our eight-L40S Flash setup has reached about 126 t/s aggregate generation with 16 sessions.
* Using two 128 GB Macs connected with RDMA, you can run 4-bit DeepSeek Flash or GLM 5.3 Flash with tensor parallelism. Larger GLM 5.2 quants need larger machines, such as Mac Studios.
* You can also use pipeline paralellism to glue together multiple systems to sum their RAM and run larger models.

## Motivations

* Capable open-weight models now fit on high-end personal machines.
* DeepSeek V4 Flash and PRO, GLM 5.2, tolerate aggressive routed-expert quantization.
* Compressed KV caches and fast local SSDs make long contexts practical.
* The idea of an inference system specialized for a few models.

# AI full disclosure

* This software is developed with **strong assistance from AI coding agents** and with humans leading the ideas, testing, and debugging. We say this openly because it shaped how the project was built. If you are not happy with AI-developed code, this software is not for you. The acknowledgement below is equally important: this would not exist without `llama.cpp` and GGML, largely written by hand.

## Acknowledgements to llama.cpp and GGML

`ds4.c` does not link against GGML, but it **exists thanks to the path opened by the
llama.cpp project and the kernels, quantization formats, GGUF ecosystem, and hard-won
engineering knowledge developed there**.
We are thankful and indebted to [`llama.cpp`](https://github.com/ggml-org/llama.cpp)
and its contributors. Their implementation, kernels, tests, and design choices were
an essential reference while building this DeepSeek V4 specific inference path.
Some source-level pieces are retained or adapted here under the MIT license: GGUF
quant layouts and tables, CPU quant/dot logic, and certain kernels. For this
reason, and because we are genuinely grateful, we keep the GGML authors copyright
notice in our `LICENSE` file.

## Status

The software is currently very fast changing. Consider it beta quality.
Before each release, a big QA run is executed, however instabilities
and regressions are definitely possible.

# How to use this project?

I (Salvatore) believe that the way projects should be shipped and used changed because of AI. The main differences today are:

1. With AI, users can modify the software in significant ways with low efforts, costs, and even lacking deep domain knowledge about the task they want to accomplish. For instance, a DwarfStar user with a specific hardware setup can ask a coding agent to improve the inference speed of this software for the specific hardware setup, asking the model to reach the maximum prefill and generation speed without impacting correctness, and also asking to do a deep QA pass.
2. Similiarly, because of "1", software may be shipped in a different way than before. It must be more a working template for the biggest use cases, without trying to cover every possible setup. If DwarfStar showcases a few good implementations of tensor parallel execution, the code will work as a rail for implementing the same feature in specific conditions, for a new model, and so forth.

So, while this project attempts to be usable for the featured models and the most common hardware setups, I ask you, if you have access to coding agents, to consider using coding agents as an interface to discover the project, make modifications, create personalized setups. This way you can likely do more than what we ship, and certain things that are not documented or implemented, and that you require, are potentially very easy to achieve.

## Start Here

```sh
git clone https://github.com/antirez/ds4.git
cd ds4
```

Choose your build. The platform guides cover prerequisites, memory sizing,
and hardware-specific setups:

| Platform guide | Build |
| --- | --- |
| [Metal on Apple Silicon](docs/METAL.md) | `make` |
| [DGX Spark](docs/DGX_SPARK.md) | `make cuda-spark` |
| [Strix Halo / Framework Desktop](docs/STRIX_HALO.md) | `make strix-halo` |
| [One or more CUDA cards, including Ada/L40S](docs/CUDA_MULTI_GPU.md) | `make cuda-generic` |

For a first run on a 96 or 128 GB machine, download DeepSeek V4 Flash Q2:

```sh
./download_model.sh ds4f-q2
```

Downloads go in `gguf/`. Repeat the command to resume an interrupted download.
Leave memory for the context and runtime buffers as well as the model.
See [other models](docs/MODELS.md) or use [SSD streaming](docs/SSD_STREAMING.md)
on a smaller Mac.

## Everyday Use

Once built and with a model downloaded:

```sh
./ds4
./ds4 -p "Explain Redis streams in one paragraph."
./ds4-agent
./ds4-server --ctx 32768
```

The default model is `ds4flash.gguf`, a link updated by main-model downloads.
Pass `-m FILE` to choose explicitly. Commands normally run from the repository
root; use `--chdir /path/to/ds4` when launching elsewhere.

The server listens at `http://127.0.0.1:8000` by default; see [serving](docs/SERVER.md)
for API access and multiple sessions.

The interactive CLI keeps a multi-turn conversation. Use `/help`, `/read FILE`,
`/ctx N`, and `/quit`. Ctrl+C interrupts generation and returns to the prompt.
Run each binary with `--help` for its full options.

### Native coding agent

`ds4-agent` runs inference directly, without a separate HTTP server. It keeps
the token history and live model state together, shows prefill progress, and
uses the model's native tool format. DeepSeek and GLM have their own templates.

Use `/hints on` for occasional, brief explanations of the programming choices
behind the work, and `/hints off` to stop them. Changes take effect at the next
conversation boundary without rebuilding the cached context. New and resumed
sessions start with hints off.

Sessions are stored in `~/.ds4/kvcache`:

| Command | Action |
| --- | --- |
| `/save` | Save the current session |
| `/list` | List saved sessions |
| `/switch <sha>` | Resume a session |
| `/del <sha>` | Delete a saved session |
| `/strip <sha>` | Keep text and title, removing the large KV payload |

Compatible local KV snapshots avoid rebuilding the prompt. Stripped sessions
and network TP restores require prefill. Sessions containing images cannot yet
be saved. Saved conversations and traces may contain private information.

For Pi, OpenCode, Codex CLI, or Claude Code, use `ds4-server` instead and follow
the [client setup guide](docs/CLIENTS.md).

### Models, images, and speculation

[Models and vision](docs/MODELS.md) lists the supported downloads and memory
requirements. DeepSeek Vision Experimental uses a different checkpoint from
Flash 0731; GLM 5.3 Flash and Qwen3.8 Flash Next add vision to the same text
model through a separate encoder.

DeepSeek V4.1 Flash text and vision run on Metal; text also runs on a DGX Spark.
Q2 runs with SSD streaming on one 128 GB Mac or Spark, or resident across two
Macs or two Sparks using RDMA. Q4 needs SSD streaming or a 512 GB Mac.
Engram tables remain on disk in every mode, so use a fast
local SSD. See the [model guide](docs/MODELS.md#deepseek-v41-flash) for downloads
and setup.

With the matching encoder passed as `--vision FILE`, use `/read image.png`
in the CLI or `view_image` in the native agent.

Qwen3.8's smaller Q2 release has **41.73 GiB** of main/MTP weights,
with imatrix IQ2_XXS gate/up experts and padded Q2_K down projections.
It is the starting option for 64 GB Macs.
The GGUF also contains 95.37 GiB of original BF16 n-grams, read directly
from disk rather than loaded into RAM. Keep it on a fast SSD. Start with 8K context:

```sh
./download_model.sh qwen38-q2
./ds4 --ctx 8192 --prefill-chunk 1024
```

The download fetches one 137.10 GiB file and updates `ds4flash.gguf`.
Add `--mtp` for speculative decoding. The larger
`qwen38-q4k` target is also available. Download the optional vision encoder
with `./download_model.sh qwen38-vision` and pass it with `--vision`.
See [Qwen setup](docs/QWEN38_FLASH_NEXT.md) for details.

Speculative decoding is opt-in. GLM and Qwen use `--mtp`; V4 Flash DSpark needs a matching
support GGUF. It can improve generation, but not every workload benefits.
Read [speculative decoding](docs/SPECULATIVE_DECODING.md) for setup and the
difference between default opportunistic sampling and `--mtp-exact-sampling`.

### Output and power

Thinking is enabled by default. Use `--nothink` or `/nothink` for direct
answers, and `--think` or `/think` to enable it again.
For V4.1, `ds4` and `ds4-agent` also accept
`--think-level 25` or `/think 25`: 1 to 100 sets the reasoning effort, and
0 disables thinking. `--think` selects 75, `--think-max` selects 100.
Changing the level in a conversation rebuilds its cached prefix.
The normal sampling defaults are temperature 1, top-p 1, and min-p 0.05;
`--temp 0` selects greedy output.

For DeepSeek V4, `--power N` trades throughput for lower sustained GPU load.
The default is 100. V4.1 and GLM currently require `--power 100`.

DeepSeek V4 Flash and GLM 5.3 Flash also support directional steering. Load a
vector with `--dir-steering-file FILE`; `/steer F` adjusts its scale for
subsequent tokens in a local CLI or agent session, without rebuilding the
existing KV cache. See [steering documentation](dir-steering/README.md).

`--prefix-file FILE` preloads complete `USER:` / `ASSISTANT:` pairs before
the live conversation. A turn marker must start a line, roles must alternate,
and the last turn must be `ASSISTANT:`.

## Capability Evaluation

`ds4-eval` runs embedded capability regression tests against a real GGUF.
These are DwarfStar integration checks, not official leaderboard scores.

```sh
./ds4-eval -m ds4flash.gguf --trace /tmp/ds4-eval.txt
./ds4-eval -m ds4flash.gguf --suite hard-smoke
./ds4-eval -m ds4flash.gguf --suite hard --retry-incomplete
```

The default suite is `core`; `--suite all` runs core and hard cases.
`--list-cases` lists tests without loading a model. `--plain` selects
non-interactive output, and `--regrade-trace FILE` scores an existing trace
without generating again. Sources and licenses are in [EVAL_DATA.md](EVAL_DATA.md).
For inference correctness and release checks, read [testing](docs/TESTING.md).

## Speed

This recorded DeepSeek V4 Flash Q2 sweep uses an M5 Max with 128 GB RAM,
2048-token continued-prefill intervals, and 128 greedy generation tokens per
frontier. It is a baseline, not a fresh benchmark of every commit.

![M5 Max Flash Q2 throughput](speed-bench/m5_max_ts.svg)

See [performance and benchmarking](docs/PERFORMANCE.md) for the full numbers,
DGX Spark results, comparison conditions, and benchmark commands.

## Detailed Guides

- [Models and vision](docs/MODELS.md): Flash, PRO, GLM, Qwen, and matching encoders.
- [Qwen3.8 Flash Next](docs/QWEN38_FLASH_NEXT.md): model setup, MTP, vision, and validation.
- [SSD streaming](docs/SSD_STREAMING.md): run larger than RAM and size the cache.
- [Inference across machines](docs/DISTRIBUTED.md): two-Mac TP/RDMA and layer pipelines.
- [Speculative decoding](docs/SPECULATIVE_DECODING.md): DSpark, GLM and Qwen MTP, and sampling.
- [Serving](docs/SERVER.md): APIs, images, batching, and disk KV caches.
- [Coding agent clients](docs/CLIENTS.md): Pi, OpenCode, Codex CLI, and Claude Code.
- [Performance](docs/PERFORMANCE.md): reproducible measurements and recorded baselines.
- [Testing and development](docs/TESTING.md): regression tests, debugging, and model-building tools.

Read [CONTRIBUTING.md](CONTRIBUTING.md) before sending a pull request.

## Logo

The DwarfStar logo was designed by hand by Salvatore Sanfilippo, made more
graphical with AI, and manually reworked by Ben Gnomino, whose human touch made
it rock.
