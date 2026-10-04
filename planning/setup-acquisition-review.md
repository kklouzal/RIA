# Pinned setup acquisition source review

Reviewed 2026-10-04. This is an offline source/boundary review, not checkpoint,
container, hardware, numerical or performance qualification.

`ria.setup_acquisition.acquire_source(workspace, *, max_bytes, deadline_ms,
token_file=None)` supervises one owned worker and returns sealed source facts.
The worker inherits an enclosing preparation process group, so controller
cancellation terminates its descendants. Its own supervisor enforces the total
deadline even if DNS or a TLS socket stalls; interrupted private files can be
resumed by the standalone module invocation.

The source is exactly `nvidia/DeepSeek-V4.1-Flash-NVFP4` revision
`3431dde3247c13b5957f682b1e3c6fcae2566079`. The packaged index selects48 shards
whose aggregate content is527,293,384,576 bytes. Only those shards, config and
index are requested. Native tokenizer, tokenizer config and chat template are
copied from reviewed packaged `deepseek-ai/DeepSeek-V4.1-Flash` metadata; the
NVIDIA repository lacks the native chat template. No source bank is exported
through cooperative setup: only the preparer's authenticated compact client
extraction is registered for transfer.

Source mode now also supplies missing `tokenizer.json`, `tokenizer_config.json`
and `chat_template.jinja` from the authenticated packaged originals. It checks
all existing named inputs before writing, never overwrites or changes their
permissions, and reports the exact conflicting name for an explicit operator
preserve/replacement decision. Symlinks, hard links and special files fail
before publication. Temporary inodes live in setup-owned private `.ria-recipes`
and publish without replacement through held source/staging descriptors. The
reviewed NVIDIA tokenizer and tokenizer-config Git blob identities equal the
packaged DeepSeek originals; a conflict is not expected for that pinned revision.

Primary documentation/source consulted:

- [Hub API reference](https://huggingface.co/docs/hub/en/api) and
  [official model_info implementation](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/hf_api.py):
  exact revision endpoint with `blobs=true` returns file sizes, Git blob IDs and
  LFS identities. Live metadata must match the already reviewed inventory.
- [Hub cache identity contract](https://huggingface.co/docs/hub/en/local-cache):
  Git-tracked content uses Git SHA1 blob identity; LFS content uses SHA256.
  Acquisition checks both size and content identity, and also binds compact
  non-LFS metadata to the packaged SHA256.
- [Official redirect handling](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/utils/_http.py):
  credentials must not reach signed storage redirects. Our narrower adapter
  removes Authorization after every redirect and permits only verified HTTPS
  Hub/storage domains, with a finite redirect count and no proxy environment.
- [Python HTTP client](https://docs.python.org/3.12/library/http.client.html) and
  [socket timeout contract](https://docs.python.org/3.12/library/socket.html):
  native argument-free HTTPS requests, incremental read1, explicit closure and
  residual socket timeouts. Exact Content-Length, Content-Range and identity
  encoding constrain initial and resumed file bodies; unexpected framing fails.

Publication is link-free, workspace-owned, private and manifest-last. Files are
streamed with bounded buffers; complete files are verified before non-replacing
publication. A crash between link and unlink is reconciled only when the two
owned names account for both inode links. Corrupt owned partials are removed;
committed corruption fails closed. Stale atomic-output temporary files are
reclaimed only by exact name/type/owner/mode/single-link proof in the marked
owned source directory. Provisioned external inputs are never cleanup targets.

Checks:108 tests passed across `test_setup_acquisition.py`,
`test_setup_artifacts.py`, and existing `test_artifacts.py`; Ruff and whitespace
checks passed. All acquisition HTTP is mocked. The only actual supervised
worker fixture supplies a one-byte budget, rejects before HTTP, and proves
subprocess/module integration. No model/weight endpoint was contacted, no
weights were downloaded, and no container/GPU/model/live tests were run.

The source-metadata addition expanded that bundle to119 passing tests. After
moving temporary publication into the private recipe area, all15 affected
metadata/source fixtures passed again; Ruff passed. Those tests include the
actual packaged compact originals and synthetic no-overwrite, link/special,
tamper, competing-publication and held-directory substitution cases. Final
integrated repository checks are owned by the root coordinator.

The full model's acquisition/preparation IO, RSS, elapsed time and performance
remain unmeasured. Operators declare acquisition bytes/deadlines, numerical
profile, context/prefill, resource caps and qualification policy. These helpers
do not infer headroom, fit GPU state, modify workload limits, loosen numerical
tolerances or claim deployment admission.
