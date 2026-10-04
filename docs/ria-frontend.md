# Native RIA frontend

The RIA build selects the prepared DeepSeek-V4.1-Flash CUDA graph with an explicit service document. It verifies the deployment lock, memory plan, placement plan, tokenizer identity and prepared tensor identities before serving. Learned client operations use CUDA; CPU tokenization, source prompt serialization, image decoding and sampling from completed logits remain native C. The donor engine remains available when `ria_service_path` is NULL.

## Commands and public API

Use the provisioned client service document; it supplies placement, numeric network addresses, TLS identities, admitted context and API policy:

```sh
/usr/local/bin/ds4-server --config /etc/dwarfstar/service.json
/usr/local/bin/ds4 --config /etc/dwarfstar/service.json --prompt 'Explain this result.' --tokens 128
/usr/local/bin/ds4-eval --config /etc/dwarfstar/service.json
```

`--ria-service PATH` is an alias for `--config PATH`. The CLI may select a smaller positive `--ctx` within admission. The HTTP server takes its context and listen address from the service document. Explicit donor model/backend/placement, MTP, CPU learned execution, disk KV and unsupported diagnostic options fail rather than alter RIA admission.

For the existing C API, zero-initialize `ds4_engine_options`, set `backend=DS4_BACKEND_CUDA` and `ria_service_path`, then use the ordinary engine, tokenization, prompt rendering, image embedding, session sync/eval and sampling APIs in [ds4.h](../ds4.h). One session claims the graph owner. Its context cannot exceed admission. Stop all users before destroying that session or closing the engine. A failed CUDA cleanup marks the process unusable; restart it instead of reusing the owner.

RIA engine startup intentionally ignores SIGPIPE for the process lifetime so disconnected TLS sockets return operation errors. Engine close does not restore the old disposition. Callers serialize signal-policy changes with startup and retain this policy while the engine is in use. The expert service blocks SIGPIPE in its transport owners. The donor path with no RIA service option retains its existing process policy.

A session's authoritative state contains incorporated token IDs and image identities. Sync reuses only a complete identical incorporated prefix, including the image positions, source identities and embedding lengths. A changed or shortened prefix creates a fresh remote binding and replays the prompt. Identical complete prefixes return the retained final logits without a new graph step. Sampled EOS is terminal control and is not incorporated. Text stops, cancellation, executor failures and failed writes invalidate the affected binding; abandoned work quiesces before the active request slot is released.

## Authenticated HTTP and SSE

The initial API exposes authenticated `GET /v1/models` and `POST /v1/chat/completions`. Every request requires the file-provisioned Bearer token, including loopback clients. The model ID is exactly `DeepSeek-V4.1-Flash`. Other model IDs, unknown fields and unqualified endpoints receive explicit errors. One generation is active and none are queued; a competing generation receives HTTP 429. Header/body/JSON/message/tool/image/connection limits and read/write deadlines come from the service document.

A minimal request body is:

```json
{"model":"DeepSeek-V4.1-Flash","messages":[{"role":"user","content":"Hello"}],"stream":true,"temperature":0,"max_tokens":32}
```

Accepted top-level fields are `model`, `messages`, `n`, `stream`, `stream_options`, `max_tokens`, `max_completion_tokens`, `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `stop`, `reasoning_effort`, `thinking`, `tools`, `tool_choice` and `ria_measurements`. `n` must equal 1. Supply one output limit, not both. Explicit sampling fields are applied, including in reasoning/tool output; omitted fields use source defaults. `reasoning_effort` accepts `low`, `high`, `max` or an integer from 1 through 100; `thinking` is Boolean. Tool choice is `auto` or `none`. Top-level tool declarations require an initial system message and cannot duplicate inline declarations. Encoded tool arguments reject duplicate keys and nonfinite JSON values.

Image inputs are uploaded JPEG/PNG data URLs in supported message content blocks. Source preprocessing, image delimiters and image embedding identities participate in exact continuation. Remote URL retrieval, filesystem paths supplied through HTTP, model-emitted tool execution, autonomous agents, durable response IDs, `/v1/responses`, legacy completions and Anthropic messages are outside the initial API. No Python model runtime is used.

Unix administration uses the configured socket, ordinarily `/run/dwarfstar/admin.sock`, mode 0600 and an authorized UID of 10001. `ds4ctl health --socket PATH` reports actual readiness. `ds4ctl drain --socket PATH` acknowledges only after admission stops and generation, graph and remote users quiesce within the configured deadline.

## Opt-in measurements

An authenticated streaming request may add `"ria_measurements":true`. Other types are rejected, and enabling it without streaming is rejected. Disabled requests perform no observer clock reads or formatting. Enabled requests emit bounded timing-only SSE events, with no token IDs or additional generated content:

```text
event: ria_measurement
data: {"index":"0","elapsed_ns":"123456","previous_write_ns":"1000"}

event: ria_diagnostics
data: {"phase":"continuation","reused_prefix_tokens":"100","sampled_tokens":"8","completion_tokens":7,"write_ns":"3000"}
```

Each index is a consecutive decimal string beginning at zero. `elapsed_ns` is a monotonic observation since generation-worker entry. A non-EOS event occurs after successful token incorporation and before byte decoding/content filtering; terminal EOS has an event after sampling and is not incorporated. `previous_write_ns` sums actual request-owned socket-writer elapsed time since the preceding event, including its write and content writes; it excludes JSON formatting. It includes the first role chunk at index zero. SSE response headers and prefill keepalives are reflected in elapsed time but are outside this writer counter.

The final diagnostics precede `[DONE]`. `phase` and `reused_prefix_tokens` report the engine's successful exact-prefix sync decision. `sampled_tokens` counts measurement events; `completion_tokens` matches API usage and excludes EOS. Their difference is one for natural EOS and zero for output/context limits. `write_ns` includes observed writes before the diagnostics; diagnostics and `[DONE]` cannot report their own delivery time. Content-delta intervals can span UTF8 fragments or withheld source markers and must not be labeled token intervals. Qualification must identify the observer-enabled region and independently compare disabled behavior; these events do not prove hardware performance by themselves.

## Teacher-forced logits

Collect native source/profile evidence with:

```sh
/usr/local/bin/ds4-eval --config /etc/dwarfstar/service.json \
  --teacher-forced /evidence/tokens.json --logits-output /evidence/logits.bin
```

This mode accepts only the config, teacher input and output path options. Input is strict JSON containing `schema_revision:1`, an array `tokens` of 2 through 1,048,576 vocabulary IDs and optional Boolean `label_mask` of length `len(tokens)-1`. The complete input is bounded to 16 MiB and 1,000,000 JSON nodes, including the optional mask; the actual sequence must fit those parser bounds and the admitted context. Parser, read and canonical-identity peaks are checked before input allocation. Mask element `i` controls the shifted target `tokens[i+1]` scored from logits after incorporating `tokens[i]`.

The output is atomically published after checked writes, flush and file synchronization. Its 128-byte little-endian header contains `RIALOG1\0` at byte 0, revision U32 at 8, vocabulary U32 at 12, position count U64 at 16, scored-label count U64 at 24, logical model SHA256 at 32, operator-contract SHA256 at 64 and complete input JCS SHA256 at 96. It is followed by one 129,280-element FP32 logits row per incorporated position, including the final position. Machine-readable stdout reports the actual output SHA256 and shifted masked negative log likelihood computed with a stable FP64 reduction. The reduction cancels the maximum and label offset before adding the log denominator, preserving uniform-logit loss even for extreme finite common offsets. Fidelity/quality thresholds belong to the preregistered hardware policy.

## Model-free checks and deferred evidence

`make ria-offline` exercises checked arithmetic, strict JSON/JCS, exact wire payloads, credits/cancellation/terminal history, TensorStore/state/expert/graph fixtures and source encoding. The independent tokenizer/prompt fixtures use the pinned tokenizer and source encoding files; they do not download model weights or run learned inference. The timing fixture uses a real socket pair and blocked reader to verify event order, counting, disabled observation and actual writer accounting.

`ds4ctl qualify-transport --config BOOTSTRAP --request REGISTERED_REQUEST` runs the real paired TLS1.3/mTLS control and bulk paths with synthetic typed payloads. It does not require a full model, final memory plan or expert bank. Both roles verify their registered request, environment/build identities, provisioned certificate SHA and configured DNS/IP against the certificate SAN; record fragmentation, protected progress, cancellation, typed rejection and a real partial-frame deadline are exercised. The sole stdout document is unqualified raw measurement JSON. The evidence wrapper verifies paired per-iteration cases, policy and provenance before sealing qualification. Private credentials and payload contents are absent from reports.

Local offline/static/sanitizer checks establish these boundary and source contracts. Model startup, GPU graph execution, full-bank deployment, real-host continuation, source/profile logits/loss and production latency/RSS/soak gates require the preregistered physical-host run. An unexecuted gate is not a pass.
