#define _GNU_SOURCE
#include "remote.h"
#include <openssl/crypto.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>

struct ria_remote {
  const ria_service *service;
  ria_tls tls;
  ria_transport control, bulk;
  ria_binding binding;
  pthread_mutex_t lifecycle;
  uint64_t generation, invocation;
  bool aborted;
};
bool ria_request_charge(uint16_t kind, uint64_t request, uint64_t response,
                        uint64_t *charge, ria_error *e) {
  /* Schema-v1 fixed credit table. The 8MiB expert allowance covers the
   * selected executor's admitted <=64x5120 / intermediate2304 tile64
   * workspace, descriptors, FP32 host row/output staging and quantizers.
   * Actual endpoint pools are separately checked against the memory plan. */
  uint64_t workspace =
      (kind == RIA_EXPERT || kind == RIA_SHARED) ? UINT64_C(8388608) : 0;
  return (ria_u64_add(request, response, charge) &&
          ria_u64_add(*charge, workspace, charge) &&
          ria_u64_add(*charge, 2 * RIA_HEADER_BYTES, charge)) ||
         ria_fail(e, RIA_RESOURCE_LIMIT, "credit charge overflow");
}
bool ria_progress_charges(const ria_limits *l, uint64_t *control, uint64_t *row,
                          uint64_t *bulk, ria_error *e) {
  uint64_t row_request, row_reply, one;
  if (!ria_u64_mul(l->row_lookup_rows, 16, &row_request) ||
      !ria_u64_add(row_request, 16, &row_request) ||
      !ria_u64_mul(l->row_lookup_rows, 280, &row_reply) ||
      !ria_u64_add(row_reply, 24, &row_reply) ||
      !ria_request_charge(RIA_ROWS, row_request, row_reply, &one, e) ||
      !ria_u64_mul(one, 2, row) ||
      !ria_request_charge(RIA_CHUNK, 16, l->bulk_data_bytes + 64, bulk, e) ||
      !ria_request_charge(RIA_HEALTH, 256, RIA_ERROR_MAX, &one, e) ||
      !ria_u64_mul(one, 4, control))
    return ria_fail(e, RIA_RESOURCE_LIMIT,
                    "protected progress charge overflow");
  return true;
}
static void retire(ria_remote *r) {
  ria_binding_invalidate(&r->binding);
  pthread_mutex_lock(&r->lifecycle);
  r->aborted = true;
  if (r->control.fd >= 0)
    shutdown(r->control.fd, SHUT_RDWR);
  if (r->bulk.fd >= 0)
    shutdown(r->bulk.fd, SHUT_RDWR);
  pthread_mutex_unlock(&r->lifecycle);
}
static bool canonical(const char *input, size_t bytes, char **output,
                      size_t *length, ria_error *e) {
  ria_json_doc d = {0};
  bool ok =
      ria_json_parse(input, bytes, (ria_json_limits){RIA_CONTROL_MAX, 8192, 32},
                     &d, e) &&
      ria_json_canonical(&d, false, output, length, e);
  ria_json_free(&d);
  return ok;
}
static bool exchange(ria_remote *r, uint16_t kind, const void *request,
                     uint64_t request_bytes, uint64_t response_bytes,
                     ria_header *reply_header, uint8_t **reply, ria_error *e) {
  *reply = NULL;
  bool bulk = kind == RIA_CHUNK;
  uint64_t channel = bulk ? 1 : 0, id, charge, deadline;
  if (!r->binding.valid || !r->binding.bound ||
      r->binding.last_request[channel] == UINT64_MAX) {
    retire(r);
    return ria_fail(
        e, RIA_NOT_READY,
        "remote binding retired; fresh session and replay required");
  }
  id = r->binding.last_request[channel] + 1;
  ria_header h = {.kind = kind,
                  .payload_length = request_bytes,
                  .request_id = id,
                  .epoch = r->binding.epoch};
  memcpy(h.session, r->binding.session, 16);
  if (!ria_request_charge(kind, request_bytes, response_bytes, &charge, e) ||
      !ria_binding_receive(&r->binding, &h, bulk, e) ||
      !ria_binding_admit(&r->binding, &h, charge, ria_monotonic_ms(), e) ||
      !ria_u64_add(ria_monotonic_ms(), r->binding.limits.operation_timeout_ms,
                   &deadline))
    goto fail;
  ria_transport *t = bulk ? &r->bulk : &r->control;
  uint64_t write_deadline;
  if (!ria_u64_add(ria_monotonic_ms(), r->binding.limits.write_timeout_ms,
                   &write_deadline))
    goto fail;
  if (write_deadline > deadline)
    write_deadline = deadline;
  if (!ria_transport_send(t, &h, request, write_deadline, e) ||
      !ria_transport_frame(t, deadline, r->binding.limits.frame_io_timeout_ms,
                           r->binding.limits.frame_payload_bytes, reply_header,
                           reply, e) ||
      !ria_binding_response(&r->binding, reply_header, e))
    goto fail;
  if (reply_header->status) {
    ria_json_doc d = {0};
    if (!ria_control_json(*reply, (size_t)reply_header->payload_length,
                          reply_header, &d, e)) {
      ria_json_free(&d);
      goto fail;
    }
    const char *message = NULL;
    size_t n = 0;
    if (ria_json_string(&d, ria_json_get(&d, 0, "message"), &message, &n, e))
      ria_error_set(e, (int)reply_header->status,
                    "remote operation failed: %.*s", (int)(n > 180 ? 180 : n),
                    message);
    ria_json_free(&d);
    goto fail;
  }
  return true;
fail:
  free(*reply);
  *reply = NULL;
  retire(r);
  return false;
}
static bool bind_control(ria_remote *r, const ria_remote_options *o,
                         ria_error *e) {
  const ria_limits *l = &o->service->limits;
  char logical[65], operator_id[65], encoding[65], layout[65], placement[65];
  ria_hex_encode(o->client_store->logical_model_digest, 32, logical);
  ria_hex_encode(o->client_store->operator_contract_digest, 32, operator_id);
  ria_hex_encode(o->client_store->encoding_digest, 32, encoding);
  ria_hex_encode(o->client_layout_digest, 32, layout);
  ria_hex_encode(o->placement_plan_digest, 32, placement);
  char input[4096];
  int n =
      snprintf(input, sizeof(input),
               "{\"role\":\"client\",\"logical_model_digest\":\"%s\","
               "\"operator_contract_digest\":\"%s\",\"encoding_digest\":\"%s\","
               "\"client_layout_digest\":\"%s\",\"placement_plan_digest\":\"%"
               "s\",\"profile\":\"%s\",\"server_executor\":\"%s\","
               "\"limits\":{\"frame_payload_bytes\":%llu,\"bulk_data_bytes\":%"
               "llu,\"expert_rows\":%llu,\"expert_requests\":%llu,"
               "\"row_lookup_rows\":%llu,\"inflight_payload_bytes\":%llu,"
               "\"operation_timeout_ms\":%llu,\"frame_io_timeout_ms\":%llu,"
               "\"write_timeout_ms\":%llu}}",
               logical, operator_id, encoding, layout, placement,
               o->client_store->profile, o->service->server_executor,
               (unsigned long long)l->frame_payload_bytes,
               (unsigned long long)l->bulk_data_bytes,
               (unsigned long long)l->expert_rows,
               (unsigned long long)l->expert_requests,
               (unsigned long long)l->row_lookup_rows,
               (unsigned long long)l->inflight_payload_bytes,
               (unsigned long long)l->operation_timeout_ms,
               (unsigned long long)l->frame_io_timeout_ms,
               (unsigned long long)l->write_timeout_ms);
  if (n < 0 || (size_t)n >= sizeof(input))
    return ria_fail(e, RIA_INTERNAL_ERROR, "bind JSON overflow");
  char *json = NULL;
  size_t length = 0;
  if (!canonical(input, (size_t)n, &json, &length, e))
    return false;
  ria_header request = {.kind = RIA_BIND,
                        .payload_length = length,
                        .request_id = 1},
             reply;
  uint8_t *payload = NULL;
  uint64_t deadline;
  bool ok =
      ria_u64_add(ria_monotonic_ms(), l->operation_timeout_ms, &deadline) &&
      ria_transport_send(&r->control, &request, json, deadline, e) &&
      ria_transport_frame(&r->control, deadline, l->frame_io_timeout_ms,
                          RIA_CONTROL_MAX, &reply, &payload, e);
  ria_json_doc response = {0}, original = {0};
  ria_limits agreed;
  if (ok)
    ok = reply.kind == RIA_BIND && reply.flags == 1 && reply.request_id == 1 &&
         !reply.status &&
         ria_control_json(payload, (size_t)reply.payload_length, &reply,
                          &response, e) &&
         ria_bind_validate(&response, true, &agreed, e) &&
         ria_json_parse(json, length,
                        (ria_json_limits){RIA_CONTROL_MAX, 8192, 32}, &original,
                        e);
  const char *keys[] = {"role",
                        "logical_model_digest",
                        "operator_contract_digest",
                        "encoding_digest",
                        "client_layout_digest",
                        "placement_plan_digest",
                        "profile",
                        "server_executor"};
  for (unsigned i = 0; ok && i < 8; i++) {
    const ria_json_node *a = ria_json_at(&original,
                                         ria_json_get(&original, 0, keys[i])),
                        *b = ria_json_at(&response,
                                         ria_json_get(&response, 0, keys[i]));
    ok = a && b && a->type == RIA_JSON_STRING && b->type == RIA_JSON_STRING &&
         a->length == b->length && !memcmp(a->text, b->text, a->length);
  }
  uint8_t session[16], cap[32], server_layout[32];
  uint64_t epoch = 0;
  const char *sid = NULL;
  size_t sid_bytes = 0;
  if (ok)
    ok = ria_json_string(&response, ria_json_get(&response, 0, "session_id"),
                         &sid, &sid_bytes, e) &&
         ria_hex_decode(sid, sid_bytes, session, 16, e) &&
         ria_json_u64(&response, ria_json_get(&response, 0, "epoch"), true,
                      &epoch, e) &&
         ria_json_digest_field(&response,
                               ria_json_get(&response, 0, "bulk_capability"),
                               cap, e) &&
         ria_json_digest_field(
             &response, ria_json_get(&response, 0, "server_layout_digest"),
             server_layout, e) &&
         !memcmp(server_layout, o->server_layout_digest, 32) &&
         epoch == reply.epoch && !memcmp(session, reply.session, 16);
  if (ok)
    ok = agreed.frame_payload_bytes <= l->frame_payload_bytes &&
         agreed.bulk_data_bytes <= l->bulk_data_bytes &&
         agreed.expert_rows <= l->expert_rows &&
         agreed.expert_requests <= l->expert_requests &&
         agreed.row_lookup_rows <= l->row_lookup_rows &&
         agreed.inflight_payload_bytes <= l->inflight_payload_bytes &&
         agreed.operation_timeout_ms <= l->operation_timeout_ms &&
         agreed.frame_io_timeout_ms <= l->frame_io_timeout_ms &&
         agreed.write_timeout_ms <= l->write_timeout_ms;
  uint64_t control, row, bulk;
  if (ok) {
    ria_binding_init(&r->binding, &agreed);
    ok = ria_progress_charges(&agreed, &control, &row, &bulk, e) &&
         ria_binding_protect(&r->binding, control, row, bulk, e) &&
         ria_binding_install(&r->binding, session, epoch, cap, e);
  }
  OPENSSL_cleanse(cap, sizeof(cap));
  free(json);
  free(payload);
  ria_json_free(&response);
  ria_json_free(&original);
  if (!ok && (!e || !e->code))
    ria_error_set(e, RIA_IDENTITY_MISMATCH,
                  "bind response changed identities/limits/session");
  return ok;
}
static bool bind_bulk(ria_remote *r, ria_error *e) {
  char session[33], cap[65], logical[65], operator_id[65], input[1024];
  ria_hex_encode(r->binding.session, 16, session);
  ria_hex_encode(r->binding.bulk_capability, 32, cap);
  ria_hex_encode(r->service->logical_model_digest, 32, logical);
  ria_hex_encode(r->service->operator_contract_digest, 32, operator_id);
  int n = snprintf(
      input, sizeof(input),
      "{\"session_id\":\"%s\",\"epoch\":\"%llu\",\"logical_model_digest\":\"%"
      "s\",\"operator_contract_digest\":\"%s\",\"bulk_capability\":\"%s\"}",
      session, (unsigned long long)r->binding.epoch, logical, operator_id, cap);
  OPENSSL_cleanse(cap, sizeof(cap));
  if (n < 0 || (size_t)n >= sizeof(input))
    return ria_fail(e, RIA_INTERNAL_ERROR, "bulk bind overflow");
  char *json = NULL;
  size_t length = 0;
  if (!canonical(input, (size_t)n, &json, &length, e))
    return false;
  ria_header h = {.kind = RIA_BIND_BULK,
                  .payload_length = length,
                  .request_id = 1,
                  .epoch = r->binding.epoch},
             reply;
  memcpy(h.session, r->binding.session, 16);
  uint8_t *payload = NULL;
  uint64_t deadline;
  bool ok = ria_binding_receive(&r->binding, &h, true, e) &&
            ria_u64_add(ria_monotonic_ms(),
                        r->service->limits.operation_timeout_ms, &deadline) &&
            ria_transport_send(&r->bulk, &h, json, deadline, e) &&
            ria_transport_frame(&r->bulk, deadline,
                                r->service->limits.frame_io_timeout_ms,
                                RIA_CONTROL_MAX, &reply, &payload, e) &&
            reply.kind == RIA_BIND_BULK && reply.flags == 1 && !reply.status &&
            reply.request_id == 1 && reply.epoch == h.epoch &&
            !memcmp(reply.session, h.session, 16);
  ria_json_doc d = {0};
  if (ok)
    ok = ria_control_json(payload, (size_t)reply.payload_length, &reply, &d,
                          e) &&
         ria_binding_bulk(&r->binding, r->binding.bulk_capability, true, e);
  free(payload);
  OPENSSL_cleanse(json, length);
  free(json);
  ria_json_free(&d);
  if (!ok && (!e || !e->code))
    ria_error_set(e, RIA_IDENTITY_MISMATCH,
                  "bulk binding failed or ambiguous; fresh pair required");
  return ok;
}
bool ria_remote_open(ria_remote **out, const ria_remote_options *o,
                     ria_error *e) {
  *out = NULL;
  if (!o || !o->service || !o->client_store ||
      strcmp(o->service->role, "client") ||
      strcmp(o->client_store->role, "client") ||
      memcmp(o->service->logical_model_digest,
             o->client_store->logical_model_digest, 32) ||
      memcmp(o->service->operator_contract_digest,
             o->client_store->operator_contract_digest, 32))
    return ria_fail(e, RIA_IDENTITY_MISMATCH,
                    "remote client package/service identity mismatch");
  ria_remote *r = calloc(1, sizeof(*r));
  if (!r)
    return ria_fail(e, RIA_RESOURCE_LIMIT, "remote session allocation failed");
  r->control.fd = r->bulk.fd = -1;
  r->service = o->service;
  if (pthread_mutex_init(&r->lifecycle, NULL)) {
    free(r);
    return ria_fail(e, RIA_RESOURCE_LIMIT,
                    "remote lifecycle mutex init failed");
  }
  struct sockaddr_storage control, bulk;
  socklen_t control_length, bulk_length;
  bool ok =
      ria_address(o->service->control_address, false, &control, &control_length,
                  e) &&
      ria_address(o->service->bulk_address, false, &bulk, &bulk_length, e) &&
      ria_tls_create(&r->tls, &o->service->tls, e) &&
      ria_transport_connect(&r->control, &r->tls, (struct sockaddr *)&control,
                            control_length, o->service->connect_timeout_ms,
                            e) &&
      bind_control(r, o, e) &&
      ria_transport_connect(&r->bulk, &r->tls, (struct sockaddr *)&bulk,
                            bulk_length, o->service->connect_timeout_ms, e);
  uint8_t a[32], b[32];
  if (ok)
    ok = ria_transport_peer_digest(&r->control, a, e) &&
         ria_transport_peer_digest(&r->bulk, b, e) &&
         (CRYPTO_memcmp(a, b, 32) == 0 ||
          ria_fail(e, RIA_UNAUTHORIZED,
                   "control/bulk peer certificates differ")) &&
         bind_bulk(r, e);
  if (!ok) {
    ria_transport_close(&r->control);
    ria_transport_close(&r->bulk);
    ria_tls_destroy(&r->tls);
    pthread_mutex_destroy(&r->lifecycle);
    free(r);
    return false;
  }
  *out = r;
  return true;
}
bool ria_remote_begin_generation(ria_remote *r, uint64_t *epoch,
                                 uint64_t *generation, ria_error *e) {
  pthread_mutex_lock(&r->lifecycle);
  bool ok = !r->aborted && r->binding.valid && r->generation < UINT64_MAX;
  if (ok) {
    *epoch = r->binding.epoch;
    *generation = ++r->generation;
  }
  pthread_mutex_unlock(&r->lifecycle);
  return ok || ria_fail(e, RIA_NOT_READY,
                        "generation requires fresh healthy binding");
}
void ria_remote_abort(ria_remote *r, uint64_t generation) {
  if (!r)
    return;
  pthread_mutex_lock(&r->lifecycle);
  if (generation == r->generation) {
    r->aborted = true;
    if (r->control.fd >= 0)
      shutdown(r->control.fd, SHUT_RDWR);
    if (r->bulk.fd >= 0)
      shutdown(r->bulk.fd, SHUT_RDWR);
  }
  pthread_mutex_unlock(&r->lifecycle);
}
bool ria_remote_evaluate(ria_remote *r, uint32_t layer, uint64_t invocation,
                         uint32_t rows, const uint64_t *ids,
                         const uint32_t *offsets, const uint16_t *experts,
                         const uint16_t *slots, const float *coefficients,
                         const float *inputs, bool shared, float *output,
                         ria_error *e) {
  if (layer >= 40 || !rows || rows > 64 || !offsets || !ids || !inputs ||
      !experts || !slots || !coefficients || !output)
    return ria_fail(e, RIA_INVALID_REQUEST, "invalid remote expert invocation");
  uint32_t entries = offsets[rows];
  uint64_t request_bytes, response_bytes;
  if (!ria_expert_lengths(rows, entries, 5120, 5120, 0, &request_bytes,
                          &response_bytes, e) ||
      request_bytes > SIZE_MAX)
    return false;
  uint8_t *p = calloc(1, (size_t)request_bytes);
  if (!p)
    return ria_fail(e, RIA_RESOURCE_LIMIT, "expert request allocation failed");
  uint64_t handle = layer + 1 + (shared ? 40 : 0);
  ria_write_u64(p, handle);
  ria_write_u64(p + 8, invocation);
  ria_write_u32(p + 16, rows);
  ria_write_u32(p + 20, 5120);
  ria_write_u32(p + 24, 5120);
  ria_write_u32(p + 28, entries);
  size_t cursor = 40;
  for (uint32_t i = 0; i < rows; i++, cursor += 8)
    ria_write_u64(p + cursor, ids[i]);
  for (uint32_t i = 0; i <= rows; i++, cursor += 4)
    ria_write_u32(p + cursor, offsets[i]);
  for (uint32_t i = 0; i < entries; i++, cursor += 8) {
    ria_write_u16(p + cursor, experts[i]);
    ria_write_u16(p + cursor + 2, slots[i]);
    ria_write_f32(p + cursor + 4, coefficients[i]);
  }
  for (uint64_t i = 0; i < (uint64_t)rows * 5120; i++, cursor += 4)
    ria_write_f32(p + cursor, inputs[i]);
  ria_operation op = {.handle = handle,
                      .input_width = 5120,
                      .output_width = 5120,
                      .max_rows = 64,
                      .expert_count = shared ? 1 : 384,
                      .max_selected = shared ? 1 : 6,
                      .coefficient_min = shared ? 1 : 0,
                      .coefficient_max = shared ? 1 : 1.5f,
                      .shared = shared,
                      .input_bf16 = true,
                      .output_bf16 = true};
  ria_expert_request parsed;
  ria_header reply;
  uint8_t *payload = NULL;
  bool ok = ria_expert_parse(p, (size_t)request_bytes,
                             shared ? RIA_SHARED : RIA_EXPERT, &op,
                             &r->binding.limits, &parsed, e) &&
            exchange(r, shared ? RIA_SHARED : RIA_EXPERT, p, request_bytes,
                     response_bytes, &reply, &payload, e) &&
            ria_expert_response(payload, (size_t)reply.payload_length, &parsed,
                                &op, e) &&
            ria_binding_terminal(&r->binding, reply.request_id, true, e);
  if (ok)
    for (uint64_t i = 0; i < (uint64_t)entries * 5120; i++)
      output[i] = ria_read_f32(payload + 32 + i * 4);
  free(p);
  free(payload);
  if (!ok)
    retire(r);
  return ok;
}
bool ria_remote_rows(ria_remote *r, uint32_t layer, const uint64_t *rows,
                     const uint64_t *associations, uint32_t count, uint8_t *out,
                     ria_error *e) {
  if ((layer != 1 && layer != 14) || !count || !rows || !associations || !out ||
      count > r->binding.limits.row_lookup_rows)
    return ria_fail(e, RIA_INVALID_REQUEST, "invalid remote Engram rows");
  uint64_t handle = layer == 1 ? 1 : 2;
  size_t length = 16 + (size_t)count * 16;
  uint8_t *p = calloc(1, length);
  if (!p)
    return ria_fail(e, RIA_RESOURCE_LIMIT, "Engram request allocation failed");
  ria_write_u64(p, handle);
  ria_write_u32(p + 8, count);
  for (uint32_t i = 0; i < count; i++) {
    ria_write_u64(p + 16 + (size_t)i * 16, rows[i]);
    ria_write_u64(p + 24 + (size_t)i * 16, associations[i]);
  }
  ria_table table = {
      handle, layer == 1 ? UINT64_C(384006168) : UINT64_C(384016682), 264, 1};
  ria_row_request parsed;
  ria_header reply;
  uint8_t *payload = NULL;
  bool ok = ria_rows_parse(p, length, &table, &r->binding.limits, &parsed, e) &&
            exchange(r, RIA_ROWS, p, length, parsed.response_bytes, &reply,
                     &payload, e) &&
            ria_rows_response(payload, (size_t)reply.payload_length, &parsed,
                              &table, e) &&
            ria_binding_terminal(&r->binding, reply.request_id, true, e);
  if (ok)
    memcpy(out, payload + 24 + (size_t)count * 16, (size_t)count * 264);
  free(p);
  free(payload);
  if (!ok)
    retire(r);
  return ok;
}
static bool callback_experts(void *c, uint32_t layer, const float *input,
                             const uint16_t *ids, const float *weights,
                             const uint16_t *slots, uint32_t count, float *out,
                             ria_error *e) {
  ria_remote *r = c;
  if (r->invocation == UINT64_MAX)
    return ria_fail(e, RIA_RESOURCE_LIMIT,
                    "logical invocation counter exhausted");
  uint64_t id = ++r->invocation;
  uint32_t offsets[2] = {0, count};
  if (count > 6)
    return ria_fail(e, RIA_INVALID_REQUEST,
                    "selected expert count exceeds native graph contract");
  return ria_remote_evaluate(r, layer, id, 1, &id, offsets, ids, slots, weights,
                             input, false, out, e);
}
static bool callback_engram(void *c, uint32_t layer, const uint64_t *ids,
                            uint32_t count, uint8_t *out, ria_error *e) {
  ria_remote *r = c;
  uint64_t associations[24];
  if (count > 24)
    return ria_fail(e, RIA_INVALID_REQUEST,
                    "Engram columns exceed source graph contract");
  for (uint32_t i = 0; i < count; i++)
    associations[i] = i;
  return ria_remote_rows(r, layer, ids, associations, count, out, e);
}
ria_graph_remote ria_remote_callbacks(ria_remote *r) {
  return (ria_graph_remote){r, callback_experts, callback_engram};
}
bool ria_remote_chunk(ria_remote *r, const ria_shard *a, uint64_t index,
                      uint8_t **out, size_t *length, ria_error *e) {
  *out = NULL;
  *length = 0;
  if (!a || index >= a->chunk_count ||
      a->chunk_size > r->binding.limits.bulk_data_bytes)
    return ria_fail(e, RIA_UNAUTHORIZED,
                    "bulk chunk outside provisioned grant");
  uint8_t request[16];
  ria_write_u64(request, a->id);
  ria_write_u64(request + 8, index);
  uint64_t offset = index * a->chunk_size, n = a->length - offset;
  if (n > a->chunk_size)
    n = a->chunk_size;
  ria_header reply;
  uint8_t *payload = NULL;
  const uint8_t *bytes;
  size_t size;
  bool ok =
      exchange(r, RIA_CHUNK, request, 16, n + 64, &reply, &payload, e) &&
      ria_chunk_response(payload, (size_t)reply.payload_length, a->id, index,
                         a->length, a->chunk_size, a->chunk_hashes + index * 32,
                         &bytes, &size, e) &&
      ria_binding_terminal_channel(&r->binding, reply.request_id, true, true,
                                   e);
  if (ok) {
    *out = malloc(size);
    if (!*out)
      ok = ria_fail(e, RIA_RESOURCE_LIMIT,
                    "verified bulk output allocation failed");
    else {
      memcpy(*out, bytes, size);
      *length = size;
    }
  }
  free(payload);
  if (!ok)
    retire(r);
  return ok;
}
bool ria_remote_close(ria_remote *r, ria_error *e) {
  if (!r)
    return true;
  bool ok = true;
  if (r->binding.valid && r->binding.bound) {
    ria_header reply;
    uint8_t *payload = NULL;
    ok = exchange(r, RIA_CLOSE, NULL, 0, 0, &reply, &payload, e);
    if (ok && reply.payload_length)
      ok = ria_fail(e, RIA_INVALID_REQUEST, "Close reply must be empty");
    if (ok)
      ok = ria_binding_terminal(&r->binding, reply.request_id, true, e);
    free(payload);
  }
  pthread_mutex_lock(&r->lifecycle);
  ria_transport_close(&r->control);
  ria_transport_close(&r->bulk);
  pthread_mutex_unlock(&r->lifecycle);
  ria_tls_destroy(&r->tls);
  pthread_mutex_destroy(&r->lifecycle);
  free(r);
  return ok;
}
