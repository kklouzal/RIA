#define _GNU_SOURCE
#include "remote.h"
#include <openssl/crypto.h>
#include <math.h>
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
  unsigned expert_group_limit, expert_batch_limit, row_group_limit, prefill_rows;
  float *expert_results,*batch_results;
  uint8_t *row_results;
  bool aborted;
};
bool ria_remote_host_required_bytes(uint32_t rows,const ria_limits *l,uint64_t *bytes,ria_error *e) {
  if (!l || !bytes || !rows || rows>64 || !l->expert_rows || l->expert_rows>64 ||
      l->frame_payload_bytes<RIA_ERROR_MAX || !l->row_lookup_rows)
    return ria_fail(e,RIA_INVALID_REQUEST,"invalid remote prefill reservation bounds");
  uint64_t fixed,request,response,transient,total,dom,canonical_bytes,keys;
  /* Bind retains its outbound JSON while receiving/validating the response;
   * the response and original DOM may coexist. Canonical validation and
   * nested key ordering own temporary storage, including invalid inputs. */
  if (!ria_json_control_required_bytes((ria_json_limits){RIA_CONTROL_MAX,8192,32},&dom,&canonical_bytes,&keys,e) ||
      !ria_u64_mul(dom,2,&transient) || !ria_u64_add(transient,2*RIA_CONTROL_MAX,&transient) ||
      !ria_u64_add(transient,canonical_bytes,&transient) || !ria_u64_add(transient,keys,&transient))
    return ria_fail(e,RIA_RESOURCE_LIMIT,"remote control reservation overflow");
  if (!ria_u64_mul((uint64_t)rows+RIA_GRAPH_SELECTED,RIA_GRAPH_DIM*sizeof(float),&fixed) ||
      !ria_u64_add(fixed,sizeof(ria_remote)+RIA_GRAPH_HASH_COLUMNS*264,&fixed))
    return ria_fail(e,RIA_RESOURCE_LIMIT,"remote publication reservation overflow");
  uint32_t n=(uint32_t)l->expert_rows;
  for (unsigned shape=0;shape<2;shape++) {
    uint32_t count=shape ? 1 : n,entries=shape ? RIA_GRAPH_SELECTED : n*RIA_GRAPH_SELECTED;
    if (!ria_expert_lengths(count,entries,RIA_GRAPH_DIM,RIA_GRAPH_DIM,0,&request,&response,e)) return false;
    if (request>l->frame_payload_bytes) request=l->frame_payload_bytes;
    if (response<RIA_ERROR_MAX) response=RIA_ERROR_MAX;
    if (response>l->frame_payload_bytes) response=l->frame_payload_bytes;
    if (!ria_u64_add(request,response,&total)) goto overflow;
    if (total>transient) transient=total;
  }
  uint64_t row_count=l->row_lookup_rows;
  uint64_t fit=(l->frame_payload_bytes-24)/280;
  if (row_count>fit) row_count=fit;
  fit=(l->frame_payload_bytes-16)/16;if (row_count>fit) row_count=fit;
  request=16+row_count*16;response=24+row_count*280;
  if (response<RIA_ERROR_MAX) response=RIA_ERROR_MAX;
  if (!ria_u64_add(request,response,&total)) goto overflow;
  if (total>transient) transient=total;
  if (!ria_u64_add(l->bulk_data_bytes,64,&response)) goto overflow;
  if (response<RIA_ERROR_MAX) response=RIA_ERROR_MAX;
  /* Verified chunk output overlaps its received envelope until the copy has
   * completed; account the caller-owned result too at this boundary. */
  if (!ria_u64_add(16,response,&total) || !ria_u64_add(total,l->bulk_data_bytes,&total)) goto overflow;
  if (total>transient) transient=total;
  if (!ria_u64_add(fixed,transient,bytes)) goto overflow;
  return true;
overflow:return ria_fail(e,RIA_RESOURCE_LIMIT,"remote transient reservation overflow");
}
bool ria_request_charge(uint16_t kind, uint64_t request, uint64_t response,
                        uint64_t *charge, ria_error *e) {
  /* Schema-v1 fixed credit table. The 8MiB expert allowance covers the
   * selected executor's admitted <=64x5120 / intermediate2304 tile64
   * workspace, descriptors, FP32 host row/output staging and quantizers.
   * A typed terminal error may exceed a small successful reply; reserve the
   * larger payload at both endpoints. Actual endpoint pools are separately
   * checked against the memory plan. */
  uint64_t workspace =
      (kind == RIA_EXPERT || kind == RIA_SHARED) ? UINT64_C(8388608) : 0;
  if (response<RIA_ERROR_MAX) response=RIA_ERROR_MAX;
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
static bool remote_error(const ria_header *h,const uint8_t *payload,ria_error *e) {
  ria_json_doc d={0};
  if (!ria_control_json(payload,(size_t)h->payload_length,h,&d,e)) { ria_json_free(&d); return false; }
  const char *message=NULL; size_t n=0;
  bool ok=ria_json_string(&d,ria_json_get(&d,0,"message"),&message,&n,e);
  if (ok) {
    size_t limit=n>192 ? 192 : n;
    while (limit && limit<n && ((unsigned char)message[limit]&0xc0)==0x80) limit--;
    ria_error_set(e,(int)h->status,"remote operation failed: %.*s",(int)limit,message);
  }
  ria_json_free(&d); return false;
}
static bool group_limits(ria_remote *r,ria_error *e) {
  const ria_limits *l=&r->binding.limits; uint64_t progress,room;
  if (!ria_u64_add(r->binding.progress_bytes[0],r->binding.progress_bytes[1],&progress) ||
      !ria_u64_add(progress,r->binding.progress_bytes[2],&progress) || progress>=l->inflight_payload_bytes)
    return ria_fail(e,RIA_RESOURCE_LIMIT,"protected progress leaves no expert unit");
  room=l->inflight_payload_bytes-progress;
  for (unsigned entries=1;entries<=RIA_GRAPH_SELECTED;entries++) {
    uint64_t request,response,charge;
    if (!ria_expert_lengths(1,entries,RIA_GRAPH_DIM,RIA_GRAPH_DIM,0,&request,&response,e) ||
        !ria_request_charge(RIA_EXPERT,request,response,&charge,e)) return false;
    if (request>l->frame_payload_bytes || response>l->frame_payload_bytes || charge>room) break;
    r->expert_group_limit=entries;
  }
  for (unsigned rows=1;rows<=r->prefill_rows && rows<=l->expert_rows;rows++) {
    uint64_t request,response,charge;
    if (!ria_expert_lengths(rows,rows,RIA_GRAPH_DIM,RIA_GRAPH_DIM,0,&request,&response,e) ||
        !ria_request_charge(RIA_EXPERT,request,response,&charge,e)) return false;
    if (request>l->frame_payload_bytes || response>l->frame_payload_bytes || charge>room) break;
    r->expert_batch_limit=rows;
  }
  uint64_t rows=(l->frame_payload_bytes-24)/(264+16);
  uint64_t request_rows=(l->frame_payload_bytes-16)/16;
  if (rows>request_rows) rows=request_rows;
  if (rows>l->row_lookup_rows) rows=l->row_lookup_rows;
  if (rows>RIA_GRAPH_HASH_COLUMNS) rows=RIA_GRAPH_HASH_COLUMNS;
  r->row_group_limit=(unsigned)rows;
  if (l->frame_payload_bytes<RIA_ERROR_MAX || !r->expert_group_limit || !r->expert_batch_limit || !r->row_group_limit)
    return ria_fail(e,RIA_RESOURCE_LIMIT,"binding cannot hold one native expert/Engram/error unit");
  /* Only split realizations need an extra atomic-publication staging pool.
   * Fixed sizes are included in endpoint runtime admission; no HOT allocation. */
  if (r->expert_group_limit<RIA_GRAPH_SELECTED) {
    r->expert_results=calloc(RIA_GRAPH_SELECTED*RIA_GRAPH_DIM,sizeof(float));
    if (!r->expert_results) return ria_fail(e,RIA_RESOURCE_LIMIT,"split expert publication pool unavailable");
  }
  if (r->row_group_limit<RIA_GRAPH_HASH_COLUMNS) {
    r->row_results=calloc(RIA_GRAPH_HASH_COLUMNS,264);
    if (!r->row_results) return ria_fail(e,RIA_RESOURCE_LIMIT,"split row publication pool unavailable");
  }
  r->batch_results=calloc((size_t)r->prefill_rows*RIA_GRAPH_DIM,sizeof(float));
  if (!r->batch_results) return ria_fail(e,RIA_RESOURCE_LIMIT,"grouped prefill publication pool unavailable");
  return true;
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
  uint64_t receive_limit=response_bytes>RIA_ERROR_MAX ? response_bytes : RIA_ERROR_MAX;
  if (receive_limit>r->binding.limits.frame_payload_bytes) receive_limit=r->binding.limits.frame_payload_bytes;
  if (!ria_transport_send(t, &h, request, write_deadline, e) ||
      !ria_transport_frame(t, deadline, r->binding.limits.frame_io_timeout_ms,
                           receive_limit, reply_header,
                           reply, e) ||
      !ria_binding_response(&r->binding, reply_header, e))
    goto fail;
  if (reply_header->status) {
    remote_error(reply_header,*reply,e);
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
  if (ok && reply.status) {
    uint8_t session_bits=0;
    for (unsigned i=0;i<16;i++) session_bits|=reply.session[i];
    if (reply.kind!=RIA_BIND || reply.flags!=1 || reply.request_id!=1 || reply.epoch || session_bits)
      ok=ria_fail(e,RIA_IDENTITY_MISMATCH,"failed initial Bind changed association/session");
    else ok=remote_error(&reply,payload,e);
  }
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
         ria_binding_install(&r->binding, session, epoch, cap, e) && group_limits(r,e);
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
            reply.kind == RIA_BIND_BULK && reply.flags == 1 &&
            reply.request_id == 1 && reply.epoch == h.epoch &&
            !memcmp(reply.session, h.session, 16);
  ria_json_doc d = {0};
  if (ok && reply.status) ok=remote_error(&reply,payload,e);
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
  r->prefill_rows=o->service->prefill_rows;
  if (!r->prefill_rows || r->prefill_rows>64) {
    free(r);return ria_fail(e,RIA_INVALID_REQUEST,"remote prefill row bound is not admitted");
  }
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
  if (ok)
    ok = ria_transport_same_peer(&r->control, &r->bulk, e) && bind_bulk(r, e);
  if (!ok) {
    ria_transport_close(&r->control);
    ria_transport_close(&r->bulk);
    ria_tls_destroy(&r->tls);
    pthread_mutex_destroy(&r->lifecycle);
    free(r->expert_results); free(r->batch_results); free(r->row_results);
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
  if (!entries || entries>rows*(shared ? 1u : RIA_GRAPH_SELECTED))
    return ria_fail(e,RIA_INVALID_REQUEST,"remote selection exceeds declared row contract");
  uint64_t request_bytes, response_bytes;
  if (!ria_expert_lengths(rows, entries, 5120, 5120, 0, &request_bytes,
                          &response_bytes, e) ||
      request_bytes > SIZE_MAX || request_bytes>r->binding.limits.frame_payload_bytes ||
      response_bytes>r->binding.limits.frame_payload_bytes || rows>r->binding.limits.expert_rows)
    return ria_fail(e,RIA_RESOURCE_LIMIT,"expert unit exceeds agreed frame/row limit");
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
  uint64_t pairs,request_bytes,response_bytes;
  if (!ria_u64_mul(count,16,&pairs) || !ria_u64_add(16,pairs,&request_bytes) ||
      !ria_u64_mul(count,280,&response_bytes) || !ria_u64_add(24,response_bytes,&response_bytes) ||
      request_bytes>r->binding.limits.frame_payload_bytes || response_bytes>r->binding.limits.frame_payload_bytes ||
      request_bytes>SIZE_MAX)
    return ria_fail(e,RIA_RESOURCE_LIMIT,"Engram unit exceeds agreed frame limit");
  size_t length=(size_t)request_bytes;
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
  if (count>RIA_GRAPH_SELECTED || !count || !input || !ids || !weights || !slots || !out || !r->expert_group_limit)
    return ria_fail(e,RIA_INVALID_REQUEST,"selected expert count exceeds native graph contract");
  unsigned groups=(count+r->expert_group_limit-1)/r->expert_group_limit;
  if (UINT64_MAX-r->invocation<groups) {
    retire(r);
    return ria_fail(e, RIA_RESOURCE_LIMIT,
                    "logical invocation counter exhausted");
  }
  uint64_t row_id=r->invocation+1;
  if (count<=r->expert_group_limit) {
    uint64_t invocation=++r->invocation; uint32_t offsets[2]={0,count};
    return ria_remote_evaluate(r,layer,invocation,1,&row_id,offsets,ids,slots,weights,input,false,out,e);
  }
  for (unsigned begin=0;begin<count;) {
    unsigned n=count-begin; if (n>r->expert_group_limit) n=r->expert_group_limit;
    uint64_t invocation=++r->invocation; uint32_t offsets[2]={0,n};
    if (!ria_remote_evaluate(r,layer,invocation,1,&row_id,offsets,ids+begin,slots+begin,weights+begin,
                            input,false,r->expert_results+(size_t)begin*RIA_GRAPH_DIM,e)) return false;
    begin+=n;
  }
  memcpy(out,r->expert_results,(size_t)count*RIA_GRAPH_DIM*sizeof(float)); return true;
}
static bool callback_engram(void *c, uint32_t layer, const uint64_t *ids,
                            uint32_t count, uint8_t *out, ria_error *e) {
  ria_remote *r = c;
  uint64_t associations[24];
  if (count > 24)
    return ria_fail(e, RIA_INVALID_REQUEST,
                    "Engram columns exceed source graph contract");
  if (!count || !ids || !out || !r->row_group_limit) return ria_fail(e,RIA_INVALID_REQUEST,"invalid Engram callback unit");
  for (uint32_t i = 0; i < count; i++)
    associations[i] = i;
  if (count<=r->row_group_limit) return ria_remote_rows(r,layer,ids,associations,count,out,e);
  for (unsigned begin=0;begin<count;) {
    unsigned n=count-begin; if (n>r->row_group_limit) n=r->row_group_limit;
    if (!ria_remote_rows(r,layer,ids+begin,associations+begin,n,r->row_results+(size_t)begin*264,e)) return false;
    begin+=n;
  }
  memcpy(out,r->row_results,(size_t)count*264); return true;
}
static bool callback_experts_batch(void *c,uint32_t layer,uint16_t expert,uint32_t rows,
                                    const uint64_t *row_ids,const float *inputs,
                                    const float *coefficients,const uint16_t *slots,
                                    float *out,ria_error *e) {
  ria_remote *r=c;
  if (!r || !rows || rows>r->prefill_rows || layer>=RIA_GRAPH_LAYERS || expert>=RIA_GRAPH_EXPERTS ||
      !row_ids || !inputs || !coefficients || !slots || !out || !r->expert_batch_limit)
    return ria_fail(e,RIA_INVALID_REQUEST,"invalid expert-grouped prompt unit");
  for (uint32_t row=0;row<rows;row++) {
    if (slots[row]>=RIA_GRAPH_SELECTED || !isfinite(coefficients[row]) || coefficients[row]<0 || coefficients[row]>1.5f)
      return ria_fail(e,RIA_INVALID_REQUEST,"invalid grouped prompt slot/coefficient");
    for (uint32_t earlier=0;earlier<row;earlier++) if (row_ids[earlier]==row_ids[row])
      return ria_fail(e,RIA_INVALID_REQUEST,"duplicate grouped prompt row identity");
  }
  /* Validate every input before the first subgroup is sent. Its wire parser
   * independently validates each final serialized unit at the boundary. */
  for (uint64_t i=0;i<(uint64_t)rows*RIA_GRAPH_DIM;i++) {
    uint32_t bits;memcpy(&bits,inputs+i,sizeof bits);
    if (!isfinite(inputs[i]) || (bits&65535u))
      return ria_fail(e,RIA_INVALID_REQUEST,"grouped prompt input is not finite BF16-logical FP32");
  }
  uint32_t groups=(rows+r->expert_batch_limit-1)/r->expert_batch_limit;
  if (UINT64_MAX-r->invocation<groups) {
    retire(r);return ria_fail(e,RIA_RESOURCE_LIMIT,"logical invocation counter exhausted");
  }
  float *destination=groups>1 ? r->batch_results : out;
  uint32_t offsets[65];uint16_t experts[64];
  for (uint32_t begin=0;begin<rows;) {
    uint32_t count=rows-begin;if (count>r->expert_batch_limit) count=r->expert_batch_limit;
    for (uint32_t row=0;row<count;row++) { offsets[row]=row;experts[row]=expert; }
    offsets[count]=count;
    uint64_t invocation=++r->invocation;
    if (!ria_remote_evaluate(r,layer,invocation,count,row_ids+begin,offsets,experts,slots+begin,
                             coefficients+begin,inputs+(size_t)begin*RIA_GRAPH_DIM,false,
                             destination+(size_t)begin*RIA_GRAPH_DIM,e)) return false;
    begin+=count;
  }
  if (groups>1) memcpy(out,destination,(size_t)rows*RIA_GRAPH_DIM*sizeof(float));
  return true;
}
ria_graph_remote ria_remote_callbacks(ria_remote *r) {
  return (ria_graph_remote){.context=r,.experts=callback_experts,.engram=callback_engram,
                            .experts_batch=callback_experts_batch};
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
  free(r->expert_results); free(r->batch_results); free(r->row_results);
  free(r);
  return ok;
}
