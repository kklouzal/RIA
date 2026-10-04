#define _POSIX_C_SOURCE 200809L
#include "engine.h"
#include "graph_cuda.h"
#include "runtime.h"
#include <math.h>
#include <openssl/crypto.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
struct ria_engine {
  ria_service service;
  ria_tensor_store store;
  ria_tokenizer *tokenizer;
  ria_graph *graph;
  ria_remote *remote;
  ria_graph_options graph_options;
  ria_remote_options remote_options;
  pthread_mutex_t ownership;
  bool mutex_ready, runtime_ready, claimed, valid, fatal, retired;
  uint64_t epoch, generation, context, frontend, frontend_fixed, outer_owner;
  float *last_logits;
  uint32_t *prefix, *token_map;
  ria_graph_local_expert *local_experts;
  uint64_t position, last_reused_tokens;
  bool last_sync_recorded, last_continuation;
  struct {
    uint32_t start, count;
    uint8_t digest[32];
  } images[128];
  size_t image_count;
};
bool ria_engine_process_policy(ria_error *e) {
  struct sigaction action;
  memset(&action, 0, sizeof action);
  action.sa_handler = SIG_IGN;
  if (sigemptyset(&action.sa_mask) || sigaction(SIGPIPE, &action, NULL))
    return ria_fail(e, RIA_INTERNAL_ERROR,
                    "cannot establish RIA process SIGPIPE policy");
  return true;
}
bool ria_logits_nll(const float *logits, uint32_t label, double *loss,
                    ria_error *e) {
  if (!logits || !loss || label >= RIA_GRAPH_VOCAB)
    return ria_fail(e, RIA_INVALID_REQUEST, "invalid native NLL inputs");
  double maximum = -INFINITY;
  for (uint32_t i = 0; i < RIA_GRAPH_VOCAB; i++) {
    if (!isfinite(logits[i]))
      return ria_fail(e, RIA_EXECUTOR_ERROR, "nonfinite native NLL logits");
    if ((double)logits[i] > maximum)
      maximum = (double)logits[i];
  }
  double sum = 0;
  for (uint32_t i = 0; i < RIA_GRAPH_VOCAB; i++)
    sum += exp((double)logits[i] - maximum);
  /* Keep the common offset cancellation separate from the log denominator. */
  double nll = log(sum) + (maximum - (double)logits[label]);
  if (!isfinite(nll) || nll < 0)
    return ria_fail(e, RIA_EXECUTOR_ERROR, "nonfinite native NLL logits");
  *loss = nll;
  return true;
}
static bool integer(const ria_json_doc *d, uint32_t o, const char *k,
                    bool string, uint64_t *v, ria_error *e) {
  return ria_json_u64(d, ria_json_get(d, o, k), string, v, e);
}
static bool frontend_prefix_bytes(uint64_t context, uint64_t *bytes) {
  uint64_t capacity = 64;
  while (capacity < context)
    capacity *= 2;
  return ria_u64_add(context, capacity, bytes) &&
         ria_u64_mul(*bytes, sizeof(uint32_t), bytes);
}
static bool client_preflight(void *context, const ria_tensor_store *store,
                             const ria_shard *shard, bool populated,
                             ria_error *e) {
  ria_engine *r = context;
  (void)populated;
  if (shard)
    return true;
  const ria_json_doc *p = &r->service.placement;
  uint32_t runtime = ria_json_get(p, 0, "runtime");
  uint64_t tokenizer, host, frontend, projection, patches, engram, owned, total,
      pinned, locked = 0;
  if (!integer(p, runtime, "tokenizer_memory_bytes", true, &tokenizer, e) ||
      !integer(p, runtime, "host_state_bytes", true, &host, e) ||
      !integer(p, runtime, "frontend_host_bytes", true, &frontend, e) ||
      !integer(p, runtime, "projection_tile_rows", false, &projection, e) ||
      projection > UINT32_MAX ||
      !integer(p, runtime, "max_image_patches", false, &patches, e) ||
      patches > UINT32_MAX ||
      !integer(p, 0, "engram_cache_bytes", false, &engram, e) ||
      !ria_tensor_owned_bytes(store, &owned, e))
    return false;
  const ria_json_node *entries =
      ria_json_at(p, ria_json_get(p, 0, "local_experts"));
  uint32_t local_count = 0;
  if (!entries || entries->type != RIA_JSON_ARRAY)
    return ria_fail(e, RIA_INVALID_REQUEST, "local membership missing");
  for (uint32_t i = entries->child; i != RIA_JSON_NONE; i = p->nodes[i].next)
    if (++local_count > 15360)
      return ria_fail(e, RIA_RESOURCE_LIMIT,
                      "local membership exceeds source population");
  ria_graph_options options = {.projection_tile_rows = (uint32_t)projection,
                               .max_image_patches = (uint32_t)patches,
                               .max_tokens = r->context,
                               .local_expert_count = local_count,
                               .engram_cache_budget = engram};
  uint64_t private_required;
  if (!ria_graph_host_required_bytes(&options, &private_required, e) ||
      private_required > host)
    return ria_fail(
        e, RIA_RESOURCE_LIMIT,
        "complete private graph state exceeds reservation before first touch");
  if (!ria_graph_pinned_required_bytes(&options, &pinned, e) ||
      pinned > r->service.pinned_cap ||
      !ria_u64_add(owned, tokenizer, &total) ||
      !ria_u64_add(total, host, &total) ||
      !ria_u64_add(total, frontend, &total) ||
      !ria_u64_add(total, pinned, &total) || total > r->service.host_cap)
    return ria_fail(
        e, RIA_RESOURCE_LIMIT,
        "complete client population exceeds host cap before first touch");
  long page = sysconf(_SC_PAGESIZE);
  if (page <= 0 || ((uint64_t)page & ((uint64_t)page - 1)))
    return ria_fail(e, RIA_UNSUPPORTED, "invalid process page size");
  for (uint64_t i = 0; i < store->shard_count; i++) {
    uint64_t bytes;
    if (!ria_u64_add(store->shards[i].length, (uint64_t)page - 1, &bytes) ||
        !ria_u64_add(locked, bytes & ~((uint64_t)page - 1), &locked))
      return ria_fail(e, RIA_RESOURCE_LIMIT,
                      "locked client population overflow");
  }
  if (!ria_u64_add(locked, host, &locked))
    return ria_fail(e, RIA_RESOURCE_LIMIT, "private locked state overflow");
  ria_runtime_observation actual;
  return ria_runtime_require(r->service.host_cap, locked, r->service.pinned_cap,
                             &actual, e);
}
static bool local_placement(ria_engine *r, ria_error *e) {
  const ria_json_doc *p = &r->service.placement;
  static const char *const fields[] = {"schema_revision",
                                       "logical_model_digest",
                                       "operator_contract_digest",
                                       "server_layout_digest",
                                       "server_executor",
                                       "schedule",
                                       "shared_placement",
                                       "expert_policy",
                                       "host_expert_cache_bytes",
                                       "device_expert_cache_bytes",
                                       "engram_cache_bytes",
                                       "local_experts",
                                       "runtime",
                                       "digest"};
  if (!ria_json_fields(p, 0, fields, 14, fields, 14, e))
    return false;
  const ria_json_node *policy =
                          ria_json_at(p, ria_json_get(p, 0, "expert_policy")),
                      *a = ria_json_at(p, ria_json_get(p, 0, "local_experts"));
  if (!policy || policy->type != RIA_JSON_STRING ||
      (strcmp(policy->text, "remote") && strcmp(policy->text, "explicit")) ||
      !a || a->type != RIA_JSON_ARRAY ||
      !integer(p, 0, "host_expert_cache_bytes", false,
               &r->graph_options.host_expert_budget, e) ||
      !integer(p, 0, "device_expert_cache_bytes", false,
               &r->graph_options.device_expert_budget, e) ||
      !integer(p, 0, "engram_cache_bytes", false,
               &r->graph_options.engram_cache_budget, e))
    return ria_fail(e, RIA_INVALID_REQUEST,
                    "invalid explicit client placement policy");
  uint32_t count = 0;
  for (uint32_t i = a->child; i != RIA_JSON_NONE; i = p->nodes[i].next)
    if (++count > 15360)
      return ria_fail(e, RIA_RESOURCE_LIMIT, "too many local expert entries");
  if (!strcmp(policy->text, "remote") && count)
    return ria_fail(e, RIA_INVALID_REQUEST,
                    "remote policy has local membership");
  if (count) {
    uint64_t bytes, minimum;
    if (!ria_u64_mul(count, sizeof *r->local_experts, &bytes) ||
        !ria_u64_add(r->outer_owner, bytes, &r->outer_owner) ||
        bytes > r->frontend || !frontend_prefix_bytes(r->context, &minimum) ||
        !ria_u64_add(minimum, r->outer_owner, &minimum) ||
        !ria_u64_add(minimum, 129280 * 8u + sizeof *r, &minimum) ||
        minimum >= r->frontend)
      return ria_fail(e, RIA_RESOURCE_LIMIT,
                      "local descriptor reservation exceeded");
    r->local_experts = calloc(count, sizeof *r->local_experts);
    if (!r->local_experts)
      return ria_fail(e, RIA_RESOURCE_LIMIT,
                      "local descriptor allocation failed");
  }
  r->graph_options.local_experts = r->local_experts;
  r->graph_options.local_expert_count = count;
  unsigned index = 0;
  for (uint32_t i = a->child; i != RIA_JSON_NONE;
       i = p->nodes[i].next, index++) {
    static const char *const entry_fields[] = {"layer", "expert", "tier",
                                               "local_phases"};
    uint64_t layer, id;
    const ria_json_node *tier = ria_json_at(p, ria_json_get(p, i, "tier")),
                        *phases =
                            ria_json_at(p, ria_json_get(p, i, "local_phases"));
    if (!ria_json_fields(p, i, entry_fields, 4, entry_fields, 4, e) ||
        !integer(p, i, "layer", false, &layer, e) || layer >= 40 ||
        !integer(p, i, "expert", false, &id, e) || id >= 384 || !tier ||
        tier->type != RIA_JSON_STRING ||
        (strcmp(tier->text, "host") && strcmp(tier->text, "vram")) || !phases ||
        phases->type != RIA_JSON_ARRAY || phases->child == RIA_JSON_NONE)
      return ria_fail(e, RIA_INVALID_REQUEST,
                      "invalid local expert placement entry");
    ria_graph_local_expert *entry = &r->local_experts[index];
    entry->layer = (uint32_t)layer;
    entry->expert_id = (uint16_t)id;
    entry->tier = !strcmp(tier->text, "host") ? RIA_GRAPH_HOST : RIA_GRAPH_VRAM;
    for (uint32_t j = phases->child; j != RIA_JSON_NONE; j = p->nodes[j].next) {
      const ria_json_node *phase = &p->nodes[j];
      unsigned bit;
      if (phase->type != RIA_JSON_STRING)
        return ria_fail(e, RIA_INVALID_REQUEST, "invalid local phase");
      if (!strcmp(phase->text, "prefill"))
        bit = 1;
      else if (!strcmp(phase->text, "decode"))
        bit = 2;
      else if (!strcmp(phase->text, "continuation"))
        bit = 4;
      else
        return ria_fail(e, RIA_UNSUPPORTED, "unrecognized local phase");
      if (entry->phase_mask & bit)
        return ria_fail(e, RIA_INVALID_REQUEST, "duplicate local phase");
      entry->phase_mask |= (uint8_t)bit;
    }
    const char *const suffixes[] = {"w1", "w3", "w2"};
    ria_expert_matrix *matrices[] = {&entry->expert.gate, &entry->expert.up,
                                     &entry->expert.down};
    for (unsigned m = 0; m < 3; m++) {
      char name[128];
      int n = snprintf(name, sizeof name, "layers.%u.ffn.experts.%u.%s.weight",
                       (unsigned)layer, (unsigned)id, suffixes[m]);
      const ria_tensor *tensor = n > 0 && (size_t)n < sizeof name
                                     ? ria_tensor_name(&r->store, name)
                                     : NULL;
      if (!tensor || !ria_tensor_matrix(&r->store, tensor, matrices[m], e))
        return ria_fail(
            e, RIA_INTEGRITY_ERROR,
            "local membership requires complete verified expert triplet");
    }
    entry->expert.clamp = 10;
  }
  uint64_t host, device;
  return ria_graph_local_experts_validate(&r->graph_options, &host, &device, e);
}
static bool config(ria_engine *r, ria_error *e) {
  const ria_json_doc *p = &r->service.placement;
  uint32_t runtime = ria_json_get(p, 0, "runtime");
  static const char *const fields[] = {
      "tokenizer_file",   "tokenizer_sha256",   "tokenizer_memory_bytes",
      "host_state_bytes", "device_state_bytes", "projection_tile_rows",
      "state_tile_rows",  "max_image_patches",  "frontend_host_bytes"};
  if (!ria_json_fields(p, runtime, fields, 9, fields, 9, e))
    return false;
  const char *path = NULL;
  size_t length = 0;
  uint8_t sha[32];
  uint64_t tokenizer_budget, host, device, projection, state, patches;
  if (!ria_json_string(p, ria_json_get(p, runtime, "tokenizer_file"), &path,
                       &length, e) ||
      !length || path[0] != '/' ||
      !ria_json_digest_field(p, ria_json_get(p, runtime, "tokenizer_sha256"),
                             sha, e) ||
      !integer(p, runtime, "tokenizer_memory_bytes", true, &tokenizer_budget,
               e) ||
      !integer(p, runtime, "frontend_host_bytes", true, &r->frontend, e) ||
      !integer(p, runtime, "host_state_bytes", true, &host, e) ||
      !integer(p, runtime, "device_state_bytes", true, &device, e) ||
      !integer(p, runtime, "projection_tile_rows", false, &projection, e) ||
      !integer(p, runtime, "state_tile_rows", false, &state, e) ||
      !integer(p, runtime, "max_image_patches", false, &patches, e) ||
      !projection || !state || !patches || projection > UINT32_MAX ||
      state > UINT32_MAX || patches > UINT32_MAX ||
      host > r->service.host_cap || device > r->service.device_cap ||
      tokenizer_budget > r->service.host_cap)
    return ria_fail(e, RIA_RESOURCE_LIMIT,
                    "placement runtime limits exceed admitted capacities");
  r->graph_options =
      (ria_graph_options){.device = 0,
                          .gpu_uuid = r->service.gpu_uuid,
                          .max_tokens = r->context,
                          .host_state_budget = host,
                          .device_budget = device,
                          .pinned_budget = r->service.pinned_cap,
                          .projection_tile_rows = (uint32_t)projection,
                          .state_tile_rows = (uint32_t)state,
                          .max_image_patches = (uint32_t)patches,
                          .compressed_vocab = 99092};
  if (!local_placement(r, e))
    return false;
  uint64_t pinned_required;
  if (!ria_graph_pinned_required_bytes(&r->graph_options, &pinned_required,
                                       e) ||
      pinned_required > r->service.pinned_cap)
    return ria_fail(e, RIA_RESOURCE_LIMIT,
                    "graph pinned staging exceeds admitted cap");
  uint8_t expected[32];
  uint64_t prefix_bytes, live, store_owned, service_owned = 0;
  const ria_json_doc *service_documents[] = {
      &r->service.document, &r->service.lock, &r->service.plan,
      &r->service.placement, &r->service.grants};
  for (size_t i = 0; i < 5; i++)
    if (!ria_u64_add(service_owned, service_documents[i]->allocated_bytes,
                     &service_owned))
      return ria_fail(e, RIA_RESOURCE_LIMIT,
                      "retained service documents overflow");
  if (!ria_u64_mul(r->context, sizeof(uint32_t), &prefix_bytes) ||
      !frontend_prefix_bytes(r->context, &r->frontend_fixed) ||
      !ria_u64_add(r->frontend_fixed,
                   129280 * (sizeof(uint32_t) + sizeof(float)) + sizeof *r,
                   &r->frontend_fixed) ||
      !ria_u64_add(r->frontend_fixed, r->outer_owner, &r->frontend_fixed) ||
      !ria_u64_add(r->frontend_fixed, service_owned, &r->frontend_fixed) ||
      !ria_u64_add(r->frontend_fixed, ria_graph_cuda_metadata_bytes(),
                   &r->frontend_fixed) ||
      !ria_u64_add(
          r->frontend_fixed,
          r->graph_options.max_image_patches ? ria_vision_metadata_bytes() : 0,
          &r->frontend_fixed) ||
      r->frontend_fixed >= r->frontend ||
      !ria_tensor_owned_bytes(&r->store, &store_owned, e) ||
      !ria_u64_add(store_owned, tokenizer_budget, &live) ||
      !ria_u64_add(live, host, &live) ||
      !ria_u64_add(live, pinned_required, &live) ||
      !ria_u64_add(live, r->frontend, &live) || live > r->service.host_cap)
    return ria_fail(
        e, RIA_RESOURCE_LIMIT,
        "complete frontend/graph/metadata reservations exceed admitted cap");
  ria_runtime_observation actual;
  if (!ria_runtime_require(r->service.host_cap, r->store.resident_bytes,
                           r->service.pinned_cap, &actual, e))
    return false;
  if (!ria_json_digest_field(
          &r->store.manifest,
          ria_json_get(&r->store.manifest, 0, "tokenizer_digest"), expected,
          e) ||
      memcmp(sha, expected, 32))
    return ria_fail(e, RIA_INTEGRITY_ERROR,
                    "runtime tokenizer differs from prepared source identity");
  if (!ria_tokenizer_runtime_begin(e))
    return false;
  r->runtime_ready = true;
  if (!ria_tokenizer_open(path, sha, tokenizer_budget, &r->tokenizer, e))
    return false;
  const ria_tensor *map = ria_tensor_name(&r->store, "engram.token_map"),
                   *prime = ria_tensor_name(&r->store, "engram.primes"),
                   *mul = ria_tensor_name(&r->store, "engram.multipliers"),
                   *pad = ria_tensor_name(&r->store, "engram.pad_id"),
                   *offset = ria_tensor_name(&r->store, "engram.offsets");
  if (!map || !prime || !mul || !pad || !offset || strcmp(map->dtype, "U32") ||
      map->length != 129280 * 4u || strcmp(prime->dtype, "U64") ||
      prime->length != 2 * 24 * 8u || strcmp(mul->dtype, "I64") ||
      mul->length != 2 * 4 * 8u || strcmp(pad->dtype, "U32") ||
      pad->length != 4 || strcmp(offset->dtype, "U64") ||
      offset->length != 2 * 24 * 8u)
    return ria_fail(e, RIA_INTEGRITY_ERROR,
                    "prepared Engram hash metadata missing or incompatible");
  r->token_map = malloc(129280 * sizeof *r->token_map);
  if (!r->token_map)
    return ria_fail(e, RIA_RESOURCE_LIMIT,
                    "Engram token map allocation failed");
  for (unsigned i = 0; i < 129280; i++) {
    uint64_t v = ria_read_u32(map->data + i * 4u);
    if (v >= 99092)
      return ria_fail(e, RIA_INTEGRITY_ERROR,
                      "invalid prepared compressed token identity");
    r->token_map[i] = (uint32_t)v;
  }
  r->graph_options.compressed_token_map = r->token_map;
  uint64_t pad_id = ria_read_u32(pad->data);
  if (pad_id >= 99092)
    return ria_fail(e, RIA_INTEGRITY_ERROR, "invalid Engram pad identity");
  r->graph_options.pad_compressed_id = (uint32_t)pad_id;
  for (unsigned layer = 0; layer < 2; layer++) {
    uint64_t sum = 0;
    for (unsigned i = 0; i < 4; i++)
      r->graph_options.hash_multipliers[layer][i] =
          ria_read_u64(mul->data + (layer * 4 + i) * 8u);
    for (unsigned i = 0; i < 24; i++) {
      uint64_t v = ria_read_u64(prime->data + (layer * 24 + i) * 8u);
      if (ria_read_u64(offset->data + (layer * 24 + i) * 8u) != sum ||
          !ria_u64_add(sum, v, &sum))
        return ria_fail(e, RIA_INTEGRITY_ERROR, "invalid Engram table offsets");
      r->graph_options.hash_primes[layer][i] = v;
    }
  }
  r->prefix = malloc((size_t)prefix_bytes);
  r->last_logits = malloc(129280 * sizeof(float));
  if (!r->prefix || !r->last_logits)
    return ria_fail(e, RIA_RESOURCE_LIMIT,
                    "incorporated prefix/logit cache allocation failed");
  r->remote_options.service = &r->service;
  r->remote_options.client_store = &r->store;
  if (!ria_json_digest_field(
          &r->store.manifest,
          ria_json_get(&r->store.manifest, 0, "layout_digest"),
          r->remote_options.client_layout_digest, e) ||
      !ria_json_digest_field(p, ria_json_get(p, 0, "server_layout_digest"),
                             r->remote_options.server_layout_digest, e) ||
      !ria_json_sha256(p, true, r->remote_options.placement_plan_digest, e))
    return false;
  return ria_graph_options_validate(&r->graph_options, e);
}
bool ria_engine_open(const char *path, uint64_t outer_owner, ria_engine **out,
                     ria_error *e) {
  if (!path || !out)
    return ria_fail(e, RIA_INVALID_REQUEST, "invalid RIA engine configuration");
  *out = NULL;
  if (!ria_engine_process_policy(e))
    return false;
  ria_engine *r = calloc(1, sizeof *r);
  if (!r)
    return ria_fail(e, RIA_RESOURCE_LIMIT, "RIA engine allocation failed");
  r->outer_owner = outer_owner;
  bool ok = ria_service_read(&r->service, path, e);
  if (ok && (strcmp(r->service.role, "client") ||
             strcmp(r->service.executor, "cuda")))
    ok = ria_fail(e, RIA_UNSUPPORTED,
                  "RIA frontend requires admitted CUDA client role");
  if (ok) {
    r->context = r->service.context_positions;
    if (r->context > 1048576 || !r->context)
      ok = ria_fail(e, RIA_RESOURCE_LIMIT, "invalid admitted source context");
  }
  ria_runtime_observation actual;
  if (ok)
    ok = ria_runtime_require(r->service.host_cap, 0, r->service.pinned_cap,
                             &actual, e);
  if (ok) {
    if (pthread_mutex_init(&r->ownership, NULL) != 0)
      ok = ria_fail(e, RIA_RESOURCE_LIMIT, "engine ownership mutex failed");
    else
      r->mutex_ready = true;
  }
  uint64_t store_cap = 0;
  if (ok) {
    const ria_json_doc *placement = &r->service.placement;
    uint32_t runtime = ria_json_get(placement, 0, "runtime");
    uint64_t tokenizer, host, frontend, projection, patches, pinned, reserved;
    ok = integer(placement, runtime, "tokenizer_memory_bytes", true, &tokenizer,
                 e) &&
         integer(placement, runtime, "host_state_bytes", true, &host, e) &&
         integer(placement, runtime, "frontend_host_bytes", true, &frontend,
                 e) &&
         integer(placement, runtime, "projection_tile_rows", false, &projection,
                 e) &&
         projection <= UINT32_MAX &&
         integer(placement, runtime, "max_image_patches", false, &patches, e) &&
         patches <= UINT32_MAX;
    ria_graph_options sizing = {
        .projection_tile_rows = ok ? (uint32_t)projection : 0,
        .max_image_patches = ok ? (uint32_t)patches : 0};
    if (ok)
      ok = ria_graph_pinned_required_bytes(&sizing, &pinned, e) &&
           pinned <= r->service.pinned_cap &&
           ria_u64_add(tokenizer, host, &reserved) &&
           ria_u64_add(reserved, frontend, &reserved) &&
           ria_u64_add(reserved, pinned, &reserved) &&
           reserved < r->service.host_cap;
    if (ok)
      store_cap = r->service.host_cap - reserved;
    else if (!e || !e->message[0])
      ria_error_set(e, RIA_RESOURCE_LIMIT,
                    "future client allocations leave no model "
                    "metadata/population capacity");
  }
  ria_tensor_load_options load = {.role = "client",
                                  .expected_digest = r->service.manifest_digest,
                                  .max_resident_bytes = store_cap,
                                  .lock_memory = true,
                                  .numa_node = -1,
                                  .max_metadata_bytes = store_cap};
  if (ok)
    ok = ria_tensor_store_open_placed(&r->store, r->service.manifest_path,
                                      &load, client_preflight, r, e) &&
         ((!memcmp(r->store.logical_model_digest,
                   r->service.logical_model_digest, 32) &&
           !memcmp(r->store.operator_contract_digest,
                   r->service.operator_contract_digest, 32) &&
           !strcmp(r->store.profile, r->service.profile)) ||
          ria_fail(e, RIA_IDENTITY_MISMATCH,
                   "client manifest differs from admitted service identity")) &&
         config(r, e) && ria_remote_open(&r->remote, &r->remote_options, e) &&
         ria_graph_create(&r->store, &r->graph_options,
                          ria_remote_callbacks(r->remote), &r->graph, e);
  if (!ok) {
    ria_error cleanup = {0};
    if (!ria_engine_close(r, &cleanup)) {
      fprintf(stderr, "engine construction cleanup failed; retaining owned backing (primary=%d cleanup=%d)\n",
              e ? e->code : 0, cleanup.code);
      _Exit(e && e->code ? e->code : cleanup.code ? cleanup.code : RIA_EXECUTOR_ERROR);
    }
    return false;
  }
  *out = r;
  return true;
}
bool ria_engine_close(ria_engine *r, ria_error *e) {
  if (!r)
    return true;
  /* A failed device drain retains the graph and all borrowed source backing.
   * The serving owner must terminate before running any further cleanup. */
  if (r->fatal)
    return ria_fail(e, RIA_EXECUTOR_ERROR,
                    "graph owner unusable after failed teardown");
  if (!ria_graph_destroy(r->graph, e)) {
    r->fatal = true;
    return false;
  }
  r->graph = NULL;
  bool ok = true;
  if (!ria_remote_close(r->remote, ok ? e : NULL))
    ok = false;
  ria_tokenizer_close(r->tokenizer);
  if (r->runtime_ready)
    ria_tokenizer_runtime_end();
  ria_tensor_store_close(&r->store);
  ria_service_free(&r->service);
  free(r->prefix);
  free(r->last_logits);
  free(r->token_map);
  free(r->local_experts);
  if (r->mutex_ready && pthread_mutex_destroy(&r->ownership) != 0) {
    if (ok)
      ria_error_set(e, RIA_INTERNAL_ERROR,
                    "engine ownership mutex teardown failed");
    ok = false;
  }
  free(r);
  return ok;
}
bool ria_engine_claim(ria_engine *r, uint64_t context, ria_error *e) {
  if (!r || !context || context > r->context)
    return ria_fail(e, RIA_RESOURCE_LIMIT, "session context exceeds admission");
  pthread_mutex_lock(&r->ownership);
  bool ok = !r->claimed && !r->fatal;
  if (ok)
    r->claimed = true;
  pthread_mutex_unlock(&r->ownership);
  return ok || ria_fail(e, RIA_RESOURCE_LIMIT,
                        "RIA engine already has its one active session");
}
void ria_engine_release(ria_engine *r) {
  if (!r)
    return;
  ria_engine_invalidate(r);
  pthread_mutex_lock(&r->ownership);
  r->claimed = false;
  pthread_mutex_unlock(&r->ownership);
}
const ria_service *ria_engine_service(const ria_engine *r) {
  return r ? &r->service : NULL;
}
ria_tokenizer *ria_engine_tokenizer(ria_engine *r) {
  return r ? r->tokenizer : NULL;
}
uint64_t ria_engine_model_bytes(const ria_engine *r) {
  return r ? r->store.resident_bytes : 0;
}
uint64_t ria_engine_context(const ria_engine *r) { return r ? r->context : 0; }
void ria_engine_invalidate(ria_engine *r) {
  if (!r)
    return;
  r->valid = false;
  r->retired = true;
  if (r->remote)
    ria_remote_abort(r->remote, r->generation);
}
static bool reconnect(ria_engine *r, ria_error *e) {
  r->valid = false;
  if (!ria_graph_destroy(r->graph, e)) {
    r->fatal = true;
    fprintf(stderr, "graph reconnect cleanup failed; retaining owned backing (code=%d)\n",e ? e->code : 0);
    _Exit(e && e->code ? e->code : RIA_EXECUTOR_ERROR);
  }
  r->graph = NULL;
  if (!ria_remote_close(r->remote, e)) {
    r->remote = NULL;
    return false;
  }
  r->remote = NULL;
  bool ok = ria_remote_open(&r->remote, &r->remote_options, e) &&
            ria_graph_create(&r->store, &r->graph_options,
                             ria_remote_callbacks(r->remote), &r->graph, e);
  if (ok)
    r->retired = false;
  return ok;
}
bool ria_engine_sync(ria_engine *r, const ds4_tokens *prompt,
                     const ds4_vision_span *images, size_t image_count,
                     float *logits, ds4_session_cancel_fn cancel, void *ud,
                     ria_error *e) {
  if (!r || r->fatal || !r->claimed || !prompt || !prompt->v ||
      prompt->len <= 0 || (uint64_t)prompt->len > r->context || !logits ||
      image_count > 128 || (image_count && !images))
    return ria_fail(e, RIA_INVALID_REQUEST,
                    "invalid incorporated prompt bounds");
  for (int i = 0; i < prompt->len; i++)
    if (prompt->v[i] < 0 || prompt->v[i] >= 129280)
      return ria_fail(e, RIA_INVALID_REQUEST,
                      "prompt token outside source vocabulary");
  r->last_sync_recorded = false;
  bool prefix = r->valid && r->position <= (uint64_t)prompt->len;
  for (uint64_t i = 0; prefix && i < r->position; i++)
    if (prompt->v[i] < 0 || r->prefix[i] != (uint32_t)prompt->v[i])
      prefix = false;
  uint64_t previous_end = 0;
  for (size_t i = 0; i < image_count; i++) {
    const ds4_vision_span *v = &images[i];
    uint64_t end = (uint64_t)v->token_start + v->embedding.token_count;
    if (v->embedding.layout != RIA_IMAGE_LAYOUT || !v->embedding.data ||
        !v->embedding.token_count || v->token_start < previous_end ||
        end > (uint64_t)prompt->len)
      return ria_fail(e, RIA_INVALID_REQUEST, "invalid source image span");
    previous_end = end;
    if ((i >= r->image_count && v->token_start < r->position) ||
        (i < r->image_count &&
         (r->images[i].start != v->token_start ||
          r->images[i].count != v->embedding.token_count ||
          memcmp(r->images[i].digest, v->embedding.fingerprint, 32))))
      prefix = false;
  }
  if (image_count < r->image_count)
    prefix = false;
  if ((r->retired || (!prefix && r->generation)) && !reconnect(r, e))
    return false;
  uint64_t reused_tokens = prefix ? r->position : 0;
  uint64_t epoch, generation;
  if (!ria_remote_begin_generation(r->remote, &epoch, &generation, e)) {
    ria_engine_invalidate(r);
    return false;
  }
  bool ok = prefix ? ria_graph_retag(r->graph, epoch, generation, e)
                   : ria_graph_reset(r->graph, epoch, generation, e);
  if (ok)
    ok = ria_graph_set_phase(
        r->graph, prefix ? RIA_GRAPH_CONTINUATION : RIA_GRAPH_PREFILL, e);
  r->epoch = epoch;
  r->generation = generation;
  if (!prefix)
    r->position = 0;
  for (size_t i = 0; i < image_count; i++) {
    r->images[i].start = images[i].token_start;
    r->images[i].count = images[i].embedding.token_count;
    memcpy(r->images[i].digest, images[i].embedding.fingerprint, 32);
  }
  r->image_count = image_count;
  for (uint64_t i = r->position; ok && i < (uint64_t)prompt->len; i++) {
    if (cancel && cancel(ud)) {
      ok = ria_fail(e, RIA_CANCELLED,
                    "generation cancelled at incorporated position");
      break;
    }
    if (prompt->v[i] < 0 || prompt->v[i] >= 129280) {
      ok = ria_fail(e, RIA_INVALID_REQUEST,
                    "prompt token outside source vocabulary");
      break;
    }
    const float *embedding = NULL;
    for (size_t j = 0; j < image_count; j++)
      if (i >= images[j].token_start &&
          i < (uint64_t)images[j].token_start +
                  images[j].embedding.token_count) {
        embedding =
            images[j].embedding.data + (i - images[j].token_start) * 5120;
        break;
      }
    ok = ria_graph_step(r->graph, (uint32_t)prompt->v[i], embedding != NULL,
                        embedding, logits, e);
    if (ok) {
      memcpy(r->last_logits, logits, 129280 * sizeof(float));
      r->prefix[i] = (uint32_t)prompt->v[i];
      r->position = i + 1;
    }
  }
  if (!ok) {
    ria_engine_invalidate(r);
    return false;
  }
  r->valid = true;
  r->last_sync_recorded = true;
  r->last_continuation = prefix;
  r->last_reused_tokens = reused_tokens;
  memcpy(logits, r->last_logits, 129280 * sizeof(float));
  return true;
}
bool ria_engine_sync_observation(const ria_engine *r, const char **phase,
                                 uint64_t *reused_tokens, ria_error *e) {
  if (!r || r->fatal || !r->claimed || !r->last_sync_recorded || !phase ||
      !reused_tokens)
    return ria_fail(e, RIA_NOT_READY,
                    "no successful sync observation is available");
  *phase = r->last_continuation ? "continuation" : "prefill";
  *reused_tokens = r->last_reused_tokens;
  return true;
}
bool ria_api_measurements(const ria_json_doc *doc, bool streaming,
                          bool *enabled, ria_error *e) {
  if (!doc || !enabled)
    return ria_fail(e, RIA_INVALID_REQUEST, "invalid measurement option owner");
  const ria_json_node *n =
      ria_json_at(doc, ria_json_get(doc, 0, "ria_measurements"));
  *enabled = false;
  if (!n)
    return true;
  if (n->type != RIA_JSON_BOOL || (n->boolean && !streaming))
    return ria_fail(
        e, RIA_INVALID_REQUEST,
        "ria_measurements requires a Boolean and streaming when enabled");
  *enabled = n->boolean;
  return true;
}
bool ria_engine_eval(ria_engine *r, uint32_t token, float *logits,
                     ria_error *e) {
  if (!r || r->fatal || !r->valid || !logits || token >= 129280 ||
      r->position >= r->context)
    return ria_fail(e, RIA_NOT_READY,
                    "session requires valid admitted incorporated prefix");
  if (!ria_graph_set_phase(r->graph, RIA_GRAPH_DECODE, e) ||
      !ria_graph_step(r->graph, token, false, NULL, logits, e)) {
    ria_engine_invalidate(r);
    return false;
  }
  memcpy(r->last_logits, logits, 129280 * sizeof(float));
  r->prefix[r->position++] = token;
  return true;
}
bool ria_engine_rewind(ria_engine *r, uint64_t position, ria_error *e) {
  if (!r || position > r->position)
    return ria_fail(e, RIA_INVALID_REQUEST, "invalid rewind position");
  ria_engine_invalidate(r);
  r->position = position;
  return true;
}
bool ria_engine_image(ria_engine *r, const uint8_t *encoded, size_t length,
                      uint64_t workspace_budget, ds4_vision_embedding *out,
                      ria_error *e) {
  if (!r || r->fatal || !out || !r->graph ||
      !r->graph_options.max_image_patches)
    return ria_fail(e, RIA_UNSUPPORTED, "image path is not admitted");
  if (workspace_budget > r->frontend - r->frontend_fixed)
    return ria_fail(e, RIA_RESOURCE_LIMIT,
                    "image workspace exceeds remaining frontend reservation");
  ria_image_input input = {0};
  if (!ria_image_prepare(encoded, length, r->graph_options.max_image_patches,
                         workspace_budget, &input, e))
    return false;
  uint64_t natural = (uint64_t)input.grid.llm_height * input.grid.llm_width,
           bytes, all_bytes;
  if (!ria_u64_mul(natural, 5120 * sizeof(float), &bytes) ||
      !ria_u64_mul(input.grid.token_count, 5120 * sizeof(float), &all_bytes) ||
      bytes > SIZE_MAX || all_bytes > SIZE_MAX ||
      input.host_bytes > workspace_budget ||
      bytes > workspace_budget - input.host_bytes ||
      all_bytes > workspace_budget - input.host_bytes - bytes) {
    ria_image_input_free(&input);
    return ria_fail(e, RIA_RESOURCE_LIMIT,
                    "image expansion exceeds admitted host workspace");
  }
  float *features = malloc((size_t)bytes), *all = malloc((size_t)all_bytes);
  if (!features || !all) {
    free(features);
    free(all);
    ria_image_input_free(&input);
    return ria_fail(e, RIA_RESOURCE_LIMIT, "image embedding allocation failed");
  }
  bool ok =
      ria_graph_encode_image(r->graph, input.patches, input.grid.vit_height,
                             input.grid.vit_width, features, natural, e);
  uint64_t j = 0;
  for (uint32_t i = 0; ok && i < input.grid.token_count; i++) {
    if (input.types[i] == 1) {
      if (j >= natural) {
        ok = ria_fail(e, RIA_INTERNAL_ERROR, "source image layout mismatch");
        break;
      }
      memcpy(all + (uint64_t)i * 5120, features + j++ * 5120,
             5120 * sizeof(float));
    } else
      ok = ria_graph_image_delimiter(r->graph, input.types[i],
                                     all + (uint64_t)i * 5120, e);
  }
  if (ok) {
    memset(out, 0, sizeof *out);
    out->data = all;
    out->token_count = input.grid.token_count;
    out->layout = RIA_IMAGE_LAYOUT;
    out->grid_width = input.grid.llm_width;
    out->grid_height = input.grid.llm_height;
    out->width = input.grid.pixel_width;
    out->height = input.grid.pixel_height;
    ok = ria_sha256(encoded, length, out->fingerprint, e);
  }
  free(features);
  ria_image_input_free(&input);
  if (!ok) {
    free(all);
    memset(out, 0, sizeof *out);
    ria_engine_invalidate(r);
  }
  return ok;
}

uint64_t ria_engine_frontend_budget(const ria_engine *r) {
  return r ? r->frontend - r->frontend_fixed : 0;
}
bool ria_api_header(const char *bytes, size_t length, const char *bearer,
                    size_t bearer_length, uint64_t max_body,
                    ria_http_header *out, ria_error *e) {
  if (!bytes || !out || !bearer || !bearer_length || bearer_length > 4096 ||
      length < 4 || length > 65536 || memchr(bytes, 0, length) ||
      memcmp(bytes + length - 4, "\r\n\r\n", 4))
    return ria_fail(e, RIA_INVALID_REQUEST, "invalid HTTP header framing");
  memset(out, 0, sizeof *out);
  const char *at = bytes, *end = bytes + length,
             *line_end = memchr(at, '\r', length);
  if (!line_end || line_end + 1 >= end || line_end[1] != '\n')
    return ria_fail(e, RIA_INVALID_REQUEST, "invalid HTTP request line");
  const char *space = memchr(at, ' ', (size_t)(line_end - at));
  if (!space || (size_t)(space - at) >= sizeof out->method)
    return ria_fail(e, RIA_INVALID_REQUEST, "invalid HTTP method");
  memcpy(out->method, at, (size_t)(space - at));
  if (strcmp(out->method, "GET") && strcmp(out->method, "POST"))
    return ria_fail(e, RIA_UNSUPPORTED, "HTTP method unsupported");
  at = space + 1;
  space = memchr(at, ' ', (size_t)(line_end - at));
  if (!space || space == at || (size_t)(space - at) >= sizeof out->path ||
      line_end - space != 9 || memcmp(space + 1, "HTTP/1.1", 8) || at[0] != '/')
    return ria_fail(e, RIA_INVALID_REQUEST, "invalid HTTP target/version");
  memcpy(out->path, at, (size_t)(space - at));
  for (const char *p = at; p < space; p++)
    if ((unsigned char)*p <= 32 || (unsigned char)*p >= 127 || *p == '#')
      return ria_fail(e, RIA_INVALID_REQUEST, "invalid HTTP target bytes");
  at = line_end + 2;
  bool auth = false, host = false, content = false, json_type = false;
  while (at < end - 2) {
    line_end = memchr(at, '\r', (size_t)(end - at));
    if (!line_end || line_end + 1 >= end || line_end[1] != '\n' ||
        at[0] == ' ' || at[0] == '\t')
      return ria_fail(e, RIA_INVALID_REQUEST, "invalid HTTP header line");
    const char *colon = memchr(at, ':', (size_t)(line_end - at));
    if (!colon || colon == at)
      return ria_fail(e, RIA_INVALID_REQUEST, "invalid HTTP header name");
    for (const char *p = at; p < colon; p++) {
      unsigned char c = (unsigned char)*p;
      if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
            (c >= '0' && c <= '9') || strchr("!#$%&'*+-.^_`|~", c)))
        return ria_fail(e, RIA_INVALID_REQUEST, "invalid HTTP header name");
    }
    const char *v = colon + 1, *vend = line_end;
    while (v < vend && (*v == ' ' || *v == '\t'))
      v++;
    while (vend > v && (vend[-1] == ' ' || vend[-1] == '\t'))
      vend--;
    for (const char *p = v; p < vend; p++)
      if (((unsigned char)*p < 32 && *p != '\t') || (unsigned char)*p == 127)
        return ria_fail(e, RIA_INVALID_REQUEST, "invalid HTTP header value");
    size_t key = (size_t)(colon - at), n = (size_t)(vend - v);
    if (key == 13 && !strncasecmp(at, "Authorization", 13)) {
      if (auth || n != 7 + bearer_length || memcmp(v, "Bearer ", 7) ||
          CRYPTO_memcmp(v + 7, bearer, bearer_length))
        return ria_fail(e, RIA_UNAUTHORIZED, "bearer authorization required");
      auth = true;
    } else if (key == 4 && !strncasecmp(at, "Host", 4)) {
      if (host || !n)
        return ria_fail(e, RIA_INVALID_REQUEST, "exactly one Host required");
      host = true;
    } else if (key == 14 && !strncasecmp(at, "Content-Length", 14)) {
      if (content || !n || n > 20)
        return ria_fail(e, RIA_INVALID_REQUEST, "invalid Content-Length");
      char number[21];
      memcpy(number, v, n);
      number[n] = 0;
      if (!ria_parse_u64(number, n, &out->body_bytes, e))
        return false;
      content = true;
    } else if (key == 17 && !strncasecmp(at, "Transfer-Encoding", 17))
      return ria_fail(e, RIA_UNSUPPORTED, "HTTP transfer encoding unsupported");
    else if (key == 12 && !strncasecmp(at, "Content-Type", 12)) {
      if (json_type || n != 16 || strncasecmp(v, "application/json", 16))
        return ria_fail(e, RIA_UNSUPPORTED,
                        "Content-Type must be application/json");
      json_type = true;
    }
    at = line_end + 2;
  }
  if (!auth)
    return ria_fail(e, RIA_UNAUTHORIZED, "bearer authorization required");
  if (!host || out->body_bytes > max_body ||
      (!strcmp(out->method, "POST") &&
       (!content || !out->body_bytes || !json_type)) ||
      (!strcmp(out->method, "GET") && out->body_bytes))
    return ria_fail(e, RIA_INVALID_REQUEST, "invalid HTTP body contract");
  return true;
}
