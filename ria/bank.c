#include "bank.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool matrix(const ria_tensor_store *s, uint32_t layer, uint32_t expert,
                   bool shared, const char *projection, ria_expert_matrix *m,
                   ria_error *e) {
  char name[160];
  int n = shared ? snprintf(name, sizeof(name),
                            "layers.%u.ffn.shared_experts.%s.weight", layer,
                            projection)
                 : snprintf(name, sizeof(name),
                            "layers.%u.ffn.experts.%u.%s.weight", layer, expert,
                            projection);
  if (n < 0 || (size_t)n >= sizeof(name))
    return ria_fail(e, RIA_INTERNAL_ERROR, "expert name exceeds fixed schema");
  const ria_tensor *t = ria_tensor_name(s, name);
  if (!t)
    return ria_fail(e, RIA_INTEGRITY_ERROR,
                    "required expert matrix is absent: %s", name);
  return ria_tensor_matrix(s, t, m, e);
}
static bool expert(const ria_tensor_store *s, uint32_t layer, uint32_t id,
                   bool shared, ria_expert *x, ria_error *e) {
  x->clamp = 10;
  if (!matrix(s, layer, id, shared, "w1", &x->gate, e) ||
      !matrix(s, layer, id, shared, "w3", &x->up, e) ||
      !matrix(s, layer, id, shared, "w2", &x->down, e))
    return false;
  if (x->gate.in_features != RIA_WIDTH ||
      x->gate.out_features != RIA_INTERMEDIATE ||
      x->up.in_features != RIA_WIDTH ||
      x->up.out_features != RIA_INTERMEDIATE ||
      x->down.in_features != RIA_INTERMEDIATE ||
      x->down.out_features != RIA_WIDTH)
    return ria_fail(e, RIA_INTEGRITY_ERROR,
                    "expert differs from pinned V4.1 Flash dimensions");
  return true;
}
bool ria_bank_open(ria_bank *b, const ria_tensor_store *s, ria_error *e) {
  memset(b, 0, sizeof(*b));
  b->store = s;
  if (strcmp(s->role, "server"))
    return ria_fail(e, RIA_INVALID_REQUEST,
                    "expert bank requires server package");
  b->experts = calloc((size_t)RIA_LAYERS * RIA_EXPERTS, sizeof(*b->experts));
  if (!b->experts)
    return ria_fail(e, RIA_RESOURCE_LIMIT,
                    "expert descriptor allocation failed");
  for (uint32_t l = 0; l < RIA_LAYERS; l++) {
    b->operations[l] = (ria_operation){.handle = l + 1,
                                       .input_width = RIA_WIDTH,
                                       .output_width = RIA_WIDTH,
                                       .max_rows = 64,
                                       .expert_count = RIA_EXPERTS,
                                       .max_selected = 6,
                                       .coefficient_min = 0,
                                       .coefficient_max = 1.5f,
                                       .input_bf16 = true,
                                       .output_bf16 = true};
    ria_expert_profile expected =
        !strcmp(s->profile, "nvfp4") ? RIA_EXPERT_NVFP4
        : !strcmp(s->profile, "fp8") ? RIA_EXPERT_FP8
                                     : RIA_EXPERT_BF16;
    for (uint32_t id = 0; id < RIA_EXPERTS; id++) {
      ria_expert *x = &b->experts[(size_t)l * RIA_EXPERTS + id];
      if (!expert(s, l, id, false, x, e) || x->gate.profile != expected ||
          x->up.profile != expected || x->down.profile != expected) {
        if (!e || !e->code)
          ria_error_set(e, RIA_IDENTITY_MISMATCH,
                        "routed matrix profile mismatch");
        goto fail;
      }
    }
    char name[160];
    snprintf(name, sizeof(name), "layers.%u.ffn.shared_experts.w1.weight", l);
    if (ria_tensor_name(s, name)) {
      if (!expert(s, l, 0, true, &b->shared[l], e))
        goto fail;
      b->shared_present[l] = true;
      b->operations[RIA_LAYERS + l] =
          (ria_operation){.handle = RIA_LAYERS + l + 1,
                          .input_width = RIA_WIDTH,
                          .output_width = RIA_WIDTH,
                          .max_rows = 64,
                          .expert_count = 1,
                          .max_selected = 1,
                          .coefficient_min = 1,
                          .coefficient_max = 1,
                          .shared = true,
                          .input_bf16 = true,
                          .output_bf16 = true};
    }
  }
  for (uint32_t i = 0; i < 2; i++) {
    char name[160];
    snprintf(name, sizeof(name), "layers.%u.engram.embed.weight", (unsigned)(i ? 14 : 1));
    const ria_tensor *t = ria_tensor_name(s, name);
    if (!t || strcmp(t->format, "engram_packed") || t->rank != 2 ||
        t->shape[1] != 256 ||
        t->shape[0] != (i ? UINT64_C(384016682) : UINT64_C(384006168)) ||
        t->length != t->shape[0] * 264) {
      ria_error_set(e, RIA_INTEGRITY_ERROR,
                    "Engram lossless packed table missing/invalid");
      goto fail;
    }
    b->engram[i] = t;
    b->tables[i] = (ria_table){.handle = i + 1,
                               .row_count = t->shape[0],
                               .packed_row_stride = 264,
                               .representation_handle = 1};
  }
  return true;
fail:
  ria_bank_close(b);
  return false;
}
void ria_bank_close(ria_bank *b) {
  free(b->experts);
  memset(b, 0, sizeof(*b));
}
const ria_operation *ria_bank_operation(const ria_bank *b, uint64_t h) {
  if (!h || h > RIA_LAYERS * 2 ||
      (h > RIA_LAYERS && !b->shared_present[h - RIA_LAYERS - 1]))
    return NULL;
  return &b->operations[h - 1];
}
const ria_expert *ria_bank_expert(const ria_bank *b, uint64_t h, uint16_t id) {
  if (!h || h > RIA_LAYERS * 2)
    return NULL;
  if (h > RIA_LAYERS)
    return id == 0 && b->shared_present[h - RIA_LAYERS - 1]
               ? &b->shared[h - RIA_LAYERS - 1]
               : NULL;
  return id < RIA_EXPERTS ? &b->experts[(h - 1) * RIA_EXPERTS + id] : NULL;
}
const ria_table *ria_bank_table(const ria_bank *b, uint64_t h) {
  return h == 1 || h == 2 ? &b->tables[h - 1] : NULL;
}
bool ria_bank_rows(const ria_bank *b, const ria_row_request *r, uint8_t *out,
                   size_t length, ria_error *e) {
  const ria_table *t = ria_bank_table(b, r->handle);
  if (!t || length != r->response_bytes)
    return ria_fail(e, RIA_INVALID_REQUEST, "invalid row response allocation");
  memset(out, 0, 24);
  ria_write_u64(out, r->handle);
  ria_write_u32(out + 8, r->row_count);
  ria_write_u32(out + 12, (uint32_t)t->packed_row_stride);
  ria_write_u64(out + 16, t->representation_handle);
  memcpy(out + 24, r->pairs, (size_t)r->row_count * 16);
  uint8_t *data = out + 24 + (size_t)r->row_count * 16;
  for (uint32_t i = 0; i < r->row_count; i++) {
    uint64_t id = ria_read_u64(r->pairs + (size_t)i * 16);
    if (id >= t->row_count)
      return ria_fail(e, RIA_INVALID_REQUEST,
                      "row index exceeds immutable table");
    memcpy(data + (size_t)i * 264, b->engram[r->handle - 1]->data + id * 264,
           264);
  }
  return true;
}
