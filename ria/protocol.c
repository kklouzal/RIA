#include "protocol.h"
#include <math.h>
#include <openssl/crypto.h>
#include <openssl/rand.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static bool known(uint16_t k)
{
    return k == 1 || k == 2 || k == 10 || k == 11 || k == 12 || k == 13 || k == 20 || k == 21 ||
           k == 22;
}
static bool zero(const uint8_t *p, size_t n)
{
    uint8_t v = 0;
    for (size_t i = 0; i < n; i++)
        v |= p[i];
    return v == 0;
}
bool ria_limits_validate(const ria_limits *l, ria_error *e)
{
    if (!l || !l->frame_payload_bytes || l->frame_payload_bytes > RIA_FRAME_MAX ||
        !l->bulk_data_bytes || l->bulk_data_bytes > RIA_BULK_MAX ||
        l->bulk_data_bytes > l->frame_payload_bytes - ((l->frame_payload_bytes >= 64) ? 64 : 0) ||
        l->frame_payload_bytes < 64 || !l->expert_rows || l->expert_rows > 64 ||
        !l->expert_requests || l->expert_requests > 2 || !l->row_lookup_rows ||
        l->row_lookup_rows > UINT32_MAX || !l->inflight_payload_bytes || !l->operation_timeout_ms ||
        !l->frame_io_timeout_ms || !l->write_timeout_ms)
        return ria_fail(e, RIA_INVALID_REQUEST, "invalid negotiated limits");
    uint64_t values[] = {l->frame_payload_bytes,  l->bulk_data_bytes,     l->expert_rows,
                         l->expert_requests,      l->row_lookup_rows,     l->inflight_payload_bytes,
                         l->operation_timeout_ms, l->frame_io_timeout_ms, l->write_timeout_ms};
    for (size_t i = 0; i < 9; i++)
        if (values[i] > RIA_JSON_SAFE_INTEGER)
            return ria_fail(e, RIA_INVALID_REQUEST, "limit exceeds safe integer");
    return true;
}
bool ria_header_decode(const uint8_t b[64], uint64_t limit, ria_header *h, ria_error *e)
{
    if (!b || !h || memcmp(b, "DSER", 4) || ria_read_u16(b + 4) != 1 || !zero(b + 56, 8))
        return ria_fail(e, RIA_INVALID_REQUEST, "invalid wire header");
    ria_header v = {0};
    v.kind = ria_read_u16(b + 6);
    v.flags = ria_read_u32(b + 8);
    v.status = ria_read_u32(b + 12);
    v.payload_length = ria_read_u64(b + 16);
    v.request_id = ria_read_u64(b + 24);
    memcpy(v.session, b + 32, 16);
    v.epoch = ria_read_u64(b + 48);
    if (!known(v.kind) || (v.flags & ~UINT32_C(1)) || v.status > 11 ||
        (!(v.flags & 1) && v.status) || !v.request_id || !limit || limit > RIA_FRAME_MAX ||
        v.payload_length > limit)
        return ria_fail(e, RIA_INVALID_REQUEST, "invalid wire fields");
    uint64_t type_limit = limit;
    if (v.status)
        type_limit = RIA_ERROR_MAX;
    else if (v.kind == RIA_BIND || v.kind == RIA_BIND_BULK)
        type_limit = RIA_CONTROL_MAX;
    else if (v.kind == RIA_HEALTH)
        type_limit = RIA_ERROR_MAX;
    else if (v.kind == RIA_CHUNK)
        type_limit = (v.flags & 1) ? RIA_BULK_MAX + 64 : 16;
    if (v.payload_length > type_limit)
        return ria_fail(e, RIA_RESOURCE_LIMIT, "message exceeds type-specific limit");
    if (!v.status && ((v.kind == RIA_CANCEL && v.payload_length != ((v.flags & 1) ? 24u : 16u)) ||
                      (v.kind == RIA_CLOSE && v.payload_length) ||
                      (v.kind == RIA_CHUNK && !(v.flags & 1) && v.payload_length != 16)))
        return ria_fail(e, RIA_INVALID_REQUEST, "invalid fixed payload length");
    *h = v;
    return true;
}
bool ria_header_encode(const ria_header *h, uint8_t b[64], ria_error *e)
{
    if (!h || !b)
        return ria_fail(e, RIA_INVALID_REQUEST, "null header");
    memset(b, 0, 64);
    memcpy(b, "DSER", 4);
    ria_write_u16(b + 4, 1);
    ria_write_u16(b + 6, h->kind);
    ria_write_u32(b + 8, h->flags);
    ria_write_u32(b + 12, h->status);
    ria_write_u64(b + 16, h->payload_length);
    ria_write_u64(b + 24, h->request_id);
    memcpy(b + 32, h->session, 16);
    ria_write_u64(b + 48, h->epoch);
    ria_header check;
    return ria_header_decode(b, RIA_FRAME_MAX, &check, e);
}
bool ria_expert_lengths(uint64_t r, uint64_t n, uint64_t d, uint64_t o, uint64_t q,
                        uint64_t *request, uint64_t *response, ria_error *e)
{
    if (!request || !response)
        return ria_fail(e, RIA_INVALID_REQUEST, "missing expert length output");
    uint64_t size = 40, a, b;
    bool ok = ria_u64_mul(r, 8, &a) && ria_u64_add(size, a, &size) && ria_u64_add(r, 1, &a) &&
              ria_u64_mul(a, 4, &a) && ria_u64_add(size, a, &size) && ria_u64_mul(n, 8, &a) &&
              ria_u64_add(size, a, &size) && ria_u64_mul(r, d, &a) && ria_u64_mul(a, 4, &a) &&
              ria_u64_add(size, a, &size) && ria_u64_add(size, q, &size) && ria_u64_mul(n, o, &b) &&
              ria_u64_mul(b, 4, &b) && ria_u64_add(b, 32, &b);
    if (!ok)
        return ria_fail(e, RIA_RESOURCE_LIMIT, "expert payload arithmetic overflow");
    *request = size;
    *response = b;
    return true;
}
static bool finite_values(const uint8_t *values, uint64_t count, bool bf16, ria_error *e)
{
    for (uint64_t i = 0; i < count; i++) {
        const uint8_t *p = values + i * 4;
        if (!isfinite(ria_read_f32(p)) || (bf16 && (ria_read_u32(p) & 65535)))
            return ria_fail(e, RIA_INVALID_REQUEST,
                            "nonfinite or incorrectly rounded floating payload");
    }
    return true;
}
bool ria_expert_parse(const uint8_t *p, size_t length, uint16_t kind, const ria_operation *op,
                      const ria_limits *limits, ria_expert_request *out, ria_error *e)
{
    if (!p || length < 40 || !op || !limits || !out || (kind != RIA_EXPERT && kind != RIA_SHARED) ||
        op->shared != (kind == RIA_SHARED))
        return ria_fail(e, RIA_INVALID_REQUEST, "invalid expert operation");
    if (op->quantizer_context_bytes)
        return ria_fail(e, RIA_UNSUPPORTED, "revision1 has no declared external statistic layout");
    ria_expert_request v = {0};
    v.operation_handle = ria_read_u64(p);
    v.invocation_id = ria_read_u64(p + 8);
    v.row_count = ria_read_u32(p + 16);
    v.input_width = ria_read_u32(p + 20);
    v.output_width = ria_read_u32(p + 24);
    v.entry_count = ria_read_u32(p + 28);
    v.quantizer_context_bytes = ria_read_u32(p + 32);
    if (ria_read_u32(p + 36) || v.operation_handle != op->handle || !v.invocation_id ||
        !v.row_count || v.row_count > op->max_rows || v.row_count > limits->expert_rows ||
        v.input_width != op->input_width || v.output_width != op->output_width || !v.input_width ||
        !v.output_width || !v.entry_count ||
        v.quantizer_context_bytes != op->quantizer_context_bytes || !op->max_selected ||
        op->max_selected > 6 || !op->expert_count || !isfinite(op->coefficient_min) ||
        !isfinite(op->coefficient_max) || op->coefficient_min > op->coefficient_max)
        return ria_fail(e, RIA_INVALID_REQUEST, "expert descriptor disagrees with bound contract");
    uint64_t request_bytes;
    if (!ria_expert_lengths(v.row_count, v.entry_count, v.input_width, v.output_width,
                            v.quantizer_context_bytes, &request_bytes, &v.response_bytes, e))
        return false;
    if (request_bytes != length || request_bytes > limits->frame_payload_bytes ||
        v.response_bytes > limits->frame_payload_bytes)
        return ria_fail(e, RIA_RESOURCE_LIMIT, "expert exact length or response bound violated");
    v.row_ids = p + 40;
    v.offsets = v.row_ids + (size_t)v.row_count * 8;
    v.entries = v.offsets + ((size_t)v.row_count + 1) * 4;
    v.inputs = v.entries + (size_t)v.entry_count * 8;
    v.quantizer_context = v.inputs + (size_t)v.row_count * v.input_width * 4;
    if (ria_read_u32(v.offsets) != 0 || ria_read_u32(v.offsets + v.row_count * 4) != v.entry_count)
        return ria_fail(e, RIA_INVALID_REQUEST, "invalid CSR endpoints");
    for (uint32_t row = 0; row < v.row_count; row++) {
        for (uint32_t j = 0; j < row; j++)
            if (ria_read_u64(v.row_ids + row * 8) == ria_read_u64(v.row_ids + j * 8))
                return ria_fail(e, RIA_INVALID_REQUEST, "duplicate invocation row");
        uint32_t begin = ria_read_u32(v.offsets + row * 4),
                 end = ria_read_u32(v.offsets + (row + 1) * 4);
        if (begin > end || end > v.entry_count || end - begin > op->max_selected ||
            (op->shared && end - begin != 1))
            return ria_fail(e, RIA_INVALID_REQUEST, "invalid CSR row");
        for (uint32_t j = begin; j < end; j++) {
            const uint8_t *entry = v.entries + (size_t)j * 8;
            uint16_t expert = ria_read_u16(entry), slot = ria_read_u16(entry + 2);
            float coefficient = ria_read_f32(entry + 4);
            if (!isfinite(coefficient) || coefficient < op->coefficient_min ||
                coefficient > op->coefficient_max)
                return ria_fail(e, RIA_INVALID_REQUEST, "invalid routing coefficient");
            if (op->shared) {
                if (expert || slot != UINT16_MAX || coefficient != 1.0f)
                    return ria_fail(e, RIA_INVALID_REQUEST, "invalid shared branch entry");
            } else if (expert >= op->expert_count || slot >= op->max_selected)
                return ria_fail(e, RIA_INVALID_REQUEST, "expert or selected slot out of range");
            for (uint32_t k = begin; k < j; k++) {
                const uint8_t *old = v.entries + (size_t)k * 8;
                if (ria_read_u16(old) == expert || ria_read_u16(old + 2) == slot)
                    return ria_fail(e, RIA_INVALID_REQUEST, "duplicate expert or selected slot");
            }
        }
    }
    if (!finite_values(v.inputs, (uint64_t)v.row_count * v.input_width, op->input_bf16, e))
        return false;
    *out = v;
    return true;
}
bool ria_expert_response(const uint8_t *p, size_t length, const ria_expert_request *r,
                         const ria_operation *op, ria_error *e)
{
    if (!p || !r || !op || length != r->response_bytes || length < 32 ||
        ria_read_u64(p) != r->operation_handle || ria_read_u64(p + 8) != r->invocation_id ||
        ria_read_u32(p + 16) != r->row_count || ria_read_u32(p + 20) != r->output_width ||
        ria_read_u32(p + 24) != r->entry_count || ria_read_u32(p + 28))
        return ria_fail(e, RIA_IDENTITY_MISMATCH, "expert reply association or length mismatch");
    return finite_values(p + 32, (uint64_t)r->entry_count * r->output_width, op->output_bf16, e);
}
bool ria_rows_parse(const uint8_t *p, size_t length, const ria_table *t, const ria_limits *l,
                    ria_row_request *out, ria_error *e)
{
    if (!p || length < 16 || !t || !l || !out)
        return ria_fail(e, RIA_INVALID_REQUEST, "invalid row request");
    ria_row_request v = {0};
    v.handle = ria_read_u64(p);
    v.row_count = ria_read_u32(p + 8);
    uint64_t pairs, size, data;
    if (v.handle != t->handle || ria_read_u32(p + 12) || !v.row_count ||
        v.row_count > l->row_lookup_rows || !t->packed_row_stride ||
        t->packed_row_stride > UINT32_MAX || !t->representation_handle ||
        !ria_u64_mul(v.row_count, 16, &pairs) || !ria_u64_add(pairs, 16, &size) || size != length ||
        !ria_u64_mul(v.row_count, t->packed_row_stride, &data) ||
        !ria_u64_add(pairs, data, &size) || !ria_u64_add(size, 24, &v.response_bytes) ||
        v.response_bytes > l->frame_payload_bytes)
        return ria_fail(e, RIA_RESOURCE_LIMIT, "row descriptor or exact length invalid");
    v.pairs = p + 16;
    for (uint32_t i = 0; i < v.row_count; i++)
        if (ria_read_u64(v.pairs + (size_t)i * 16) >= t->row_count)
            return ria_fail(e, RIA_INVALID_REQUEST, "row outside bound table");
    *out = v;
    return true;
}
bool ria_rows_response(const uint8_t *p, size_t length, const ria_row_request *r,
                       const ria_table *t, ria_error *e)
{
    uint64_t pairs;
    if (!p || !r || !t || length != r->response_bytes || length < 24 ||
        ria_read_u64(p) != r->handle || ria_read_u32(p + 8) != r->row_count ||
        ria_read_u32(p + 12) != t->packed_row_stride ||
        ria_read_u64(p + 16) != t->representation_handle ||
        !ria_u64_mul(r->row_count, 16, &pairs) || memcmp(p + 24, r->pairs, (size_t)pairs))
        return ria_fail(e, RIA_IDENTITY_MISMATCH, "row response association mismatch");
    return true;
}
bool ria_chunk_request(const uint8_t *p, size_t length, uint64_t *handle, uint64_t *index,
                       ria_error *e)
{
    if (!p || length != 16 || !handle || !index || !ria_read_u64(p))
        return ria_fail(e, RIA_INVALID_REQUEST, "invalid chunk request");
    *handle = ria_read_u64(p);
    *index = ria_read_u64(p + 8);
    return true;
}
bool ria_chunk_response(const uint8_t *p, size_t length, uint64_t handle, uint64_t index,
                        uint64_t shard, uint32_t chunk, const uint8_t expected[32],
                        const uint8_t **data, size_t *data_length, ria_error *e)
{
    uint64_t offset;
    if (!p || length < 64 || !expected || !data || !data_length || !chunk || chunk > RIA_BULK_MAX ||
        !ria_u64_mul(index, chunk, &offset) || offset >= shard)
        return ria_fail(e, RIA_INVALID_REQUEST, "invalid chunk range");
    uint64_t n = shard - offset;
    if (n > chunk)
        n = chunk;
    if (ria_read_u64(p) != handle || ria_read_u64(p + 8) != index ||
        ria_read_u64(p + 16) != offset || ria_read_u32(p + 24) != n || ria_read_u32(p + 28) ||
        n != length - 64 || CRYPTO_memcmp(p + 32, expected, 32))
        return ria_fail(e, RIA_INTEGRITY_ERROR, "chunk descriptor or expected identity mismatch");
    uint8_t actual[32];
    if (!ria_sha256(p + 64, (size_t)n, actual, e))
        return false;
    if (CRYPTO_memcmp(actual, expected, 32))
        return ria_fail(e, RIA_INTEGRITY_ERROR, "chunk data digest mismatch");
    *data = p + 64;
    *data_length = (size_t)n;
    return true;
}
bool ria_cancel_parse(const uint8_t *p, size_t length, uint64_t id, uint64_t epoch,
                      uint64_t *target, ria_error *e)
{
    if (!p || length != 16 || !target || !ria_read_u64(p) || ria_read_u64(p) >= id ||
        ria_read_u64(p + 8) != epoch)
        return ria_fail(e, RIA_INVALID_REQUEST, "invalid cancellation target");
    *target = ria_read_u64(p);
    return true;
}
bool ria_cancel_response(const uint8_t *p, size_t length, uint64_t target, uint64_t epoch,
                         uint32_t *state, ria_error *e)
{
    if (!p || length != 24 || !state || ria_read_u64(p) != target || ria_read_u64(p + 8) != epoch ||
        ria_read_u32(p + 16) > 2 || ria_read_u32(p + 20))
        return ria_fail(e, RIA_INVALID_REQUEST, "invalid cancel response");
    *state = ria_read_u32(p + 16);
    return true;
}
bool ria_control_json(const uint8_t *p, size_t length, const ria_header *h, ria_json_doc *doc,
                      ria_error *e)
{
    if (!h || !doc || length != h->payload_length ||
        length > (h->status ? RIA_ERROR_MAX
                            : (h->kind == RIA_BIND || h->kind == RIA_BIND_BULK ? RIA_CONTROL_MAX
                                                                               : RIA_ERROR_MAX)))
        return ria_fail(e, RIA_INVALID_REQUEST, "invalid control JSON length");
    ria_json_limits bounds = {length, 8192, 32};
    if (!ria_json_parse(p, length, bounds, doc, e))
        return false;
    bool ok = false;
    if (h->status) {
        static const char *const fields[] = {"code", "message"};
        uint64_t code;
        const char *message;
        size_t n;
        ok = ria_json_fields(doc, 0, fields, 2, fields, 2, e) &&
             ria_json_u64(doc, ria_json_get(doc, 0, "code"), false, &code, e) &&
             code == h->status &&
             ria_json_string(doc, ria_json_get(doc, 0, "message"), &message, &n, e) && n <= 4096;
        (void)message;
        if (!ok)
            ria_error_set(e, RIA_INVALID_REQUEST, "invalid typed error JSON");
    } else if (h->kind == RIA_BIND) {
        ria_limits limits;
        ok = ria_bind_validate(doc, h->flags == 1, &limits, e);
        if (ok && h->flags == 1) {
            const char *text = NULL;
            size_t n = 0;
            uint8_t session[16];
            uint64_t epoch = 0;
            ok = h->request_id == 1 &&
                 ria_json_string(doc, ria_json_get(doc, 0, "session_id"), &text, &n, e) &&
                 ria_hex_decode(text, n, session, 16, e) && !memcmp(session, h->session, 16) &&
                 ria_json_u64(doc, ria_json_get(doc, 0, "epoch"), true, &epoch, e) &&
                 epoch == h->epoch;
        }
    } else if (h->kind == RIA_BIND_BULK) {
        if (h->flags == 1) {
            static const char *const fields[] = {"bound"};
            const ria_json_node *n = ria_json_at(doc, ria_json_get(doc, 0, "bound"));
            ok = ria_json_fields(doc, 0, fields, 1, fields, 1, e) && n &&
                 n->type == RIA_JSON_BOOL && n->boolean;
        } else {
            static const char *const fields[] = {"session_id", "epoch", "logical_model_digest",
                                                 "operator_contract_digest", "bulk_capability"};
            const char *s;
            size_t n;
            uint8_t sid[16], digest[32];
            uint64_t epoch;
            ok = ria_json_fields(doc, 0, fields, 5, fields, 5, e) &&
                 ria_json_string(doc, ria_json_get(doc, 0, "session_id"), &s, &n, e) &&
                 ria_hex_decode(s, n, sid, 16, e) && !memcmp(sid, h->session, 16) &&
                 ria_json_u64(doc, ria_json_get(doc, 0, "epoch"), true, &epoch, e) &&
                 epoch == h->epoch &&
                 ria_json_digest_field(doc, ria_json_get(doc, 0, "logical_model_digest"), digest,
                                       e) &&
                 ria_json_digest_field(doc, ria_json_get(doc, 0, "operator_contract_digest"),
                                       digest, e) &&
                 ria_json_digest_field(doc, ria_json_get(doc, 0, "bulk_capability"), digest, e);
        }
    } else if (h->kind == RIA_HEALTH) {
        if (h->flags == 0)
            ok = ria_json_fields(doc, 0, NULL, 0, NULL, 0, e);
        else {
            static const char *const fields[] = {"state", "ready", "counters"};
            const ria_json_node *state = ria_json_at(doc, ria_json_get(doc, 0, "state")),
                                *ready = ria_json_at(doc, ria_json_get(doc, 0, "ready")),
                                *counters = ria_json_at(doc, ria_json_get(doc, 0, "counters"));
            ok = ria_json_fields(doc, 0, fields, 3, fields, 3, e) && state &&
                 state->type == RIA_JSON_STRING && state->length <= 32 && ready &&
                 ready->type == RIA_JSON_BOOL && counters && counters->type == RIA_JSON_OBJECT;
            if (ok)
                for (uint32_t i = counters->child; i != RIA_JSON_NONE; i = doc->nodes[i].next) {
                    uint64_t value;
                    if (!ria_json_u64(doc, i, false, &value, e)) {
                        ok = false;
                        break;
                    }
                }
        }
    }
    if (ok && (h->kind == RIA_BIND || h->kind == RIA_BIND_BULK) && !h->status) {
        char *canonical = NULL;
        size_t n = 0;
        ok = ria_json_canonical(doc, false, &canonical, &n, e);
        if (ok && (n != length || memcmp(p, canonical, n)))
            ok = ria_fail(e, RIA_INVALID_REQUEST, "Bind JSON is not canonical");
        free(canonical);
    }
    if (!ok) {
        ria_json_free(doc);
        return ria_fail(e, RIA_INVALID_REQUEST, "invalid control message schema");
    }
    return true;
}
static bool string_is(const ria_json_doc *d, uint32_t i, const char *s)
{
    const ria_json_node *n = ria_json_at(d, i);
    return n && n->type == RIA_JSON_STRING && strlen(s) == n->length &&
           !memcmp(n->text, s, n->length);
}
bool ria_bind_validate(const ria_json_doc *d, bool response, ria_limits *out, ria_error *e)
{
    static const char *const fields[] = {"role",
                                         "logical_model_digest",
                                         "operator_contract_digest",
                                         "encoding_digest",
                                         "client_layout_digest",
                                         "placement_plan_digest",
                                         "profile",
                                         "server_executor",
                                         "limits",
                                         "server_layout_digest",
                                         "session_id",
                                         "epoch",
                                         "bulk_capability"};
    size_t count = response ? 13 : 9;
    if (!ria_json_fields(d, 0, fields, count, fields, count, e))
        return false;
    if (!string_is(d, ria_json_get(d, 0, "role"), "client") ||
        (!string_is(d, ria_json_get(d, 0, "profile"), "nvfp4") &&
         !string_is(d, ria_json_get(d, 0, "profile"), "fp8") &&
         !string_is(d, ria_json_get(d, 0, "profile"), "bf16")) ||
        (!string_is(d, ria_json_get(d, 0, "server_executor"), "cpu") &&
         !string_is(d, ria_json_get(d, 0, "server_executor"), "cuda")))
        return ria_fail(e, RIA_INVALID_REQUEST, "invalid binding role/profile/executor");
    uint8_t digest[32];
    for (size_t i = 1; i <= 5; i++)
        if (!ria_json_digest_field(d, ria_json_get(d, 0, fields[i]), digest, e))
            return false;
    static const char *const limits[] = {
        "frame_payload_bytes",  "bulk_data_bytes",     "expert_rows",
        "expert_requests",      "row_lookup_rows",     "inflight_payload_bytes",
        "operation_timeout_ms", "frame_io_timeout_ms", "write_timeout_ms"};
    uint32_t object = ria_json_get(d, 0, "limits");
    if (!ria_json_fields(d, object, limits, 9, limits, 9, e))
        return false;
    uint64_t numbers[9];
    for (size_t i = 0; i < 9; i++)
        if (!ria_json_u64(d, ria_json_get(d, object, limits[i]), false, &numbers[i], e))
            return false;
    ria_limits l = {numbers[0], numbers[1], numbers[2], numbers[3], numbers[4],
                    numbers[5], numbers[6], numbers[7], numbers[8]};
    if (!ria_limits_validate(&l, e))
        return false;
    if (response) {
        uint8_t session[16];
        const char *s;
        size_t n;
        uint64_t epoch;
        if (!ria_json_digest_field(d, ria_json_get(d, 0, "server_layout_digest"), digest, e) ||
            !ria_json_string(d, ria_json_get(d, 0, "session_id"), &s, &n, e) ||
            !ria_hex_decode(s, n, session, 16, e) || zero(session, 16) ||
            !ria_json_u64(d, ria_json_get(d, 0, "epoch"), true, &epoch, e) || !epoch ||
            !ria_json_digest_field(d, ria_json_get(d, 0, "bulk_capability"), digest, e))
            return ria_fail(e, RIA_INVALID_REQUEST, "invalid issued binding identity");
    }
    *out = l;
    return true;
}
bool ria_bind_response_json(const ria_json_doc *request, const ria_limits *agreed,
                            const uint8_t layout[32], const uint8_t session[16], uint64_t epoch,
                            const uint8_t capability[32], char **json, size_t *length, ria_error *e)
{
    ria_limits requested;
    if (!json || !length || !layout || !session || !capability || !epoch || zero(session, 16) ||
        zero(capability, 32) || !ria_bind_validate(request, false, &requested, e) ||
        !ria_limits_validate(agreed, e))
        return ria_fail(e, RIA_INVALID_REQUEST, "invalid bind response inputs");
    uint64_t a[] = {
        agreed->frame_payload_bytes,  agreed->bulk_data_bytes,     agreed->expert_rows,
        agreed->expert_requests,      agreed->row_lookup_rows,     agreed->inflight_payload_bytes,
        agreed->operation_timeout_ms, agreed->frame_io_timeout_ms, agreed->write_timeout_ms};
    uint64_t r[] = {requested.frame_payload_bytes,  requested.bulk_data_bytes,
                    requested.expert_rows,          requested.expert_requests,
                    requested.row_lookup_rows,      requested.inflight_payload_bytes,
                    requested.operation_timeout_ms, requested.frame_io_timeout_ms,
                    requested.write_timeout_ms};
    for (unsigned i = 0; i < 9; i++)
        if (a[i] > r[i])
            return ria_fail(e, RIA_RESOURCE_LIMIT, "bind response enlarges requested limit");
    static const char *const names[] = {"logical_model_digest",  "operator_contract_digest",
                                        "encoding_digest",       "client_layout_digest",
                                        "placement_plan_digest", "profile",
                                        "server_executor"};
    const char *s[7];
    for (unsigned i = 0; i < 7; i++)
        if (!ria_json_string(request, ria_json_get(request, 0, names[i]), &s[i], NULL, e))
            return false;
    char physical[65], sid[33], cap[65];
    ria_hex_encode(layout, 32, physical);
    ria_hex_encode(session, 16, sid);
    ria_hex_encode(capability, 32, cap);
    char buffer[4096];
    int n = snprintf(
        buffer, sizeof buffer,
        "{\"role\":\"client\",\"logical_model_digest\":\"%s\",\"operator_contract_digest\":\"%s\","
        "\"encoding_digest\":\"%s\",\"client_layout_digest\":\"%s\",\"placement_plan_digest\":\"%"
        "s\",\"profile\":\"%s\",\"server_executor\":\"%s\",\"server_layout_digest\":\"%s\","
        "\"session_id\":\"%s\",\"epoch\":\"%llu\",\"bulk_capability\":\"%s\",\"limits\":{\"frame_"
        "payload_bytes\":%llu,\"bulk_data_bytes\":%llu,\"expert_rows\":%llu,\"expert_requests\":%"
        "llu,\"row_lookup_rows\":%llu,\"inflight_payload_bytes\":%llu,\"operation_timeout_ms\":%"
        "llu,\"frame_io_timeout_ms\":%llu,\"write_timeout_ms\":%llu}}",
        s[0], s[1], s[2], s[3], s[4], s[5], s[6], physical, sid, (unsigned long long)epoch, cap,
        (unsigned long long)a[0], (unsigned long long)a[1], (unsigned long long)a[2],
        (unsigned long long)a[3], (unsigned long long)a[4], (unsigned long long)a[5],
        (unsigned long long)a[6], (unsigned long long)a[7], (unsigned long long)a[8]);
    if (n < 0 || (size_t)n >= sizeof buffer)
        return ria_fail(e, RIA_INTERNAL_ERROR, "bind response encoding exceeds bound");
    ria_json_doc d = {0};
    ria_json_limits bounds = {sizeof buffer, 128, 8};
    bool ok = ria_json_parse(buffer, (size_t)n, bounds, &d, e) &&
              ria_json_canonical(&d, false, json, length, e);
    ria_json_free(&d);
    return ok;
}
void ria_binding_init(ria_binding *b, const ria_limits *l)
{
    memset(b, 0, sizeof *b);
    if (l)
        b->limits = *l;
    b->valid = true;
}
bool ria_binding_protect(ria_binding *b, uint64_t control, uint64_t rows, uint64_t bulk,
                         ria_error *e)
{
    uint64_t total;
    if (!b || !b->valid || b->reserved_bytes || !control || !ria_u64_add(control, rows, &total) ||
        !ria_u64_add(total, bulk, &total) || total >= b->limits.inflight_payload_bytes)
        return ria_fail(e, RIA_RESOURCE_LIMIT, "invalid protected progress reservation");
    b->progress_bytes[0] = control;
    b->progress_bytes[1] = rows;
    b->progress_bytes[2] = bulk;
    return true;
}
bool ria_binding_install(ria_binding *b, const uint8_t s[16], uint64_t epoch, const uint8_t cap[32],
                         ria_error *e)
{
    if (!b || !b->valid || b->bound || !s || zero(s, 16) || !epoch || !cap || zero(cap, 32) ||
        !ria_limits_validate(&b->limits, e))
        return ria_fail(e, RIA_IDENTITY_MISMATCH, "invalid initial binding transition");
    memcpy(b->session, s, 16);
    memcpy(b->bulk_capability, cap, 32);
    b->epoch = epoch;
    b->bound = true;
    b->last_request[0] = 1;
    b->last_admitted[0] = 1;
    return true;
}
bool ria_binding_fresh(ria_binding *b, ria_error *e)
{
    uint8_t s[16], cap[32], epoch[8];
    if (RAND_bytes(s, 16) != 1 || RAND_bytes(cap, 32) != 1 || RAND_bytes(epoch, 8) != 1)
        return ria_fail(e, RIA_INTERNAL_ERROR, "binding entropy unavailable");
    return ria_binding_install(b, s, ria_read_u64(epoch), cap, e);
}
bool ria_binding_bulk(ria_binding *b, const uint8_t cap[32], bool peer, ria_error *e)
{
    if (!b || !b->valid || !b->bound || b->bulk_bound || b->bulk_capability_used || !peer || !cap ||
        CRYPTO_memcmp(cap, b->bulk_capability, 32))
        return ria_fail(e, RIA_UNAUTHORIZED, "bulk capability or peer rejected");
    b->bulk_bound = b->bulk_capability_used = true;
    OPENSSL_cleanse(b->bulk_capability, 32);
    return true;
}
bool ria_binding_receive(ria_binding *b, const ria_header *h, bool bulk, ria_error *e)
{
    if (!b || !h || !b->valid || h->flags || h->status || !known(h->kind) || !h->request_id)
        return ria_fail(e, RIA_INVALID_REQUEST, "invalid request direction or binding");
    unsigned channel = bulk ? 1 : 0;
    if (!b->bound) {
        if (bulk || h->kind != RIA_BIND || h->request_id != 1 || h->epoch ||
            !zero(h->session, 16) || b->last_request[0])
            return ria_fail(e, RIA_IDENTITY_MISMATCH, "invalid initial bind request");
        b->last_request[0] = 1;
        return true;
    }
    if (h->kind == RIA_BIND || h->request_id <= b->last_request[channel] || h->epoch != b->epoch ||
        CRYPTO_memcmp(h->session, b->session, 16) ||
        (bulk ? (h->kind != RIA_BIND_BULK && h->kind != RIA_CHUNK)
              : (h->kind == RIA_BIND_BULK || h->kind == RIA_CHUNK)) ||
        (bulk && !b->bulk_bound && (h->kind != RIA_BIND_BULK || h->request_id != 1)) ||
        (bulk && b->bulk_bound && h->kind == RIA_BIND_BULK))
        return ria_fail(e, RIA_IDENTITY_MISMATCH, "stale, reused or wrong-channel request");
    b->last_request[channel] = h->request_id;
    if (h->kind == RIA_CLOSE)
        b->draining = true;
    return true;
}
static unsigned category(uint16_t kind)
{
    return kind == RIA_EXPERT || kind == RIA_SHARED ? 0
           : kind == RIA_ROWS                       ? 1
           : kind == RIA_CHUNK                      ? 2
                                                    : 3;
}
bool ria_binding_admit(ria_binding *b, const ria_header *h, uint64_t cost, uint64_t now,
                       ria_error *e)
{
    if (!b || !h || !b->valid || !b->bound || !cost || !h->request_id || !known(h->kind) ||
        h->kind == RIA_BIND || h->kind == RIA_BIND_BULK || h->flags || h->status ||
        h->epoch != b->epoch || CRYPTO_memcmp(h->session, b->session, 16) ||
        h->payload_length > cost || h->payload_length > b->limits.frame_payload_bytes ||
        (b->draining && (category(h->kind) != 3)))
        return ria_fail(e, RIA_NOT_READY, "binding cannot admit work");
    unsigned cat = category(h->kind), count = 0;
    bool bulk = h->kind == RIA_CHUNK;
    if (h->request_id <= b->last_admitted[bulk ? 1 : 0])
        return ria_fail(e, RIA_INVALID_REQUEST, "nonmonotone admitted request id");
    size_t free_slot = 9;
    uint64_t used[4] = {0};
    for (size_t i = 0; i < 9; i++) {
        if (b->pending[i].used) {
            if (b->pending[i].id == h->request_id && b->pending[i].bulk == bulk)
                return ria_fail(e, RIA_INVALID_REQUEST, "duplicate outstanding request");
            unsigned pc = category(b->pending[i].kind);
            if (pc == cat)
                count++;
            if (!ria_u64_add(used[pc], b->pending[i].cost, &used[pc]))
                return ria_fail(e, RIA_INTERNAL_ERROR, "corrupt pending accounting");
        } else
            free_slot = i;
    }
    unsigned maximum = cat == 0   ? (unsigned)b->limits.expert_requests
                       : cat == 1 ? 2
                       : cat == 2 ? 1
                                  : 4;
    uint64_t total, deadline;
    if (!b->progress_bytes[0] || (cat == 1 && !b->progress_bytes[1]) ||
        (cat == 2 && (!b->bulk_bound || !b->progress_bytes[2])) || count >= maximum ||
        free_slot == 9 || !ria_u64_add(b->reserved_bytes, cost, &total) ||
        total > b->limits.inflight_payload_bytes ||
        !ria_u64_add(now, b->limits.operation_timeout_ms, &deadline))
        return ria_fail(e, RIA_RESOURCE_LIMIT, "request credit unavailable");
    if (!ria_u64_add(used[cat], cost, &used[cat]))
        return ria_fail(e, RIA_RESOURCE_LIMIT, "pending byte overflow");
    for (unsigned k = 0; k < 3; k++) {
        static const unsigned protected_classes[3] = {3, 1, 2};
        unsigned protected_class = protected_classes[k];
        if (used[protected_class] < b->progress_bytes[k] &&
            !ria_u64_add(total, b->progress_bytes[k] - used[protected_class], &total))
            return ria_fail(e, RIA_RESOURCE_LIMIT, "progress byte overflow");
    }
    if (total > b->limits.inflight_payload_bytes)
        return ria_fail(e, RIA_RESOURCE_LIMIT, "request would exhaust protected progress bytes");
    b->pending[free_slot] =
        (ria_pending){h->request_id, cost, deadline, h->kind, true, false, bulk};
    b->reserved_bytes += cost;
    b->last_admitted[bulk ? 1 : 0] = h->request_id;
    return true;
}
bool ria_binding_terminal_channel(ria_binding *b, uint64_t id, bool bulk, bool quiescent,
                                  ria_error *e)
{
    if (!b || !b->valid || !quiescent)
        return ria_fail(e, RIA_INTERNAL_ERROR, "terminal requires valid binding and quiescence");
    for (size_t i = 0; i < 9; i++) {
        ria_pending *p = &b->pending[i];
        if (p->used && p->id == id && p->bulk == bulk) {
            if (p->kind == RIA_CLOSE) {
                for (size_t j = 0; j < 9; j++)
                    if (j != i && b->pending[j].used)
                        return ria_fail(e, RIA_NOT_READY, "close waits for admitted work");
            }
            if (p->cost > b->reserved_bytes)
                return ria_fail(e, RIA_INTERNAL_ERROR, "corrupt credit accounting");
            b->reserved_bytes -= p->cost;
            if (p->kind == RIA_EXPERT || p->kind == RIA_SHARED || p->kind == RIA_ROWS) {
                b->terminal[b->terminal_next] = id;
                b->terminal_next = (b->terminal_next + 1) % 16;
                if (b->terminal_count < 16)
                    b->terminal_count++;
            }
            memset(p, 0, sizeof *p);
            return true;
        }
    }
    return ria_fail(e, RIA_INVALID_REQUEST, "no outstanding terminal target");
}
bool ria_binding_terminal(ria_binding *b, uint64_t id, bool quiescent, ria_error *e)
{
    return ria_binding_terminal_channel(b, id, false, quiescent, e);
}
bool ria_binding_cancel(ria_binding *b, uint64_t target, uint32_t *state, ria_error *e)
{
    if (!b || !b->valid || !state || !target)
        return ria_fail(e, RIA_INVALID_REQUEST, "invalid cancel lookup");
    for (size_t i = 0; i < 9; i++)
        if (b->pending[i].used && !b->pending[i].bulk && b->pending[i].id == target &&
            (b->pending[i].kind == RIA_EXPERT || b->pending[i].kind == RIA_SHARED ||
             b->pending[i].kind == RIA_ROWS)) {
            b->pending[i].cancelled = true;
            *state = 0;
            return true;
        }
    for (unsigned i = 0; i < b->terminal_count; i++)
        if (b->terminal[i] == target) {
            *state = 1;
            return true;
        }
    *state = 2;
    return true;
}
bool ria_binding_response(ria_binding *b, const ria_header *h, ria_error *e)
{
    if (!b || !h || !b->valid || !b->bound || h->flags != 1 || h->status > 11 ||
        h->epoch != b->epoch || CRYPTO_memcmp(h->session, b->session, 16))
        return ria_fail(e, RIA_IDENTITY_MISMATCH, "response binding mismatch");
    for (size_t i = 0; i < 9; i++)
        if (b->pending[i].used && b->pending[i].id == h->request_id &&
            b->pending[i].bulk == (h->kind == RIA_CHUNK)) {
            if (b->pending[i].kind != h->kind)
                return ria_fail(e, RIA_IDENTITY_MISMATCH, "response operation mismatch");
            return true;
        }
    return ria_fail(e, RIA_INVALID_REQUEST, "unsolicited or duplicate response");
}
bool ria_binding_expired(const ria_binding *b, uint64_t now, uint64_t *id)
{
    if (!b || !b->valid)
        return false;
    for (size_t i = 0; i < 9; i++)
        if (b->pending[i].used && now >= b->pending[i].deadline) {
            if (id)
                *id = b->pending[i].id;
            return true;
        }
    return false;
}
void ria_binding_invalidate(ria_binding *b)
{
    if (!b)
        return; /* Callers retain executor buffers until quiescent; only network credits die. */
    b->valid = false;
    b->bound = false;
    b->draining = true;
    b->bulk_bound = false;
    b->bulk_capability_used = true;
    b->reserved_bytes = 0;
    OPENSSL_cleanse(b->bulk_capability, 32);
}
