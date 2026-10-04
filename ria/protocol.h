#ifndef RIA_PROTOCOL_H
#define RIA_PROTOCOL_H
#include "common.h"
#include "json.h"
#ifdef __cplusplus
extern "C" {
#endif
#define RIA_HEADER_BYTES 64u
#define RIA_FRAME_MAX UINT64_C(16777216)
#define RIA_BULK_MAX UINT64_C(4194304)
#define RIA_CONTROL_MAX UINT64_C(262144)
#define RIA_ERROR_MAX UINT64_C(16384)
enum {
  RIA_BIND = 1,
  RIA_BIND_BULK = 2,
  RIA_EXPERT = 10,
  RIA_ROWS = 11,
  RIA_CHUNK = 12,
  RIA_SHARED = 13,
  RIA_CANCEL = 20,
  RIA_HEALTH = 21,
  RIA_CLOSE = 22
};
typedef struct {
  uint16_t kind;
  uint32_t flags, status;
  uint64_t payload_length, request_id, epoch;
  uint8_t session[16];
} ria_header;
typedef struct {
  uint64_t frame_payload_bytes, bulk_data_bytes, expert_rows, expert_requests,
      row_lookup_rows, inflight_payload_bytes;
  uint64_t operation_timeout_ms, frame_io_timeout_ms, write_timeout_ms;
} ria_limits;
bool ria_limits_validate(const ria_limits *limits, ria_error *error);
bool ria_header_decode(const uint8_t bytes[64], uint64_t frame_limit,
                       ria_header *header, ria_error *error);
bool ria_header_encode(const ria_header *header, uint8_t bytes[64],
                       ria_error *error);
typedef struct {
  uint64_t handle;
  uint32_t input_width, output_width, max_rows, quantizer_context_bytes;
  uint16_t expert_count, max_selected;
  float coefficient_min, coefficient_max;
  bool shared, input_bf16, output_bf16;
} ria_operation;
typedef struct {
  uint64_t operation_handle, invocation_id;
  uint32_t row_count, input_width, output_width, entry_count,
      quantizer_context_bytes;
  const uint8_t *row_ids, *offsets, *entries, *inputs, *quantizer_context;
  uint64_t response_bytes;
} ria_expert_request;
bool ria_expert_lengths(uint64_t rows, uint64_t entries, uint64_t input_width,
                        uint64_t output_width, uint64_t context_bytes,
                        uint64_t *request_bytes, uint64_t *response_bytes,
                        ria_error *error);
/* Views borrow the exact payload; validation completes before any
 * execution/allocation. */
bool ria_expert_parse(const uint8_t *payload, size_t length, uint16_t kind,
                      const ria_operation *operation, const ria_limits *limits,
                      ria_expert_request *request, ria_error *error);
bool ria_expert_response(const uint8_t *payload, size_t length,
                         const ria_expert_request *request,
                         const ria_operation *operation, ria_error *error);
typedef struct {
  uint64_t handle, row_count, packed_row_stride, representation_handle;
} ria_table;
typedef struct {
  uint64_t handle;
  uint32_t row_count;
  const uint8_t *pairs;
  uint64_t response_bytes;
} ria_row_request;
bool ria_rows_parse(const uint8_t *payload, size_t length,
                    const ria_table *table, const ria_limits *limits,
                    ria_row_request *request, ria_error *error);
bool ria_rows_response(const uint8_t *payload, size_t length,
                       const ria_row_request *request, const ria_table *table,
                       ria_error *error);
bool ria_chunk_request(const uint8_t *payload, size_t length, uint64_t *handle,
                       uint64_t *chunk_index, ria_error *error);
bool ria_chunk_response(const uint8_t *payload, size_t length, uint64_t handle,
                        uint64_t index, uint64_t shard_length,
                        uint32_t chunk_bytes, const uint8_t expected_hash[32],
                        const uint8_t **data, size_t *data_length,
                        ria_error *error);
bool ria_cancel_parse(const uint8_t *payload, size_t length,
                      uint64_t cancel_request_id, uint64_t epoch,
                      uint64_t *target, ria_error *error);
bool ria_cancel_response(const uint8_t *payload, size_t length, uint64_t target,
                         uint64_t epoch, uint32_t *state, ria_error *error);
/* Strict bounded JSON control schema; Bind JSON must already be canonical on
 * wire. */
bool ria_control_json(const uint8_t *payload, size_t length,
                      const ria_header *header, ria_json_doc *doc,
                      ria_error *error);
bool ria_bind_validate(const ria_json_doc *doc, bool response,
                       ria_limits *limits, ria_error *error);
bool ria_bind_response_json(const ria_json_doc *request,
                            const ria_limits *agreed,
                            const uint8_t server_layout[32],
                            const uint8_t session[16], uint64_t epoch,
                            const uint8_t capability[32], char **json,
                            size_t *length, ria_error *error);

typedef struct {
  uint64_t id, cost, deadline;
  uint16_t kind;
  bool used, cancelled, bulk;
} ria_pending;
/* One event-loop/thread owns a binding. Worker completion must be marshaled to
 * it. */
typedef struct {
  ria_limits limits;
  uint8_t session[16], bulk_capability[32];
  uint64_t epoch, last_request[2], last_admitted[2], reserved_bytes;
  bool valid, bound, draining, bulk_bound, bulk_capability_used;
  ria_pending pending[9];
  uint64_t progress_bytes[3], terminal[16];
  unsigned terminal_count, terminal_next;
} ria_binding;
void ria_binding_init(ria_binding *binding, const ria_limits *limits);
/* Reserve disjoint control/row/bulk bytes from the admitted total before work.
 */
bool ria_binding_protect(ria_binding *binding, uint64_t control_bytes,
                         uint64_t row_bytes, uint64_t bulk_bytes,
                         ria_error *error);
bool ria_binding_install(ria_binding *binding, const uint8_t session[16],
                         uint64_t epoch, const uint8_t capability[32],
                         ria_error *error);
bool ria_binding_fresh(ria_binding *binding, ria_error *error);
bool ria_binding_bulk(ria_binding *binding, const uint8_t capability[32],
                      bool same_authorized_peer, ria_error *error);
bool ria_binding_receive(ria_binding *binding, const ria_header *header,
                         bool bulk, ria_error *error);
bool ria_binding_admit(ria_binding *binding, const ria_header *header,
                       uint64_t worst_case_bytes, uint64_t now_ms,
                       ria_error *error);
bool ria_binding_terminal(ria_binding *binding, uint64_t id, bool quiescent,
                          ria_error *error);
bool ria_binding_terminal_channel(ria_binding *binding, uint64_t id, bool bulk,
                                  bool quiescent, ria_error *error);
/* 0 pending, 1 terminal, 2 unknown. No credit is returned by cancellation. */
bool ria_binding_cancel(ria_binding *binding, uint64_t target, uint32_t *state,
                        ria_error *error);
bool ria_binding_response(ria_binding *binding, const ria_header *header,
                          ria_error *error);
bool ria_binding_expired(const ria_binding *binding, uint64_t now_ms,
                         uint64_t *id);
void ria_binding_invalidate(ria_binding *binding);
#ifdef __cplusplus
}
#endif
#endif
