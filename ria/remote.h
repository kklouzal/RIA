#ifndef RIA_REMOTE_H
#define RIA_REMOTE_H
#include "graph.h"
#include "service.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ria_remote ria_remote;
typedef struct {
  const ria_service *service;
  const ria_tensor_store *client_store;
  uint8_t client_layout_digest[32], server_layout_digest[32],
      placement_plan_digest[32];
} ria_remote_options;
/* Graph callbacks split selected slots, expert-grouped prompt rows and Engram
 * associations to agreed frame/count/byte bounds. They publish contributions
 * atomically and assign a fresh invocation ID to each expert subgroup. Prompt
 * row identities/coefficients/original slots survive every split. Direct
 * multirow callers pass already bounded units. Bind must support one native
 * expert/table/error unit;
 * authorized verification chunks always retain their manifest granule.
 * Graph thread owns all SSL and binding transitions. Abort only shuts down
 * stable descriptors under the lifecycle mutex; destruction follows joining
 * that graph thread. Required-operation failure retires the entire pair. */
bool ria_remote_open(ria_remote **remote, const ria_remote_options *options,
                     ria_error *error);
bool ria_remote_close(ria_remote *remote, ria_error *error);
void ria_remote_abort(ria_remote *remote, uint64_t generation);
bool ria_remote_begin_generation(ria_remote *remote, uint64_t *epoch,
                                 uint64_t *generation, ria_error *error);
ria_graph_remote ria_remote_callbacks(ria_remote *remote);
/* Conservative endpoint reservation: owner, bounded publication pools and
 * maximum one synchronous request/reply envelope. SSL/runtime margin is an
 * independently declared frontend reservation. Pure host sizing, no network. */
bool ria_remote_host_required_bytes(uint32_t prefill_rows,const ria_limits *,
                                    uint64_t *bytes,ria_error *error);
bool ria_remote_evaluate(ria_remote *remote, uint32_t layer,
                         uint64_t invocation, uint32_t rows,
                         const uint64_t *row_ids, const uint32_t *offsets,
                         const uint16_t *experts, const uint16_t *slots,
                         const float *coefficients, const float *inputs,
                         bool shared, float *contributions, ria_error *error);
bool ria_remote_rows(ria_remote *remote, uint32_t layer, const uint64_t *rows,
                     const uint64_t *associations, uint32_t count,
                     uint8_t *packed, ria_error *error);
bool ria_remote_chunk(ria_remote *remote, const ria_shard *trusted,
                      uint64_t index, uint8_t **bytes, size_t *length,
                      ria_error *error);
/* Revision-1 agreed charge: request + max(success reply,16KiB typed error)
 * + 128 framing bytes + 8MiB workspace for Expert/Shared (zero otherwise).
 * Actual endpoint allocations have independent memory-plan admission. */
bool ria_request_charge(uint16_t kind, uint64_t request_bytes,
                        uint64_t response_bytes, uint64_t *charge,
                        ria_error *error);
bool ria_progress_charges(const ria_limits *limits, uint64_t *control,
                          uint64_t *row, uint64_t *bulk, ria_error *error);
#ifdef __cplusplus
}
#endif
#endif
