#ifndef RIA_SERVER_H
#define RIA_SERVER_H
#include "bank.h"
#include "service.h"
#include <pthread.h>
#ifdef __cplusplus
extern "C" {
#endif
/* A single network owner serializes every binding transition and TLS stream.
 * Workers own numeric contexts and disjoint contribution slots, borrow request
 * bytes until quiescence, and never touch SSL/binding state or publish replies. */
#define RIA_SERVER_JOBS 768u
#define RIA_SERVER_REQUESTS 2u
typedef struct { uint64_t shard, begin, end; } ria_chunk_grant;
typedef struct { ria_chunk_grant *ranges; size_t count; } ria_server_grants;
bool ria_server_grants_open(ria_server_grants *grants,
                            const ria_tensor_store *store, ria_error *error);
void ria_server_grants_close(ria_server_grants *grants);
bool ria_server_grants_chunk(const ria_server_grants *grants,
                             const ria_shard *shard, uint64_t index,
                             ria_error *error);
/* Shared bounded queue mechanics are testable without NUMA/hardware/model
 * execution. Caller holds its lifecycle mutex; fixed records never allocate. */
typedef struct { void *request; unsigned node, row, entry; } ria_server_job;
typedef struct { ria_server_job jobs[RIA_SERVER_JOBS]; unsigned count; } ria_server_queue;
bool ria_server_queue_push(ria_server_queue *queue, ria_server_job job,
                           ria_error *error);
bool ria_server_queue_take(ria_server_queue *queue, unsigned node,
                           ria_server_job *job);
/* The validated wire view remains borrowed until its worker finishes. These
 * bridges bound the worker's private float buffers and exact contribution slot. */
bool ria_server_contribution_unpack(const ria_expert_request *,unsigned row,unsigned entry,
                                    float *,uint64_t,float *,ria_error *);
bool ria_server_contribution_pack(const ria_expert_request *,unsigned entry,const float *,uint64_t,
                                  uint8_t *,uint64_t,ria_error *);
/* Complete native process entry. CPU builds never initialize/link CUDA.
 * Fatal sticky executor failure invalidates the pair and returns nonzero after
 * bounded quiescent drain; an unquiescent timeout terminates the process. */
bool ria_server_run(const char *configuration, ria_error *error);
#ifdef __cplusplus
}
#endif
#endif
