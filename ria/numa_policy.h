#ifndef RIA_NUMA_H
#define RIA_NUMA_H
#include "bank.h"
#include "service.h"
#include <string.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Topology and physical verification run at startup, never in numeric loops.
 * All bind calls precede first touch and use flags=0. Query-only move_pages
 * verifies every base page; no migration flags or host-wide changes are used. */
typedef struct ria_numa ria_numa;
/* Exact values are shared with the generated reviewed seccomp policy. */
#define RIA_NUMA_BIND_MODE 2u
#define RIA_NUMA_QUERY_PAGES 64u
#define RIA_NUMA_WORKER_BYTES UINT64_C(2097152)
#define RIA_NUMA_WORKER_STACK_BYTES UINT64_C(1048576)
/* One fixed per-node worker arena: stack, CPU quantizer owner/scratch (CPU
 * only), and distinct widened-FP32 input/output rows. Shared by admission and
 * execution; it does not inspect topology or allocate/initialize a runtime. */
static inline bool ria_numa_worker_rows(uint32_t bound,const char *executor,uint32_t *rows,uint64_t *scratch,ria_error *e) {
  if (!rows || !scratch || !executor || !bound || bound>64 ||
      (strcmp(executor,"cpu") && strcmp(executor,"cuda")))
    return ria_fail(e,RIA_INVALID_REQUEST,"invalid admitted worker row bound");
  *scratch=0;
  if (!strcmp(executor,"cpu") && !ria_expert_cpu_required_bytes(RIA_WIDTH,RIA_INTERMEDIATE,RIA_WIDTH,scratch,e)) return false;
  uint64_t fixed,capacity;
  if (!ria_u64_add(RIA_NUMA_WORKER_STACK_BYTES,*scratch,&fixed) || fixed>=RIA_NUMA_WORKER_BYTES)
    return ria_fail(e,RIA_RESOURCE_LIMIT,"worker scratch exceeds fixed NUMA arena");
  capacity=(RIA_NUMA_WORKER_BYTES-fixed)/(2*RIA_WIDTH*sizeof(float));
  if (!capacity) return ria_fail(e,RIA_RESOURCE_LIMIT,"worker arena cannot hold one expert row");
  *rows=(uint32_t)(capacity<bound ? capacity : bound);return true;
}
typedef struct {
  uint64_t canonical_bytes, replica_bytes, total_bytes;
  uint64_t node_bytes[RIA_NUMA_MAX_NODES];
} ria_numa_account;
bool ria_numa_config_validate(const ria_expert_config *config, ria_error *error);
bool ria_numa_account_store(const ria_tensor_store *store,
                            const ria_expert_config *config,
                            ria_numa_account *account, ria_error *error);
bool ria_numa_topology(const ria_expert_config *config, ria_error *error);
bool ria_numa_place_shard(void *config, const ria_tensor_store *store,
                          const ria_shard *shard, bool populated,
                          ria_error *error);
bool ria_numa_arena(unsigned node, uint64_t bytes, void **arena,
                     uint64_t *mapped_bytes, ria_error *error);
bool ria_numa_verify(unsigned node, const void *arena, uint64_t bytes,
                      ria_error *error);
void ria_numa_release(void *arena, uint64_t bytes);
bool ria_numa_affinity(const ria_expert_node *node, unsigned worker,
                       ria_error *error);
bool ria_numa_open(ria_numa **out, const ria_bank *bank,
                   const ria_expert_config *config, ria_error *error);
bool ria_numa_open_controlled(ria_numa **,const ria_bank *,const ria_expert_config *,ria_tensor_progress,void *,ria_error *);
void ria_numa_close(ria_numa *local);
const ria_expert *ria_numa_expert(const ria_numa *local, unsigned node,
                                 uint64_t handle, uint16_t expert);
const uint8_t *ria_numa_table(const ria_numa *local, unsigned node,
                             unsigned table);
unsigned ria_numa_owner(const ria_expert_config *config, uint64_t handle,
                         uint16_t expert);
#ifdef __cplusplus
}
#endif
#endif
