#ifndef RIA_RUNTIME_H
#define RIA_RUNTIME_H
#include "common.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
  uint64_t host_limit, host_available, swap_limit, memlock, locked_bytes;
  unsigned cpu_count;
  char cpu_mask[4096], memory_node_mask[4096];
} ria_runtime_observation;
/* Read actual process/cgroup restrictions; no model or GPU initialization and
 * no allocation benchmark. Runtime entry points reject mismatched deployment
 * restrictions even if a provisioned plan claims admission. */
bool ria_runtime_inspect(ria_runtime_observation *, ria_error *);
bool ria_runtime_require(uint64_t host_cap, uint64_t locked_population,
                         uint64_t pinned_cap, ria_runtime_observation *,
                         ria_error *);
bool ria_runtime_memory_counter(const char *path, const char *key,
                                uint64_t *bytes, ria_error *);
#ifdef __cplusplus
}
#endif
#endif
