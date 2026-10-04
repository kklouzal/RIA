#ifndef RIA_ADMIN_H
#define RIA_ADMIN_H
#include "common.h"
#include <sys/types.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ria_admin ria_admin;
#define RIA_ADMIN_STACK_BYTES UINT64_C(1048576)
/* Fixed owner, stack/guard and bounded parser reservation; callback-owned
 * resources are accounted by the role. Pure arithmetic, no listener starts. */
uint64_t ria_admin_reserved_bytes(void);
typedef struct {
  void *context;
  /* Callbacks synchronize with the service owner. No operation may outlive
   * context. Drain stops admission and returns only after real quiescence;
   * it must honor the absolute monotonic deadline, including GPU work. */
  bool (*health)(void *, bool *ready, bool *active, ria_error *);
  bool (*drain)(void *, uint64_t deadline_ms, ria_error *);
} ria_admin_callbacks;
bool ria_admin_start(const char *path, uid_t authorized_uid,
                     uint64_t request_timeout_ms, ria_admin_callbacks callbacks,
                     ria_admin **out, ria_error *error);
/* If post-creation cleanup fails, start returns false with a non-NULL owned
 * output. The caller must stop it before destroying callback context. */
/* On timeout state/resources remain owned: caller must terminate the service
 * nonzero or retry joining after completion; never free a live callback. */
bool ria_admin_stop(ria_admin *admin, uint64_t timeout_ms, ria_error *error);
#ifdef __cplusplus
}
#endif
#endif
