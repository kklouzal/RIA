#ifndef RIA_STATE_H
#define RIA_STATE_H
#include "common.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  uint64_t id, epoch, generation, first_position, position_count, bytes;
  uint64_t dirty_begin, dirty_end, event;
  uint32_t representation, leases;
  int64_t device_slot;
  uint8_t *host;
  bool host_valid, device_valid, pending, cancelled;
} ria_state_page;
typedef struct {
  void *context;
  bool (*upload)(void *, uint64_t, const void *, uint64_t, uint64_t *,
                 ria_error *);
  bool (*download)(void *, uint64_t, void *, uint64_t, uint64_t *, ria_error *);
  bool (*wait)(void *, uint64_t, ria_error *);
  bool (*zero)(void *, uint64_t, uint64_t, ria_error *);
} ria_state_io;
typedef struct {
  ria_state_page *pages;
  uint64_t page_count, page_limit, host_bytes, host_limit;
  uint64_t *slots, *ages, slot_count, tick, slot_bytes;
  uint64_t epoch, generation;
  bool unusable;
  ria_state_io io;
} ria_state_store;

/* One graph/session owner; asynchronous I/O callbacks own completion events.
 * No page becomes valid before wait succeeds. Read leases protect eviction.
 * Packed representation bytes are moved verbatim, never requantized here. */
bool ria_state_create(ria_state_store *store, uint64_t page_limit,
                      uint64_t host_limit, uint64_t slots, uint64_t slot_bytes,
                      ria_state_io io, ria_error *error);
bool ria_state_begin(ria_state_store *store, uint64_t epoch,
                     uint64_t generation, ria_error *error);
bool ria_state_add(ria_state_store *store, uint64_t id, uint32_t representation,
                   uint64_t first, uint64_t count, uint64_t bytes,
                   ria_state_page **page, ria_error *error);
bool ria_state_acquire(ria_state_store *store, uint64_t id,
                       ria_state_page **page, ria_error *error);
bool ria_state_release(ria_state_store *store, ria_state_page *page,
                       ria_error *error);
bool ria_state_dirty(ria_state_store *store, ria_state_page *page,
                     uint64_t begin, uint64_t end, uint64_t event,
                     ria_error *error);
bool ria_state_flush(ria_state_store *store, ria_state_page *page,
                     ria_error *error);
bool ria_state_cancel(ria_state_store *store, ria_error *error);
bool ria_state_destroy(ria_state_store *store, ria_error *error);
#ifdef __cplusplus
}
#endif
#endif
