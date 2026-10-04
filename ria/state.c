#define _GNU_SOURCE
#include "state.h"
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

static bool valid(ria_state_store *s, ria_state_page *p, ria_error *e) {
  if (!p || !s->pages)
    return ria_fail(e, RIA_INVALID_REQUEST,
                    "state page/store has no owned allocation");
  bool owned = false;
  for (uint64_t i = 0; i < s->page_count; i++)
    if (p == &s->pages[i]) {
      owned = true;
      break;
    }
  if (!owned)
    return ria_fail(e, RIA_INVALID_REQUEST,
                    "state page is not owned by this store");
  if (s->unusable || p->epoch != s->epoch || p->generation != s->generation ||
      p->cancelled)
    return ria_fail(e, RIA_NOT_READY,
                    "state belongs to an unusable/stale generation");
  return true;
}
static bool complete(ria_state_store *s, ria_state_page *p, ria_error *e) {
  if (!p->pending)
    return true;
  if (!s->io.wait(s->io.context, p->event, e)) {
    s->unusable = true;
    return false;
  }
  p->pending = false;
  p->event = 0;
  if (p->cancelled)
    return ria_fail(e, RIA_CANCELLED,
                    "cancelled state completion cannot be published");
  p->device_valid = true;
  return true;
}
bool ria_state_create(ria_state_store *s, uint64_t pages, uint64_t host,
                      uint64_t slots, uint64_t slot_bytes, ria_state_io io,
                      ria_error *e) {
  memset(s, 0, sizeof(*s));
  if (!pages || pages > SIZE_MAX / sizeof(*s->pages) || !host || !slots ||
      slots > INT64_MAX || slots > SIZE_MAX / sizeof(uint64_t) || !slot_bytes ||
      !io.upload || !io.download || !io.wait || !io.zero)
    return ria_fail(e, RIA_INVALID_REQUEST,
                    "invalid bounded state manager configuration");
  s->pages = calloc((size_t)pages, sizeof(*s->pages));
  s->slots = malloc((size_t)slots * sizeof(uint64_t));
  s->ages = calloc((size_t)slots, sizeof(uint64_t));
  if (!s->pages || !s->slots || !s->ages) {
    free(s->pages);
    free(s->slots);
    free(s->ages);
    memset(s, 0, sizeof(*s));
    return ria_fail(e, RIA_RESOURCE_LIMIT, "state metadata allocation failed");
  }
  for (uint64_t i = 0; i < slots; i++)
    s->slots[i] = UINT64_MAX;
  s->page_limit = pages;
  s->host_limit = host;
  s->slot_count = slots;
  s->slot_bytes = slot_bytes;
  s->io = io;
  return true;
}
bool ria_state_begin(ria_state_store *s, uint64_t epoch, uint64_t generation,
                     ria_error *e) {
  if (!epoch || !generation || s->page_count || s->unusable)
    return ria_fail(e, RIA_INVALID_REQUEST,
                    "state begin requires fresh empty manager");
  s->epoch = epoch;
  s->generation = generation;
  return true;
}
bool ria_state_add(ria_state_store *s, uint64_t id, uint32_t representation,
                   uint64_t first, uint64_t count, uint64_t bytes,
                   ria_state_page **out, ria_error *e) {
  *out = NULL;
  uint64_t end, total;
  if (s->unusable || !s->epoch || !s->generation || !count || !bytes ||
      bytes > s->slot_bytes || s->page_count == s->page_limit ||
      !ria_u64_add(first, count, &end) ||
      !ria_u64_add(s->host_bytes, bytes, &total) || total > s->host_limit ||
      bytes > SIZE_MAX)
    return ria_fail(e, RIA_RESOURCE_LIMIT,
                    "state page exceeds declared bounds");
  for (uint64_t i = 0; i < s->page_count; i++)
    if (s->pages[i].id == id)
      return ria_fail(e, RIA_INVALID_REQUEST, "duplicate state page ID");
  /* Anonymous mmap ignores fd; Cppcheck's POSIX fd range is too narrow. */
  void *host = mmap(NULL, (size_t)bytes, PROT_READ | PROT_WRITE,
                    // cppcheck-suppress invalidFunctionArg
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (host == MAP_FAILED)
    return ria_fail(e, RIA_RESOURCE_LIMIT, "private state allocation failed");
  if (!host) {
    /* Linux munmap accepts address zero; its POSIX model assumes nonnull. */
    // cppcheck-suppress [nullPointer,nullPointerRedundantCheck]
    if (munmap(host, (size_t)bytes))
      return ria_fail(e, RIA_RESOURCE_LIMIT, "null state mapping cleanup failed");
    return ria_fail(e, RIA_RESOURCE_LIMIT, "private state mapped the null address");
  }
  if (madvise(host, (size_t)bytes, MADV_DONTDUMP)) {
    munmap(host, (size_t)bytes);
    return ria_fail(e, RIA_RESOURCE_LIMIT,
                    "private state dump exclusion failed");
  }
  ria_state_page *p = &s->pages[s->page_count++];
  *p = (ria_state_page){.id = id,
                        .epoch = s->epoch,
                        .generation = s->generation,
                        .first_position = first,
                        .position_count = count,
                        .bytes = bytes,
                        .representation = representation,
                        .device_slot = -1,
                        .host = host,
                        .host_valid = true};
  s->host_bytes = total;
  *out = p;
  return true;
}
bool ria_state_flush(ria_state_store *s, ria_state_page *p, ria_error *e) {
  if (!valid(s, p, e) || !complete(s, p, e))
    return false;
  if (p->host_valid)
    return true;
  if (!p->device_valid || p->device_slot < 0) {
    s->unusable = true;
    return ria_fail(e, RIA_INTERNAL_ERROR, "state has no authoritative copy");
  }
  uint64_t event;
  /* Copy the exact representation. Dirty bounds are retained as diagnostic
   * producer metadata; full-page transfer preserves bytes outside the range. */
  if (!s->io.download(s->io.context, (uint64_t)p->device_slot, p->host,
                      p->bytes, &event, e) ||
      !s->io.wait(s->io.context, event, e)) {
    s->unusable = true;
    return false;
  }
  p->host_valid = true;
  p->dirty_begin = p->dirty_end = 0;
  return true;
}
bool ria_state_acquire(ria_state_store *s, uint64_t id, ria_state_page **out,
                       ria_error *e) {
  *out = NULL;
  ria_state_page *p = NULL;
  for (uint64_t i = 0; i < s->page_count; i++)
    if (s->pages[i].id == id) {
      p = &s->pages[i];
      break;
    }
  if (!p)
    return ria_fail(e, RIA_INVALID_REQUEST, "unknown private state page ID");
  if (!valid(s, p, e) || !complete(s, p, e))
    return false;
  if (p->leases == UINT32_MAX)
    return ria_fail(e, RIA_RESOURCE_LIMIT, "state lease counter exhausted");
  if (p->device_slot < 0) {
    uint64_t slot = UINT64_MAX, oldest = UINT64_MAX;
    for (uint64_t i = 0; i < s->slot_count; i++) {
      if (s->slots[i] == UINT64_MAX) {
        slot = i;
        break;
      }
      ria_state_page *victim = &s->pages[s->slots[i]];
      if (!victim->leases && s->ages[i] < oldest) {
        oldest = s->ages[i];
        slot = i;
      }
    }
    if (slot == UINT64_MAX)
      return ria_fail(e, RIA_RESOURCE_LIMIT,
                      "all state slots have active read leases");
    if (s->slots[slot] != UINT64_MAX) {
      ria_state_page *victim = &s->pages[s->slots[slot]];
      if (!ria_state_flush(s, victim, e) ||
          !s->io.zero(s->io.context, slot, s->slot_bytes, e)) {
        s->unusable = true;
        return false;
      }
      victim->device_valid = false;
      victim->device_slot = -1;
    }
    if (!p->host_valid) {
      s->unusable = true;
      return ria_fail(e, RIA_INTERNAL_ERROR,
                      "evicted page has no host backing");
    }
    if (!s->io.upload(s->io.context, slot, p->host, p->bytes, &p->event, e)) {
      s->unusable = true;
      return false;
    }
    p->device_slot = (int64_t)slot;
    p->pending = true;
    s->slots[slot] = (uint64_t)(p - s->pages);
    if (!complete(s, p, e))
      return false;
  }
  /* Retire before tick wrap; normalize LRU ages while preserving order. */
  if (s->tick == UINT64_MAX)
    return ria_fail(e, RIA_RESOURCE_LIMIT, "state lifetime counter exhausted");
  s->ages[(uint64_t)p->device_slot] = ++s->tick;
  ++p->leases;
  *out = p;
  return true;
}
bool ria_state_release(ria_state_store *s, ria_state_page *p, ria_error *e) {
  if (!valid(s, p, e) || !p->leases)
    return ria_fail(e, RIA_INVALID_REQUEST, "unbalanced state read lease");
  --p->leases;
  return true;
}
bool ria_state_dirty(ria_state_store *s, ria_state_page *p, uint64_t begin,
                     uint64_t end, uint64_t event, ria_error *e) {
  if (!valid(s, p, e) || p->device_slot < 0 || !p->device_valid || p->pending ||
      !p->leases || begin >= end || end > p->bytes)
    return ria_fail(e, RIA_INVALID_REQUEST, "invalid state write publication");
  if (p->host_valid) {
    p->dirty_begin = begin;
    p->dirty_end = end;
  } else {
    if (begin < p->dirty_begin)
      p->dirty_begin = begin;
    if (end > p->dirty_end)
      p->dirty_end = end;
  }
  p->host_valid = false;
  p->device_valid = false;
  p->event = event;
  p->pending = true;
  return true;
}
bool ria_state_cancel(ria_state_store *s, ria_error *e) {
  bool ok = true;
  for (uint64_t i = 0; i < s->page_count; i++) {
    ria_state_page *p = &s->pages[i];
    if (p->leases)
      ok = ria_fail(e, RIA_NOT_READY,
                    "cancellation requires released graph leases");
  }
  if (!ok)
    return false;
  for (uint64_t i = 0; i < s->page_count; i++) {
    ria_state_page *p = &s->pages[i];
    if (p->pending && !s->io.wait(s->io.context, p->event, e)) {
      s->unusable = true;
      return false;
    }
    p->pending = false;
    p->cancelled = true;
    p->host_valid = p->device_valid = false;
  }
  for (uint64_t i = 0; i < s->slot_count; i++)
    if (!s->io.zero(s->io.context, i, s->slot_bytes, e)) {
      s->unusable = true;
      return false;
    }
  s->unusable = true;
  return true;
}
bool ria_state_destroy(ria_state_store *s, ria_error *e) {
  if (!s)
    return true;
  for (uint64_t i = 0; i < s->page_count; i++)
    if (s->pages[i].leases)
      return ria_fail(e, RIA_NOT_READY,
                      "cannot free state while graph holds leases");
  bool ok = true;
  for (uint64_t i = 0; i < s->page_count; i++)
    if (s->pages[i].pending && !s->io.wait(s->io.context, s->pages[i].event, e))
      ok = false;
  /* If the device reports a sticky failure, the owner must terminate the
   * process. Do not release buffers that may still be referenced by DMA. */
  if (!ok) {
    s->unusable = true;
    return false;
  }
  for (uint64_t i = 0; i < s->page_count; i++) {
    ria_state_page *p = &s->pages[i];
    if (p->host) {
      memset(p->host, 0, (size_t)p->bytes);
      if (munmap(p->host, (size_t)p->bytes))
        ok = ria_fail(e, RIA_INTERNAL_ERROR, "state munmap failed: %s",
                      strerror(errno));
    }
  }
  free(s->pages);
  free(s->slots);
  free(s->ages);
  memset(s, 0, sizeof(*s));
  return ok;
}
