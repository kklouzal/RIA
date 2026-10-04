#include "ria/state.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x)                                                               \
  do {                                                                         \
    if (!(x)) {                                                                \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x);                  \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
typedef struct {
  uint8_t device[2][64];
  uint64_t events, waits, uploads, downloads, zeroes;
  bool fail_wait;
} fake;
static bool upload(void *c, uint64_t slot, const void *p, uint64_t bytes,
                   uint64_t *event, ria_error *e) {
  fake *f = c;
  (void)e;
  CHECK(slot < 2 && bytes <= 64);
  memcpy(f->device[slot], p, (size_t)bytes);
  ++f->uploads;
  *event = ++f->events;
  return true;
}
static bool download(void *c, uint64_t slot, void *p, uint64_t bytes,
                     uint64_t *event, ria_error *e) {
  fake *f = c;
  (void)e;
  CHECK(slot < 2 && bytes <= 64);
  memcpy(p, f->device[slot], (size_t)bytes);
  ++f->downloads;
  *event = ++f->events;
  return true;
}
static bool wait_event(void *c, uint64_t event, ria_error *e) {
  fake *f = c;
  CHECK(event && event <= f->events);
  ++f->waits;
  if (f->fail_wait) {
    f->fail_wait = false;
    return ria_fail(e, RIA_EXECUTOR_ERROR, "fixture completion failure");
  }
  return true;
}
static bool zero(void *c, uint64_t slot, uint64_t bytes, ria_error *e) {
  fake *f = c;
  (void)e;
  CHECK(slot < 2 && bytes == 64);
  memset(f->device[slot], 0, (size_t)bytes);
  ++f->zeroes;
  return true;
}
static ria_state_io io(fake *f) {
  return (ria_state_io){f, upload, download, wait_event, zero};
}
int main(void) {
  ria_error e = {0};
  fake f = {0};
  ria_state_store s;
  ria_state_page *a, *b, *c, *p;
  CHECK(ria_state_create(&s, 3, 96, 1, 64, io(&f), &e));
  CHECK(!ria_state_add(&s, 1, 7, 0, 1, 32, &a, &e));
  CHECK(ria_state_begin(&s, 3, 9, &e));
  CHECK(ria_state_add(&s, 1, 7, 0, 2, 32, &a, &e));
  CHECK(ria_state_add(&s, 2, 7, 2, 2, 32, &b, &e));
  CHECK(ria_state_add(&s, 3, 7, 4, 2, 32, &c, &e));
  CHECK(!ria_state_add(&s, 4, 7, 6, 2, 32, &p, &e));
  CHECK(!ria_state_add(&s, 1, 7, 0, 2, 32, &p, &e));
  for (unsigned i = 0; i < 32; i++)
    a->host[i] = (uint8_t)i;
  CHECK(ria_state_acquire(&s, 1, &p, &e) && p == a && a->device_valid &&
        a->leases == 1);
  CHECK(f.waits == 1 && f.uploads == 1 && !memcmp(f.device[0], a->host, 32));
  CHECK(!ria_state_acquire(&s, 2, &p, &e));
  CHECK(!ria_state_destroy(&s, &e));
  CHECK(!ria_state_dirty(&s, a, 31, 33, ++f.events, &e));
  f.device[0][5] = 211;
  CHECK(ria_state_dirty(&s, a, 5, 6, ++f.events, &e) && !a->host_valid &&
        !a->device_valid && a->pending);
  CHECK(ria_state_release(&s, a, &e));
  CHECK(!ria_state_release(&s, a, &e));
  CHECK(ria_state_acquire(&s, 2, &p, &e) && p == b);
  CHECK(a->host_valid && a->host[5] == 211 && a->device_slot == -1 &&
        f.downloads == 1 && f.zeroes == 1);
  CHECK(ria_state_release(&s, b, &e));
  CHECK(ria_state_acquire(&s, 1, &p, &e) && f.device[0][5] == 211);
  CHECK(ria_state_release(&s, a, &e));
  ria_state_page alien = {0};
  CHECK(!ria_state_release(&s, &alien, &e));
  CHECK(ria_state_cancel(&s, &e) && s.unusable && a->cancelled &&
        !a->host_valid && !a->device_valid);
  CHECK(!ria_state_acquire(&s, 1, &p, &e));
  CHECK(ria_state_destroy(&s, &e));
  memset(&f, 0, sizeof(f));
  CHECK(ria_state_create(&s, 1, 32, 1, 64, io(&f), &e));
  CHECK(ria_state_begin(&s, 4, 1, &e) &&
        ria_state_add(&s, 4, 2, UINT64_MAX - 1, 1, 32, &a, &e));
  f.fail_wait = true;
  CHECK(!ria_state_acquire(&s, 4, &p, &e) && s.unusable && !a->device_valid);
  CHECK(ria_state_destroy(&s, &e));
  puts("RIA exact-state fixtures passed");
  return 0;
}
