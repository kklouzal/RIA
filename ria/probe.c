#define _GNU_SOURCE
#include "probe.h"
#include "runtime.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <linux/magic.h>
#include <numa.h>
#include <numaif.h>
#include <poll.h>
#include <sched.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/wait.h>
#include <unistd.h>

static bool num(const ria_json_doc *d, const char *key, uint64_t *v,
                ria_error *e) {
  return ria_json_u64(d, ria_json_get(d, 0, key), false, v, e);
}
bool ria_probe_config_read(const char *path, ria_probe_config *c,
                           ria_error *e) {
  ria_json_doc d = {0};
  memset(c, 0, sizeof(*c));
  const char *const f[] = {"schema_revision",
                           "role",
                           "executor",
                           "device_index",
                           "expected_gpu_uuid",
                           "numa_nodes",
                           "max_host_test_bytes",
                           "max_device_test_bytes",
                           "max_pinned_test_bytes",
                           "deadline_ms",
                           "disable_core_dumps",
                           "environment_digest",
                           "build_digest",
                           "build_info_file"};
  if (!ria_json_read(path, (ria_json_limits){65536, 1024, 16}, &d, e))
    return false;
  bool ok = false;
  uint64_t v;
  const char *s;
  size_t n;
  if (!ria_json_fields(&d, 0, f, 14, f, 14, e) ||
      !num(&d, "schema_revision", &v, e) || v != 1 ||
      !ria_json_string(&d, ria_json_get(&d, 0, "role"), &s, &n, e) ||
      strlen(s) != n || (strcmp(s, "client") && strcmp(s, "expert")))
    goto invalid;
  c->client = !strcmp(s, "client");
  if (!ria_json_string(&d, ria_json_get(&d, 0, "executor"), &s, &n, e) ||
      strlen(s) != n || (strcmp(s, "cpu") && strcmp(s, "cuda")))
    goto invalid;
  c->cuda = !strcmp(s, "cuda");
  if (c->client && !c->cuda)
    goto invalid;
  const ria_json_node *gpu =
      ria_json_at(&d, ria_json_get(&d, 0, "expected_gpu_uuid"));
  const ria_json_node *device =
      ria_json_at(&d, ria_json_get(&d, 0, "device_index"));
  const ria_json_node *dumps =
      ria_json_at(&d, ria_json_get(&d, 0, "disable_core_dumps"));
  if (!gpu || !device || !dumps || dumps->type != RIA_JSON_BOOL ||
      !dumps->boolean)
    goto invalid;
  if (c->cuda) {
    if (!ria_json_u64(&d, ria_json_get(&d, 0, "device_index"), false, &v, e) ||
        v || gpu->type != RIA_JSON_STRING || gpu->length != 40 ||
        memcmp(gpu->text, "GPU-", 4))
      goto invalid;
    for (size_t i = 4; i < 40; ++i) {
      char ch = gpu->text[i];
      if ((i == 12 || i == 17 || i == 22 || i == 27)
              ? ch != '-'
              : !((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f') ||
                  (ch >= 'A' && ch <= 'F')))
        goto invalid;
    }
    memcpy(c->gpu_uuid, gpu->text, 40);
  } else if (device->type != RIA_JSON_NULL || gpu->type != RIA_JSON_NULL)
    goto invalid;
  if (!num(&d, "max_host_test_bytes", &c->host_test_bytes, e) ||
      !c->host_test_bytes ||
      !num(&d, "max_device_test_bytes", &c->device_test_bytes, e) ||
      !num(&d, "max_pinned_test_bytes", &c->pinned_test_bytes, e) ||
      !num(&d, "deadline_ms", &c->deadline_ms, e) || !c->deadline_ms ||
      c->deadline_ms > INT_MAX ||
      (!c->cuda && (c->device_test_bytes || c->pinned_test_bytes)) ||
      (c->cuda && (!c->device_test_bytes || !c->pinned_test_bytes)))
    goto invalid;
  const ria_json_node *a = ria_json_at(&d, ria_json_get(&d, 0, "numa_nodes"));
  if (!a || a->type != RIA_JSON_ARRAY)
    goto invalid;
  uint64_t seen = 0;
  for (uint32_t i = a->child; i != RIA_JSON_NONE; i = d.nodes[i].next) {
    if (!ria_json_u64(&d, i, false, &v, e) || v >= 64 ||
        (seen & (UINT64_C(1) << v)) || c->node_count >= 64)
      goto invalid;
    seen |= UINT64_C(1) << v;
    c->nodes[c->node_count++] = (unsigned)v;
  }
  if (!c->node_count)
    goto invalid;
  if (!ria_json_digest_field(&d, ria_json_get(&d, 0, "environment_digest"),
                             c->environment_digest, e) ||
      !ria_json_digest_field(&d, ria_json_get(&d, 0, "build_digest"),
                             c->build_digest, e) ||
      !ria_json_string(&d, ria_json_get(&d, 0, "build_info_file"), &s, &n, e) ||
      !n || n >= sizeof(c->build_info_file) || s[0] != '/' || strlen(s) != n)
    goto invalid;
  memcpy(c->build_info_file, s, n + 1);
  ok = true;
  goto done;
invalid:
  ria_error_set(
      e, RIA_INVALID_REQUEST,
      "probe configuration violates the explicit target/budget schema");
done:
  ria_json_free(&d);
  return ok;
}

#define PROBE_OUTPUT 32768u
typedef struct {
  ria_error error;
  uint32_t report_bytes, detail_bytes;
  char bytes[PROBE_OUTPUT];
} probe_message;
static bool append(char *b, size_t cap, size_t *n, const char *format, ...)
    __attribute__((format(printf, 4, 5)));
#include <stdarg.h>
static bool append(char *b, size_t cap, size_t *n, const char *format, ...) {
  if (*n >= cap)
    return false;
  va_list args;
  va_start(args, format);
  int count = vsnprintf(b + *n, cap - *n, format, args);
  va_end(args);
  if (count < 0 || (size_t)count >= cap - *n)
    return false;
  *n += (size_t)count;
  return true;
}
static bool seal(const char *json, size_t length, char **out, size_t *out_n,
                 ria_error *e) {
  ria_json_doc d = {0};
  uint8_t hash[32];
  char hex[65];
  if (!ria_json_parse(json, length, (ria_json_limits){PROBE_OUTPUT, 4096, 24},
                      &d, e))
    return false;
  bool ok = ria_json_sha256(&d, true, hash, e);
  ria_json_free(&d);
  if (!ok)
    return false;
  ria_hex_encode(hash, 32, hex);
  char *sealed = malloc(length + 80);
  if (!sealed)
    return ria_fail(e, RIA_RESOURCE_LIMIT, "probe report allocation failed");
  memcpy(sealed, json, length - 1);
  int written = snprintf(sealed + length - 1, 81, ",\"digest\":\"%s\"}", hex);
  ok = written == 77 &&
       ria_json_parse(sealed, length - 1 + (size_t)written,
                      (ria_json_limits){PROBE_OUTPUT, 4096, 24}, &d, e);
  if (ok)
    ok = ria_json_canonical(&d, false, out, out_n, e);
  ria_json_free(&d);
  free(sealed);
  return ok;
}

static bool probe_body(const ria_probe_config *c, probe_message *m) {
  ria_error *e = &m->error;
#if !defined(__x86_64__)
  (void)c;
  return ria_fail(e, RIA_UNSUPPORTED,
                  "physical release probe requires native Linux x86-64");
#endif
  uint64_t start = ria_monotonic_ms();
  if (!ria_disable_dumps(e))
    return false;
  ria_json_doc build = {0};
  uint8_t build_hash[32], build_claimed[32];
  bool build_valid =
      ria_json_read(c->build_info_file, (ria_json_limits){1048576, 32768, 32},
                    &build, e) &&
      ria_json_digest_field(&build, ria_json_get(&build, 0, "digest"),
                            build_claimed, e) &&
      ria_json_sha256(&build, true, build_hash, e) &&
      !memcmp(build_hash, build_claimed, 32) &&
      !memcmp(build_hash, c->build_digest, 32);
  ria_json_free(&build);
  if (!build_valid)
    return e->code ? false
                   : ria_fail(e, RIA_IDENTITY_MISMATCH,
                              "actual native build differs from probe grant");
  ria_runtime_observation observation;
  if (!ria_runtime_inspect(&observation, e))
    return false;
  uint64_t available = observation.host_available,
           limit = observation.host_limit;
  uint64_t sysfree = observation.host_available,
           locked = observation.locked_bytes;
  char *cpu_mask = observation.cpu_mask,
       *mem_mask = observation.memory_node_mask;
  uint64_t lock_limit = observation.memlock;
  if (sysfree < available)
    available = sysfree;
  uint64_t lockfree = lock_limit > locked ? lock_limit - locked : 0;
  if (lockfree < available)
    available = lockfree;
  if (available > RIA_JSON_SAFE_INTEGER || !available ||
      c->host_test_bytes > available)
    return ria_fail(e, RIA_RESOURCE_LIMIT,
                    "bounded host probe exceeds current memory/lock headroom");
  if (numa_available() < 0)
    return ria_fail(e, RIA_UNSUPPORTED, "NUMA self-allocation is unavailable");
  struct bitmask *allowed = numa_get_mems_allowed();
  if (!allowed)
    return ria_fail(e, RIA_NOT_READY, "effective NUMA mask is unavailable");
  bool ok = true;
  uint64_t node_bytes[64] = {0}, total = 0;
  long pg = sysconf(_SC_PAGESIZE);
  if (pg <= 0 || c->host_test_bytes / (uint64_t)pg < c->node_count)
    ok = false;
  for (unsigned i = 0; ok && i < c->node_count; ++i) {
    unsigned node = c->nodes[i];
    char path[128];
    if (!numa_bitmask_isbitset(allowed, node)) {
      ok = ria_fail(e, RIA_RESOURCE_LIMIT,
                    "requested NUMA node is outside effective mask");
      break;
    }
    snprintf(path, sizeof(path), "/sys/devices/system/node/node%u/meminfo",
             node);
    if (!ria_runtime_memory_counter(path, "MemFree:", &node_bytes[node], e)) {
      ok = false;
      break;
    }
    if (node_bytes[node] > available)
      node_bytes[node] = available;
    if (!ria_u64_add(total, node_bytes[node], &total)) {
      ok = false;
      break;
    }
    uint64_t test =
        c->host_test_bytes / c->node_count / (uint64_t)pg * (uint64_t)pg;
    size_t size;
    if (test > node_bytes[node] || !ria_size(test, &size, e)) {
      ok = false;
      break;
    }
    /* Anonymous mmap ignores fd; Cppcheck's POSIX fd range is too narrow. */
    void *arena = mmap(NULL, size, PROT_READ | PROT_WRITE,
                       // cppcheck-suppress invalidFunctionArg
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (arena == MAP_FAILED) {
      ok = ria_fail(e, RIA_RESOURCE_LIMIT, "bounded NUMA allocation failed");
      break;
    }
    if (!arena) {
      /* Linux munmap accepts address zero; its POSIX model assumes nonnull. */
      // cppcheck-suppress [nullPointer,nullPointerRedundantCheck]
      bool released = munmap(arena, size) == 0;
      ok = ria_fail(e, RIA_RESOURCE_LIMIT, released ? "NUMA allocation mapped the null address" : "null NUMA mapping cleanup failed");
      break;
    }
    unsigned long mask = 1ul << node;
    bool allocated = madvise(arena, size, MADV_DONTDUMP) == 0 &&
                     mbind(arena, size, MPOL_BIND, &mask, 64, 0) == 0;
    if (allocated) {
      for (size_t p = 0; p < size; p += (size_t)pg)
        ((volatile unsigned char *)arena)[p] = (unsigned char)(p / (size_t)pg);
      allocated = mlock(arena, size) == 0;
    }
    if (allocated) {
      void *pages[64];
      int status[64];
      unsigned samples =
          size / (size_t)pg < 64 ? (unsigned)(size / (size_t)pg) : 64;
      for (unsigned j = 0; j < samples; ++j)
        pages[j] =
            (char *)arena + (j * (size / (size_t)pg) / samples) * (size_t)pg;
      allocated = move_pages(0, samples, pages, NULL, status, 0) == 0;
      for (unsigned j = 0; allocated && j < samples; ++j)
        allocated = status[j] == (int)node;
    }
    int saved = errno;
    if (munlock(arena, size) && allocated) {
      allocated = false;
      saved = errno;
    }
    if (munmap(arena, size) && allocated) {
      allocated = false;
      saved = errno;
    }
    if (!allocated)
      ok = ria_fail(e, RIA_NOT_READY,
                    "NUMA bind/lock/locality proof failed on node %u: %s", node,
                    strerror(saved));
  }
  numa_bitmask_free(allowed);
  if (!ok) {
    if (!e->code)
      ria_error_set(e, RIA_RESOURCE_LIMIT, "NUMA probe budget is invalid");
    return false;
  }
  if (total < available)
    available = total;
  if (c->pinned_test_bytes > lockfree - c->host_test_bytes)
    return ria_fail(e, RIA_RESOURCE_LIMIT,
                    "combined host and CUDA pin test exceeds memlock budget");
  uint64_t host_elapsed = ria_monotonic_ms() - start;
  ria_probe_gpu_result gpu = {0};
  if (c->cuda && !ria_probe_gpu(c, &gpu, e))
    return false;
  char compact[16384], detail[16384];
  size_t n = 0, dn = 0;
  char environment_hex[65], build_hex[65], evidence_hex[65];
  ria_hex_encode(c->environment_digest, 32, environment_hex);
  ria_hex_encode(c->build_digest, 32, build_hex);
  /* Masks are kernel-produced decimal/range syntax, validate before JSON use.
   */
  if (!*cpu_mask || !*mem_mask ||
      strspn(cpu_mask, "0123456789,-") != strlen(cpu_mask) ||
      strspn(mem_mask, "0123456789,-") != strlen(mem_mask))
    ok = false;
  ok =
      ok &&
      append(
          detail, sizeof(detail), &dn,
          "{\"schema_revision\":1,\"kind\":\"probe_evidence\","
          "\"role\":\"%s\",\"executor\":\"%s\","
          "\"environment_digest\":\"%s\",\"build_digest\":\"%s\","
          "\"architecture\":"
          "\"x86_64\",\"uid\":10001,\"dumpable\":false,\"seccomp_mode\":2,\"no_"
          "new_privileges\":true,\"cgroup_limit_bytes\":\"%llu\",\"cgroup_"
          "available_bytes\":\"%llu\",\"swap_limit_bytes\":\"0\",\"memlock_"
          "bytes\":\"%llu\",\"cpu_mask\":\"%s\",\"memory_node_mask\":\"%s\","
          "\"host_test_bytes\":\"%llu\",\"host_test_ms\":\"%llu\",\"numa_"
          "locality_proven\":true,\"driver_version\":%d,\"runtime_version\":%d,"
          "\"gpu_uuid\":%s,\"compute_major\":%d,\"compute_minor\":%d,\"gpu_"
          "allocation_ms\":\"%llu\",\"native_kernel_ms\":\"%llu\",\"native_"
          "results\":[%g,%g,%g],\"host_parent_preflight_required\":true,"
          "\"host_available_bytes\":\"%llu\",\"device_available_bytes\":\"%"
          "llu\","
          "\"pinned_test_bytes\":\"%llu\",\"numa_available\":[",
          c->client ? "client" : "expert", c->cuda ? "cuda" : "cpu",
          environment_hex, build_hex, (unsigned long long)limit,
          (unsigned long long)observation.host_available,
          (unsigned long long)lock_limit, cpu_mask, mem_mask,
          (unsigned long long)c->host_test_bytes,
          (unsigned long long)host_elapsed, gpu.driver_version,
          gpu.runtime_version, c->cuda ? "\"GPU_PLACEHOLDER\"" : "null",
          gpu.major, gpu.minor, (unsigned long long)gpu.allocation_ms,
          (unsigned long long)gpu.kernel_ms, (double)gpu.native_results[0],
          (double)gpu.native_results[1], (double)gpu.native_results[2],
          (unsigned long long)available, (unsigned long long)gpu.device_bytes,
          (unsigned long long)gpu.pinned_bytes);
  for (unsigned i = 0; ok && i < c->node_count; ++i)
    ok = append(detail, sizeof(detail), &dn,
                "%s{\"node\":%u,\"bytes\":\"%llu\"}", i ? "," : "", c->nodes[i],
                (unsigned long long)node_bytes[c->nodes[i]]);
  ok = ok && append(detail, sizeof(detail), &dn, "]}");
  if (ok && c->cuda) {
    char *placeholder = strstr(detail, "GPU_PLACEHOLDER");
    size_t old = strlen("GPU_PLACEHOLDER"), uuid_n = strlen(gpu.uuid);
    if (!placeholder || dn + uuid_n - old >= sizeof(detail))
      ok = false;
    else {
      size_t offset = (size_t)(placeholder - detail);
      memmove(detail + offset + uuid_n, detail + offset + old,
              dn - offset - old + 1);
      memcpy(detail + offset, gpu.uuid, uuid_n);
      dn += uuid_n - old;
    }
  }
  char *sealed_detail = NULL, *sealed = NULL;
  size_t sd = 0, sn = 0;
  if (ok)
    ok = seal(detail, dn, &sealed_detail, &sd, e);
  ria_json_doc evidence = {0};
  uint8_t evidence_hash[32];
  if (ok)
    ok = ria_json_parse(sealed_detail, sd, (ria_json_limits){16384, 2048, 24},
                        &evidence, e) &&
         ria_json_digest_field(&evidence, ria_json_get(&evidence, 0, "digest"),
                               evidence_hash, e);
  ria_json_free(&evidence);
  if (ok) {
    ria_hex_encode(evidence_hash, 32, evidence_hex);
    ok = append(
        compact, sizeof(compact), &n,
        "{\"schema_revision\":1,\"role\":\"%s\",\"executor\":\"%s\","
        "\"host_bytes\":%llu,\"device_bytes\":%llu,\"pinned_bytes\":%llu,"
        "\"qualified\":true,\"environment_digest\":\"%s\","
        "\"build_digest\":\"%s\",\"evidence_digest\":\"%s\",\"numa\":[",
        c->client ? "client" : "expert", c->cuda ? "cuda" : "cpu",
        (unsigned long long)available, (unsigned long long)gpu.device_bytes,
        (unsigned long long)gpu.pinned_bytes, environment_hex, build_hex,
        evidence_hex);
    for (unsigned i = 0; ok && i < c->node_count; ++i)
      ok = append(compact, sizeof(compact), &n,
                  "%s{\"node\":%u,\"bytes\":%llu}", i ? "," : "", c->nodes[i],
                  (unsigned long long)node_bytes[c->nodes[i]]);
    ok = ok && append(compact, sizeof(compact), &n, "]}");
  }
  if (ok)
    ok = seal(compact, n, &sealed, &sn, e);
  if (ok && sn + sd <= sizeof(m->bytes)) {
    memcpy(m->bytes, sealed, sn);
    memcpy(m->bytes + sn, sealed_detail, sd);
    m->report_bytes = (uint32_t)sn;
    m->detail_bytes = (uint32_t)sd;
  } else if (ok)
    ok = ria_fail(e, RIA_RESOURCE_LIMIT, "probe evidence exceeds IPC limit");
  free(sealed_detail);
  free(sealed);
  return ok;
}

#ifndef RIA_WITH_CUDA
bool ria_probe_gpu(const ria_probe_config *c, ria_probe_gpu_result *r,
                   ria_error *e) {
  (void)c;
  (void)r;
  return ria_fail(e, RIA_UNSUPPORTED, "this CPU image cannot initialize CUDA");
}
#endif
bool ria_probe_run(const ria_probe_config *c, char **report, size_t *rn,
                   char **details, size_t *dn, ria_error *e) {
  *report = NULL;
  *details = NULL;
  *rn = 0;
  *dn = 0;
  int fds[2];
  if (pipe2(fds, O_CLOEXEC))
    return ria_fail(e, RIA_INTERNAL_ERROR, "probe supervision pipe failed");
  pid_t pid = fork();
  if (pid < 0) {
    close(fds[0]);
    close(fds[1]);
    return ria_fail(e, RIA_INTERNAL_ERROR, "probe child creation failed");
  }
  if (!pid) {
    close(fds[0]);
    probe_message message = {0};
    bool ok = probe_body(c, &message);
    if (!ok && !message.error.code)
      ria_error_set(&message.error, RIA_INTERNAL_ERROR,
                    "probe failed without context");
    size_t done = 0;
    while (done < sizeof(message)) {
      ssize_t count =
          write(fds[1], (char *)&message + done, sizeof(message) - done);
      if (count < 0 && errno == EINTR)
        continue;
      if (count <= 0)
        _exit(RIA_INTERNAL_ERROR);
      done += (size_t)count;
    }
    close(fds[1]);
    _exit(ok ? 0 : message.error.code);
  }
  close(fds[1]);
  int flags = fcntl(fds[0], F_GETFL);
  if (flags < 0 || fcntl(fds[0], F_SETFL, flags | O_NONBLOCK) < 0) {
    kill(pid, SIGKILL);
    waitpid(pid, NULL, 0);
    close(fds[0]);
    return ria_fail(e, RIA_INTERNAL_ERROR,
                    "probe supervision initialization failed");
  }
  uint64_t now = ria_monotonic_ms(), deadline;
  bool ok = now != UINT64_MAX && ria_u64_add(now, c->deadline_ms, &deadline);
  probe_message message = {0};
  size_t received = 0;
  int status = 0;
  while (ok && received < sizeof(message)) {
    now = ria_monotonic_ms();
    if (now == UINT64_MAX || now >= deadline) {
      ok = ria_fail(e, RIA_DEADLINE_EXCEEDED,
                    "bounded physical probe deadline exceeded");
      break;
    }
    struct pollfd p = {fds[0], POLLIN, 0};
    int polled = poll(&p, 1, (int)(deadline - now));
    if (polled < 0 && errno == EINTR)
      continue;
    if (polled <= 0) {
      ok = ria_fail(e, polled ? RIA_INTERNAL_ERROR : RIA_DEADLINE_EXCEEDED,
                    "probe supervision wait failed");
      break;
    }
    ssize_t count =
        read(fds[0], (char *)&message + received, sizeof(message) - received);
    if (count < 0 && (errno == EAGAIN || errno == EINTR))
      continue;
    if (count <= 0) {
      ok = ria_fail(
          e, RIA_EXECUTOR_ERROR,
          "probe child terminated before publishing complete evidence");
      break;
    }
    received += (size_t)count;
  }
  close(fds[0]);
  if (!ok)
    kill(pid, SIGKILL);
  while (waitpid(pid, &status, 0) < 0) {
    if (errno != EINTR)
      return ria_fail(e, RIA_INTERNAL_ERROR, "cannot reap probe child");
  }
  if (!ok)
    return false;
  if (!WIFEXITED(status) || WEXITSTATUS(status) || message.error.code) {
    *e = message.error;
    if (!e->code)
      ria_error_set(e, RIA_EXECUTOR_ERROR, "physical probe child failed");
    return false;
  }
  if (!message.report_bytes || !message.detail_bytes ||
      (uint64_t)message.report_bytes + message.detail_bytes >
          sizeof(message.bytes))
    return ria_fail(e, RIA_INTERNAL_ERROR,
                    "invalid probe child evidence bounds");
  *report = malloc((size_t)message.report_bytes + 1);
  *details = malloc((size_t)message.detail_bytes + 1);
  if (!*report || !*details) {
    free(*report);
    free(*details);
    *report = NULL;
    *details = NULL;
    return ria_fail(e, RIA_RESOURCE_LIMIT,
                    "probe publication allocation failed");
  }
  memcpy(*report, message.bytes, message.report_bytes);
  (*report)[message.report_bytes] = 0;
  memcpy(*details, message.bytes + message.report_bytes, message.detail_bytes);
  (*details)[message.detail_bytes] = 0;
  *rn = message.report_bytes;
  *dn = message.detail_bytes;
  return true;
}
bool ria_report_write(const char *path, const void *bytes, size_t n,
                      ria_error *e) {
  if (!path || !*path || strlen(path) > PATH_MAX - 16)
    return ria_fail(e, RIA_INVALID_REQUEST, "invalid report destination");
  char temp[PATH_MAX], parent[PATH_MAX];
  snprintf(temp, sizeof(temp), "%s.tmp.XXXXXX", path);
  int fd = mkstemp(temp);
  if (fd < 0)
    return ria_fail(e, RIA_RESOURCE_LIMIT,
                    "cannot create protected report temporary file");
  bool ok = fchmod(fd, 0600) == 0;
  size_t done = 0;
  while (ok && done < n) {
    ssize_t count = write(fd, (const char *)bytes + done, n - done);
    if (count < 0 && errno == EINTR)
      continue;
    if (count <= 0)
      ok = false;
    else
      done += (size_t)count;
  }
  if (ok)
    ok = fsync(fd) == 0;
  if (close(fd))
    ok = false;
  if (ok)
    ok = rename(temp, path) == 0;
  if (ok) {
    strcpy(parent, path);
    char *slash = strrchr(parent, '/');
    if (!slash)
      strcpy(parent, ".");
    else if (slash == parent)
      slash[1] = 0;
    else
      *slash = 0;
    int dir = open(parent, O_DIRECTORY | O_RDONLY | O_CLOEXEC);
    if (dir < 0)
      ok = false;
    else {
      if (fsync(dir))
        ok = false;
      if (close(dir))
        ok = false;
    }
  }
  if (!ok) {
    int saved = errno;
    unlink(temp);
    return ria_fail(e, RIA_INTEGRITY_ERROR, "report publication incomplete: %s",
                    strerror(saved));
  }
  return true;
}
