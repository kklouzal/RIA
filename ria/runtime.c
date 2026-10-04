#define _GNU_SOURCE
#include "runtime.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <linux/magic.h>
#include <sched.h>
#include <stdio.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/statfs.h>
#include <unistd.h>
static bool read_small(const char *path, char *out, size_t cap, ria_error *e) {
  int fd = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0)
    return ria_fail(e, RIA_NOT_READY, "cannot inspect %s: %s", path,
                    strerror(errno));
  size_t n = 0;
  bool ok = true;
  while (n < cap - 1) {
    ssize_t got = read(fd, out + n, cap - 1 - n);
    if (got < 0 && errno == EINTR)
      continue;
    if (got < 0) {
      ok = false;
      break;
    }
    if (!got)
      break;
    n += (size_t)got;
  }
  char extra;
  if (ok && n == cap - 1 && read(fd, &extra, 1) != 0)
    ok = false;
  if (close(fd))
    ok = false;
  if (!ok)
    return ria_fail(e, RIA_NOT_READY, "bounded system observation failed: %s",
                    path);
  out[n] = 0;
  return true;
}
static bool scalar_file(const char *path, uint64_t *v, ria_error *e) {
  char b[64];
  if (!read_small(path, b, sizeof(b), e))
    return false;
  size_t n = strcspn(b, "\r\n");
  if (n == 3 && !memcmp(b, "max", 3)) {
    *v = UINT64_MAX;
    return true;
  }
  return ria_parse_u64(b, n, v, e);
}
static bool cgroup_limits(uint64_t *available, uint64_t *limit, uint64_t *swap,
                          char cpu_mask[4096], char mem_mask[4096],
                          ria_error *e) {
  struct statfs fs;
  if (statfs("/sys/fs/cgroup", &fs) || fs.f_type != CGROUP2_SUPER_MAGIC)
    return ria_fail(e, RIA_UNSUPPORTED, "probe requires cgroup v2");
  char b[8192];
  if (!read_small("/proc/self/cgroup", b, sizeof(b), e))
    return false;
  char *line = NULL;
  for (char *p = b; p && *p; p = strchr(p, '\n')) {
    if (*p == '\n')
      ++p;
    if (!strncmp(p, "0::/", 4)) {
      line = p + 3;
      break;
    }
  }
  if (!line)
    return ria_fail(e, RIA_NOT_READY, "unified cgroup membership is absent");
  size_t n = strcspn(line, "\r\n");
  line[n] = 0;
  if (n >= PATH_MAX - 32 || strstr(line, "/../") || strstr(line, "/./") ||
      (n >= 3 && !strcmp(line + n - 3, "/..")))
    return ria_fail(e, RIA_NOT_READY,
                    "unresolvable cgroup namespace; host preflight required");
  char dir[PATH_MAX], path[PATH_MAX];
  int len = snprintf(dir, sizeof(dir), "/sys/fs/cgroup%s", line);
  if (len < 0 || (size_t)len >= sizeof(dir))
    return ria_fail(e, RIA_RESOURCE_LIMIT, "cgroup path too long");
  if (n == 1)
    dir[strlen(dir) - 1] = 0;
  *available = UINT64_MAX;
  *limit = UINT64_MAX;
  *swap = UINT64_MAX;
  const size_t root = strlen("/sys/fs/cgroup");
  bool first = true;
  for (;;) {
    uint64_t cap, used, sw;
    if (snprintf(path, sizeof(path), "%s/memory.max", dir) >=
            (int)sizeof(path) ||
        !scalar_file(path, &cap, e))
      return false;
    if (snprintf(path, sizeof(path), "%s/memory.current", dir) >=
            (int)sizeof(path) ||
        !scalar_file(path, &used, e))
      return false;
    if (snprintf(path, sizeof(path), "%s/memory.swap.max", dir) >=
            (int)sizeof(path) ||
        !scalar_file(path, &sw, e))
      return false;
    if (cap < *limit)
      *limit = cap;
    if (sw < *swap)
      *swap = sw;
    uint64_t free_bytes = cap == UINT64_MAX ? UINT64_MAX
                          : cap > used      ? cap - used
                                            : 0;
    if (free_bytes < *available)
      *available = free_bytes;
    if (first) {
      if (snprintf(path, sizeof(path), "%s/cpuset.cpus.effective", dir) >=
              (int)sizeof(path) ||
          !read_small(path, cpu_mask, 4096, e) ||
          snprintf(path, sizeof(path), "%s/cpuset.mems.effective", dir) >=
              (int)sizeof(path) ||
          !read_small(path, mem_mask, 4096, e))
        return false;
      cpu_mask[strcspn(cpu_mask, "\r\n")] = 0;
      mem_mask[strcspn(mem_mask, "\r\n")] = 0;
      first = false;
    }
    if (strlen(dir) == root)
      break;
    char *last = strrchr(dir, '/');
    if (!last || (size_t)(last - dir) < root)
      return ria_fail(e, RIA_NOT_READY, "cgroup ancestor resolution failed");
    *last = 0;
  }
  if (!*available || *limit == UINT64_MAX || *swap)
    return ria_fail(
        e, RIA_RESOURCE_LIMIT,
        "probe requires finite effective memory and zero effective swap");
  return true;
}
bool ria_runtime_memory_counter(const char *path, const char *key,
                                uint64_t *bytes, ria_error *e) {
  char b[16384];
  if (!read_small(path, b, sizeof(b), e))
    return false;
  const char *p = strstr(b, key);
  if (!p)
    return ria_fail(e, RIA_NOT_READY,
                    "required memory counter %s is unavailable", key);
  p += strlen(key);
  while (*p == ' ' || *p == '\t')
    ++p;
  size_t n = strspn(p, "0123456789");
  uint64_t kib;
  if (!ria_parse_u64(p, n, &kib, e) || strncmp(p + n, " kB", 3) ||
      !ria_u64_mul(kib, 1024, bytes))
    return ria_fail(e, RIA_NOT_READY, "invalid memory counter %s", key);
  return true;
}

bool ria_runtime_inspect(ria_runtime_observation *o, ria_error *e) {
  if (!o)
    return ria_fail(e, RIA_INVALID_REQUEST,
                    "runtime observation output is required");
  memset(o, 0, sizeof(*o));
  struct rlimit core, lock;
  if (getuid() != 10001 || geteuid() != 10001 || prctl(PR_GET_DUMPABLE) != 0 ||
      prctl(PR_GET_NO_NEW_PRIVS, 0, 0, 0, 0) != 1 ||
      prctl(PR_GET_SECCOMP, 0, 0, 0, 0) != 2)
    return ria_fail(e, RIA_UNAUTHORIZED,
                    "runtime requires UID10001, disabled dumps, "
                    "no-new-privileges and the reviewed seccomp policy");
  if (getrlimit(RLIMIT_CORE, &core) || core.rlim_cur || core.rlim_max ||
      getrlimit(RLIMIT_MEMLOCK, &lock) || lock.rlim_cur == RLIM_INFINITY ||
      lock.rlim_max == RLIM_INFINITY || !lock.rlim_cur)
    return ria_fail(
        e, RIA_RESOURCE_LIMIT,
        "runtime requires finite positive memlock and zero core limits");
  o->memlock = (uint64_t)lock.rlim_cur;
  cpu_set_t cpus;
  CPU_ZERO(&cpus);
  if (sched_getaffinity(0, sizeof(cpus), &cpus) || !CPU_COUNT(&cpus))
    return ria_fail(e, RIA_NOT_READY, "effective CPU affinity is unavailable");
  o->cpu_count = (unsigned)CPU_COUNT(&cpus);
  uint64_t available;
  if (!cgroup_limits(&o->host_available, &o->host_limit, &o->swap_limit,
                     o->cpu_mask, o->memory_node_mask, e) ||
      !ria_runtime_memory_counter("/proc/meminfo", "MemAvailable:", &available,
                                  e) ||
      !ria_runtime_memory_counter("/proc/self/status",
                                  "VmLck:", &o->locked_bytes, e))
    return false;
  if (available < o->host_available)
    o->host_available = available;
  return true;
}
bool ria_runtime_require(uint64_t cap, uint64_t locked, uint64_t pinned,
                         ria_runtime_observation *o, ria_error *e) {
#if !defined(__x86_64__)
  (void)cap;
  (void)locked;
  (void)pinned;
  (void)o;
  return ria_fail(e, RIA_UNSUPPORTED,
                  "production runtime requires native Linux x86-64");
#endif
  uint64_t lock_budget;
  if (!ria_runtime_inspect(o, e))
    return false;
  if (!cap || cap > o->host_limit ||
      !ria_u64_add(locked, pinned, &lock_budget) || lock_budget > o->memlock ||
      lock_budget > cap)
    return ria_fail(e, RIA_RESOURCE_LIMIT,
                    "actual cgroup/memlock restrictions do not admit the "
                    "declared population and pinned budget");
  uint64_t additional =
      lock_budget > o->locked_bytes ? lock_budget - o->locked_bytes : 0;
  if (additional > o->host_available)
    return ria_fail(e, RIA_RESOURCE_LIMIT,
                    "current host/cgroup headroom cannot populate the "
                    "remaining locked and pinned reservation");
  return true;
}
