#define _GNU_SOURCE
#include "admin.h"
#include "json.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <poll.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

struct ria_admin {
  int listener, wake, done;
  char path[sizeof(((struct sockaddr_un *)0)->sun_path)];
  uid_t uid;
  uint64_t timeout_ms;
  ria_admin_callbacks callbacks;
  pthread_t thread;
  ria_error thread_error;
  dev_t socket_device;
  ino_t socket_inode;
};
uint64_t ria_admin_reserved_bytes(void) {
  /* The native target uses 4KiB pages. The 64KiB parser/allocator allowance
   * exceeds the fixed 255-byte/16-node command limit and thread TLS metadata.
   */
  return sizeof(ria_admin) + RIA_ADMIN_STACK_BYTES + 4096 + 65536;
}
static bool deadline_after(uint64_t delay, uint64_t *deadline, ria_error *e) {
  uint64_t now = ria_monotonic_ms();
  return now != UINT64_MAX && delay && delay <= INT_MAX &&
                 ria_u64_add(now, delay, deadline)
             ? true
             : ria_fail(e, RIA_INVALID_REQUEST,
                        "administrative deadline is invalid");
}
static int wait_fd(ria_admin *a, int fd, short events, uint64_t deadline) {
  for (;;) {
    uint64_t now = ria_monotonic_ms();
    if (now == UINT64_MAX || now >= deadline)
      return 0;
    struct pollfd p[2] = {{fd, events, 0}, {a->wake, POLLIN, 0}};
    int n = poll(p, 2, (int)(deadline - now));
    if (n < 0 && errno == EINTR)
      continue;
    if (n <= 0)
      return n;
    if (p[1].revents)
      return -2;
    if (p[0].revents & events)
      return 1;
    if (p[0].revents & (POLLERR | POLLHUP | POLLNVAL))
      return -1;
  }
}
static bool reply(ria_admin *a, int fd, uint64_t deadline, const char *text,
                  size_t n) {
  size_t sent = 0;
  while (sent < n) {
    ssize_t got = send(fd, text + sent, n - sent, MSG_NOSIGNAL);
    if (got < 0 && errno == EINTR)
      continue;
    if (got < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      if (wait_fd(a, fd, POLLOUT, deadline) != 1)
        return false;
      continue;
    }
    if (got <= 0)
      return false;
    sent += (size_t)got;
  }
  return true;
}
static void request(ria_admin *a, int fd) {
  struct ucred peer;
  socklen_t size = sizeof(peer);
  if (getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &peer, &size) ||
      size != sizeof(peer) || peer.uid != a->uid)
    return;
  ria_error e = {0};
  uint64_t deadline;
  if (!deadline_after(a->timeout_ms, &deadline, &e))
    return;
  char bytes[256];
  size_t n = 0;
  bool complete = false;
  while (n < sizeof(bytes)) {
    ssize_t count = recv(fd, bytes + n, sizeof(bytes) - n, 0);
    if (count < 0 && errno == EINTR)
      continue;
    if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      if (wait_fd(a, fd, POLLIN, deadline) != 1)
        return;
      continue;
    }
    if (count <= 0)
      return;
    n += (size_t)count;
    const char *newline = memchr(bytes, '\n', n);
    if (newline) {
      /* One command per connection. Coalesced extra data is rejected. */
      if ((size_t)(newline - bytes) + 1 != n)
        return;
      --n;
      complete = true;
      break;
    }
  }
  if (!complete)
    return;
  const char *const fields[] = {"command", "schema_revision"};
  ria_json_doc d = {0};
  const char *command = NULL;
  size_t length;
  uint64_t revision;
  bool ok = ria_json_parse(bytes, n, (ria_json_limits){255, 16, 4}, &d, &e) &&
            ria_json_fields(&d, 0, fields, 2, fields, 2, &e) &&
            ria_json_u64(&d, ria_json_get(&d, 0, "schema_revision"), false,
                         &revision, &e) &&
            revision == 1 &&
            ria_json_string(&d, ria_json_get(&d, 0, "command"), &command,
                            &length, &e) &&
            strlen(command) == length;
  bool ready = false, active = false;
  if (ok && !strcmp(command, "health"))
    ok = a->callbacks.health(a->callbacks.context, &ready, &active, &e);
  else if (ok && !strcmp(command, "drain"))
    ok = a->callbacks.drain(a->callbacks.context, deadline, &e);
  else if (ok)
    ok =
        ria_fail(&e, RIA_INVALID_REQUEST, "unsupported administrative command");
  char output[128];
  int written;
  if (ok)
    written = snprintf(output, sizeof(output),
                       "{\"ok\":true,\"ready\":%s,\"active\":%s}\n",
                       ready ? "true" : "false", active ? "true" : "false");
  else
    written = snprintf(output, sizeof(output), "{\"ok\":false,\"code\":%d}\n",
                       e.code ? e.code : RIA_INTERNAL_ERROR);
  if (written > 0 && (size_t)written < sizeof(output))
    (void)reply(a, fd, deadline, output, (size_t)written);
  ria_json_free(&d);
}
static void *owner(void *context) {
  ria_admin *a = context;
  for (;;) {
    struct pollfd f[2] = {{a->listener, POLLIN, 0}, {a->wake, POLLIN, 0}};
    int n = poll(f, 2, -1);
    if (n < 0 && errno == EINTR)
      continue;
    if (n < 0) {
      ria_error_set(&a->thread_error, RIA_INTERNAL_ERROR,
                    "administrative listener poll failed");
      break;
    }
    if (f[1].revents)
      break;
    if (f[0].revents & (POLLERR | POLLHUP | POLLNVAL)) {
      ria_error_set(&a->thread_error, RIA_INTERNAL_ERROR,
                    "administrative listener failed");
      break;
    }
    if (!(f[0].revents & POLLIN))
      continue;
    int fd = accept4(a->listener, NULL, NULL, SOCK_NONBLOCK | SOCK_CLOEXEC);
    if (fd < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)
        continue;
      ria_error_set(&a->thread_error, RIA_INTERNAL_ERROR,
                    "administrative accept failed");
      break;
    }
    request(a, fd);
    if (close(fd)) {
      ria_error_set(&a->thread_error, RIA_INTERNAL_ERROR,
                    "administrative connection close failed");
      break;
    }
  }
  uint64_t one = 1;
  ssize_t written;
  do {
    written = write(a->done, &one, sizeof(one));
  } while (written < 0 && errno == EINTR);
  if (written != sizeof(one))
    ria_error_set(&a->thread_error, RIA_INTERNAL_ERROR,
                  "administrative completion signaling failed");
  return NULL;
}
bool ria_admin_start(const char *path, uid_t uid, uint64_t timeout,
                     ria_admin_callbacks cb, ria_admin **out, ria_error *e) {
  if (!out)
    return ria_fail(e, RIA_INVALID_REQUEST, "missing admin owner destination");
  *out = NULL;
  uint64_t deadline;
  if (!path || path[0] != '/' ||
      strlen(path) >= sizeof(((struct sockaddr_un *)0)->sun_path) ||
      !cb.health || !cb.drain || !deadline_after(timeout, &deadline, e))
    return ria_fail(e, RIA_INVALID_REQUEST,
                    "invalid bounded administrative listener configuration");
  struct stat st;
  char parent[sizeof(((struct sockaddr_un *)0)->sun_path)];
  strcpy(parent, path);
  char *slash = strrchr(parent, '/');
  if (slash == parent)
    slash[1] = 0;
  else
    *slash = 0;
  int directory = open(parent, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
  bool private_parent = directory >= 0 && !fstat(directory, &st) &&
                        S_ISDIR(st.st_mode) && st.st_uid == geteuid() &&
                        !(st.st_mode & 0022);
  if (directory >= 0 && close(directory))
    private_parent = false;
  if (!private_parent)
    return ria_fail(e, RIA_UNAUTHORIZED,
                    "administrative socket parent must be owned by this "
                    "effective UID and not writable by another UID");
  if (!lstat(path, &st) || errno != ENOENT)
    return ria_fail(e, RIA_UNAUTHORIZED,
                    "administrative socket destination already exists or "
                    "cannot be inspected");
  ria_admin *a = calloc(1, sizeof(*a));
  if (!a)
    return ria_fail(e, RIA_RESOURCE_LIMIT,
                    "administrative owner allocation failed");
  a->listener = a->wake = a->done = -1;
  a->uid = uid;
  a->timeout_ms = timeout;
  a->callbacks = cb;
  strcpy(a->path, path);
  a->listener = socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
  a->wake = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  a->done = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  struct sockaddr_un address;
  memset(&address, 0, sizeof(address));
  address.sun_family = AF_UNIX;
  strcpy(address.sun_path, path);
  bool bound = false, ok = a->listener >= 0 && a->wake >= 0 && a->done >= 0;
  if (ok) {
    ok = bind(a->listener, (struct sockaddr *)&address, sizeof(address)) == 0;
    bound = ok;
  }
  if (ok) {
    ok = chmod(path, 0600) == 0 && lstat(path, &st) == 0 &&
         S_ISSOCK(st.st_mode) && listen(a->listener, 1) == 0;
    if (ok) {
      a->socket_device = st.st_dev;
      a->socket_inode = st.st_ino;
      pthread_attr_t attributes;
      int initialized = pthread_attr_init(&attributes);
      ok = !initialized;
      if (ok)
        ok = !pthread_attr_setstacksize(&attributes,
                                        (size_t)RIA_ADMIN_STACK_BYTES) &&
             !pthread_attr_setguardsize(&attributes, 4096) &&
             !pthread_create(&a->thread, &attributes, owner, a);
      if (!initialized) {
        int destroyed = pthread_attr_destroy(&attributes);
        /* Once the owner starts it must be joined, even if attr cleanup fails.
         */
        if (destroyed && ok) {
          *out = a;
          return ria_fail(e, RIA_INTERNAL_ERROR,
                          "administrative thread attribute cleanup failed; "
                          "started owner remains available for shutdown");
        }
      }
    }
  }
  if (!ok) {
    if (a->listener >= 0)
      close(a->listener);
    if (a->wake >= 0)
      close(a->wake);
    if (a->done >= 0)
      close(a->done);
    if (bound)
      unlink(path);
    free(a);
    return ria_fail(e, RIA_RESOURCE_LIMIT,
                    "administrative listener creation failed");
  }
  *out = a;
  return true;
}
bool ria_admin_stop(ria_admin *a, uint64_t timeout, ria_error *e) {
  if (!a)
    return true;
  uint64_t deadline;
  if (!deadline_after(timeout, &deadline, e))
    return false;
  uint64_t one = 1;
  ssize_t count;
  do {
    count = write(a->wake, &one, sizeof(one));
  } while (count < 0 && errno == EINTR);
  if (count != sizeof(one) && errno != EAGAIN)
    return ria_fail(e, RIA_INTERNAL_ERROR, "cannot wake administrative owner");
  for (;;) {
    uint64_t now = ria_monotonic_ms();
    if (now == UINT64_MAX || now >= deadline)
      return ria_fail(
          e, RIA_DEADLINE_EXCEEDED,
          "administrative callback failed to quiesce; live owner retained");
    struct pollfd p = {a->done, POLLIN, 0};
    int polled = poll(&p, 1, (int)(deadline - now));
    if (polled < 0 && errno == EINTR)
      continue;
    if (polled <= 0)
      return ria_fail(
          e, polled ? RIA_INTERNAL_ERROR : RIA_DEADLINE_EXCEEDED,
          "administrative completion wait failed; live owner retained");
    if (p.revents & POLLIN)
      break;
    return ria_fail(e, RIA_INTERNAL_ERROR,
                    "administrative completion descriptor failed");
  }
  int joined = pthread_join(a->thread, NULL);
  if (joined)
    return ria_fail(e, RIA_INTERNAL_ERROR, "cannot join administrative owner");
  bool ok = !a->thread_error.code;
  if (!ok && e)
    *e = a->thread_error;
  if (close(a->listener) && ok)
    ok =
        ria_fail(e, RIA_INTERNAL_ERROR, "administrative listener close failed");
  if (close(a->wake) && ok)
    ok = ria_fail(e, RIA_INTERNAL_ERROR, "administrative wake close failed");
  if (close(a->done) && ok)
    ok = ria_fail(e, RIA_INTERNAL_ERROR,
                  "administrative completion close failed");
  struct stat st;
  bool owned = !lstat(a->path, &st) && S_ISSOCK(st.st_mode) &&
               st.st_dev == a->socket_device && st.st_ino == a->socket_inode;
  if ((!owned || unlink(a->path)) && ok)
    ok =
        ria_fail(e, RIA_INTERNAL_ERROR,
                 "administrative socket cleanup failed; replacement preserved");
  free(a);
  return ok;
}
