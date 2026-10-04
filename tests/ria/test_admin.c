#define _GNU_SOURCE
#include "ria/admin.h"
#include "ria/service.h"
#include <assert.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

typedef struct {
  pthread_mutex_t mutex;
  bool ready, active, drained;
} fixture;
static bool health(void *context, bool *ready, bool *active, ria_error *e) {
  (void)e;
  fixture *f = context;
  assert(!pthread_mutex_lock(&f->mutex));
  *ready = f->ready;
  *active = f->active;
  assert(!pthread_mutex_unlock(&f->mutex));
  return true;
}
static bool drain(void *context, uint64_t deadline, ria_error *e) {
  (void)e;
  assert(ria_monotonic_ms() < deadline);
  fixture *f = context;
  assert(!pthread_mutex_lock(&f->mutex));
  f->ready = false;
  f->active = false;
  f->drained = true;
  assert(!pthread_mutex_unlock(&f->mutex));
  return true;
}
static bool drained(fixture *f) {
  assert(!pthread_mutex_lock(&f->mutex));
  bool value = f->drained;
  assert(!pthread_mutex_unlock(&f->mutex));
  return value;
}
static int connect_socket(const char *path) {
  int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  assert(fd >= 0);
  struct sockaddr_un a = {0};
  a.sun_family = AF_UNIX;
  strcpy(a.sun_path, path);
  assert(!connect(fd, (struct sockaddr *)&a, sizeof(a)));
  return fd;
}
int main(void) {
  char dir[] = "/tmp/ria-admin-fixture-XXXXXX";
  assert(mkdtemp(dir));
  char path[108];
  assert(snprintf(path, sizeof(path), "%s/admin.sock", dir) <
         (int)sizeof(path));
  fixture f = {0};
  assert(!pthread_mutex_init(&f.mutex, NULL));
  f.ready = f.active = true;
  ria_admin *a = NULL;
  ria_error e = {0};
  assert(ria_admin_start(path, getuid(), 1000,
                         (ria_admin_callbacks){&f, health, drain}, &a, &e));
  struct stat st;
  assert(!lstat(path, &st) && S_ISSOCK(st.st_mode) &&
         (st.st_mode & 0777) == 0600);
  char *response = NULL;
  size_t n = 0;
  assert(ria_admin_request(path, "health", 1000, &response, &n, &e));
  assert(strstr(response, "\"ready\":true") &&
         strstr(response, "\"active\":true"));
  free(response);
  int fd = connect_socket(path);
  const char *bad = "{\"command\":\"health\",\"command\":\"drain\"}\n";
  assert(send(fd, bad, strlen(bad), MSG_NOSIGNAL) == (ssize_t)strlen(bad));
  char buffer[256] = {0};
  assert(recv(fd, buffer, sizeof(buffer) - 1, 0) > 0);
  assert(strstr(buffer, "\"ok\":false") && !drained(&f));
  assert(!close(fd));
  assert(ria_admin_request(path, "drain", 1000, &response, &n, &e));
  assert(strstr(response, "\"ok\":true") && drained(&f));
  free(response);
  fd = connect_socket(path);
  assert(send(fd, "{", 1, MSG_NOSIGNAL) == 1);
  assert(ria_admin_stop(a, 1000, &e));
  assert(!close(fd));
  assert(lstat(path, &st) < 0);
  assert(ria_admin_start(path, getuid() + 1, 1000,
                         (ria_admin_callbacks){&f, health, drain}, &a, &e));
  assert(!ria_admin_request(path, "health", 1000, &response, &n, &e));
  free(response);
  assert(ria_admin_stop(a, 1000, &e));
  fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0600);
  assert(fd >= 0);
  assert(!close(fd));
  assert(!ria_admin_start(path, getuid(), 1000,
                          (ria_admin_callbacks){&f, health, drain}, &a, &e));
  assert(!lstat(path, &st) && S_ISREG(st.st_mode));
  assert(!unlink(path));
  assert(!rmdir(dir));
  assert(!pthread_mutex_destroy(&f.mutex));
  puts("RIA admin peer authorization, lifecycle, malformed input and shutdown "
       "fixtures passed");
  return 0;
}
