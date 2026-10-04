#define _GNU_SOURCE
#define DS4_NO_GPU
#define DS4_RIA
#define DS4_SERVER_TEST
#define DS4_SERVER_TEST_NO_MAIN
#include "../../ds4_server.c"
#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "measurement fixture failure at %s:%d: %s\n", __FILE__,  \
              __LINE__, #condition);                                           \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

typedef struct {
  int fd;
  size_t bytes;
} drain_request;
static void *drain_socket(void *context) {
  drain_request *r = context;
  char bytes[4096];
  while (r->bytes) {
    ssize_t n =
        read(r->fd, bytes, r->bytes < sizeof bytes ? r->bytes : sizeof bytes);
    CHECK(n > 0);
    r->bytes -= (size_t)n;
    struct timespec pause = {0, 1000000};
    CHECK(nanosleep(&pause, NULL) == 0);
  }
  return NULL;
}
int main(void) {
  int sockets[2];
  CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
  request r = {0};
  ria_error error = {0};
  r.ria_observer = &r.ria_measurement;
  r.ria_observer->phase = "continuation";
  r.ria_observer->reused_prefix_tokens = 17;
  CHECK(ria_monotonic_ns(&r.ria_observer->started_ns, &error));
  CHECK(ria_measure_token(sockets[0], &r));
  CHECK(ria_measure_token(sockets[0], &r));
  CHECK(ria_measure_final(sockets[0], &r, 1));
  char bytes[2048] = {0};
  ssize_t n = recv(sockets[1], bytes, sizeof bytes - 1, MSG_DONTWAIT);
  CHECK(n > 0);
  CHECK(strstr(bytes, "event: ria_measurement\ndata: {\"index\":\"0\""));
  CHECK(strstr(bytes, "event: ria_measurement\ndata: {\"index\":\"1\""));
  CHECK(strstr(bytes,
               "event: ria_diagnostics\ndata: "
               "{\"phase\":\"continuation\",\"reused_prefix_tokens\":\"17\""));
  CHECK(strstr(bytes, "\"sampled_tokens\":\"2\",\"completion_tokens\":1"));
  CHECK(!strstr(bytes, "token_id"));
  CHECK(r.ria_observer->sampled_tokens == 2);
  CHECK(r.ria_observer->write_ns > 0);
  int buffer = 4096;
  CHECK(setsockopt(sockets[0], SOL_SOCKET, SO_SNDBUF, &buffer, sizeof buffer) ==
        0);
  int flags = fcntl(sockets[0], F_GETFL);
  CHECK(flags >= 0 && fcntl(sockets[0], F_SETFL, flags | O_NONBLOCK) == 0);
  size_t length = 262144;
  char *large = calloc(length, 1);
  CHECK(large);
  drain_request drain = {sockets[1], length};
  pthread_t thread;
  CHECK(pthread_create(&thread, NULL, drain_socket, &drain) == 0);
  uint64_t previous = r.ria_observer->write_ns;
  CHECK(request_send_all(sockets[0], large, length, &r));
  CHECK(pthread_join(thread, NULL) == 0);
  CHECK(r.ria_observer->write_ns - previous >= 5000000);
  CHECK(r.ria_observer->pending_write_ns >= 5000000);
  free(large);
  ria_measurement unchanged = r.ria_measurement;
  r.ria_observer = NULL;
  CHECK(request_send_all(sockets[0], "x", 1, &r));
  CHECK(!memcmp(&unchanged, &r.ria_measurement, sizeof unchanged));
  CHECK(close(sockets[0]) == 0 && close(sockets[1]) == 0);
  puts("native timing/blocked writer contracts: ok");
  return 0;
}
