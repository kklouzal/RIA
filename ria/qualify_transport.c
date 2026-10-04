#define _GNU_SOURCE
#include "qualify_transport.h"
#include "protocol.h"
#include "service.h"
#include "transport.h"
#include <errno.h>
#include <limits.h>
#include <openssl/crypto.h>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <poll.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

typedef struct {
  const char *id;
  uint16_t kind;
  uint32_t status, iteration;
  uint64_t request_bytes, reply_bytes, elapsed_ns;
  uint8_t request_hash[32], reply_hash[32];
} transport_case;
typedef struct {
  ria_json_doc config, request;
  ria_tls tls;
  ria_transport control, bulk;
  ria_binding binding;
  ria_limits limits;
  bool server;
  int listeners[2];
  uint8_t identities[8][32], peer[32], authorized_peer[32];
  uint8_t session[16], capability[32];
  uint64_t deadline, warmup, repeats, seed, control_credit, expert_credit,
      row_credit, bulk_credit, connect_timeout, started, startup_ns, timeout_ns,
      request_bytes, reply_bytes, owned_peak;
  uint32_t checks;
  transport_case *cases;
  size_t case_count, case_capacity, chunk_bytes;
  uint8_t *chunk;
  struct rusage initial_usage;
} qualifier;
static const char *const identity_names[8] = {
    "environment_digest",     "build_digest",       "policy_digest",
    "logical_model_digest",   "source_lock_digest", "operator_contract_digest",
    "preregistration_digest", "request_digest"};
static bool number(const ria_json_doc *d, uint32_t object, const char *key,
                   bool string, uint64_t *out, ria_error *e) {
  return ria_json_u64(d, ria_json_get(d, object, key), string, out, e);
}
static bool text_field(const ria_json_doc *d, uint32_t object, const char *key,
                       const char **out, ria_error *e) {
  size_t n;
  if (!ria_json_string(d, ria_json_get(d, object, key), out, &n, e) || !n ||
      memchr(*out, 0, n))
    return ria_fail(e, RIA_INVALID_REQUEST, "invalid bootstrap text field");
  return true;
}
static uint64_t nanoseconds(void) {
  struct timespec t;
  if (clock_gettime(CLOCK_MONOTONIC, &t))
    return 0;
  return (uint64_t)t.tv_sec * UINT64_C(1000000000) + (uint64_t)t.tv_nsec;
}
static uint64_t phase_deadline(qualifier *q, uint64_t timeout) {
  uint64_t now = ria_monotonic_ms(), end;
  if (!ria_u64_add(now, timeout, &end) || end > q->deadline)
    return q->deadline;
  return end;
}
static bool canonical_source(const char *bytes, size_t n, char **out,
                             size_t *length, ria_error *e) {
  ria_json_doc d = {0};
  bool ok = ria_json_parse(bytes, n, (ria_json_limits){n, 8192, 32}, &d, e) &&
            ria_json_canonical(&d, false, out, length, e);
  ria_json_free(&d);
  return ok;
}
static bool configuration(qualifier *q, const char *config, const char *request,
                          ria_error *e) {
  static const char *const config_fields[] = {"schema_revision",
                                              "role",
                                              "environment_digest",
                                              "build_digest",
                                              "build_info_file",
                                              "request_digest",
                                              "network",
                                              "tls"};
  static const char *const request_fields[] = {"schema_revision",
                                               "kind",
                                               "environment_digest",
                                               "build_digest",
                                               "policy_digest",
                                               "logical_model_digest",
                                               "source_lock_digest",
                                               "operator_contract_digest",
                                               "preregistration_digest",
                                               "deadline_ms",
                                               "warmup",
                                               "repeats",
                                               "fixture_seed",
                                               "max_frame_bytes",
                                               "control_credit",
                                               "expert_credit",
                                               "row_credit",
                                               "bulk_credit",
                                               "digest"};
  uint64_t revision, duration, frame;
  const char *role, *kind, *build_path;
  if (!ria_json_read(config, (ria_json_limits){65536, 512, 8}, &q->config, e) ||
      !ria_json_read(request, (ria_json_limits){65536, 512, 8}, &q->request,
                     e) ||
      !ria_json_fields(&q->config, 0, config_fields, 8, config_fields, 8, e) ||
      !ria_json_fields(&q->request, 0, request_fields, 19, request_fields, 19,
                       e) ||
      !number(&q->config, 0, "schema_revision", false, &revision, e) ||
      revision != 1 ||
      !number(&q->request, 0, "schema_revision", false, &revision, e) ||
      revision != 1 || !text_field(&q->config, 0, "role", &role, e) ||
      (strcmp(role, "client") && strcmp(role, "expert")) ||
      !text_field(&q->request, 0, "kind", &kind, e) ||
      strcmp(kind, "transport_request"))
    return ria_fail(e, RIA_INVALID_REQUEST, "invalid bootstrap/request schema");
  q->server = !strcmp(role, "expert");
  uint8_t actual[32], claimed[32];
  if (!ria_json_sha256(&q->request, true, actual, e) ||
      !ria_json_digest_field(
          &q->request, ria_json_get(&q->request, 0, "digest"), claimed, e) ||
      CRYPTO_memcmp(actual, claimed, 32) ||
      !ria_json_digest_field(&q->config,
                             ria_json_get(&q->config, 0, "request_digest"),
                             claimed, e) ||
      CRYPTO_memcmp(actual, claimed, 32))
    return ria_fail(e, RIA_IDENTITY_MISMATCH,
                    "request is not authorized by bootstrap configuration");
  memcpy(q->identities[7], actual, 32);
  for (unsigned i = 0; i < 7; i++) {
    if (!ria_json_digest_field(&q->request,
                               ria_json_get(&q->request, 0, identity_names[i]),
                               q->identities[i], e))
      return false;
    if (i < 2 &&
        (!ria_json_digest_field(&q->config,
                                ria_json_get(&q->config, 0, identity_names[i]),
                                actual, e) ||
         CRYPTO_memcmp(actual, q->identities[i], 32)))
      return ria_fail(
          e, RIA_IDENTITY_MISMATCH,
          "bootstrap environment/build differs from registered request");
  }
  if (!text_field(&q->config, 0, "build_info_file", &build_path, e) ||
      build_path[0] != '/')
    return false;
  ria_json_doc build = {0};
  bool build_ok =
      ria_json_read(build_path, (ria_json_limits){1048576, 8192, 16}, &build,
                    e) &&
      ria_json_sha256(&build, true, actual, e) &&
      ria_json_digest_field(&build, ria_json_get(&build, 0, "digest"), claimed,
                            e) &&
      !CRYPTO_memcmp(actual, claimed, 32) &&
      !CRYPTO_memcmp(actual, q->identities[1], 32);
  ria_json_free(&build);
  if (!build_ok)
    return ria_fail(
        e, RIA_IDENTITY_MISMATCH,
        "build-info identity differs from registered executable build");
  if (!number(&q->request, 0, "deadline_ms", false, &duration, e) ||
      !duration || duration > 3600000 ||
      !ria_u64_add(ria_monotonic_ms(), duration, &q->deadline) ||
      !number(&q->request, 0, "warmup", false, &q->warmup, e) ||
      q->warmup > 100 ||
      !number(&q->request, 0, "repeats", false, &q->repeats, e) ||
      !q->repeats || q->repeats > 1000 ||
      !number(&q->request, 0, "fixture_seed", false, &q->seed, e) ||
      !number(&q->request, 0, "max_frame_bytes", true, &frame, e) ||
      frame < 4096 || frame > RIA_FRAME_MAX ||
      !number(&q->request, 0, "control_credit", true, &q->control_credit, e) ||
      !number(&q->request, 0, "expert_credit", true, &q->expert_credit, e) ||
      !number(&q->request, 0, "row_credit", true, &q->row_credit, e) ||
      !number(&q->request, 0, "bulk_credit", true, &q->bulk_credit, e))
    return false;
  if (q->control_credit < 4 * (2 * RIA_ERROR_MAX + RIA_HEADER_BYTES) ||
      q->expert_credit < 2 * (68 + RIA_ERROR_MAX + 2 * RIA_HEADER_BYTES) ||
      q->row_credit < 2 * (32 + 56 + 2 * RIA_HEADER_BYTES) ||
      q->bulk_credit < 1024)
    return ria_fail(
        e, RIA_RESOURCE_LIMIT,
        "registered credits cannot protect bounded fixture progress");
  uint64_t total;
  if (!ria_u64_add(q->control_credit, q->expert_credit, &total) ||
      !ria_u64_add(total, q->row_credit, &total) ||
      !ria_u64_add(total, q->bulk_credit, &total) || total > 67108864)
    return ria_fail(
        e, RIA_RESOURCE_LIMIT,
        "registered inflight credit total exceeds native fixture bound");
  uint32_t network = ria_json_get(&q->config, 0, "network"),
           tls = ria_json_get(&q->config, 0, "tls");
  const char *const network_fields[] = {
      "control_address",      "bulk_address",        "connect_timeout_ms",
      "handshake_timeout_ms", "frame_io_timeout_ms", "write_timeout_ms",
      "operation_timeout_ms"};
  const char *const tls_fields[] = {"ca_file", "certificate_file",
                                    "private_key_file", "expected_peer_name",
                                    "authorized_peer_sha256"};
  uint64_t handshake, frame_time, write_time, operation;
  if (!ria_json_fields(&q->config, network, network_fields, 7, network_fields,
                       7, e) ||
      !ria_json_fields(&q->config, tls, tls_fields, 5, tls_fields, 5, e) ||
      !number(&q->config, network, "connect_timeout_ms", false,
              &q->connect_timeout, e) ||
      !number(&q->config, network, "handshake_timeout_ms", false, &handshake,
              e) ||
      !number(&q->config, network, "frame_io_timeout_ms", false, &frame_time,
              e) ||
      !number(&q->config, network, "write_timeout_ms", false, &write_time, e) ||
      !number(&q->config, network, "operation_timeout_ms", false, &operation,
              e) ||
      !q->connect_timeout || q->connect_timeout > duration || !handshake ||
      handshake > duration || !frame_time || frame_time > duration ||
      !write_time || write_time > duration || !operation ||
      operation > duration ||
      !ria_json_digest_field(
          &q->config, ria_json_get(&q->config, tls, "authorized_peer_sha256"),
          q->authorized_peer, e))
    return false;
  uint64_t chunk = frame - 64;
  if (chunk > RIA_BULK_MAX)
    chunk = RIA_BULK_MAX;
  if (chunk > q->bulk_credit - 208)
    chunk = q->bulk_credit - 208;
  q->limits = (ria_limits){frame, chunk,     1,          2,         1,
                           total, operation, frame_time, write_time};
  ria_binding_init(&q->binding, &q->limits);
  if (!ria_limits_validate(&q->limits, e) ||
      !ria_binding_protect(&q->binding, q->control_credit, q->row_credit,
                           q->bulk_credit, e))
    return false;
  ria_tls_config tc = {.server = q->server, .handshake_timeout_ms = handshake};
  if (!text_field(&q->config, tls, "ca_file", &tc.ca_file, e) ||
      !text_field(&q->config, tls, "certificate_file", &tc.certificate_file,
                  e) ||
      !text_field(&q->config, tls, "private_key_file", &tc.private_key_file,
                  e) ||
      !text_field(&q->config, tls, "expected_peer_name", &tc.expected_peer_name,
                  e) ||
      tc.ca_file[0] != '/' || tc.certificate_file[0] != '/' ||
      tc.private_key_file[0] != '/')
    return false;
  q->case_capacity = (size_t)q->repeats * 9;
  q->cases = calloc(q->case_capacity, sizeof *q->cases);
  q->chunk_bytes = (size_t)chunk;
  q->chunk = malloc(q->chunk_bytes + 64);
  if (!q->cases || !q->chunk)
    return ria_fail(e, RIA_RESOURCE_LIMIT, "fixture buffer allocation failed");
  q->owned_peak = sizeof *q + q->config.allocated_bytes +
                  q->request.allocated_bytes +
                  q->case_capacity * sizeof *q->cases + q->chunk_bytes + 64;
  memset(q->chunk, 0, 64);
  ria_write_u64(q->chunk, 3);
  ria_write_u32(q->chunk + 24, (uint32_t)q->chunk_bytes);
  for (size_t i = 0; i < q->chunk_bytes; i++)
    q->chunk[64 + i] =
        (uint8_t)((q->seed + i * UINT64_C(37) + (i >> 8)) & 255u);
  if (!ria_sha256(q->chunk + 64, q->chunk_bytes, q->chunk + 32, e))
    return false;
  return ria_tls_create(&q->tls, &tc, e);
}
static bool address(qualifier *q, unsigned channel,
                    struct sockaddr_storage *out, socklen_t *length,
                    ria_error *e) {
  const char *value;
  return text_field(&q->config, ria_json_get(&q->config, 0, "network"),
                    channel ? "bulk_address" : "control_address", &value, e) &&
         ria_address(value, q->server, out, length, e);
}
static bool listener(qualifier *q, unsigned channel, ria_error *e) {
  struct sockaddr_storage a;
  socklen_t length;
  if (!address(q, channel, &a, &length, e))
    return false;
  int fd = socket(a.ss_family, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
  if (fd < 0)
    return ria_fail(e, RIA_RESOURCE_LIMIT, "fixture listener creation failed");
  q->listeners[channel] = fd;
  int yes = 1;
  if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof yes) ||
      bind(fd, (struct sockaddr *)&a, length) || listen(fd, 1))
    return ria_fail(e, RIA_RESOURCE_LIMIT,
                    "fixture listener bind/listen failed");
  return true;
}
static bool pair_channel(qualifier *q, unsigned channel, ria_error *e) {
  ria_transport *t = channel ? &q->bulk : &q->control;
  bool ok;
  if (!q->server) {
    struct sockaddr_storage a;
    socklen_t length;
    if (!address(q, channel, &a, &length, e))
      return false;
    uint64_t end = phase_deadline(q, q->connect_timeout);
    ok = false;
    while (!ok) {
      uint64_t now = ria_monotonic_ms();
      if (now >= end || q->deadline - now < 2)
        return ria_fail(e, RIA_DEADLINE_EXCEEDED,
                        "fixture connection deadline elapsed");
      uint64_t timeout = end - now;
      if (timeout > (q->deadline - now) / 2)
        timeout = (q->deadline - now) / 2;
      uint64_t handshake = q->tls.handshake_timeout_ms;
      if (q->tls.handshake_timeout_ms > (q->deadline - now) / 2)
        q->tls.handshake_timeout_ms = (q->deadline - now) / 2;
      ok = ria_transport_connect(t, &q->tls, (struct sockaddr *)&a, length,
                                 timeout, e);
      q->tls.handshake_timeout_ms = handshake;
      if (!ok && e->code != RIA_NOT_READY)
        return false;
      if (!ok) {
        /* Connection refusal before the paired supervisor starts its listener
         * is the only retry class. No Bind mutation has been sent yet. */
        uint64_t wait =
            end - (ria_monotonic_ms() < end ? ria_monotonic_ms() : end);
        int duration = wait > 20 ? 20 : (int)wait;
        if (poll(NULL, 0, duration) < 0 && errno != EINTR)
          return ria_fail(e, RIA_EXECUTOR_ERROR,
                          "fixture connection retry wait failed");
        memset(e, 0, sizeof *e);
      }
    }
  } else {
    int fd = -1;
    while (fd < 0) {
      uint64_t now = ria_monotonic_ms();
      if (now >= q->deadline)
        return ria_fail(e, RIA_DEADLINE_EXCEEDED,
                        "fixture accept deadline elapsed");
      struct pollfd p = {q->listeners[channel], POLLIN, 0};
      uint64_t remain = q->deadline - now;
      int status = poll(&p, 1, remain > INT_MAX ? INT_MAX : (int)remain);
      if (status < 0 && errno == EINTR)
        continue;
      if (status <= 0 || (p.revents & (POLLERR | POLLHUP | POLLNVAL)))
        return ria_fail(e, RIA_DEADLINE_EXCEEDED,
                        "fixture accept failed or timed out");
      fd = accept4(q->listeners[channel], NULL, NULL,
                   SOCK_CLOEXEC | SOCK_NONBLOCK);
      if (fd < 0 && errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK)
        return ria_fail(e, RIA_EXECUTOR_ERROR, "fixture accept failed");
    }
    uint64_t now = ria_monotonic_ms(), handshake = q->tls.handshake_timeout_ms;
    if (now >= q->deadline) {
      close(fd);
      return ria_fail(e, RIA_DEADLINE_EXCEEDED,
                      "fixture handshake deadline elapsed");
    }
    if (q->tls.handshake_timeout_ms > q->deadline - now)
      q->tls.handshake_timeout_ms = q->deadline - now;
    ok = ria_transport_accept(t, &q->tls, fd, e);
    q->tls.handshake_timeout_ms = handshake;
    if (!ok)
      close(fd);
  }
  uint8_t peer[32];
  if (!ok || !ria_transport_peer_digest(t, peer, e) ||
      CRYPTO_memcmp(peer, q->authorized_peer, 32))
    return ria_fail(
        e, RIA_UNAUTHORIZED,
        "fixture peer certificate is not provisioned for this request");
  if (channel && CRYPTO_memcmp(peer, q->peer, 32))
    return ria_fail(e, RIA_UNAUTHORIZED, "control/bulk certificates differ");
  memcpy(q->peer, peer, 32);
  return true;
}
static ria_header header(qualifier *q, uint16_t kind, uint64_t id,
                         uint64_t length, bool reply, uint32_t status) {
  ria_header h = {.kind = kind,
                  .flags = reply,
                  .status = status,
                  .payload_length = length,
                  .request_id = id,
                  .epoch = 1};
  memcpy(h.session, q->session, 16);
  return h;
}
static bool send_frame(qualifier *q, ria_transport *t, const ria_header *h,
                       const void *payload, ria_error *e) {
  uint8_t encoded[64];
  if (!ria_header_encode(h, encoded, e))
    return false;
  uint64_t deadline = phase_deadline(q, q->limits.write_timeout_ms);
  /* Deliberate record fragmentation tests first-byte/header/payload partial IO
   * using the production TLS writer under one absolute write deadline. */
  for (size_t i = 0; i < 64;) {
    size_t n = i ? 7 : 1;
    if (n > 64 - i)
      n = 64 - i;
    if (!ria_transport_write(t, encoded + i, n, deadline, e))
      return false;
    i += n;
  }
  for (size_t i = 0; i < h->payload_length;) {
    size_t n = 2048 + (size_t)((q->seed + i) & 2047);
    if (n > h->payload_length - i)
      n = (size_t)h->payload_length - i;
    if (!ria_transport_write(t, (const uint8_t *)payload + i, n, deadline, e))
      return false;
    i += n;
  }
  uint64_t *counter = h->flags ? &q->reply_bytes : &q->request_bytes;
  return ria_u64_add(*counter, h->payload_length + 64, counter) ||
         ria_fail(e, RIA_RESOURCE_LIMIT,
                  "fixture application byte counter overflow");
}
static bool receive_frame(qualifier *q, ria_transport *t, ria_header *h,
                          uint8_t **payload, ria_error *e) {
  bool ok =
      ria_transport_frame(t, phase_deadline(q, q->limits.operation_timeout_ms),
                          q->limits.frame_io_timeout_ms,
                          q->limits.frame_payload_bytes, h, payload, e);
  if (ok) {
    uint64_t bytes = sizeof *q + q->config.allocated_bytes +
                     q->request.allocated_bytes +
                     q->case_capacity * sizeof *q->cases + q->chunk_bytes + 64 +
                     h->payload_length;
    if (bytes > q->owned_peak)
      q->owned_peak = bytes;
    uint64_t *counter = h->flags ? &q->reply_bytes : &q->request_bytes;
    if (!ria_u64_add(*counter, h->payload_length + 64, counter)) {
      free(*payload);
      *payload = NULL;
      return ria_fail(e, RIA_RESOURCE_LIMIT,
                      "fixture application byte counter overflow");
    }
  }
  return ok;
}
static bool frame_hash(const ria_header *h, const void *payload,
                       uint8_t out[32], ria_error *e) {
  uint8_t bytes[64];
  if (!ria_header_encode(h, bytes, e))
    return false;
  EVP_MD_CTX *ctx = EVP_MD_CTX_new();
  unsigned length = 0;
  bool ok = ctx && EVP_DigestInit_ex(ctx, EVP_sha256(), NULL) == 1 &&
            EVP_DigestUpdate(ctx, bytes, sizeof bytes) == 1 &&
            EVP_DigestUpdate(ctx, payload, (size_t)h->payload_length) == 1 &&
            EVP_DigestFinal_ex(ctx, out, &length) == 1 && length == 32;
  EVP_MD_CTX_free(ctx);
  return ok || ria_fail(e, RIA_EXECUTOR_ERROR, "fixture frame SHA256 failed");
}
static bool control(qualifier *q, const ria_header *h, uint8_t *bytes,
                    ria_json_doc *d, ria_error *e) {
  (void)q;
  return ria_control_json(bytes, (size_t)h->payload_length, h, d, e);
}
static bool same_request_identity(qualifier *q, const ria_json_doc *d,
                                  ria_error *e) {
  const char *const fields[] = {"logical_model_digest",
                                "operator_contract_digest", "encoding_digest",
                                "placement_plan_digest"};
  const unsigned index[] = {3, 5, 4, 6};
  uint8_t actual[32];
  for (unsigned i = 0; i < 4; i++)
    if (!ria_json_digest_field(d, ria_json_get(d, 0, fields[i]), actual, e) ||
        CRYPTO_memcmp(actual, q->identities[index[i]], 32))
      return ria_fail(e, RIA_IDENTITY_MISMATCH,
                      "paired fixture registration/model contract differs");
  return true;
}
static bool bind_pair(qualifier *q, ria_error *e) {
  if (q->server && (!listener(q, 0, e) || !listener(q, 1, e)))
    return false;
  if (!pair_channel(q, 0, e))
    return false;
  ria_header h;
  char *json = NULL;
  uint8_t *received = NULL;
  size_t n = 0;
  ria_json_doc d = {0};
  bool ok;
  if (!q->server) {
    char hashes[5][65];
    const unsigned source[] = {3, 5, 4, 0, 6};
    for (unsigned i = 0; i < 5; i++)
      ria_hex_encode(q->identities[source[i]], 32, hashes[i]);
    char buffer[2048];
    int length =
        snprintf(buffer, sizeof buffer,
                 "{\"role\":\"client\",\"logical_model_digest\":\"%s\","
                 "\"operator_contract_digest\":\"%s\","
                 "\"encoding_digest\":\"%s\",\"client_layout_digest\":\"%s\","
                 "\"placement_plan_digest\":\"%s\","
                 "\"profile\":\"bf16\",\"server_executor\":\"cpu\",\"limits\":{"
                 "\"frame_payload_bytes\":%llu,"
                 "\"bulk_data_bytes\":%llu,\"expert_rows\":1,\"expert_"
                 "requests\":2,\"row_lookup_rows\":1,"
                 "\"inflight_payload_bytes\":%llu,\"operation_timeout_ms\":%"
                 "llu,\"frame_io_timeout_ms\":%llu,\"write_timeout_ms\":%llu}}",
                 hashes[0], hashes[1], hashes[2], hashes[3], hashes[4],
                 (unsigned long long)q->limits.frame_payload_bytes,
                 (unsigned long long)q->limits.bulk_data_bytes,
                 (unsigned long long)q->limits.inflight_payload_bytes,
                 (unsigned long long)q->limits.operation_timeout_ms,
                 (unsigned long long)q->limits.frame_io_timeout_ms,
                 (unsigned long long)q->limits.write_timeout_ms);
    ok = length > 0 && (size_t)length < sizeof buffer &&
         canonical_source(buffer, (size_t)length, &json, &n, e);
    h = (ria_header){.kind = RIA_BIND, .payload_length = n, .request_id = 1};
    if (ok)
      ok = send_frame(q, &q->control, &h, json, e) &&
           receive_frame(q, &q->control, &h, &received, e) &&
           h.kind == RIA_BIND && h.flags == 1 && !h.status &&
           control(q, &h, received, &d, e) && same_request_identity(q, &d, e);
    if (ok) {
      const char *text;
      size_t length_text;
      uint64_t epoch;
      ok = ria_json_string(&d, ria_json_get(&d, 0, "session_id"), &text,
                           &length_text, e) &&
           ria_hex_decode(text, length_text, q->session, 16, e) &&
           number(&d, 0, "epoch", true, &epoch, e) && epoch == 1 &&
           ria_json_digest_field(&d, ria_json_get(&d, 0, "bulk_capability"),
                                 q->capability, e);
      ria_limits agreed;
      if (ok)
        ok = ria_bind_validate(&d, true, &agreed, e) &&
             !memcmp(&agreed, &q->limits, sizeof agreed);
    }
  } else {
    ok = receive_frame(q, &q->control, &h, &received, e) &&
         h.kind == RIA_BIND && !h.flags && h.request_id == 1 && !h.epoch &&
         control(q, &h, received, &d, e) && same_request_identity(q, &d, e);
    ria_limits requested;
    if (ok)
      ok = ria_bind_validate(&d, false, &requested, e) &&
           !memcmp(&requested, &q->limits, sizeof requested) &&
           RAND_bytes(q->session, 16) == 1 &&
           RAND_bytes(q->capability, 32) == 1 &&
           ria_bind_response_json(&d, &q->limits, q->identities[0], q->session,
                                  1, q->capability, &json, &n, e);
    if (ok) {
      h = header(q, RIA_BIND, 1, n, true, 0);
      ok = send_frame(q, &q->control, &h, json, e);
    }
  }
  free(json);
  free(received);
  ria_json_free(&d);
  if (!ok ||
      !ria_binding_install(&q->binding, q->session, 1, q->capability, e) ||
      !pair_channel(q, 1, e))
    return false;
  json = NULL;
  received = NULL;
  n = 0;
  if (!q->server) {
    char sid[33], capability[65], logical[65], operation[65], buffer[512];
    ria_hex_encode(q->session, 16, sid);
    ria_hex_encode(q->capability, 32, capability);
    ria_hex_encode(q->identities[3], 32, logical);
    ria_hex_encode(q->identities[5], 32, operation);
    int length = snprintf(
        buffer, sizeof buffer,
        "{\"session_id\":\"%s\",\"epoch\":\"1\",\"logical_model_digest\":\"%"
        "s\",\"operator_contract_digest\":\"%s\",\"bulk_capability\":\"%s\"}",
        sid, logical, operation, capability);
    ok = length > 0 && (size_t)length < sizeof buffer &&
         canonical_source(buffer, (size_t)length, &json, &n, e);
    h = header(q, RIA_BIND_BULK, 1, n, false, 0);
    if (ok)
      ok = send_frame(q, &q->bulk, &h, json, e) &&
           receive_frame(q, &q->bulk, &h, &received, e) &&
           h.kind == RIA_BIND_BULK && h.flags == 1 && !h.status &&
           h.request_id == 1 && h.epoch == 1 &&
           !CRYPTO_memcmp(h.session, q->session, 16) &&
           control(q, &h, received, &d, e);
  } else {
    ok = receive_frame(q, &q->bulk, &h, &received, e) &&
         h.kind == RIA_BIND_BULK && h.request_id == 1 && !h.flags &&
         h.epoch == 1 && !CRYPTO_memcmp(h.session, q->session, 16) &&
         control(q, &h, received, &d, e);
    uint8_t actual[32];
    if (ok)
      ok =
          ria_json_digest_field(&d, ria_json_get(&d, 0, "bulk_capability"),
                                actual, e) &&
          !CRYPTO_memcmp(actual, q->capability, 32) &&
          ria_json_digest_field(&d, ria_json_get(&d, 0, "logical_model_digest"),
                                actual, e) &&
          !CRYPTO_memcmp(actual, q->identities[3], 32) &&
          ria_json_digest_field(
              &d, ria_json_get(&d, 0, "operator_contract_digest"), actual, e) &&
          !CRYPTO_memcmp(actual, q->identities[5], 32);
    if (ok) {
      h = header(q, RIA_BIND_BULK, 1, 14, true, 0);
      ok = send_frame(q, &q->bulk, &h, "{\"bound\":true}", e);
    }
  }
  free(json);
  free(received);
  ria_json_free(&d);
  if (!ok || !ria_binding_bulk(&q->binding, q->capability, true, e))
    return false;
  ria_error expected = {0};
  if (ria_binding_bulk(&q->binding, q->capability, true, &expected) ||
      expected.code != RIA_UNAUTHORIZED)
    return ria_fail(e, RIA_INTERNAL_ERROR, "bulk capability was reusable");
  q->checks |= 3u;
  q->startup_ns = nanoseconds() - q->started;
  return true;
}

enum {
  CASE_HEALTH,
  CASE_ROWS,
  CASE_BULK,
  CASE_CANCEL_PENDING,
  CASE_EXPERT_CANCELLED,
  CASE_EXPERT_SUCCESS,
  CASE_CANCEL_TERMINAL,
  CASE_CANCEL_UNKNOWN,
  CASE_MALFORMED,
  CASE_COUNT
};
static const char *const check_names[] = {"mtls_san_certificate_pair",
                                          "bind_and_bulk_one_use",
                                          "credit_and_protected_progress",
                                          "cancel_no_early_credit",
                                          "terminal_history",
                                          "malformed_typed_error",
                                          "partial_frame_deadline",
                                          "response_payload_integrity"};
static const char *const case_names[CASE_COUNT] = {
    "health",           "rows",           "bulk_chunk",      "cancel_pending",
    "expert_cancelled", "expert_success", "cancel_terminal", "cancel_unknown",
    "malformed_rows"};
static const uint16_t case_kinds[CASE_COUNT] = {
    RIA_HEALTH, RIA_ROWS,   RIA_CHUNK,  RIA_CANCEL, RIA_EXPERT,
    RIA_EXPERT, RIA_CANCEL, RIA_CANCEL, RIA_ROWS};
typedef struct {
  transport_case result;
  ria_header request;
  uint8_t payload[68], reply[256];
  size_t reply_length;
  uint64_t start;
} fixture_event;
static const ria_operation operation = {.handle = 1,
                                        .input_width = 1,
                                        .output_width = 1,
                                        .max_rows = 1,
                                        .expert_count = 1,
                                        .max_selected = 1,
                                        .coefficient_min = 1,
                                        .coefficient_max = 1};
static const ria_table table = {2, 4, 16, 3};
static uint64_t owned_base(const qualifier *q) {
  return sizeof *q + q->config.allocated_bytes + q->request.allocated_bytes +
         q->case_capacity * sizeof *q->cases + q->chunk_bytes + 64 +
         CASE_COUNT * sizeof(fixture_event);
}
static void track_owned(qualifier *q, uint64_t extra) {
  uint64_t total = owned_base(q) + extra;
  if (total > q->owned_peak)
    q->owned_peak = total;
}
static void expert_payload(uint8_t *p, uint64_t invocation, uint64_t seed) {
  ria_write_u64(p, 1);
  ria_write_u64(p + 8, invocation);
  ria_write_u32(p + 16, 1);
  ria_write_u32(p + 20, 1);
  ria_write_u32(p + 24, 1);
  ria_write_u32(p + 28, 1);
  ria_write_u64(p + 40, invocation);
  ria_write_u32(p + 52, 1);
  ria_write_f32(p + 60, 1.0f);
  ria_write_f32(p + 64, (float)(seed & 65535u) / 256.0f);
}
static bool events_prepare(qualifier *q, fixture_event events[CASE_COUNT],
                           uint64_t sequence) {
  memset(events, 0, CASE_COUNT * sizeof *events);
  uint64_t base = 2 + sequence * 16;
  const unsigned delta[CASE_COUNT] = {3, 2, 0, 4, 0, 1, 5, 6, 7};
  const unsigned lengths[CASE_COUNT] = {2, 32, 16, 16, 68, 68, 16, 16, 33};
  for (unsigned i = 0; i < CASE_COUNT; i++) {
    fixture_event *v = &events[i];
    v->result.id = case_names[i];
    v->result.kind = case_kinds[i];
    v->request =
        header(q, case_kinds[i], base + delta[i], lengths[i], false, 0);
    v->result.request_bytes = lengths[i] + 64;
  }
  memcpy(events[CASE_HEALTH].payload, "{}", 2);
  uint8_t *row = events[CASE_ROWS].payload;
  ria_write_u64(row, 2);
  ria_write_u32(row + 8, 1);
  ria_write_u64(row + 16, (q->seed + sequence) & 3u);
  ria_write_u64(row + 24, base);
  uint8_t *reply = events[CASE_ROWS].reply;
  ria_write_u64(reply, 2);
  ria_write_u32(reply + 8, 1);
  ria_write_u32(reply + 12, 16);
  ria_write_u64(reply + 16, 3);
  memcpy(reply + 24, row + 16, 16);
  for (unsigned i = 0; i < 16; i++)
    reply[40 + i] = (uint8_t)((q->seed + sequence + i) & 255u);
  events[CASE_ROWS].reply_length = 56;
  memcpy(events[CASE_MALFORMED].payload, row, 32);
  events[CASE_MALFORMED].payload[32] = 255;
  ria_write_u64(events[CASE_BULK].payload, 3);
  expert_payload(events[CASE_EXPERT_CANCELLED].payload, base,
                 q->seed + sequence);
  expert_payload(events[CASE_EXPERT_SUCCESS].payload, base + 1,
                 q->seed + sequence + 1);
  reply = events[CASE_EXPERT_SUCCESS].reply;
  ria_write_u64(reply, 1);
  ria_write_u64(reply + 8, base + 1);
  ria_write_u32(reply + 16, 1);
  ria_write_u32(reply + 20, 1);
  ria_write_u32(reply + 24, 1);
  memcpy(reply + 32, events[CASE_EXPERT_SUCCESS].payload + 64, 4);
  events[CASE_EXPERT_SUCCESS].reply_length = 36;
  const unsigned cancels[] = {CASE_CANCEL_PENDING, CASE_CANCEL_TERMINAL,
                              CASE_CANCEL_UNKNOWN};
  for (unsigned i = 0; i < 3; i++) {
    fixture_event *v = &events[cancels[i]];
    ria_write_u64(v->payload, i == 2 ? base - 1 : base);
    ria_write_u64(v->payload + 8, 1);
    memcpy(v->reply, v->payload, 16);
    ria_write_u32(v->reply + 16, i);
    v->reply_length = 24;
  }
  const char cancelled[] = "{\"code\":8,\"message\":\"fixture cancellation\"}";
  const char malformed[] =
      "{\"code\":5,\"message\":\"fixture exact-length rejection\"}";
  memcpy(events[CASE_EXPERT_CANCELLED].reply, cancelled, sizeof cancelled - 1);
  events[CASE_EXPERT_CANCELLED].reply_length = sizeof cancelled - 1;
  events[CASE_EXPERT_CANCELLED].result.status = RIA_CANCELLED;
  memcpy(events[CASE_MALFORMED].reply, malformed, sizeof malformed - 1);
  events[CASE_MALFORMED].reply_length = sizeof malformed - 1;
  events[CASE_MALFORMED].result.status = RIA_RESOURCE_LIMIT;
  return track_owned(q, 0), true;
}
static bool payload_request(qualifier *q, fixture_event *v, ria_error *e) {
  if (v->request.kind == RIA_EXPERT) {
    ria_expert_request r;
    return ria_expert_parse(v->payload, (size_t)v->request.payload_length,
                            RIA_EXPERT, &operation, &q->limits, &r, e);
  }
  if (v->request.kind == RIA_ROWS) {
    ria_row_request r;
    if (v->result.status) {
      ria_error expected = {0};
      if (ria_rows_parse(v->payload, (size_t)v->request.payload_length, &table,
                         &q->limits, &r, &expected) ||
          expected.code != RIA_RESOURCE_LIMIT)
        return ria_fail(
            e, RIA_INTERNAL_ERROR,
            "malformed fixture rows were accepted or misclassified");
      return true;
    }
    return ria_rows_parse(v->payload, (size_t)v->request.payload_length, &table,
                          &q->limits, &r, e);
  }
  if (v->request.kind == RIA_CHUNK) {
    uint64_t handle, index;
    return ria_chunk_request(v->payload, (size_t)v->request.payload_length,
                             &handle, &index, e) &&
           handle == 3 && !index;
  }
  if (v->request.kind == RIA_CANCEL) {
    uint64_t target;
    return ria_cancel_parse(v->payload, (size_t)v->request.payload_length,
                            v->request.request_id, 1, &target, e);
  }
  ria_json_doc d = {0};
  bool ok = control(q, &v->request, v->payload, &d, e);
  track_owned(q, d.allocated_bytes);
  ria_json_free(&d);
  return ok;
}
static uint64_t case_cost(qualifier *q, unsigned index) {
  if (index == CASE_EXPERT_CANCELLED)
    return q->expert_credit / 2;
  if (index == CASE_EXPERT_SUCCESS)
    return q->expert_credit - q->expert_credit / 2;
  if (index == CASE_ROWS || index == CASE_MALFORMED)
    return q->row_credit;
  if (index == CASE_BULK)
    return q->bulk_credit;
  return q->control_credit / 4;
}
static bool event_request(qualifier *q, fixture_event *v, unsigned index,
                          ria_error *e) {
  ria_transport *t = index == CASE_BULK ? &q->bulk : &q->control;
  v->start = nanoseconds();
  if (!v->start)
    return ria_fail(e, RIA_EXECUTOR_ERROR, "fixture monotonic clock failed");
  if (q->server) {
    ria_header h;
    uint8_t *payload = NULL;
    bool ok = receive_frame(q, t, &h, &payload, e);
    if (ok) {
      track_owned(q, h.payload_length);
      uint8_t actual[64], expected[64];
      ok = ria_header_encode(&h, actual, e) &&
           ria_header_encode(&v->request, expected, e) &&
           !memcmp(actual, expected, 64) &&
           !memcmp(payload, v->payload, (size_t)h.payload_length);
    }
    free(payload);
    if (!ok)
      return ria_fail(e, e->code ? e->code : RIA_IDENTITY_MISMATCH,
                      "fixture request bytes or transport mismatch");
  }
  if (!payload_request(q, v, e) ||
      !ria_binding_receive(&q->binding, &v->request, index == CASE_BULK, e) ||
      !ria_binding_admit(&q->binding, &v->request, case_cost(q, index),
                         ria_monotonic_ms(), e) ||
      !frame_hash(&v->request, v->payload, v->result.request_hash, e))
    return false;
  return q->server || send_frame(q, t, &v->request, v->payload, e);
}
static bool cancellation(qualifier *q, fixture_event *v, uint32_t expected,
                         ria_error *e) {
  uint64_t before = q->binding.reserved_bytes;
  uint32_t state;
  if (!ria_binding_cancel(&q->binding, ria_read_u64(v->payload), &state, e) ||
      state != expected || before != q->binding.reserved_bytes)
    return ria_fail(e, RIA_INTERNAL_ERROR,
                    "cancel state or retained credit mismatch");
  q->checks |= expected == 0 ? 1u << 3 : 1u << 4;
  return true;
}
static bool payload_reply(qualifier *q, fixture_event *v, const ria_header *h,
                          const uint8_t *p, size_t n, ria_error *e) {
  if (h->status || h->kind == RIA_HEALTH) {
    ria_json_doc d = {0};
    bool ok = control(q, h, (uint8_t *)p, &d, e);
    track_owned(q, n + d.allocated_bytes);
    ria_json_free(&d);
    return ok;
  }
  if (h->kind == RIA_EXPERT) {
    ria_expert_request r;
    return ria_expert_parse(v->payload, (size_t)v->request.payload_length,
                            h->kind, &operation, &q->limits, &r, e) &&
           ria_expert_response(p, n, &r, &operation, e);
  }
  if (h->kind == RIA_ROWS) {
    ria_row_request r;
    return ria_rows_parse(v->payload, (size_t)v->request.payload_length, &table,
                          &q->limits, &r, e) &&
           ria_rows_response(p, n, &r, &table, e);
  }
  if (h->kind == RIA_CHUNK) {
    const uint8_t *data;
    size_t length;
    return ria_chunk_response(p, n, 3, 0, q->chunk_bytes,
                              (uint32_t)q->chunk_bytes, q->chunk + 32, &data,
                              &length, e);
  }
  uint32_t state;
  return ria_cancel_response(p, n, ria_read_u64(v->payload), 1, &state, e) &&
         state == ria_read_u32(v->reply + 16);
}
static bool event_reply(qualifier *q, fixture_event *v, unsigned index,
                        ria_error *e) {
  const uint8_t *expected = index == CASE_BULK ? q->chunk : v->reply;
  size_t n = index == CASE_BULK ? q->chunk_bytes + 64 : v->reply_length;
  ria_header h = header(q, v->request.kind, v->request.request_id, n, true,
                        v->result.status);
  ria_transport *t = index == CASE_BULK ? &q->bulk : &q->control;
  uint8_t *received = NULL;
  if (q->server) {
    if (!payload_reply(q, v, &h, expected, n, e) ||
        !send_frame(q, t, &h, expected, e))
      return false;
  } else {
    ria_header actual;
    if (!receive_frame(q, t, &actual, &received, e))
      return false;
    track_owned(q, actual.payload_length);
    uint8_t a[64], b[64];
    bool ok = ria_header_encode(&actual, a, e) && ria_header_encode(&h, b, e) &&
              !memcmp(a, b, 64) && !memcmp(received, expected, n) &&
              payload_reply(q, v, &actual, received, n, e);
    free(received);
    if (!ok)
      return ria_fail(e, e->code ? e->code : RIA_INTEGRITY_ERROR,
                      "fixture reply bytes or typed payload mismatch");
  }
  if (!ria_binding_response(&q->binding, &h, e) ||
      !frame_hash(&h, expected, v->result.reply_hash, e) ||
      !ria_binding_terminal_channel(&q->binding, h.request_id,
                                    index == CASE_BULK, true, e))
    return false;
  uint64_t now = nanoseconds();
  if (!now || now < v->start)
    return ria_fail(e, RIA_EXECUTOR_ERROR, "fixture monotonic clock failed");
  v->result.elapsed_ns = now - v->start;
  v->result.reply_bytes = n + 64;
  if (index == CASE_MALFORMED)
    q->checks |= 1u << 5;
  q->checks |= 1u << 7;
  return true;
}
static bool episode(qualifier *q, uint64_t sequence, bool measured,
                    uint32_t iteration, ria_error *e) {
  fixture_event events[CASE_COUNT];
  if (!events_prepare(q, events, sequence))
    return false;
  static const unsigned requests[] = {
      CASE_EXPERT_CANCELLED, CASE_EXPERT_SUCCESS, CASE_ROWS,
      CASE_HEALTH,           CASE_CANCEL_PENDING, CASE_BULK};
  for (unsigned i = 0; i < sizeof requests / sizeof *requests; i++)
    if (!event_request(q, &events[requests[i]], requests[i], e))
      return false;
  if (!cancellation(q, &events[CASE_CANCEL_PENDING], 0, e))
    return false;
  uint64_t initial_credit = q->binding.reserved_bytes;
  char health[256];
  int health_n = snprintf(health, sizeof health,
                          "{\"counters\":{\"reserved_bytes\":%llu},\"ready\":"
                          "true,\"state\":\"fixture\"}",
                          (unsigned long long)initial_credit);
  if (health_n < 0 || (size_t)health_n >= sizeof health)
    return ria_fail(e, RIA_INTERNAL_ERROR, "fixture health encoding failed");
  memcpy(events[CASE_HEALTH].reply, health, (size_t)health_n);
  events[CASE_HEALTH].reply_length = (size_t)health_n;
  static const unsigned replies[] = {
      CASE_HEALTH,           CASE_ROWS,          CASE_BULK, CASE_CANCEL_PENDING,
      CASE_EXPERT_CANCELLED, CASE_EXPERT_SUCCESS};
  for (unsigned i = 0; i < sizeof replies / sizeof *replies; i++) {
    if (!event_reply(q, &events[replies[i]], replies[i], e))
      return false;
    if (i == 2) {
      if (q->binding.reserved_bytes < q->expert_credit)
        return ria_fail(e, RIA_INTERNAL_ERROR,
                        "protected progress released expert credit early");
      q->checks |= 1u << 2;
    }
  }
  if (q->binding.reserved_bytes)
    return ria_fail(e, RIA_INTERNAL_ERROR,
                    "fixture terminal credit did not quiesce");
  const unsigned final[] = {CASE_CANCEL_TERMINAL, CASE_CANCEL_UNKNOWN,
                            CASE_MALFORMED};
  for (unsigned i = 0; i < sizeof final / sizeof *final; i++) {
    unsigned index = final[i];
    if (!event_request(q, &events[index], index, e) ||
        (i < 2 && !cancellation(q, &events[index], i + 1, e)) ||
        !event_reply(q, &events[index], index, e))
      return false;
  }
  if (q->binding.reserved_bytes)
    return ria_fail(e, RIA_INTERNAL_ERROR,
                    "fixture failure credit did not quiesce");
  if (measured)
    for (unsigned i = 0; i < CASE_COUNT; i++) {
      if (q->case_count == q->case_capacity)
        return ria_fail(e, RIA_INTERNAL_ERROR,
                        "fixture case capacity mismatch");
      events[i].result.iteration = iteration;
      q->cases[q->case_count++] = events[i].result;
    }
  return true;
}
static bool partial_deadline(qualifier *q, ria_error *e) {
  uint64_t now_ms = ria_monotonic_ms();
  if (now_ms >= q->deadline ||
      q->deadline - now_ms <= q->limits.frame_io_timeout_ms + 100)
    return ria_fail(
        e, RIA_DEADLINE_EXCEEDED,
        "insufficient registered time for partial-frame deadline case");
  uint64_t start = nanoseconds();
  if (!q->server) {
    ria_header next =
        header(q, RIA_HEALTH, 2 + (q->warmup + q->repeats) * 16, 2, false, 0);
    uint8_t encoded[64];
    if (!ria_header_encode(&next, encoded, e) ||
        !ria_transport_write(&q->control, encoded, 1,
                             phase_deadline(q, q->limits.write_timeout_ms), e))
      return false;
  }
  ria_header unused;
  uint8_t *payload = NULL;
  ria_error expected = {0};
  bool ok = ria_transport_frame(
      &q->control, phase_deadline(q, q->limits.operation_timeout_ms),
      q->limits.frame_io_timeout_ms, q->limits.frame_payload_bytes, &unused,
      &payload, &expected);
  free(payload);
  uint64_t end = nanoseconds();
  if (!start || !end || end < start || ok ||
      (q->server && expected.code != RIA_DEADLINE_EXCEEDED))
    return ria_fail(
        e, RIA_INTERNAL_ERROR,
        "partial fixture frame did not fail at the production deadline");
  q->timeout_ns = end - start;
  /* The server times the started partial frame. The client observes that exact
   * peer timeout as a close; a quick unrelated disconnect cannot pass. */
  uint64_t floor = (q->limits.frame_io_timeout_ms - 1) * UINT64_C(1000000);
  if (q->timeout_ns < floor)
    return ria_fail(e, RIA_INTERNAL_ERROR,
                    "partial frame failed before its declared timeout");
  ria_binding_invalidate(&q->binding);
  ria_transport_close(&q->control);
  q->checks |= 1u << 6;
  return true;
}

typedef struct {
  char *bytes;
  size_t length, capacity;
} report_buffer;
static bool report_append(report_buffer *b, ria_error *e, const char *format,
                          ...) RIA_PRINTF(3, 4);
static bool report_append(report_buffer *b, ria_error *e, const char *format,
                          ...) {
  va_list args;
  va_start(args, format);
  int n =
      vsnprintf(b->bytes + b->length, b->capacity - b->length, format, args);
  va_end(args);
  if (n < 0 || (size_t)n >= b->capacity - b->length)
    return ria_fail(e, RIA_RESOURCE_LIMIT,
                    "transport report exceeds bounded capacity");
  b->length += (size_t)n;
  return true;
}
static uint64_t cpu_time(const struct rusage *u) {
  return ((uint64_t)u->ru_utime.tv_sec + (uint64_t)u->ru_stime.tv_sec) *
             UINT64_C(1000000000) +
         ((uint64_t)u->ru_utime.tv_usec + (uint64_t)u->ru_stime.tv_usec) *
             UINT64_C(1000);
}
static bool report(qualifier *q, char **out, size_t *length, ria_error *e) {
  struct rusage usage;
  uint64_t end = nanoseconds();
  if (!end || end < q->started || getrusage(RUSAGE_SELF, &usage))
    return ria_fail(e, RIA_EXECUTOR_ERROR,
                    "transport resource measurement failed");
  uint64_t elapsed = end - q->started, cpu_start = cpu_time(&q->initial_usage),
           cpu_end = cpu_time(&usage);
  if (cpu_end < cpu_start || usage.ru_maxrss < 0 ||
      elapsed > RIA_JSON_SAFE_INTEGER ||
      q->startup_ns > RIA_JSON_SAFE_INTEGER ||
      q->timeout_ns > RIA_JSON_SAFE_INTEGER ||
      cpu_end - cpu_start > RIA_JSON_SAFE_INTEGER || q->checks != 255u ||
      q->case_count != q->case_capacity)
    return ria_fail(e, RIA_INTERNAL_ERROR,
                    "transport measurement invariants failed");
  size_t capacity = q->case_count * 512 + 8192;
  report_buffer b = {.bytes = malloc(capacity), .capacity = capacity};
  if (!b.bytes)
    return ria_fail(e, RIA_RESOURCE_LIMIT,
                    "transport report allocation failed");
  track_owned(q, capacity);
  bool ok = report_append(
      &b, e,
      "{\"schema_revision\":1,\"kind\":\"native_transport_measurements\","
      "\"qualified\":false,"
      "\"qualification_scope\":\"initial_fixture\",\"role\":\"%s\"",
      q->server ? "expert" : "client");
  char hex[65];
  for (unsigned i = 0; ok && i < 8; i++) {
    ria_hex_encode(q->identities[i], 32, hex);
    ok = report_append(&b, e, ",\"%s\":\"%s\"", identity_names[i], hex);
  }
  ria_hex_encode(q->peer, 32, hex);
  if (ok)
    ok = report_append(
        &b, e,
        ",\"peer_certificate_digest\":\"%s\",\"warmup_completed\":%llu,"
        "\"iterations_completed\":%llu,"
        "\"startup_ns\":%llu,\"elapsed_ns\":%llu,\"cpu_ns\":%llu,\"max_rss_"
        "bytes\":\"%llu\","
        "\"owned_buffers_peak_bytes\":\"%llu\",\"measured_request_bytes\":\"%"
        "llu\","
        "\"measured_response_bytes\":\"%llu\",\"timeout_elapsed_ns\":%llu,"
        "\"checks\":[",
        hex, (unsigned long long)q->warmup, (unsigned long long)q->repeats,
        (unsigned long long)q->startup_ns, (unsigned long long)elapsed,
        (unsigned long long)(cpu_end - cpu_start),
        (unsigned long long)((uint64_t)usage.ru_maxrss * 1024),
        (unsigned long long)q->owned_peak, (unsigned long long)q->request_bytes,
        (unsigned long long)q->reply_bytes, (unsigned long long)q->timeout_ns);
  for (unsigned i = 0; ok && i < 8; i++)
    ok = report_append(&b, e, "%s{\"id\":\"%s\",\"passed\":true}", i ? "," : "",
                       check_names[i]);
  if (ok)
    ok = report_append(&b, e, "],\"cases\":[");
  for (size_t i = 0; ok && i < q->case_count; i++) {
    transport_case *v = &q->cases[i];
    char request_hash[65], reply_hash[65];
    ria_hex_encode(v->request_hash, 32, request_hash);
    ria_hex_encode(v->reply_hash, 32, reply_hash);
    if (v->elapsed_ns > RIA_JSON_SAFE_INTEGER) {
      ok = ria_fail(e, RIA_INTERNAL_ERROR,
                    "case timing exceeds JSON numeric contract");
      break;
    }
    ok = report_append(&b, e,
                       "%s{\"iteration\":%u,\"id\":\"%s\",\"kind\":%u,"
                       "\"status\":%u,\"request_bytes\":\"%llu\","
                       "\"reply_bytes\":\"%llu\",\"request_sha256\":\"%s\","
                       "\"response_sha256\":\"%s\",\"elapsed_ns\":%llu}",
                       i ? "," : "", v->iteration, v->id, (unsigned)v->kind,
                       v->status, (unsigned long long)v->request_bytes,
                       (unsigned long long)v->reply_bytes, request_hash,
                       reply_hash, (unsigned long long)v->elapsed_ns);
  }
  if (ok)
    ok = report_append(&b, e, "]}");
  ria_json_doc d = {0};
  if (ok)
    ok = ria_json_parse(
        b.bytes, b.length,
        (ria_json_limits){capacity, q->case_count * 32 + 256, 16}, &d, e);
  char owned_text[21], rss_text[21];
  if (ok) {
    /* Exact canonical writer capacity from the production allocator; root
     * object sorting owns at most 64 pointers, case objects fewer. */
    uint64_t canonical_bytes =
        d.string_bytes * UINT64_C(6) + (uint64_t)d.count * 64 + 1;
    track_owned(q, capacity + d.allocated_bytes + canonical_bytes +
                       64 * sizeof(void *));
    int n = snprintf(owned_text, sizeof owned_text, "%llu",
                     (unsigned long long)q->owned_peak);
    ria_json_node *node =
        &d.nodes[ria_json_get(&d, 0, "owned_buffers_peak_bytes")];
    node->text = owned_text;
    node->length = (size_t)n;
    ok = ria_json_canonical(&d, false, out, length, e);
    if (ok && getrusage(RUSAGE_SELF, &usage))
      ok = ria_fail(e, RIA_EXECUTOR_ERROR,
                    "transport peak RSS measurement failed");
    if (ok) {
      n = snprintf(rss_text, sizeof rss_text, "%llu",
                   (unsigned long long)((uint64_t)usage.ru_maxrss * 1024));
      node = &d.nodes[ria_json_get(&d, 0, "max_rss_bytes")];
      node->text = rss_text;
      node->length = (size_t)n;
      free(*out);
      *out = NULL;
      ok = ria_json_canonical(&d, false, out, length, e);
    }
  }
  if (!ok) {
    free(*out);
    *out = NULL;
    *length = 0;
  }
  ria_json_free(&d);
  free(b.bytes);
  return ok;
}
bool ria_qualify_transport(const char *config_path, const char *request_path,
                           char **report_json, size_t *report_length,
                           ria_error *e) {
  if (!config_path || !request_path || !report_json || !report_length || !e)
    return ria_fail(e, RIA_INVALID_REQUEST,
                    "invalid transport qualification arguments");
  *report_json = NULL;
  *report_length = 0;
  memset(e, 0, sizeof *e);
  if (!ria_disable_dumps(e))
    return false;
  qualifier q = {
      .listeners = {-1, -1}, .control = {.fd = -1}, .bulk = {.fd = -1}};
  q.started = nanoseconds();
  bool ok = q.started && !getrusage(RUSAGE_SELF, &q.initial_usage) &&
            configuration(&q, config_path, request_path, e) && bind_pair(&q, e);
  for (uint64_t i = 0; ok && i < q.warmup; i++)
    ok = episode(&q, i, false, 0, e);
  q.request_bytes = q.reply_bytes = 0;
  for (uint64_t i = 0; ok && i < q.repeats; i++)
    ok = episode(&q, q.warmup + i, true, (uint32_t)i, e);
  if (ok)
    ok = partial_deadline(&q, e) && report(&q, report_json, report_length, e);
  ria_transport_close(&q.control);
  ria_transport_close(&q.bulk);
  ria_tls_destroy(&q.tls);
  for (unsigned i = 0; i < 2; i++)
    if (q.listeners[i] >= 0)
      close(q.listeners[i]);
  ria_json_free(&q.config);
  ria_json_free(&q.request);
  free(q.cases);
  free(q.chunk);
  OPENSSL_cleanse(q.capability, sizeof q.capability);
  if (!ok && !e->code)
    return ria_fail(
        e, RIA_EXECUTOR_ERROR,
        "transport qualification failed before measurement completion");
  return ok;
}
