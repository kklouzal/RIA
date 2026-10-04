#define _POSIX_C_SOURCE 200809L
#include "transport.h"
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <netinet/tcp.h>
#include <openssl/err.h>
#include <openssl/ssl.h>
#include <openssl/x509v3.h>
#include <poll.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
static bool deadline(uint64_t timeout, uint64_t *end, ria_error *e) {
  uint64_t now = ria_monotonic_ms();
  if (!timeout || now == UINT64_MAX || !ria_u64_add(now, timeout, end))
    return ria_fail(e, RIA_INVALID_REQUEST, "invalid I/O deadline");
  return true;
}
static bool ready(int fd, short events, uint64_t end, ria_error *e) {
  for (;;) {
    uint64_t now = ria_monotonic_ms();
    if (now == UINT64_MAX || now >= end)
      return ria_fail(e, RIA_DEADLINE_EXCEEDED, "transport deadline exceeded");
    uint64_t left = end - now;
    struct pollfd p = {fd, events, 0};
    int rc = poll(&p, 1, left > INT_MAX ? INT_MAX : (int)left);
    if (rc < 0 && errno == EINTR)
      continue;
    if (rc < 0)
      return ria_fail(e, RIA_INTERNAL_ERROR, "transport poll failed");
    if (!rc)
      continue;
    if (p.revents & POLLNVAL)
      return ria_fail(e, RIA_INTERNAL_ERROR, "transport descriptor invalid");
    if (p.revents & (events | POLLERR | POLLHUP))
      return true;
  }
}
static int no_password(char *buffer, int size, int writing, void *context) {
  (void)buffer;
  (void)size;
  (void)writing;
  (void)context;
  return 0; /* Runtime credentials never trigger interactive input. */
}
static bool credential_file(const char *path, ria_error *e) {
  int fd = open(path, O_RDONLY | O_CLOEXEC | O_NONBLOCK);
  if (fd < 0)
    return ria_fail(e, RIA_UNAUTHORIZED,
                    "cannot open provisioned TLS credential");
  struct stat metadata;
  bool ok = fstat(fd, &metadata) == 0 && S_ISREG(metadata.st_mode) &&
            metadata.st_size > 0 && metadata.st_size <= 262144;
  if (close(fd) != 0)
    ok = false;
  return ok || ria_fail(e, RIA_UNAUTHORIZED,
                        "TLS credential is not a bounded regular PEM file");
}
bool ria_tls_create(ria_tls *t, const ria_tls_config *c, ria_error *e) {
  if (!t || !c || !c->handshake_timeout_ms)
    return ria_fail(e, RIA_INVALID_REQUEST, "incomplete transport configuration");
  memset(t, 0, sizeof *t);
  if (c->plaintext) {
    if (c->ca_file || c->certificate_file || c->private_key_file ||
        c->expected_peer_name)
      return ria_fail(e, RIA_INVALID_REQUEST,
                      "trusted-network transport does not accept TLS credentials");
    t->plaintext = true;
    t->server = c->server;
    t->handshake_timeout_ms = c->handshake_timeout_ms;
    return true;
  }
  if (!c->ca_file || !c->certificate_file || !c->private_key_file ||
      !c->expected_peer_name || !c->expected_peer_name[0] ||
      strlen(c->expected_peer_name) >= sizeof t->expected_peer_name)
    return ria_fail(e, RIA_INVALID_REQUEST, "incomplete TLS configuration");
  if (!credential_file(c->ca_file, e) ||
      !credential_file(c->certificate_file, e) ||
      !credential_file(c->private_key_file, e))
    return false;
  ERR_clear_error();
  SSL_CTX *ctx =
      SSL_CTX_new(c->server ? TLS_server_method() : TLS_client_method());
  if (!ctx)
    return ria_fail(e, RIA_INTERNAL_ERROR, "TLS context allocation failed");
  SSL_CTX_set_default_passwd_cb(ctx, no_password);
  bool ok = SSL_CTX_set_min_proto_version(ctx, TLS1_3_VERSION) == 1 &&
            SSL_CTX_set_max_proto_version(ctx, TLS1_3_VERSION) == 1 &&
            SSL_CTX_set_max_early_data(ctx, 0) == 1 &&
            SSL_CTX_load_verify_locations(ctx, c->ca_file, NULL) == 1 &&
            SSL_CTX_use_certificate_chain_file(ctx, c->certificate_file) == 1 &&
            SSL_CTX_use_PrivateKey_file(ctx, c->private_key_file,
                                        SSL_FILETYPE_PEM) == 1 &&
            SSL_CTX_check_private_key(ctx) == 1;
  if (!ok) {
    SSL_CTX_free(ctx);
    return ria_fail(e, RIA_UNAUTHORIZED,
                    "TLS credentials or trust configuration invalid");
  }
  SSL_CTX_set_verify(
      ctx, SSL_VERIFY_PEER | (c->server ? SSL_VERIFY_FAIL_IF_NO_PEER_CERT : 0),
      NULL);
  SSL_CTX_set_verify_depth(ctx, 8);
  SSL_CTX_set_max_cert_list(ctx, 262144);
  SSL_CTX_set_options(ctx, SSL_OP_NO_RENEGOTIATION | SSL_OP_NO_COMPRESSION);
  SSL_CTX_set_session_cache_mode(ctx, SSL_SESS_CACHE_OFF);
  SSL_CTX_set_num_tickets(ctx, 0);
  t->context = ctx;
  t->server = c->server;
  t->handshake_timeout_ms = c->handshake_timeout_ms;
  memcpy(t->expected_peer_name, c->expected_peer_name,
         strlen(c->expected_peer_name) + 1);
  return true;
}
void ria_tls_destroy(ria_tls *t) {
  if (t) {
    SSL_CTX_free(t->context);
    memset(t, 0, sizeof *t);
  }
}
static bool nonblocking(int fd, ria_error *e) {
  int flags = fcntl(fd, F_GETFL);
  if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0)
    return ria_fail(e, RIA_INTERNAL_ERROR,
                    "cannot configure nonblocking transport");
  struct sockaddr_storage address;
  socklen_t n = sizeof address;
  if (getsockname(fd, (struct sockaddr *)&address, &n) != 0)
    return ria_fail(e, RIA_INTERNAL_ERROR, "cannot inspect socket");
  if (address.ss_family == AF_INET || address.ss_family == AF_INET6) {
    int one = 1;
    if (setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one) != 0)
      return ria_fail(e, RIA_INTERNAL_ERROR,
                      "cannot configure short-packet transport");
  }
  return true;
}
static bool wait_ssl(SSL *ssl, int result, int fd, uint64_t end, ria_error *e) {
  int code = SSL_get_error(ssl, result);
  if (code == SSL_ERROR_WANT_READ)
    return ready(fd, POLLIN, end, e);
  if (code == SSL_ERROR_WANT_WRITE)
    return ready(fd, POLLOUT, end, e);
  if (code == SSL_ERROR_ZERO_RETURN)
    return ria_fail(e, RIA_NOT_READY, "peer closed authenticated transport");
  return ria_fail(e, RIA_INTEGRITY_ERROR, "authenticated transport failed");
}
static bool establish(ria_transport *t, ria_tls *ctx, int fd, bool owns,
                      ria_error *e) {
  if (!t || !ctx || (!ctx->plaintext && !ctx->context))
    return ria_fail(e, RIA_INVALID_REQUEST, "invalid transport context");
  memset(t, 0, sizeof *t);
  t->fd = -1;
  if (!nonblocking(fd, e))
    return false;
  if (ctx->plaintext) {
    t->plaintext = true;
    t->fd = fd;
    t->owns_fd = owns;
    return true;
  }
  SSL *ssl = SSL_new(ctx->context);
  if (!ssl)
    return ria_fail(e, RIA_RESOURCE_LIMIT, "TLS connection allocation failed");
  X509_VERIFY_PARAM *verify = SSL_get0_param(ssl);
  X509_VERIFY_PARAM_set_hostflags(verify,
                                  X509_CHECK_FLAG_NEVER_CHECK_SUBJECT |
                                      X509_CHECK_FLAG_NO_PARTIAL_WILDCARDS);
  uint8_t address[16];
  bool ip = inet_pton(AF_INET, ctx->expected_peer_name, address) == 1 ||
            inet_pton(AF_INET6, ctx->expected_peer_name, address) == 1;
  int named =
      ip ? X509_VERIFY_PARAM_set1_ip_asc(verify, ctx->expected_peer_name)
         : SSL_set1_host(ssl, ctx->expected_peer_name);
  if (named != 1 || SSL_set_fd(ssl, fd) != 1 ||
      (!ctx->server && !ip &&
       SSL_set_tlsext_host_name(ssl, ctx->expected_peer_name) != 1)) {
    SSL_free(ssl);
    return ria_fail(e, RIA_UNAUTHORIZED, "peer identity policy invalid");
  }
  if (ctx->server)
    SSL_set_accept_state(ssl);
  else
    SSL_set_connect_state(ssl);
  uint64_t end;
  if (!deadline(ctx->handshake_timeout_ms, &end, e)) {
    SSL_free(ssl);
    return false;
  }
  for (;;) {
    if (ria_monotonic_ms() >= end) {
      SSL_free(ssl);
      return ria_fail(e, RIA_DEADLINE_EXCEEDED,
                      "TLS handshake deadline exceeded");
    }
    /* SSL_get_error requires the calling thread's queue to be empty before I/O. */
    ERR_clear_error();
    int rc = SSL_do_handshake(ssl);
    if (rc == 1)
      break;
    if (!wait_ssl(ssl, rc, fd, end, e)) {
      SSL_free(ssl);
      return false;
    }
  }
  X509 *peer = SSL_get1_peer_certificate(ssl);
  bool authenticated = peer && SSL_get_verify_result(ssl) == X509_V_OK &&
                       SSL_version(ssl) == TLS1_3_VERSION;
  X509_free(peer);
  if (!authenticated) {
    SSL_free(ssl);
    return ria_fail(e, RIA_UNAUTHORIZED,
                    "peer certificate authorization failed");
  }
  t->ssl = ssl;
  t->fd = fd;
  t->owns_fd = owns;
  return true;
}
bool ria_transport_connect(ria_transport *t, ria_tls *ctx,
                           const struct sockaddr *address, socklen_t n,
                           uint64_t timeout, ria_error *e) {
  if (!address || !ctx || ctx->server ||
      (address->sa_family != AF_INET && address->sa_family != AF_INET6))
    return ria_fail(e, RIA_INVALID_REQUEST, "invalid connect address");
  uint64_t end;
  if (!deadline(timeout, &end, e))
    return false;
  int fd = socket(address->sa_family, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (fd < 0)
    return ria_fail(e, RIA_RESOURCE_LIMIT, "socket creation failed");
  if (!nonblocking(fd, e)) {
    close(fd);
    return false;
  }
  if (connect(fd, address, n) != 0) {
    if (errno != EINPROGRESS) {
      close(fd);
      return ria_fail(e, RIA_NOT_READY, "peer connection failed");
    }
    if (!ready(fd, POLLOUT, end, e)) {
      close(fd);
      return false;
    }
    int status = 0;
    socklen_t len = sizeof status;
    if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &status, &len) != 0 || status) {
      close(fd);
      return ria_fail(e, RIA_NOT_READY, "peer connection failed");
    }
  }
  if (!establish(t, ctx, fd, true, e)) {
    close(fd);
    return false;
  }
  return true;
}
bool ria_transport_accept(ria_transport *t, ria_tls *ctx, int fd,
                          ria_error *e) {
  if (!ctx || !ctx->server || fd < 0)
    return ria_fail(e, RIA_INVALID_REQUEST, "invalid accepted transport socket");
  return establish(t, ctx, fd, true, e);
}
bool ria_transport_step(ria_transport *t, bool writing, void *buffer,
                        size_t length, size_t *amount, short *wait_event,
                        ria_error *e) {
  if (!t || (!t->plaintext && !t->ssl) || t->fd < 0 || t->unusable)
    return ria_fail(e, RIA_NOT_READY, "transport is not established");
  if (!buffer || !length || !amount || !wait_event || length > SSIZE_MAX) {
    t->unusable = true;
    return ria_fail(e, RIA_INVALID_REQUEST, "invalid transport transfer");
  }
  *amount = 0;
  *wait_event = 0;
  if (t->plaintext) {
    ssize_t count = writing ? send(t->fd, buffer, length, MSG_NOSIGNAL)
                            : recv(t->fd, buffer, length, 0);
    if (count > 0) {
      *amount = (size_t)count;
      return true;
    }
    if (count < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) {
      *wait_event = writing ? POLLOUT : POLLIN;
      return true;
    }
    t->unusable = true;
    return ria_fail(e, RIA_NOT_READY, "trusted-network channel disconnected/failed");
  }
  ERR_clear_error();
  int result = writing ? SSL_write_ex(t->ssl, buffer, length, amount)
                       : SSL_read_ex(t->ssl, buffer, length, amount);
  if (result == 1) {
    if (*amount && *amount <= length)
      return true;
    t->unusable = true;
    return ria_fail(e, RIA_INTERNAL_ERROR, "TLS transfer made invalid progress");
  }
  int code = SSL_get_error(t->ssl, result);
  if (code == SSL_ERROR_WANT_READ || code == SSL_ERROR_WANT_WRITE) {
    *amount = 0;
    *wait_event = code == SSL_ERROR_WANT_READ ? POLLIN : POLLOUT;
    return true;
  }
  t->unusable = true;
  return ria_fail(e, code == SSL_ERROR_ZERO_RETURN ? RIA_NOT_READY : RIA_INTEGRITY_ERROR,
                  "authenticated channel disconnected/failed");
}
bool ria_transport_pending(const ria_transport *t) {
  return t && !t->plaintext && t->ssl && !t->unusable && SSL_pending(t->ssl) > 0;
}
static bool transfer(ria_transport *t, void *buffer, size_t length,
                     uint64_t end, bool write, ria_error *e) {
  if (!t || (!t->plaintext && !t->ssl) || t->fd < 0 || t->unusable || (!buffer && length))
    return ria_fail(e, RIA_NOT_READY, "transport is not established");
  size_t done = 0;
  while (done < length) {
    uint64_t now = ria_monotonic_ms();
    if (now == UINT64_MAX || now >= end) {
      t->unusable = true;
      return ria_fail(e, RIA_DEADLINE_EXCEEDED,
                      "transport I/O deadline exceeded");
    }
    size_t amount = 0;
    short event = 0;
    if (!ria_transport_step(t, write, (unsigned char *)buffer + done,
                            length - done, &amount, &event, e))
      return false;
    if (amount) {
      done += amount;
    } else if (!ready(t->fd, event, end, e)) {
      t->unusable = true;
      return false;
    }
  }
  return true;
}
bool ria_transport_read(ria_transport *t, void *b, size_t n, uint64_t end,
                        ria_error *e) {
  return transfer(t, b, n, end, false, e);
}
bool ria_transport_write(ria_transport *t, const void *b, size_t n,
                         uint64_t end, ria_error *e) {
  return transfer(t, (void *)b, n, end, true, e);
}
bool ria_transport_frame(ria_transport *t, uint64_t first, uint64_t timeout,
                         uint64_t limit, ria_header *h, uint8_t **payload,
                         ria_error *e) {
  if (!payload)
    return ria_fail(e, RIA_INVALID_REQUEST, "missing frame output");
  *payload = NULL;
  uint8_t bytes[64];
  if (!ria_transport_read(t, bytes, 1, first, e))
    return false;
  uint64_t end;
  if (!deadline(timeout, &end, e)) {
    t->unusable = true;
    return false;
  }
  if (end > first)
    end = first;
  if (!ria_transport_read(t, bytes + 1, 63, end, e) ||
      !ria_header_decode(bytes, limit, h, e)) {
    t->unusable = true;
    return false;
  }
  size_t n;
  if (!ria_size(h->payload_length, &n, e)) {
    t->unusable = true;
    return false;
  }
  if (!n)
    return true;
  uint8_t *p = malloc(n);
  if (!p) {
    t->unusable = true;
    return ria_fail(e, RIA_RESOURCE_LIMIT, "frame buffer allocation failed");
  }
  if (!ria_transport_read(t, p, n, end, e)) {
    free(p);
    return false;
  }
  *payload = p;
  return true;
}
bool ria_transport_send(ria_transport *t, const ria_header *h, const void *p,
                        uint64_t end, ria_error *e) {
  if (!h || (!p && h->payload_length))
    return ria_fail(e, RIA_INVALID_REQUEST, "missing send payload");
  uint8_t bytes[64];
  size_t n;
  if (!ria_header_encode(h, bytes, e) || !ria_size(h->payload_length, &n, e))
    return false;
  return ria_transport_write(t, bytes, 64, end, e) &&
         ria_transport_write(t, p, n, end, e);
}
bool ria_transport_peer_digest(const ria_transport *t, uint8_t digest[32],
                               ria_error *e) {
  if (!t || t->plaintext || !t->ssl || t->unusable || !digest)
    return ria_fail(e, RIA_NOT_READY, "transport peer is unavailable");
  X509 *peer = SSL_get1_peer_certificate(t->ssl);
  unsigned length = 0;
  bool ok = peer && X509_digest(peer, EVP_sha256(), digest, &length) == 1 &&
            length == 32;
  X509_free(peer);
  return ok || ria_fail(e, RIA_UNAUTHORIZED,
                        "authenticated certificate digest unavailable");
}
static bool peer_address(const ria_transport *t, uint8_t address[16],
                         uint32_t *scope, ria_error *e) {
  struct sockaddr_storage peer;
  socklen_t length = sizeof peer;
  if (getpeername(t->fd, (struct sockaddr *)&peer, &length) != 0)
    return ria_fail(e, RIA_NOT_READY, "transport peer address unavailable");
  *scope = 0;
  if (peer.ss_family == AF_INET && length >= sizeof(struct sockaddr_in)) {
    const struct sockaddr_in *v4 = (const struct sockaddr_in *)&peer;
    memset(address, 0, 10);
    address[10] = address[11] = 255;
    memcpy(address + 12, &v4->sin_addr, 4);
    return true;
  }
  if (peer.ss_family == AF_INET6 && length >= sizeof(struct sockaddr_in6)) {
    const struct sockaddr_in6 *v6 = (const struct sockaddr_in6 *)&peer;
    memcpy(address, &v6->sin6_addr, 16);
    *scope = v6->sin6_scope_id;
    return true;
  }
  return ria_fail(e, RIA_INVALID_REQUEST, "transport requires an IP peer");
}
bool ria_transport_same_peer(const ria_transport *a, const ria_transport *b,
                             ria_error *e) {
  if (!a || !b || a->fd < 0 || b->fd < 0 || a->unusable || b->unusable ||
      a->plaintext != b->plaintext)
    return ria_fail(e, RIA_UNAUTHORIZED, "control/bulk transport modes differ or are unavailable");
  uint8_t left[32], right[32];
  if (a->plaintext) {
    uint32_t left_scope, right_scope;
    if (!peer_address(a, left, &left_scope, e) ||
        !peer_address(b, right, &right_scope, e))
      return false;
    return (left_scope == right_scope && !memcmp(left, right, 16)) ||
           ria_fail(e, RIA_UNAUTHORIZED, "control/bulk trusted-network peer IPs differ");
  }
  return ria_transport_peer_digest(a, left, e) &&
         ria_transport_peer_digest(b, right, e) &&
         (CRYPTO_memcmp(left, right, 32) == 0 ||
          ria_fail(e, RIA_UNAUTHORIZED, "control/bulk peer certificates differ"));
}
void ria_transport_close(ria_transport *t) {
  if (t) {
    if (t->ssl) {
      /* Fatal errors and abandoned partial operations prohibit SSL_shutdown. */
      if (!t->unusable) {
        ERR_clear_error();
        (void)SSL_shutdown(t->ssl);
      }
      SSL_free(t->ssl);
    }
    if (t->owns_fd && t->fd >= 0)
      close(t->fd);
    memset(t, 0, sizeof *t);
    t->fd = -1;
  }
}
