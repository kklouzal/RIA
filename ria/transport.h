#ifndef RIA_TRANSPORT_H
#define RIA_TRANSPORT_H
#include "protocol.h"
#include <sys/socket.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
  const char *ca_file, *certificate_file, *private_key_file,
      *expected_peer_name;
  bool server;
  uint64_t handshake_timeout_ms;
} ria_tls_config;
typedef struct {
  void *context;
  bool server;
  char expected_peer_name[256];
  uint64_t handshake_timeout_ms;
} ria_tls;
typedef struct {
  void *ssl;
  int fd;
  bool owns_fd, unusable;
} ria_transport;
/* Connections have one serialized I/O/cleanup owner; concurrent SSL use is
 * prohibited. A published TLS context remains immutable until destruction.
 * The process owner must ignore SIGPIPE or block it in every connection owner
 * thread before TLS use: the pinned OpenSSL socket BIO performs Unix write().
 * RIA engine startup and expert service startup establish this policy. */
/* Provisioned credential paths remain immutable through creation. Each input
 * is a nonempty regular PEM file of at most 256KiB; encrypted keys are rejected
 * without interactive callbacks. Deployment mounts these inputs read-only. */
bool ria_tls_create(ria_tls *tls, const ria_tls_config *config,
                    ria_error *error);
void ria_tls_destroy(ria_tls *tls);
/* Caller supplies a validated numeric address; no DNS or resolver blocking
 * occurs here. Connect has a finite deadline. */
bool ria_transport_connect(ria_transport *transport, ria_tls *tls,
                           const struct sockaddr *address, socklen_t length,
                           uint64_t connect_timeout_ms, ria_error *error);
/* Adopt fd on success; caller retains ownership on failure. mTLS/SAN checked
 * here. */
bool ria_transport_accept(ria_transport *transport, ria_tls *tls, int fd,
                          ria_error *error);
bool ria_transport_read(ria_transport *transport, void *buffer, size_t length,
                        uint64_t deadline_ms, ria_error *error);
bool ria_transport_write(ria_transport *transport, const void *buffer,
                         size_t length, uint64_t deadline_ms, ria_error *error);
/* First byte uses operation deadline; a started frame also uses
 * frame_io_timeout_ms. */
bool ria_transport_frame(ria_transport *transport,
                         uint64_t first_byte_deadline_ms,
                         uint64_t frame_io_timeout_ms, uint64_t frame_limit,
                         ria_header *header, uint8_t **payload,
                         ria_error *error);
bool ria_transport_send(ria_transport *transport, const ria_header *header,
                        const void *payload, uint64_t deadline_ms,
                        ria_error *error);
bool ria_transport_peer_digest(const ria_transport *transport,
                               uint8_t digest[32], ria_error *error);
/* Any I/O failure retires the connection. Close consumes SSL/fd ownership;
 * close_notify is best effort and is skipped after an abandoned/fatal I/O. */
void ria_transport_close(ria_transport *transport);
#ifdef __cplusplus
}
#endif
#endif
