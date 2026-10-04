#ifndef RIA_ENGINE_H
#define RIA_ENGINE_H
#include "../ds4.h"
#include "prompt.h"
#include "remote.h"
#include "tokenizer.h"
#include "vision.h"
#ifdef __cplusplus
extern "C" {
#endif
#define RIA_IMAGE_LAYOUT 41u
/* One graph owner/session, borrowed immutable metadata. Tokenization may run
 * independently; graph/image/sync/eval are serialized by the service owner.
 * Close follows stopping/joining every graph user. */
typedef struct ria_engine ria_engine;
/* Explicit process-lifetime RIA policy: ignore SIGPIPE so an authenticated
 * socket disconnect returns an operation error. Open installs this policy
 * before role allocation; close does not restore the caller's disposition.
 * Callers must serialize process signal-policy changes with engine startup. */
bool ria_engine_process_policy(ria_error *);
/* Pure FP64 shifted-label NLL over exactly 129280 FP32 logits. Reject missing
 * buffers, out-of-vocabulary labels, nonfinite logits and nonfinite results. */
bool ria_logits_nll(const float *logits, uint32_t label, double *loss,
                    ria_error *);
typedef struct {
  char method[8], path[256];
  uint64_t body_bytes;
} ria_http_header;
bool ria_api_header(const char *bytes, size_t length, const char *bearer,
                    size_t bearer_length, uint64_t maximum_body,
                    ria_http_header *, ria_error *);
/* Provisioned credentials are bounded regular files, opened without following
 * a final symlink or blocking on a FIFO/device. Accept 1..4096 printable ASCII
 * token bytes with an optional LF/CRLF terminator. Output needs 4097 bytes;
 * failure clears it and length, without exposing credential contents. */
bool ria_api_bearer_read(const char *path, char *bearer, size_t capacity,
                         size_t *length, ria_error *);
uint64_t ria_engine_frontend_budget(const ria_engine *);
bool ria_engine_open(const char *service_path, uint64_t frontend_owner_bytes,
                     ds4_startup_cancel_fn startup_cancel, void *cancel_context,
                     ria_engine **, ria_error *);
/* Failure may retain a graph with unproved CUDA completion and its borrowed
 * TensorStore/backing. The serving owner must terminate the process; it must
 * not release external graph users/operands or attempt engine recovery. */
bool ria_engine_close(ria_engine *, ria_error *);
bool ria_engine_claim(ria_engine *, uint64_t context, ria_error *);
void ria_engine_release(ria_engine *);
const ria_service *ria_engine_service(const ria_engine *);
ria_tokenizer *ria_engine_tokenizer(ria_engine *);
uint64_t ria_engine_model_bytes(const ria_engine *);
uint64_t ria_engine_context(const ria_engine *);
bool ria_engine_sync(ria_engine *, const ds4_tokens *, const ds4_vision_span *,
                     size_t, float *logits, ds4_session_cancel_fn, void *,
                     ria_error *);
/* Most recent successful sync only: exact incorporated-prefix reuse, including
 * image identities; borrowed literal phase. The generation owner serializes. */
bool ria_engine_sync_observation(const ria_engine *, const char **phase,
                                 uint64_t *reused_tokens, ria_error *);
/* Authenticated caller opts in; measurements require streaming. */
bool ria_api_measurements(const ria_json_doc *, bool streaming, bool *enabled,
                          ria_error *);
bool ria_engine_eval(ria_engine *, uint32_t, float *, ria_error *);
void ria_engine_invalidate(ria_engine *);
bool ria_engine_rewind(ria_engine *, uint64_t, ria_error *);
bool ria_engine_image(ria_engine *, const uint8_t *, size_t,
                      uint64_t workspace_budget, ds4_vision_embedding *,
                      ria_error *);
#ifdef __cplusplus
}
#endif
#endif
