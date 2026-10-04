#ifndef RIA_TOKENIZER_H
#define RIA_TOKENIZER_H
#include "json.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ria_tokenizer ria_tokenizer;
typedef struct {
  uint8_t pending[4];
  unsigned length;
} ria_utf8_decoder;
/* Source ByteLevel decoder: UTF8 malformed subsequences become U+FFFD. A
 * nonfinal call retains only an incomplete valid prefix. Caller output space
 * must hold 3*(input bytes + pending bytes); input and output are disjoint,
 * and state is explicitly owned. */
bool ria_utf8_decode(ria_utf8_decoder *, const uint8_t *, size_t, bool final,
                     char *, size_t capacity, size_t *written, ria_error *);
/* One process owner initializes Oniguruma before constructing tokenizer
 * objects; all tokenizers are destroyed before the owner releases the library
 * runtime. */
bool ria_tokenizer_runtime_begin(ria_error *error);
void ria_tokenizer_runtime_end(void);
/* The provisioned source is a nonempty regular file of at most 16MiB, opened
 * without following a final symlink or blocking on a FIFO/device. Its size and
 * modification identity must remain stable while reading, and the consumed
 * bytes must match source_sha256. Failure leaves a valid output pointer NULL. */
bool ria_tokenizer_open(const char *path, const uint8_t source_sha256[32],
                        uint64_t memory_budget, ria_tokenizer **tokenizer,
                        ria_error *error);
void ria_tokenizer_close(ria_tokenizer *tokenizer);
uint64_t ria_tokenizer_bytes(const ria_tokenizer *tokenizer);
uint64_t ria_tokenizer_max_token_bytes(const ria_tokenizer *);
uint32_t ria_tokenizer_vocab(const ria_tokenizer *tokenizer);
bool ria_tokenizer_special(const ria_tokenizer *tokenizer, const char *text,
                           uint32_t *id, ria_error *error);
/* Input is UTF8; returned token array is owned by caller. Explicit work limits
 * bound native regex and merge storage. No truncation or unknown-token default.
 */
bool ria_tokenizer_encode(ria_tokenizer *tokenizer, const char *text,
                          size_t length, bool recognize_special,
                          uint64_t max_tokens, uint32_t **tokens, size_t *count,
                          ria_error *error);
bool ria_tokenizer_decode(const ria_tokenizer *tokenizer, uint32_t id,
                          char **bytes, size_t *length, ria_error *error);
#ifdef __cplusplus
}
#endif
#endif
