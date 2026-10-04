#ifndef RIA_JSON_H
#define RIA_JSON_H
#include "common.h"
#ifdef __cplusplus
extern "C" {
#endif
#define RIA_JSON_NONE UINT32_MAX
#define RIA_JSON_SAFE_INTEGER UINT64_C(9007199254740991)
typedef enum {
  RIA_JSON_NULL,
  RIA_JSON_BOOL,
  RIA_JSON_NUMBER,
  RIA_JSON_STRING,
  RIA_JSON_ARRAY,
  RIA_JSON_OBJECT
} ria_json_type;
typedef struct {
  ria_json_type type;
  uint32_t child, next;
  const char *key, *text;
  size_t key_length, length;
  double number;
  bool boolean;
} ria_json_node;
typedef struct {
  ria_json_node *nodes;
  uint32_t count;
  char *strings;
  size_t string_bytes;
  /* Exact owned node/string allocation capacity; borrowed assembled DOMs
   * declare their own allocation size. Includes unused parser capacity. */
  uint64_t allocated_bytes;
} ria_json_doc;
typedef struct {
  size_t max_bytes;
  uint32_t max_nodes, max_depth;
} ria_json_limits;
/* DOM owns all strings; no retained input. Reject duplicate keys and invalid
 * UTF-8. Output must not already own resources; free before reuse. Parse/read
 * leave an empty document on failure, including failures before parsing. */
bool ria_json_parse(const void *bytes, size_t length, ria_json_limits limits,
                    ria_json_doc *doc, ria_error *error);
bool ria_json_read(const char *path, ria_json_limits limits, ria_json_doc *doc,
                   ria_error *error);
void ria_json_free(ria_json_doc *doc);
const ria_json_node *ria_json_at(const ria_json_doc *doc, uint32_t index);
uint32_t ria_json_get(const ria_json_doc *doc, uint32_t object,
                      const char *key);
bool ria_json_fields(const ria_json_doc *doc, uint32_t object,
                     const char *const *allowed, size_t allowed_count,
                     const char *const *required, size_t required_count,
                     ria_error *error);
bool ria_json_u64(const ria_json_doc *doc, uint32_t index, bool decimal_string,
                  uint64_t *out, ria_error *error);
bool ria_json_string(const ria_json_doc *doc, uint32_t index, const char **out,
                     size_t *length, ria_error *error);
bool ria_json_digest_field(const ria_json_doc *doc, uint32_t index,
                           uint8_t out[32], ria_error *error);
/* RFC8785, UTF-16 object-key ordering, no normalization. Caller frees output.
 */
bool ria_json_canonical(const ria_json_doc *doc, bool manifest_identity,
                        char **out, size_t *length, ria_error *error);
bool ria_json_sha256(const ria_json_doc *doc, bool manifest_identity,
                     uint8_t digest[32], ria_error *error);
#ifdef __cplusplus
}
#endif
#endif
