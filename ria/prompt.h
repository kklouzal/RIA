#ifndef RIA_PROMPT_H
#define RIA_PROMPT_H
#include "json.h"
#ifdef __cplusplus
extern "C" {
#endif
#define RIA_BOS "<｜begin▁of▁sentence｜>"
#define RIA_EOS "<｜end▁of▁sentence｜>"
#define RIA_USER "<｜User｜>"
#define RIA_ASSISTANT "<｜Assistant｜>"
#define RIA_SYSTEM "<｜System｜>"
#define RIA_IMAGE_PLACEHOLDER "<｜deepseek_image｜>"
typedef struct {
  bool thinking, drop_thinking;
  unsigned reasoning_effort;
  size_t maximum_bytes;
} ria_prompt_options;
/* Render the pinned V4.1 source message semantics, including tool definitions,
 * DSML calls/results, numeric effort and exact assistant/user transitions.
 * Image sources are retained as ordered descriptors, never fetched here. */
bool ria_prompt_render(const ria_json_doc *messages, uint32_t array,
                       ria_prompt_options options, char **text, size_t *length,
                       ria_error *error);
bool ria_prompt_completion(const char *text, size_t length, bool thinking,
                           char **message_json, size_t *json_length,
                           ria_error *error);
/* Source-compatible JSON spelling preserves object order and string values;
 * intended only for encoded tool schemas/results, never identity hashing. */
bool ria_prompt_json(const ria_json_doc *doc, uint32_t node, char **text,
                     size_t *length, ria_error *error);
#ifdef __cplusplus
}
#endif
#endif
