#include "prompt.h"
#include "ryu/ryu.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct {
  char *text;
  size_t length, capacity, maximum;
  ria_error *error;
} builder;
static bool put(builder *b, const char *s, size_t n) {
  if (n > b->maximum - b->length)
    return ria_fail(b->error, RIA_RESOURCE_LIMIT,
                    "encoded prompt exceeds admitted byte limit");
  size_t need = b->length + n + 1;
  if (need > b->capacity) {
    size_t cap = b->capacity ? b->capacity : 1024;
    while (cap < need) {
      if (cap > (b->maximum + 1) / 2) {
        cap = b->maximum + 1;
        break;
      }
      cap *= 2;
    }
    char *text = realloc(b->text, cap);
    if (!text)
      return ria_fail(b->error, RIA_RESOURCE_LIMIT, "prompt allocation failed");
    b->text = text;
    b->capacity = cap;
  }
  if (n)
    memcpy(b->text + b->length, s, n);
  b->length += n;
  b->text[b->length] = 0;
  return true;
}
static bool literal(builder *b, const char *s) { return put(b, s, strlen(s)); }
static bool quoted(builder *b, const char *s, size_t n) {
  static const char hex[] = "0123456789abcdef";
  if (!literal(b, "\""))
    return false;
  for (size_t i = 0; i < n; i++) {
    unsigned c = (unsigned char)s[i];
    const char *escape = NULL;
    switch (c) {
    case 8:
      escape = "\\b";
      break;
    case 9:
      escape = "\\t";
      break;
    case 10:
      escape = "\\n";
      break;
    case 12:
      escape = "\\f";
      break;
    case 13:
      escape = "\\r";
      break;
    case '"':
      escape = "\\\"";
      break;
    case '\\':
      escape = "\\\\";
      break;
    }
    if (escape) {
      if (!literal(b, escape))
        return false;
    } else if (c < 32) {
      char bytes[6] = {'\\', 'u', '0', '0', hex[c >> 4], hex[c & 15]};
      if (!put(b, bytes, 6))
        return false;
    } else if (!put(b, s + i, 1))
      return false;
  }
  return literal(b, "\"");
}
static size_t python_float(double v, char out[64]) {
  if (v == 0) {
    const char *s = signbit(v) ? "-0.0" : "0.0";
    strcpy(out, s);
    return strlen(s);
  }
  char raw[32] = {0};
  int n = d2s_buffered_n(v, raw);
  if (n < 3 || n > 31)
    return 0;
  int at = 0;
  size_t used = 0;
  if (raw[0] == '-') {
    out[used++] = '-';
    at++;
  }
  char digits[20];
  int count = 0;
  while (at < n && raw[at] != 'E') {
    if (raw[at] != '.')
      digits[count++] = raw[at];
    at++;
  }
  if (!count)
    return 0;
  int exponent = 0;
  bool negative = false;
  if (at < n) {
    at++;
    if (raw[at] == '-') {
      negative = true;
      at++;
    }
    for (; at < n; at++)
      exponent = exponent * 10 + raw[at] - '0';
    if (negative)
      exponent = -exponent;
  }
  while (count > 1 && digits[count - 1] == '0')
    count--;
  int point = exponent + 1;
  if (exponent >= -4 && exponent < 16) {
    if (point <= 0) {
      out[used++] = '0';
      out[used++] = '.';
      while (point++ < 0)
        out[used++] = '0';
      for (int i = 0; i < count; i++)
        out[used++] = digits[i];
    } else {
      for (int i = 0; i < count || i < point; i++) {
        if (i == point)
          out[used++] = '.';
        out[used++] = i < count ? digits[i] : '0';
      }
      if (count <= point) {
        out[used++] = '.';
        out[used++] = '0';
      }
    }
  } else {
    out[used++] = digits[0];
    if (count > 1) {
      out[used++] = '.';
      for (int i = 1; i < count; i++)
        out[used++] = digits[i];
    }
    out[used++] = 'e';
    out[used++] = exponent < 0 ? '-' : '+';
    if (exponent < 0)
      exponent = -exponent;
    char reverse[5];
    int k = 0;
    do {
      reverse[k++] = (char)('0' + exponent % 10);
      exponent /= 10;
    } while (exponent);
    if (k < 2)
      reverse[k++] = '0';
    while (k)
      out[used++] = reverse[--k];
  }
  return used;
}
static bool json(builder *b, const ria_json_doc *d, uint32_t index,
                 unsigned depth) {
  const ria_json_node *n = ria_json_at(d, index);
  if (!n || depth > 64)
    return ria_fail(b->error, RIA_INVALID_REQUEST, "invalid prompt JSON node");
  switch (n->type) {
  case RIA_JSON_NULL:
    return literal(b, "null");
  case RIA_JSON_BOOL:
    return literal(b, n->boolean ? "true" : "false");
  case RIA_JSON_STRING:
    return quoted(b, n->text, n->length);
  case RIA_JSON_NUMBER: {
    if (!isfinite(n->number))
      return ria_fail(b->error, RIA_INVALID_REQUEST, "nonfinite prompt number");
    bool integer = n->text && !memchr(n->text, '.', n->length) &&
                   !memchr(n->text, 'e', n->length) &&
                   !memchr(n->text, 'E', n->length);
    char value[64];
    size_t length;
    if (integer) {
      if (fabs(n->number) > 9007199254740991.0 || trunc(n->number) != n->number)
        return ria_fail(b->error, RIA_INVALID_REQUEST,
                        "tool integer outside safe range");
      int result =
          snprintf(value, sizeof value, "%.0f", n->number == 0 ? 0 : n->number);
      if (result < 0 || (size_t)result >= sizeof value)
        return false;
      length = (size_t)result;
    } else
      length = python_float(n->number, value);
    return length ? put(b, value, length)
                  : ria_fail(b->error, RIA_INTERNAL_ERROR,
                             "prompt number conversion failed");
  }
  case RIA_JSON_ARRAY:
  case RIA_JSON_OBJECT: {
    bool object = n->type == RIA_JSON_OBJECT;
    if (!literal(b, object ? "{" : "["))
      return false;
    bool first = true;
    for (uint32_t i = n->child; i != RIA_JSON_NONE; i = d->nodes[i].next) {
      if (!first && !literal(b, ", "))
        return false;
      first = false;
      if (object && (!quoted(b, d->nodes[i].key, d->nodes[i].key_length) ||
                     !literal(b, ": ")))
        return false;
      if (!json(b, d, i, depth + 1))
        return false;
    }
    return literal(b, object ? "}" : "]");
  }
  }
  return false;
}
bool ria_prompt_json(const ria_json_doc *d, uint32_t index, char **out,
                     size_t *length, ria_error *e) {
  if (!out || !length)
    return ria_fail(e, RIA_INVALID_REQUEST, "missing prompt JSON output");
  builder b = {NULL, 0, 0, 67108864, e};
  if (!json(&b, d, index, 0)) {
    free(b.text);
    return false;
  }
  *out = b.text;
  *length = b.length;
  return true;
}
static const ria_json_node *field(const ria_json_doc *d, uint32_t object,
                                  const char *key) {
  return ria_json_at(d, ria_json_get(d, object, key));
}
static bool is(const ria_json_doc *d, uint32_t object, const char *key,
               const char *text) {
  const ria_json_node *n = field(d, object, key);
  return n && n->type == RIA_JSON_STRING && n->length == strlen(text) &&
         !memcmp(n->text, text, n->length);
}
static bool text(builder *b, const ria_json_node *n) {
  if (!n || n->type == RIA_JSON_NULL)
    return true;
  if (n->type != RIA_JSON_STRING)
    return ria_fail(b->error, RIA_INVALID_REQUEST,
                    "message content must be text");
  for (size_t i = 0; i + strlen(RIA_IMAGE_PLACEHOLDER) <= n->length; i++)
    if (!memcmp(n->text + i, RIA_IMAGE_PLACEHOLDER,
                strlen(RIA_IMAGE_PLACEHOLDER)))
      return ria_fail(b->error, RIA_INVALID_REQUEST,
                      "image placeholder is reserved for image descriptors");
  return put(b, n->text, n->length);
}
static bool attribute(builder *b, const char *s, size_t n) {
  if (!n || n > 256 || memchr(s, '"', n) || memchr(s, '\n', n) ||
      memchr(s, '\r', n) || memchr(s, 0, n))
    return ria_fail(b->error, RIA_INVALID_REQUEST, "invalid DSML attribute");
  return put(b, s, n);
}
static bool qualified_name(builder *b, const ria_json_doc *d, uint32_t function,
                           const ria_json_node *namespace) {
  const ria_json_node *name = field(d, function, "name");
  if (!name || name->type != RIA_JSON_STRING)
    return ria_fail(b->error, RIA_INVALID_REQUEST, "tool name missing");
  if (namespace && namespace->type == RIA_JSON_OBJECT)
  namespace = field(d, (uint32_t)(namespace - d->nodes), "name");
  if (namespace && namespace->type != RIA_JSON_NULL) {
    if (namespace->type != RIA_JSON_STRING || strstr(namespace->text, "::"))
      return ria_fail(b->error, RIA_INVALID_REQUEST, "invalid tool namespace");
    const char *separator = strstr(name->text, "::");
    if (separator) {
      size_t prefix = (size_t)(separator - name->text);
      if (prefix != namespace->length ||
          memcmp(namespace->text, name->text, prefix) ||
          strstr(separator + 2, "::"))
        return ria_fail(b->error, RIA_INVALID_REQUEST,
                        "conflicting tool namespace");
      return attribute(b, name->text, name->length);
    }
    if (!attribute(b, namespace->text, namespace->length) || !literal(b, "::"))
      return false;
  } else {
    const char *separator = strstr(name->text, "::");
    if (separator && strstr(separator + 2, "::"))
      return ria_fail(b->error, RIA_INVALID_REQUEST,
                      "invalid qualified tool name");
  }
  return attribute(b, name->text, name->length);
}
static const char tools_prefix[] =
    "## Tools\n\nYou have access to a set of tools to help answer the user's "
    "question. You can invoke tools by writing a \"<｜DSML｜ calls>\" block "
    "like the following:\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke "
    "name=\"$TOOL_NAME\">\n<｜DSML｜ parameter name=\"$PARAMETER_NAME\" "
    "string=\"true|false\">$PARAMETER_VALUE</｜DSML｜ "
    "parameter>\n...\n</｜DSML｜ invoke>\n<｜DSML｜ invoke "
    "name=\"$TOOL_NAME2\">\n...\n</｜DSML｜ invoke>\n</｜DSML｜ "
    "calls>\n\nString parameters should be specified as is and set "
    "`string=\"true\"`. For all other types (numbers, booleans, arrays, "
    "objects), pass the value in JSON format and set `string=\"false\"`.\n\nIf "
    "thinking_mode is enabled (triggered by <think>), you MUST output your "
    "complete reasoning inside <think>...</think> BEFORE any tool calls or "
    "final response.\n\nOtherwise, output directly after </think> with tool "
    "calls or final response.\n\n### Available Tool Schemas\n\n";
static const char tools_suffix[] =
    "\n\nYou MUST strictly follow the above defined tool name and parameter "
    "schemas to invoke tool calls.\n";
static bool tools(builder *b, const ria_json_doc *d, uint32_t array) {
  const ria_json_node *n = ria_json_at(d, array);
  if (!n || n->type != RIA_JSON_ARRAY)
    return ria_fail(b->error, RIA_INVALID_REQUEST, "tools must be an array");
  if (n->child == RIA_JSON_NONE)
    return true;
  if (!literal(b, "\n\n") || !literal(b, tools_prefix))
    return false;
  bool first = true;
  for (uint32_t i = n->child; i != RIA_JSON_NONE; i = d->nodes[i].next) {
    uint32_t function = ria_json_get(d, i, "function");
    const ria_json_node *f = ria_json_at(d, function),
                        *namespace = field(d, i, "namespace");
    if (!namespace)
    namespace = field(d, function, "namespace");
    if (!is(d, i, "type", "function") || !f || f->type != RIA_JSON_OBJECT)
      return ria_fail(b->error, RIA_INVALID_REQUEST, "unsupported tool kind");
    builder name = {NULL, 0, 0, 512, b->error};
    if (!qualified_name(&name, d, function, namespace)) {
      free(name.text);
      return false;
    }
    const ria_json_node *namespace_description =
        namespace && namespace->type == RIA_JSON_OBJECT
            ? field(d, (uint32_t)(namespace - d->nodes), "description")
            : NULL;
    bool prepend = namespace_description &&
                   namespace_description->type == RIA_JSON_STRING &&
                   namespace_description->length;
    if (!first && !literal(b, "\n")) {
      free(name.text);
      return false;
    }
    first = false;
    bool ok = literal(b, "{");
    bool field_first = true, had_description = false;
    for (uint32_t j = f->child; j != RIA_JSON_NONE && ok;
         j = d->nodes[j].next) {
      const ria_json_node *value = &d->nodes[j];
      if (value->key_length == 9 && !memcmp(value->key, "namespace", 9))
        continue;
      if (!field_first)
        ok = literal(b, ", ");
      field_first = false;
      if (ok)
        ok = quoted(b, value->key, value->key_length) && literal(b, ": ");
      if (value->key_length == 4 && !memcmp(value->key, "name", 4)) {
        if (ok)
          ok = quoted(b, name.text, name.length);
      } else if (value->key_length == 11 &&
                 !memcmp(value->key, "description", 11) && prepend) {
        had_description = true;
        builder description = {NULL, 0, 0, b->maximum, b->error};
        ok = ok &&
             put(&description, namespace_description->text,
                 namespace_description->length) &&
             literal(&description, "\n") && text(&description, value) &&
             quoted(b, description.text, description.length);
        free(description.text);
      } else if (ok)
        ok = json(b, d, j, 0);
    }
    if (ok && prepend && !had_description) {
      if (!field_first)
        ok = literal(b, ", ");
      builder description = {NULL, 0, 0, b->maximum, b->error};
      ok = ok &&
           put(&description, namespace_description->text,
               namespace_description->length) &&
           literal(&description, "\n") && literal(b, "\"description\": ") &&
           quoted(b, description.text, description.length);
      free(description.text);
    }
    if (ok)
      ok = literal(b, "}");
    free(name.text);
    if (!ok)
      return false;
  }
  return literal(b, tools_suffix);
}
/* Python accepts these extensions while the service's JSON boundary requires
 * finite values. Recognize complete unquoted extension tokens only. */
static bool nonfinite_argument(const char *bytes, size_t length) {
  size_t first = 0;
  while (first < length && strchr(" \r\n\t", bytes[first]))
    first++;
  if (first == length)
    return false;
  if (bytes[first] != '{' && bytes[first] != '[' && bytes[first] != 'N' &&
      bytes[first] != 'I' && bytes[first] != '-')
    return false;
  bool quoted_string = false, escaped = false;
  for (size_t i = first; i < length; i++) {
    unsigned char c = (unsigned char)bytes[i];
    if (quoted_string) {
      if (escaped)
        escaped = false;
      else if (c == '\\')
        escaped = true;
      else if (c == '"')
        quoted_string = false;
      continue;
    }
    if (c == '"') {
      quoted_string = true;
      continue;
    }
    if (i > first && !strchr(" \r\n\t[:,", bytes[i - 1]))
      continue;
    const char *tokens[] = {"NaN", "Infinity", "-Infinity"};
    for (size_t j = 0; j < 3; j++) {
      size_t n = strlen(tokens[j]);
      if (n <= length - i && !memcmp(bytes + i, tokens[j], n) &&
          (i + n == length || strchr(" \r\n\t,]}", bytes[i + n])))
        return true;
    }
  }
  return false;
}
static bool arguments(builder *b, const ria_json_doc *original,
                      uint32_t index) {
  uint32_t original_index = index;
  const ria_json_doc *d = original;
  ria_json_doc decoded[2] = {{0}, {0}};
  const ria_json_node *n = ria_json_at(d, index);
  for (unsigned level = 0; level < 2 && n && n->type == RIA_JSON_STRING;
       level++) {
    ria_error ignored = {0};
    if (!ria_json_parse(n->text, n->length,
                        (ria_json_limits){16777216, 200000, 64},
                        &decoded[level], &ignored)) {
      if (ignored.code != RIA_INVALID_REQUEST ||
          strstr(ignored.message, "duplicate key") ||
          strstr(ignored.message, "nonfinite") ||
          nonfinite_argument(n->text, n->length)) {
        ria_json_free(&decoded[0]);
        ria_json_free(&decoded[1]);
        if (b->error)
          *b->error = ignored;
        return ria_fail(
            b->error,
            ignored.code == RIA_INVALID_REQUEST ? RIA_UNSUPPORTED
                                                : ignored.code,
            "encoded tool arguments violate finite, unique-key JSON contract");
      }
      break;
    }
    d = &decoded[level];
    index = 0;
    n = ria_json_at(d, index);
  }
  bool object = n && n->type == RIA_JSON_OBJECT;
  bool first = true, ok = true;
  if (object) {
    for (uint32_t i = n->child; i != RIA_JSON_NONE && ok;
         i = d->nodes[i].next) {
      const ria_json_node *v = &d->nodes[i];
      if (!first)
        ok = literal(b, "\n");
      first = false;
      if (ok)
        ok = literal(b, "<｜DSML｜ parameter name=\"") &&
             attribute(b, v->key, v->key_length) &&
             literal(b, v->type == RIA_JSON_STRING ? "\" string=\"true\">"
                                                   : "\" string=\"false\">") &&
             (v->type == RIA_JSON_STRING ? put(b, v->text, v->length)
                                         : json(b, d, i, 0)) &&
             literal(b, "</｜DSML｜ parameter>");
    }
  } else {
    const ria_json_node *v = ria_json_at(original, original_index);
    index = original_index;
    if (!v) {
      ria_json_free(&decoded[0]);
      ria_json_free(&decoded[1]);
      return ria_fail(b->error, RIA_INVALID_REQUEST, "tool arguments missing");
    }
    ok = v && literal(b, "<｜DSML｜ parameter name=\"arguments\"") &&
         literal(b, v->type == RIA_JSON_STRING ? " string=\"true\">"
                                               : " string=\"false\">") &&
         (v->type == RIA_JSON_STRING ? put(b, v->text, v->length)
                                     : json(b, original, index, 0)) &&
         literal(b, "</｜DSML｜ parameter>");
  }
  ria_json_free(&decoded[0]);
  ria_json_free(&decoded[1]);
  return ok;
}
static bool calls(builder *b, const ria_json_doc *d, uint32_t array) {
  const ria_json_node *n = ria_json_at(d, array);
  if (!n || n->type != RIA_JSON_ARRAY)
    return ria_fail(b->error, RIA_INVALID_REQUEST,
                    "tool calls must be an array");
  if (n->child == RIA_JSON_NONE)
    return true;
  if (!literal(b, "\n\n<｜DSML｜ calls>\n"))
    return false;
  bool first = true;
  for (uint32_t i = n->child; i != RIA_JSON_NONE; i = d->nodes[i].next) {
    if (!first && !literal(b, "\n"))
      return false;
    first = false;
    uint32_t function = ria_json_get(d, i, "function");
    const ria_json_node *namespace = field(d, i, "namespace");
    if (!namespace)
    namespace = field(d, function, "namespace");
    if (!literal(b, "<｜DSML｜ invoke name=\"") ||
        !qualified_name(b, d, function, namespace) || !literal(b, "\">\n") ||
        !arguments(b, d, ria_json_get(d, function, "arguments")) ||
        !literal(b, "\n</｜DSML｜ invoke>"))
      return false;
  }
  return literal(b, "\n</｜DSML｜ calls>");
}
typedef struct {
  uint32_t message, block;
  unsigned kind;
  const ria_json_node *id;
  uint32_t rank, ordinal;
} block;
typedef struct {
  uint32_t source;
  unsigned role;
  size_t begin, count;
} message;
enum { SYSTEM = 1, USER = 2, ASSISTANT = 3, REMINDER = 4 };
static bool blocks(builder *b, const ria_json_doc *d, uint32_t array,
                   unsigned depth) {
  const ria_json_node *n = ria_json_at(d, array);
  if (!n || n->type != RIA_JSON_ARRAY || depth > 32)
    return ria_fail(b->error, RIA_INVALID_REQUEST, "invalid message blocks");
  bool first = true;
  for (uint32_t i = n->child; i != RIA_JSON_NONE; i = d->nodes[i].next) {
    if (!first && !literal(b, "\n\n"))
      return false;
    first = false;
    if (is(d, i, "type", "text")) {
      if (!text(b, field(d, i, "text")))
        return false;
    } else if (is(d, i, "type", "image") || is(d, i, "type", "image_url")) {
      if (!literal(b, RIA_IMAGE_PLACEHOLDER))
        return false;
    } else if (is(d, i, "type", "tool_result")) {
      const ria_json_node *content = field(d, i, "content");
      if (!literal(b, "<tool_result>"))
        return false;
      if (content && content->type == RIA_JSON_ARRAY) {
        if (!blocks(b, d, (uint32_t)(content - d->nodes), depth + 1))
          return false;
      } else if (!text(b, content))
        return false;
      if (!literal(b, "</tool_result>"))
        return false;
    } else
      return ria_fail(b->error, RIA_UNSUPPORTED,
                      "unsupported message content block");
  }
  return true;
}
static bool user_piece(builder *b, const ria_json_doc *d, const block *piece) {
  if (piece->kind == 1)
    return text(b, field(d, piece->message, "content"));
  if (piece->kind == 2) {
    if (!literal(b, "<tool_result>") ||
        !(field(d, piece->message, "content") &&
                  field(d, piece->message, "content")->type == RIA_JSON_ARRAY
              ? blocks(b, d, ria_json_get(d, piece->message, "content"), 1)
              : text(b, field(d, piece->message, "content"))))
      return false;
    return literal(b, "</tool_result>");
  }
  if (piece->kind == 4) {
    uint32_t k = piece->block;
    if (is(d, k, "type", "text"))
      return text(b, field(d, k, "text"));
    if (is(d, k, "type", "image") || is(d, k, "type", "image_url"))
      return literal(b, RIA_IMAGE_PLACEHOLDER);
    if (is(d, k, "type", "tool_result")) {
      const ria_json_node *c = field(d, k, "content");
      return literal(b, "<tool_result>") &&
             (c && c->type == RIA_JSON_ARRAY
                  ? blocks(b, d, (uint32_t)(c - d->nodes), 1)
                  : text(b, c)) &&
             literal(b, "</tool_result>");
    }
    return ria_fail(b->error, RIA_UNSUPPORTED,
                    "unsupported user content block");
  }
  return blocks(b, d, piece->block, 0);
}
static int block_order(const void *left, const void *right) {
  const block *a = left, *b = right;
  return a->rank < b->rank         ? -1
         : a->rank > b->rank       ? 1
         : a->ordinal < b->ordinal ? -1
                                   : a->ordinal > b->ordinal;
}
static bool node_equal(const ria_json_node *a, const ria_json_node *b) {
  return a && b && a->type == RIA_JSON_STRING && b->type == RIA_JSON_STRING &&
         a->length == b->length && !memcmp(a->text, b->text, a->length);
}
bool ria_prompt_render(const ria_json_doc *d, uint32_t array,
                       ria_prompt_options options, char **out, size_t *length,
                       ria_error *e) {
  if (!out || !length || !options.maximum_bytes ||
      options.maximum_bytes > 67108864 || options.reasoning_effort < 1 ||
      options.reasoning_effort > 100)
    return ria_fail(e, RIA_INVALID_REQUEST,
                    "invalid prompt rendering contract");
  const ria_json_node *input = ria_json_at(d, array);
  if (!input || input->type != RIA_JSON_ARRAY || input->child == RIA_JSON_NONE)
    return ria_fail(e, RIA_INVALID_REQUEST,
                    "messages must be a nonempty array");
  size_t count = 0;
  for (uint32_t i = input->child; i != RIA_JSON_NONE; i = d->nodes[i].next)
    count++;
  if (count > 10000)
    return ria_fail(e, RIA_RESOURCE_LIMIT, "message count exceeds bound");
  message *messages = calloc(count, sizeof *messages);
  block *pieces = calloc(d->count, sizeof *pieces);
  if (!messages || !pieces) {
    free(messages);
    free(pieces);
    return ria_fail(e, RIA_RESOURCE_LIMIT,
                    "message normalization allocation failed");
  }
  size_t m = 0, p = 0;
  bool any_tools = false, ok = true;
  static const char *const fields[] = {"role",
                                       "content",
                                       "content_blocks",
                                       "tool_calls",
                                       "reasoning_content",
                                       "tools",
                                       "response_format",
                                       "wo_eos",
                                       "task",
                                       "tool_call_id",
                                       "name"};
  static const char *const required[] = {"role"};
  for (uint32_t i = input->child; i != RIA_JSON_NONE && ok;
       i = d->nodes[i].next) {
    if (!ria_json_fields(d, i, fields, 11, required, 1, e)) {
      ok = false;
      break;
    }
    unsigned role = is(d, i, "role", "system") ? SYSTEM
                    : is(d, i, "role", "user") || is(d, i, "role", "tool")
                        ? USER
                    : is(d, i, "role", "assistant")       ? ASSISTANT
                    : is(d, i, "role", "latest_reminder") ? REMINDER
                                                          : 0;
    if (!role) {
      ok = ria_fail(e, RIA_UNSUPPORTED, "unsupported message role");
      break;
    }
    const ria_json_node *content = field(d, i, "content"),
                        *reasoning = field(d, i, "reasoning_content");
    if ((content && content->type == RIA_JSON_STRING &&
         strstr(content->text, RIA_IMAGE_PLACEHOLDER)) ||
        (reasoning && reasoning->type == RIA_JSON_STRING &&
         strstr(reasoning->text, RIA_IMAGE_PLACEHOLDER))) {
      ok = ria_fail(e, RIA_INVALID_REQUEST,
                    "text contains reserved image token");
      break;
    }
    const ria_json_node *defined_tools = field(d, i, "tools");
    if (defined_tools && defined_tools->type == RIA_JSON_ARRAY &&
        defined_tools->child != RIA_JSON_NONE) {
      if (role != SYSTEM) {
        ok = ria_fail(e, RIA_INVALID_REQUEST,
                      "tool declarations require a system message");
        break;
      }
      any_tools = true;
    }
    bool tool = is(d, i, "role", "tool");
    bool merge_user =
        role == USER && m && messages[m - 1].role == USER &&
        (!field(d, messages[m - 1].source, "task") ||
         field(d, messages[m - 1].source, "task")->type == RIA_JSON_NULL);
    if (!merge_user)
      messages[m++] = (message){i, role, p, 0};
    message *current = &messages[m - 1];
    if (role == USER) {
      uint32_t block_array = ria_json_get(d, i, "content_blocks");
      if (block_array == RIA_JSON_NONE && content &&
          content->type == RIA_JSON_ARRAY)
        block_array = (uint32_t)(content - d->nodes);
      if (block_array != RIA_JSON_NONE && !tool) {
        const ria_json_node *array_node = ria_json_at(d, block_array);
        if (!array_node || array_node->type != RIA_JSON_ARRAY) {
          ok = ria_fail(e, RIA_INVALID_REQUEST, "user blocks must be an array");
          break;
        }
        for (uint32_t k = array_node->child; k != RIA_JSON_NONE;
             k = d->nodes[k].next) {
          pieces[p++] = (block){i, k, 4, field(d, k, "tool_use_id"), 0, 0};
          current->count++;
        }
      } else {
        pieces[p++] = (block){
            i, block_array, tool ? 2 : 1, field(d, i, "tool_call_id"), 0, 0};
        current->count++;
      }
    }
  }
  uint32_t previous_calls = RIA_JSON_NONE;
  for (size_t i = 0; i < m && ok; i++) {
    message *msg = &messages[i];
    if (msg->role == ASSISTANT) {
      uint32_t calls = ria_json_get(d, msg->source, "tool_calls");
      const ria_json_node *c = ria_json_at(d, calls);
      if (c && c->type == RIA_JSON_ARRAY && c->child != RIA_JSON_NONE)
        previous_calls = calls;
    } else if (msg->role == USER && previous_calls != RIA_JSON_NONE &&
               msg->count > 1) {
      block *ordered = malloc(msg->count * sizeof *ordered);
      if (!ordered) {
        ok = ria_fail(e, RIA_RESOURCE_LIMIT,
                      "tool result ordering allocation failed");
        break;
      }
      size_t num = 0;
      for (size_t j = 0; j < msg->count; j++) {
        block v = pieces[msg->begin + j];
        if (v.kind != 2 &&
            !(v.kind == 4 && is(d, v.block, "type", "tool_result")))
          continue;
        v.ordinal = (uint32_t)j;
        v.rank = 0;
        uint32_t rank = 0;
        const ria_json_node *c = ria_json_at(d, previous_calls);
        for (uint32_t k = c->child; k != RIA_JSON_NONE;
             k = d->nodes[k].next, rank++) {
          const ria_json_node *id = field(d, k, "id");
          if (!id || id->type == RIA_JSON_NULL ||
              (id->type == RIA_JSON_STRING && !id->length))
            id = field(d, ria_json_get(d, k, "function"), "id");
          if (node_equal(id, v.id))
            v.rank = rank;
        }
        ordered[num++] = v;
      }
      qsort(ordered, num, sizeof *ordered, block_order);
      size_t k = 0;
      for (size_t j = 0; j < msg->count; j++) {
        block *v = &pieces[msg->begin + j];
        if (v->kind == 2 ||
            (v->kind == 4 && is(d, v->block, "type", "tool_result")))
          *v = ordered[k++];
      }
      free(ordered);
    }
  }
  int64_t last_user = -1;
  for (size_t i = 0; i < m; i++)
    if (messages[i].role == USER || (messages[i].role == SYSTEM && i > 0))
      last_user = (int64_t)i;
  bool drop = options.drop_thinking && !any_tools;
  builder b = {NULL, 0, 0, options.maximum_bytes, e};
  if (ok)
    ok = literal(&b, RIA_BOS);
  for (size_t i = 0; i < m && ok; i++) {
    message *msg = &messages[i];
    uint32_t source = msg->source;
    if (i == 0 && (options.thinking || msg->role == SYSTEM))
      ok = literal(&b, RIA_SYSTEM);
    if (ok && i == 0 && options.thinking) {
      char effort[160];
      snprintf(effort, sizeof effort,
               "Reasoning Effort: %u (range 1-100, the higher the value, the "
               "more thorough the reasoning)\n\n",
               options.reasoning_effort);
      ok = literal(&b, effort);
    }
    if (!ok)
      break;
    if (msg->role == SYSTEM) {
      if (i)
        ok = literal(&b, RIA_SYSTEM);
      if (ok)
        ok = text(&b, field(d, source, "content"));
      uint32_t t = ria_json_get(d, source, "tools");
      if (ok && t != RIA_JSON_NONE)
        ok = tools(&b, d, t);
      uint32_t format = ria_json_get(d, source, "response_format");
      if (ok && format != RIA_JSON_NONE)
        ok = literal(&b, "\n\n## Response Format:\n\nYou MUST strictly adhere "
                         "to the following schema to reply:\n") &&
             json(&b, d, format, 0);
    } else if (msg->role == USER) {
      ok = literal(&b, RIA_USER);
      for (size_t j = 0; j < msg->count && ok; j++) {
        if (j)
          ok = literal(&b, "\n\n");
        if (ok)
          ok = user_piece(&b, d, &pieces[msg->begin + j]);
      }
    } else if (msg->role == REMINDER)
      ok = literal(&b, "<｜latest_reminder｜>") &&
           text(&b, field(d, source, "content"));
    else {
      bool previous_task =
          i && field(d, messages[i - 1].source, "task") &&
          field(d, messages[i - 1].source, "task")->type != RIA_JSON_NULL;
      if (options.thinking && !previous_task &&
          (!drop || (int64_t)i > last_user))
        ok = text(&b, field(d, source, "reasoning_content")) &&
             literal(&b, "</think>");
      if (ok)
        ok = text(&b, field(d, source, "content"));
      uint32_t tc = ria_json_get(d, source, "tool_calls");
      if (ok && tc != RIA_JSON_NONE)
        ok = calls(&b, d, tc);
      const ria_json_node *wo_eos = field(d, source, "wo_eos");
      if (wo_eos && wo_eos->type != RIA_JSON_BOOL) {
        ok = ria_fail(e, RIA_INVALID_REQUEST, "wo_eos must be boolean");
        break;
      }
      if (ok && (!wo_eos || !wo_eos->boolean))
        ok = literal(&b, RIA_EOS);
    }
    if (!ok || ((i + 1 < m) && messages[i + 1].role != ASSISTANT &&
                messages[i + 1].role != REMINDER))
      continue;
    const ria_json_node *task = field(d, source, "task");
    if (task && task->type != RIA_JSON_NULL) {
      static const char *const task_names[] = {"action", "query", "authority",
                                               "domain", "title", "read_url"};
      unsigned selected = 6;
      for (unsigned k = 0; k < 6; k++)
        if (task->type == RIA_JSON_STRING &&
            task->length == strlen(task_names[k]) &&
            !memcmp(task->text, task_names[k], task->length))
          selected = k;
      if (selected == 6) {
        ok = ria_fail(e, RIA_UNSUPPORTED, "unsupported quick instruction task");
        break;
      }
      if (selected == 0)
        ok = literal(&b, RIA_ASSISTANT) &&
             literal(&b, options.thinking ? "<think>" : "</think>");
      if (ok)
        ok = literal(&b, "<｜") && literal(&b, task_names[selected]) &&
             literal(&b, "｜>");
    } else if (msg->role == USER || (msg->role == SYSTEM && i > 0))
      ok = literal(&b, RIA_ASSISTANT) &&
           literal(&b, options.thinking && (!drop || (int64_t)i >= last_user)
                           ? "<think>"
                           : "</think>");
  }
  free(messages);
  free(pieces);
  if (!ok) {
    free(b.text);
    return false;
  }
  *out = b.text;
  *length = b.length;
  return true;
}
static const char *find_bytes(const char *s, const char *limit,
                              const char *needle) {
  size_t n = strlen(needle);
  if (s > limit || n > (size_t)(limit - s))
    return NULL;
  for (const char *p = s; (size_t)(limit - p) >= n; p++)
    if (!memcmp(p, needle, n))
      return p;
  return NULL;
}
static bool consume(const char **at, const char *literal_text) {
  size_t n = strlen(literal_text);
  if (strncmp(*at, literal_text, n))
    return false;
  *at += n;
  return true;
}
static bool clean_content(const char *s, size_t n) {
  static const char *const reserved[] = {RIA_BOS, RIA_EOS, "<think>",
                                         "</think>", "｜DSML｜"};
  for (size_t i = 0; i < sizeof reserved / sizeof reserved[0]; i++) {
    const char *p = find_bytes(s, s + n, reserved[i]);
    if (p && (size_t)(p - s) < n)
      return false;
  }
  return true;
}
static bool parse_calls(builder *out, const char **cursor, const char *limit) {
  const char *at = *cursor;
  bool first_call = true;
  if (!consume(&at, "\n\n<｜DSML｜ calls>\n") || !literal(out, "["))
    return false;
  while (strncmp(at, "</｜DSML｜ calls>", strlen("</｜DSML｜ calls>"))) {
    if (!first_call && !literal(out, ", "))
      return false;
    first_call = false;
    if (!consume(&at, "<｜DSML｜ invoke"))
      return ria_fail(out->error, RIA_INVALID_REQUEST,
                      "incomplete DSML invoke");
    while (*at == ' ' || *at == '\t' || *at == '\r' || *at == '\n')
      at++;
    if (!consume(&at, "name=\""))
      return ria_fail(out->error, RIA_INVALID_REQUEST,
                      "DSML invoke name missing");
    const char *name = at, *end = find_bytes(at, limit, "\">\n");
    if (!end || end - name > 256 || end == name)
      return ria_fail(out->error, RIA_INVALID_REQUEST,
                      "invalid DSML invoke name");
    size_t name_length = (size_t)(end - name);
    at = end + 3;
    const char *namespace_end = NULL;
    for (size_t i = 0; i + 1 < name_length; i++)
      if (name[i] == ':' && name[i + 1] == ':') {
        if (namespace_end)
          return ria_fail(out->error, RIA_INVALID_REQUEST,
                          "invalid DSML namespace");
        namespace_end = name + i;
        i++;
      }
    builder args = {NULL, 0, 0, out->maximum, out->error};
    bool ok = literal(&args, "{");
    const char *keys[1024];
    size_t key_lengths[1024], key_count = 0;
    while (ok &&
           !strncmp(at, "<｜DSML｜ parameter", strlen("<｜DSML｜ parameter"))) {
      at += strlen("<｜DSML｜ parameter");
      if (!consume(&at, " name=\"")) {
        ok = ria_fail(out->error, RIA_INVALID_REQUEST,
                      "invalid DSML parameter name");
        break;
      }
      const char *key = at;
      end = find_bytes(at, limit, "\" string=\"");
      if (!end || !(end > key) || end - key > 256 || key_count == 1024) {
        ok = ria_fail(out->error, RIA_RESOURCE_LIMIT,
                      "DSML parameter bounds exceeded");
        break;
      }
      size_t key_length = (size_t)(end - key);
      for (size_t i = 0; i < key_count; i++)
        if (key_length == key_lengths[i] && !memcmp(key, keys[i], key_length)) {
          ok = ria_fail(out->error, RIA_INVALID_REQUEST,
                        "duplicate DSML parameter");
          break;
        }
      if (!ok)
        break;
      if (key_count && !literal(&args, ", ")) {
        ok = false;
        break;
      }
      keys[key_count] = key;
      key_lengths[key_count++] = key_length;
      at = end + strlen("\" string=\"");
      bool string_value = consume(&at, "true\">");
      if (!string_value && !consume(&at, "false\">")) {
        ok = ria_fail(out->error, RIA_INVALID_REQUEST,
                      "invalid DSML parameter type");
        break;
      }
      const char *value = at;
      end = find_bytes(at, limit, "</｜DSML｜ parameter>");
      if (!end) {
        ok = ria_fail(out->error, RIA_INVALID_REQUEST,
                      "incomplete DSML parameter");
        break;
      }
      ok = quoted(&args, key, key_length) && literal(&args, ": ") &&
           (string_value ? quoted(&args, value, (size_t)(end - value))
                         : put(&args, value, (size_t)(end - value)));
      at = end + strlen("</｜DSML｜ parameter>");
      if (!consume(&at, "\n")) {
        ok = ria_fail(out->error, RIA_INVALID_REQUEST,
                      "DSML parameter newline missing");
        break;
      }
    }
    /* An empty invocation has the source encoder's empty-arguments newline. */
    if (!key_count && *at == '\n')
      at++;
    if (ok && !consume(&at, "</｜DSML｜ invoke>\n"))
      ok = ria_fail(out->error, RIA_INVALID_REQUEST,
                    "incomplete DSML invoke terminator");
    if (ok)
      ok = literal(&args, "}") &&
           literal(out, "{\"type\": \"function\", \"function\": {\"name\": ") &&
           quoted(out, namespace_end ? namespace_end + 2 : name,
                  namespace_end
                      ? name_length - (size_t)(namespace_end - name) - 2
                      : name_length) &&
           literal(out, ", \"arguments\": ") &&
           quoted(out, args.text, args.length) && literal(out, "}");
    if (ok && namespace_end)
      ok = literal(out, ", \"namespace\": ") &&
           quoted(out, name, (size_t)(namespace_end - name));
    if (ok)
      ok = literal(out, "}");
    free(args.text);
    if (!ok)
      return false;
  }
  if (!consume(&at, "</｜DSML｜ calls>") || !literal(out, "]"))
    return false;
  *cursor = at;
  return true;
}
bool ria_prompt_completion(const char *text, size_t input_length, bool thinking,
                           char **json_out, size_t *json_length, ria_error *e) {
  if (!text || !json_out || !json_length || input_length > 67108864)
    return ria_fail(e, RIA_INVALID_REQUEST, "invalid completion bytes");
  char *copy = malloc(input_length + 1);
  if (!copy)
    return ria_fail(e, RIA_RESOURCE_LIMIT, "completion allocation failed");
  memcpy(copy, text, input_length);
  copy[input_length] = 0;
  const char *at = copy, *reasoning = "";
  size_t reasoning_length = 0;
  bool ok = true;
  if (thinking) {
    const char *end = find_bytes(at, copy + input_length, "</think>");
    if (!end)
      ok = ria_fail(e, RIA_INVALID_REQUEST, "incomplete reasoning output");
    else {
      reasoning = at;
      reasoning_length = (size_t)(end - at);
      at = end + 8;
    }
  }
  const char *eos = ok ? find_bytes(at, copy + input_length, RIA_EOS) : NULL,
             *tool =
                 ok ? find_bytes(at, copy + input_length, "\n\n<｜DSML｜ calls")
                    : NULL;
  if (ok && (!eos || eos + strlen(RIA_EOS) != copy + input_length))
    ok =
        ria_fail(e, RIA_INVALID_REQUEST, "completion lacks exact terminal EOS");
  size_t content_length =
      ok ? (size_t)((tool && tool < eos ? tool : eos) - at) : 0;
  if (ok && (!clean_content(reasoning, reasoning_length) ||
             !clean_content(at, content_length)))
    ok = ria_fail(e, RIA_INVALID_REQUEST,
                  "reserved token in completion content");
  builder b = {NULL, 0, 0, 67108864, e};
  if (ok)
    ok = literal(&b, "{\"role\": \"assistant\", \"content\": ") &&
         quoted(&b, at, content_length) &&
         literal(&b, ", \"reasoning_content\": ") &&
         quoted(&b, reasoning, reasoning_length) &&
         literal(&b, ", \"tool_calls\": ");
  if (ok && tool && tool < eos) {
    at = tool;
    ok = parse_calls(&b, &at, eos);
    if (ok && at != eos)
      ok = ria_fail(e, RIA_INVALID_REQUEST,
                    "unexpected content after DSML calls");
  } else if (ok)
    ok = literal(&b, "[]");
  if (ok)
    ok = literal(&b, "}");
  free(copy);
  if (ok) {
    ria_json_doc validated = {0};
    ok = ria_json_parse(b.text, b.length,
                        (ria_json_limits){67108864, 200000, 64}, &validated, e);
    ria_json_free(&validated);
  }
  if (!ok) {
    free(b.text);
    return false;
  }
  *json_out = b.text;
  *json_length = b.length;
  return true;
}
