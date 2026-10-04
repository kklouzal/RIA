#define _GNU_SOURCE
#include "json.h"
#include "ryu/ryu.h"
#include <errno.h>
#include <fcntl.h>
#include <locale.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
  const unsigned char *p, *end;
  ria_json_doc *doc;
  uint32_t cap, depth;
  size_t string_cap;
  locale_t locale;
  ria_error *error;
} parser;
static int key_compare(const void *a, const void *b);
/* Duplicate detection requires equality only. UTF8 byte ordering avoids the
 * UTF16 conversion needed exclusively for canonical object ordering. */
static int duplicate_key_compare(const void *a, const void *b) {
  const ria_json_node *x = *(const ria_json_node *const *)a,
                      *y = *(const ria_json_node *const *)b;
  size_t length = x->key_length < y->key_length ? x->key_length : y->key_length;
  int order = memcmp(x->key, y->key, length);
  if (order)
    return order;
  return x->key_length == y->key_length  ? 0
         : x->key_length < y->key_length ? -1
                                         : 1;
}
static bool distinct(parser *p, const ria_json_node *object) {
  size_t count = 0;
  for (uint32_t i = object->child; i != RIA_JSON_NONE;
       i = p->doc->nodes[i].next)
    count++;
  if (count < 2)
    return true;
  const ria_json_node **keys = malloc(count * sizeof *keys);
  if (!keys)
    return ria_fail(p->error, RIA_RESOURCE_LIMIT,
                    "duplicate-key check allocation failed");
  size_t at = 0;
  for (uint32_t i = object->child; i != RIA_JSON_NONE;
       i = p->doc->nodes[i].next)
    keys[at++] = &p->doc->nodes[i];
  qsort(keys, count, sizeof *keys, duplicate_key_compare);
  bool ok = true;
  for (size_t i = 1; i < count; i++)
    if (keys[i - 1]->key_length == keys[i]->key_length &&
        !memcmp(keys[i - 1]->key, keys[i]->key, keys[i]->key_length)) {
      ok = false;
      break;
    }
  free(keys);
  return ok || ria_fail(p->error, RIA_INVALID_REQUEST, "JSON: duplicate key");
}
static bool invalid(parser *p, const char *why) {
  return ria_fail(p->error, RIA_INVALID_REQUEST, "JSON: %s", why);
}
static void whitespace(parser *p) {
  while (p->p < p->end &&
         (*p->p == ' ' || *p->p == '\t' || *p->p == '\r' || *p->p == '\n'))
    p->p++;
}
static bool utf8(const unsigned char *s, size_t n, uint32_t *cp, size_t *used) {
  if (!n)
    return false;
  uint32_t v = s[0];
  size_t k = 1;
  uint32_t low = 0;
  if (v < 0x80) {
  } else if (v >= 0xc2 && v <= 0xdf) {
    v &= 31;
    k = 2;
    low = 0x80;
  } else if (v >= 0xe0 && v <= 0xef) {
    v &= 15;
    k = 3;
    low = 0x800;
  } else if (v >= 0xf0 && v <= 0xf4) {
    v &= 7;
    k = 4;
    low = 0x10000;
  } else
    return false;
  if (k > n)
    return false;
  for (size_t i = 1; i < k; i++) {
    if ((s[i] & 0xc0) != 0x80)
      return false;
    v = v << 6 | (s[i] & 63);
  }
  if (v < low || v > 0x10ffff || (v >= 0xd800 && v <= 0xdfff))
    return false;
  *cp = v;
  *used = k;
  return true;
}
static bool append_byte(parser *p, unsigned char b) {
  if (p->doc->string_bytes >= p->string_cap)
    return invalid(p, "string bound exceeded");
  p->doc->strings[p->doc->string_bytes++] = (char)b;
  return true;
}
static bool codepoint(parser *p, uint32_t cp) {
  if (cp < 0x80)
    return append_byte(p, (unsigned char)cp);
  if (cp < 0x800)
    return append_byte(p, (unsigned char)(0xc0 | (cp >> 6))) &&
           append_byte(p, (unsigned char)(0x80 | (cp & 63)));
  if (cp < 0x10000)
    return append_byte(p, (unsigned char)(0xe0 | (cp >> 12))) &&
           append_byte(p, (unsigned char)(0x80 | ((cp >> 6) & 63))) &&
           append_byte(p, (unsigned char)(0x80 | (cp & 63)));
  return append_byte(p, (unsigned char)(0xf0 | (cp >> 18))) &&
         append_byte(p, (unsigned char)(0x80 | ((cp >> 12) & 63))) &&
         append_byte(p, (unsigned char)(0x80 | ((cp >> 6) & 63))) &&
         append_byte(p, (unsigned char)(0x80 | (cp & 63)));
}
static bool hex4(parser *p, uint32_t *out) {
  if ((size_t)(p->end - p->p) < 4)
    return invalid(p, "truncated Unicode escape");
  uint32_t v = 0;
  for (unsigned i = 0; i < 4; i++) {
    unsigned c = *p->p++;
    unsigned d = c >= '0' && c <= '9'   ? c - '0'
                 : c >= 'a' && c <= 'f' ? c - 'a' + 10
                 : c >= 'A' && c <= 'F' ? c - 'A' + 10
                                        : 16;
    if (d > 15)
      return invalid(p, "invalid Unicode escape");
    v = v * 16 + d;
  }
  *out = v;
  return true;
}
static bool string_parse(parser *p, const char **out, size_t *length) {
  if (p->p == p->end || *p->p++ != '"')
    return invalid(p, "expected string");
  size_t start = p->doc->string_bytes;
  while (p->p < p->end) {
    unsigned c = *p->p++;
    if (c == '"') {
      *length = p->doc->string_bytes - start;
      if (!append_byte(p, 0))
        return false;
      *out = p->doc->strings + start;
      return true;
    }
    if (c < 32)
      return invalid(p, "unescaped control character");
    if (c == '\\') {
      if (p->p == p->end)
        return invalid(p, "truncated escape");
      c = *p->p++;
      switch (c) {
      case '"':
      case '\\':
      case '/':
        break;
      case 'b':
        c = 8;
        break;
      case 'f':
        c = 12;
        break;
      case 'n':
        c = 10;
        break;
      case 'r':
        c = 13;
        break;
      case 't':
        c = 9;
        break;
      case 'u': {
        uint32_t cp;
        if (!hex4(p, &cp))
          return false;
        if (cp >= 0xd800 && cp <= 0xdbff) {
          if (p->end - p->p < 2 || p->p[0] != '\\' || p->p[1] != 'u')
            return invalid(p, "lone high surrogate");
          p->p += 2;
          uint32_t tail;
          if (!hex4(p, &tail))
            return false;
          if (tail < 0xdc00 || tail > 0xdfff)
            return invalid(p, "invalid surrogate pair");
          cp = 0x10000 + ((cp - 0xd800) << 10) + (tail - 0xdc00);
        } else if (cp >= 0xdc00 && cp <= 0xdfff)
          return invalid(p, "lone low surrogate");
        if (!codepoint(p, cp))
          return false;
        continue;
      }
      default:
        return invalid(p, "unknown escape");
      }
      if (!append_byte(p, (unsigned char)c))
        return false;
    } else if (c >= 0x80) {
      p->p--;
      uint32_t cp;
      size_t used;
      if (!utf8(p->p, (size_t)(p->end - p->p), &cp, &used))
        return invalid(p, "malformed UTF-8");
      if (!codepoint(p, cp))
        return false;
      p->p += used;
    } else if (!append_byte(p, (unsigned char)c))
      return false;
  }
  return invalid(p, "unterminated string");
}
static bool value(parser *p, uint32_t depth, uint32_t *out) {
  whitespace(p);
  if (p->p == p->end)
    return invalid(p, "missing value");
  if (depth > p->depth)
    return invalid(p, "nesting limit exceeded");
  if (p->doc->count == p->cap)
    return invalid(p, "node limit exceeded");
  uint32_t i = p->doc->count++;
  ria_json_node *node = &p->doc->nodes[i];
  node->child = node->next = RIA_JSON_NONE;
  *out = i;
  unsigned c = *p->p;
  if (c == '"') {
    node->type = RIA_JSON_STRING;
    return string_parse(p, &node->text, &node->length);
  }
  if (c == '{' || c == '[') {
    bool object = c == '{';
    node->type = object ? RIA_JSON_OBJECT : RIA_JSON_ARRAY;
    p->p++;
    whitespace(p);
    if (p->p < p->end && *p->p == (object ? '}' : ']')) {
      p->p++;
      return true;
    }
    uint32_t tail = RIA_JSON_NONE;
    for (;;) {
      const char *key = NULL;
      size_t key_length = 0;
      if (object) {
        if (!string_parse(p, &key, &key_length))
          return false;
        whitespace(p);
        if (p->p == p->end || *p->p++ != ':')
          return invalid(p, "expected colon");
      }
      uint32_t child;
      if (!value(p, depth + 1, &child))
        return false;
      p->doc->nodes[child].key = key;
      p->doc->nodes[child].key_length = key_length;
      if (tail == RIA_JSON_NONE)
        node->child = child;
      else
        p->doc->nodes[tail].next = child;
      tail = child;
      whitespace(p);
      if (p->p == p->end)
        return invalid(p, "unterminated container");
      c = *p->p++;
      if (c == (object ? '}' : ']'))
        return !object || distinct(p, node);
      if (c != ',')
        return invalid(p, "expected comma");
      whitespace(p);
    }
  }
  if (c == 't' || c == 'f' || c == 'n') {
    const char *literal = c == 't' ? "true" : c == 'f' ? "false" : "null";
    size_t n = strlen(literal);
    if ((size_t)(p->end - p->p) < n || memcmp(p->p, literal, n))
      return invalid(p, "invalid literal");
    p->p += n;
    node->type = c == 'n' ? RIA_JSON_NULL : RIA_JSON_BOOL;
    node->boolean = c == 't';
    return true;
  }
  const unsigned char *start = p->p;
  if (c == '-')
    p->p++;
  if (p->p == p->end)
    return invalid(p, "truncated number");
  if (*p->p == '0')
    p->p++;
  else {
    if (*p->p < '1' || *p->p > '9')
      return invalid(p, "invalid number");
    while (p->p < p->end && *p->p >= '0' && *p->p <= '9')
      p->p++;
  }
  if (p->p < p->end && *p->p == '.') {
    p->p++;
    const unsigned char *digits = p->p;
    while (p->p < p->end && *p->p >= '0' && *p->p <= '9')
      p->p++;
    if (digits == p->p)
      return invalid(p, "empty fraction");
  }
  if (p->p < p->end && (*p->p == 'e' || *p->p == 'E')) {
    p->p++;
    if (p->p < p->end && (*p->p == '+' || *p->p == '-'))
      p->p++;
    const unsigned char *digits = p->p;
    while (p->p < p->end && *p->p >= '0' && *p->p <= '9')
      p->p++;
    if (digits == p->p)
      return invalid(p, "empty exponent");
  }
  node->type = RIA_JSON_NUMBER;
  node->length = (size_t)(p->p - start);
  size_t off = p->doc->string_bytes;
  for (size_t j = 0; j < node->length; j++)
    if (!append_byte(p, start[j]))
      return false;
  if (!append_byte(p, 0))
    return false;
  node->text = p->doc->strings + off;
  char *end = NULL;
  errno = 0;
  node->number = strtod_l(node->text, &end, p->locale);
  if (!isfinite(node->number) || end != node->text + node->length)
    return invalid(p, "nonfinite or invalid number");
  return true;
}
bool ria_json_parse(const void *bytes, size_t n, ria_json_limits limits,
                    ria_json_doc *doc, ria_error *e) {
  if (!doc)
    return ria_fail(e, RIA_INVALID_REQUEST, "null JSON document");
  memset(doc, 0, sizeof *doc);
  if (!bytes || !n || n > limits.max_bytes || n > PTRDIFF_MAX ||
      limits.max_nodes == 0 || limits.max_nodes > 1000000 ||
      limits.max_depth == 0 || limits.max_depth > 128)
    return ria_fail(e, RIA_RESOURCE_LIMIT, "invalid JSON bounds");
  uint64_t possible = (uint64_t)n / 2 + 1;
  uint32_t cap =
      possible < limits.max_nodes ? (uint32_t)possible : limits.max_nodes;
  uint64_t node_bytes;
  if (!ria_u64_mul(cap, sizeof *doc->nodes, &node_bytes) ||
      !ria_u64_add(node_bytes, n + 1, &doc->allocated_bytes))
    return ria_fail(e, RIA_RESOURCE_LIMIT, "JSON allocation capacity overflow");
  doc->nodes = calloc(cap, sizeof *doc->nodes);
  doc->strings = malloc(n + 1);
  if (!doc->nodes || !doc->strings) {
    ria_json_free(doc);
    return ria_fail(e, RIA_RESOURCE_LIMIT, "JSON allocation failed");
  }
  locale_t locale = newlocale(LC_NUMERIC_MASK, "C", (locale_t)0);
  if (!locale) {
    ria_json_free(doc);
    return ria_fail(e, RIA_INTERNAL_ERROR, "C numeric locale unavailable");
  }
  parser p = {(const unsigned char *)bytes,
              (const unsigned char *)bytes + n,
              doc,
              cap,
              limits.max_depth,
              n + 1,
              locale,
              e};
  uint32_t root;
  bool ok = value(&p, 0, &root);
  whitespace(&p);
  if (ok && p.p != p.end)
    ok = invalid(&p, "trailing bytes");
  freelocale(locale);
  if (!ok)
    ria_json_free(doc);
  return ok;
}
bool ria_json_read(const char *path, ria_json_limits limits, ria_json_doc *doc,
                   ria_error *e) {
  if (!doc)
    return ria_fail(e, RIA_INVALID_REQUEST, "null JSON document");
  memset(doc, 0, sizeof *doc);
  if (!path || !limits.max_bytes || limits.max_bytes >= PTRDIFF_MAX)
    return ria_fail(e, RIA_INVALID_REQUEST, "invalid JSON read bounds");
  int fd = open(path, O_RDONLY | O_CLOEXEC | O_NONBLOCK);
  if (fd < 0)
    return ria_fail(e, RIA_INVALID_REQUEST, "cannot open JSON input");
  struct stat metadata;
  if (fstat(fd, &metadata) || !S_ISREG(metadata.st_mode) ||
      metadata.st_size <= 0 || (uint64_t)metadata.st_size > limits.max_bytes) {
    close(fd);
    return ria_fail(e, RIA_RESOURCE_LIMIT,
                    "JSON input is not a bounded nonempty regular file");
  }
  size_t expected = (size_t)metadata.st_size;
  FILE *f = fdopen(fd, "rb");
  if (!f) {
    close(fd);
    return ria_fail(e, RIA_RESOURCE_LIMIT,
                    "cannot initialize JSON input reader");
  }
  char *buffer = malloc(expected + 1);
  if (!buffer) {
    fclose(f);
    return ria_fail(e, RIA_RESOURCE_LIMIT, "JSON read allocation failed");
  }
  size_t n = fread(buffer, 1, expected + 1, f);
  bool ok = !ferror(f) && n == expected;
  if (fclose(f) != 0)
    ok = false;
  if (!ok) {
    free(buffer);
    return ria_fail(e, RIA_INVALID_REQUEST,
                    "JSON input changed size or read/close failed");
  }
  ok = ria_json_parse(buffer, n, limits, doc, e);
  free(buffer);
  return ok;
}
void ria_json_free(ria_json_doc *doc) {
  if (doc) {
    free(doc->nodes);
    free(doc->strings);
    memset(doc, 0, sizeof *doc);
  }
}
const ria_json_node *ria_json_at(const ria_json_doc *doc, uint32_t i) {
  return doc && i < doc->count ? &doc->nodes[i] : NULL;
}
uint32_t ria_json_get(const ria_json_doc *doc, uint32_t object,
                      const char *key) {
  const ria_json_node *n = ria_json_at(doc, object);
  if (!n || n->type != RIA_JSON_OBJECT || !key)
    return RIA_JSON_NONE;
  size_t length = strlen(key);
  for (uint32_t i = n->child; i != RIA_JSON_NONE; i = doc->nodes[i].next)
    if (doc->nodes[i].key_length == length &&
        !memcmp(doc->nodes[i].key, key, length))
      return i;
  return RIA_JSON_NONE;
}
bool ria_json_fields(const ria_json_doc *doc, uint32_t object,
                     const char *const *allowed, size_t ac,
                     const char *const *required, size_t rc, ria_error *e) {
  const ria_json_node *n = ria_json_at(doc, object);
  if (!n || n->type != RIA_JSON_OBJECT)
    return ria_fail(e, RIA_INVALID_REQUEST, "expected JSON object");
  for (uint32_t i = n->child; i != RIA_JSON_NONE; i = doc->nodes[i].next) {
    bool found = false;
    for (size_t j = 0; j < ac; j++) {
      size_t l = strlen(allowed[j]);
      if (l == doc->nodes[i].key_length &&
          !memcmp(doc->nodes[i].key, allowed[j], l)) {
        found = true;
        break;
      }
    }
    if (!found)
      return ria_fail(e, RIA_INVALID_REQUEST, "unknown JSON field");
  }
  for (size_t j = 0; j < rc; j++)
    if (ria_json_get(doc, object, required[j]) == RIA_JSON_NONE)
      return ria_fail(e, RIA_INVALID_REQUEST, "required JSON field missing: %s",
                      required[j]);
  return true;
}
bool ria_json_u64(const ria_json_doc *doc, uint32_t index, bool string,
                  uint64_t *out, ria_error *e) {
  const ria_json_node *n = ria_json_at(doc, index);
  if (!n || !out)
    return ria_fail(e, RIA_INVALID_REQUEST, "missing integer");
  if (string) {
    if (n->type != RIA_JSON_STRING)
      return ria_fail(e, RIA_INVALID_REQUEST, "u64 must be decimal string");
    return ria_parse_u64(n->text, n->length, out, e);
  }
  if (n->type != RIA_JSON_NUMBER || n->number < 0 ||
      n->number > (double)RIA_JSON_SAFE_INTEGER ||
      trunc(n->number) != n->number)
    return ria_fail(e, RIA_INVALID_REQUEST, "expected unsigned safe integer");
  /* Check the source decimal exactly, independently of binary64 rounding and
   * locale. */
  const char *s = n->text;
  size_t at = 0;
  bool negative = s[at] == '-';
  if (negative)
    at++;
  size_t first = SIZE_MAX, last = 0, digits = 0, fraction = 0;
  bool fractional = false;
  for (; at < n->length && s[at] != 'e' && s[at] != 'E'; at++) {
    if (s[at] == '.') {
      fractional = true;
      continue;
    }
    if (s[at] != '0') {
      if (first == SIZE_MAX)
        first = digits;
      last = digits;
    }
    digits++;
    if (fractional)
      fraction++;
  }
  if (first == SIZE_MAX) {
    *out = 0;
    return true;
  }
  if (negative)
    return ria_fail(e, RIA_INVALID_REQUEST, "negative unsigned integer");
  int64_t exponent = 0;
  bool exponent_negative = false;
  if (at < n->length) {
    at++;
    if (s[at] == '+' || s[at] == '-') {
      exponent_negative = s[at] == '-';
      at++;
    }
    for (; at < n->length; at++)
      if (exponent < 1000000)
        exponent = exponent * 10 + s[at] - '0';
    if (exponent_negative)
      exponent = -exponent;
  }
  exponent -= (int64_t)fraction;
  exponent += (int64_t)(digits - last - 1);
  size_t significant = last - first + 1;
  if (exponent < 0 || significant > 16 || exponent > 16 ||
      (uint64_t)exponent + significant > 16)
    return ria_fail(e, RIA_INVALID_REQUEST,
                    "JSON integer is fractional or out of exact range");
  uint64_t result = 0;
  size_t di = 0;
  for (at = negative ? 1 : 0; at < n->length && s[at] != 'e' && s[at] != 'E';
       at++) {
    if (s[at] == '.')
      continue;
    if (di >= first && di <= last)
      result = result * 10 + (unsigned)(s[at] - '0');
    di++;
  }
  while (exponent--)
    result *= 10;
  if (result > RIA_JSON_SAFE_INTEGER)
    return ria_fail(e, RIA_INVALID_REQUEST, "JSON integer exceeds safe range");
  *out = result;
  return true;
}
bool ria_json_string(const ria_json_doc *doc, uint32_t index, const char **out,
                     size_t *length, ria_error *e) {
  const ria_json_node *n = ria_json_at(doc, index);
  if (!n || n->type != RIA_JSON_STRING)
    return ria_fail(e, RIA_INVALID_REQUEST, "expected JSON string");
  if (out)
    *out = n->text;
  if (length)
    *length = n->length;
  return true;
}
bool ria_json_digest_field(const ria_json_doc *doc, uint32_t index,
                           uint8_t out[32], ria_error *e) {
  const char *text = NULL;
  size_t n = 0;
  return ria_json_string(doc, index, &text, &n, e) &&
         ria_hex_decode(text, n, out, 32, e);
}

typedef struct {
  char *bytes;
  size_t length, cap;
  ria_error *error;
} writer;
static bool emit(writer *w, const char *bytes, size_t n) {
  if (n > w->cap - w->length)
    return ria_fail(w->error, RIA_RESOURCE_LIMIT,
                    "canonical JSON bound exceeded");
  memcpy(w->bytes + w->length, bytes, n);
  w->length += n;
  return true;
}
static bool quoted(writer *w, const char *s, size_t n) {
  if (!emit(w, "\"", 1))
    return false;
  static const char hex[] = "0123456789abcdef";
  for (size_t i = 0; i < n; i++) {
    unsigned c = (unsigned char)s[i];
    const char *escape = NULL;
    switch (c) {
    case '"':
      escape = "\\\"";
      break;
    case '\\':
      escape = "\\\\";
      break;
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
    }
    if (escape) {
      if (!emit(w, escape, 2))
        return false;
    } else if (c < 32) {
      char b[6] = {'\\', 'u', '0', '0', hex[c >> 4], hex[c & 15]};
      if (!emit(w, b, 6))
        return false;
    } else if (!emit(w, s + i, 1))
      return false;
  }
  return emit(w, "\"", 1);
}
static size_t number_text(double v, char out[64]) {
  if (v == 0) {
    out[0] = '0';
    return 1;
  }
  char raw[32] = {0};
  int n = d2s_buffered_n(v, raw);
  if (n < 3 || n > 31)
    return 0;
  int pos = 0;
  size_t used = 0;
  if (raw[0] == '-') {
    out[used++] = '-';
    pos = 1;
  }
  char digits[20];
  int count = 0;
  while (pos < n && raw[pos] != 'E') {
    if (raw[pos] != '.') {
      if (raw[pos] < '0' || raw[pos] > '9' || count == 20)
        return 0;
      digits[count++] = raw[pos];
    }
    pos++;
  }
  if (!count)
    return 0;
  int exponent = 0;
  if (pos < n) {
    pos++;
    bool negative = false;
    if (pos < n && raw[pos] == '-') {
      negative = true;
      pos++;
    }
    for (; pos < n; pos++)
      exponent = exponent * 10 + raw[pos] - '0';
    if (negative)
      exponent = -exponent;
  }
  while (count > 1 && digits[count - 1] == '0')
    count--;
  int point = exponent + 1;
  if (exponent >= -6 && exponent < 21) {
    if (point <= 0) {
      out[used++] = '0';
      out[used++] = '.';
      for (int i = 0; i < -point; i++)
        out[used++] = '0';
      for (int i = 0; i < count; i++)
        out[used++] = digits[i];
    } else {
      for (int i = 0; i < point || i < count; i++) {
        if (i == point)
          out[used++] = '.';
        out[used++] = i < count ? digits[i] : '0';
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
    if (exponent >= 0)
      out[used++] = '+';
    else {
      out[used++] = '-';
      exponent = -exponent;
    }
    char rev[5];
    int k = 0;
    do {
      rev[k++] = (char)('0' + exponent % 10);
      exponent /= 10;
    } while (exponent);
    while (k)
      out[used++] = rev[--k];
  }
  return used;
}
typedef struct {
  const unsigned char *s;
  size_t n, pos;
  uint16_t low;
} utf16_iterator;
static uint32_t utf16_next(utf16_iterator *it) {
  if (it->low) {
    uint16_t v = it->low;
    it->low = 0;
    return v;
  }
  if (it->pos == it->n)
    return UINT32_MAX;
  uint32_t cp;
  size_t used;
  if (!utf8(it->s + it->pos, it->n - it->pos, &cp, &used))
    return UINT32_MAX;
  it->pos += used;
  if (cp > 0xffff) {
    cp -= 0x10000;
    it->low = (uint16_t)(0xdc00 + (cp & 1023));
    return 0xd800 + (cp >> 10);
  }
  return cp;
}
static int key_compare(const void *a, const void *b) {
  const ria_json_node *x = *(const ria_json_node *const *)a,
                      *y = *(const ria_json_node *const *)b;
  utf16_iterator ix = {(const unsigned char *)x->key, x->key_length, 0, 0},
                 iy = {(const unsigned char *)y->key, y->key_length, 0, 0};
  for (;;) {
    uint32_t cx = utf16_next(&ix), cy = utf16_next(&iy);
    if (cx != cy) {
      if (cx == UINT32_MAX)
        return -1;
      if (cy == UINT32_MAX)
        return 1;
      return cx < cy ? -1 : 1;
    }
    if (cx == UINT32_MAX)
      return 0;
  }
}
static bool encode(const ria_json_doc *doc, uint32_t i, writer *w, bool exclude,
                   unsigned depth) {
  if (depth > 128)
    return ria_fail(w->error, RIA_INVALID_REQUEST,
                    "canonical nesting exceeded");
  const ria_json_node *n = ria_json_at(doc, i);
  if (!n)
    return ria_fail(w->error, RIA_INVALID_REQUEST, "invalid JSON node");
  switch (n->type) {
  case RIA_JSON_NULL:
    return emit(w, "null", 4);
  case RIA_JSON_BOOL:
    return emit(w, n->boolean ? "true" : "false", n->boolean ? 4 : 5);
  case RIA_JSON_STRING:
    return quoted(w, n->text, n->length);
  case RIA_JSON_NUMBER: {
    if (!isfinite(n->number))
      return ria_fail(w->error, RIA_INVALID_REQUEST,
                      "nonfinite canonical number");
    char b[64];
    size_t l = number_text(n->number, b);
    if (!l)
      return ria_fail(w->error, RIA_INTERNAL_ERROR,
                      "shortest number formatter violated its contract");
    return emit(w, b, l);
  }
  case RIA_JSON_ARRAY: {
    if (!emit(w, "[", 1))
      return false;
    bool first = true;
    for (uint32_t j = n->child; j != RIA_JSON_NONE; j = doc->nodes[j].next) {
      if (!first && !emit(w, ",", 1))
        return false;
      first = false;
      if (!encode(doc, j, w, false, depth + 1))
        return false;
    }
    return emit(w, "]", 1);
  }
  case RIA_JSON_OBJECT: {
    size_t count = 0;
    for (uint32_t j = n->child; j != RIA_JSON_NONE; j = doc->nodes[j].next)
      count++;
    const ria_json_node **keys = malloc((count ? count : 1) * sizeof *keys);
    if (!keys)
      return ria_fail(w->error, RIA_RESOURCE_LIMIT,
                      "canonical key allocation failed");
    size_t at = 0;
    for (uint32_t j = n->child; j != RIA_JSON_NONE; j = doc->nodes[j].next) {
      const ria_json_node *k = &doc->nodes[j];
      if (exclude &&
          ((k->key_length == 6 && !memcmp(k->key, "digest", 6)) ||
           (k->key_length == 10 && !memcmp(k->key, "signatures", 10))))
        continue;
      keys[at++] = k;
    }
    qsort(keys, at, sizeof *keys, key_compare);
    bool ok = emit(w, "{", 1);
    for (size_t j = 0; j < at && ok; j++) {
      if (j)
        ok = emit(w, ",", 1);
      if (ok)
        ok = quoted(w, keys[j]->key, keys[j]->key_length) && emit(w, ":", 1) &&
             encode(doc, (uint32_t)(keys[j] - doc->nodes), w, false, depth + 1);
    }
    if (ok)
      ok = emit(w, "}", 1);
    free(keys);
    return ok;
  }
  }
  return ria_fail(w->error, RIA_INVALID_REQUEST, "unsupported JSON type");
}
bool ria_json_canonical(const ria_json_doc *doc, bool exclude, char **out,
                        size_t *length, ria_error *e) {
  if (!doc || !doc->count || !out || !length)
    return ria_fail(e, RIA_INVALID_REQUEST, "empty canonical input");
  uint64_t cap, a;
  if (!ria_u64_mul(doc->string_bytes, 6, &cap) ||
      !ria_u64_mul(doc->count, 64, &a) || !ria_u64_add(cap, a, &cap) ||
      !ria_u64_add(cap, 1, &cap))
    return ria_fail(e, RIA_RESOURCE_LIMIT, "canonical size overflow");
  size_t bytes;
  if (!ria_size(cap, &bytes, e))
    return false;
  writer w = {malloc(bytes), 0, bytes - 1, e};
  if (!w.bytes)
    return ria_fail(e, RIA_RESOURCE_LIMIT, "canonical allocation failed");
  if (!encode(doc, 0, &w, exclude, 0)) {
    free(w.bytes);
    return false;
  }
  w.bytes[w.length] = 0;
  *out = w.bytes;
  *length = w.length;
  return true;
}
bool ria_json_sha256(const ria_json_doc *doc, bool exclude, uint8_t digest[32],
                     ria_error *e) {
  char *bytes;
  size_t length;
  if (!ria_json_canonical(doc, exclude, &bytes, &length, e))
    return false;
  bool ok = ria_sha256(bytes, length, digest, e);
  free(bytes);
  return ok;
}
