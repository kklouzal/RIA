#define _POSIX_C_SOURCE 200809L
#include "tokenizer.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <oniguruma.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define TOKEN_VOCAB 129280u
#define VOCAB_TABLE 524288u
#define MERGE_TABLE 262144u
typedef struct {
  const char *encoded;
  size_t encoded_length;
  char *decoded;
  size_t decoded_length;
  bool special, added;
} token;
typedef struct {
  uint32_t child, sibling, token_id;
  unsigned char byte;
} special_node;
typedef struct {
  uint64_t key;
  uint32_t rank, output;
} merge;
struct ria_tokenizer {
  ria_json_doc source;
  token *vocab;
  uint32_t *index;
  merge *merges;
  OnigRegex regex[3];
  uint32_t bytes[256];
  uint64_t owned, budget;
  size_t source_bytes;
  special_node *specials;
  uint32_t special_count, special_capacity;
};
/* Oniguruma owns process-wide Unicode runtime state. This explicit owner
 * serializes first initialization and last teardown; objects keep a runtime
 * reference for their entire lifetime. It does not hold tokenizer data. */
static struct {
  pthread_mutex_t mutex;
  uint64_t owners;
} unicode_runtime = {PTHREAD_MUTEX_INITIALIZER, 0};
bool ria_tokenizer_runtime_begin(ria_error *e) {
  pthread_mutex_lock(&unicode_runtime.mutex);
  bool ok = unicode_runtime.owners < UINT64_MAX;
  if (ok && !unicode_runtime.owners) {
    OnigEncoding encodings[] = {ONIG_ENCODING_UTF8};
    ok = onig_initialize(encodings, 1) == ONIG_NORMAL;
  }
  if (ok)
    unicode_runtime.owners++;
  pthread_mutex_unlock(&unicode_runtime.mutex);
  return ok || ria_fail(e, RIA_INTERNAL_ERROR,
                        "tokenizer Unicode runtime initialization failed");
}
void ria_tokenizer_runtime_end(void) {
  pthread_mutex_lock(&unicode_runtime.mutex);
  if (unicode_runtime.owners && !--unicode_runtime.owners)
    (void)onig_end();
  pthread_mutex_unlock(&unicode_runtime.mutex);
}
static uint64_t hash(const void *s, size_t length) {
  const unsigned char *p = s;
  uint64_t h = UINT64_C(14695981039346656037);
  for (size_t i = 0; i < length; i++)
    h = (h ^ p[i]) * UINT64_C(1099511628211);
  return h;
}
static uint32_t lookup(const ria_tokenizer *t, const char *s, size_t n) {
  uint32_t at = (uint32_t)(hash(s, n) & (VOCAB_TABLE - 1));
  for (uint32_t probes = 0; probes < VOCAB_TABLE; probes++) {
    uint32_t entry = t->index[at];
    if (!entry)
      return UINT32_MAX;
    const token *v = &t->vocab[entry - 1];
    if (v->encoded_length == n && !memcmp(v->encoded, s, n))
      return entry - 1;
    at = (at + 1) & (VOCAB_TABLE - 1);
  }
  return UINT32_MAX;
}
static bool insert(ria_tokenizer *t, uint32_t id, ria_error *e) {
  token *v = &t->vocab[id];
  uint32_t at =
      (uint32_t)(hash(v->encoded, v->encoded_length) & (VOCAB_TABLE - 1));
  for (uint32_t i = 0; i < VOCAB_TABLE; i++) {
    if (!t->index[at]) {
      t->index[at] = id + 1;
      return true;
    }
    token *old = &t->vocab[t->index[at] - 1];
    if (old->encoded_length == v->encoded_length &&
        !memcmp(old->encoded, v->encoded, v->encoded_length))
      return ria_fail(e, RIA_INTEGRITY_ERROR,
                      "duplicate tokenizer vocabulary text");
    at = (at + 1) & (VOCAB_TABLE - 1);
  }
  return ria_fail(e, RIA_RESOURCE_LIMIT, "tokenizer hash table exhausted");
}
static bool utf8_one(const unsigned char *s, size_t n, uint32_t *cp,
                     size_t *used) {
  if (!n)
    return false;
  uint32_t v = s[0], minimum = 0;
  size_t k = 1;
  if (v < 128) {
  } else if (v >= 0xc2 && v <= 0xdf) {
    k = 2;
    v &= 31;
    minimum = 128;
  } else if (v >= 0xe0 && v <= 0xef) {
    k = 3;
    v &= 15;
    minimum = 2048;
  } else if (v >= 0xf0 && v <= 0xf4) {
    k = 4;
    v &= 7;
    minimum = 65536;
  } else
    return false;
  if (k > n)
    return false;
  for (size_t i = 1; i < k; i++) {
    if ((s[i] & 0xc0) != 0x80)
      return false;
    v = v << 6 | (s[i] & 63);
  }
  if (v < minimum || v > 0x10ffff || (v >= 0xd800 && v <= 0xdfff))
    return false;
  *cp = v;
  *used = k;
  return true;
}
bool ria_utf8_decode(ria_utf8_decoder *d, const uint8_t *s, size_t n,
                     bool final, char *out, size_t cap, size_t *written,
                     ria_error *e) {
  uint64_t maximum;
  if (!d || (!s && n) || !out || !written || d->length > 3 ||
      !ria_u64_add(n, d->length, &maximum) ||
      !ria_u64_mul(maximum, 3, &maximum) || maximum > cap)
    return ria_fail(e, RIA_RESOURCE_LIMIT, "UTF8 decoder capacity exceeded");
  *written = 0;
  for (;;) {
    if (!d->length) {
      if (!n)
        break;
      d->pending[d->length++] = *s++;
      n--;
    }
    unsigned lead = d->pending[0];
    unsigned need = lead < 0x80                    ? 1
                    : lead >= 0xc2 && lead <= 0xdf ? 2
                    : lead >= 0xe0 && lead <= 0xef ? 3
                    : lead >= 0xf0 && lead <= 0xf4 ? 4
                                                   : 0;
    bool invalid = !need;
    while (!invalid && d->length < need && n) {
      unsigned next = *s;
      bool valid = next >= 0x80 && next <= 0xbf;
      if (d->length == 1)
        valid = valid && !(lead == 0xe0 && next < 0xa0) &&
                !(lead == 0xed && next >= 0xa0) &&
                !(lead == 0xf0 && next < 0x90) &&
                !(lead == 0xf4 && next >= 0x90);
      if (!valid) {
        invalid = true;
        break;
      }
      d->pending[d->length++] = *s++;
      n--;
    }
    if (!invalid && d->length < need) {
      if (!final)
        break;
      invalid = true;
    }
    if (invalid) {
      memcpy(out + *written, "\xef\xbf\xbd", 3);
      *written += 3;
    } else {
      memcpy(out + *written, d->pending, need);
      *written += need;
    }
    d->length = 0;
  }
  return true;
}
static unsigned alphabet(uint32_t b) {
  return (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174;
}
static uint32_t codepoint_for_byte(unsigned byte) {
  if (alphabet(byte))
    return byte;
  /* Source ByteLevel appends the excluded bytes in their integer order:
   * 0..32, 127..160, then 173. */
  if (byte <= 32)
    return 256 + byte;
  if (byte <= 160)
    return 289 + byte - 127;
  return 323;
}
static unsigned byte_for_codepoint(uint32_t cp) {
  if (cp < 256 && alphabet(cp))
    return cp;
  if (cp >= 256 && cp < 289)
    return cp - 256;
  if (cp >= 289 && cp < 323)
    return cp - 162;
  if (cp == 323)
    return 173;
  return 256;
}
static size_t glyph(uint32_t cp, char out[4]) {
  if (cp < 128) {
    out[0] = (char)cp;
    return 1;
  }
  out[0] = (char)(0xc0 | (cp >> 6));
  out[1] = (char)(0x80 | (cp & 63));
  return 2;
}
static bool json_string_is(const ria_json_doc *d, uint32_t i, const char *s) {
  const ria_json_node *n = ria_json_at(d, i);
  return n && n->type == RIA_JSON_STRING && n->length == strlen(s) &&
         !memcmp(n->text, s, n->length);
}
static bool plain_bool(const ria_json_doc *d, uint32_t o, const char *name,
                       bool expected) {
  const ria_json_node *n = ria_json_at(d, ria_json_get(d, o, name));
  return n && n->type == RIA_JSON_BOOL && n->boolean == expected;
}
static uint32_t child_at(const ria_json_doc *d, uint32_t array,
                         unsigned position) {
  const ria_json_node *n = ria_json_at(d, array);
  if (!n || n->type != RIA_JSON_ARRAY)
    return RIA_JSON_NONE;
  uint32_t i = n->child;
  while (position-- && i != RIA_JSON_NONE)
    i = d->nodes[i].next;
  return i;
}
static bool metadata(ria_tokenizer *t, ria_error *e) {
  const ria_json_doc *d = &t->source;
  static const char *const top[] = {
      "version",        "truncation", "padding",
      "added_tokens",   "normalizer", "pre_tokenizer",
      "post_processor", "decoder",    "model"};
  if (!ria_json_fields(d, 0, top, 9, top, 9, e) ||
      !json_string_is(d, ria_json_get(d, 0, "version"), "1.0"))
    return ria_fail(e, RIA_UNSUPPORTED, "unsupported tokenizer document");
  const ria_json_node *n = ria_json_at(d, ria_json_get(d, 0, "truncation")),
                      *p = ria_json_at(d, ria_json_get(d, 0, "padding"));
  if (!n || !p || n->type != RIA_JSON_NULL || p->type != RIA_JSON_NULL)
    return ria_fail(e, RIA_UNSUPPORTED,
                    "runtime tokenizer truncation/padding forbidden");
  uint32_t normalizer = ria_json_get(d, 0, "normalizer");
  const ria_json_node *normalizers =
      ria_json_at(d, ria_json_get(d, normalizer, "normalizers"));
  if (!json_string_is(d, ria_json_get(d, normalizer, "type"), "Sequence") ||
      !normalizers || normalizers->type != RIA_JSON_ARRAY ||
      normalizers->child != RIA_JSON_NONE)
    return ria_fail(e, RIA_UNSUPPORTED, "unsupported tokenizer normalization");
  uint32_t pretok = ria_json_get(d, 0, "pre_tokenizer"),
           splits = ria_json_get(d, pretok, "pretokenizers");
  if (!json_string_is(d, ria_json_get(d, pretok, "type"), "Sequence"))
    return ria_fail(e, RIA_UNSUPPORTED,
                    "unsupported tokenizer pre-split pipeline");
  for (unsigned i = 0; i < 3; i++) {
    uint32_t split = child_at(d, splits, i);
    const char *regex = NULL;
    size_t length = 0;
    uint32_t pattern = ria_json_get(d, split, "pattern");
    if (!json_string_is(d, ria_json_get(d, split, "type"), "Split") ||
        !json_string_is(d, ria_json_get(d, split, "behavior"), "Isolated") ||
        !plain_bool(d, split, "invert", false) ||
        !ria_json_string(d, ria_json_get(d, pattern, "Regex"), &regex, &length,
                         e))
      return ria_fail(e, RIA_UNSUPPORTED, "unsupported tokenizer split");
    OnigErrorInfo info;
    int rc = onig_new(&t->regex[i], (const OnigUChar *)regex,
                      (const OnigUChar *)regex + length, ONIG_OPTION_NONE,
                      ONIG_ENCODING_UTF8, ONIG_SYNTAX_RUBY, &info);
    if (rc != ONIG_NORMAL)
      return ria_fail(e, RIA_UNSUPPORTED, "tokenizer regex compilation failed");
  }
  uint32_t bytelevel = child_at(d, splits, 3);
  if (child_at(d, splits, 4) != RIA_JSON_NONE ||
      !json_string_is(d, ria_json_get(d, bytelevel, "type"), "ByteLevel") ||
      !plain_bool(d, bytelevel, "add_prefix_space", false) ||
      !plain_bool(d, bytelevel, "use_regex", false))
    return ria_fail(e, RIA_UNSUPPORTED, "unsupported tokenizer byte encoding");
  uint32_t model = ria_json_get(d, 0, "model");
  static const char *const model_fields[] = {"type",
                                             "dropout",
                                             "unk_token",
                                             "continuing_subword_prefix",
                                             "end_of_word_suffix",
                                             "fuse_unk",
                                             "byte_fallback",
                                             "vocab",
                                             "merges",
                                             "ignore_merges"};
  if (!ria_json_fields(d, model, model_fields, 10, model_fields, 9, e) ||
      !json_string_is(d, ria_json_get(d, model, "type"), "BPE") ||
      !plain_bool(d, model, "fuse_unk", false) ||
      !plain_bool(d, model, "byte_fallback", false))
    return ria_fail(e, RIA_UNSUPPORTED, "unsupported BPE realization");
  for (unsigned i = 1; i <= 4; i++) {
    const ria_json_node *v =
        ria_json_at(d, ria_json_get(d, model, model_fields[i]));
    if (!v || v->type != RIA_JSON_NULL)
      return ria_fail(e, RIA_UNSUPPORTED, "noncanonical BPE options");
  }
  if (ria_json_get(d, model, "ignore_merges") != RIA_JSON_NONE &&
      !plain_bool(d, model, "ignore_merges", false))
    return ria_fail(e, RIA_UNSUPPORTED, "BPE merge skipping forbidden");
  const ria_json_node *vocab = ria_json_at(d, ria_json_get(d, model, "vocab"));
  if (!vocab || vocab->type != RIA_JSON_OBJECT)
    return ria_fail(e, RIA_INVALID_REQUEST, "tokenizer vocabulary missing");
  uint64_t count = 0;
  for (uint32_t i = vocab->child; i != RIA_JSON_NONE; i = d->nodes[i].next) {
    uint64_t id;
    if (!ria_json_u64(d, i, false, &id, e) || id >= TOKEN_VOCAB ||
        t->vocab[id].encoded || !d->nodes[i].key_length)
      return ria_fail(e, RIA_INTEGRITY_ERROR,
                      "invalid tokenizer vocabulary identity");
    t->vocab[id].encoded = d->nodes[i].key;
    t->vocab[id].encoded_length = d->nodes[i].key_length;
    count++;
  }
  if (count != 128000)
    return ria_fail(e, RIA_INTEGRITY_ERROR,
                    "tokenizer source vocabulary size mismatch");
  const ria_json_node *added =
      ria_json_at(d, ria_json_get(d, 0, "added_tokens"));
  if (!added || added->type != RIA_JSON_ARRAY)
    return ria_fail(e, RIA_INVALID_REQUEST, "tokenizer added tokens missing");
  for (uint32_t i = added->child; i != RIA_JSON_NONE; i = d->nodes[i].next) {
    uint64_t id;
    const char *s = NULL;
    size_t length = 0;
    static const char *const fields[] = {"id",     "content", "single_word",
                                         "lstrip", "rstrip",  "normalized",
                                         "special"};
    if (!ria_json_fields(d, i, fields, 7, fields, 7, e) ||
        !ria_json_u64(d, ria_json_get(d, i, "id"), false, &id, e) ||
        id >= TOKEN_VOCAB ||
        !ria_json_string(d, ria_json_get(d, i, "content"), &s, &length, e) ||
        !length || !plain_bool(d, i, "single_word", false) ||
        !plain_bool(d, i, "lstrip", false) ||
        !plain_bool(d, i, "rstrip", false) ||
        (!plain_bool(d, i, "normalized", false) &&
         !plain_bool(d, i, "normalized", true)) ||
        (!plain_bool(d, i, "special", false) &&
         !plain_bool(d, i, "special", true)) ||
        t->vocab[id].added)
      return ria_fail(e, RIA_UNSUPPORTED, "unsupported added-token semantics");
    token *v = &t->vocab[id];
    if (v->encoded &&
        (v->encoded_length != length || memcmp(v->encoded, s, length)))
      return ria_fail(e, RIA_INTEGRITY_ERROR,
                      "added-token identity conflicts with BPE");
    v->encoded = s;
    v->encoded_length = length;
    v->special = plain_bool(d, i, "special", true);
    v->added = true;
  }
  uint64_t special_capacity = 1;
  for (uint32_t i = 0; i < TOKEN_VOCAB; i++)
    if (t->vocab[i].added)
      if (!ria_u64_add(special_capacity, t->vocab[i].encoded_length,
                       &special_capacity))
        return ria_fail(e, RIA_RESOURCE_LIMIT,
                        "special token trie size overflow");
  uint64_t special_bytes;
  if (special_capacity > UINT32_MAX ||
      !ria_u64_mul(special_capacity, sizeof(special_node), &special_bytes) ||
      !ria_u64_add(t->owned, special_bytes, &t->owned) || t->owned > t->budget)
    return ria_fail(e, RIA_RESOURCE_LIMIT,
                    "special token trie exceeds metadata budget");
  t->specials = calloc((size_t)special_capacity, sizeof *t->specials);
  if (!t->specials)
    return ria_fail(e, RIA_RESOURCE_LIMIT,
                    "special token trie allocation failed");
  t->special_capacity = (uint32_t)special_capacity;
  t->special_count = 1;
  for (uint32_t i = 0; i < TOKEN_VOCAB; i++) {
    token *v = &t->vocab[i];
    if (!v->encoded || !insert(t, i, e))
      return ria_fail(e, RIA_INTEGRITY_ERROR,
                      "incomplete tokenizer vocabulary");
    if (v->added) {
      uint32_t node = 0;
      for (size_t j = 0; j < v->encoded_length; j++) {
        unsigned char byte = (unsigned char)v->encoded[j];
        uint32_t child = t->specials[node].child;
        while (child && t->specials[child].byte != byte)
          child = t->specials[child].sibling;
        if (!child) {
          child = t->special_count++;
          t->specials[child].byte = byte;
          t->specials[child].sibling = t->specials[node].child;
          t->specials[node].child = child;
        }
        node = child;
      }
      t->specials[node].token_id = i + 1;
    }
    if (!ria_u64_add(t->owned, v->encoded_length + 1, &t->owned) ||
        t->owned > t->budget)
      return ria_fail(e, RIA_RESOURCE_LIMIT,
                      "tokenizer metadata budget exceeded");
    v->decoded = malloc(v->encoded_length + 1);
    if (!v->decoded)
      return ria_fail(e, RIA_RESOURCE_LIMIT,
                      "tokenizer decoding allocation failed");

    if (v->added) {
      memcpy(v->decoded, v->encoded, v->encoded_length);
      v->decoded_length = v->encoded_length;
    } else
      for (size_t at = 0; at < v->encoded_length;) {
        uint32_t cp;
        size_t used;
        if (!utf8_one((const unsigned char *)v->encoded + at,
                      v->encoded_length - at, &cp, &used))
          return ria_fail(e, RIA_INTEGRITY_ERROR, "invalid vocabulary Unicode");
        unsigned b = byte_for_codepoint(cp);
        if (b > 255)
          return ria_fail(e, RIA_INTEGRITY_ERROR,
                          "vocabulary glyph outside byte alphabet");
        v->decoded[v->decoded_length++] = (char)b;
        at += used;
      }
    v->decoded[v->decoded_length] = 0;
  }
  for (unsigned b = 0; b < 256; b++) {
    char text[4];
    size_t length = glyph(codepoint_for_byte(b), text);
    uint32_t id = lookup(t, text, length);
    if (id == UINT32_MAX || t->vocab[id].special)
      return ria_fail(e, RIA_INTEGRITY_ERROR, "BPE byte alphabet incomplete");
    t->bytes[b] = id;
  }
  const ria_json_node *merges =
      ria_json_at(d, ria_json_get(d, model, "merges"));
  if (!merges || merges->type != RIA_JSON_ARRAY)
    return ria_fail(e, RIA_INVALID_REQUEST, "BPE merge ranks missing");
  uint32_t rank = 0;
  for (uint32_t i = merges->child; i != RIA_JSON_NONE; i = d->nodes[i].next) {
    const char *s = NULL;
    size_t length = 0;
    if (!ria_json_string(d, i, &s, &length, e) || !length || length > 65536 ||
        rank >= 127741)
      return ria_fail(e, RIA_INTEGRITY_ERROR, "invalid BPE merge rank");
    const char *space = memchr(s, ' ', length);
    if (!space)
      return ria_fail(e, RIA_INTEGRITY_ERROR, "BPE merge separator missing");
    size_t left = (size_t)(space - s), right = length - left - 1;
    uint32_t a = lookup(t, s, left), b = lookup(t, space + 1, right);
    char *joined = malloc(length);
    if (!joined)
      return ria_fail(e, RIA_RESOURCE_LIMIT,
                      "merge validation allocation failed");
    memcpy(joined, s, left);
    memcpy(joined + left, space + 1, right);
    uint32_t out = lookup(t, joined, left + right);
    free(joined);
    if (a == UINT32_MAX || b == UINT32_MAX || out == UINT32_MAX ||
        t->vocab[a].special || t->vocab[b].special || t->vocab[out].special)
      return ria_fail(e, RIA_INTEGRITY_ERROR,
                      "BPE merge references unknown symbol");
    uint64_t key = (uint64_t)a << 32 | b;
    uint32_t at = (uint32_t)(key * UINT64_C(11400714819323198485) >> (64 - 18));
    while (t->merges[at].rank) {
      if (t->merges[at].key == key)
        return ria_fail(e, RIA_INTEGRITY_ERROR, "duplicate BPE merge pair");
      at = (at + 1) & (MERGE_TABLE - 1);
    }
    t->merges[at] = (merge){key, rank + 1, out};
    rank++;
  }
  if (rank != 127741)
    return ria_fail(e, RIA_INTEGRITY_ERROR, "incomplete BPE merge ranks");
  return true;
}
bool ria_tokenizer_open(const char *path, const uint8_t expected[32],
                        uint64_t budget, ria_tokenizer **out, ria_error *e) {
  if (out)
    *out = NULL;
  if (!path || !expected || !out || budget < UINT64_C(134217728))
    return ria_fail(e, RIA_INVALID_REQUEST, "invalid tokenizer load contract");
  int fd = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
  if (fd < 0)
    return ria_fail(e, RIA_INTEGRITY_ERROR,
                    "cannot read provisioned tokenizer");
  const size_t cap = 16777216;
  struct stat before, after;
  if (fstat(fd, &before) || !S_ISREG(before.st_mode) || before.st_size <= 0 ||
      (uint64_t)before.st_size > cap) {
    close(fd);
    return ria_fail(e, RIA_INTEGRITY_ERROR,
                    "tokenizer must be a nonempty regular file of at most 16MiB");
  }
  size_t n = (size_t)before.st_size;
  char *bytes = malloc(n + 1);
  if (!bytes) {
    close(fd);
    return ria_fail(e, RIA_RESOURCE_LIMIT, "tokenizer read allocation failed");
  }
  bool ok = true;
  size_t got = 0;
  while (got < n) {
    ssize_t count = read(fd, bytes + got, n - got);
    if (count < 0 && errno == EINTR)
      continue;
    if (count <= 0) {
      ok = false;
      break;
    }
    got += (size_t)count;
  }
  char extra;
  ssize_t count = -1;
  if (ok) {
    do count = read(fd, &extra, 1); while (count < 0 && errno == EINTR);
    ok = count == 0 && fstat(fd, &after) == 0 &&
         before.st_dev == after.st_dev && before.st_ino == after.st_ino &&
         before.st_size == after.st_size &&
         before.st_mtim.tv_sec == after.st_mtim.tv_sec &&
         before.st_mtim.tv_nsec == after.st_mtim.tv_nsec &&
         before.st_ctim.tv_sec == after.st_ctim.tv_sec &&
         before.st_ctim.tv_nsec == after.st_ctim.tv_nsec;
  }
  if (close(fd))
    ok = false;
  uint8_t actual[32];
  if (ok)
    ok = ria_sha256(bytes, n, actual, e) && !memcmp(actual, expected, 32);
  if (!ok) {
    free(bytes);
    return ria_fail(e, RIA_INTEGRITY_ERROR,
                    "tokenizer source SHA256 mismatch or unreadable input");
  }
  ria_tokenizer *t = calloc(1, sizeof *t);
  if (!t) {
    free(bytes);
    return ria_fail(e, RIA_RESOURCE_LIMIT, "tokenizer owner allocation failed");
  }
  t->budget = budget;
  t->source_bytes = n + 1;
  ria_json_limits bounds = {cap, 600000, 32};
  ok = ria_json_parse(bytes, n, bounds, &t->source, e);
  free(bytes);
  if (!ok) {
    ria_tokenizer_close(t);
    return false;
  }
  t->vocab = calloc(TOKEN_VOCAB, sizeof *t->vocab);
  t->index = calloc(VOCAB_TABLE, sizeof *t->index);
  t->merges = calloc(MERGE_TABLE, sizeof *t->merges);
  t->owned = sizeof *t + (uint64_t)TOKEN_VOCAB * sizeof *t->vocab +
             (uint64_t)VOCAB_TABLE * sizeof *t->index +
             (uint64_t)MERGE_TABLE * sizeof *t->merges + n + 1;
  /* Account the full parser capacity and read buffer at peak before retained
     nodes are released. The fixed tokenizer source is bounded at 16MiB. */
  uint64_t parse_peak = (uint64_t)600000 * sizeof(ria_json_node) + cap + 1;
  if (t->owned > budget || parse_peak > budget - t->owned)
    ok = ria_fail(e, RIA_RESOURCE_LIMIT,
                  "tokenizer startup peak exceeds metadata budget");
  if (ok && (!t->vocab || !t->index || !t->merges))
    ok = ria_fail(e, RIA_RESOURCE_LIMIT, "tokenizer table allocation failed");
  if (ok)
    ok = metadata(t, e);
  if (!ok) {
    ria_tokenizer_close(t);
    return false;
  }
  free(t->source.nodes);
  t->source.nodes = NULL;
  t->source.count = 0;
  *out = t;
  return true;
}
void ria_tokenizer_close(ria_tokenizer *t) {
  if (!t)
    return;
  for (unsigned i = 0; i < 3; i++)
    if (t->regex[i])
      onig_free(t->regex[i]);
  if (t->vocab)
    for (uint32_t i = 0; i < TOKEN_VOCAB; i++)
      free(t->vocab[i].decoded);
  free(t->vocab);
  free(t->index);
  free(t->merges);
  free(t->specials);
  ria_json_free(&t->source);
  free(t);
}
uint64_t ria_tokenizer_bytes(const ria_tokenizer *t) {
  return t ? t->owned : 0;
}
uint32_t ria_tokenizer_vocab(const ria_tokenizer *t) {
  return t ? TOKEN_VOCAB : 0;
}
bool ria_tokenizer_special(const ria_tokenizer *t, const char *text,
                           uint32_t *id, ria_error *e) {
  if (!t || !text || !id)
    return ria_fail(e, RIA_INVALID_REQUEST, "invalid special-token query");
  uint32_t value = lookup(t, text, strlen(text));
  if (value == UINT32_MAX || !t->vocab[value].special)
    return ria_fail(e, RIA_INTEGRITY_ERROR, "required special token missing");
  *id = value;
  return true;
}
typedef struct {
  uint32_t id, next, previous, version;
  bool live;
} symbol;
typedef struct {
  uint32_t rank, left, right, left_version, right_version, output;
} edge;
typedef struct {
  ria_tokenizer *t;
  uint32_t *out;
  size_t count, maximum;
  OnigRegion *region;
  OnigMatchParam *parameter;
  ria_error *error;
  uint64_t scratch;
} encoding;
static const merge *pair(const ria_tokenizer *t, uint32_t left,
                         uint32_t right) {
  uint64_t key = (uint64_t)left << 32 | right;
  uint32_t at = (uint32_t)(key * UINT64_C(11400714819323198485) >> (64 - 18));
  for (uint32_t i = 0; i < MERGE_TABLE; i++) {
    const merge *m = &t->merges[at];
    if (!m->rank)
      return NULL;
    if (m->key == key)
      return m;
    at = (at + 1) & (MERGE_TABLE - 1);
  }
  return NULL;
}
static bool earlier(edge a, edge b) {
  return a.rank < b.rank || (a.rank == b.rank && a.left < b.left);
}
static void heap_push(edge *heap, size_t *count, edge value) {
  size_t at = (*count)++;
  while (at) {
    size_t parent = (at - 1) / 2;
    if (!earlier(value, heap[parent]))
      break;
    heap[at] = heap[parent];
    at = parent;
  }
  heap[at] = value;
}
static edge heap_pop(edge *heap, size_t *count) {
  edge result = heap[0], last = heap[--*count];
  size_t at = 0;
  while (at * 2 + 1 < *count) {
    size_t child = at * 2 + 1;
    if (child + 1 < *count && earlier(heap[child + 1], heap[child]))
      child++;
    if (!earlier(heap[child], last))
      break;
    heap[at] = heap[child];
    at = child;
  }
  if (*count)
    heap[at] = last;
  return result;
}
static void candidate(encoding *e, symbol *nodes, edge *heap, size_t *count,
                      uint32_t left) {
  uint32_t right = nodes[left].next;
  if (right == UINT32_MAX)
    return;
  const merge *m = pair(e->t, nodes[left].id, nodes[right].id);
  if (m)
    heap_push(heap, count,
              (edge){m->rank, left, right, nodes[left].version,
                     nodes[right].version, m->output});
}
static bool output_token(encoding *e, uint32_t token_id) {
  if (e->count == e->maximum)
    return ria_fail(e->error, RIA_RESOURCE_LIMIT,
                    "tokenized prompt exceeds admitted context");
  e->out[e->count++] = token_id;
  return true;
}
static bool bpe(encoding *e, const char *s, size_t n) {
  if (!n)
    return true;
  if (n > UINT32_MAX)
    return ria_fail(e->error, RIA_RESOURCE_LIMIT,
                    "BPE piece exceeds index range");
  uint64_t node_bytes, heap_bytes, total;
  if (!ria_u64_mul(n, sizeof(symbol), &node_bytes) ||
      !ria_u64_mul(n, 3 * sizeof(edge), &heap_bytes) ||
      !ria_u64_add(node_bytes, heap_bytes, &total) || total > e->scratch)
    return ria_fail(e->error, RIA_RESOURCE_LIMIT,
                    "BPE work exceeds admitted tokenizer scratch");
  symbol *nodes = malloc((size_t)node_bytes);
  edge *heap = malloc((size_t)heap_bytes);
  if (!nodes || !heap) {
    free(nodes);
    free(heap);
    return ria_fail(e->error, RIA_RESOURCE_LIMIT,
                    "BPE scratch allocation failed");
  }
  size_t edges = 0;
  for (uint32_t i = 0; i < n; i++)
    nodes[i] = (symbol){e->t->bytes[(unsigned char)s[i]],
                        i + 1 < n ? i + 1 : UINT32_MAX, i ? i - 1 : UINT32_MAX,
                        0, true};
  for (uint32_t i = 0; i < n; i++)
    candidate(e, nodes, heap, &edges, i);
  while (edges) {
    edge selected = heap_pop(heap, &edges);
    symbol *left = &nodes[selected.left], *right = &nodes[selected.right];
    if (!left->live || !right->live || left->next != selected.right ||
        left->version != selected.left_version ||
        right->version != selected.right_version)
      continue;
    left->id = selected.output;
    left->version++;
    left->next = right->next;
    right->live = false;
    right->version++;
    if (right->next != UINT32_MAX)
      nodes[right->next].previous = selected.left;
    if (left->previous != UINT32_MAX)
      candidate(e, nodes, heap, &edges, left->previous);
    candidate(e, nodes, heap, &edges, selected.left);
  }
  bool ok = true;
  for (uint32_t i = 0; i != UINT32_MAX; i = nodes[i].next)
    if (!output_token(e, nodes[i].id)) {
      ok = false;
      break;
    }
  free(nodes);
  free(heap);
  return ok;
}
static bool split(encoding *e, const char *text, size_t n, unsigned level) {
  if (!n)
    return true;
  if (level == 3)
    return bpe(e, text, n);
  size_t cursor = 0;
  while (cursor < n) {
    int result = onig_search_with_param(
        e->t->regex[level], (const OnigUChar *)text,
        (const OnigUChar *)text + n, (const OnigUChar *)text + cursor,
        (const OnigUChar *)text + n, e->region,
        ONIG_OPTION_CHECK_VALIDITY_OF_STRING, e->parameter);
    if (result == ONIG_MISMATCH)
      return split(e, text + cursor, n - cursor, level + 1);
    if (result < 0 || e->region->beg[0] < 0 ||
        e->region->end[0] <= e->region->beg[0])
      return ria_fail(e->error, RIA_INVALID_REQUEST,
                      "tokenizer Unicode split failed or exhausted its bound");
    size_t begin = (size_t)e->region->beg[0], end = (size_t)e->region->end[0];
    if (begin < cursor || end > n)
      return ria_fail(
          e->error, RIA_INTERNAL_ERROR,
          "tokenizer regex returned invalid range"); /* Recursion reuses the
                                                        region only after range
                                                        values are copied. */
    if (begin > cursor && !split(e, text + cursor, begin - cursor, level + 1))
      return false;
    if (!split(e, text + begin, end - begin, level + 1))
      return false;
    cursor = end;
  }
  return true;
}
bool ria_tokenizer_encode(ria_tokenizer *t, const char *text, size_t n,
                          bool recognize, uint64_t maximum, uint32_t **out,
                          size_t *count, ria_error *e) {
  if (!t || !text || !out || !count || n > 67108864 || n > INT_MAX ||
      !maximum || maximum > 1048576)
    return ria_fail(e, RIA_INVALID_REQUEST, "invalid tokenizer input bounds");
  *out = NULL;
  *count = 0;
  for (size_t at = 0; at < n;) {
    uint32_t cp;
    size_t used;
    if (!utf8_one((const unsigned char *)text + at, n - at, &cp, &used))
      return ria_fail(e, RIA_INVALID_REQUEST,
                      "tokenizer input is not valid UTF8");
    at += used;
  }
  uint64_t output_bytes = maximum * sizeof(uint32_t),
           overhead = UINT64_C(4194304);
  if (t->owned > t->budget || output_bytes > t->budget - t->owned ||
      overhead > t->budget - t->owned - output_bytes)
    return ria_fail(e, RIA_RESOURCE_LIMIT,
                    "tokenizer output/regex scratch exceeds budget");
  encoding ctx = {t,
                  malloc((size_t)output_bytes),
                  0,
                  (size_t)maximum,
                  onig_region_new(),
                  onig_new_match_param(),
                  e,
                  t->budget - t->owned - output_bytes - overhead};
  if (!ctx.out || !ctx.region || !ctx.parameter) {
    free(ctx.out);
    if (ctx.region)
      onig_region_free(ctx.region, 1);
    if (ctx.parameter)
      onig_free_match_param(ctx.parameter);
    return ria_fail(e, RIA_RESOURCE_LIMIT,
                    "tokenizer output allocation failed");
  }
  bool ok = onig_set_match_stack_limit_size_of_match_param(ctx.parameter,
                                                           1048576) == 0 &&
            onig_set_retry_limit_in_match_of_match_param(ctx.parameter,
                                                         10000000) == 0 &&
            onig_set_retry_limit_in_search_of_match_param(ctx.parameter,
                                                          10000000) == 0;
  size_t start = 0, at = 0;
  while (ok && at < n) {
    uint32_t selected = UINT32_MAX;
    size_t length = 0;
    {
      uint32_t node = 0;
      for (size_t j = at; j < n; j++) {
        uint32_t child = t->specials[node].child;
        while (child && t->specials[child].byte != (unsigned char)text[j])
          child = t->specials[child].sibling;
        if (!child)
          break;
        node = child;
        if (t->specials[node].token_id &&
            (recognize || !t->vocab[t->specials[node].token_id - 1].special)) {
          selected = t->specials[node].token_id - 1;
          length = j - at + 1;
        }
      }
    }
    if (selected != UINT32_MAX) {
      ok = split(&ctx, text + start, at - start, 0) &&
           output_token(&ctx, selected);
      at += length;
      start = at;
    } else
      at++;
  }
  if (ok)
    ok = split(&ctx, text + start, n - start, 0);
  onig_region_free(ctx.region, 1);
  onig_free_match_param(ctx.parameter);
  if (!ok) {
    free(ctx.out);
    return false;
  }
  *out = ctx.out;
  *count = ctx.count;
  return true;
}
bool ria_tokenizer_decode(const ria_tokenizer *t, uint32_t id, char **out,
                          size_t *length, ria_error *e) {
  if (!t || id >= TOKEN_VOCAB || !out || !length)
    return ria_fail(e, RIA_INVALID_REQUEST, "invalid decoded token identity");
  const token *v = &t->vocab[id];
  char *s = malloc(v->decoded_length + 1);
  if (!s)
    return ria_fail(e, RIA_RESOURCE_LIMIT, "decoded token allocation failed");
  memcpy(s, v->decoded, v->decoded_length + 1);
  *out = s;
  *length = v->decoded_length;
  return true;
}

uint64_t ria_tokenizer_max_token_bytes(const ria_tokenizer *t) {
  uint64_t max = 0;
  if (t)
    for (unsigned i = 0; i < TOKEN_VOCAB; i++)
      if (t->vocab[i].decoded_length > max)
        max = t->vocab[i].decoded_length;
  return max;
}
