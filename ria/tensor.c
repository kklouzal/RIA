#define _GNU_SOURCE
#include "tensor.h"
#include "protocol.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <numa.h>
#include <numaif.h>
#include <openssl/evp.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static const ria_json_limits manifest_limits = {256u << 10, 100000, 64};
static bool number(const ria_json_doc *d, uint32_t p, const char *k, bool wide,
                   uint64_t *v, ria_error *e) {
  return ria_json_u64(d, ria_json_get(d, p, k), wide, v, e);
}
static bool string(const ria_json_doc *d, uint32_t p, const char *k,
                   const char **v, ria_error *e) {
  size_t n;
  return ria_json_string(d, ria_json_get(d, p, k), v, &n, e) &&
         (strlen(*v) == n ||
          ria_fail(e, RIA_INVALID_REQUEST, "embedded NUL in %s", k));
}
static bool hash(const ria_json_doc *d, uint32_t p, const char *k,
                 uint8_t out[32], ria_error *e) {
  return ria_json_digest_field(d, ria_json_get(d, p, k), out, e);
}
static bool identity(const ria_json_doc *d, const uint8_t *expected,
                     uint8_t actual[32], ria_error *e) {
  uint8_t claimed[32];
  return hash(d, 0, "digest", claimed, e) &&
         ria_json_sha256(d, true, actual, e) &&
         ((!memcmp(actual, claimed, 32) &&
           (!expected || !memcmp(actual, expected, 32))) ||
          ria_fail(e, RIA_INTEGRITY_ERROR,
                   "manifest identity does not match trusted root"));
}
static bool float_bits(const ria_json_doc *d, uint32_t p, const char *k,
                       float *v, ria_error *e) {
  uint32_t i = ria_json_get(d, p, k);
  if (i == RIA_JSON_NONE) {
    *v = 1;
    return true;
  }
  const char *s;
  size_t n;
  uint8_t bytes[4];
  if (!ria_json_string(d, i, &s, &n, e) || !ria_hex_decode(s, n, bytes, 4, e))
    return false;
  uint32_t bits = ((uint32_t)bytes[0] << 24) | ((uint32_t)bytes[1] << 16) |
                  ((uint32_t)bytes[2] << 8) | bytes[3];
  memcpy(v, &bits, 4);
  return (isfinite(*v) && *v > 0) ||
         ria_fail(e, RIA_INVALID_REQUEST, "invalid positive float bits: %s", k);
}
static bool array(const ria_json_doc *d, uint32_t p, const char *k,
                  uint64_t *out, uint32_t *count, uint32_t maximum, bool wide,
                  ria_error *e) {
  const ria_json_node *a = ria_json_at(d, ria_json_get(d, p, k));
  if (!a || a->type != RIA_JSON_ARRAY)
    return ria_fail(e, RIA_INVALID_REQUEST, "missing array %s", k);
  *count = 0;
  for (uint32_t i = a->child; i != RIA_JSON_NONE; i = d->nodes[i].next) {
    if (*count == maximum)
      return ria_fail(e, RIA_RESOURCE_LIMIT, "too many %s entries", k);
    if (!ria_json_u64(d, i, wide, &out[*count], e))
      return false;
    ++*count;
  }
  return true;
}
static int compare_tensor(const void *a, const void *b) {
  uint64_t x = ((const ria_tensor *)a)->id, y = ((const ria_tensor *)b)->id;
  return (x > y) - (x < y);
}
static int compare_shard(const void *a, const void *b) {
  uint64_t x = ((const ria_shard *)a)->id, y = ((const ria_shard *)b)->id;
  return (x > y) - (x < y);
}
static int compare_range(const void *a, const void *b) {
  uint64_t x = ((const uint64_t *)a)[0], y = ((const uint64_t *)b)[0];
  return (x > y) - (x < y);
}
static int compare_name(const void *a, const void *b) {
  return strcmp((*(const ria_tensor *const *)a)->name,
                (*(const ria_tensor *const *)b)->name);
}
const ria_tensor *ria_tensor_id(const ria_tensor_store *s, uint64_t id) {
  uint64_t lo = 0, hi = s->tensor_count;
  while (lo < hi) {
    uint64_t m = lo + (hi - lo) / 2;
    if (s->tensors[m].id < id)
      lo = m + 1;
    else
      hi = m;
  }
  return lo < s->tensor_count && s->tensors[lo].id == id ? &s->tensors[lo]
                                                         : NULL;
}
const ria_tensor *ria_tensor_name(const ria_tensor_store *s, const char *name) {
  uint64_t lo = 0, hi = s->tensor_count;
  while (lo < hi) {
    uint64_t m = lo + (hi - lo) / 2;
    if (strcmp(s->names[m]->name, name) < 0)
      lo = m + 1;
    else
      hi = m;
  }
  return lo < s->tensor_count && !strcmp(s->names[lo]->name, name)
             ? s->names[lo]
             : NULL;
}
const ria_shard *ria_shard_id(const ria_tensor_store *s, uint64_t id) {
  uint64_t lo = 0, hi = s->shard_count;
  while (lo < hi) {
    uint64_t m = lo + (hi - lo) / 2;
    if (s->shards[m].id < id)
      lo = m + 1;
    else
      hi = m;
  }
  return lo < s->shard_count && s->shards[lo].id == id ? &s->shards[lo] : NULL;
}
static bool shape_product(const uint64_t *shape, uint32_t n, uint64_t *product,
                          ria_error *e) {
  *product = 1;
  for (uint32_t i = 0; i < n; i++)
    if (!shape[i] || !ria_u64_mul(*product, shape[i], product))
      return ria_fail(e, RIA_INVALID_REQUEST,
                      "zero or overflowing tensor dimensions");
  return true;
}
static bool dtype_size(const char *dtype, uint64_t elements, uint64_t *bytes,
                       ria_error *e) {
  uint64_t width;
  if (!strcmp(dtype, "F32") || !strcmp(dtype, "I32") || !strcmp(dtype, "U32"))
    width = 4;
  else if (!strcmp(dtype, "BF16") || !strcmp(dtype, "F16") ||
           !strcmp(dtype, "I16") || !strcmp(dtype, "U16"))
    width = 2;
  else if (!strcmp(dtype, "F64") || !strcmp(dtype, "I64") ||
           !strcmp(dtype, "U64"))
    width = 8;
  else if (!strcmp(dtype, "U8") || !strcmp(dtype, "I8") ||
           !strcmp(dtype, "BOOL") || !strcmp(dtype, "F8_E4M3") ||
           !strcmp(dtype, "F8_E4M3FN") || !strcmp(dtype, "F8_E8M0"))
    width = 1;
  else
    return ria_fail(e, RIA_UNSUPPORTED, "unsupported safetensors dtype %s",
                    dtype);
  return ria_u64_mul(elements, width, bytes) ||
         ria_fail(e, RIA_INVALID_REQUEST, "tensor byte overflow");
}
/* Every path component is opened below the provisioned artifact directory.
 * Symlinks and empty/dot components are rejected without pathname races. */
static int open_below(int root, const char *path, ria_error *e) {
  if (!path || !*path || path[0] == '/' || strchr(path, '\\')) {
    ria_error_set(e, RIA_INVALID_REQUEST, "invalid shard path");
    return -1;
  }
  char *copy = strdup(path);
  int dir = dup(root), result = -1;
  if (!copy || dir < 0) {
    free(copy);
    if (dir >= 0)
      close(dir);
    ria_error_set(e, RIA_RESOURCE_LIMIT, "path allocation failed");
    return -1;
  }
  char *part = copy;
  for (;;) {
    char *slash = strchr(part, '/');
    if (slash)
      *slash = 0;
    if (!*part || !strcmp(part, ".") || !strcmp(part, "..")) {
      ria_error_set(e, RIA_INVALID_REQUEST, "invalid shard path component");
      break;
    }
    int next =
        openat(dir, part,
               O_RDONLY | O_CLOEXEC | O_NOFOLLOW | (slash ? O_DIRECTORY : 0));
    if (next < 0) {
      ria_error_set(e, RIA_INTEGRITY_ERROR,
                    "cannot open artifact component: %s", strerror(errno));
      break;
    }
    close(dir);
    dir = -1;
    if (!slash) {
      result = next;
      break;
    }
    dir = next;
    part = slash + 1;
  }
  if (dir >= 0)
    close(dir);
  free(copy);
  return result;
}
static bool metadata_peak(uint64_t retained,uint64_t input,ria_json_limits limits,uint64_t budget,ria_error *e) {
  uint64_t nodes=input/2+1,bytes;
  if (nodes>limits.max_nodes) nodes=limits.max_nodes;
  if (!ria_u64_mul(nodes,sizeof(ria_json_node),&bytes) || !ria_u64_add(bytes,input+1,&bytes) ||
      !ria_u64_add(bytes,input,&bytes) || !ria_u64_add(bytes,retained,&bytes) || bytes>budget)
    return ria_fail(e,RIA_RESOURCE_LIMIT,"metadata read/DOM peak exceeds explicit startup reservation");
  return true;
}
static bool read_document(int fd,ria_json_doc *d,ria_json_limits limits,uint64_t retained,uint64_t budget,ria_error *e) {
  struct stat st;
  if (fstat(fd, &st) || !S_ISREG(st.st_mode) || st.st_size <= 0 ||
      (uint64_t)st.st_size > limits.max_bytes) {
    close(fd);
    return ria_fail(e, RIA_RESOURCE_LIMIT,
                    "invalid tensor index page file/size");
  }
  size_t length = (size_t)st.st_size, got = 0;
  if (!metadata_peak(retained,length,limits,budget,e)) { close(fd); return false; }
  uint8_t *bytes = malloc(length);
  if (!bytes) {
    close(fd);
    return ria_fail(e, RIA_RESOURCE_LIMIT, "index page read allocation failed");
  }
  bool ok = true;
  while (got < length) {
    ssize_t n = read(fd, bytes + got, length - got);
    if (n < 0 && errno == EINTR)
      continue;
    if (n <= 0) {
      ok = ria_fail(e, RIA_INTEGRITY_ERROR, "short index page read");
      break;
    }
    got += (size_t)n;
  }
  uint8_t extra;
  ssize_t tail;
  do tail=read(fd,&extra,1); while (tail<0 && errno==EINTR);
  if (tail!=0 && ok) ok=ria_fail(e,RIA_INTEGRITY_ERROR,"metadata file grew or trailing read failed");
  if (close(fd) && ok)
    ok = ria_fail(e, RIA_INTEGRITY_ERROR, "index page close failed");
  if (ok)
    ok = ria_json_parse(bytes, length,limits,d,e);
  free(bytes);
  return ok;
}
static bool read_page(int root,const char *path,ria_json_doc *d,uint64_t retained,uint64_t budget,ria_error *e) {
  int fd=open_below(root,path,e);
  return fd>=0 && read_document(fd,d,(ria_json_limits){16u<<20,1000000,64},retained,budget,e);
}
static bool indexes(ria_tensor_store *s, int root,uint64_t budget, ria_error *e) {
  ria_json_doc *m = &s->manifest;
  uint32_t array_id = ria_json_get(m, 0, "tensor_pages");
  const ria_json_node *refs = ria_json_at(m, array_id);
  if (refs && refs->type != RIA_JSON_ARRAY)
    return ria_fail(e, RIA_INVALID_REQUEST, "invalid tensor_pages array");
  for (uint32_t i = refs ? refs->child : RIA_JSON_NONE; i != RIA_JSON_NONE;
       i = m->nodes[i].next)
    if (++s->page_count > 4096)
      return ria_fail(e, RIA_RESOURCE_LIMIT, "too many tensor index pages");
  uint64_t retained=s->manifest.allocated_bytes+(uint64_t)(s->page_count ? s->page_count : 1)*sizeof(*s->pages);
  if (retained>budget) return ria_fail(e,RIA_RESOURCE_LIMIT,"metadata page owners exceed startup reservation");
  s->pages = calloc(s->page_count ? s->page_count : 1, sizeof(*s->pages));
  if (!s->pages)
    return ria_fail(e, RIA_RESOURCE_LIMIT,
                    "index page metadata allocation failed");
  uint64_t total = m->count;
  uint32_t page = 0;
  const char *const rf[] = {"path", "digest", "kind"};
  const char *const pf[] = {"schema_revision", "tensors", "shards", "digest"};
  for (uint32_t i = refs ? refs->child : RIA_JSON_NONE; i != RIA_JSON_NONE;
       i = m->nodes[i].next) {
    const char *path, *kind;
    uint8_t expected[32], actual[32];
    uint64_t revision;
    if (!ria_json_fields(m, i, rf, 3, rf, 3, e) ||
        !string(m, i, "path", &path, e) || !string(m, i, "kind", &kind, e) ||
        strcmp(kind, "index") || !hash(m, i, "digest", expected, e) ||
        !read_page(root, path, &s->pages[page],retained,budget,e))
      return false;
    ria_json_doc *d = &s->pages[page++];
    if (!ria_u64_add(retained,d->allocated_bytes,&retained) || retained>budget)
      return ria_fail(e,RIA_RESOURCE_LIMIT,"retained metadata pages exceed startup reservation");
    if (!ria_json_fields(d, 0, pf, 4, pf, 4, e) ||
        !identity(d, expected, actual, e) ||
        !number(d, 0, "schema_revision", false, &revision, e) || revision != 1)
      return false;
    if (!ria_u64_add(total, d->count, &total) || total > 16000000 ||
        total > SIZE_MAX / sizeof(ria_json_node))
      return ria_fail(e, RIA_RESOURCE_LIMIT,
                      "aggregate tensor index complexity exceeded");
  }
  s->index.count = (uint32_t)total;
  s->index.allocated_bytes=total*sizeof(*s->index.nodes);
  if (!ria_u64_add(retained,s->index.allocated_bytes,&retained) || retained>budget)
    return ria_fail(e,RIA_RESOURCE_LIMIT,"assembled metadata index exceeds startup reservation");
  s->index.nodes = malloc((size_t)total * sizeof(*s->index.nodes));
  if (!s->index.nodes)
    return ria_fail(e, RIA_RESOURCE_LIMIT, "tensor index allocation failed");
  memcpy(s->index.nodes, m->nodes, m->count * sizeof(*m->nodes));
  /* Retain page-owned strings; remap only integer DOM links. The immutable
   * root remains untouched so its externally provisioned identity is stable. */
  uint32_t base = m->count;
  const char *keys[] = {"tensors", "shards"};
  uint32_t targets[2], tails[2];
  for (unsigned k=0;k<2;k++) {
    targets[k]=ria_json_get(&s->index,0,keys[k]);
    const ria_json_node *array=ria_json_at(&s->index,targets[k]);
    if (!array || array->type!=RIA_JSON_ARRAY)
      return ria_fail(e,RIA_INVALID_REQUEST,"root requires tensor/shard arrays");
    tails[k]=array->child;
    while (tails[k]!=RIA_JSON_NONE && s->index.nodes[tails[k]].next!=RIA_JSON_NONE)
      tails[k]=s->index.nodes[tails[k]].next;
  }
  for (uint32_t j = 0; j < s->page_count; j++) {
    ria_json_doc *d = &s->pages[j];
    for (uint32_t k = 0; k < d->count; k++) {
      ria_json_node node = d->nodes[k];
      if (node.child != RIA_JSON_NONE)
        node.child += base;
      if (node.next != RIA_JSON_NONE)
        node.next += base;
      s->index.nodes[base + k] = node;
    }
    for (unsigned k = 0; k < 2; k++) {
      uint32_t source = ria_json_get(d, 0, keys[k]);
      const ria_json_node *src = ria_json_at(d, source);
      ria_json_node *dst = &s->index.nodes[targets[k]];
      if (!src || src->type != RIA_JSON_ARRAY || !dst ||
          dst->type != RIA_JSON_ARRAY)
        return ria_fail(e, RIA_INVALID_REQUEST,
                        "index page requires tensor/shard arrays");
      if (src->child == RIA_JSON_NONE)
        continue;
      if (dst->child == RIA_JSON_NONE)
        dst->child = base + src->child;
      else {
        s->index.nodes[tails[k]].next = base + src->child;
      }
      tails[k]=base+src->child;
      while (s->index.nodes[tails[k]].next!=RIA_JSON_NONE)
        tails[k]=s->index.nodes[tails[k]].next;
    }
    base += d->count;
  }
  return true;
}
static bool descriptors(ria_tensor_store *s, const ria_tensor_load_options *o,
                        ria_error *e) {
  ria_json_doc *d = &s->index;
  const char *const allowed[] = {"schema_revision",
                                 "role",
                                 "model_id",
                                 "source_revision",
                                 "profile",
                                 "logical_model_digest",
                                 "operator_contract_digest",
                                 "tokenizer_digest",
                                 "encoding_digest",
                                 "layout_digest",
                                 "tensors",
                                 "shards",
                                 "metadata",
                                 "feature_exclusions",
                                 "unexplained_required_tensors",
                                 "digest",
                                 "signatures",
                                 "tensor_pages"};
  const char *const required[] = {"schema_revision",
                                  "role",
                                  "model_id", "source_revision", "tokenizer_digest", "layout_digest", "metadata", "feature_exclusions", "tensor_pages",
                                  "profile",
                                  "logical_model_digest",
                                  "operator_contract_digest",
                                  "encoding_digest",
                                  "tensors",
                                  "shards",
                                  "digest",
                                  "unexplained_required_tensors"};
  if (!ria_json_fields(d, 0, allowed, sizeof(allowed) / sizeof(*allowed),
                       required, sizeof(required) / sizeof(*required), e))
    return false;
  uint64_t revision;
  const char *role, *profile, *model, *source;
  uint8_t extra[32];
  if (!number(d, 0, "schema_revision", false, &revision, e) || revision != 1 ||
      !string(d, 0, "role", &role, e) ||
      !string(d, 0, "profile", &profile, e) || strcmp(role, o->role) ||
      (strcmp(role, "client") && strcmp(role, "server")) ||
      (strcmp(profile, "bf16") && strcmp(profile, "fp8") &&
       strcmp(profile, "nvfp4")))
    return ria_fail(e, RIA_IDENTITY_MISMATCH,
                    "unsupported manifest revision/role/profile");
  if (!string(d,0,"model_id",&model,e) || strcmp(model,"deepseek-ai/DeepSeek-V4.1-Flash") ||
      !string(d,0,"source_revision",&source,e) || strlen(source)!=40 ||
      !ria_hex_decode(source,40,extra,20,e) || !hash(d,0,"tokenizer_digest",extra,e) || !hash(d,0,"layout_digest",extra,e))
    return ria_fail(e,RIA_INVALID_REQUEST,"required model/source/tokenizer/layout identity invalid");
  const char *const reference_fields[]={"path","digest","kind"};
  const ria_json_node *metadata=ria_json_at(d,ria_json_get(d,0,"metadata"));
  const ria_json_node *exclusions=ria_json_at(d,ria_json_get(d,0,"feature_exclusions"));
  if (!metadata || metadata->type!=RIA_JSON_ARRAY || !exclusions || exclusions->type!=RIA_JSON_ARRAY)
    return ria_fail(e,RIA_INVALID_REQUEST,"required provenance/exclusion arrays missing");
  unsigned references=0;
  for (uint32_t i=metadata->child;i!=RIA_JSON_NONE;i=d->nodes[i].next) {
    const char *path,*kind;
    if (++references>4096 || !ria_json_fields(d,i,reference_fields,3,reference_fields,3,e) ||
        !string(d,i,"path",&path,e) || !*path || path[0]=='/' || !string(d,i,"kind",&kind,e) ||
        (strcmp(kind,"operator") && strcmp(kind,"index") && strcmp(kind,"metadata") && strcmp(kind,"root") && strcmp(kind,"layout")) || !hash(d,i,"digest",extra,e))
      return ria_fail(e,RIA_INVALID_REQUEST,"invalid provenance reference");
  }
  references=0;
  for (uint32_t i=exclusions->child;i!=RIA_JSON_NONE;i=d->nodes[i].next) {
    const char *text; size_t length;
    if (++references>100 || !ria_json_string(d,i,&text,&length,e) || !length || strlen(text)!=length)
      return ria_fail(e,RIA_INVALID_REQUEST,"invalid feature exclusion");
  }
  memcpy(s->role, role, strlen(role) + 1);
  memcpy(s->profile, profile, strlen(profile) + 1);
  if (!hash(d, 0, "logical_model_digest", s->logical_model_digest, e) ||
      !hash(d, 0, "operator_contract_digest", s->operator_contract_digest, e) ||
      !hash(d, 0, "encoding_digest", s->encoding_digest, e))
    return false;
  const ria_json_node *unknown =
      ria_json_at(d, ria_json_get(d, 0, "unexplained_required_tensors"));
  if (!unknown || unknown->type != RIA_JSON_ARRAY ||
      unknown->child != RIA_JSON_NONE)
    return ria_fail(e, RIA_INTEGRITY_ERROR,
                    "manifest has unexplained required tensors");
  const ria_json_node *shards = ria_json_at(d, ria_json_get(d, 0, "shards"));
  const ria_json_node *tensors = ria_json_at(d, ria_json_get(d, 0, "tensors"));
  if (!shards || shards->type != RIA_JSON_ARRAY || !tensors ||
      tensors->type != RIA_JSON_ARRAY)
    return ria_fail(e, RIA_INVALID_REQUEST, "invalid shard/tensor arrays");
  for (uint32_t i = shards->child; i != RIA_JSON_NONE; i = d->nodes[i].next)
    if (++s->shard_count > 4096)
      return ria_fail(e, RIA_RESOURCE_LIMIT, "too many shards");
  for (uint32_t i = tensors->child; i != RIA_JSON_NONE; i = d->nodes[i].next)
    if (++s->tensor_count > 200000)
      return ria_fail(e, RIA_RESOURCE_LIMIT, "too many tensors");
  if (!s->shard_count || !s->tensor_count)
    return ria_fail(e, RIA_INVALID_REQUEST, "empty prepared payload");
  uint64_t retained=s->manifest.allocated_bytes+s->index.allocated_bytes+(uint64_t)(s->page_count ? s->page_count : 1)*sizeof(ria_json_doc);
  uint64_t extra_bytes;
  for (unsigned i=0;i<s->page_count;i++) if (!ria_u64_add(retained,s->pages[i].allocated_bytes,&retained))
    return ria_fail(e,RIA_RESOURCE_LIMIT,"metadata accounting overflow");
  if (!ria_u64_mul(s->tensor_count,sizeof(*s->tensors)+sizeof(*s->names),&extra_bytes) || !ria_u64_add(retained,extra_bytes,&retained) ||
      !ria_u64_mul(s->shard_count,sizeof(*s->shards),&extra_bytes) || !ria_u64_add(retained,extra_bytes,&retained) ||
      retained>(o->max_metadata_bytes ? o->max_metadata_bytes : o->max_resident_bytes))
    return ria_fail(e,RIA_RESOURCE_LIMIT,"descriptor/name owners exceed metadata startup reservation");
  s->shards = calloc((size_t)s->shard_count, sizeof(*s->shards));
  s->tensors = calloc((size_t)s->tensor_count, sizeof(*s->tensors));
  if (!s->shards || !s->tensors)
    return ria_fail(e, RIA_RESOURCE_LIMIT, "descriptor allocation failed");
  uint64_t n = 0;
  const char *const sf[] = {"id",         "path",         "length",    "sha256",
                            "chunk_size", "chunk_hashes", "data_start"};
  for (uint32_t i = shards->child; i != RIA_JSON_NONE; i = d->nodes[i].next) {
    ria_shard *a = &s->shards[n++];
    uint64_t chunk;
    if (!ria_json_fields(d, i, sf, 7, sf, 7, e) ||
        !number(d, i, "id", true, &a->id, e) ||
        !number(d, i, "length", true, &a->length, e) ||
        !number(d, i, "data_start", true, &a->data_start, e) ||
        !number(d, i, "chunk_size", false, &chunk, e) ||
        !string(d, i, "path", &a->path, e) ||
        !hash(d, i, "sha256", a->digest, e))
      return false;
    if (!chunk || chunk > RIA_BULK_MAX || a->length < 10 ||
        a->length > SIZE_MAX || a->data_start >= a->length)
      return ria_fail(e, RIA_INVALID_REQUEST, "invalid shard size/chunk size");
    if (!ria_u64_add(s->resident_bytes, a->length, &s->resident_bytes) ||
        s->resident_bytes > o->max_resident_bytes)
      return ria_fail(e, RIA_RESOURCE_LIMIT,
                      "prepared population exceeds admitted resident bytes");
    a->chunk_size = (uint32_t)chunk;
    a->chunk_count = a->length / chunk + (a->length % chunk != 0);
    const ria_json_node *h = ria_json_at(d, ria_json_get(d, i, "chunk_hashes"));
    if (!h || h->type != RIA_JSON_ARRAY || a->chunk_count > SIZE_MAX / 32)
      return ria_fail(e, RIA_INVALID_REQUEST, "invalid chunk hash array");
    if (!ria_u64_mul(a->chunk_count,32,&extra_bytes) || !ria_u64_add(retained,extra_bytes,&retained) ||
        retained>(o->max_metadata_bytes ? o->max_metadata_bytes : o->max_resident_bytes))
      return ria_fail(e,RIA_RESOURCE_LIMIT,"chunk index exceeds metadata startup reservation");
    a->chunk_hashes = malloc((size_t)a->chunk_count * 32);
    if (!a->chunk_hashes)
      return ria_fail(e, RIA_RESOURCE_LIMIT, "chunk index allocation failed");
    uint64_t j = 0;
    for (uint32_t k = h->child; k != RIA_JSON_NONE; k = d->nodes[k].next) {
      if (j == a->chunk_count ||
          !ria_json_digest_field(d, k, a->chunk_hashes + j * 32, e))
        return false;
      ++j;
    }
    if (j != a->chunk_count)
      return ria_fail(e, RIA_INTEGRITY_ERROR, "incomplete chunk index");
  }
  qsort(s->shards, (size_t)s->shard_count, sizeof(*s->shards), compare_shard);
  for (uint64_t i = 1; i < s->shard_count; i++)
    if (s->shards[i - 1].id == s->shards[i].id)
      return ria_fail(e, RIA_INVALID_REQUEST, "duplicate shard ID");
  n = 0;
  const char *const tf[] = {"id",
                            "name",
                            "logical_shape",
                            "physical_shape",
                            "dtype",
                            "format",
                            "layout",
                            "shard",
                            "offset",
                            "length",
                            "sha256",
                            "scale_ids",
                            "operation",
                            "placement",
                            "alias_of",
                            "byte_order",
                            "bytes_per_block",
                            "block_values",
                            "group_shape",
                            "value_row_stride",
                            "scale_row_stride",
                            "weight_global_scale_bits",
                            "activation_global_scale_bits",
                            "source_sha256",
                            "source_scale_sha256"};
  const char *const tr[] = {
      "id",     "name",      "logical_shape", "physical_shape", "dtype",
      "format", "layout",    "shard",         "offset",         "length",
      "sha256", "scale_ids", "placement", "operation", "alias_of", "source_sha256", "bytes_per_block", "block_values", "group_shape", "byte_order"};
  for (uint32_t i = tensors->child; i != RIA_JSON_NONE; i = d->nodes[i].next) {
    ria_tensor *t = &s->tensors[n++];
    const char *order;
    uint64_t product, bytes, bytes_per_block, block_values, group[8]; uint32_t group_rank;
    if (!ria_json_fields(d, i, tf, sizeof(tf) / sizeof(*tf), tr,
                         sizeof(tr) / sizeof(*tr), e) ||
        !number(d, i, "id", true, &t->id, e) ||
        !number(d, i, "shard", true, &t->shard, e) ||
        !number(d, i, "offset", true, &t->offset, e) ||
        !number(d, i, "length", true, &t->length, e) ||
        !string(d, i, "name", &t->name, e) ||
        !string(d, i, "dtype", &t->dtype, e) ||
        !string(d, i, "format", &t->format, e) ||
        !string(d, i, "layout", &t->layout, e) ||
        !string(d, i, "placement", &t->placement, e) ||
        !string(d, i, "operation", &t->operation, e) ||
        !hash(d,i,"sha256",extra,e) || !hash(d,i,"source_sha256",extra,e) ||
        !string(d, i, "byte_order", &order, e) || strcmp(order, "little") ||
        !array(d, i, "logical_shape", t->shape, &t->rank, 8, false, e) ||
        !array(d, i, "physical_shape", t->physical_shape, &t->physical_rank, 8, false, e) ||
        !shape_product(t->physical_shape, t->physical_rank, &product, e) ||
        !dtype_size(t->dtype, product, &bytes, e) || bytes != t->length ||
        !number(d,i,"bytes_per_block",false,&bytes_per_block,e) || !number(d,i,"block_values",false,&block_values,e) ||
        !array(d,i,"group_shape",group,&group_rank,8,false,e) ||
        !array(d, i, "scale_ids", t->scale_ids, &t->scale_count, 4, true, e) ||
        !float_bits(d, i, "weight_global_scale_bits", &t->weight_global_scale,
                    e) ||
        !float_bits(d, i, "activation_global_scale_bits",
                    &t->activation_global_scale, e))
      return ria_fail(e, RIA_INVALID_REQUEST, "invalid tensor descriptor");
    static const char *const operations[]={"embedding","attention","attention_state","index","router","norm","mhc","expert_gate","expert_up","expert_down","shared_gate","shared_up","shared_down","engram","vision","output","scale","inactive"};
    bool known=false;
    for (unsigned j=0;j<sizeof(operations)/sizeof(*operations);j++) if (!strcmp(t->operation,operations[j])) known=true;
    if (!known || (strcmp(t->placement,"client") && strcmp(t->placement,"server") && strcmp(t->placement,"both") && strcmp(t->placement,"cache") && strcmp(t->placement,"inactive")) ||
        (strcmp(t->layout,"row_major_le") && strcmp(t->layout,"opaque_source")))
      return ria_fail(e,RIA_INVALID_REQUEST,"unknown tensor operation/placement/layout");
    bool quantized=!strcmp(t->format,"nvfp4") || !strcmp(t->format,"fp8_block32") || !strcmp(t->format,"source_mxfp4");
    if (quantized) {
      bool packed=strcmp(t->format,"fp8_block32")!=0; uint64_t k=t->rank==2 ? t->shape[1] : 0;
      uint64_t group_width=!strcmp(t->format,"nvfp4") ? 16 : 32;
      if (t->rank!=2 || t->physical_rank!=2 || t->physical_shape[0]!=t->shape[0] ||
          (packed && k%2) || t->physical_shape[1]!=(packed ? k/2 : k) ||
          strcmp(t->dtype,packed ? "U8" : "F8_E4M3") || bytes_per_block!=1 || block_values!=(packed ? 2u : 1u) ||
          group_rank!=2 || group[0]!=(packed ? 1u : 32u) || group[1]!=group_width || t->scale_count!=1)
        return ria_fail(e,RIA_INVALID_REQUEST,"quantized tensor packing/group shape invalid");
      if (!strcmp(t->format,"nvfp4") && (ria_json_get(d,i,"weight_global_scale_bits")==RIA_JSON_NONE || ria_json_get(d,i,"activation_global_scale_bits")==RIA_JSON_NONE))
        return ria_fail(e,RIA_INVALID_REQUEST,"NVFP4 requires explicit calibrated global multipliers");
      if (!number(d,i,"scale_row_stride",true,&t->scale_row_stride,e) || t->scale_row_stride!=(k+group_width-1)/group_width)
        return ria_fail(e,RIA_INVALID_REQUEST,"quantized scale row stride invalid");
      if (!strncmp(t->operation,"expert_",7) &&
          strcmp(t->format,!strcmp(profile,"fp8") ? "fp8_block32" : profile))
        return ria_fail(e,RIA_IDENTITY_MISMATCH,"expert profile disagrees with model profile");
    } else if (!strcmp(t->format,"engram_packed")) {
      if (t->rank!=2 || t->physical_rank!=2 || t->shape[1]!=256 || t->physical_shape[0]!=t->shape[0] || t->physical_shape[1]!=264 || strcmp(t->dtype,"U8") || t->scale_count || bytes_per_block!=1 || block_values!=1 || group_rank ||
          !hash(d,i,"source_scale_sha256",extra,e)) return ria_fail(e,RIA_INVALID_REQUEST,"Engram row264 provenance/shape invalid");
    } else if (!strcmp(t->format,"plain") || !strcmp(t->format,"bf16")) {
      uint64_t width;
      if (t->rank!=t->physical_rank || memcmp(t->shape,t->physical_shape,t->rank*sizeof(*t->shape)) || group_rank || block_values!=1 ||
          !dtype_size(t->dtype,1,&width,e) || bytes_per_block!=width || (!strcmp(t->format,"bf16") && strcmp(t->dtype,"BF16")))
        return ria_fail(e,RIA_INVALID_REQUEST,"plain/BF16 shape/block/type invalid");
    } else return ria_fail(e,RIA_UNSUPPORTED,"unsupported prepared tensor format");
    if (!strncmp(t->operation,"expert_",7) &&
        (strcmp(t->format,!strcmp(profile,"fp8") ? "fp8_block32" : profile) && !( !strcmp(profile,"bf16") && !strcmp(t->format,"plain") && !strcmp(t->dtype,"BF16"))))
      return ria_fail(e,RIA_IDENTITY_MISMATCH,"routed expert precision disagrees with declared profile");
    if (t->physical_rank==2 && ria_json_get(d,i,"value_row_stride")!=RIA_JSON_NONE) {
      uint64_t width,stride;
      if (!dtype_size(t->dtype,1,&width,e) || !ria_u64_mul(t->physical_shape[1],width,&stride) || !number(d,i,"value_row_stride",true,&t->value_row_stride,e) || t->value_row_stride!=stride)
        return ria_fail(e,RIA_INVALID_REQUEST,"row-major value stride invalid");
    }
    uint64_t logical;
    if (!shape_product(t->shape, t->rank, &logical, e))
      return false;
    const ria_shard *a = ria_shard_id(s, t->shard);
    if (!a || t->offset > a->length - a->data_start ||
        t->length > a->length - a->data_start - t->offset)
      return ria_fail(e, RIA_INVALID_REQUEST, "tensor exceeds shard data area");
    uint32_t alias = ria_json_get(d, i, "alias_of");
    if (alias != RIA_JSON_NONE && d->nodes[alias].type != RIA_JSON_NULL) {
      t->alias = true;
      if (!ria_json_u64(d, alias, true, &t->alias_of, e))
        return false;
    }
    uint32_t stride = ria_json_get(d, i, "value_row_stride");
    if (stride != RIA_JSON_NONE &&
        !ria_json_u64(d, stride, true, &t->value_row_stride, e))
      return false;
    stride = ria_json_get(d, i, "scale_row_stride");
    if (stride != RIA_JSON_NONE &&
        !ria_json_u64(d, stride, true, &t->scale_row_stride, e))
      return false;
  }
  qsort(s->tensors, (size_t)s->tensor_count, sizeof(*s->tensors),
        compare_tensor);
  s->names = malloc((size_t)s->tensor_count * sizeof(*s->names));
  if (!s->names)
    return ria_fail(e, RIA_RESOURCE_LIMIT,
                    "tensor name index allocation failed");
  for (uint64_t i = 0; i < s->tensor_count; i++)
    s->names[i] = &s->tensors[i];
  qsort(s->names, (size_t)s->tensor_count, sizeof(*s->names), compare_name);
  for (uint64_t i = 1; i < s->tensor_count; i++)
    if (!strcmp(s->names[i - 1]->name, s->names[i]->name))
      return ria_fail(e, RIA_INVALID_REQUEST, "duplicate tensor name");
  for (uint64_t i = 0; i < s->tensor_count; i++) {
    const ria_tensor *t = &s->tensors[i];
    if (i && s->tensors[i - 1].id == t->id)
      return ria_fail(e, RIA_INVALID_REQUEST, "duplicate tensor ID");
    if (t->alias) {
      const ria_tensor *a = ria_tensor_id(s, t->alias_of);
      if (!a || a->alias || a->id == t->id || a->shard != t->shard ||
          a->offset != t->offset || a->length != t->length ||
          a->rank != t->rank ||
          memcmp(a->shape, t->shape, t->rank * sizeof(uint64_t)) || a->physical_rank!=t->physical_rank ||
          memcmp(a->physical_shape,t->physical_shape,t->physical_rank*sizeof(uint64_t)) || strcmp(a->dtype,t->dtype) || strcmp(a->format,t->format) ||
          strcmp(a->placement,t->placement) || strcmp(a->operation,t->operation) ||
          a->scale_count!=t->scale_count || memcmp(a->scale_ids,t->scale_ids,t->scale_count*sizeof(uint64_t)) || a->weight_global_scale!=t->weight_global_scale || a->activation_global_scale!=t->activation_global_scale)
        return ria_fail(e, RIA_INTEGRITY_ERROR, "invalid tensor alias");
    }
    for (uint32_t j = 0; j < t->scale_count; j++)
      if (!ria_tensor_id(s, t->scale_ids[j]))
        return ria_fail(e, RIA_INVALID_REQUEST, "unknown scale tensor ID");
    if (!strcmp(t->format,"nvfp4") || !strcmp(t->format,"source_mxfp4") || !strcmp(t->format,"fp8_block32")) {
      const ria_tensor *scale=ria_tensor_id(s,t->scale_ids[0]); bool packed=strcmp(t->format,"fp8_block32")!=0;
      uint64_t group=!strcmp(t->format,"nvfp4") ? 16 : 32;
      if (scale->physical_rank!=2 || scale->physical_shape[0]!=(packed ? t->shape[0] : (t->shape[0]+31)/32) ||
          scale->physical_shape[1]!=(t->shape[1]+group-1)/group || strcmp(scale->operation,"scale") ||
          (strcmp(scale->dtype,"U8") && strcmp(scale->dtype,!strcmp(t->format,"nvfp4") ? "F8_E4M3" : "F8_E8M0")))
        return ria_fail(e,RIA_INVALID_REQUEST,"quantized scale shape/type/operation invalid");
    }
  }
  return true;
}
static bool safetensors(const ria_tensor_store *s, const ria_shard *a,uint64_t metadata_budget,
                        ria_error *e) {
  uint64_t header = ria_read_u64(a->data);
  if (header > (16u << 20) || header > a->length - 8 ||
      header + 8 != a->data_start)
    return ria_fail(e, RIA_INTEGRITY_ERROR,
                    "safetensors header/data start mismatch");
  ria_json_doc d = {0};
  uint64_t owned,arenas=0,retained;
  long page=sysconf(_SC_PAGESIZE);
  if (page<=0 || !ria_tensor_owned_bytes(s,&owned,e)) return false;
  for (uint64_t i=0;i<s->shard_count;i++) {
    uint64_t rounded;
    if (!ria_u64_add(s->shards[i].length,(uint64_t)page-1,&rounded) || !ria_u64_add(arenas,rounded&~((uint64_t)page-1),&arenas))
      return ria_fail(e,RIA_RESOURCE_LIMIT,"header metadata accounting overflow");
  }
  if (owned<arenas || !ria_u64_add(owned-arenas,s->tensor_count*16,&retained) ||
      !metadata_peak(retained,header,(ria_json_limits){16u<<20,1000000,16},metadata_budget,e)) return false;
  if (!ria_json_parse(a->data + 8, (size_t)header,
                      (ria_json_limits){16u << 20, 1000000, 16}, &d, e))
    return false;
  bool ok = true;
  for (uint64_t j = 0; j < s->tensor_count && ok; j++) {
    const ria_tensor *t = &s->tensors[j];
    if (t->shard != a->id || t->alias)
      continue;
    uint32_t p = ria_json_get(&d, 0, t->name);
    const char *dtype;
    uint64_t offsets[2];
    uint32_t count;
    uint64_t shape[8], product, bytes;
    uint32_t rank;
    ok = p != RIA_JSON_NONE && string(&d, p, "dtype", &dtype, e) &&
         !strcmp(dtype, t->dtype) &&
         array(&d, p, "data_offsets", offsets, &count, 2, false, e) &&
         count == 2 && offsets[0] == t->offset && offsets[1] >= offsets[0] &&
         offsets[1] - offsets[0] == t->length &&
         array(&d, p, "shape", shape, &rank, 8, false, e) &&
         rank==t->physical_rank && !memcmp(shape,t->physical_shape,rank*sizeof(*shape)) &&
         shape_product(shape, rank, &product, e) &&
         dtype_size(dtype, product, &bytes, e) && bytes == t->length;
    if (!ok)
      ria_error_set(e, RIA_INTEGRITY_ERROR,
                    "prepared tensor %s does not match safetensors directory",
                    t->name);
  }
  /* Source directory overlap is rejected even for tensors not selected by
   * the logical inventory; explicitly tied views are manifest aliases. */
  uint64_t previous_end = 0, seen = 0;
  const ria_json_node *root = ria_json_at(&d, 0);
  if (!root || root->type != RIA_JSON_OBJECT)
    ok = ria_fail(e, RIA_INVALID_REQUEST, "invalid safetensors directory");
  if (ok) {
    uint64_t entries = 0;
    for (uint32_t k = root->child; k != RIA_JSON_NONE; k = d.nodes[k].next)
      if (strcmp(d.nodes[k].key, "__metadata__"))
        ++entries;
    if (entries>s->tensor_count) { ria_json_free(&d); return ria_fail(e,RIA_RESOURCE_LIMIT,"header inventory exceeds declared descriptor bound"); }
    uint64_t(*ranges)[2] = calloc((size_t)entries, sizeof(*ranges));
    if (entries && !ranges)
      ok = ria_fail(e, RIA_RESOURCE_LIMIT, "directory allocation failed");
    for (uint32_t k = root->child; ok && k != RIA_JSON_NONE;
         k = d.nodes[k].next) {
      if (!strcmp(d.nodes[k].key, "__metadata__"))
        continue;
      const ria_tensor *declared=ria_tensor_name(s,d.nodes[k].key);
      const char *const directory_fields[]={"dtype","shape","data_offsets"};
      if (!declared || declared->shard!=a->id || declared->alias ||
          !ria_json_fields(&d,k,directory_fields,3,directory_fields,3,e)) {
        ok=ria_fail(e,RIA_INTEGRITY_ERROR,"undeclared safetensors payload tensor"); break;
      }
      uint32_t count;
      ok = array(&d, k, "data_offsets", ranges[seen], &count, 2, false, e) &&
           count == 2;
      if (ok && (ranges[seen][0] > ranges[seen][1] ||
                 ranges[seen][1] > a->length - a->data_start))
        ok = ria_fail(e, RIA_INTEGRITY_ERROR,
                      "invalid safetensors data offsets");
      ++seen;
    }
    /* Insertion-free sort using the first uint64 and the same comparator. */
    if (ok)
      qsort(ranges, (size_t)seen, sizeof(*ranges), compare_range);
    for (uint64_t k = 0; ok && k < seen; k++) {
      if (ranges[k][0] != previous_end)
        ok =
            ria_fail(e, RIA_INTEGRITY_ERROR, "overlap or unexplained gap in safetensors tensors");
      previous_end = ranges[k][1];
    }
    if (ok && previous_end!=a->length-a->data_start)
      ok=ria_fail(e,RIA_INTEGRITY_ERROR,"unexplained trailing shard bytes");
    free(ranges);
  }
  ria_json_free(&d);
  return ok;
}
static bool value_equal(const ria_json_doc *a,uint32_t ai,const ria_json_doc *b,uint32_t bi,unsigned depth) {
  const ria_json_node *x=ria_json_at(a,ai),*y=ria_json_at(b,bi);
  if (!x || !y || x->type!=y->type || depth>64) return false;
  if (x->type==RIA_JSON_STRING) return x->length==y->length && !memcmp(x->text,y->text,x->length);
  if (x->type==RIA_JSON_NUMBER) return x->number==y->number;
  if (x->type==RIA_JSON_BOOL) return x->boolean==y->boolean;
  if (x->type==RIA_JSON_NULL) return true;
  uint32_t p=x->child,q=y->child;
  if (x->type==RIA_JSON_ARRAY) {
    for (;p!=RIA_JSON_NONE && q!=RIA_JSON_NONE;p=a->nodes[p].next,q=b->nodes[q].next)
      if (!value_equal(a,p,b,q,depth+1)) return false;
    return p==q;
  }
  unsigned left=0,right=0;
  for (;p!=RIA_JSON_NONE;p=a->nodes[p].next) {
    if (!value_equal(a,p,b,ria_json_get(b,bi,a->nodes[p].key),depth+1)) return false;
    left++;
  }
  for (;q!=RIA_JSON_NONE;q=b->nodes[q].next) right++;
  return left==right;
}
static bool contracts(ria_tensor_store *s,int root,uint64_t budget,ria_error *e) {
  uint64_t retained=s->manifest.allocated_bytes+s->index.allocated_bytes+(uint64_t)(s->page_count ? s->page_count : 1)*sizeof(ria_json_doc)+
      s->tensor_count*(sizeof(ria_tensor)+sizeof(void *))+s->shard_count*sizeof(ria_shard);
  for (unsigned i=0;i<s->page_count;i++) if (!ria_u64_add(retained,s->pages[i].allocated_bytes,&retained)) return ria_fail(e,RIA_RESOURCE_LIMIT,"contract metadata capacity overflow");
  for (uint64_t i=0;i<s->shard_count;i++) if (!ria_u64_add(retained,s->shards[i].chunk_count*32,&retained)) return ria_fail(e,RIA_RESOURCE_LIMIT,"contract chunk capacity overflow");
  const ria_json_doc *m=&s->manifest;
  const ria_json_node *refs=ria_json_at(m,ria_json_get(m,0,"metadata"));
  bool operator_seen=false,layout_seen=false;
  for (uint32_t i=refs->child;i!=RIA_JSON_NONE;i=m->nodes[i].next) {
    const char *kind,*path; uint8_t expected[32],actual[32];
    if (!string(m,i,"kind",&kind,e) || !string(m,i,"path",&path,e)) return false;
    if (strcmp(kind,"operator") && strcmp(kind,"layout")) continue;
    ria_json_doc d={0};
    bool ok=hash(m,i,"digest",expected,e) && read_page(root,path,&d,retained,budget,e) && identity(&d,expected,actual,e);
    if (ok && !strcmp(kind,"operator")) {
      const char *const fields[]={"schema_revision","profile","source_revision","graph","weight_format","activation_group","weight_scale_block","activation_quantizer","clamp_f32_bits","gate_clamp","up_clamp","coefficient_position","accumulator","reduction_order","scale_reduction_domain","calibration_digest","rounding","nibble_order","digest"};
      uint64_t revision,group,block[2]; uint32_t count;
      const char *const keys[]={"profile","source_revision","graph","weight_format","activation_quantizer","clamp_f32_bits","gate_clamp","up_clamp","coefficient_position","accumulator","reduction_order","scale_reduction_domain","rounding","nibble_order"};
      const char *source;
      if (!string(m,0,"source_revision",&source,e)) { ria_json_free(&d); return false; }
      const char *expected_strings[]={s->profile,source,"deepseek_v41_flash",!strcmp(s->profile,"fp8") ? "fp8_block32" : s->profile,
        !strcmp(s->profile,"nvfp4") ? "nvfp4_dynamic16_calibrated" : !strcmp(s->profile,"fp8") ? "fp8_e4m3fn_ue8m0_32" : "bf16_rne",
        "41200000","upper_only","two_sided","before_down_quantizer","fp32","increasing_expert_id_then_shared","full_original_population","ties_to_even","low_first"};
      ok=!operator_seen && !memcmp(actual,s->operator_contract_digest,32) && ria_json_fields(&d,0,fields,19,fields,19,e) &&
        number(&d,0,"schema_revision",false,&revision,e) && revision==1 && number(&d,0,"activation_group",false,&group,e) &&
        group==(!strcmp(s->profile,"nvfp4") ? 16u : !strcmp(s->profile,"fp8") ? 32u : 0u) && array(&d,0,"weight_scale_block",block,&count,2,false,e);
      for (unsigned j=0;ok && j<14;j++) { const char *value; ok=string(&d,0,keys[j],&value,e) && !strcmp(value,expected_strings[j]); }
      if (ok) ok=!strcmp(s->profile,"bf16") ? count==0 : count==2 && block[0]==(!strcmp(s->profile,"nvfp4") ? 1u : 32u) && block[1]==group;
      const ria_json_node *calibration=ria_json_at(&d,ria_json_get(&d,0,"calibration_digest"));
      if (ok) ok=!strcmp(s->profile,"nvfp4") ? hash(&d,0,"calibration_digest",actual,e) : calibration && calibration->type==RIA_JSON_NULL;
      operator_seen=ok;
    } else if (ok) {
      const char *const fields[]={"schema_revision","logical_model_digest","operator_contract_digest","backend","tensors","tensor_pages","digest"};
      uint64_t revision; const char *backend; uint8_t layout[32];
      ok=!layout_seen && hash(m,0,"layout_digest",layout,e) && !memcmp(layout,actual,32) && ria_json_fields(&d,0,fields,7,fields,7,e) &&
          number(&d,0,"schema_revision",false,&revision,e) && revision==1 && hash(&d,0,"logical_model_digest",actual,e) && !memcmp(actual,s->logical_model_digest,32) &&
          hash(&d,0,"operator_contract_digest",actual,e) && !memcmp(actual,s->operator_contract_digest,32) && string(&d,0,"backend",&backend,e) &&
          (!strcmp(backend,"source") || !strcmp(backend,"cpu") || !strcmp(backend,"cuda_sm120a")) &&
          value_equal(m,ria_json_get(m,0,"tensors"),&d,ria_json_get(&d,0,"tensors"),0) && value_equal(m,ria_json_get(m,0,"tensor_pages"),&d,ria_json_get(&d,0,"tensor_pages"),0);
      layout_seen=ok;
    }
    ria_json_free(&d);
    if (!ok) return e->code ? false : ria_fail(e,RIA_IDENTITY_MISMATCH,"operator/physical-layout contract does not match native realization");
  }
  return (operator_seen && layout_seen) || ria_fail(e,RIA_INTEGRITY_ERROR,"required operator/layout metadata missing");
}
static bool populate(ria_tensor_store *s, ria_shard *a, int root, const ria_tensor_load_options *o,
                     ria_tensor_place place, void *context,ria_tensor_progress progress,void *progress_context,
                     ria_error *e) {
  int fd = open_below(root, a->path, e);
  if (fd < 0)
    return false;
  struct stat st;
  bool ok = fstat(fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_size >= 0 &&
            (uint64_t)st.st_size == a->length;
  if (!ok) {
    close(fd);
    return ria_fail(e, RIA_INTEGRITY_ERROR, "shard file type/size mismatch");
  }
  /* MAP_ANONYMOUS ignores fd; -1 deliberately names no file. */
  a->data = mmap(NULL, (size_t)a->length, PROT_READ | PROT_WRITE,
                 // cppcheck-suppress invalidFunctionArg
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (a->data == MAP_FAILED || !a->data) {
    if (!a->data) {
      /* Linux permits munmap at address0; release that valid rejected map. */
      // cppcheck-suppress [nullPointer, nullPointerRedundantCheck]
      munmap(a->data, (size_t)a->length);
    }
    a->data = NULL;
    close(fd);
    return ria_fail(e, RIA_RESOURCE_LIMIT, "resident shard allocation: %s",
                    strerror(errno));
  }
  if (madvise(a->data, (size_t)a->length, MADV_DONTDUMP))
    ok = ria_fail(e, RIA_RESOURCE_LIMIT, "MADV_DONTDUMP failed");
  if (ok && o->numa_node >= 0) {
    if (numa_available() < 0 || o->numa_node > numa_max_node() ||
        !numa_bitmask_isbitset(numa_all_nodes_ptr, (unsigned)o->numa_node))
      ok = ria_fail(e, RIA_INVALID_REQUEST, "NUMA node is unavailable");
    else {
      struct bitmask *mask = numa_allocate_nodemask();
      if (!mask)
        ok = ria_fail(e, RIA_RESOURCE_LIMIT, "NUMA mask allocation failed");
      else {
        numa_bitmask_setbit(mask, (unsigned)o->numa_node);
        if (mbind(a->data, (unsigned long)a->length, MPOL_BIND, mask->maskp,
                  mask->size + 1, 0))
          ok = ria_fail(e, RIA_RESOURCE_LIMIT, "NUMA memory binding failed: %s",
                        strerror(errno));
        numa_bitmask_free(mask);
      }
    }
  }
  if (ok && place)
    ok = place(context, s, a, false, e);
  EVP_MD_CTX *md = EVP_MD_CTX_new();
  if (!md || EVP_DigestInit_ex(md, EVP_sha256(), NULL) != 1)
    ok = ria_fail(e, RIA_INTERNAL_ERROR, "SHA context creation failed");
  for (uint64_t i = 0; ok && i < a->chunk_count; i++) {
    if (progress && !progress(progress_context,e)) { ok=false; break; }
    uint64_t begin = i * a->chunk_size, n = a->length - begin;
    if (n > a->chunk_size)
      n = a->chunk_size;
    size_t got = 0;
    while (got < n) {
      ssize_t r = read(fd, a->data + begin + got, (size_t)n - got);
      if (r < 0 && errno == EINTR)
        continue;
      if (r <= 0) {
        ok = ria_fail(e, RIA_INTEGRITY_ERROR, "short/failed shard read");
        break;
      }
      got += (size_t)r;
    }
    uint8_t h[32];
    if (ok)
      ok = ria_sha256(a->data + begin, (size_t)n, h, e) &&
           !memcmp(h, a->chunk_hashes + 32 * i, 32) &&
           EVP_DigestUpdate(md, a->data + begin, (size_t)n) == 1;
    if (!ok && (!e || !e->code))
      ria_error_set(e, RIA_INTEGRITY_ERROR, "chunk SHA mismatch");
  }
  uint8_t full[32];
  unsigned length = 0;
  if (ok && (EVP_DigestFinal_ex(md, full, &length) != 1 || length != 32 ||
             memcmp(full, a->digest, 32)))
    ok = ria_fail(e, RIA_INTEGRITY_ERROR, "complete shard SHA mismatch");
  EVP_MD_CTX_free(md);
  if (close(fd) && ok)
    ok = ria_fail(e, RIA_INTEGRITY_ERROR, "shard close failed");
  if (ok && place)
    ok = place(context, s, a, true, e);
  if (ok && o->lock_memory) {
    if (mlock(a->data, (size_t)a->length))
      ok = ria_fail(e, RIA_RESOURCE_LIMIT, "mandatory resident lock failed: %s",
                    strerror(errno));
    else
      a->locked = true;
  }
  if (ok && mprotect(a->data, (size_t)a->length, PROT_READ))
    ok = ria_fail(e, RIA_RESOURCE_LIMIT, "immutable shard protection failed");
  return ok;
}
static bool store_read(ria_tensor_store *,const char *,const ria_tensor_load_options *,
                       ria_tensor_place,void *,ria_tensor_progress,void *,bool,ria_error *);
bool ria_tensor_store_inspect(ria_tensor_store *s,const char *path,
                              const ria_tensor_load_options *o,ria_error *e) {
  return store_read(s,path,o,NULL,NULL,NULL,NULL,true,e);
}
bool ria_tensor_store_open(ria_tensor_store *s, const char *path,
                           const ria_tensor_load_options *o, ria_error *e) {
  return ria_tensor_store_open_placed(s, path, o, NULL, NULL, e);
}
bool ria_tensor_store_open_placed(ria_tensor_store *s, const char *path,
                           const ria_tensor_load_options *o,
                           ria_tensor_place place, void *context, ria_error *e) {
  return ria_tensor_store_open_controlled(s,path,o,place,context,NULL,NULL,e);
}
bool ria_tensor_store_open_controlled(ria_tensor_store *s,const char *path,const ria_tensor_load_options *o,
                                     ria_tensor_place place,void *context,ria_tensor_progress progress,void *progress_context,ria_error *e) {
  return store_read(s,path,o,place,context,progress,progress_context,false,e);
}
static bool store_read(ria_tensor_store *s,const char *path,const ria_tensor_load_options *o,
                       ria_tensor_place place,void *context,ria_tensor_progress progress,void *progress_context,
                       bool metadata_only,ria_error *e) {
  memset(s, 0, sizeof(*s));
  if (!o || !o->expected_digest || !o->role || !o->max_resident_bytes)
    return ria_fail(e, RIA_INVALID_REQUEST, "missing trusted tensor admission");
  uint64_t metadata_budget=o->max_metadata_bytes ? o->max_metadata_bytes : o->max_resident_bytes;
  int manifest_fd=open(path,O_RDONLY|O_CLOEXEC|O_NOFOLLOW);
  if (manifest_fd<0) return ria_fail(e,RIA_INTEGRITY_ERROR,"bounded manifest open failed");
  if (!read_document(manifest_fd,&s->manifest,manifest_limits,0,metadata_budget,e) ||
      !identity(&s->manifest, o->expected_digest, s->digest, e))
    goto fail;
  char *copy = strdup(path);
  if (!copy) {
    ria_error_set(e, RIA_RESOURCE_LIMIT, "path allocation failed");
    goto fail;
  }
  char *slash = strrchr(copy, '/');
  const char *directory = ".";
  if (slash) {
    if (slash == copy)
      slash[1] = 0;
    else
      *slash = 0;
    directory = copy;
  }
  int root = open(directory, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  free(copy);
  if (root < 0) {
    ria_error_set(e, RIA_INTEGRITY_ERROR, "artifact directory open failed");
    goto fail;
  }
  bool ok = indexes(s, root, metadata_budget,e) && descriptors(s, o, e) && contracts(s,root,metadata_budget,e);
  if (metadata_only) {
    if (close(root) && ok) ok=ria_fail(e,RIA_INTEGRITY_ERROR,"metadata directory close failed");
    if (!ok) goto fail;
    return true;
  }
  if (ok && place) ok=place(context,s,NULL,false,e);
  for (uint64_t i = 0; ok && i < s->shard_count; i++)
    ok =
        populate(s, &s->shards[i], root, o, place, context, progress,progress_context,e) && safetensors(s, &s->shards[i],metadata_budget,e);
  if (close(root) && ok)
    ok = ria_fail(e, RIA_INTEGRITY_ERROR, "artifact directory close failed");
  if (!ok)
    goto fail;
  /* Payload hashes bind individual descriptors independently of shard layout.
   */
  const ria_json_node *ts =
      ria_json_at(&s->index, ria_json_get(&s->index, 0, "tensors"));
  for (uint32_t i = ts->child; i != RIA_JSON_NONE; i = s->index.nodes[i].next) {
    uint64_t id;
    uint8_t expected[32], actual[32];
    if (!number(&s->index, i, "id", true, &id, e) ||
        !hash(&s->index, i, "sha256", expected, e))
      goto fail;
    ria_tensor *t = (ria_tensor *)ria_tensor_id(s, id);
    const ria_shard *a = ria_shard_id(s, t->shard);
    t->data = a->data + a->data_start + t->offset;
    EVP_MD_CTX *md=EVP_MD_CTX_new(); bool hashed=md && EVP_DigestInit_ex(md,EVP_sha256(),NULL)==1;
    for (uint64_t offset=0;hashed && offset<t->length;) {
      uint64_t count=t->length-offset;
      if (count>RIA_BULK_MAX) count=RIA_BULK_MAX;
      if (progress && !progress(progress_context,e)) { hashed=false; break; }
      hashed=EVP_DigestUpdate(md,t->data+offset,(size_t)count)==1; offset+=count;
    }
    unsigned digest_length=0;
    if (hashed) hashed=EVP_DigestFinal_ex(md,actual,&digest_length)==1 && digest_length==32;
    EVP_MD_CTX_free(md);
    if (!hashed ||
        memcmp(expected, actual, 32)) {
      if (!e->code) ria_error_set(e, RIA_INTEGRITY_ERROR, "tensor SHA mismatch: %s", t->name);
      goto fail;
    }
  }
  return true;
fail:
  ria_tensor_store_close(s);
  return false;
}
void ria_tensor_store_close(ria_tensor_store *s) {
  if (!s)
    return;
  for (uint64_t i = 0; s->shards && i < s->shard_count; i++) {
    ria_shard *a = &s->shards[i];
    if (a->data) {
      if (a->locked)
        munlock(a->data, (size_t)a->length);
      munmap(a->data, (size_t)a->length);
    }
    free(a->chunk_hashes);
  }
  free(s->names);
  free(s->tensors);
  free(s->shards);
  ria_json_free(&s->index);
  for (uint32_t i = 0; i < s->page_count; i++)
    ria_json_free(&s->pages[i]);
  free(s->pages);
  ria_json_free(&s->manifest);
  memset(s, 0, sizeof(*s));
}
bool ria_tensor_owned_bytes(const ria_tensor_store *s,uint64_t *out,ria_error *e) {
  if (!s || !out) return ria_fail(e,RIA_INVALID_REQUEST,"missing TensorStore accounting output");
  uint64_t bytes=sizeof(*s),n;
#define ADD(value) do { if (!ria_u64_add(bytes,(value),&bytes)) goto overflow; } while (0)
  ADD(s->manifest.allocated_bytes); ADD(s->index.allocated_bytes);
  if (!ria_u64_mul(s->tensor_count,sizeof(ria_tensor)+sizeof(void *),&n)) goto overflow;
  ADD(n);
  if (!ria_u64_mul(s->shard_count,sizeof(ria_shard),&n)) goto overflow;
  ADD(n);
  if (!ria_u64_mul(s->page_count ? s->page_count : 1,sizeof(ria_json_doc),&n)) goto overflow;
  ADD(n);
  for (unsigned i=0;i<s->page_count;i++) ADD(s->pages[i].allocated_bytes);
  long page=sysconf(_SC_PAGESIZE);
  if (page<=0 || (page&(page-1))) return ria_fail(e,RIA_UNSUPPORTED,"invalid native page size");
  for (uint64_t i=0;i<s->shard_count;i++) {
    if (!ria_u64_add(s->shards[i].length,(uint64_t)page-1,&n)) goto overflow;
    ADD(n&~((uint64_t)page-1));
    if (!ria_u64_mul(s->shards[i].chunk_count,32,&n)) goto overflow;
    ADD(n);
  }
#undef ADD
  *out=bytes; return true;
overflow: return ria_fail(e,RIA_RESOURCE_LIMIT,"TensorStore owned byte accounting overflow");
}
bool ria_tensor_locked_bytes(const ria_tensor_store *s,uint64_t *out,ria_error *e) {
  if (!s || !out) return ria_fail(e,RIA_INVALID_REQUEST,"missing locked population output");
  uint64_t bytes=0; long page=sysconf(_SC_PAGESIZE);
  if (page<=0 || (page&(page-1))) return ria_fail(e,RIA_UNSUPPORTED,"invalid native page size");
  for (uint64_t i=0;i<s->shard_count;i++) if (s->shards[i].locked) {
    uint64_t n;
    if (!ria_u64_add(s->shards[i].length,(uint64_t)page-1,&n) || !ria_u64_add(bytes,n&~((uint64_t)page-1),&bytes))
      return ria_fail(e,RIA_RESOURCE_LIMIT,"locked population overflow");
  }
  *out=bytes; return true;
}
bool ria_tensor_matrix(const ria_tensor_store *s, const ria_tensor *t,
                       ria_expert_matrix *m, ria_error *e) {
  memset(m, 0, sizeof(*m));
  if (!t || t->rank != 2 || !t->data || strcmp(t->layout, "row_major_le"))
    return ria_fail(e, RIA_UNSUPPORTED,
                    "matrix requires immutable row-major rank two tensor");
  m->out_features = t->shape[0];
  m->in_features = t->shape[1];
  m->values = t->data;
  m->values_bytes = t->length;
  m->weight_global_scale = t->weight_global_scale;
  m->activation_global_scale = t->activation_global_scale;
  if (!strcmp(t->format, "plain") && !strcmp(t->dtype, "F32"))
    m->profile = RIA_EXPERT_F32;
  else if (!strcmp(t->format, "bf16") ||
           (!strcmp(t->format, "plain") && !strcmp(t->dtype, "BF16")))
    m->profile = RIA_EXPERT_BF16;
  else if (!strcmp(t->format, "fp8_block32"))
    m->profile = RIA_EXPERT_FP8;
  else if (!strcmp(t->format, "nvfp4"))
    m->profile = RIA_EXPERT_NVFP4;
  else
    return ria_fail(e, RIA_UNSUPPORTED, "unsupported expert matrix format %s",
                    t->format);
  m->value_row_stride = t->value_row_stride;
  if (!m->value_row_stride)
    m->value_row_stride = m->profile == RIA_EXPERT_F32    ? m->in_features * 4
                          : m->profile == RIA_EXPERT_BF16 ? m->in_features * 2
                          : m->profile == RIA_EXPERT_FP8
                              ? m->in_features
                              : (m->in_features + 1) / 2;
  if (m->profile != RIA_EXPERT_BF16 && m->profile != RIA_EXPERT_F32) {
    if (t->scale_count != 1)
      return ria_fail(e, RIA_INVALID_REQUEST,
                      "quantized matrix requires one explicit scale tensor");
    const ria_tensor *scale = ria_tensor_id(s, t->scale_ids[0]);
    m->scales = scale->data;
    m->scales_bytes = scale->length;
    m->scale_row_stride = t->scale_row_stride;
    if (!m->scale_row_stride)
      m->scale_row_stride =
          (m->in_features + (m->profile == RIA_EXPERT_FP8 ? 31 : 15)) /
          (m->profile == RIA_EXPERT_FP8 ? 32 : 16);
  }
  return ria_expert_matrix_validate(m, e);
}
bool ria_tensor_chunk(const ria_tensor_store *s, uint64_t id, uint64_t index,
                      const uint8_t **bytes, uint32_t *length,
                      const uint8_t **hash_out, ria_error *e) {
  const ria_shard *a = ria_shard_id(s, id);
  if (!a || index >= a->chunk_count)
    return ria_fail(e, RIA_UNAUTHORIZED, "chunk is outside prepared grant");
  uint64_t offset = index * a->chunk_size, n = a->length - offset;
  if (n > a->chunk_size)
    n = a->chunk_size;
  *bytes = a->data + offset;
  *length = (uint32_t)n;
  *hash_out = a->chunk_hashes + 32 * index;
  return true;
}
