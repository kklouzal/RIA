#define _POSIX_C_SOURCE 200809L
#include "common.h"
#include <errno.h>
#include <float.h>
#include <limits.h>
#include <openssl/evp.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#ifdef __linux__
#include <sys/prctl.h>
#endif
_Static_assert(sizeof(float) == 4, "wire requires binary32 float");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128,
               "wire requires IEEE754 binary32");
void ria_error_set(ria_error *e, int code, const char *format, ...) {
  if (e) {
    e->code = code;
    va_list args;
    va_start(args, format);
    (void)vsnprintf(e->message, sizeof e->message, format, args);
    va_end(args);
  }
}
bool ria_u64_add(uint64_t a, uint64_t b, uint64_t *out) {
  if (!out || b > UINT64_MAX - a)
    return false;
  *out = a + b;
  return true;
}
bool ria_u64_mul(uint64_t a, uint64_t b, uint64_t *out) {
  if (!out || (a && b > UINT64_MAX / a))
    return false;
  *out = a * b;
  return true;
}
bool ria_size(uint64_t bytes, size_t *out, ria_error *e) {
  if (!out || bytes > SIZE_MAX || bytes > PTRDIFF_MAX)
    return ria_fail(e, RIA_RESOURCE_LIMIT,
                    "allocation exceeds addressable object size");
  *out = (size_t)bytes;
  return true;
}
bool ria_parse_u64(const char *text, size_t n, uint64_t *out, ria_error *e) {
  if (!text || !n || !out || (n > 1 && text[0] == '0'))
    return ria_fail(e, RIA_INVALID_REQUEST, "invalid u64 decimal string");
  uint64_t v = 0;
  for (size_t i = 0; i < n; i++) {
    unsigned d = (unsigned char)text[i] - '0';
    if (d > 9 || v > (UINT64_MAX - d) / 10)
      return ria_fail(e, RIA_INVALID_REQUEST,
                      "u64 decimal string out of range");
    v = v * 10 + d;
  }
  *out = v;
  return true;
}
bool ria_hex_decode(const char *text, size_t n, uint8_t *out, size_t bytes,
                    ria_error *e) {
  if (!text || !out || bytes > SIZE_MAX / 2 || n != bytes * 2)
    return ria_fail(e, RIA_INVALID_REQUEST,
                    "invalid hexadecimal identity length");
  for (size_t i = 0; i < n; i++) {
    unsigned char c = (unsigned char)text[i];
    unsigned d = c >= '0' && c <= '9'   ? c - '0'
                 : c >= 'a' && c <= 'f' ? c - 'a' + 10
                                        : 16;
    if (d > 15)
      return ria_fail(e, RIA_INVALID_REQUEST,
                      "identity requires lowercase hexadecimal");
    if (!(i & 1))
      out[i / 2] = (uint8_t)(d << 4);
    else
      out[i / 2] |= (uint8_t)d;
  }
  return true;
}
void ria_hex_encode(const uint8_t *bytes, size_t n, char *out) {
  static const char digits[] = "0123456789abcdef";
  for (size_t i = 0; i < n; i++) {
    out[2 * i] = digits[bytes[i] >> 4];
    out[2 * i + 1] = digits[bytes[i] & 15];
  }
  out[2 * n] = 0;
}
bool ria_sha256(const void *bytes, size_t n, uint8_t out[32], ria_error *e) {
  unsigned length = 0;
  if (!out || (!bytes && n) ||
      EVP_Digest(bytes, n, out, &length, EVP_sha256(), NULL) != 1 ||
      length != 32)
    return ria_fail(e, RIA_INTERNAL_ERROR, "SHA256 failed");
  return true;
}
bool ria_sha256_file(const char *path, uint8_t out[32], ria_error *e) {
  FILE *f = fopen(path, "rb");
  if (!f)
    return ria_fail(e, RIA_INTEGRITY_ERROR, "cannot read integrity input");
  EVP_MD_CTX *ctx = EVP_MD_CTX_new();
  bool ok = ctx && EVP_DigestInit_ex(ctx, EVP_sha256(), NULL) == 1;
  unsigned char buffer[65536];
  while (ok) {
    size_t n = fread(buffer, 1, sizeof buffer, f);
    if (n && EVP_DigestUpdate(ctx, buffer, n) != 1) {
      ok = false;
      break;
    }
    if (n < sizeof buffer) {
      if (ferror(f))
        ok = false;
      break;
    }
  }
  unsigned length = 0;
  if (ok)
    ok = EVP_DigestFinal_ex(ctx, out, &length) == 1 && length == 32;
  EVP_MD_CTX_free(ctx);
  if (fclose(f) != 0)
    ok = false;
  if (!ok)
    return ria_fail(e, RIA_INTEGRITY_ERROR, "file SHA256 failed");
  return true;
}
bool ria_monotonic_ns(uint64_t *out, ria_error *e) {
  struct timespec t;
  uint64_t seconds;
  if (!out || clock_gettime(CLOCK_MONOTONIC, &t) || t.tv_sec < 0 ||
      t.tv_nsec < 0 || t.tv_nsec >= 1000000000 ||
      !ria_u64_mul((uint64_t)t.tv_sec, UINT64_C(1000000000), &seconds) ||
      !ria_u64_add(seconds, (uint64_t)t.tv_nsec, out))
    return ria_fail(e, RIA_EXECUTOR_ERROR,
                    "monotonic nanosecond observation failed");
  return true;
}
uint64_t ria_monotonic_ms(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
    return UINT64_MAX;
  return (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;
}
uint16_t ria_read_u16(const uint8_t *p) {
  return (uint16_t)((uint16_t)p[0] | (uint16_t)p[1] << 8);
}
uint32_t ria_read_u32(const uint8_t *p) {
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 |
         (uint32_t)p[3] << 24;
}
uint64_t ria_read_u64(const uint8_t *p) {
  return (uint64_t)ria_read_u32(p) | (uint64_t)ria_read_u32(p + 4) << 32;
}
float ria_read_f32(const uint8_t *p) {
  uint32_t v = ria_read_u32(p);
  float f;
  memcpy(&f, &v, 4);
  return f;
}
void ria_write_u16(uint8_t *p, uint16_t v) {
  p[0] = (uint8_t)v;
  p[1] = (uint8_t)(v >> 8);
}
void ria_write_u32(uint8_t *p, uint32_t v) {
  for (unsigned i = 0; i < 4; i++)
    p[i] = (uint8_t)(v >> (i * 8));
}
void ria_write_u64(uint8_t *p, uint64_t v) {
  for (unsigned i = 0; i < 8; i++)
    p[i] = (uint8_t)(v >> (i * 8));
}
void ria_write_f32(uint8_t *p, float f) {
  uint32_t v;
  memcpy(&v, &f, 4);
  ria_write_u32(p, v);
}
bool ria_disable_dumps(ria_error *e) {
  struct rlimit limit = {0, 0};
  if (setrlimit(RLIMIT_CORE, &limit) != 0)
    return ria_fail(e, RIA_INTERNAL_ERROR, "cannot disable core dumps");
#ifdef __linux__
  if (prctl(PR_SET_DUMPABLE, 0) != 0 || prctl(PR_GET_DUMPABLE) != 0)
    return ria_fail(e, RIA_INTERNAL_ERROR, "cannot disable process dumps");
#endif
  return true;
}
