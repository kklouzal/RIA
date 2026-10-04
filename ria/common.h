#ifndef RIA_COMMON_H
#define RIA_COMMON_H
#if defined(__FAST_MATH__) ||                                                  \
    (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error                                                                         \
    "RIA numerical contracts require finite checks and strict floating-point semantics"
#endif
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
typedef enum {
  RIA_OK = 0,
  RIA_INVALID_REQUEST = 1,
  RIA_UNAUTHORIZED = 2,
  RIA_IDENTITY_MISMATCH = 3,
  RIA_UNSUPPORTED = 4,
  RIA_RESOURCE_LIMIT = 5,
  RIA_NOT_READY = 6,
  RIA_DEADLINE_EXCEEDED = 7,
  RIA_CANCELLED = 8,
  RIA_INTEGRITY_ERROR = 9,
  RIA_EXECUTOR_ERROR = 10,
  RIA_INTERNAL_ERROR = 11
} ria_status;
typedef struct {
  int code;
  char message[256];
} ria_error;
#if defined(__GNUC__) || defined(__clang__)
#define RIA_PRINTF(format_index, argument_index)                               \
  __attribute__((format(printf, format_index, argument_index)))
#else
#define RIA_PRINTF(format_index, argument_index)
#endif
void ria_error_set(ria_error *error, int code, const char *format, ...)
    RIA_PRINTF(3, 4);
/* The false result is explicit to callers and single-translation-unit
 * analyzers. */
#define ria_fail(...) (ria_error_set(__VA_ARGS__), false)
bool ria_u64_add(uint64_t a, uint64_t b, uint64_t *out);
bool ria_u64_mul(uint64_t a, uint64_t b, uint64_t *out);
bool ria_size(uint64_t bytes, size_t *out, ria_error *error);
bool ria_parse_u64(const char *text, size_t length, uint64_t *out,
                   ria_error *error);
bool ria_hex_decode(const char *text, size_t length, uint8_t *out,
                    size_t out_length, ria_error *error);
void ria_hex_encode(const uint8_t *bytes, size_t length, char *out);
bool ria_sha256(const void *bytes, size_t length, uint8_t out[32],
                ria_error *error);
bool ria_sha256_file(const char *path, uint8_t out[32], ria_error *error);
uint64_t ria_monotonic_ms(void);
/* Checked monotonic observation; zero is a valid timestamp. */
bool ria_monotonic_ns(uint64_t *out, ria_error *error);
uint16_t ria_read_u16(const uint8_t *p);
uint32_t ria_read_u32(const uint8_t *p);
uint64_t ria_read_u64(const uint8_t *p);
float ria_read_f32(const uint8_t *p);
void ria_write_u16(uint8_t *p, uint16_t v);
void ria_write_u32(uint8_t *p, uint32_t v);
void ria_write_u64(uint8_t *p, uint64_t v);
void ria_write_f32(uint8_t *p, float v);
/* Disable process dumps before secrets. Linux MADV_DONTDUMP is separate. */
bool ria_disable_dumps(ria_error *error);
#ifdef __cplusplus
}
#endif
#endif
