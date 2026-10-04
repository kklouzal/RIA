#ifndef RIA_PROBE_H
#define RIA_PROBE_H
#include "json.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
  bool cuda, client;
  uint64_t host_test_bytes, device_test_bytes, pinned_test_bytes, deadline_ms;
  unsigned nodes[64], node_count;
  char gpu_uuid[41];
  uint8_t environment_digest[32], build_digest[32];
  char build_info_file[4096];
} ria_probe_config;
typedef struct {
  uint64_t device_bytes, pinned_bytes, allocation_ms, kernel_ms;
  int driver_version, runtime_version, major, minor;
  float native_results[3];
  char uuid[41];
} ria_probe_gpu_result;
bool ria_probe_config_read(const char *path, ria_probe_config *config,
                           ria_error *error);
/* A supervised child performs the independently capped allocations/kernels.
 * The parent kills and reaps it on deadline. No model or final plan is read.
 * CPU builds contain no CUDA dependency. Never invoke as an offline test. */
bool ria_probe_run(const ria_probe_config *config, char **report,
                   size_t *report_length, char **details,
                   size_t *details_length, ria_error *error);
bool ria_probe_gpu(const ria_probe_config *config, ria_probe_gpu_result *result,
                   ria_error *error);
/* Canonical JCS output, protected temporary file, fsync and atomic publication.
 * Caller authorizes replacement of the specified output report. */
bool ria_report_write(const char *path, const void *bytes, size_t length,
                      ria_error *error);
#ifdef __cplusplus
}
#endif
#endif
