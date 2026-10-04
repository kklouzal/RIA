#ifndef RIA_ADMISSION_H
#define RIA_ADMISSION_H
#include "json.h"
#ifdef __cplusplus
extern "C" {
#endif
#define RIA_PHASES 6u
#define RIA_NUMA_MAX 64u
typedef struct {
  uint64_t host, device, pinned;
  uint64_t numa[RIA_NUMA_MAX];
} ria_memory_peak;
typedef struct {
  ria_memory_peak phases[RIA_PHASES], peak, caps;
  uint64_t context_positions, allocation_count;
  char role[7], executor[5], profile[6];
  uint8_t logical_model_digest[32], operator_contract_digest[32],
      request_digest[32], inventory_digest[32], probe_digest[32],
      calibration_digest[32];
  uint8_t environment_digest[32], build_digest[32], policy_digest[32];
} ria_admission_plan;
extern const char *const ria_phase_names[RIA_PHASES];
/* The only equations used by CLI and offline renderer. Input schemas are
 * strict. */
bool ria_admission_compute(const ria_json_doc *request,
                           const ria_json_doc *inventory,
                           const ria_json_doc *probe,
                           const ria_json_doc *calibration,
                           ria_admission_plan *plan, ria_error *error);
bool ria_admission_json(const ria_admission_plan *plan, char **json,
                        size_t *length, ria_error *error);
/* Four independent input files, no already-admitted memory plan. Atomic output.
 */
bool ria_admission_files(const char *request, const char *inventory,
                         const char *probe, const char *calibration,
                         const char *output, ria_error *error);
#ifdef __cplusplus
}
#endif
#endif
