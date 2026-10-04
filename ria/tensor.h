#ifndef RIA_TENSOR_H
#define RIA_TENSOR_H
#include "expert.h"
#include "json.h"
#ifdef __cplusplus
extern "C" {
#endif

#define RIA_TENSOR_DIMS 8u
typedef struct {
  uint64_t id, shard, offset, length, shape[RIA_TENSOR_DIMS], physical_shape[RIA_TENSOR_DIMS];
  uint32_t rank, physical_rank;
  const char *name, *dtype, *format, *layout, *placement, *operation;
  uint64_t scale_ids[4];
  uint32_t scale_count;
  uint64_t value_row_stride, scale_row_stride;
  float weight_global_scale, activation_global_scale;
  const uint8_t *data;
  uint64_t alias_of;
  bool alias;
} ria_tensor;
typedef struct {
  uint64_t id, length, data_start;
  uint32_t chunk_size;
  uint8_t digest[32];
  const char *path;
  uint8_t *data, *chunk_hashes;
  uint64_t chunk_count;
  bool locked;
} ria_shard;
typedef struct {
  ria_json_doc manifest;
  ria_json_doc index;
  ria_json_doc *pages;
  uint32_t page_count;
  ria_tensor *tensors;
  const ria_tensor **names;
  ria_shard *shards;
  uint64_t tensor_count, shard_count, resident_bytes;
  uint8_t digest[32], logical_model_digest[32], operator_contract_digest[32],
      encoding_digest[32];
  char role[7], profile[6];
} ria_tensor_store;
typedef struct {
  const char *role;
  const uint8_t *expected_digest; /* provisioned trusted root, required */
  uint64_t max_resident_bytes;
  bool lock_memory;
  int numa_node; /* -1: caller-selected default policy; otherwise explicit bind
                  */
  uint64_t max_metadata_bytes; /* 0 derives max_resident_bytes; includes peak
                               * retained DOM/index and transient parse bytes. */
} ria_tensor_load_options;

/* Validate the complete descriptor before allocation. Payloads are copied to
 * owned anonymous RAM while hashing the actual copied bytes; source files are
 * never demand-paged during execution. Each shard is one coarse allocation.
 * Store and borrowed descriptors outlive every worker/device transfer. */
bool ria_tensor_store_open(ria_tensor_store *store, const char *manifest_path,
                           const ria_tensor_load_options *options,
                           ria_error *error);
/* Preparation/admission inspection only. Authenticate the complete bounded
 * metadata graph and validate descriptors without opening/copying payloads,
 * allocating a bank, touching NUMA policy or initializing CUDA. Every data
 * pointer remains NULL. This does not establish payload integrity/readiness.
 * The returned metadata owner must be closed through store_close. */
bool ria_tensor_store_inspect(ria_tensor_store *, const char *,
                             const ria_tensor_load_options *, ria_error *);
/* Startup-only callback first receives shard=NULL for complete metadata-only
 * admission, before any payload arena allocation. Then before any shard page
 * is touched and again after
 * first-touch authenticated copy. It may bind untouched anonymous page ranges
 * and verify physical locations. Never migrate or mutate published payloads.
 * Store descriptors are complete but data pointers are not yet published. */
typedef bool (*ria_tensor_place)(void *context, const ria_tensor_store *store,
                               const ria_shard *shard, bool populated,
                               ria_error *error);
bool ria_tensor_store_open_placed(ria_tensor_store *store, const char *path,
                                 const ria_tensor_load_options *options,
                                 ria_tensor_place place, void *context,
                                 ria_error *error);
typedef bool (*ria_tensor_progress)(void *context,ria_error *error);
/* Progress callback runs between bounded authentication/copy blocks. A process
 * owner can request cancellation without freeing arenas used by startup. */
bool ria_tensor_store_open_controlled(ria_tensor_store *,const char *,const ria_tensor_load_options *,
                                     ria_tensor_place,void *,ria_tensor_progress,void *,ria_error *);
void ria_tensor_store_close(ria_tensor_store *store);
/* Pure checked accounting, including metadata capacities and rounded shard
 * arenas. Valid after descriptor validation, before payload allocation. */
bool ria_tensor_owned_bytes(const ria_tensor_store *,uint64_t *,ria_error *);
bool ria_tensor_locked_bytes(const ria_tensor_store *,uint64_t *,ria_error *);
const ria_tensor *ria_tensor_id(const ria_tensor_store *store, uint64_t id);
const ria_tensor *ria_tensor_name(const ria_tensor_store *store,
                                  const char *name);
const ria_shard *ria_shard_id(const ria_tensor_store *store, uint64_t id);
bool ria_tensor_matrix(const ria_tensor_store *store, const ria_tensor *tensor,
                       ria_expert_matrix *matrix, ria_error *error);
/* The trusted chunk index authorizes every byte including metadata/overfetch.
 */
bool ria_tensor_chunk(const ria_tensor_store *store, uint64_t shard_id,
                      uint64_t index, const uint8_t **bytes, uint32_t *length,
                      const uint8_t **hash, ria_error *error);
#ifdef __cplusplus
}
#endif
#endif
