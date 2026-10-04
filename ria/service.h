#ifndef RIA_SERVICE_H
#define RIA_SERVICE_H
#include "tensor.h"
#include "transport.h"
#ifdef __cplusplus
extern "C" {
#endif
#define RIA_NUMA_MAX_NODES 64u
#define RIA_NUMA_MAX_CPUS 256u
typedef enum { RIA_NUMA_SHARDED=1, RIA_NUMA_REPLICATED_EXPERTS=2,
               RIA_NUMA_REPLICATED_SERVER_MODEL=3 } ria_numa_policy;
typedef struct {
  unsigned node, workers, cpu_count;
  unsigned cpus[RIA_NUMA_MAX_CPUS];
  uint64_t local_bytes;
} ria_expert_node;
typedef struct {
  ria_numa_policy policy;
  unsigned node_count, worker_count;
  ria_expert_node nodes[RIA_NUMA_MAX_NODES];
  uint64_t projection_tile_rows, host_runtime_bytes, startup_host_bytes,
      device_workspace_bytes, pinned_workspace_bytes, drain_timeout_ms;
} ria_expert_config;
typedef struct {
  ria_json_doc document, lock, plan, grants, placement;
  const char *role, *executor, *profile, *manifest_path, *admin_socket,
      *peer_grants;
  const char *control_address, *bulk_address, *gpu_uuid, *server_executor;
  ria_tls_config tls;
  ria_limits limits;
  ria_expert_config expert;
  uint64_t connect_timeout_ms, host_cap, device_cap, pinned_cap,
      context_positions, prefill_rows;
  uint8_t manifest_digest[32], logical_model_digest[32],
      operator_contract_digest[32];
} ria_service;
/* Pure configuration parsing shared by metadata-only inventory and serving.
 * It checks chosen reservations/affinities against caps, without requiring an
 * admitted model or inspecting physical topology. Serving additionally checks
 * the immutable plan and actual placement before readiness. */
bool ria_expert_config_parse(const ria_json_doc *,uint32_t,const char *,uint32_t prefill_rows,
                             uint64_t,uint64_t,uint64_t,ria_expert_config *,ria_error *);
/* Pure boundary parsing for the shared transport memory/lifecycle contract.
 * Updates only network fields; returned strings borrow the input document. */
bool ria_service_network_parse(const ria_json_doc *,uint32_t,ria_service *,ria_error *);
bool ria_service_read(ria_service *service, const char *path, ria_error *error);
void ria_service_free(ria_service *service);
bool ria_service_grant(const ria_service *service,
                       const ria_tensor_store *store, const ria_json_doc *bind,
                       ria_error *error);
/* Explicit numeric address forms are IPv4:port or [IPv6]:port. No DNS or
 * externally timed resolution occurs. TLS peer-name verification is separate.
 */
bool ria_address(const char *text, bool listen, struct sockaddr_storage *out,
                 socklen_t *length, ria_error *error);
bool ria_admin_request(const char *path, const char *command,
                       uint64_t timeout_ms, char **reply, size_t *length,
                       ria_error *error);
#ifdef __cplusplus
}
#endif
#endif
