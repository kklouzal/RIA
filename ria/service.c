#define _GNU_SOURCE
#include "service.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <netdb.h>
#include <poll.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

static bool text(const ria_json_doc *d, uint32_t p, const char *key,
                 const char **out, ria_error *e) {
  size_t n = 0;
  if (!ria_json_string(d, ria_json_get(d, p, key), out, &n, e))
    return false;
  return n && n < 4096 && strlen(*out) == n
             ? true
             : ria_fail(e, RIA_INVALID_REQUEST, "invalid string %s", key);
}
static bool integer(const ria_json_doc *d, uint32_t p, const char *key,
                    uint64_t *out, ria_error *e) {
  return ria_json_u64(d, ria_json_get(d, p, key), false, out, e);
}
static bool digest(const ria_json_doc *d, uint32_t p, const char *key,
                   uint8_t out[32], ria_error *e) {
  return ria_json_digest_field(d, ria_json_get(d, p, key), out, e);
}
static bool checked_doc(const char *path, const uint8_t expected[32],
                        ria_json_doc *d, ria_error *e) {
  uint8_t actual[32], claimed[32];
  return ria_json_read(path, (ria_json_limits){256u << 10, 32768, 48}, d, e) &&
         digest(d, 0, "digest", claimed, e) &&
         ria_json_sha256(d, true, actual, e) &&
         ((!memcmp(actual, claimed, 32) &&
           (!expected || !memcmp(actual, expected, 32))) ||
          ria_fail(e, RIA_INTEGRITY_ERROR,
                   "provisioned document digest mismatch"));
}
bool ria_expert_config_parse(const ria_json_doc *d,uint32_t x,const char *executor,
                              uint64_t host_cap,uint64_t device_cap,uint64_t pinned_cap,
                              ria_expert_config *c,ria_error *e) {
  if (!d || !executor || !c || (strcmp(executor,"cpu") && strcmp(executor,"cuda")))
    return ria_fail(e,RIA_INVALID_REQUEST,"invalid expert configuration parse input");
  memset(c,0,sizeof(*c)); const char *policy;
  const char *const fields[]={"numa_policy","nodes","projection_tile_rows","host_runtime_bytes",
    "startup_host_bytes","device_workspace_bytes","pinned_workspace_bytes","drain_timeout_ms"};
  if (!ria_json_fields(d,x,fields,8,fields,8,e) || !text(d,x,"numa_policy",&policy,e)) return false;
  c->policy=!strcmp(policy,"sharded") ? RIA_NUMA_SHARDED : !strcmp(policy,"replicated_experts") ?
      RIA_NUMA_REPLICATED_EXPERTS : !strcmp(policy,"replicated_server_model") ? RIA_NUMA_REPLICATED_SERVER_MODEL : 0;
  if (!c->policy || !integer(d,x,"projection_tile_rows",&c->projection_tile_rows,e) ||
      !integer(d,x,"drain_timeout_ms",&c->drain_timeout_ms,e) || !c->projection_tile_rows || c->projection_tile_rows>64 ||
      !c->drain_timeout_ms || !ria_json_u64(d,ria_json_get(d,x,"host_runtime_bytes"),true,&c->host_runtime_bytes,e) ||
      !ria_json_u64(d,ria_json_get(d,x,"startup_host_bytes"),true,&c->startup_host_bytes,e) ||
      !ria_json_u64(d,ria_json_get(d,x,"device_workspace_bytes"),true,&c->device_workspace_bytes,e) ||
      !ria_json_u64(d,ria_json_get(d,x,"pinned_workspace_bytes"),true,&c->pinned_workspace_bytes,e) ||
      !c->host_runtime_bytes || c->host_runtime_bytes>c->startup_host_bytes || c->startup_host_bytes>host_cap ||
      c->device_workspace_bytes>device_cap || c->pinned_workspace_bytes>pinned_cap)
    return ria_fail(e,RIA_RESOURCE_LIMIT,"expert reservations exceed admitted caps");
  const ria_json_node *nodes=ria_json_at(d,ria_json_get(d,x,"nodes"));
  if (!nodes || nodes->type!=RIA_JSON_ARRAY) return ria_fail(e,RIA_INVALID_REQUEST,"expert NUMA nodes missing");
  const char *const nf[]={"node","cpus","workers","local_bytes"};
  for (uint32_t i=nodes->child;i!=RIA_JSON_NONE;i=d->nodes[i].next) {
    uint64_t id,workers; if (c->node_count==64) return ria_fail(e,RIA_RESOURCE_LIMIT,"too many expert NUMA nodes");
    ria_expert_node *n=&c->nodes[c->node_count++];
    if (!ria_json_fields(d,i,nf,4,nf,4,e) || !integer(d,i,"node",&id,e) || id>63 ||
        !integer(d,i,"workers",&workers,e) || !workers || workers>64 ||
        !ria_json_u64(d,ria_json_get(d,i,"local_bytes"),true,&n->local_bytes,e) || !n->local_bytes) return false;
    n->node=(unsigned)id; n->workers=(unsigned)workers; c->worker_count+=n->workers;
    const ria_json_node *cpus=ria_json_at(d,ria_json_get(d,i,"cpus"));
    if (!cpus || cpus->type!=RIA_JSON_ARRAY) return ria_fail(e,RIA_INVALID_REQUEST,"explicit expert CPU affinities missing");
    for (uint32_t j=cpus->child;j!=RIA_JSON_NONE;j=d->nodes[j].next) {
      uint64_t cpu;
      if (n->cpu_count==256 || !ria_json_u64(d,j,false,&cpu,e) || cpu>=65536) return ria_fail(e,RIA_RESOURCE_LIMIT,"CPU affinity bound exceeded");
      n->cpus[n->cpu_count++]=(unsigned)cpu;
      for (unsigned k=0;k<c->node_count;k++) for (unsigned l=0;l<c->nodes[k].cpu_count;l++)
        if ((k!=c->node_count-1 || l!=n->cpu_count-1) && c->nodes[k].cpus[l]==cpu)
          return ria_fail(e,RIA_INVALID_REQUEST,"duplicate expert CPU assignment");
    }
    if (n->workers>n->cpu_count) return ria_fail(e,RIA_RESOURCE_LIMIT,"workers exceed explicit local CPUs");
    for (unsigned j=0;j+1<c->node_count;j++) if (c->nodes[j].node==n->node)
      return ria_fail(e,RIA_INVALID_REQUEST,"duplicate expert NUMA node");
  }
  if (!c->node_count || c->worker_count>128 ||
      (!strcmp(executor,"cpu") && (c->device_workspace_bytes || c->pinned_workspace_bytes)) ||
      (!strcmp(executor,"cuda") && (c->worker_count!=1 || !c->device_workspace_bytes || !c->pinned_workspace_bytes)))
    return ria_fail(e,RIA_INVALID_REQUEST,"executor/node/worker reservation mismatch");
  return true;
}
static bool expert_config(ria_service *s,ria_error *e) {
  ria_expert_config *c=&s->expert;
  if (!ria_expert_config_parse(&s->document,ria_json_get(&s->document,0,"expert"),s->executor,
                               s->host_cap,s->device_cap,s->pinned_cap,c,e)) return false;
  const ria_json_node *caps=ria_json_at(&s->plan,ria_json_get(&s->plan,ria_json_get(&s->plan,0,"caps"),"numa"));
  uint32_t startup=ria_json_get(&s->plan,ria_json_get(&s->plan,0,"phases"),"startup");
  uint64_t startup_host,startup_device,startup_pinned;
  if (!integer(&s->plan,startup,"host_bytes",&startup_host,e) || !integer(&s->plan,startup,"device_bytes",&startup_device,e) ||
      !integer(&s->plan,startup,"pinned_bytes",&startup_pinned,e) || c->startup_host_bytes>startup_host ||
      c->device_workspace_bytes>startup_device || c->pinned_workspace_bytes>startup_pinned)
    return ria_fail(e,RIA_RESOURCE_LIMIT,"expert physical reservations exceed admitted startup phase peak");
  const ria_json_node *startup_nodes=ria_json_at(&s->plan,ria_json_get(&s->plan,startup,"numa"));
  for (unsigned i=0;i<c->node_count;i++) {
    const ria_expert_node *n=&c->nodes[i];
    const ria_json_node *sets[]={caps,startup_nodes};
    for (unsigned k=0;k<2;k++) {
      bool found=false;
      for (uint32_t j=sets[k] ? sets[k]->child : RIA_JSON_NONE;j!=RIA_JSON_NONE;j=s->plan.nodes[j].next) {
        uint64_t node,bytes;
        if (!integer(&s->plan,j,"node",&node,e) || !integer(&s->plan,j,"bytes",&bytes,e)) return false;
        if (node==n->node && n->local_bytes<=bytes) found=true;
      }
      if (!found) return ria_fail(e,RIA_RESOURCE_LIMIT,"expert local reserve exceeds admitted node peak/capacity");
    }
  }
  return true;
}
bool ria_service_read(ria_service *s, const char *path, ria_error *e) {
  memset(s, 0, sizeof(*s));
  if (!ria_disable_dumps(e))
    return false;
  const char *lock_path, *plan_path;
  if (!ria_json_read(path, (ria_json_limits){256u << 10, 32768, 48},
                     &s->document, e))
    goto fail;
  ria_json_doc *d = &s->document;
  const char *const fields[] = {"schema_revision", "role",
                                "executor",        "model_manifest",
                                "memory_plan",     "deployment_lock",
                                "device_index",    "management_socket",
                                "artifacts_dir",   "tls",
                                "network",         "api",
                                "peer_grants",     "placement_plan", "expert"};
  const char *const required[] = {"schema_revision", "role",
                                  "executor",        "model_manifest",
                                  "memory_plan",     "deployment_lock",
                                  "device_index",    "management_socket",
                                  "artifacts_dir",   "tls",
                                  "network"};
  uint64_t revision;
  if (!ria_json_fields(d, 0, fields, 15, required, 11, e) ||
      !integer(d, 0, "schema_revision", &revision, e) || revision != 1 ||
      !text(d, 0, "role", &s->role, e) ||
      !text(d, 0, "executor", &s->executor, e) ||
      !text(d, 0, "model_manifest", &s->manifest_path, e) ||
      !text(d, 0, "memory_plan", &plan_path, e) ||
      !text(d, 0, "deployment_lock", &lock_path, e) ||
      !text(d, 0, "management_socket", &s->admin_socket, e) ||
      (strcmp(s->role, "client") && strcmp(s->role, "expert")) ||
      (strcmp(s->executor, "cpu") && strcmp(s->executor, "cuda")))
    goto invalid;
  if (s->manifest_path[0] != '/' || plan_path[0] != '/' ||
      lock_path[0] != '/' || s->admin_socket[0] != '/' ||
      (!strcmp(s->role, "client") && strcmp(s->executor, "cuda")))
    goto invalid;
  if (!checked_doc(lock_path, NULL, &s->lock, e))
    goto fail;
  uint8_t service_digest[32], expected[32];
  if (!digest(&s->lock, 0, "service_digest", expected, e) ||
      !ria_json_sha256(d, true, service_digest, e) ||
      memcmp(expected, service_digest, 32))
    goto integrity;
  if (!digest(&s->lock, 0, "memory_plan_digest", expected, e) ||
      !checked_doc(plan_path, expected, &s->plan, e) ||
      !digest(&s->lock, 0, "model_manifest_digest", s->manifest_digest, e) ||
      !digest(&s->lock, 0, "logical_model_digest", s->logical_model_digest,
              e) ||
      !digest(&s->lock, 0, "operator_contract_digest",
              s->operator_contract_digest, e) ||
      !text(&s->plan, 0, "profile", &s->profile, e) ||
      !integer(&s->plan, 0, "context_positions", &s->context_positions, e))
    goto fail;
  const ria_json_node *admitted =
      ria_json_at(&s->plan, ria_json_get(&s->plan, 0, "admitted"));
  const char *plan_role, *plan_executor;
  if (!admitted || admitted->type != RIA_JSON_BOOL || !admitted->boolean ||
      !text(&s->plan, 0, "role", &plan_role, e) || strcmp(plan_role, s->role) ||
      !text(&s->plan, 0, "executor", &plan_executor, e) ||
      strcmp(plan_executor, s->executor))
    goto invalid;
  const char *const provenance[] = {"environment_digest", "build_digest",
                                     "policy_digest", "logical_model_digest",
                                     "operator_contract_digest"};
  for (unsigned i = 0; i < sizeof(provenance) / sizeof(*provenance); i++) {
    uint8_t locked[32], planned[32];
    if (!digest(&s->lock, 0, provenance[i], locked, e) ||
        !digest(&s->plan, 0, provenance[i], planned, e))
      goto fail;
    if (memcmp(locked, planned, sizeof(locked)))
      goto integrity;
  }
  uint32_t caps = ria_json_get(&s->plan, 0, "caps");
  if (!integer(&s->plan, caps, "host_bytes", &s->host_cap, e) ||
      !integer(&s->plan, caps, "device_bytes", &s->device_cap, e) ||
      !integer(&s->plan, caps, "pinned_bytes", &s->pinned_cap, e) ||
      !s->host_cap)
    goto fail;
  uint32_t tls = ria_json_get(d, 0, "tls");
  const char *const tf[] = {"ca_file",          "certificate_file",
                            "private_key_file", "expected_peer_name",
                            "minimum_version",  "early_data"};
  const char *version;
  if (!ria_json_fields(d, tls, tf, 6, tf, 6, e) ||
      !text(d, tls, "ca_file", &s->tls.ca_file, e) ||
      !text(d, tls, "certificate_file", &s->tls.certificate_file, e) ||
      !text(d, tls, "private_key_file", &s->tls.private_key_file, e) ||
      !text(d, tls, "expected_peer_name", &s->tls.expected_peer_name, e) ||
      !text(d, tls, "minimum_version", &version, e) ||
      strcmp(version, "TLS1.3"))
    goto invalid;
  const ria_json_node *early =
      ria_json_at(d, ria_json_get(d, tls, "early_data"));
  if (!early || early->type != RIA_JSON_BOOL || early->boolean)
    goto invalid;
  s->tls.server = !strcmp(s->role, "expert");
  uint32_t network = ria_json_get(d, 0, "network");
  const char *const nf[] = {"control_address",
                            "bulk_address",
                            "connect_timeout_ms",
                            "handshake_timeout_ms",
                            "operation_timeout_ms",
                            "frame_io_timeout_ms",
                            "write_timeout_ms",
                            "max_row_lookup_rows",
                            "max_inflight_payload_bytes",
                            "max_frame_payload_bytes",
                            "max_bulk_data_bytes",
                            "max_inflight_expert_requests",
                            "server_executor"};
  if (!ria_json_fields(d, network, nf, 13, nf, 13, e) ||
      !text(d, network, "control_address", &s->control_address, e) ||
      !text(d, network, "bulk_address", &s->bulk_address, e) ||
      !integer(d, network, "connect_timeout_ms", &s->connect_timeout_ms, e) ||
      !text(d, network, "server_executor", &s->server_executor, e) ||
      !integer(d, network, "handshake_timeout_ms", &s->tls.handshake_timeout_ms,
               e) ||
      !integer(d, network, "operation_timeout_ms",
               &s->limits.operation_timeout_ms, e) ||
      !integer(d, network, "frame_io_timeout_ms",
               &s->limits.frame_io_timeout_ms, e) ||
      !integer(d, network, "write_timeout_ms", &s->limits.write_timeout_ms,
               e) ||
      !integer(d, network, "max_row_lookup_rows", &s->limits.row_lookup_rows,
               e) ||
      !integer(d, network, "max_inflight_payload_bytes",
               &s->limits.inflight_payload_bytes, e) ||
      !integer(d, network, "max_frame_payload_bytes",
               &s->limits.frame_payload_bytes, e) ||
      !integer(d, network, "max_bulk_data_bytes", &s->limits.bulk_data_bytes,
               e) ||
      !integer(d, network, "max_inflight_expert_requests",
               &s->limits.expert_requests, e))
    goto fail;
  s->limits.expert_rows = 64;
  if (!s->connect_timeout_ms || !s->tls.handshake_timeout_ms ||
      !ria_limits_validate(&s->limits, e) ||
      (strcmp(s->server_executor, "cpu") && strcmp(s->server_executor, "cuda")))
    goto invalid;
  if (!strcmp(s->role, "expert")) {
    if (strcmp(s->server_executor, s->executor))
      goto invalid;
    if (!text(d, 0, "peer_grants", &s->peer_grants, e) ||
        s->peer_grants[0] != '/' ||
        !digest(&s->lock, 0, "peer_grants_digest", expected, e) ||
        !checked_doc(s->peer_grants, expected, &s->grants, e))
      goto fail;
  } else {
    const char *placement_path;
    if (!text(d, 0, "placement_plan", &placement_path, e) ||
        placement_path[0] != '/' ||
        !digest(&s->lock, 0, "placement_plan_digest", expected, e) ||
        !checked_doc(placement_path, expected, &s->placement, e))
      goto fail;
  }
  if (!strcmp(s->executor, "cuda")) {
    if (!text(&s->lock, 0, "gpu_uuid", &s->gpu_uuid, e))
      goto fail;
    uint64_t device;
    if (!integer(d, 0, "device_index", &device, e) || device != 0)
      goto invalid;
  } else {
    const ria_json_node *device =
        ria_json_at(d, ria_json_get(d, 0, "device_index"));
    if (!device || device->type != RIA_JSON_NULL || s->device_cap ||
        s->pinned_cap)
      goto invalid;
  }
  if (!strcmp(s->role,"expert") && !expert_config(s,e)) goto fail;
  if (!strcmp(s->role,"client") && ria_json_get(d,0,"expert")!=RIA_JSON_NONE) goto invalid;
  return true;
invalid:
  ria_error_set(e, RIA_INVALID_REQUEST,
                "service violates supported role/security/budget contract");
  goto fail;
integrity:
  ria_error_set(e, RIA_INTEGRITY_ERROR,
                "service does not match immutable deployment lock");
fail:
  ria_service_free(s);
  return false;
}
void ria_service_free(ria_service *s) {
  ria_json_free(&s->placement);
  ria_json_free(&s->grants);
  ria_json_free(&s->plan);
  ria_json_free(&s->lock);
  ria_json_free(&s->document);
  memset(s, 0, sizeof(*s));
}
bool ria_service_grant(const ria_service *s, const ria_tensor_store *store,
                       const ria_json_doc *bind, ria_error *e) {
  const ria_json_node *grants =
      ria_json_at(&s->grants, ria_json_get(&s->grants, 0, "grants"));
  if (!grants || grants->type != RIA_JSON_ARRAY)
    return ria_fail(e, RIA_UNAUTHORIZED, "no provisioned peer grants");
  const char *keys[] = {"logical_model_digest",  "operator_contract_digest",
                        "encoding_digest",       "client_layout_digest",
                        "placement_plan_digest", "profile",
                        "server_executor"};
  uint8_t layout[32];
  if (!digest(&store->manifest, 0, "layout_digest", layout, e))
    return false;
  for (uint32_t i = grants->child; i != RIA_JSON_NONE;
       i = s->grants.nodes[i].next) {
    const char *peer;
    bool match = text(&s->grants, i, "expected_peer_name", &peer, e) &&
                 !strcmp(peer, s->tls.expected_peer_name);
    for (unsigned k = 0; match && k < 7; k++) {
      const char *want, *actual;
      match = text(&s->grants, i, keys[k], &want, e) &&
              text(bind, 0, keys[k], &actual, e) && !strcmp(want, actual);
    }
    uint8_t want[32];
    if (match && digest(&s->grants, i, "server_layout_digest", want, e) &&
        !memcmp(want, layout, 32))
      return true;
  }
  return ria_fail(
      e, RIA_UNAUTHORIZED,
      "binding has no exact authorized identity/layout/placement grant");
}
bool ria_address(const char *text_value, bool listen,
                 struct sockaddr_storage *out, socklen_t *length,
                 ria_error *e) {
  size_t n = strlen(text_value);
  if (!n || n >= 512)
    return ria_fail(e, RIA_INVALID_REQUEST, "invalid address length");
  char host[512], service[6];
  const char *port;
  if (text_value[0] == '[') {
    const char *end = strchr(text_value, ']');
    if (!end || end[1] != ':')
      return ria_fail(e, RIA_INVALID_REQUEST, "invalid IPv6 address");
    size_t size = (size_t)(end - text_value) - 1;
    memcpy(host, text_value + 1, size);
    host[size] = 0;
    port = end + 2;
  } else {
    port = strrchr(text_value, ':');
    if (!port || port == text_value)
      return ria_fail(e, RIA_INVALID_REQUEST, "address requires host:port");
    size_t size = (size_t)(port - text_value);
    memcpy(host, text_value, size);
    host[size] = 0;
    ++port;
  }
  uint64_t value;
  if (!ria_parse_u64(port, strlen(port), &value, e) || !value ||
      value > 65535 || strlen(port) > 5)
    return ria_fail(e, RIA_INVALID_REQUEST, "invalid TCP port");
  memcpy(service, port, strlen(port) + 1);
  struct addrinfo hints = {0}, *addresses = NULL;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_family = text_value[0]=='[' ? AF_INET6 : AF_INET;
  hints.ai_flags = AI_NUMERICHOST | AI_NUMERICSERV | (listen ? AI_PASSIVE : 0);
  int status = getaddrinfo(host, service, &hints, &addresses);
  if (status || !addresses)
    return ria_fail(e, RIA_INVALID_REQUEST,
                    "configured peer address must be a numeric IPv4/[IPv6] literal");
  bool ok = addresses->ai_addrlen <= sizeof(*out);
  if (ok) {
    memset(out, 0, sizeof(*out));
    memcpy(out, addresses->ai_addr, addresses->ai_addrlen);
    *length = (socklen_t)addresses->ai_addrlen;
  }
  freeaddrinfo(addresses);
  return ok ||
         ria_fail(e, RIA_INVALID_REQUEST, "unsupported socket address size");
}
static bool ready(int fd, short events, uint64_t deadline, ria_error *e) {
  for (;;) {
    uint64_t now = ria_monotonic_ms();
    if (now >= deadline)
      return ria_fail(e, RIA_DEADLINE_EXCEEDED, "admin operation deadline");
    uint64_t remaining = deadline - now;
    int timeout = remaining > INT_MAX ? INT_MAX : (int)remaining;
    struct pollfd p = {fd, events, 0};
    int status = poll(&p, 1, timeout);
    if (status < 0 && errno == EINTR)
      continue;
    if (status <= 0)
      return ria_fail(e, status ? RIA_INTERNAL_ERROR : RIA_DEADLINE_EXCEEDED,
                      "admin poll failed/deadline");
    if (p.revents & events)
      return true;
    return ria_fail(e, RIA_NOT_READY, "admin endpoint disconnected");
  }
}
bool ria_admin_request(const char *path, const char *command, uint64_t timeout,
                       char **reply, size_t *length, ria_error *e) {
  *reply = NULL;
  *length = 0;
  if ((strcmp(command, "health") && strcmp(command, "drain")) || !timeout ||
      strlen(path) >= sizeof(((struct sockaddr_un *)0)->sun_path))
    return ria_fail(e, RIA_INVALID_REQUEST,
                    "invalid admin command/path/deadline");
  uint64_t deadline;
  if (!ria_u64_add(ria_monotonic_ms(), timeout, &deadline))
    return ria_fail(e, RIA_INVALID_REQUEST, "admin deadline overflow");
  int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
  if (fd < 0)
    return ria_fail(e, RIA_NOT_READY, "admin socket creation failed");
  struct sockaddr_un address = {0};
  address.sun_family = AF_UNIX;
  memcpy(address.sun_path, path, strlen(path) + 1);
  bool ok = true;
  if (connect(fd, (struct sockaddr *)&address, sizeof(address)) &&
      errno != EINPROGRESS)
    ok = ria_fail(e, RIA_NOT_READY, "admin endpoint unavailable");
  if (ok)
    ok = ready(fd, POLLOUT, deadline, e);
  int status = 0;
  socklen_t status_bytes = sizeof(status);
  if (ok &&
      (getsockopt(fd, SOL_SOCKET, SO_ERROR, &status, &status_bytes) || status))
    ok = ria_fail(e, RIA_NOT_READY, "admin connect failed");
  const char *request = !strcmp(command, "health")
                            ? "{\"command\":\"health\",\"schema_revision\":1}\n"
                            : "{\"command\":\"drain\",\"schema_revision\":1}\n";
  size_t sent = 0, request_length = strlen(request);
  while (ok && sent < request_length) {
    if (!ready(fd, POLLOUT, deadline, e)) {
      ok = false;
      break;
    }
    ssize_t n = send(fd, request + sent, request_length - sent, MSG_NOSIGNAL);
    if (n < 0 && (errno == EAGAIN || errno == EINTR))
      continue;
    if (n <= 0) {
      ok = ria_fail(e, RIA_NOT_READY, "admin request write failed");
      break;
    }
    sent += (size_t)n;
  }
  char *buffer = malloc(RIA_ERROR_MAX + 1);
  if (!buffer)
    ok = ria_fail(e, RIA_RESOURCE_LIMIT, "admin reply allocation failed");
  size_t used = 0;
  while (ok) {
    if (!ready(fd, POLLIN, deadline, e)) {
      ok = false;
      break;
    }
    ssize_t n = recv(fd, buffer + used, RIA_ERROR_MAX - used, 0);
    if (n < 0 && (errno == EINTR || errno == EAGAIN))
      continue;
    if (n <= 0) {
      ok = ria_fail(e, RIA_NOT_READY, "admin reply incomplete");
      break;
    }
    used += (size_t)n;
    if (memchr(buffer, '\n', used))
      break;
    if (used == RIA_ERROR_MAX) {
      ok = ria_fail(e, RIA_RESOURCE_LIMIT, "admin reply too large");
      break;
    }
  }
  if (close(fd) && ok)
    ok = ria_fail(e, RIA_INTERNAL_ERROR, "admin socket close failed");
  if (ok) {
    ria_json_doc doc = {0};
    ok = ria_json_parse(buffer, used,
                        (ria_json_limits){RIA_ERROR_MAX, 8192, 16}, &doc, e);
    ria_json_free(&doc);
  }
  if (!ok) {
    free(buffer);
    return false;
  }
  buffer[used] = 0;
  *reply = buffer;
  *length = used;
  return true;
}
