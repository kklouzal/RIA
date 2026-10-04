#define _POSIX_C_SOURCE 200809L
#include "inventory.h"
#include "numa_policy.h"
#include "probe.h"
#include "prefill.h"
#include "remote.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { char bytes[65536]; size_t used; unsigned count; ria_error *error; } output;
static bool emit(output *o,const char *format,...) RIA_PRINTF(2,3);
static bool emit(output *o,const char *format,...) {
  va_list a;va_start(a,format);int n=vsnprintf(o->bytes+o->used,sizeof(o->bytes)-o->used,format,a);va_end(a);
  if (n<0 || (size_t)n>=sizeof(o->bytes)-o->used) return ria_fail(o->error,RIA_RESOURCE_LIMIT,"inventory output bound exceeded");
  o->used+=(size_t)n;return true;
}
static bool integer(const ria_json_doc *d,uint32_t p,const char *key,bool wide,uint64_t *v,ria_error *e) {
  return ria_json_u64(d,ria_json_get(d,p,key),wide,v,e);
}
static bool text(const ria_json_doc *d,uint32_t p,const char *key,const char **s,ria_error *e) {
  size_t n;return ria_json_string(d,ria_json_get(d,p,key),s,&n,e) &&
    (strlen(*s)==n || ria_fail(e,RIA_INVALID_REQUEST,"embedded NUL in inventory field"));
}
static bool allocation(output *o,const char *name,uint64_t bytes,int node,bool device,bool pinned,bool progress) {
  if (!bytes) return true;
  if (bytes>RIA_JSON_SAFE_INTEGER) return ria_fail(o->error,RIA_RESOURCE_LIMIT,"inventory bytes exceed JSON range");
  unsigned id=++o->count;
  if (!emit(o,"%s{\"id\":\"%u\",\"name\":\"%s\",\"resource\":\"%s\",\"base_bytes\":%llu,\"bytes_per_position\":0,\"numa_node\":",
             id>1 ? "," : "",id,name,device ? "device" : "host",(unsigned long long)bytes)) return false;
  if (!(node<0 ? emit(o,"null") : emit(o,"%d",node))) return false;
  return emit(o,",\"pinned\":%s,\"protected_progress\":%s,\"phases\":[\"startup\",\"prefill\",\"decode\",\"continuation\",\"image\",\"drain\"]}",
               pinned ? "true" : "false",progress ? "true" : "false");
}
bool ria_inventory_files(const char *manifest,const char *request,const char *destination,ria_error *e) {
  ria_json_doc d={0},result={0};ria_tensor_store store={0};output *o=NULL;bool ok=false;
  const char *const fields[]={"schema_revision","role","executor","profile","manifest_digest","context_positions","prefill_rows",
    "max_metadata_bytes","host_cap","device_cap","pinned_cap","host_runtime_bytes","device_runtime_bytes","pinned_runtime_bytes","expert","runtime_policy_digest",
    "graph_host_state_bytes","projection_tile_rows","frontend_host_bytes","network"};
  uint64_t rev,context,rows,projection,graph_host,frontend,metadata,host_cap,device_cap,pinned_cap,host,device,pinned;
  uint8_t expected[32],runtime[32],request_hash[32];const char *role,*executor,*profile;
  if (!ria_json_read(request,(ria_json_limits){256u<<10,32768,32},&d,e) ||
      !ria_json_fields(&d,0,fields,20,fields,20,e) || !integer(&d,0,"schema_revision",false,&rev,e) || rev!=1 ||
      !text(&d,0,"role",&role,e) || !text(&d,0,"executor",&executor,e) || !text(&d,0,"profile",&profile,e) ||
      !integer(&d,0,"context_positions",false,&context,e) || !context || context>1048576 ||
      !integer(&d,0,"prefill_rows",false,&rows,e) || !rows || rows>64 || rows>context ||
      !integer(&d,0,"projection_tile_rows",false,&projection,e) ||
      !integer(&d,0,"graph_host_state_bytes",true,&graph_host,e) ||
      !integer(&d,0,"frontend_host_bytes",true,&frontend,e) ||
      !integer(&d,0,"max_metadata_bytes",false,&metadata,e) || !metadata ||
      !integer(&d,0,"host_cap",false,&host_cap,e) || !host_cap || metadata>host_cap ||
      !integer(&d,0,"device_cap",false,&device_cap,e) || !integer(&d,0,"pinned_cap",false,&pinned_cap,e) ||
      !integer(&d,0,"host_runtime_bytes",true,&host,e) || !host ||
      !integer(&d,0,"device_runtime_bytes",true,&device,e) || !integer(&d,0,"pinned_runtime_bytes",true,&pinned,e) ||
      host>host_cap || graph_host>host || frontend>host || device>device_cap || pinned>pinned_cap || pinned>host ||
      !ria_json_digest_field(&d,ria_json_get(&d,0,"manifest_digest"),expected,e) ||
      !ria_json_digest_field(&d,ria_json_get(&d,0,"runtime_policy_digest"),runtime,e) ||
      !ria_json_sha256(&d,true,request_hash,e)) goto done;
  bool client=!strcmp(role,"client"),cpu=!strcmp(executor,"cpu");
  if ((!client && strcmp(role,"expert")) || (strcmp(executor,"cpu") && strcmp(executor,"cuda")) ||
      (client ? !projection || projection>4096 || !graph_host || !frontend : projection || graph_host || frontend) ||
      (client && cpu) || (cpu && (device || pinned || device_cap || pinned_cap)) ||
      (!cpu && (!device || !pinned))) { ria_error_set(e,RIA_INVALID_REQUEST,"inventory role/executor reservations invalid");goto done; }
  ria_service network={0};
  if (!ria_service_network_parse(&d,ria_json_get(&d,0,"network"),&network,e)) goto done;
  if (client) {
    uint64_t batch_host,batch_device,batch_pinned,remote_host,owners;
    if (!ria_graph_prefill_required_bytes((uint32_t)rows,(uint32_t)projection,&batch_host,&batch_device,&batch_pinned,e) ||
        !ria_remote_host_required_bytes((unsigned)rows,&network.limits,&remote_host,e) ||
        !ria_u64_add(graph_host,frontend,&owners) || !ria_u64_add(owners,pinned,&owners) || owners>host ||
        batch_host>graph_host || batch_device>device || batch_pinned>pinned || remote_host>frontend) {
      if (!e || !e->code) ria_error_set(e,RIA_RESOURCE_LIMIT,"prefill workspace exceeds client host/device/pinned reservation");
      goto done;
    }
  }
  ria_tensor_load_options load={.role=client ? "client" : "server",.expected_digest=expected,
    .max_resident_bytes=host_cap,.max_metadata_bytes=metadata,.numa_node=-1};
  if (!ria_tensor_store_inspect(&store,manifest,&load,e) || strcmp(profile,store.profile)) {
    if (!e || !e->code) ria_error_set(e,RIA_IDENTITY_MISMATCH,"inventory profile differs from manifest");
    goto done;
  }
  uint64_t owned;
  if (!ria_tensor_owned_bytes(&store,&owned,e)) goto done;
  o=calloc(1,sizeof(*o));if (!o) { ria_error_set(e,RIA_RESOURCE_LIMIT,"inventory output allocation failed");goto done; }o->error=e;
  char logical[65],op[65],manifest_hash[65],policy[65],request_digest[65];
  ria_hex_encode(store.logical_model_digest,32,logical);ria_hex_encode(store.operator_contract_digest,32,op);
  ria_hex_encode(expected,32,manifest_hash);ria_hex_encode(runtime,32,policy);ria_hex_encode(request_hash,32,request_digest);
  if (!emit(o,"{\"schema_revision\":1,\"logical_model_digest\":\"%s\",\"operator_contract_digest\":\"%s\",\"semantic_max_positions\":%llu,\"derivation\":{\"manifest_digest\":\"%s\",\"runtime_policy_digest\":\"%s\",\"request_digest\":\"%s\",\"context_positions\":%llu,\"prefill_rows\":%llu},\"allocations\":[",
            logical,op,(unsigned long long)context,manifest_hash,policy,request_digest,(unsigned long long)context,(unsigned long long)rows)) goto done;
  if (client) {
    const ria_json_node *expert=ria_json_at(&d,ria_json_get(&d,0,"expert"));uint64_t total;
    if (!expert || expert->type!=RIA_JSON_NULL || !ria_u64_add(owned,host,&total) || total>host_cap) {
      ria_error_set(e,RIA_RESOURCE_LIMIT,"client population and explicit runtime reservations exceed cap");goto done;
    }
    if (!allocation(o,"authenticated_population_and_metadata",owned,-1,false,false,false) ||
        !allocation(o,"client_runtime_reservation",host-pinned,-1,false,false,true) ||
        !allocation(o,"client_pinned_reservation",pinned,-1,false,true,true)) goto done;
  } else {
    ria_expert_config c;ria_numa_account a;uint64_t reserved=0;
    if (!ria_expert_config_parse(&d,ria_json_get(&d,0,"expert"),executor,(uint32_t)rows,host_cap,device_cap,pinned_cap,&c,e) ||
        c.host_runtime_bytes!=host || c.device_workspace_bytes!=device || c.pinned_workspace_bytes!=pinned ||
        !ria_numa_account_store(&store,&c,&a,e) || owned<a.canonical_bytes || owned-a.canonical_bytes>host) {
      if (!e || !e->code) ria_error_set(e,RIA_RESOURCE_LIMIT,"expert metadata/runtime reservations disagree");
      goto done;
    }
    uint64_t worker_total;
    if (!ria_u64_mul(c.worker_count,RIA_NUMA_WORKER_BYTES,&worker_total) || worker_total>host) goto overflow;
    for (unsigned i=0;i<c.node_count;i++) {
      uint64_t runtime_bytes=(uint64_t)c.nodes[i].workers*RIA_NUMA_WORKER_BYTES;
      if (!i && !ria_u64_add(runtime_bytes,host-worker_total,&runtime_bytes)) goto overflow;
      if (runtime_bytes>c.nodes[i].local_bytes || (!i && pinned>runtime_bytes) ||
          !ria_u64_add(reserved,c.nodes[i].local_bytes,&reserved)) goto overflow;
      char name[96];int n=snprintf(name,sizeof(name),"node_%u_population_replica_reservation",c.nodes[i].node);
      if (n<0 || (size_t)n>=sizeof(name) || !allocation(o,name,c.nodes[i].local_bytes-runtime_bytes,(int)c.nodes[i].node,false,false,false)) goto done;
      n=snprintf(name,sizeof(name),"node_%u_runtime_reservation",c.nodes[i].node);
      if (n<0 || (size_t)n>=sizeof(name) || !allocation(o,name,runtime_bytes-(!i ? pinned : 0),(int)c.nodes[i].node,false,false,!i)) goto done;
    }
    if (reserved>c.startup_host_bytes) { ria_error_set(e,RIA_RESOURCE_LIMIT,"NUMA reservations exceed global startup reserve");goto done; }
    if (!allocation(o,"expert_pinned_reservation",pinned,(int)c.nodes[0].node,false,true,true) ||
        !allocation(o,"expert_global_startup_headroom",c.startup_host_bytes-reserved,-1,false,false,false)) goto done;
  }
  if (!allocation(o,"device_runtime_reservation",device,-1,true,false,true) || !emit(o,"]}")) goto done;
  if (!ria_json_parse(o->bytes,o->used,(ria_json_limits){sizeof(o->bytes),8192,32},&result,e) ||
      !ria_json_sha256(&result,true,expected,e)) goto done;
  ria_hex_encode(expected,32,manifest_hash);o->used--;
  if (!emit(o,",\"digest\":\"%s\"}\n",manifest_hash)) goto done;
  ok=ria_report_write(destination,o->bytes,o->used,e);goto done;
overflow:ria_error_set(e,RIA_RESOURCE_LIMIT,"inventory reservation arithmetic overflow");
done:
  if (!ok && e && !e->code) ria_error_set(e,RIA_INVALID_REQUEST,"invalid metadata-only inventory request");
  free(o);ria_json_free(&result);ria_tensor_store_close(&store);ria_json_free(&d);return ok;
}
