#define _GNU_SOURCE
#include "ria/server.h"
#include "ria/numa_policy.h"
#include "ria/remote.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr,"server contract %s:%d: %s\n",__FILE__,__LINE__,#x);abort(); } } while (0)

static void accounting(void) {
  ria_error e={0}; ria_expert_config c={0};
  c.policy=RIA_NUMA_SHARDED; c.node_count=2; c.worker_count=2;
  c.projection_tile_rows=64; c.host_runtime_bytes=4u<<20;
  c.startup_host_bytes=16u<<20; c.drain_timeout_ms=1000;
  for (unsigned i=0;i<2;i++) {
    c.nodes[i].node=i; c.nodes[i].workers=c.nodes[i].cpu_count=1;
    c.nodes[i].cpus[0]=i; c.nodes[i].local_bytes=8u<<20;
  }
  ria_shard shard={.id=1,.length=20480,.data_start=4096,.chunk_size=4096,.chunk_count=5};
  ria_tensor tensors[5]={
    {.id=1,.shard=1,.offset=0,.length=4096,.name="layers.0.ffn.experts.0.w1.weight",.placement="server"},
    {.id=2,.shard=1,.offset=4096,.length=4096,.name="layers.0.ffn.experts.1.w1.weight",.placement="server"},
    {.id=3,.shard=1,.offset=8192,.length=4096,.name="layers.1.engram.embed.weight",.placement="server"},
    {.id=4,.shard=1,.offset=12288,.length=4096,.name="embed.weight",.placement="client"},
    {.id=5,.shard=1,.offset=4096,.length=4096,.name="layers.0.ffn.experts.3.w1.weight",.placement="server",.alias=true,.alias_of=2}
  };
  ria_tensor_store s={.tensors=tensors,.tensor_count=5,.shards=&shard,.shard_count=1};
  ria_numa_account a;
  CHECK(ria_numa_account_store(&s,&c,&a,&e));
  CHECK(a.canonical_bytes==20480 && a.replica_bytes==0 && a.total_bytes==20480+(4u<<20));
  CHECK(a.node_bytes[0]==16384+(2u<<20) && a.node_bytes[1]==4096+(2u<<20));
  CHECK(ria_numa_owner(&c,1,1)==1 && ria_numa_owner(&c,41,0)==0);
  c.policy=RIA_NUMA_REPLICATED_EXPERTS;
  CHECK(ria_numa_account_store(&s,&c,&a,&e));
  CHECK(a.replica_bytes==8192 && a.node_bytes[0]==20480+(2u<<20) && a.node_bytes[1]==8192+(2u<<20));
  c.policy=RIA_NUMA_REPLICATED_SERVER_MODEL;
  CHECK(ria_numa_account_store(&s,&c,&a,&e));
  CHECK(a.replica_bytes==20480 && a.total_bytes==40960+(4u<<20));
  c.policy=RIA_NUMA_SHARDED;
  tensors[4].name="layers.0.ffn.experts.2.w1.weight";
  CHECK(!ria_numa_account_store(&s,&c,&a,&e)); /* tied pages cannot acquire conflicting owners */
  tensors[4].name="layers.0.ffn.experts.3.w1.weight";
  shard.data_start=64;
  CHECK(!ria_numa_account_store(&s,&c,&a,&e));
  shard.data_start=4096;
  c.nodes[1].local_bytes=4096;
  CHECK(!ria_numa_account_store(&s,&c,&a,&e));
  c.nodes[1].local_bytes=8u<<20; c.startup_host_bytes=1u<<20;
  CHECK(!ria_numa_account_store(&s,&c,&a,&e));
  c.startup_host_bytes=16u<<20; c.nodes[1].cpus[0]=0;
  CHECK(!ria_numa_config_validate(&c,&e));
  c.nodes[1].cpus[0]=1; c.host_runtime_bytes=1u<<20;
  CHECK(!ria_numa_account_store(&s,&c,&a,&e));
}
static void seccomp_contract(void) {
  ria_json_doc d={0}; ria_error e={0};
  CHECK(ria_json_read("deploy/seccomp-numa.json",(ria_json_limits){256u<<10,32768,24},&d,&e));
  const ria_json_node *calls=ria_json_at(&d,ria_json_get(&d,0,"syscalls"));
  bool bind=false,query=false;
  CHECK(calls && calls->type==RIA_JSON_ARRAY);
  for (uint32_t i=calls->child;i!=RIA_JSON_NONE;i=d.nodes[i].next) {
    const ria_json_node *names=ria_json_at(&d,ria_json_get(&d,i,"names"));
    bool is_bind=false,is_query=false;
    for (uint32_t j=names ? names->child : RIA_JSON_NONE;j!=RIA_JSON_NONE;j=d.nodes[j].next) {
      const char *name; size_t n;
      CHECK(ria_json_string(&d,j,&name,&n,&e));
      is_bind|=n==5 && !memcmp(name,"mbind",5);
      is_query|=n==10 && !memcmp(name,"move_pages",10);
    }
    if (!is_bind && !is_query) continue;
    const ria_json_node *args=ria_json_at(&d,ria_json_get(&d,i,"args"));
    for (uint32_t j=args ? args->child : RIA_JSON_NONE;j!=RIA_JSON_NONE;j=d.nodes[j].next) {
      uint64_t index,value; const char *op; size_t n;
      CHECK(ria_json_u64(&d,ria_json_get(&d,j,"index"),false,&index,&e));
      CHECK(ria_json_u64(&d,ria_json_get(&d,j,"value"),false,&value,&e));
      CHECK(ria_json_string(&d,ria_json_get(&d,j,"op"),&op,&n,&e));
      if (is_bind && index==2 && !strcmp(op,"SCMP_CMP_EQ")) { CHECK(value==RIA_NUMA_BIND_MODE); bind=true; }
      if (is_query && index==1 && !strcmp(op,"SCMP_CMP_LE")) { CHECK(value>=RIA_NUMA_QUERY_PAGES); query=true; }
    }
  }
  CHECK(bind && query && RIA_NUMA_QUERY_PAGES==64);
  ria_json_free(&d);
}
static void grants(void) {
  ria_error e={0}; ria_server_grants g;
  ria_shard shard={.id=7,.length=24576,.data_start=4096,.chunk_size=4096,.chunk_count=6};
  ria_tensor tensors[]={
    {.shard=7,.offset=0,.length=4096,.name="embed.weight",.placement="client"},
    {.shard=7,.offset=4096,.length=8192,.name="layers.1.engram.embed.weight",.placement="server"},
    {.shard=7,.offset=12288,.length=2048,.name="cache.weight",.placement="cache"},
    {.shard=7,.offset=14336,.length=2048,.name="__ria_padding_0",.placement="inactive"},
    {.shard=7,.offset=16384,.length=4096,.name="layers.0.ffn.experts.0.w1.weight",.placement="server"}
  };
  ria_tensor_store s={.tensors=tensors,.tensor_count=5,.shards=&shard,.shard_count=1};
  CHECK(ria_server_grants_open(&g,&s,&e));
  CHECK(ria_server_grants_chunk(&g,&shard,0,&e));
  CHECK(ria_server_grants_chunk(&g,&shard,1,&e));
  CHECK(!ria_server_grants_chunk(&g,&shard,2,&e));
  CHECK(!ria_server_grants_chunk(&g,&shard,3,&e));
  CHECK(ria_server_grants_chunk(&g,&shard,4,&e));
  CHECK(!ria_server_grants_chunk(&g,&shard,5,&e));
  CHECK(!ria_server_grants_chunk(&g,&shard,6,&e));
  shard.chunk_size=8192; shard.chunk_count=3;
  CHECK(!ria_server_grants_chunk(&g,&shard,2,&e)); /* authorized prefix cannot expose expert tail */
  ria_server_grants_close(&g);
}
typedef struct {
  ria_server_queue queue; pthread_mutex_t lock; pthread_cond_t changed;
  bool stop; unsigned counts[768]; unsigned consumed;
} queue_fixture;
typedef struct { queue_fixture *fixture; unsigned node; } consumer;
static void *consume(void *arg) {
  consumer *c=arg; queue_fixture *f=c->fixture;
  pthread_mutex_lock(&f->lock);
  for (;;) {
    ria_server_job job;
    while (!ria_server_queue_take(&f->queue,c->node,&job)) {
      if (f->stop) { pthread_mutex_unlock(&f->lock); return NULL; }
      pthread_cond_wait(&f->changed,&f->lock);
    }
    CHECK(job.node==c->node && job.entry<768);
    f->counts[job.entry]++; f->consumed++;
    pthread_cond_broadcast(&f->changed);
  }
}
static void queue_stress(void) {
  queue_fixture f={0}; ria_error e={0};
  CHECK(!pthread_mutex_init(&f.lock,NULL)); CHECK(!pthread_cond_init(&f.changed,NULL));
  for (unsigned i=0;i<768;i++) CHECK(ria_server_queue_push(&f.queue,(ria_server_job){&f,i%4,i/6,i},&e));
  CHECK(!ria_server_queue_push(&f.queue,(ria_server_job){&f,0,0,0},&e));
  pthread_t threads[4]; consumer c[4];
  for (unsigned i=0;i<4;i++) { c[i]=(consumer){&f,i}; CHECK(!pthread_create(&threads[i],NULL,consume,&c[i])); }
  pthread_mutex_lock(&f.lock);
  while (f.consumed<768) pthread_cond_wait(&f.changed,&f.lock);
  f.stop=true; pthread_cond_broadcast(&f.changed); pthread_mutex_unlock(&f.lock);
  for (unsigned i=0;i<4;i++) CHECK(!pthread_join(threads[i],NULL));
  for (unsigned i=0;i<768;i++) CHECK(f.counts[i]==1);
  CHECK(!f.queue.count);
  pthread_cond_destroy(&f.changed); pthread_mutex_destroy(&f.lock);
}
static void cancellation_progress(void) {
  ria_error e={0}; ria_limits l={.frame_payload_bytes=RIA_FRAME_MAX,.bulk_data_bytes=RIA_BULK_MAX,
    .expert_rows=64,.expert_requests=2,.row_lookup_rows=64,.inflight_payload_bytes=64u<<20,
    .operation_timeout_ms=1000,.frame_io_timeout_ms=1000,.write_timeout_ms=1000};
  ria_binding b; uint64_t control,row,bulk,charge;
  CHECK(ria_progress_charges(&l,&control,&row,&bulk,&e));
  ria_binding_init(&b,&l); CHECK(ria_binding_protect(&b,control,row,bulk,&e));
  uint8_t session[16]={1},capability[32]={2};
  CHECK(ria_binding_install(&b,session,1,capability,&e));
  CHECK(!ria_binding_bulk(&b,capability,false,&e));
  CHECK(ria_binding_bulk(&b,capability,true,&e));
  CHECK(!ria_binding_bulk(&b,capability,true,&e));
  ria_header h={.kind=RIA_EXPERT,.payload_length=256,.request_id=2,.epoch=1}; memcpy(h.session,session,16);
  CHECK(ria_request_charge(h.kind,h.payload_length,256,&charge,&e));
  CHECK(ria_binding_receive(&b,&h,false,&e)); CHECK(ria_binding_admit(&b,&h,charge,100,&e));
  uint64_t target_reserved=b.reserved_bytes; uint32_t state;
  CHECK(ria_binding_cancel(&b,2,&state,&e) && state==0 && b.reserved_bytes==target_reserved);
  CHECK(!ria_binding_terminal(&b,2,false,&e));
  h.kind=RIA_CANCEL; h.payload_length=16; h.request_id=3;
  CHECK(ria_binding_receive(&b,&h,false,&e)); CHECK(ria_binding_admit(&b,&h,168,101,&e));
  CHECK(ria_binding_terminal(&b,3,true,&e)); CHECK(b.reserved_bytes==target_reserved);
  h.kind=RIA_ROWS; h.request_id=4;
  CHECK(ria_binding_receive(&b,&h,false,&e)); CHECK(ria_binding_admit(&b,&h,1024,102,&e));
  h.kind=RIA_HEALTH; h.request_id=5;
  CHECK(ria_binding_receive(&b,&h,false,&e)); CHECK(ria_binding_admit(&b,&h,256,103,&e));
  h.kind=RIA_CHUNK; h.request_id=2;
  CHECK(ria_binding_receive(&b,&h,true,&e)); CHECK(ria_binding_admit(&b,&h,1024,104,&e));
  CHECK(ria_binding_terminal(&b,2,true,&e));
  CHECK(!ria_binding_terminal(&b,2,true,&e));
  CHECK(ria_binding_cancel(&b,2,&state,&e) && state==1);
  ria_binding_invalidate(&b); CHECK(!b.valid);
  CHECK(!ria_binding_receive(&b,&h,true,&e));
}
static void error_credit(void) {
  ria_error e={0}; uint64_t charge;
  /* Independent constants include the 16KiB error alternative for tiny work. */
  CHECK(ria_request_charge(RIA_ROWS,32,320,&charge,&e) && charge==16544);
  CHECK(ria_request_charge(RIA_CHUNK,16,65,&charge,&e) && charge==16528);
  CHECK(ria_request_charge(RIA_CANCEL,16,24,&charge,&e) && charge==16528);
  CHECK(ria_request_charge(RIA_CLOSE,0,0,&charge,&e) && charge==16512);
  CHECK(ria_request_charge(RIA_EXPERT,68,36,&charge,&e) && charge==8405188);
  CHECK(ria_request_charge(RIA_CHUNK,16,4194368,&charge,&e) && charge==4194512);
  CHECK(!ria_request_charge(RIA_CHUNK,UINT64_MAX,0,&charge,&e) && e.code==RIA_RESOURCE_LIMIT);
}
static void addresses(void) {
  struct sockaddr_storage address; socklen_t length; ria_error e={0};
  CHECK(ria_address("127.0.0.1:7443",true,&address,&length,&e));
  CHECK(ria_address("[::1]:7444",false,&address,&length,&e));
  CHECK(!ria_address("unresolvable.example:7443",false,&address,&length,&e));
  CHECK(!ria_address("::1:7443",false,&address,&length,&e));
  CHECK(!ria_address("127.0.0.1:65536",true,&address,&length,&e));
}
static void contribution(void) {
  ria_error e={0}; uint8_t offsets[12],entries[24]={0},inputs[16],result[56];
  ria_write_u32(offsets,0); ria_write_u32(offsets+4,2); ria_write_u32(offsets+8,3);
  ria_write_f32(entries+4,.5f); ria_write_f32(entries+12,.25f); ria_write_f32(entries+20,1.f);
  for (unsigned i=0;i<4;i++) ria_write_f32(inputs+i*4,(float)i);
  ria_expert_request r={.row_count=2,.input_width=2,.output_width=2,.entry_count=3,
    .offsets=offsets,.entries=entries,.inputs=inputs,.response_bytes=sizeof(result)};
  float input[2],output[2]={3.f,4.f},coefficient;
  memset(result,0x5a,sizeof(result));
  CHECK(ria_server_contribution_unpack(&r,0,1,input,2,&coefficient,&e));
  CHECK(input[0]==0 && input[1]==1 && coefficient==.25f);
  CHECK(!ria_server_contribution_unpack(&r,1,1,input,2,&coefficient,&e));
  CHECK(!ria_server_contribution_unpack(&r,0,0,input,1,&coefficient,&e));
  CHECK(ria_server_contribution_pack(&r,1,output,2,result,sizeof(result),&e));
  CHECK(ria_read_f32(result+40)==3.f && ria_read_f32(result+44)==4.f);
  for (unsigned i=0;i<sizeof(result);i++) if (i<40 || i>=48) CHECK(result[i]==0x5a);
  CHECK(!ria_server_contribution_pack(&r,3,output,2,result,sizeof(result),&e));
  CHECK(!ria_server_contribution_pack(&r,1,output,2,result,sizeof(result)-1,&e));
}
static ria_json_doc transport_document(const char *text) {
  ria_json_doc d={0};ria_error e={0};
  CHECK(ria_json_parse(text,strlen(text),(ria_json_limits){8192,512,16},&d,&e));return d;
}
static void transport_policy(void) {
  const char *valid[]={
    "{\"ca_file\":\"/ca.pem\",\"certificate_file\":\"/peer.pem\",\"private_key_file\":\"/peer.key\",\"expected_peer_name\":\"peer.test\",\"minimum_version\":\"TLS1.3\",\"early_data\":false}",
    "{\"enabled\":true,\"ca_file\":\"/ca.pem\",\"certificate_file\":\"/peer.pem\",\"private_key_file\":\"/peer.key\",\"expected_peer_name\":\"peer.test\",\"minimum_version\":\"TLS1.3\",\"early_data\":false}",
    "{\"enabled\":false}"
  };
  for (unsigned i=0;i<3;i++) {
    ria_json_doc d=transport_document(valid[i]);ria_error e={0};
    ria_tls_config config={.server=true,.handshake_timeout_ms=17};
    CHECK(ria_service_tls_parse(&d,0,&config,&e) && !e.code);
    CHECK(config.server && config.handshake_timeout_ms==17 && config.plaintext==(i==2));
    if (config.plaintext) CHECK(!config.ca_file && !config.certificate_file && !config.private_key_file && !config.expected_peer_name);
    else CHECK(!strcmp(config.ca_file,"/ca.pem") && !strcmp(config.certificate_file,"/peer.pem") &&
      !strcmp(config.private_key_file,"/peer.key") && !strcmp(config.expected_peer_name,"peer.test"));
    ria_json_free(&d);
  }
  const char *invalid[]={"{}","[]","null","{\"enabled\":null}","{\"enabled\":0}",
    "{\"enabled\":\"false\"}","{\"enabled\":true}",
    "{\"enabled\":false,\"ca_file\":\"/ca.pem\"}",
    "{\"enabled\":false,\"expected_peer_name\":null}",
    "{\"enabled\":false,\"unexpected\":false}",
    "{\"ca_file\":\"/ca.pem\",\"certificate_file\":\"/peer.pem\",\"private_key_file\":\"/peer.key\",\"expected_peer_name\":\"peer.test\",\"minimum_version\":\"TLS1.2\",\"early_data\":false}",
    "{\"ca_file\":\"/ca.pem\",\"certificate_file\":\"/peer.pem\",\"private_key_file\":\"/peer.key\",\"expected_peer_name\":\"peer.test\",\"minimum_version\":\"TLS1.3\",\"early_data\":true}"
  };
  for (unsigned i=0;i<sizeof invalid/sizeof *invalid;i++) {
    ria_json_doc d=transport_document(invalid[i]);ria_tls_config config={0};ria_error e={0};
    CHECK(!ria_service_tls_parse(&d,0,&config,&e) && e.code==RIA_INVALID_REQUEST);ria_json_free(&d);
  }
}
#define GRANT_SHA "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
#define GRANT_IDENTITIES "\"logical_model_digest\":\"" GRANT_SHA "\",\"operator_contract_digest\":\"" GRANT_SHA "\",\"encoding_digest\":\"" GRANT_SHA "\",\"client_layout_digest\":\"" GRANT_SHA "\",\"placement_plan_digest\":\"" GRANT_SHA "\",\"profile\":\"bf16\",\"server_executor\":\"cpu\""
static void check_model_grant(const char *grants,bool plaintext,bool accepted) {
  ria_service service={0};service.tls.plaintext=plaintext;
  service.tls.expected_peer_name=plaintext ? NULL : "peer.test";
  service.grants=transport_document(grants);
  ria_tensor_store store={0};store.manifest=transport_document("{\"layout_digest\":\"" GRANT_SHA "\"}");
  ria_json_doc bind=transport_document("{\"role\":\"client\"," GRANT_IDENTITIES ",\"limits\":{\"frame_payload_bytes\":65536,\"bulk_data_bytes\":4096,\"expert_rows\":1,\"expert_requests\":1,\"row_lookup_rows\":1,\"inflight_payload_bytes\":33554432,\"operation_timeout_ms\":200,\"frame_io_timeout_ms\":100,\"write_timeout_ms\":100}}");
  ria_error e={0};ria_limits limits;
  CHECK(ria_bind_validate(&bind,false,&limits,&e));
  bool allowed=ria_service_grant(&service,&store,&bind,&e);
  CHECK(allowed==accepted && e.code==(accepted ? 0 : RIA_UNAUTHORIZED));
  ria_json_free(&bind);ria_json_free(&store.manifest);ria_json_free(&service.grants);
}
static void transport_model_grants(void) {
  char named[1024],plain[1024],document[2304];
  int n=snprintf(named,sizeof named,"{\"expected_peer_name\":\"peer.test\"," GRANT_IDENTITIES ",\"server_layout_digest\":\"" GRANT_SHA "\"}");
  CHECK(n>0 && (size_t)n<sizeof named);
  n=snprintf(plain,sizeof plain,"{\"expected_peer_name\":null," GRANT_IDENTITIES ",\"server_layout_digest\":\"" GRANT_SHA "\"}");
  CHECK(n>0 && (size_t)n<sizeof plain);
  for (unsigned mode=0;mode<2;mode++) {
    n=snprintf(document,sizeof document,"{\"grants\":[%s]}",mode ? plain : named);CHECK(n>0 && (size_t)n<sizeof document);
    check_model_grant(document,mode!=0,true);check_model_grant(document,mode==0,false);
    /* Both entry orders must skip the other mode without leaving stale errors. */
    n=snprintf(document,sizeof document,"{\"grants\":[%s,%s]}",mode ? named : plain,mode ? plain : named);
    CHECK(n>0 && (size_t)n<sizeof document);check_model_grant(document,false,true);check_model_grant(document,true,true);
  }
  const char *keys[]={"logical_model_digest","operator_contract_digest","encoding_digest","client_layout_digest",
    "placement_plan_digest","server_layout_digest","profile","server_executor"};
  for (unsigned i=0;i<sizeof keys/sizeof *keys;i++) {
    n=snprintf(document,sizeof document,"{\"grants\":[%s]}",plain);CHECK(n>0 && (size_t)n<sizeof document);
    char prefix[64];n=snprintf(prefix,sizeof prefix,"\"%s\":\"",keys[i]);CHECK(n>0 && (size_t)n<sizeof prefix);
    char *value=strstr(document,prefix);CHECK(value);value+=strlen(prefix);
    if (i==6) { memmove(value+3,value+4,strlen(value+4)+1);memcpy(value,"fp8",3); }
    else if (i==7) { memmove(value+4,value+3,strlen(value+3)+1);memcpy(value,"cuda",4); }
    else *value='b';
    check_model_grant(document,true,false);
  }
  check_model_grant("{\"grants\":[]}",true,false);
  check_model_grant("{\"grants\":[{}]}",true,false);
}
int main(void) {
  accounting(); seccomp_contract(); grants(); queue_stress(); cancellation_progress(); error_credit(); addresses(); contribution();
  transport_policy();transport_model_grants();
  puts("RIA server contracts: NUMA policy/alias accounting, whole-chunk grants, bounded queues, cancellation/progress, explicit transport policy and exact TLS/trusted-network model grants passed; no model/NUMA/GPU execution");
  return 0;
}
