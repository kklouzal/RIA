#define _POSIX_C_SOURCE 200809L
/* Actual RIA construction/cleanup control flow, with explicit model/runtime
 * boundary shims. No checkpoint, physical probe, socket or CUDA operation. */
#include "ria/engine.c"
#define REQUIRE(condition) do { if (!(condition)) { \
  fprintf(stderr,"fixture failure %s:%d: %s\n",__FILE__,__LINE__,#condition); \
  exit(1); } } while (0)
#include <signal.h>
#define SHA "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
struct ria_graph { unsigned owner; };
struct ria_remote { unsigned owner; };
struct ria_tokenizer { unsigned owner; };
typedef enum { NONE, EARLY, SERVICE, RUNTIME, BEFORE_LOAD, COPY_FIRST, COPY_LAST,
               HASH_FIRST, HASH_LAST, CONFIG, BIND, GRAPH } stop_phase;
typedef struct {
  stop_phase stop;
  volatile sig_atomic_t halted;
  unsigned polls,services,runtimes,loads,blocks,binds,graphs;
  unsigned graph_closes,remote_closes,store_closes,service_closes,tokenizer_closes,runtime_ends;
  ria_tensor metadata[5];
  uint8_t *map;
  uint8_t primes[2*24*8],multipliers[2*4*8],pad[4],offsets[2*24*8];
} fixture;
/* Test-owned foreign-boundary dispatch: one fixture, reset between cases. */
static fixture *active;
static bool stopped(void *context) {
  fixture *f=context;REQUIRE(f==active);f->polls++;return f->halted!=0;
}
static ria_json_doc document(const char *text) {
  ria_json_doc d={0};ria_error e={0};REQUIRE(ria_json_parse(text,strlen(text),(ria_json_limits){4096,256,16},&d,&e));return d;
}
bool ria_service_read(ria_service *s,const char *path,ria_error *e) {
  (void)e;REQUIRE(active && !strcmp(path,"/fixture/service.json"));active->services++;
  memset(s,0,sizeof *s);s->role="client";s->executor="cuda";s->profile="bf16";
  s->manifest_path="/fixture/manifest.json";s->context_positions=128;s->prefill_rows=64;
  s->host_cap=UINT64_C(1073741824);s->device_cap=UINT64_C(1048576);s->pinned_cap=128;
  memset(s->logical_model_digest,0xaa,32);memset(s->operator_contract_digest,0xaa,32);
  s->placement=document("{\"schema_revision\":1,\"logical_model_digest\":\"" SHA "\",\"operator_contract_digest\":\"" SHA "\",\"server_layout_digest\":\"" SHA "\",\"server_executor\":\"cpu\",\"schedule\":[],\"shared_placement\":\"client\",\"expert_policy\":\"remote\",\"host_expert_cache_bytes\":0,\"device_expert_cache_bytes\":0,\"engram_cache_bytes\":0,\"local_experts\":[],\"runtime\":{\"tokenizer_file\":\"/fixture/tokenizer.json\",\"tokenizer_sha256\":\"" SHA "\",\"tokenizer_memory_bytes\":\"1048576\",\"host_state_bytes\":\"1048576\",\"device_state_bytes\":\"1048576\",\"projection_tile_rows\":8,\"state_tile_rows\":8,\"max_image_patches\":1,\"frontend_host_bytes\":\"16777216\",\"prefill_rows\":64},\"digest\":\"" SHA "\"}");
  if (active->stop==SERVICE) active->halted=1;
  return true;
}
void ria_service_free(ria_service *s) {
  REQUIRE(active);active->service_closes++;ria_json_free(&s->placement);memset(s,0,sizeof *s);
}
bool ria_runtime_require(uint64_t host,uint64_t locked,uint64_t pinned,ria_runtime_observation *out,ria_error *e) {
  (void)locked;(void)e;REQUIRE(active && host==UINT64_C(1073741824) && pinned==128);active->runtimes++;memset(out,0,sizeof *out);
  if (active->stop==RUNTIME && active->runtimes==1) active->halted=1;
  return true;
}
bool ria_graph_pinned_required_bytes(const ria_graph_options *o,uint64_t *bytes,ria_error *e) {
  (void)e;REQUIRE(o->prefill_rows==64);*bytes=16;
  if (active->stop==BEFORE_LOAD && !active->loads) active->halted=1;
  return true;
}
bool ria_graph_host_required_bytes(const ria_graph_options *o,uint64_t *bytes,ria_error *e) { (void)o;(void)e;*bytes=64;return true; }
bool ria_tensor_owned_bytes(const ria_tensor_store *s,uint64_t *bytes,ria_error *e) { (void)s;(void)e;*bytes=1024;return true; }
uint64_t ria_graph_cuda_metadata_bytes(void) { return 8; }
uint64_t ria_vision_metadata_bytes(void) { return 8; }
bool ria_graph_local_experts_validate(const ria_graph_options *o,uint64_t *host,uint64_t *device,ria_error *e) {
  (void)e;REQUIRE(o->local_expert_count==0);*host=*device=0;return true;
}
bool ria_tensor_matrix(const ria_tensor_store *s,const ria_tensor *t,ria_expert_matrix *out,ria_error *e) {
  (void)s;(void)t;(void)out;return ria_fail(e,RIA_INTEGRITY_ERROR,"unexpected synthetic matrix request");
}
bool ria_remote_host_required_bytes(uint32_t rows,const ria_limits *l,uint64_t *bytes,ria_error *e) {
  (void)l;(void)e;REQUIRE(rows==64);*bytes=1024;return true;
}
bool ria_tensor_store_open_controlled(ria_tensor_store *s,const char *path,const ria_tensor_load_options *o,
                                      ria_tensor_place place,void *place_context,ria_tensor_progress progress,void *progress_context,ria_error *e) {
  REQUIRE(active && !strcmp(path,"/fixture/manifest.json") && o->lock_memory &&
         !strcmp(o->role,"client") && place && progress && place_context==progress_context);
  active->loads++;memset(s,0,sizeof *s);strcpy(s->profile,"bf16");
  memset(s->logical_model_digest,0xaa,32);memset(s->operator_contract_digest,0xaa,32);
  s->manifest=document("{\"tokenizer_digest\":\"" SHA "\",\"layout_digest\":\"" SHA "\"}");
  if (!place(place_context,s,NULL,false,e)) return false;
  /* Two bounded synthetic phases model the actual TensorStore copy and hash
   * progress contract. Every cancellation must preserve its typed cause. */
  for (unsigned block=0;block<8;block++) {
    if ((active->stop==COPY_FIRST && block==0) || (active->stop==COPY_LAST && block==3) ||
        (active->stop==HASH_FIRST && block==4) || (active->stop==HASH_LAST && block==7)) active->halted=1;
    if (!progress(progress_context,e)) return false;
    active->blocks++;
  }
  return true;
}
void ria_tensor_store_close(ria_tensor_store *s) { REQUIRE(active);active->store_closes++;ria_json_free(&s->manifest);memset(s,0,sizeof *s); }
const ria_tensor *ria_tensor_name(const ria_tensor_store *s,const char *name) {
  (void)s;REQUIRE(active);for (unsigned i=0;i<5;i++) if (!strcmp(name,active->metadata[i].name)) return &active->metadata[i];return NULL;
}
bool ria_tokenizer_runtime_begin(ria_error *e) { (void)e;return true; }
void ria_tokenizer_runtime_end(void) { REQUIRE(active);active->runtime_ends++; }
bool ria_tokenizer_open(const char *path,const uint8_t sha[32],uint64_t budget,ria_tokenizer **out,ria_error *e) {
  (void)sha;(void)budget;(void)e;REQUIRE(!strcmp(path,"/fixture/tokenizer.json"));*out=calloc(1,sizeof **out);REQUIRE(*out);
  if (active->stop==CONFIG) active->halted=1;
  return true;
}
void ria_tokenizer_close(ria_tokenizer *t) { REQUIRE(active);if(t) active->tokenizer_closes++;free(t); }
bool ria_graph_options_validate(const ria_graph_options *o,ria_error *e) { (void)e;REQUIRE(o->max_tokens==128 && o->prefill_rows==64);return true; }
bool ria_remote_open(ria_remote **out,const ria_remote_options *o,ria_error *e) {
  (void)o;(void)e;active->binds++;*out=calloc(1,sizeof **out);REQUIRE(*out);if(active->stop==BIND) active->halted=1;
  return true;
}
bool ria_remote_close(ria_remote *r,ria_error *e) { (void)e;REQUIRE(active);if(r) active->remote_closes++;free(r);return true; }
ria_graph_remote ria_remote_callbacks(ria_remote *r) { REQUIRE(r);return (ria_graph_remote){0}; }
bool ria_graph_create(const ria_tensor_store *s,const ria_graph_options *o,ria_graph_remote remote,ria_graph **out,ria_error *e) {
  (void)s;(void)o;(void)remote;(void)e;active->graphs++;*out=calloc(1,sizeof **out);REQUIRE(*out);if(active->stop==GRAPH) active->halted=1;
  return true;
}
bool ria_graph_destroy(ria_graph *g,ria_error *e) { (void)e;REQUIRE(active);if(g) active->graph_closes++;free(g);return true; }
static void initialize(fixture *f,stop_phase phase) {
  memset(f,0,sizeof *f);f->stop=phase;f->halted=phase==EARLY;f->map=calloc(129280,4);REQUIRE(f->map);
  for(unsigned i=0;i<48;i++) { ria_write_u64(f->primes+(size_t)i*8,1);ria_write_u64(f->offsets+(size_t)i*8,i%24); }
  f->metadata[0]=(ria_tensor){.name="engram.token_map",.dtype="U32",.length=129280*4u,.data=f->map};
  f->metadata[1]=(ria_tensor){.name="engram.primes",.dtype="U64",.length=sizeof f->primes,.data=f->primes};
  f->metadata[2]=(ria_tensor){.name="engram.multipliers",.dtype="I64",.length=sizeof f->multipliers,.data=f->multipliers};
  f->metadata[3]=(ria_tensor){.name="engram.pad_id",.dtype="U32",.length=sizeof f->pad,.data=f->pad};
  f->metadata[4]=(ria_tensor){.name="engram.offsets",.dtype="U64",.length=sizeof f->offsets,.data=f->offsets};
  active=f;
}
int main(void) {
  for(stop_phase phase=NONE;phase<=GRAPH;phase++) {
    fixture f;initialize(&f,phase);ria_error e={0};ria_engine *engine=NULL;
    bool ok=ria_engine_open("/fixture/service.json",0,stopped,&f,&engine,&e);
    if(phase==NONE) {
      REQUIRE(ok && engine && !engine->startup_cancel && !engine->startup_cancel_context && f.blocks==8);
      unsigned polls=f.polls;REQUIRE(ria_engine_close(engine,&e) && polls==f.polls);
    } else {
      REQUIRE(!ok && !engine && e.code==RIA_CANCELLED);
      REQUIRE(f.binds==(phase>=BIND) && f.graphs==(phase>=GRAPH));
    }
    if(phase!=EARLY) REQUIRE(f.store_closes==1 && f.service_closes==1);
    REQUIRE(f.remote_closes==f.binds && f.graph_closes==f.graphs);
    REQUIRE(f.tokenizer_closes==f.runtime_ends);
    free(f.map);active=NULL;
  }
  fixture f;initialize(&f,NONE);ria_error e={0};ria_engine *engine=NULL;
  REQUIRE(ria_engine_open("/fixture/service.json",0,NULL,NULL,&engine,&e) && f.polls==0);
  REQUIRE(ria_engine_close(engine,&e));free(f.map);active=NULL;
  puts("RIA actual constructor: explicit startup cancellation, bounded copy/hash checks, phase/final publication gates, typed cleanup and callback lifetime passed");return 0;
}
