#define _POSIX_C_SOURCE 200809L
/* Execute the real frontend prompt synchronizer with an explicit graph command
 * recorder. No tensor math, network, runtime probing or CUDA is executed. */
#include "ria/engine.c"
#include <assert.h>
struct ria_graph { uint64_t position;ria_graph_phase phase; };
struct ria_remote { unsigned unused; };
typedef struct {
  ria_graph graph;ria_remote remote;
  unsigned calls,resets,retags,aborts,reconnects,polls,cancel_at,fail_at;
  uint64_t generation,expected_position;
  uint32_t expected_rows,seen_images;
} fixture;
static fixture recorder;
static bool cancelled(void *p) { fixture *f=p;return f->cancel_at && ++f->polls>=f->cancel_at; }
bool ria_remote_begin_generation(ria_remote *r,uint64_t *epoch,uint64_t *generation,ria_error *e) {
  (void)e;assert(r==&recorder.remote);*epoch=37;*generation=++recorder.generation;return true;
}
void ria_remote_abort(ria_remote *r,uint64_t generation) {
  assert(r==&recorder.remote && generation==recorder.generation);recorder.aborts++;
}
bool ria_remote_close(ria_remote *r,ria_error *e) { (void)e;assert(r==&recorder.remote);return true; }
bool ria_remote_open(ria_remote **r,const ria_remote_options *o,ria_error *e) {
  (void)o;(void)e;*r=&recorder.remote;recorder.reconnects++;return true;
}
ria_graph_remote ria_remote_callbacks(ria_remote *r) { (void)r;return (ria_graph_remote){0}; }
bool ria_graph_create(const ria_tensor_store *s,const ria_graph_options *o,ria_graph_remote remote,ria_graph **g,ria_error *e) {
  (void)s;(void)o;(void)remote;(void)e;*g=&recorder.graph;return true;
}
bool ria_graph_destroy(ria_graph *g,ria_error *e) { (void)e;assert(g==&recorder.graph);return true; }
bool ria_graph_reset(ria_graph *g,uint64_t epoch,uint64_t generation,ria_error *e) {
  (void)e;assert(g==&recorder.graph && epoch==37 && generation==recorder.generation);
  g->position=0;recorder.expected_position=0;recorder.resets++;return true;
}
bool ria_graph_retag(ria_graph *g,uint64_t epoch,uint64_t generation,ria_error *e) {
  (void)e;assert(g==&recorder.graph && epoch==37 && generation==recorder.generation);recorder.retags++;return true;
}
bool ria_graph_set_phase(ria_graph *g,ria_graph_phase phase,ria_error *e) { (void)e;g->phase=phase;return true; }
bool ria_graph_prefill_rows(ria_graph *g,const uint32_t *tokens,const float *const *images,uint32_t rows,
                            float *last,float *all,ria_graph_cancel_fn cancel,void *context,ria_error *e) {
  assert(g==&recorder.graph && rows && rows<=recorder.expected_rows && !all &&
    g->position==recorder.expected_position && g->phase!=RIA_GRAPH_DECODE);
  recorder.calls++;
  if (recorder.calls==recorder.fail_at) return ria_fail(e,RIA_EXECUTOR_ERROR,"injected prompt chunk failure");
  for (unsigned i=0;i<rows;i++) {
    if (cancel && cancel(context)) return ria_fail(e,RIA_CANCELLED,"injected bounded graph cancellation");
    assert(tokens[i]==g->position+i+1);
    if (images && images[i]) { assert(images[i][0]==100+(float)(g->position+i));recorder.seen_images++; }
  }
  for (unsigned i=0;i<RIA_GRAPH_VOCAB;i++) last[i]=(float)tokens[rows-1];
  g->position+=rows;recorder.expected_position=g->position;return true;
}
static ria_engine *create(unsigned rows) {
  memset(&recorder,0,sizeof recorder);recorder.expected_rows=rows;
  ria_engine *r=calloc(1,sizeof *r);assert(r);r->claimed=true;r->context=130;r->graph_options.prefill_rows=rows;
  r->graph=&recorder.graph;r->remote=&recorder.remote;r->prefix=calloc(130,sizeof(uint32_t));
  r->last_logits=calloc(RIA_GRAPH_VOCAB,sizeof(float));assert(r->prefix && r->last_logits);return r;
}
static void release(ria_engine *r) { free(r->prefix);free(r->last_logits);free(r); }
int main(void) {
  int values[130];for (unsigned i=0;i<130;i++) values[i]=(int)i+1;
  ds4_tokens tokens={.v=values,.len=130};float *logits=calloc(RIA_GRAPH_VOCAB,sizeof(float));assert(logits);
  float embeddings[3*RIA_GRAPH_DIM]={0};for (unsigned i=0;i<3;i++) embeddings[i*RIA_GRAPH_DIM]=102+(float)i;
  ds4_vision_span image={.token_start=2,.embedding={.data=embeddings,.token_count=3,.layout=RIA_IMAGE_LAYOUT}};
  memset(image.embedding.fingerprint,0xa5,32);
  const unsigned bounds[]={1,3,17,64};ria_error e={0};
  for (unsigned b=0;b<sizeof(bounds)/sizeof(*bounds);b++) {
    ria_engine *r=create(bounds[b]);tokens.len=67;
    assert(ria_engine_sync(r,&tokens,&image,1,logits,NULL,NULL,&e));
    assert(r->position==67 && r->valid && !r->retired && recorder.seen_images==3 && recorder.resets==1 && logits[0]==67);
    unsigned calls=recorder.calls;assert(calls==(67+bounds[b]-1)/bounds[b]);
    tokens.len=130;assert(ria_engine_sync(r,&tokens,&image,1,logits,NULL,NULL,&e));
    assert(r->position==130 && r->last_continuation && r->last_reused_tokens==67 &&
      recorder.retags==1 && recorder.seen_images==3 && logits[0]==130 && recorder.graph.phase==RIA_GRAPH_CONTINUATION);
    assert(recorder.calls==calls+(63+bounds[b]-1)/bounds[b]);
    calls=recorder.calls;assert(ria_engine_sync(r,&tokens,&image,1,logits,NULL,NULL,&e) && recorder.calls==calls);
    image.embedding.fingerprint[0]^=1;assert(ria_engine_sync(r,&tokens,&image,1,logits,NULL,NULL,&e));
    assert(recorder.reconnects==1 && recorder.resets==2 && r->last_reused_tokens==0);
    image.embedding.fingerprint[0]^=1;release(r);
  }
  ria_engine *r=create(64);tokens.len=5;recorder.cancel_at=4;
  assert(!ria_engine_sync(r,&tokens,NULL,0,logits,cancelled,&recorder,&e) && e.code==RIA_CANCELLED &&
    !r->valid && r->retired && r->position==0 && recorder.aborts==1);release(r);
  r=create(3);tokens.len=5;recorder.fail_at=2;
  assert(!ria_engine_sync(r,&tokens,NULL,0,logits,NULL,NULL,&e) && e.code==RIA_EXECUTOR_ERROR &&
    !r->valid && r->retired && r->position==3 && r->prefix[3]==0 && recorder.aborts==1);release(r);
  r=create(3);tokens.len=5;values[4]=129280;
  assert(!ria_engine_sync(r,&tokens,NULL,0,logits,NULL,NULL,&e) && recorder.calls==0 && recorder.resets==0);values[4]=5;release(r);
  free(logits);puts("RIA actual engine grouped prompt synchronization: chunks, images, continuation/prefix replay, cancellation and failure publication passed");return 0;
}
