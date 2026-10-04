#define _GNU_SOURCE
#include "server.h"
#include "numa_policy.h"
#include "remote.h"
#include "admin.h"
#include "runtime.h"
#ifdef RIA_WITH_CUDA
#include "expert_cuda.h"
#endif
#include <errno.h>
#include <fcntl.h>
#include <openssl/crypto.h>
#include <openssl/err.h>
#include <openssl/ssl.h>
#include <poll.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/signalfd.h>
#include <sys/socket.h>
#include <unistd.h>

#define WORKER_BYTES RIA_NUMA_WORKER_BYTES
#define STACK_BYTES RIA_NUMA_WORKER_STACK_BYTES
typedef struct server server;
typedef struct {
  bool used, cancelled, completed;
  ria_header header;
  ria_expert_request parsed;
  uint8_t *input, *output;
  uint64_t deadline;
  unsigned remaining;
  ria_error error;
} work;
typedef struct {
  server *owner; unsigned node,index; pthread_t thread; bool started, initialized, exited;
  void *arena; uint64_t arena_bytes;
  float *input,*output; ria_expert_cpu *cpu; uint32_t batch_rows;
#ifdef RIA_WITH_CUDA
  ria_expert_cuda *cuda;
#endif
} worker;
typedef struct {
  ria_header header; uint8_t encoded[64],*payload; size_t cursor;
  uint64_t deadline, write_deadline; bool terminal, retire, write_started;
} reply;
typedef struct {
  ria_transport transport; uint8_t header[64], *input;
  size_t received; ria_header frame; uint64_t frame_deadline, operation_deadline;
  short read_event,write_event; bool decoded,write_retry;
  reply output[9]; unsigned output_count;
} channel;
struct server {
  ria_service service; ria_tensor_store store; ria_bank bank; ria_numa *numa;
  ria_tls tls; ria_binding binding; ria_server_grants grants;
  channel channels[2]; int listeners[2],wake,signals;
  pthread_mutex_t mutex; pthread_cond_t condition;
  ria_server_queue queue; work requests[2]; worker workers[128]; unsigned worker_count,initialized;
  bool stopping,draining,ready,sticky,quiescent,startup_done,startup_ok; uint64_t drain_deadline;
  uint64_t binding_deadline, retirement_deadline; bool binding_retiring;
  uint64_t minimum_frame, minimum_bulk, minimum_expert_charge;
  ria_error worker_error; uint8_t peer_digest[32];
  void *network_arena; uint64_t network_bytes;
  /* Admin state is owned until the bounded admin thread has joined. */
  ria_admin *admin;
};
static uint64_t end_after(uint64_t interval) {
  uint64_t end; return ria_u64_add(ria_monotonic_ms(),interval,&end) ? end : UINT64_MAX;
}
bool ria_server_queue_push(ria_server_queue *q,ria_server_job job,ria_error *e) {
  if (!q || !job.request || q->count==RIA_SERVER_JOBS) return ria_fail(e,RIA_RESOURCE_LIMIT,"bounded worker queue exhausted");
  q->jobs[q->count++]=job; return true;
}
bool ria_server_queue_take(ria_server_queue *q,unsigned node,ria_server_job *out) {
  if (!q || !out) return false;
  for (unsigned i=0;i<q->count;i++) if (q->jobs[i].node==node) {
    *out=q->jobs[i]; memmove(q->jobs+i,q->jobs+i+1,(q->count-i-1)*sizeof(*q->jobs)); q->count--; return true;
  }
  return false;
}
bool ria_server_contribution_unpack(const ria_expert_request *r,unsigned row,unsigned entry,float *input,uint64_t capacity,float *coefficient,ria_error *e) {
  if (!r || !input || !coefficient || row>=r->row_count || entry>=r->entry_count || capacity<r->input_width ||
      entry<ria_read_u32(r->offsets+(size_t)row*4) || entry>=ria_read_u32(r->offsets+(size_t)(row+1)*4))
    return ria_fail(e,RIA_INVALID_REQUEST,"contribution does not belong to validated CSR row/private input buffer");
  *coefficient=ria_read_f32(r->entries+(size_t)entry*8+4);
  for (unsigned i=0;i<r->input_width;i++) input[i]=ria_read_f32(r->inputs+((size_t)row*r->input_width+i)*4);
  return true;
}
bool ria_server_contribution_pack(const ria_expert_request *r,unsigned entry,const float *output,uint64_t capacity,uint8_t *result,uint64_t bytes,ria_error *e) {
  uint64_t offset,length;
  if (!r || !output || !result || entry>=r->entry_count || capacity<r->output_width || bytes!=r->response_bytes ||
      !ria_u64_mul(entry,r->output_width,&offset) || !ria_u64_mul(offset,4,&offset) || !ria_u64_add(offset,32,&offset) ||
      !ria_u64_mul(r->output_width,4,&length) || offset>bytes || length>bytes-offset)
    return ria_fail(e,RIA_INVALID_REQUEST,"contribution output slot exceeds validated response/private buffer");
  for (unsigned i=0;i<r->output_width;i++) ria_write_f32(result+offset+(size_t)i*4,output[i]);
  return true;
}
static int grant_compare(const void *a,const void *b) {
  const ria_chunk_grant *x=a,*y=b;
  if (x->shard!=y->shard) return (x->shard>y->shard)-(x->shard<y->shard);
  return (x->begin>y->begin)-(x->begin<y->begin);
}
bool ria_server_grants_open(ria_server_grants *g,const ria_tensor_store *s,ria_error *e) {
  memset(g,0,sizeof(*g));
  if (!s || s->tensor_count>200000 || s->shard_count>4096 || s->tensor_count+s->shard_count>SIZE_MAX/sizeof(*g->ranges))
    return ria_fail(e,RIA_RESOURCE_LIMIT,"bootstrap grant inventory exceeds bound");
  g->ranges=calloc((size_t)(s->tensor_count+s->shard_count),sizeof(*g->ranges));
  if (!g->ranges) return ria_fail(e,RIA_RESOURCE_LIMIT,"bootstrap grant index allocation");
  for (uint64_t i=0;i<s->shard_count;i++) g->ranges[g->count++]=(ria_chunk_grant){s->shards[i].id,0,s->shards[i].data_start};
  for (uint64_t i=0;i<s->tensor_count;i++) {
    const ria_tensor *t=&s->tensors[i];
    bool public_padding=!strncmp(t->name,"__ria_padding_",14) && !strcmp(t->placement,"inactive");
    if (strcmp(t->placement,"both") && strcmp(t->placement,"client") && strcmp(t->placement,"cache") && !public_padding) continue;
    const ria_shard *a=ria_shard_id(s,t->shard); uint64_t begin,end;
    if (!a || !ria_u64_add(a->data_start,t->offset,&begin) || !ria_u64_add(begin,t->length,&end) || end>a->length) {
      ria_server_grants_close(g); return ria_fail(e,RIA_INTEGRITY_ERROR,"bootstrap grant exceeds trusted shard");
    }
    g->ranges[g->count++]=(ria_chunk_grant){a->id,begin,end};
  }
  qsort(g->ranges,g->count,sizeof(*g->ranges),grant_compare);
  size_t count=0;
  for (size_t i=0;i<g->count;i++) {
    ria_chunk_grant x=g->ranges[i];
    if (count && g->ranges[count-1].shard==x.shard && x.begin<=g->ranges[count-1].end) {
      if (x.end>g->ranges[count-1].end) g->ranges[count-1].end=x.end;
    } else g->ranges[count++]=x;
  }
  g->count=count; return true;
}
void ria_server_grants_close(ria_server_grants *g) { free(g->ranges); memset(g,0,sizeof(*g)); }
bool ria_server_grants_chunk(const ria_server_grants *g,const ria_shard *a,uint64_t index,ria_error *e) {
  uint64_t begin,end;
  if (!g || !a || index>=a->chunk_count || !ria_u64_mul(index,a->chunk_size,&begin) ||
      !ria_u64_add(begin,a->chunk_size,&end)) return ria_fail(e,RIA_UNAUTHORIZED,"chunk outside bound grant");
  if (end>a->length) end=a->length;
  size_t lo=0,hi=g->count;
  while (lo<hi) {
    size_t mid=lo+(hi-lo)/2; const ria_chunk_grant *x=&g->ranges[mid];
    if (x->shard<a->id || (x->shard==a->id && x->begin<=begin)) lo=mid+1; else hi=mid;
  }
  return (lo && g->ranges[lo-1].shard==a->id && g->ranges[lo-1].begin<=begin && g->ranges[lo-1].end>=end) ||
      ria_fail(e,RIA_UNAUTHORIZED,"whole verification chunk contains unauthorized server bytes");
}
/* Resolve immutable operation/granule minima once, before listener readiness.
 * Whole chunks retain their manifest hash; lowering a limit never subchunks. */
static bool server_minimum(server *s,ria_error *e) {
  s->minimum_frame=RIA_ERROR_MAX; s->minimum_bulk=s->minimum_expert_charge=0;
  for (unsigned i=0;i<RIA_LAYERS*2;i++) {
    const ria_operation *op=&s->bank.operations[i]; uint64_t request,response,charge;
    if (!op->handle) continue;
    if (!ria_expert_lengths(1,1,op->input_width,op->output_width,op->quantizer_context_bytes,&request,&response,e) ||
        !ria_request_charge(op->shared ? RIA_SHARED : RIA_EXPERT,request,response,&charge,e)) return false;
    if (request>s->minimum_frame) s->minimum_frame=request;
    if (response>s->minimum_frame) s->minimum_frame=response;
    if (charge>s->minimum_expert_charge) s->minimum_expert_charge=charge;
  }
  for (unsigned i=0;i<2;i++) if (s->bank.tables[i].handle) {
    uint64_t response;
    if (!ria_u64_add(40,s->bank.tables[i].packed_row_stride,&response))
      return ria_fail(e,RIA_RESOURCE_LIMIT,"minimum row reply overflow");
    if (response>s->minimum_frame) s->minimum_frame=response;
  }
  for (size_t i=0;i<s->grants.count;i++) {
    const ria_chunk_grant *g=&s->grants.ranges[i]; const ria_shard *a=ria_shard_id(&s->store,g->shard);
    uint64_t start;
    if (!a || !a->chunk_size || !ria_u64_add(g->begin,(a->chunk_size-g->begin%a->chunk_size)%a->chunk_size,&start))
      return ria_fail(e,RIA_INTEGRITY_ERROR,"invalid immutable chunk grant granule");
    if (start>=a->length || start>=g->end) continue;
    uint64_t length=a->length-start; if (length>a->chunk_size) length=a->chunk_size;
    if (length<=g->end-start && length>s->minimum_bulk) s->minimum_bulk=length;
  }
  uint64_t bulk_frame;
  if (!ria_u64_add(s->minimum_bulk,64,&bulk_frame)) return ria_fail(e,RIA_RESOURCE_LIMIT,"minimum bulk frame overflow");
  if (bulk_frame>s->minimum_frame) s->minimum_frame=bulk_frame;
  return true;
}
static bool minimum_limits(const server *s,const ria_limits *l,ria_error *e) {
  uint64_t control,row,bulk,progress,minimum;
  if (l->frame_payload_bytes<s->minimum_frame || l->bulk_data_bytes<s->minimum_bulk)
    return ria_fail(e,RIA_RESOURCE_LIMIT,"limits cannot hold a registered operation or whole verification chunk");
  if (!ria_progress_charges(l,&control,&row,&bulk,e)) return false;
  if (!ria_u64_add(control,row,&progress) || !ria_u64_add(progress,bulk,&progress) ||
      !ria_u64_add(progress,s->minimum_expert_charge,&minimum) || minimum>l->inflight_payload_bytes ||
      progress>=l->inflight_payload_bytes)
    return ria_fail(e,RIA_RESOURCE_LIMIT,"limits leave no mandatory operation credit after protected progress");
  return true;
}
static void notify(server *s) {
  uint64_t one=1; ssize_t n;
  do n=write(s->wake,&one,sizeof(one)); while (n<0 && errno==EINTR);
  /* A saturated event counter already guarantees a wake; the owner also polls
   * every ten milliseconds and checks the mutex-protected completion state. */
  if (n<0 && errno!=EAGAIN) abort();
}
static void *worker_main(void *argument) {
  worker *w=argument; server *s=w->owner; ria_error e={0};
  bool ok=ria_numa_affinity(&s->service.expert.nodes[w->node],w->index,&e);
  uint64_t scratch=0;
  if (ok) ok=ria_numa_worker_rows(s->service.prefill_rows,s->service.executor,&w->batch_rows,&scratch,&e);
  if (ok && !strcmp(s->service.executor,"cpu")) {
    ok=ria_expert_cpu_create_in(5120,2304,5120,(uint8_t *)w->arena+STACK_BYTES,scratch,&w->cpu,&e);
    /* Raw arena ownership supplies float alignment and storage; no byte
     * object is accessed through an incompatible typed lvalue. */
    void *input=(uint8_t *)w->arena+STACK_BYTES+scratch;
    w->input=input;
    w->output=w->input+(size_t)w->batch_rows*RIA_GRAPH_DIM;
  }
#ifdef RIA_WITH_CUDA
  else if (ok) {
    ok=ria_expert_cuda_device_require(0,s->service.gpu_uuid,NULL,&e) &&
       ria_expert_cuda_create_pooled(0,5120,2304,5120,w->batch_rows,s->service.expert.projection_tile_rows,
                                    s->service.expert.device_workspace_bytes,s->service.expert.pinned_workspace_bytes,&w->cuda,&e);
    void *input=(uint8_t *)w->arena+STACK_BYTES;
    w->input=input; w->output=w->input+(size_t)w->batch_rows*RIA_GRAPH_DIM;
    if (!ok && !e.code) ria_error_set(&e,RIA_RESOURCE_LIMIT,"CUDA workspace exceeds admitted allocation");
  }
#else
  else if (ok) ok=ria_fail(&e,RIA_UNSUPPORTED,"CPU service build cannot select CUDA");
#endif
  pthread_mutex_lock(&s->mutex);
  w->initialized=true; s->initialized++;
  if (!ok) { s->worker_error=e; s->stopping=true; }
  pthread_cond_broadcast(&s->condition); pthread_mutex_unlock(&s->mutex); notify(s);
  if (!ok) goto finish;
  for (;;) {
    ria_server_job task,group[64];unsigned group_count=1;
    pthread_mutex_lock(&s->mutex);
    while (!s->stopping && !ria_server_queue_take(&s->queue,w->node,&task)) pthread_cond_wait(&s->condition,&s->mutex);
    if (s->stopping) { pthread_mutex_unlock(&s->mutex); break; }
    work *r=task.request; bool abandoned=r->cancelled || r->error.code || ria_monotonic_ms()>=r->deadline;
    group[0]=task;
    uint16_t expert_id=ria_read_u16(r->parsed.entries+(size_t)task.entry*8);
    /* Same request/expert/node owns every gathered row. Remove queued jobs
     * under the lifecycle mutex, then retain all borrowed slots until the
     * grouped executor completes. No later cancellation frees that backing. */
    for (unsigned i=0;!abandoned && i<s->queue.count && group_count<w->batch_rows;) {
      ria_server_job candidate=s->queue.jobs[i];
      if (candidate.request==r && candidate.node==w->node &&
          ria_read_u16(r->parsed.entries+(size_t)candidate.entry*8)==expert_id) {
        group[group_count++]=candidate;
        memmove(s->queue.jobs+i,s->queue.jobs+i+1,(s->queue.count-i-1)*sizeof(*s->queue.jobs));s->queue.count--;
      } else i++;
    }
    pthread_mutex_unlock(&s->mutex);
    ria_error error={0}; bool success=true;
    if (!abandoned) {
      float coefficients[64];
      const ria_expert *x=ria_numa_expert(s->numa,w->node,r->parsed.operation_handle,expert_id);
      if (!x) success=ria_fail(&error,RIA_INTEGRITY_ERROR,"worker selected absent local expert");
      for (unsigned i=0;success && i<group_count;i++)
        success=ria_server_contribution_unpack(&r->parsed,group[i].row,group[i].entry,
                    w->input+(size_t)i*RIA_GRAPH_DIM,RIA_GRAPH_DIM,&coefficients[i],&error);
      if (success && w->cpu) success=ria_expert_cpu_evaluate(w->cpu,x,w->input,group_count,RIA_GRAPH_DIM,
                            r->header.kind==RIA_SHARED ? NULL : coefficients,w->output,RIA_GRAPH_DIM,&error);
#ifdef RIA_WITH_CUDA
      else if (success) success=ria_expert_cuda_evaluate(w->cuda,x,w->input,group_count,RIA_GRAPH_DIM,
                            r->header.kind==RIA_SHARED ? NULL : coefficients,w->output,RIA_GRAPH_DIM,&error);
#endif
      for (unsigned i=0;success && i<group_count;i++)
        success=ria_server_contribution_pack(&r->parsed,group[i].entry,w->output+(size_t)i*RIA_GRAPH_DIM,
                                             RIA_GRAPH_DIM,r->output,r->parsed.response_bytes,&error);
    }
    pthread_mutex_lock(&s->mutex);
    if (!success && !r->error.code) r->error=error;
    if (!success && !strcmp(s->service.executor,"cuda")) { s->sticky=true; s->draining=true; }
    if (abandoned || ria_monotonic_ms()>=r->deadline) r->cancelled=true;
    if (r->remaining<group_count) abort();
    r->remaining-=group_count;if (!r->remaining) r->completed=true;
    pthread_cond_broadcast(&s->condition); pthread_mutex_unlock(&s->mutex); notify(s);
  }
finish:
  pthread_mutex_lock(&s->mutex); w->exited=true;
  pthread_cond_broadcast(&s->condition); pthread_mutex_unlock(&s->mutex); notify(s);
  return NULL;
}
static bool queue_reply(server *s,unsigned channel_id,const ria_header *request,uint8_t *payload,
                        size_t length,int status,bool terminal,bool retire,uint64_t deadline,ria_error *e) {
  channel *c=&s->channels[channel_id];
  if (c->output_count==9) { free(payload); return ria_fail(e,RIA_RESOURCE_LIMIT,"bounded response queue exhausted"); }
  reply *r=&c->output[c->output_count]; memset(r,0,sizeof(*r)); r->header=*request;
  r->header.flags=1; r->header.status=(uint32_t)status; r->header.payload_length=length;
  memcpy(r->header.session,s->binding.session,16); r->header.epoch=s->binding.epoch;
  if (request->kind==RIA_BIND && status) { memset(r->header.session,0,16); r->header.epoch=0; }
  r->payload=payload; r->deadline=deadline; r->terminal=terminal; r->retire=retire;
  if (!ria_header_encode(&r->header,r->encoded,e)) { free(payload); return false; }
  c->output_count++; return true;
}
static bool error_reply(server *s,unsigned channel_id,const ria_header *h,int status,bool terminal,
                        bool retire,uint64_t deadline,ria_error *e) {
  const char *message=status==RIA_CANCELLED ? "operation cancelled after quiescence" :
      status==RIA_DEADLINE_EXCEEDED ? "operation deadline exceeded" : "required operation failed";
  char json[256]; int n=snprintf(json,sizeof(json),"{\"code\":%d,\"message\":\"%s\"}",status,message);
  if (n<0 || (size_t)n>=sizeof(json)) return ria_fail(e,RIA_INTERNAL_ERROR,"error response bound");
  uint8_t *payload=malloc((size_t)n);
  if (!payload) return ria_fail(e,RIA_RESOURCE_LIMIT,"error reply allocation");
  memcpy(payload,json,(size_t)n);
  return queue_reply(s,channel_id,h,payload,(size_t)n,status,terminal,retire,deadline,e);
}
static void abandon_requests(server *s) {
  pthread_mutex_lock(&s->mutex);
  for (unsigned i=0;i<2;i++) if (s->requests[i].used) {
    s->requests[i].cancelled=true;
    if (!s->retirement_deadline) s->retirement_deadline=end_after(s->service.expert.drain_timeout_ms);
  }
  pthread_cond_broadcast(&s->condition); pthread_mutex_unlock(&s->mutex);
}
static void invalidate(server *s) {
  ria_binding_invalidate(&s->binding);
  abandon_requests(s); s->binding_deadline=0; s->binding_retiring=false;
  for (unsigned i=0;i<2;i++) {
    channel *c=&s->channels[i];
    if (c->received || c->write_retry) c->transport.unusable=true;
    ria_transport_close(&c->transport);
    for (unsigned j=0;j<c->output_count;j++) free(c->output[j].payload);
    c->output_count=0; c->received=0; c->decoded=false; c->write_retry=false;
  }
}
static bool work_active(server *s) {
  bool active=false;
  pthread_mutex_lock(&s->mutex);
  for (unsigned i=0;i<2;i++) if (s->requests[i].used && !s->requests[i].completed) active=true;
  pthread_mutex_unlock(&s->mutex); return active;
}
static bool completed(server *s,ria_error *e) {
  pthread_mutex_lock(&s->mutex);
  bool sticky=s->sticky;
  for (unsigned i=0;i<2;i++) {
    work *r=&s->requests[i];
    if (!r->used || !r->completed) continue;
    uint8_t *result=r->output; r->output=NULL;
    ria_header h=r->header; uint64_t deadline=r->deadline, response_bytes=r->parsed.response_bytes;
    int status=r->error.code;
    if (r->cancelled) status=ria_monotonic_ms()>=deadline ? RIA_DEADLINE_EXCEEDED : RIA_CANCELLED;
    free(r->input); r->input=NULL; r->used=false;
    pthread_mutex_unlock(&s->mutex);
    bool ok=true;
    /* A completion belongs to its original binding even if a new authenticated
     * pair was installed after the old executor became quiescent. */
    if (!s->binding.valid || !s->binding.bound || s->binding_retiring || sticky || h.epoch!=s->binding.epoch ||
        memcmp(h.session,s->binding.session,16)) free(result);
    else if (status) {
      free(result);
      /* The operation deadline includes execution and publication. If it is
       * already expired, invalidate rather than extending it to emit a reply. */
      if (ria_monotonic_ms()>=deadline) invalidate(s);
      else {
        bool retire=status!=RIA_CANCELLED;
        if (retire) { s->binding_retiring=true; s->binding.draining=true; abandon_requests(s); }
        ok=error_reply(s,0,&h,status,true,retire,deadline,e);
      }
    } else ok=queue_reply(s,0,&h,result,(size_t)response_bytes,0,true,false,deadline,e);
    if (!ok) return false;
    pthread_mutex_lock(&s->mutex);
  }
  pthread_mutex_unlock(&s->mutex);
  if (sticky) invalidate(s);
  return true;
}
static bool admit(server *s,const ria_header *h,uint64_t response,uint64_t start,ria_error *e) {
  uint64_t charge;
  if (!ria_request_charge(h->kind,h->payload_length,response,&charge,e) ||
      !ria_binding_admit(&s->binding,h,charge,start,e)) return false;
  return true;
}
static bool bind_frame(server *s,unsigned ch,ria_header *h,const uint8_t *p,uint64_t start,ria_error *e) {
  ria_json_doc d={0}; bool ok=ria_control_json(p,(size_t)h->payload_length,h,&d,e);
  if (ok && !ch) {
    ria_limits requested;
    ok=ria_bind_validate(&d,false,&requested,e) && ria_service_grant(&s->service,&s->store,&d,e);
    /* Exact local ceilings are agreed here. Smaller requests are accepted only
     * if every requested field can hold the selected realization; no enlargement. */
#define CEILING(field) do { if (requested.field>s->service.limits.field) requested.field=s->service.limits.field; } while (0)
    CEILING(frame_payload_bytes); CEILING(bulk_data_bytes); CEILING(expert_rows);
    CEILING(expert_requests); CEILING(row_lookup_rows); CEILING(inflight_payload_bytes);
    CEILING(operation_timeout_ms); CEILING(frame_io_timeout_ms); CEILING(write_timeout_ms);
#undef CEILING
    uint64_t control,row,bulk;
    if (ok) {
      pthread_mutex_lock(&s->mutex); s->quiescent=false; pthread_mutex_unlock(&s->mutex);
      ria_binding_init(&s->binding,&requested);
      ok=minimum_limits(s,&requested,e) && ria_progress_charges(&requested,&control,&row,&bulk,e) &&
         ria_binding_protect(&s->binding,control,row,bulk,e) && ria_binding_fresh(&s->binding,e);
    }
    uint8_t layout[32]; char *json=NULL; size_t length=0;
    if (ok) ok=ria_json_digest_field(&s->store.manifest,ria_json_get(&s->store.manifest,0,"layout_digest"),layout,e) &&
       ria_bind_response_json(&d,&requested,layout,s->binding.session,s->binding.epoch,s->binding.bulk_capability,&json,&length,e);
    if (ok) ok=queue_reply(s,0,h,(uint8_t *)json,length,0,false,false,start+requested.operation_timeout_ms,e);
    else free(json);
  } else if (ok) {
    uint8_t capability[32],logical[32],operator_id[32],peer[32];
    ok=ria_json_digest_field(&d,ria_json_get(&d,0,"bulk_capability"),capability,e) &&
       ria_json_digest_field(&d,ria_json_get(&d,0,"logical_model_digest"),logical,e) &&
       ria_json_digest_field(&d,ria_json_get(&d,0,"operator_contract_digest"),operator_id,e) &&
       !memcmp(logical,s->store.logical_model_digest,32) && !memcmp(operator_id,s->store.operator_contract_digest,32) &&
       ria_transport_peer_digest(&s->channels[1].transport,peer,e) &&
       ria_binding_bulk(&s->binding,capability,CRYPTO_memcmp(peer,s->peer_digest,32)==0,e);
    OPENSSL_cleanse(capability,sizeof(capability));
    if (ok) {
      const char *json="{\"bound\":true}"; uint8_t *copy=malloc(strlen(json));
      if (!copy) ok=ria_fail(e,RIA_RESOURCE_LIMIT,"bulk bind reply allocation");
      else { memcpy(copy,json,strlen(json)); ok=queue_reply(s,1,h,copy,strlen(json),0,false,false,start+s->binding.limits.operation_timeout_ms,e); }
    }
  }
  ria_json_free(&d); return ok;
}
static unsigned choose_node(server *s,uint64_t handle,uint16_t id) {
  if (!strcmp(s->service.executor,"cuda")) return 0;
  if (s->service.expert.policy==RIA_NUMA_SHARDED ||
      (handle>40 && s->service.expert.policy==RIA_NUMA_REPLICATED_EXPERTS))
    return ria_numa_owner(&s->service.expert,handle,id);
  unsigned best=0,count=UINT_MAX;
  for (unsigned n=0;n<s->service.expert.node_count;n++) {
    unsigned current=0;
    for (unsigned i=0;i<s->queue.count;i++) if (s->queue.jobs[i].node==n) current++;
    if (current<count) { count=current; best=n; }
  }
  return best;
}
static bool expert_frame(server *s,const ria_header *h,const uint8_t *p,uint64_t start,ria_error *e) {
  if (h->payload_length<40) return ria_fail(e,RIA_INVALID_REQUEST,"truncated expert prefix");
  const ria_operation *operation=ria_bank_operation(&s->bank,ria_read_u64(p)); ria_expert_request parsed;
  if (!operation || !ria_expert_parse(p,(size_t)h->payload_length,h->kind,operation,&s->binding.limits,&parsed,e) ||
      !admit(s,h,parsed.response_bytes,start,e)) return false;
  pthread_mutex_lock(&s->mutex);
  work *r=NULL;
  for (unsigned i=0;i<2;i++) if (!s->requests[i].used) { r=&s->requests[i]; break; }
  if (!r || parsed.entry_count>RIA_SERVER_JOBS) {
    pthread_mutex_unlock(&s->mutex); return ria_fail(e,RIA_RESOURCE_LIMIT,"expert request/queue capacity exhausted");
  }
  if (parsed.entry_count>RIA_SERVER_JOBS-s->queue.count) { pthread_mutex_unlock(&s->mutex); return ria_fail(e,RIA_RESOURCE_LIMIT,"worker queue admission exceeded"); }
  memset(r,0,sizeof(*r)); r->input=malloc((size_t)h->payload_length); r->output=calloc(1,(size_t)parsed.response_bytes);
  if (!r->input || !r->output) { free(r->input); free(r->output); memset(r,0,sizeof(*r)); pthread_mutex_unlock(&s->mutex); return ria_fail(e,RIA_RESOURCE_LIMIT,"admitted expert buffers allocation failed"); }
  memcpy(r->input,p,(size_t)h->payload_length);
  if (!ria_expert_parse(r->input,(size_t)h->payload_length,h->kind,operation,&s->binding.limits,&r->parsed,e)) {
    free(r->input); free(r->output); memset(r,0,sizeof(*r)); pthread_mutex_unlock(&s->mutex); return false;
  }
  r->used=true; r->header=*h; r->deadline=start+s->binding.limits.operation_timeout_ms; r->remaining=parsed.entry_count;
  ria_write_u64(r->output,parsed.operation_handle); ria_write_u64(r->output+8,parsed.invocation_id);
  ria_write_u32(r->output+16,parsed.row_count); ria_write_u32(r->output+20,parsed.output_width); ria_write_u32(r->output+24,parsed.entry_count);
  for (unsigned row=0;row<parsed.row_count;row++) {
    unsigned begin=ria_read_u32(parsed.offsets+row*4),end=ria_read_u32(parsed.offsets+(row+1)*4);
    for (unsigned entry=begin;entry<end;entry++) {
      uint16_t id=ria_read_u16(parsed.entries+(size_t)entry*8); unsigned node=choose_node(s,parsed.operation_handle,id);
      if (!ria_server_queue_push(&s->queue,(ria_server_job){r,node,row,entry},e)) abort();
    }
  }
  if (!r->remaining) r->completed=true;
  pthread_cond_broadcast(&s->condition); pthread_mutex_unlock(&s->mutex); return true;
}
static bool cancel_frame(server *s,const ria_header *h,const uint8_t *p,uint64_t start,ria_error *e) {
  uint64_t target; uint32_t state;
  if (!ria_cancel_parse(p,(size_t)h->payload_length,h->request_id,h->epoch,&target,e) ||
      !admit(s,h,24,start,e) || !ria_binding_cancel(&s->binding,target,&state,e)) return false;
  pthread_mutex_lock(&s->mutex);
  for (unsigned i=0;i<2;i++) if (s->requests[i].used && s->requests[i].header.request_id==target) s->requests[i].cancelled=true;
  pthread_cond_broadcast(&s->condition); pthread_mutex_unlock(&s->mutex);
  channel *c=&s->channels[0];
  for (unsigned i=0;i<c->output_count;i++) if (c->output[i].header.request_id==target) {
    if (c->output[i].write_started) return ria_fail(e,RIA_CANCELLED,"cancel races a started TLS reply; retire pair");
    reply old=c->output[i]; free(old.payload);
    memmove(c->output+i,c->output+i+1,(c->output_count-i-1)*sizeof(*c->output)); c->output_count--;
    if (!error_reply(s,0,&old.header,RIA_CANCELLED,true,false,old.deadline,e)) return false;
    break;
  }
  uint8_t *result=calloc(1,24);
  if (!result) return ria_fail(e,RIA_RESOURCE_LIMIT,"cancel acknowledgment allocation");
  ria_write_u64(result,target); ria_write_u64(result+8,h->epoch); ria_write_u32(result+16,state);
  return queue_reply(s,0,h,result,24,0,true,false,start+s->binding.limits.operation_timeout_ms,e);
}
static bool dispatch_operation(server *s,unsigned ch,ria_header *h,uint8_t *p,uint64_t start,ria_error *e) {
  if (h->kind==RIA_BIND || h->kind==RIA_BIND_BULK) return bind_frame(s,ch,h,p,start,e);
  if (!s->binding.bulk_bound && h->kind!=RIA_HEALTH && h->kind!=RIA_CLOSE)
    return ria_fail(e,RIA_NOT_READY,"both authenticated channels required before work");
  if (h->kind==RIA_EXPERT || h->kind==RIA_SHARED) return expert_frame(s,h,p,start,e);
  if (h->kind==RIA_CANCEL) return cancel_frame(s,h,p,start,e);
  if (h->kind==RIA_ROWS) {
    if (h->payload_length<16) return ria_fail(e,RIA_INVALID_REQUEST,"truncated row prefix");
    const ria_table *table=ria_bank_table(&s->bank,ria_read_u64(p)); ria_row_request r;
    if (!table || !ria_rows_parse(p,(size_t)h->payload_length,table,&s->binding.limits,&r,e) || !admit(s,h,r.response_bytes,start,e)) return false;
    uint8_t *result=malloc((size_t)r.response_bytes);
    if (!result) return ria_fail(e,RIA_RESOURCE_LIMIT,"row progress allocation");
    if (!ria_bank_rows(&s->bank,&r,result,(size_t)r.response_bytes,e)) { free(result); return false; }
    return queue_reply(s,ch,h,result,(size_t)r.response_bytes,0,true,false,start+s->binding.limits.operation_timeout_ms,e);
  }
  if (h->kind==RIA_CHUNK) {
    uint64_t handle,index; const uint8_t *bytes,*hash; uint32_t length;
    if (!ria_chunk_request(p,(size_t)h->payload_length,&handle,&index,e)) return false;
    const ria_shard *a=ria_shard_id(&s->store,handle);
    if (!ria_server_grants_chunk(&s->grants,a,index,e) || !ria_tensor_chunk(&s->store,handle,index,&bytes,&length,&hash,e) ||
        length>s->binding.limits.bulk_data_bytes || !admit(s,h,(uint64_t)length+64,start,e)) return false;
    uint8_t *result=calloc(1,(size_t)length+64);
    if (!result) return ria_fail(e,RIA_RESOURCE_LIMIT,"bulk progress allocation");
    ria_write_u64(result,handle); ria_write_u64(result+8,index); ria_write_u64(result+16,index*a->chunk_size);
    ria_write_u32(result+24,length); memcpy(result+32,hash,32); memcpy(result+64,bytes,length);
    return queue_reply(s,ch,h,result,(size_t)length+64,0,true,false,start+s->binding.limits.operation_timeout_ms,e);
  }
  if (h->kind==RIA_HEALTH) {
    ria_json_doc d={0};
    if (!ria_control_json(p,(size_t)h->payload_length,h,&d,e)) return false;
    ria_json_free(&d);
    const char *json="{\"counters\":{},\"ready\":true,\"state\":\"ready\"}";
    if (s->binding.draining) json="{\"counters\":{},\"ready\":false,\"state\":\"draining\"}";
    if (!admit(s,h,strlen(json),start,e)) return false;
    uint8_t *result=malloc(strlen(json));
    if (!result) return ria_fail(e,RIA_RESOURCE_LIMIT,"health progress allocation");
    memcpy(result,json,strlen(json)); return queue_reply(s,ch,h,result,strlen(json),0,true,false,start+s->binding.limits.operation_timeout_ms,e);
  }
  if (h->kind==RIA_CLOSE) return !h->payload_length && admit(s,h,0,start,e);
  return ria_fail(e,RIA_INVALID_REQUEST,"unsupported bound operation");
}
static bool dispatch(server *s,unsigned ch,ria_header *h,uint8_t *p,uint64_t start,ria_error *e) {
  pthread_mutex_lock(&s->mutex); bool unavailable=s->draining || s->sticky; pthread_mutex_unlock(&s->mutex);
  if (unavailable || s->binding_retiring) return ria_fail(e,RIA_NOT_READY,"service drain forbids new admissions");
  /* Corrupt direction, identity and request ordering close without admission. */
  if (!ria_binding_receive(&s->binding,h,ch!=0,e)) return false;
  if (dispatch_operation(s,ch,h,p,start,e)) return true;
  int status=e->code;
  bool initial=h->kind==RIA_BIND;
  if (!initial && (status==RIA_INVALID_REQUEST || status==RIA_IDENTITY_MISMATCH || !status)) return false;
  if (!status) status=RIA_INTERNAL_ERROR;
  bool accepted=false;
  for (unsigned i=0;i<9;i++) accepted|=s->binding.pending[i].used &&
    s->binding.pending[i].id==h->request_id && s->binding.pending[i].bulk==(ch!=0);
  bool retire=initial || h->kind==RIA_BIND_BULK || h->kind==RIA_EXPERT || h->kind==RIA_SHARED || h->kind==RIA_ROWS;
  if (retire) { s->binding_retiring=true; s->binding.draining=true; abandon_requests(s); }
  uint64_t deadline;
  if (!ria_u64_add(start,s->binding.limits.operation_timeout_ms,&deadline)) return false;
  memset(e,0,sizeof(*e));
  /* A pre-admission refusal has no pending slot and returns no server credit. */
  return error_reply(s,ch,h,status,accepted,retire,deadline,e);
}
/* Incremental TLS I/O: one owner, fixed read buffers and bounded reply slots.
 * SSL retry arguments stay stable until that operation makes progress. */
static bool ssl_step(channel *c,bool writing,void *buffer,size_t length,size_t *done,ria_error *e) {
  size_t amount=0;
  ERR_clear_error();
  int rc=writing ? SSL_write_ex(c->transport.ssl,buffer,length,&amount) : SSL_read_ex(c->transport.ssl,buffer,length,&amount);
  if (rc==1) {
    if (!amount || amount>length) { c->transport.unusable=true; return ria_fail(e,RIA_INTERNAL_ERROR,"TLS made invalid progress"); }
    if (writing) c->write_retry=false;
    *done+=amount; return true;
  }
  int code=SSL_get_error(c->transport.ssl,rc); short event=code==SSL_ERROR_WANT_WRITE ? POLLOUT : POLLIN;
  if (code==SSL_ERROR_WANT_READ || code==SSL_ERROR_WANT_WRITE) {
    if (writing) { c->write_event=event; c->write_retry=true; } else c->read_event=event; return true;
  }
  c->transport.unusable=true;
  return ria_fail(e,RIA_NOT_READY,"authenticated channel disconnected/failed");
}
static bool channel_read(server *s,unsigned id,ria_error *e) {
  channel *c=&s->channels[id]; uint64_t now=ria_monotonic_ms();
  if (c->received && now>=c->frame_deadline) return ria_fail(e,RIA_DEADLINE_EXCEEDED,"started frame deadline exceeded");
  c->read_event=POLLIN;
  for (unsigned steps=0;steps<8;steps++) {
    size_t before=c->received;
    if (!c->decoded) {
      if (!ssl_step(c,false,c->header+c->received,64-c->received,&c->received,e)) return false;
      if (!before && c->received) {
        c->operation_deadline=end_after(s->binding.limits.operation_timeout_ms);
        c->frame_deadline=end_after(s->binding.limits.frame_io_timeout_ms);
        if (c->frame_deadline>c->operation_deadline) c->frame_deadline=c->operation_deadline;
      }
      if (c->received<64) return true;
      if (!ria_header_decode(c->header,s->binding.limits.frame_payload_bytes,&c->frame,e)) return false;
      c->decoded=true;
    }
    size_t payload_received=c->received-64;
    if (payload_received<c->frame.payload_length) {
      size_t total=c->received;
      if (!ssl_step(c,false,c->input+payload_received,(size_t)c->frame.payload_length-payload_received,&total,e)) return false;
      c->received=total;
      if (c->received==64+payload_received) return true;
    }
    if (c->received==64+c->frame.payload_length) {
      uint64_t started=c->operation_deadline-s->binding.limits.operation_timeout_ms;
      if (!dispatch(s,id,&c->frame,c->input,started,e)) return false;
      c->received=0; c->decoded=false;
      if (s->binding_retiring) return true;
      if (!SSL_pending(c->transport.ssl)) return true;
    }
  }
  return true;
}
static bool channel_write(server *s,unsigned id,ria_error *e) {
  channel *c=&s->channels[id]; c->write_event=POLLOUT;
  if (!c->output_count) return true;
  reply *r=&c->output[0]; uint64_t now=ria_monotonic_ms();
  if (!r->write_started) {
    r->write_started=true; r->write_deadline=end_after(s->binding.limits.write_timeout_ms);
    if (r->write_deadline>r->deadline) r->write_deadline=r->deadline;
  }
  if (now>=r->write_deadline) return ria_fail(e,RIA_DEADLINE_EXCEEDED,"reply deadline includes queue and socket publication");
  if (r->cursor<64) {
    if (!ssl_step(c,true,r->encoded+r->cursor,64-r->cursor,&r->cursor,e) || r->cursor<64) return !e->code;
  }
  if (r->cursor<64+r->header.payload_length) {
    if (!ssl_step(c,true,r->payload+r->cursor-64,(size_t)(64+r->header.payload_length-r->cursor),&r->cursor,e)) return false;
    if (r->cursor<64+r->header.payload_length) return true;
  }
  bool retire=r->retire;
  if (r->terminal && !ria_binding_terminal_channel(&s->binding,r->header.request_id,id!=0,true,e)) return false;
  if (r->header.kind==RIA_BIND_BULK && !r->header.status) s->binding_deadline=0;
  free(r->payload); memmove(c->output,c->output+1,(c->output_count-1)*sizeof(*c->output)); c->output_count--;
  if (retire) invalidate(s);
  return true;
}
static bool close_ready(server *s,ria_error *e) {
  ria_pending *close=NULL; unsigned other=0;
  for (unsigned i=0;i<9;i++) if (s->binding.pending[i].used) {
    if (s->binding.pending[i].kind==RIA_CLOSE) close=&s->binding.pending[i]; else other++;
  }
  if (!close || other) return true;
  for (unsigned i=0;i<s->channels[0].output_count;i++) if (s->channels[0].output[i].header.kind==RIA_CLOSE) return true;
  ria_header h={.kind=RIA_CLOSE,.request_id=close->id,.epoch=s->binding.epoch}; memcpy(h.session,s->binding.session,16);
  return queue_reply(s,0,&h,NULL,0,0,true,true,close->deadline,e);
}
static int listen_tcp(const char *text,ria_error *e) {
  struct sockaddr_storage address; socklen_t length;
  if (!ria_address(text,true,&address,&length,e)) return -1;
  int fd=socket(address.ss_family,SOCK_STREAM|SOCK_CLOEXEC|SOCK_NONBLOCK,0);
  if (fd<0) { ria_error_set(e,RIA_RESOURCE_LIMIT,"expert listener allocation failed"); return -1; }
  int one=1;
  if (setsockopt(fd,SOL_SOCKET,SO_REUSEADDR,&one,sizeof(one)) || bind(fd,(struct sockaddr *)&address,length) || listen(fd,2)) {
    close(fd); ria_error_set(e,RIA_NOT_READY,"expert listen bind failed: %s",strerror(errno)); return -1;
  }
  return fd;
}
static bool accept_channel(server *s,unsigned id,ria_error *e) {
  int fd=accept4(s->listeners[id],NULL,NULL,SOCK_NONBLOCK|SOCK_CLOEXEC);
  if (fd<0) return (errno==EAGAIN || errno==EINTR) || ria_fail(e,RIA_NOT_READY,"expert accept failed");
  /* No execution is admitted until both handshakes finish. Subsequent new
   * connections cannot interrupt an established pair or create another client. */
  channel *c=&s->channels[id];
  pthread_mutex_lock(&s->mutex);
  bool retained=false;
  for (unsigned i=0;i<2;i++) retained|=s->requests[i].used;
  pthread_mutex_unlock(&s->mutex);
  if (c->transport.fd>=0 || (id && (!s->binding.bound || s->binding.bulk_capability_used)) ||
      (!id && (s->binding.bound || retained))) { close(fd); return true; }
  if (!id) {
    uint64_t interval;
    if (!ria_u64_add(s->service.tls.handshake_timeout_ms,s->service.limits.operation_timeout_ms,&interval) ||
        !ria_u64_add(ria_monotonic_ms(),interval,&s->binding_deadline)) {
      close(fd); return ria_fail(e,RIA_INVALID_REQUEST,"initial pair deadline overflow");
    }
    /* The initial pair has one finite handshake+operation budget, including
     * silence before Bind and absence of the required bulk connection. */
    ria_binding_init(&s->binding,&s->service.limits);
  }
  pthread_mutex_lock(&s->mutex); s->quiescent=false; pthread_mutex_unlock(&s->mutex);
  if (!ria_transport_accept(&c->transport,&s->tls,fd,e)) { close(fd); return false; }
  c->read_event=POLLIN; c->write_event=POLLOUT; c->received=0; c->decoded=false;
  if (!id && !ria_transport_peer_digest(&c->transport,s->peer_digest,e)) return false;
  return true;
}
static bool admin_health(void *context,bool *ready,bool *active,ria_error *e) {
  (void)e; server *s=context; pthread_mutex_lock(&s->mutex);
  *ready=s->ready && !s->draining && !s->sticky && !s->retirement_deadline; *active=!s->quiescent;
  pthread_mutex_unlock(&s->mutex); return true;
}
static bool admin_drain(void *context,uint64_t deadline,ria_error *e) {
  server *s=context; pthread_mutex_lock(&s->mutex);
  s->draining=true; s->ready=false;
  if (!s->drain_deadline || deadline<s->drain_deadline) s->drain_deadline=deadline;
  pthread_mutex_unlock(&s->mutex); notify(s);
  pthread_mutex_lock(&s->mutex);
  while (!s->quiescent) {
    uint64_t now=ria_monotonic_ms();
    if (now>=deadline) { pthread_mutex_unlock(&s->mutex); return ria_fail(e,RIA_DEADLINE_EXCEEDED,"expert drain awaits actual worker/transport quiescence"); }
    struct timespec absolute={(time_t)(deadline/1000),(long)(deadline%1000)*1000000};
    int status=pthread_cond_timedwait(&s->condition,&s->mutex,&absolute);
    if (status && status!=ETIMEDOUT) { pthread_mutex_unlock(&s->mutex); return ria_fail(e,RIA_INTERNAL_ERROR,"expert drain condition failed"); }
  }
  pthread_mutex_unlock(&s->mutex); return true;
}
static uint64_t doc_bytes(const ria_json_doc *d) {
  return d->allocated_bytes;
}
static bool runtime_budget(server *s,ria_error *e) {
  /* One index-page transient exists at a time. Reserve that peak separately
   * from bounded SSL/library baseline and the supervised startup thread stack. */
  uint64_t bytes=sizeof(*s)+(32u<<20)+STACK_BYTES+4096+ria_admin_reserved_bytes(), workers,frames;
  if (!ria_u64_mul(s->service.expert.worker_count,WORKER_BYTES,&workers) ||
      !ria_u64_mul(s->service.limits.frame_payload_bytes,2,&frames) ||
      !ria_u64_add(bytes,workers,&bytes) || !ria_u64_add(bytes,frames,&bytes) ||
      !ria_u64_add(bytes,s->service.limits.inflight_payload_bytes,&bytes) ||
      !ria_u64_add(bytes,s->service.expert.pinned_workspace_bytes,&bytes)) goto fail;
  uint64_t metadata=doc_bytes(&s->store.manifest)+doc_bytes(&s->store.index)+doc_bytes(&s->service.document)+doc_bytes(&s->service.lock)+doc_bytes(&s->service.plan)+doc_bytes(&s->service.grants)+
      (uint64_t)s->store.tensor_count*(sizeof(ria_tensor)+sizeof(void *)+sizeof(ria_chunk_grant)*2)+
      (uint64_t)s->store.shard_count*sizeof(ria_shard)+40*384*sizeof(ria_expert)*s->service.expert.node_count;
  for (unsigned i=0;i<s->store.page_count;i++) if (!ria_u64_add(metadata,doc_bytes(&s->store.pages[i]),&metadata)) goto fail;
  for (uint64_t i=0;i<s->store.shard_count;i++) if (!ria_u64_add(metadata,s->store.shards[i].chunk_count*32,&metadata)) goto fail;
  if (!ria_u64_add(bytes,metadata,&bytes) || bytes>s->service.expert.host_runtime_bytes)
    return ria_fail(e,RIA_RESOURCE_LIMIT,"expert runtime/metadata/worker pools exceed explicit host reservation");
  return true;
fail: return ria_fail(e,RIA_RESOURCE_LIMIT,"expert runtime allocation arithmetic overflow");
}
static bool place_shard(void *context,const ria_tensor_store *store,const ria_shard *shard,bool populated,ria_error *e) {
  server *s=context;
  if (!shard) {
    ria_runtime_observation observation;
    if (!runtime_budget(s,e) || !ria_runtime_require(s->service.host_cap,0,s->service.expert.pinned_workspace_bytes,&observation,e)) return false;
  }
  return ria_numa_place_shard(&s->service.expert,store,shard,populated,e);
}
static bool startup_progress(void *context,ria_error *e) {
  server *s=context; pthread_mutex_lock(&s->mutex); bool cancelled=s->stopping; pthread_mutex_unlock(&s->mutex);
  return !cancelled || ria_fail(e,RIA_CANCELLED,"expert startup cancelled between authenticated blocks");
}
static void *startup_main(void *context) {
  server *s=context; ria_error e={0};
  uint64_t resident=s->service.host_cap-s->service.expert.host_runtime_bytes;
  ria_tensor_load_options load={.role="server",.expected_digest=s->service.manifest_digest,
    .max_resident_bytes=resident,.lock_memory=false,.numa_node=-1,.max_metadata_bytes=s->service.expert.host_runtime_bytes};
  bool ok=ria_tensor_store_open_controlled(&s->store,s->service.manifest_path,&load,place_shard,s,startup_progress,s,&e) &&
      startup_progress(s,&e) && runtime_budget(s,&e) && ria_bank_open(&s->bank,&s->store,&e) && startup_progress(s,&e) &&
      ria_numa_open_controlled(&s->numa,&s->bank,&s->service.expert,startup_progress,s,&e) &&
      ria_server_grants_open(&s->grants,&s->store,&e) && server_minimum(s,&e) &&
      minimum_limits(s,&s->service.limits,&e) && ria_tls_create(&s->tls,&s->service.tls,&e);
  pthread_mutex_lock(&s->mutex); s->startup_done=true; s->startup_ok=ok;
  if (!ok) s->worker_error=e;
  pthread_cond_broadcast(&s->condition); pthread_mutex_unlock(&s->mutex); notify(s); return NULL;
}
static bool startup_supervise(server *s,ria_error *e) {
  pthread_t thread; pthread_attr_t attr;
  if (pthread_attr_init(&attr)) return ria_fail(e,RIA_RESOURCE_LIMIT,"startup thread attributes");
  int code=pthread_attr_setstacksize(&attr,STACK_BYTES);
  if (!code) code=pthread_create(&thread,&attr,startup_main,s);
  pthread_attr_destroy(&attr);
  if (code) return ria_fail(e,RIA_RESOURCE_LIMIT,"startup supervisor thread allocation: %s",strerror(code));
  uint64_t stop_deadline=0;
  for (;;) {
    pthread_mutex_lock(&s->mutex); bool done=s->startup_done,ok=s->startup_ok; ria_error result=s->worker_error; pthread_mutex_unlock(&s->mutex);
    if (done) {
      if (pthread_join(thread,NULL)) { fputs("expert startup join failed\n",stderr); _Exit(RIA_INTERNAL_ERROR); }
      if (!ok) *e=result;
      return ok;
    }
    if (stop_deadline && ria_monotonic_ms()>=stop_deadline) {
      fputs("expert startup cancellation deadline exceeded; terminating with live owner intact\n",stderr); _Exit(RIA_DEADLINE_EXCEEDED);
    }
    struct pollfd fds[2]={{s->signals,POLLIN,0},{s->wake,POLLIN,0}};
    int status=poll(fds,2,10);
    if (status<0 && errno==EINTR) continue;
    if (status<0 || (fds[0].revents&POLLIN)) {
      if (status>=0) { struct signalfd_siginfo info; while (read(s->signals,&info,sizeof(info))==(ssize_t)sizeof(info)) {} }
      pthread_mutex_lock(&s->mutex); s->stopping=true; pthread_mutex_unlock(&s->mutex);
      if (!stop_deadline) stop_deadline=end_after(s->service.expert.drain_timeout_ms);
    }
    if (fds[1].revents&POLLIN) { uint64_t count; while (read(s->wake,&count,sizeof(count))>0) {} }
  }
}
static bool start_workers(server *s,ria_error *e) {
  pthread_attr_t attr;
  if (pthread_attr_init(&attr)) return ria_fail(e,RIA_RESOURCE_LIMIT,"worker attributes initialization");
  bool ok=true;
  for (unsigned node=0;ok && node<s->service.expert.node_count;node++) for (unsigned i=0;ok && i<s->service.expert.nodes[node].workers;i++) {
    worker *w=&s->workers[s->worker_count]; w->owner=s; w->node=node; w->index=i;
    ok=ria_numa_arena(s->service.expert.nodes[node].node,WORKER_BYTES,&w->arena,&w->arena_bytes,e);
    if (ok) { memset(w->arena,0,(size_t)w->arena_bytes); ok=ria_numa_verify(s->service.expert.nodes[node].node,w->arena,w->arena_bytes,e); }
    if (ok && (pthread_attr_setstack(&attr,w->arena,STACK_BYTES) || pthread_create(&w->thread,&attr,worker_main,w)))
      ok=ria_fail(e,RIA_RESOURCE_LIMIT,"node-local bounded worker creation failed");
    if (ok) { w->started=true; s->worker_count++; }
    else ria_numa_release(w->arena,w->arena_bytes);
  }
  pthread_attr_destroy(&attr);
  if (!ok) return false;
  pthread_mutex_lock(&s->mutex);
  uint64_t deadline=end_after(s->service.expert.drain_timeout_ms);
  while (s->initialized<s->worker_count && !s->stopping) {
    struct timespec absolute={(time_t)(deadline/1000),(long)(deadline%1000)*1000000};
    int status=pthread_cond_timedwait(&s->condition,&s->mutex,&absolute);
    if (status) { ok=ria_fail(e,RIA_DEADLINE_EXCEEDED,"executor worker startup deadline exceeded"); break; }
  }
  if (s->worker_error.code) { *e=s->worker_error; ok=false; }
  pthread_mutex_unlock(&s->mutex); return ok;
}
static bool event_loop(server *s,ria_error *e) {
  bool terminate=false;
  for (;;) {
    pthread_mutex_lock(&s->mutex);
    bool draining=s->draining,sticky=s->sticky; uint64_t drain_deadline=s->drain_deadline;
    if (sticky && !drain_deadline) { drain_deadline=end_after(s->service.expert.drain_timeout_ms); s->drain_deadline=drain_deadline; }
    pthread_mutex_unlock(&s->mutex);
    if (draining) invalidate(s);
    if (s->binding_deadline && ria_monotonic_ms()>=s->binding_deadline) invalidate(s);
    if (!completed(s,e)) return false;
    bool active=work_active(s);
    pthread_mutex_lock(&s->mutex);
    if (!active) s->retirement_deadline=0;
    uint64_t retirement_deadline=s->retirement_deadline;
    pthread_mutex_unlock(&s->mutex);
    if (active && retirement_deadline && ria_monotonic_ms()>=retirement_deadline) {
      fputs("expert disconnected-binding drain exceeded deadline with live executor; terminating\n",stderr);
      _Exit(RIA_DEADLINE_EXCEEDED);
    }
    if (!s->binding.valid && !active) {
      pthread_mutex_lock(&s->mutex); s->quiescent=true; pthread_cond_broadcast(&s->condition); pthread_mutex_unlock(&s->mutex);
      if (terminate || sticky) return !sticky || ria_fail(e,RIA_EXECUTOR_ERROR,"sticky CUDA failure retired service");
    }
    if (draining && active && drain_deadline && ria_monotonic_ms()>=drain_deadline) {
      /* Numeric work cannot safely be detached or have borrowed RAM reclaimed.
       * Termination is the explicit bounded containment policy. */
      fputs("expert drain exceeded deadline with live executor; terminating\n",stderr); _Exit(RIA_DEADLINE_EXCEEDED);
    }
    if (s->binding.valid && s->binding.bound) {
      uint64_t expired;
      if (ria_binding_expired(&s->binding,ria_monotonic_ms(),&expired)) { invalidate(s); continue; }
      if (!close_ready(s,e)) { invalidate(s); memset(e,0,sizeof(*e)); }
    }
    struct pollfd fds[6]={{s->wake,POLLIN,0},{s->signals,POLLIN,0},
      {s->listeners[0],draining?0:POLLIN,0},{s->listeners[1],draining?0:POLLIN,0},
      {s->channels[0].transport.fd,POLLIN,0},{s->channels[1].transport.fd,POLLIN,0}};
    for (unsigned i=0;i<2;i++) {
      channel *c=&s->channels[i]; fds[4+i].events=s->binding_retiring || c->write_retry ? 0 : c->read_event;
      if (c->output_count) fds[4+i].events|=c->write_event;
      if (c->received && ria_monotonic_ms()>=c->frame_deadline) { invalidate(s); break; }
      if (c->output_count && (ria_monotonic_ms()>=c->output[0].deadline ||
          (c->output[0].write_started && ria_monotonic_ms()>=c->output[0].write_deadline))) { invalidate(s); break; }
    }
    int n=poll(fds,6,10);
    if (n<0 && errno==EINTR) continue;
    if (n<0) return ria_fail(e,RIA_INTERNAL_ERROR,"expert event poll failed");
    if (fds[0].revents&POLLIN) { uint64_t count; while (read(s->wake,&count,sizeof(count))>0) {} }
    if (fds[1].revents&POLLIN) {
      struct signalfd_siginfo info; while (read(s->signals,&info,sizeof(info))==(ssize_t)sizeof(info)) {}
      terminate=true; pthread_mutex_lock(&s->mutex); s->draining=true; s->ready=false;
      if (!s->drain_deadline) s->drain_deadline=end_after(s->service.expert.drain_timeout_ms);
      pthread_mutex_unlock(&s->mutex); continue;
    }
    for (unsigned i=0;i<2;i++) if (fds[2+i].revents&POLLIN) {
      if (!accept_channel(s,i,e)) { invalidate(s); memset(e,0,sizeof(*e)); }
    }
    for (unsigned i=0;i<2;i++) {
      channel *c=&s->channels[i]; short events=fds[4+i].revents;
      if (c->transport.fd<0) continue;
      bool ok=true;
      if (events&(POLLERR|POLLHUP|POLLNVAL)) { c->transport.unusable=true; ok=false; }
      /* OpenSSL requires a WANT_WRITE operation to be retried before another
       * operation can trigger I/O. Keep its borrowed buffer and arguments. */
      bool retried=c->write_retry;
      if (ok && retried && (events&c->write_event)) ok=channel_write(s,i,e);
      if (ok && c->transport.fd>=0 && !s->binding_retiring && !c->write_retry &&
          ((events&c->read_event) || SSL_pending(c->transport.ssl))) ok=channel_read(s,i,e);
      pthread_mutex_lock(&s->mutex); bool stopped=s->sticky || s->draining; pthread_mutex_unlock(&s->mutex);
      if (stopped) ok=false;
      if (ok && !retried && c->transport.fd>=0 && c->output_count && (events&c->write_event)) ok=channel_write(s,i,e);
      if (!ok) { invalidate(s); memset(e,0,sizeof(*e)); break; }
    }
    active=work_active(s);
    pthread_mutex_lock(&s->mutex); s->quiescent=!s->binding.bound && !active && s->channels[0].transport.fd<0 && s->channels[1].transport.fd<0;
    pthread_cond_broadcast(&s->condition); pthread_mutex_unlock(&s->mutex);
  }
}
bool ria_server_run(const char *path,ria_error *e) {
  if (!ria_disable_dumps(e)) return false;
  server *s=calloc(1,sizeof(*s));
  if (!s) return ria_fail(e,RIA_RESOURCE_LIMIT,"expert owner allocation failed");
  s->wake=s->signals=s->listeners[0]=s->listeners[1]=-1;
  s->channels[0].transport.fd=s->channels[1].transport.fd=-1;
  bool mutex_ready=false,condition_ready=false,ok=false;
  pthread_condattr_t ca;
  if (pthread_mutex_init(&s->mutex,NULL)) { ria_error_set(e,RIA_RESOURCE_LIMIT,"expert lifecycle mutex initialization"); goto finish; }
  mutex_ready=true;
  if (pthread_condattr_init(&ca)) { ria_error_set(e,RIA_RESOURCE_LIMIT,"expert condition attributes"); goto finish; }
  int condition_status=pthread_condattr_setclock(&ca,CLOCK_MONOTONIC);
  if (!condition_status) condition_status=pthread_cond_init(&s->condition,&ca);
  pthread_condattr_destroy(&ca);
  if (condition_status) { ria_error_set(e,RIA_RESOURCE_LIMIT,"expert monotonic condition initialization"); goto finish; }
  condition_ready=true;
  if (!ria_service_read(&s->service,path,e)) goto finish;
  if (strcmp(s->service.role,"expert")) { ria_error_set(e,RIA_INVALID_REQUEST,"expert entry requires an expert service configuration"); goto finish; }
  if (!ria_numa_topology(&s->service.expert,e) || !ria_numa_affinity(&s->service.expert.nodes[0],0,e)) goto finish;
#ifndef RIA_WITH_CUDA
  if (strcmp(s->service.executor,"cpu")) { ria_error_set(e,RIA_UNSUPPORTED,"CPU image does not provide CUDA executor"); goto finish; }
#endif
  sigset_t signals; sigemptyset(&signals); sigaddset(&signals,SIGTERM); sigaddset(&signals,SIGINT); sigaddset(&signals,SIGPIPE);
  if (pthread_sigmask(SIG_BLOCK,&signals,NULL)) { ria_error_set(e,RIA_INTERNAL_ERROR,"expert signal policy initialization"); goto finish; }
  sigdelset(&signals,SIGPIPE);
  s->signals=signalfd(-1,&signals,SFD_NONBLOCK|SFD_CLOEXEC); s->wake=eventfd(0,EFD_NONBLOCK|EFD_CLOEXEC);
  if (s->signals<0 || s->wake<0) { ria_error_set(e,RIA_RESOURCE_LIMIT,"expert event descriptors unavailable"); goto finish; }
  if (s->service.expert.host_runtime_bytes>=s->service.host_cap) { ria_error_set(e,RIA_RESOURCE_LIMIT,"runtime reserve leaves no model capacity"); goto finish; }
  if (!startup_supervise(s,e)) goto finish;
  uint64_t frame_bytes=s->service.limits.frame_payload_bytes;
  if (!ria_numa_arena(s->service.expert.nodes[0].node,frame_bytes*2,&s->network_arena,&s->network_bytes,e)) goto finish;
  memset(s->network_arena,0,(size_t)s->network_bytes);
  if (!ria_numa_verify(s->service.expert.nodes[0].node,s->network_arena,s->network_bytes,e)) goto finish;
  s->channels[0].input=s->network_arena; s->channels[1].input=(uint8_t *)s->network_arena+frame_bytes;
  ria_binding_init(&s->binding,&s->service.limits);
  if (!start_workers(s,e)) goto finish;
  s->listeners[0]=listen_tcp(s->service.control_address,e); s->listeners[1]=listen_tcp(s->service.bulk_address,e);
  if (s->listeners[0]<0 || s->listeners[1]<0) goto finish;
  if (!ria_admin_start(s->service.admin_socket,10001,s->service.expert.drain_timeout_ms,
       (ria_admin_callbacks){s,admin_health,admin_drain},&s->admin,e)) goto finish;
  pthread_mutex_lock(&s->mutex); s->ready=true; s->quiescent=true; pthread_mutex_unlock(&s->mutex);
  ok=event_loop(s,e);
finish:
  if (mutex_ready) {
    pthread_mutex_lock(&s->mutex); s->stopping=true; if (condition_ready) pthread_cond_broadcast(&s->condition); pthread_mutex_unlock(&s->mutex);
  }
  if (condition_ready && s->wake>=0) invalidate(s);
  uint64_t join_deadline=end_after(s->service.expert.drain_timeout_ms ? s->service.expert.drain_timeout_ms : 1000);
  for (unsigned i=0;i<s->worker_count;i++) {
    worker *w=&s->workers[i];
    pthread_mutex_lock(&s->mutex);
    while (w->started && !w->exited) {
      struct timespec until={(time_t)(join_deadline/1000),(long)(join_deadline%1000)*1000000};
      if (ria_monotonic_ms()>=join_deadline || pthread_cond_timedwait(&s->condition,&s->mutex,&until)) {
        fputs("expert worker cleanup deadline exceeded; terminating with owned arenas intact\n",stderr); _Exit(RIA_DEADLINE_EXCEEDED);
      }
    }
    pthread_mutex_unlock(&s->mutex);
    if (w->started && pthread_join(w->thread,NULL)) { fputs("expert worker join failed\n",stderr); _Exit(RIA_INTERNAL_ERROR); }
    ria_expert_cpu_destroy(w->cpu);
#ifdef RIA_WITH_CUDA
    ria_error cleanup={0};
    if (w->cuda && !ria_expert_cuda_destroy(w->cuda,&cleanup)) {
      /* Failed synchronization does not prove DMA has released host/model
       * ownership. Process teardown retains every borrowed arena until exit. */
      fprintf(stderr,"expert CUDA cleanup could not prove quiescence (primary=%d cleanup=%d); terminating\n",e->code,cleanup.code);
      _Exit(e->code ? e->code : cleanup.code ? cleanup.code : RIA_EXECUTOR_ERROR);
    }
#endif
    ria_numa_release(w->arena,w->arena_bytes);
  }
  for (unsigned i=0;i<2;i++) { free(s->requests[i].input); free(s->requests[i].output); }
  if (condition_ready) {
    pthread_mutex_lock(&s->mutex); s->quiescent=true; s->ready=false; s->draining=true;
    pthread_cond_broadcast(&s->condition); pthread_mutex_unlock(&s->mutex);
  }
  if (s->admin) {
    ria_error cleanup={0};
    if (!ria_admin_stop(s->admin,s->service.expert.drain_timeout_ms,&cleanup)) {
      fprintf(stderr,"expert admin cleanup failed: %s\n",cleanup.message); _Exit(cleanup.code ? cleanup.code : RIA_INTERNAL_ERROR);
    }
  }
  for (unsigned i=0;i<2;i++) { ria_transport_close(&s->channels[i].transport); if (s->listeners[i]>=0) close(s->listeners[i]); }
  if (s->signals>=0) close(s->signals);
  if (s->wake>=0) close(s->wake);
  ria_tls_destroy(&s->tls); ria_server_grants_close(&s->grants); ria_numa_close(s->numa); ria_bank_close(&s->bank);
  ria_numa_release(s->network_arena,s->network_bytes); ria_tensor_store_close(&s->store); ria_service_free(&s->service);
  if (condition_ready) pthread_cond_destroy(&s->condition);
  if (mutex_ready) pthread_mutex_destroy(&s->mutex);
  free(s); return ok;
}
