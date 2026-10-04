#define _GNU_SOURCE
#include "numa_policy.h"
#include <errno.h>
#include <numa.h>
#include <numaif.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define PAGE_BYTES UINT64_C(4096)
typedef struct { uint64_t begin, end; unsigned node; } span;
typedef struct { uintptr_t begin, end; uint8_t *target; } relocation;
typedef struct { void *arena; uint64_t bytes; ria_expert *experts; ria_expert shared[40]; const uint8_t *tables[2]; } replica;
struct ria_numa { const ria_bank *bank; ria_expert_config config; replica nodes[64]; };
static bool rounded(uint64_t n, uint64_t *r, ria_error *e) {
  uint64_t v;
  if (!ria_u64_add(n, PAGE_BYTES-1, &v))
    return ria_fail(e,RIA_RESOURCE_LIMIT,"NUMA page extent overflow");
  *r=v&~(PAGE_BYTES-1); return true;
}
bool ria_numa_config_validate(const ria_expert_config *c, ria_error *e) {
  if (!c || !c->node_count || c->node_count>64 ||
      c->policy<RIA_NUMA_SHARDED || c->policy>RIA_NUMA_REPLICATED_SERVER_MODEL ||
      !c->host_runtime_bytes || !c->startup_host_bytes || !c->drain_timeout_ms ||
      !c->projection_tile_rows || c->projection_tile_rows>64)
    return ria_fail(e,RIA_INVALID_REQUEST,"invalid expert NUMA policy/budgets");
  unsigned workers=0;
  for (unsigned i=0;i<c->node_count;i++) {
    const ria_expert_node *n=&c->nodes[i];
    if (n->node>63 || !n->local_bytes || !n->workers || n->workers>64 ||
        !n->cpu_count || n->cpu_count>256 || n->workers>n->cpu_count)
      return ria_fail(e,RIA_INVALID_REQUEST,"invalid NUMA node/worker affinity");
    workers+=n->workers;
    for (unsigned j=0;j<i;j++) if (n->node==c->nodes[j].node)
      return ria_fail(e,RIA_INVALID_REQUEST,"duplicate selected NUMA node");
    for (unsigned j=0;j<n->cpu_count;j++) {
      if (n->cpus[j]>=65536) return ria_fail(e,RIA_INVALID_REQUEST,"CPU ID exceeds bounded mask");
      for (unsigned k=0;k<i;k++) for (unsigned l=0;l<c->nodes[k].cpu_count;l++)
        if (n->cpus[j]==c->nodes[k].cpus[l])
          return ria_fail(e,RIA_INVALID_REQUEST,"CPU belongs to multiple worker pools");
      for (unsigned k=0;k<j;k++) if (n->cpus[j]==n->cpus[k])
        return ria_fail(e,RIA_INVALID_REQUEST,"duplicate worker CPU");
    }
  }
  if (workers>128 || workers!=c->worker_count)
    return ria_fail(e,RIA_RESOURCE_LIMIT,"bounded worker count mismatch");
  return true;
}
unsigned ria_numa_owner(const ria_expert_config *c,uint64_t handle,uint16_t expert) {
  return handle>40 ? 0 : (unsigned)(((handle-1)*384+expert)%c->node_count);
}
static bool tensor_node(const ria_tensor *t,const ria_expert_config *c,unsigned *out,ria_error *e) {
  unsigned layer,id; int used=0;
  if (sscanf(t->name,"layers.%u.ffn.experts.%u.%n",&layer,&id,&used)==2 && used) {
    if (layer>=40 || id>=384)
      return ria_fail(e,RIA_INTEGRITY_ERROR,"expert tensor owner outside target");
    *out=c->policy==RIA_NUMA_SHARDED ? ria_numa_owner(c,layer+1,(uint16_t)id) : 0;
  } else *out=0;
  return true;
}
static int compare_span(const void *a,const void *b) {
  const span *x=a,*y=b;
  return (x->begin>y->begin)-(x->begin<y->begin);
}
static int compare_relocation(const void *a,const void *b) {
  const relocation *x=a,*y=b;
  return (x->begin>y->begin)-(x->begin<y->begin);
}
static const uint8_t *relocate(const relocation *map,size_t count,const uint8_t *p) {
  if (!p) return NULL;
  uintptr_t value=(uintptr_t)p; size_t lo=0,hi=count;
  while (lo<hi) { size_t mid=lo+(hi-lo)/2; if (map[mid].begin<=value) lo=mid+1; else hi=mid; }
  return lo && value<map[lo-1].end ? map[lo-1].target+(value-map[lo-1].begin) : p;
}
static bool shard_spans(const ria_tensor_store *s,const ria_shard *a,const ria_expert_config *c,
                        span **out,size_t *count,ria_error *e) {
  *out=NULL; *count=0;
  if (a->data_start%PAGE_BYTES)
    return ria_fail(e,RIA_INTEGRITY_ERROR,"NUMA source data_start must be page aligned");
  if (s->tensor_count>SIZE_MAX/sizeof(span))
    return ria_fail(e,RIA_RESOURCE_LIMIT,"NUMA interval inventory overflow");
  span *ranges=malloc((size_t)s->tensor_count*sizeof(*ranges));
  if (!ranges) return ria_fail(e,RIA_RESOURCE_LIMIT,"NUMA interval inventory allocation");
  for (uint64_t i=0;i<s->tensor_count;i++) {
    const ria_tensor *t=&s->tensors[i];
    if (t->shard!=a->id || !t->length || !strncmp(t->name,"__ria_padding_",14)) continue;
    unsigned owner;
    uint64_t begin,end;
    if (!tensor_node(t,c,&owner,e) || !ria_u64_add(a->data_start,t->offset,&begin) ||
        !ria_u64_add(begin,t->length,&end) || !rounded(end,&end,e)) { free(ranges); return false; }
    ranges[(*count)++]=(span){begin&~(PAGE_BYTES-1),end,owner};
  }
  qsort(ranges,*count,sizeof(*ranges),compare_span);
  size_t merged=0;
  for (size_t i=0;i<*count;i++) {
    span x=ranges[i];
    if (merged && x.begin<ranges[merged-1].end && x.node!=ranges[merged-1].node) {
      free(ranges); return ria_fail(e,RIA_INTEGRITY_ERROR,"different NUMA owners share physical expert pages");
    }
    if (merged && x.begin<=ranges[merged-1].end && x.node==ranges[merged-1].node) {
      if (x.end>ranges[merged-1].end) ranges[merged-1].end=x.end;
    } else ranges[merged++]=x;
  }
  *count=merged; *out=ranges; return true;
}
static bool bind_pages(unsigned node,void *p,uint64_t n,ria_error *e) {
  if (!n) return true;
  struct bitmask *mask=numa_allocate_nodemask();
  if (!mask) return ria_fail(e,RIA_RESOURCE_LIMIT,"NUMA policy mask allocation");
  numa_bitmask_setbit(mask,node);
  bool ok=mbind(p,(unsigned long)n,RIA_NUMA_BIND_MODE,mask->maskp,mask->size+1,0)==0;
  numa_bitmask_free(mask);
  return ok || ria_fail(e,RIA_RESOURCE_LIMIT,"pre-touch NUMA bind failed: %s",strerror(errno));
}
bool ria_numa_arena(unsigned node,uint64_t bytes,void **arena,uint64_t *mapped_bytes,ria_error *e) {
  *arena=NULL; *mapped_bytes=0;
  if (sysconf(_SC_PAGESIZE)!=4096 || !bytes || !rounded(bytes,mapped_bytes,e) || *mapped_bytes>SIZE_MAX)
    return ria_fail(e,RIA_UNSUPPORTED,"NUMA arenas require admitted 4096-byte base pages");
  /* MAP_ANONYMOUS ignores fd; -1 deliberately names no file. */
  // cppcheck-suppress invalidFunctionArg
  void *p=mmap(NULL,(size_t)*mapped_bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
  if (p==MAP_FAILED || !p) {
    if (!p) {
      /* Linux permits munmap at address0; release that valid rejected map. */
      // cppcheck-suppress [nullPointer, nullPointerRedundantCheck]
      munmap(p,(size_t)*mapped_bytes);
    }
    return ria_fail(e,RIA_RESOURCE_LIMIT,"NUMA arena allocation failed");
  }
  if (madvise(p,(size_t)*mapped_bytes,MADV_DONTDUMP) || madvise(p,(size_t)*mapped_bytes,MADV_NOHUGEPAGE) ||
      madvise(p,(size_t)*mapped_bytes,MADV_UNMERGEABLE) || !bind_pages(node,p,*mapped_bytes,e)) {
    munmap(p,(size_t)*mapped_bytes); return e->code ? false : ria_fail(e,RIA_RESOURCE_LIMIT,"NUMA arena page policy failed");
  }
  *arena=p; return true;
}
void ria_numa_release(void *p,uint64_t n) { if (p) munmap(p,(size_t)n); }
bool ria_numa_verify(unsigned node,const void *p,uint64_t n,ria_error *e) {
  if ((uintptr_t)p%PAGE_BYTES || n%PAGE_BYTES)
    return ria_fail(e,RIA_INVALID_REQUEST,"NUMA verification range is not page aligned");
  void *pages[RIA_NUMA_QUERY_PAGES]; int status[RIA_NUMA_QUERY_PAGES];
  for (uint64_t offset=0;offset<n;) {
    uint64_t left=(n-offset)/PAGE_BYTES;
    unsigned count=left>RIA_NUMA_QUERY_PAGES ? RIA_NUMA_QUERY_PAGES : (unsigned)left;
    for (unsigned i=0;i<count;i++) pages[i]=(uint8_t *)(uintptr_t)p+offset+(uint64_t)i*PAGE_BYTES;
    if (move_pages(0,count,pages,NULL,status,0)<0)
      return ria_fail(e,RIA_RESOURCE_LIMIT,"query-only NUMA locality verification failed: %s",strerror(errno));
    for (unsigned i=0;i<count;i++) if (status[i]!=(int)node)
      return ria_fail(e,RIA_RESOURCE_LIMIT,"NUMA page absent or placed on another node");
    offset+=(uint64_t)count*PAGE_BYTES;
  }
  return true;
}
bool ria_numa_place_shard(void *context,const ria_tensor_store *s,const ria_shard *a,bool populated,ria_error *e) {
  const ria_expert_config *c=context;
  if (!a) { ria_numa_account account; return ria_numa_account_store(s,c,&account,e); }
  span *ranges=NULL; size_t count=0; uint64_t bytes;
  if (!ria_numa_config_validate(c,e) || sysconf(_SC_PAGESIZE)!=4096 ||
      !rounded(a->length,&bytes,e) || !shard_spans(s,a,c,&ranges,&count,e)) return false;
  bool ok=true;
  if (!populated) {
    if (madvise(a->data,(size_t)bytes,MADV_NOHUGEPAGE) || madvise(a->data,(size_t)bytes,MADV_UNMERGEABLE))
      ok=ria_fail(e,RIA_RESOURCE_LIMIT,"canonical NUMA page policy failed");
    if (ok) ok=bind_pages(c->nodes[0].node,a->data,bytes,e);
    for (size_t i=0;ok && i<count;i++) if (ranges[i].node)
      ok=bind_pages(c->nodes[ranges[i].node].node,a->data+ranges[i].begin,ranges[i].end-ranges[i].begin,e);
  } else {
    uint64_t cursor=0;
    for (size_t i=0;ok && i<count;i++) {
      if (cursor<ranges[i].begin) ok=ria_numa_verify(c->nodes[0].node,a->data+cursor,ranges[i].begin-cursor,e);
      if (ok) ok=ria_numa_verify(c->nodes[ranges[i].node].node,a->data+ranges[i].begin,ranges[i].end-ranges[i].begin,e);
      cursor=ranges[i].end;
    }
    if (ok && cursor<bytes) ok=ria_numa_verify(c->nodes[0].node,a->data+cursor,bytes-cursor,e);
  }
  free(ranges); return ok;
}
bool ria_numa_topology(const ria_expert_config *c,ria_error *e) {
  if (!ria_numa_config_validate(c,e) || numa_available()<0 || sysconf(_SC_PAGESIZE)!=4096)
    return ria_fail(e,RIA_UNSUPPORTED,"required NUMA/base-page capability unavailable");
  int possible=numa_num_possible_cpus();
  if (possible<=0 || possible>65536) return ria_fail(e,RIA_UNSUPPORTED,"CPU topology exceeds bounded affinity mask");
  cpu_set_t *allowed=CPU_ALLOC((size_t)possible); size_t size=CPU_ALLOC_SIZE((size_t)possible);
  if (!allowed) return ria_fail(e,RIA_RESOURCE_LIMIT,"affinity mask allocation failed");
  bool ok=sched_getaffinity(0,size,allowed)==0;
  struct bitmask *mems=numa_get_mems_allowed();
  if (!mems) ok=ria_fail(e,RIA_RESOURCE_LIMIT,"allowed NUMA mask allocation");
  for (unsigned i=0;ok && i<c->node_count;i++) {
    const ria_expert_node *n=&c->nodes[i];
    struct bitmask *cpus=numa_allocate_cpumask();
    if (!cpus || !numa_bitmask_isbitset(mems,n->node) ||
        numa_node_to_cpus((int)n->node,cpus)) ok=ria_fail(e,RIA_UNSUPPORTED,"selected NUMA node is outside process policy");
    for (unsigned j=0;ok && j<n->cpu_count;j++) if (n->cpus[j]>=(unsigned)possible ||
        !CPU_ISSET_S(n->cpus[j],size,allowed) || !numa_bitmask_isbitset(cpus,n->cpus[j]))
      ok=ria_fail(e,RIA_UNSUPPORTED,"worker CPU is not allowed/local to selected NUMA node");
    if (cpus) numa_bitmask_free(cpus);
  }
  CPU_FREE(allowed);
  if (mems) numa_bitmask_free(mems);
  return ok || ria_fail(e,RIA_UNSUPPORTED,"cannot verify effective CPU affinity");
}
bool ria_numa_affinity(const ria_expert_node *n,unsigned worker,ria_error *e) {
  if (!n || worker>=n->workers || worker>=n->cpu_count)
    return ria_fail(e,RIA_INVALID_REQUEST,"worker affinity index invalid");
  unsigned cpu=n->cpus[worker]; cpu_set_t *set=CPU_ALLOC((size_t)cpu+1);
  if (!set) return ria_fail(e,RIA_RESOURCE_LIMIT,"worker affinity allocation");
  size_t size=CPU_ALLOC_SIZE((size_t)cpu+1); CPU_ZERO_S(size,set); CPU_SET_S(cpu,size,set);
  int code=pthread_setaffinity_np(pthread_self(),size,set); CPU_FREE(set);
  return !code || ria_fail(e,RIA_RESOURCE_LIMIT,"worker affinity failed: %s",strerror(code));
}
static bool routed(const ria_tensor *t) { return strstr(t->name,".ffn.experts.")!=NULL; }
bool ria_numa_account_store(const ria_tensor_store *s,const ria_expert_config *c,ria_numa_account *a,ria_error *e) {
  memset(a,0,sizeof(*a));
  if (!ria_numa_config_validate(c,e)) return false;
  uint64_t expert_bytes=0;
  for (uint64_t i=0;i<s->shard_count;i++) {
    const ria_shard *sh=&s->shards[i]; uint64_t n;
    span *ranges=NULL; size_t count=0;
    if (!rounded(sh->length,&n,e) || !shard_spans(s,sh,c,&ranges,&count,e)) return false;
    if (!ria_u64_add(a->canonical_bytes,n,&a->canonical_bytes) || !ria_u64_add(a->node_bytes[0],n,&a->node_bytes[0])) { free(ranges); goto overflow; }
    for (size_t j=0;j<count;j++) if (ranges[j].node) {
      uint64_t length=ranges[j].end-ranges[j].begin;
      a->node_bytes[0]-=length;
      if (!ria_u64_add(a->node_bytes[ranges[j].node],length,&a->node_bytes[ranges[j].node])) { free(ranges); goto overflow; }
    }
    free(ranges);
  }
  for (uint64_t i=0;i<s->tensor_count;i++) if (routed(&s->tensors[i]) && !s->tensors[i].alias) {
    uint64_t n;
    if (!rounded(s->tensors[i].length,&n,e) || !ria_u64_add(expert_bytes,n,&expert_bytes)) goto overflow;
  }
  for (unsigned i=1;i<c->node_count;i++) {
    uint64_t n=c->policy==RIA_NUMA_REPLICATED_EXPERTS ? expert_bytes :
               c->policy==RIA_NUMA_REPLICATED_SERVER_MODEL ? a->canonical_bytes : 0;
    if (!ria_u64_add(a->node_bytes[i],n,&a->node_bytes[i]) || !ria_u64_add(a->replica_bytes,n,&a->replica_bytes)) goto overflow;
  }
  if (!ria_u64_add(a->canonical_bytes,a->replica_bytes,&a->total_bytes) ||
      !ria_u64_add(a->total_bytes,c->host_runtime_bytes,&a->total_bytes)) goto overflow;
  /* Worker arenas are physically bound to their pool's node. Network, TLS,
   * metadata and the lifecycle owner live on the first selected node. */
  uint64_t worker_bytes;
  if (!ria_u64_mul(c->worker_count,UINT64_C(2097152),&worker_bytes) || worker_bytes>c->host_runtime_bytes)
    return ria_fail(e,RIA_RESOURCE_LIMIT,"host runtime reserve cannot hold fixed node-local worker arenas");
  for (unsigned i=0;i<c->node_count;i++) {
    uint64_t n;
    if (!ria_u64_mul(c->nodes[i].workers,UINT64_C(2097152),&n)) goto overflow;
    if (!i && !ria_u64_add(n,c->host_runtime_bytes-worker_bytes,&n)) goto overflow;
    if (!ria_u64_add(a->node_bytes[i],n,&a->node_bytes[i]) || a->node_bytes[i]>c->nodes[i].local_bytes)
      return ria_fail(e,RIA_RESOURCE_LIMIT,"physical NUMA population/reserve exceeds local admitted budget");
  }
  if (a->total_bytes>c->startup_host_bytes)
    return ria_fail(e,RIA_RESOURCE_LIMIT,"canonical/replica/startup reserve exceeds admitted startup budget");
  return true;
overflow:
  return ria_fail(e,RIA_RESOURCE_LIMIT,"NUMA physical allocation accounting overflow");
}
/* Replica storage consists of independent prebound anonymous arenas. The
 * original canonical pages are replica zero and remain the chunk authority. */
bool ria_numa_open(ria_numa **out,const ria_bank *b,const ria_expert_config *c,ria_error *e) {
  return ria_numa_open_controlled(out,b,c,NULL,NULL,e);
}
bool ria_numa_open_controlled(ria_numa **out,const ria_bank *b,const ria_expert_config *c,ria_tensor_progress progress,void *progress_context,ria_error *e) {
  *out=NULL; ria_numa_account account;
  if (!b || !b->store || !ria_numa_account_store(b->store,c,&account,e)) return false;
  ria_numa *n=calloc(1,sizeof(*n));
  if (!n) return ria_fail(e,RIA_RESOURCE_LIMIT,"NUMA view owner allocation");
  n->bank=b; n->config=*c;
  for (unsigned node=0;node<c->node_count;node++) {
    replica *r=&n->nodes[node];
    r->tables[0]=b->engram[0]->data; r->tables[1]=b->engram[1]->data;
    if (!node || c->policy==RIA_NUMA_SHARDED) continue;
    uint64_t bytes=c->policy==RIA_NUMA_REPLICATED_SERVER_MODEL ? account.canonical_bytes : 0;
    if (c->policy==RIA_NUMA_REPLICATED_EXPERTS) for (uint64_t i=0;i<b->store->tensor_count;i++) {
      const ria_tensor *t=&b->store->tensors[i]; uint64_t length;
      if (!routed(t) || t->alias) continue;
      if (!rounded(t->length,&length,e) || !ria_u64_add(bytes,length,&bytes)) goto fail;
    }
    if (!ria_numa_arena(c->nodes[node].node,bytes,&r->arena,&r->bytes,e)) goto fail;
    r->experts=malloc(40*384*sizeof(*r->experts));
    if (!r->experts) { ria_error_set(e,RIA_RESOURCE_LIMIT,"replica descriptor allocation"); goto fail; }
    memcpy(r->experts,b->experts,40*384*sizeof(*r->experts)); memcpy(r->shared,b->shared,sizeof(r->shared));
    size_t map_capacity=(size_t)(c->policy==RIA_NUMA_REPLICATED_SERVER_MODEL ? b->store->shard_count : b->store->tensor_count);
    relocation *map=calloc(map_capacity,sizeof(*map)); size_t map_count=0;
    if (!map) { ria_error_set(e,RIA_RESOURCE_LIMIT,"replica relocation index allocation"); goto fail; }
    uint64_t cursor=0;
    for (uint64_t i=0;i<(c->policy==RIA_NUMA_REPLICATED_SERVER_MODEL ? b->store->shard_count : b->store->tensor_count);i++) {
      const uint8_t *source; uint64_t length, extent;
      if (c->policy==RIA_NUMA_REPLICATED_SERVER_MODEL) { source=b->store->shards[i].data; length=b->store->shards[i].length; }
      else { const ria_tensor *t=&b->store->tensors[i]; if (!routed(t) || t->alias) continue; source=t->data; length=t->length; }
      if (!rounded(length,&extent,e) || cursor>r->bytes || extent>r->bytes-cursor) { free(map); goto fail; }
      uint8_t *target=(uint8_t *)r->arena+cursor;
      for (uint64_t offset=0;offset<length;) {
        uint64_t count=length-offset; if (count>RIA_BULK_MAX) count=RIA_BULK_MAX;
        if (progress && !progress(progress_context,e)) { free(map); goto fail; }
        memcpy(target+offset,source+offset,(size_t)count);
        uint8_t a[32],z[32];
        if (!ria_sha256(source+offset,(size_t)count,a,e) || !ria_sha256(target+offset,(size_t)count,z,e) || memcmp(a,z,32)) {
          free(map); ria_error_set(e,RIA_INTEGRITY_ERROR,"independent replica copy SHA mismatch"); goto fail;
        }
        offset+=count;
      }
      memset(target+length,0,(size_t)(extent-length));
      map[map_count++]=(relocation){(uintptr_t)source,(uintptr_t)source+(uintptr_t)length,target};
      cursor+=extent;
    }
    qsort(map,map_count,sizeof(*map),compare_relocation);
    for (unsigned x=0;x<40*384+40;x++) {
      ria_expert *expert=x<40*384 ? &r->experts[x] : &r->shared[x-40*384];
      ria_expert_matrix *matrices[3]={&expert->gate,&expert->up,&expert->down};
      for (unsigned j=0;j<3;j++) {
        matrices[j]->values=relocate(map,map_count,matrices[j]->values);
        matrices[j]->scales=relocate(map,map_count,matrices[j]->scales);
      }
    }
    for (unsigned j=0;j<2;j++) r->tables[j]=relocate(map,map_count,r->tables[j]);
    free(map);
    if (cursor!=r->bytes || !ria_numa_verify(c->nodes[node].node,r->arena,r->bytes,e) || mprotect(r->arena,(size_t)r->bytes,PROT_READ)) {
      if (!e->code) ria_error_set(e,RIA_RESOURCE_LIMIT,"replica locality/protection failed");
      goto fail;
    }
  }
  *out=n; return true;
fail:
  ria_numa_close(n); return false;
}
void ria_numa_close(ria_numa *n) {
  if (!n) return;
  for (unsigned i=0;i<n->config.node_count;i++) { ria_numa_release(n->nodes[i].arena,n->nodes[i].bytes); free(n->nodes[i].experts); }
  free(n);
}
const ria_expert *ria_numa_expert(const ria_numa *n,unsigned node,uint64_t handle,uint16_t id) {
  if (!n || node>=n->config.node_count || !handle || handle>80) return NULL;
  const replica *r=&n->nodes[node];
  if (handle>40) return id==0 && n->bank->shared_present[handle-41] ?
      (r->experts && n->config.policy==RIA_NUMA_REPLICATED_SERVER_MODEL ? &r->shared[handle-41] : &n->bank->shared[handle-41]) : NULL;
  if (id>=384) return NULL;
  return r->experts ? &r->experts[(handle-1)*384+id] : &n->bank->experts[(handle-1)*384+id];
}
const uint8_t *ria_numa_table(const ria_numa *n,unsigned node,unsigned table) {
  return n && node<n->config.node_count && table<2 ? n->nodes[node].tables[table] : NULL;
}
