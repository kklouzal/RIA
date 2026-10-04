#define _GNU_SOURCE
#include "graph.h"
#include "graph_cuda.h"
#include "vision.h"
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

typedef struct {
    ria_expert_matrix hc_attn,hc_ffn,wqa,wqb,wkv,woa,wob,router,shared_gate,shared_up,shared_down;
    ria_expert_matrix compressor_kv,compressor_gate,index_q,index_weights,index_key,engram_kv;
    const ria_tensor *attn_norm,*ffn_norm,*q_norm,*kv_norm,*sink,*attn_scale,*attn_base,*ffn_scale,*ffn_base;
    const ria_tensor *bias,*bias_vl,*compressor_norm,*index_norm,*engram_q,*engram_k;
    uint32_t ratio,kv_source,index_source;
} graph_layer;
typedef ria_graph_row_cache_entry engram_cache_entry;
typedef struct { char padding;engram_cache_entry entry; } engram_cache_alignment;
#define RIA_CACHE_ALIGNMENT offsetof(engram_cache_alignment,entry)
struct ria_graph {
    const ria_tensor_store *store;
    ria_graph_options options;
    ria_graph_remote remote;
    graph_layer layers[RIA_GRAPH_LAYERS];
    ria_expert_matrix head;
    const ria_tensor *embedding,*norm,*image_delimiters[3];
    ria_graph_cuda *cuda;
    ria_vision *vision;
    uint8_t *host_allocation,*windows[40],*compressed[4],*index_keys[4];
    float *pool_kv[4],*pool_gate[4];
    uint64_t host_bytes,owner_bytes,state_bytes,epoch,generation,position;
    int64_t history[4];
    uint32_t selected[512],selected_count;
    ria_graph_phase phase;
    int32_t local_index[40][384];
    uint64_t local_device_bytes,engram_capacity;
    engram_cache_entry *engram_cache;
    float input[5120],contributions[6*5120],remote_contributions[6*5120];
    uint8_t packed_rows[24*264],packed_window[128*528],packed_selected[512*288];
    bool poisoned,locked,owner_locked;
    ria_expert_cuda_resident local_device[];
};
static const uint32_t kv_sources[4]={2,8,14,20},index_sources[8]={2,8,14,20,24,28,32,36};
bool ria_graph_client_device_validate(int count,int device,int major,int minor,
                                     const char *name,const char *uuid,const char *expected_uuid,ria_error *e) {
    if (count!=1 || device!=0 || major!=12 || minor!=0 || !name || strcmp(name,"NVIDIA GeForce RTX 5090"))
        return ria_fail(e,RIA_UNSUPPORTED,"client requires one RTX 5090 at device 0 with physical CC12.0");
    if (!uuid || !expected_uuid || strcmp(uuid,expected_uuid))
        return ria_fail(e,RIA_IDENTITY_MISMATCH,"graph device UUID differs from admission");
    return true;
}
static bool page_round(uint64_t bytes,uint64_t *rounded,ria_error *e) {
    long page=sysconf(_SC_PAGESIZE);uint64_t sum;
    if (page<=0 || !ria_u64_add(bytes,(uint64_t)page-1,&sum) || sum>SIZE_MAX)
        return ria_fail(e,RIA_RESOURCE_LIMIT,"private graph allocation rounding overflow");
    *rounded=sum/(uint64_t)page*(uint64_t)page;return true;
}
static bool private_sizes(uint64_t max_tokens,uint32_t locals,uint64_t cache_budget,uint64_t *owner,uint64_t *state,ria_error *e) {
    uint64_t bytes=40*128*528;
    if (!max_tokens || max_tokens>1048576) return ria_fail(e,RIA_INVALID_REQUEST,"invalid graph state extent");
    for (unsigned i=0;i<4;++i) {
        uint64_t count=max_tokens/(i==3 ? 1 : 2),part;
        if (!ria_u64_mul(count,356,&part) || !ria_u64_add(bytes,part,&bytes) || !ria_u64_add(bytes,8192,&bytes))
            return ria_fail(e,RIA_RESOURCE_LIMIT,"graph private state size overflow");
    }
    uint64_t metadata,cache=cache_budget/sizeof(engram_cache_entry)*sizeof(engram_cache_entry);
    if (locals>40*384 || !ria_u64_mul(locals,sizeof(ria_expert_cuda_resident),&metadata) ||
        !ria_u64_add(metadata,sizeof(ria_graph),&metadata) ||
        !ria_u64_add(bytes,cache ? RIA_CACHE_ALIGNMENT-1 : 0,&bytes) || !ria_u64_add(bytes,cache,&bytes))
        return ria_fail(e,RIA_RESOURCE_LIMIT,"local metadata/Engram cache allocation overflow");
    return page_round(metadata,owner,e) && page_round(bytes,state,e);
}
bool ria_graph_host_state_required(uint64_t max_tokens,uint64_t *bytes,ria_error *e) {
    uint64_t owner,state;
    if (!bytes) return ria_fail(e,RIA_INVALID_REQUEST,"missing graph state size result");
    if (!private_sizes(max_tokens,0,0,&owner,&state,e)) return false;
    return ria_u64_add(owner,state,bytes) || ria_fail(e,RIA_RESOURCE_LIMIT,"graph total private size overflow");
}
bool ria_graph_host_required_bytes(const ria_graph_options *o,uint64_t *bytes,ria_error *e) {
    uint64_t owner,state;
    if (!o || !bytes || !private_sizes(o->max_tokens,o->local_expert_count,o->engram_cache_budget,&owner,&state,e)) return false;
    return ria_u64_add(owner,state,bytes) || ria_fail(e,RIA_RESOURCE_LIMIT,"complete host graph size overflow");
}
bool ria_graph_local_experts_validate(const ria_graph_options *o,uint64_t *host,uint64_t *device,ria_error *e) {
    if (!o || !host || !device || o->local_expert_count>40*384 || (o->local_expert_count && !o->local_experts))
        return ria_fail(e,RIA_INVALID_REQUEST,"invalid explicit local expert population");
    *host=0;*device=0;uint32_t previous=0;
    for (uint32_t i=0;i<o->local_expert_count;++i) {
        const ria_graph_local_expert *x=&o->local_experts[i];uint32_t key=x->layer*384u+x->expert_id;uint64_t bytes;
        if (x->layer>=40 || x->expert_id>=384 || (i && key<=previous) ||
            (x->tier!=RIA_GRAPH_HOST && x->tier!=RIA_GRAPH_VRAM) || !x->phase_mask || x->phase_mask>7 ||
            x->expert.gate.in_features!=5120 || x->expert.gate.out_features!=2304 || x->expert.down.out_features!=5120 ||
            x->expert.clamp!=10 || !ria_expert_cuda_resident_required_bytes(&x->expert,&bytes,e))
            return ria_fail(e,RIA_INTEGRITY_ERROR,"local expert membership/profile/source shape is invalid");
        previous=key;
        const ria_expert_matrix *m[3]={&x->expert.gate,&x->expert.up,&x->expert.down};
        for (unsigned j=0;j<3;++j) if (!ria_u64_add(*host,m[j]->values_bytes,host) || !ria_u64_add(*host,m[j]->scales_bytes,host))
            return ria_fail(e,RIA_RESOURCE_LIMIT,"local host expert population overflow");
        if (x->tier==RIA_GRAPH_VRAM && !ria_u64_add(*device,bytes,device))
            return ria_fail(e,RIA_RESOURCE_LIMIT,"local device expert population overflow");
    }
    if (*host>o->host_expert_budget || *device>o->device_expert_budget ||
        (o->engram_cache_budget && o->engram_cache_budget<sizeof(engram_cache_entry)))
        return ria_fail(e,RIA_RESOURCE_LIMIT,"explicit local expert/row population exceeds cache reservation");
    return true;
}
static int32_t local_lookup(const ria_graph_options *o,uint32_t layer,uint16_t id,ria_graph_phase phase) {
    uint32_t key=layer*384u+id,lo=0,hi=o->local_expert_count;
    while (lo<hi) { uint32_t mid=lo+(hi-lo)/2;const ria_graph_local_expert *x=&o->local_experts[mid];
        uint32_t value=x->layer*384u+x->expert_id;if (value<key) lo=mid+1;else hi=mid; }
    if (lo<o->local_expert_count) { const ria_graph_local_expert *x=&o->local_experts[lo];
        if (x->layer==layer && x->expert_id==id && (x->phase_mask&(1u<<(unsigned)phase))) return (int32_t)lo; }
    return -1;
}
bool ria_graph_rows_fetch(ria_graph_remote remote,uint32_t layer,const uint64_t *ids,uint32_t count,
                          ria_graph_row_cache_entry *cache,uint64_t capacity,uint8_t *packed,ria_error *e) {
    uint64_t population=layer==1 ? UINT64_C(384006168) : layer==14 ? UINT64_C(384016682) : 0;
    if (!population || !ids || !packed || !remote.engram || !count || count>24 || (capacity && !cache) || capacity>SIZE_MAX/sizeof(*cache))
        return ria_fail(e,RIA_INVALID_REQUEST,"invalid exact Engram cache lookup");
    uint64_t misses[24];uint8_t slots[24],reply[24*264];uint32_t missing=0;
    for (uint32_t slot=0;slot<count;++slot) {
        if (ids[slot]>=population) return ria_fail(e,RIA_INVALID_REQUEST,"Engram row is outside original table");
        uint64_t bucket=capacity ? (ids[slot]^((uint64_t)layer*UINT64_C(0x9e3779b97f4a7c15)))%capacity : 0;
        const ria_graph_row_cache_entry *entry=capacity ? &cache[bucket] : NULL;
        if (entry && entry->valid && entry->layer==layer && entry->row==ids[slot]) memcpy(packed+(uint64_t)slot*264,entry->packed,264);
        else { misses[missing]=ids[slot];slots[missing++]=(uint8_t)slot; }
    }
    if (missing && !remote.engram(remote.context,layer,misses,missing,reply,e)) return false;
    for (uint32_t i=0;i<missing;++i) {
        memcpy(packed+(uint64_t)slots[i]*264,reply+(uint64_t)i*264,264);
        if (capacity) {
            uint64_t bucket=(misses[i]^((uint64_t)layer*UINT64_C(0x9e3779b97f4a7c15)))%capacity;
            ria_graph_row_cache_entry *entry=&cache[bucket];entry->row=misses[i];entry->layer=layer;memcpy(entry->packed,reply+(uint64_t)i*264,264);entry->valid=true;
        }
    }
    return true;
}
bool ria_graph_partition(const ria_graph_options *o,uint32_t layer,ria_graph_phase phase,const uint16_t *ids,uint32_t count,
                         int32_t *local,uint8_t *slots,uint32_t *remote,ria_error *e) {
    if (!o || layer>=40 || phase<RIA_GRAPH_PREFILL || phase>RIA_GRAPH_CONTINUATION || !ids || !local || !slots || !remote || !count || count>6 ||
        o->local_expert_count>40*384 || (o->local_expert_count && !o->local_experts))
        return ria_fail(e,RIA_INVALID_REQUEST,"invalid selected-slot partition contract");
    *remote=0;
    for (uint32_t slot=0;slot<count;++slot) {
        if (ids[slot]>=384) return ria_fail(e,RIA_INVALID_REQUEST,"selected expert is outside original population");
        for (uint32_t earlier=0;earlier<slot;++earlier) if (ids[earlier]==ids[slot]) return ria_fail(e,RIA_INVALID_REQUEST,"duplicate original selected expert");
        local[slot]=local_lookup(o,layer,ids[slot],phase);
        if (local[slot]<0) slots[(*remote)++]=(uint8_t)slot;
    }
    return true;
}
bool ria_graph_pinned_required_bytes(const ria_graph_options *o,uint64_t *bytes,ria_error *e) {
    uint64_t graph,vision=0;
    if (!o || !bytes || !ria_expert_cuda_pinned_required_bytes(32768,32768,129280,1,o->projection_tile_rows,&graph,e)) return false;
    if (o->max_image_patches && !ria_expert_cuda_pinned_required_bytes(9216,5632,5120,64,128,&vision,e)) return false;
    return ria_u64_add(graph,vision,bytes) || ria_fail(e,RIA_RESOURCE_LIMIT,"complete graph pinned reservation overflow");
}
bool ria_graph_dependencies(uint32_t layer,uint32_t *kv,uint32_t *index,uint32_t *ratio,ria_error *e) {
    if (layer>=40 || !kv || !index || !ratio) return ria_fail(e,RIA_INVALID_REQUEST,"invalid graph layer/dependency result");
    *kv=UINT32_MAX;*index=UINT32_MAX;*ratio=layer<2 ? 0 : layer<20 ? 2 : 1;
    for (unsigned i=0;i<4;++i) if (kv_sources[i]<=layer) *kv=kv_sources[i];
    for (unsigned i=0;i<8;++i) if (index_sources[i]<=layer) *index=index_sources[i];
    return true;
}
bool ria_graph_attention_kind_for_layer(uint32_t layer,ria_graph_attention_kind *kind,ria_error *e) {
    uint32_t kv,index,ratio;
    if (!kind || !ria_graph_dependencies(layer,&kv,&index,&ratio,e)) return false;
    *kind=!ratio ? RIA_GRAPH_WINDOW : kv==layer ? RIA_GRAPH_FULL : index==layer ? RIA_GRAPH_REINDEX : RIA_GRAPH_REUSE;
    return true;
}
bool ria_graph_options_validate(const ria_graph_options *o,ria_error *e) {
    if (!o || o->device!=0 || !o->gpu_uuid || strncmp(o->gpu_uuid,"GPU-",4) || strlen(o->gpu_uuid)!=40 ||
        !o->max_tokens || o->max_tokens>1048576 || !o->host_state_budget || !o->device_budget || !o->pinned_budget ||
        !o->projection_tile_rows || o->projection_tile_rows>4096 || !o->state_tile_rows || o->state_tile_rows>4096 ||
        o->max_image_patches>9216 || !o->compressed_token_map || o->compressed_vocab!=99092 || o->pad_compressed_id>=99092)
        return ria_fail(e,RIA_INVALID_REQUEST,"graph options differ from pinned V4.1 contract/admission");
    uint64_t pinned;
    if (!ria_graph_pinned_required_bytes(o,&pinned,e) || pinned>o->pinned_budget)
        return ria_fail(e,RIA_RESOURCE_LIMIT,"complete graph/vision pinned pools exceed pinned admission");
    uint64_t local_host,local_device;
    if (!ria_graph_local_experts_validate(o,&local_host,&local_device,e)) return false;
    for (unsigned i=4;i<40;++i) {
        char ch=o->gpu_uuid[i];bool separator=i==12 || i==17 || i==22 || i==27;
        if (separator ? ch!='-' : !((ch>='0' && ch<='9') || (ch>='a' && ch<='f')))
            return ria_fail(e,RIA_INVALID_REQUEST,"graph requires canonical physical GPU UUID");
    }
    for (unsigned i=0;i<129280;++i) if (o->compressed_token_map[i]>=99092)
        return ria_fail(e,RIA_INTEGRITY_ERROR,"Engram compressed token map is out of range");
    for (unsigned layer=0;layer<2;++layer) {
        uint64_t sum=0;
        for (unsigned i=0;i<4;++i) if (!(o->hash_multipliers[layer][i]&1) ||
            o->hash_multipliers[layer][i]>(uint64_t)INT64_MAX/99091)
            return ria_fail(e,RIA_INTEGRITY_ERROR,"Engram multiplier violates signed product bounds");
        for (unsigned i=0;i<24;++i) {
            uint64_t prime=o->hash_primes[layer][i];
            if (prime<16000000 || prime>17000000 || !ria_u64_add(sum,prime,&sum))
                return ria_fail(e,RIA_INTEGRITY_ERROR,"Engram bucket constant is invalid");
        }
        if (sum!=(layer ? UINT64_C(384016682) : UINT64_C(384006168)))
            return ria_fail(e,RIA_INTEGRITY_ERROR,"Engram bucket population differs from source");
    }
    return true;
}
bool ria_graph_hash(const ria_graph_options *o,uint32_t layer,const int64_t history[4],uint64_t ids[24],ria_error *e) {
    if (!o || !history || !ids || layer>=2) return ria_fail(e,RIA_INVALID_REQUEST,"invalid Engram hash arguments");
    uint64_t rolling=0,offset=0;bool blocked=false;
    for (unsigned lookback=0;lookback<4;++lookback) {
        blocked=blocked || history[lookback]<0;
        uint64_t token=blocked ? o->pad_compressed_id : (uint64_t)history[lookback];
        if (token>=o->compressed_vocab || !o->hash_multipliers[layer][lookback] ||
            token>(uint64_t)INT64_MAX/o->hash_multipliers[layer][lookback])
            return ria_fail(e,RIA_INTEGRITY_ERROR,"Engram hash product is not representable");
        rolling^=token*o->hash_multipliers[layer][lookback];
        if (lookback) for (unsigned head=0;head<8;++head) {
            unsigned column=(lookback-1)*8+head;uint64_t prime=o->hash_primes[layer][column];
            if (!prime || !ria_u64_add(offset,rolling%prime,&ids[column]) || !ria_u64_add(offset,prime,&offset))
                return ria_fail(e,RIA_INTEGRITY_ERROR,"Engram hash bucket overflows");
        }
    }
    return true;
}
static const ria_tensor *required(const ria_tensor_store *s,const char *name,ria_error *e) {
    const ria_tensor *t=ria_tensor_name(s,name);
    if (!t) (void)ria_fail(e,RIA_INTEGRITY_ERROR,"required graph tensor missing: %s",name);
    return t;
}
static const ria_tensor *layer_tensor(const ria_tensor_store *s,unsigned layer,const char *suffix,ria_error *e) {
    char name[192];int n=snprintf(name,sizeof(name),"layers.%u.%s",layer,suffix);
    if (n<0 || (size_t)n>=sizeof(name)) { (void)ria_fail(e,RIA_INTERNAL_ERROR,"graph tensor name exceeds schema");return NULL; }
    return required(s,name,e);
}
static bool bind_matrix(const ria_tensor_store *s,unsigned layer,const char *name,uint64_t n,uint64_t k,
                        ria_expert_matrix *m,ria_error *e) {
    const ria_tensor *t=layer_tensor(s,layer,name,e);
    if (!t || !ria_tensor_matrix(s,t,m,e)) return false;
    if (m->out_features!=n || m->in_features!=k) return ria_fail(e,RIA_INTEGRITY_ERROR,"graph matrix shape differs: layer %u %s",layer,name);
    return true;
}
static bool bind_vector(const ria_tensor_store *s,unsigned layer,const char *name,uint64_t elements,
                        const ria_tensor **out,ria_error *e) {
    const ria_tensor *t=layer_tensor(s,layer,name,e);uint64_t count=1;
    if (!t) return false;
    for (unsigned i=0;i<t->rank;++i) if (!ria_u64_mul(count,t->shape[i],&count)) return ria_fail(e,RIA_INTEGRITY_ERROR,"graph vector shape overflow");
    if (count!=elements || (strcmp(t->format,"plain") && strcmp(t->format,"bf16") && strcmp(t->format,"f32")) ||
        (strcmp(t->dtype,"BF16") && strcmp(t->dtype,"F32")))
        return ria_fail(e,RIA_INTEGRITY_ERROR,"graph vector shape/precision differs: layer %u %s",layer,name);
    *out=t;return true;
}
static bool bind_layer(ria_graph *g,unsigned l,ria_error *e) {
    graph_layer *x=&g->layers[l];const ria_tensor_store *s=g->store;
    if (!ria_graph_dependencies(l,&x->kv_source,&x->index_source,&x->ratio,e)) return false;
#define MATRIX(member,name,n,k) if (!bind_matrix(s,l,name,n,k,&x->member,e)) return false
#define VECTOR(member,name,count) if (!bind_vector(s,l,name,count,&x->member,e)) return false
    MATRIX(hc_attn,"hc_attn_fn",24,20480);MATRIX(hc_ffn,"hc_ffn_fn",24,20480);
    VECTOR(attn_scale,"hc_attn_scale",3);VECTOR(attn_base,"hc_attn_base",24);
    VECTOR(ffn_scale,"hc_ffn_scale",3);VECTOR(ffn_base,"hc_ffn_base",24);
    VECTOR(attn_norm,"attn_norm.weight",5120);VECTOR(ffn_norm,"ffn_norm.weight",5120);
    MATRIX(wqa,"attn.wq_a.weight",1280,5120);MATRIX(wqb,"attn.wq_b.weight",32768,1280);
    MATRIX(wkv,"attn.wkv.weight",512,5120);MATRIX(woa,"attn.wo_a.weight",8192,4096);MATRIX(wob,"attn.wo_b.weight",5120,8192);
    VECTOR(q_norm,"attn.q_norm.weight",1280);VECTOR(kv_norm,"attn.kv_norm.weight",512);VECTOR(sink,"attn.attn_sink",64);
    MATRIX(router,"ffn.gate.weight",384,5120);VECTOR(bias,"ffn.gate.bias",384);VECTOR(bias_vl,"ffn.gate.bias_vl",384);
    MATRIX(shared_gate,"ffn.shared_experts.w1.weight",2304,5120);
    MATRIX(shared_up,"ffn.shared_experts.w3.weight",2304,5120);
    MATRIX(shared_down,"ffn.shared_experts.w2.weight",5120,2304);
    if (x->kv_source==l) {
        MATRIX(compressor_kv,"attn.compressor.wkv.weight",512,5120);
        VECTOR(compressor_norm,"attn.compressor.norm.weight",512);
        if (x->ratio>1) MATRIX(compressor_gate,"attn.compressor.wgate.weight",512,5120);
        MATRIX(index_key,"attn.indexer.wk.weight",128,512);VECTOR(index_norm,"attn.indexer.k_norm.weight",128);
    }
    if (x->index_source==l) {
        MATRIX(index_q,"attn.indexer.wq_b.weight",4096,1280);MATRIX(index_weights,"attn.indexer.weights_proj.weight",32,5120);
    }
    if (l==1 || l==14) {
        MATRIX(engram_kv,"engram.wkv.weight",25600,6144);
        VECTOR(engram_q,"engram.q_weight",20480);VECTOR(engram_k,"engram.k_weight",20480);
    }
#undef MATRIX
#undef VECTOR
    if (x->hc_attn.profile!=RIA_EXPERT_F32 || x->hc_ffn.profile!=RIA_EXPERT_F32 ||
        x->woa.profile!=RIA_EXPERT_BF16 || (x->router.profile!=RIA_EXPERT_BF16 && x->router.profile!=RIA_EXPERT_F32) ||
        (x->kv_source==l && (x->index_key.profile!=RIA_EXPERT_BF16 || (x->ratio==1 && x->compressor_kv.profile!=RIA_EXPERT_BF16))) ||
        (x->index_source==l && x->index_weights.profile!=RIA_EXPERT_BF16) ||
        (x->kv_source==l && x->ratio>1 && (x->compressor_kv.profile!=RIA_EXPERT_F32 || x->compressor_gate.profile!=RIA_EXPERT_F32)))
        return ria_fail(e,RIA_INTEGRITY_ERROR,"sensitive graph projection requires source FP32 prepared values");
    return true;
}
static unsigned source_slot(uint32_t source) { for (unsigned i=0;i<4;++i) if (kv_sources[i]==source) return i;return 4; }
static bool allocate_private(ria_graph *g,ria_error *e) {
    /* MAP_ANONYMOUS ignores fd; -1 prevents an accidental file mapping. */
    // cppcheck-suppress invalidFunctionArg
    g->host_allocation=mmap(NULL,(size_t)g->state_bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if (g->host_allocation==MAP_FAILED) { g->host_allocation=NULL;return ria_fail(e,RIA_RESOURCE_LIMIT,"allocate complete private graph backing failed"); }
    if (!g->host_allocation) {
        /* Linux munmap accepts virtual address zero without dereferencing it. */
        // cppcheck-suppress [nullPointer,nullPointerRedundantCheck]
        if (munmap(g->host_allocation,(size_t)g->state_bytes)!=0)
            return ria_fail(e,RIA_RESOURCE_LIMIT,"release unusable null private graph mapping failed");
        return ria_fail(e,RIA_RESOURCE_LIMIT,"private graph backing mapped an unusable null address");
    }
    if (madvise(g->host_allocation,(size_t)g->state_bytes,MADV_DONTDUMP)!=0 || mlock(g->host_allocation,(size_t)g->state_bytes)!=0)
        return ria_fail(e,RIA_RESOURCE_LIMIT,"protect/lock private graph backing failed");
    g->locked=true;uint8_t *cursor=g->host_allocation;
    for (unsigned i=0;i<40;++i) { g->windows[i]=cursor;cursor+=128*528; }
    for (unsigned i=0;i<4;++i) {
        uint64_t count=g->options.max_tokens/(i==3 ? 1 : 2);
        g->compressed[i]=cursor;cursor+=count*288;g->index_keys[i]=cursor;cursor+=count*68;
        /* mmap owns untyped storage, not encoded float bytes. Every preceding
         * packed stride is divisible by four; each pool is float-aligned and
         * obtains its effective type through float stores/copies. */
        // cppcheck-suppress invalidPointerCast
        g->pool_kv[i]=(float *)cursor;cursor+=4096;
        // cppcheck-suppress invalidPointerCast
        g->pool_gate[i]=(float *)cursor;cursor+=4096;
    }
    g->engram_capacity=g->options.engram_cache_budget/sizeof(engram_cache_entry);
    if (g->engram_capacity) { uintptr_t aligned=((uintptr_t)cursor+RIA_CACHE_ALIGNMENT-1)/RIA_CACHE_ALIGNMENT*RIA_CACHE_ALIGNMENT;
        g->engram_cache=(engram_cache_entry *)aligned; }
    return true;
}
bool ria_graph_create(const ria_tensor_store *store,const ria_graph_options *options,ria_graph_remote remote,ria_graph **out,ria_error *e) {
    if (!out) return ria_fail(e,RIA_INVALID_REQUEST,"missing graph output");
    *out=NULL;
    if (!store || strcmp(store->role,"client") || !remote.experts || !remote.engram)
        return ria_fail(e,RIA_INVALID_REQUEST,"graph requires client store and authenticated remote callbacks");
    if (!ria_graph_options_validate(options,e)) return false;
    uint64_t owner,state,total;
    if (!private_sizes(options->max_tokens,options->local_expert_count,options->engram_cache_budget,&owner,&state,e) || !ria_u64_add(owner,state,&total)) return false;
    if (total>options->host_state_budget) return ria_fail(e,RIA_RESOURCE_LIMIT,"complete private graph state exceeds host budget");
    /* MAP_ANONYMOUS ignores fd; -1 prevents an accidental file mapping. */
    // cppcheck-suppress invalidFunctionArg
    ria_graph *g=mmap(NULL,(size_t)owner,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if (g==MAP_FAILED) return ria_fail(e,RIA_RESOURCE_LIMIT,"graph owner allocation failed");
    if (!g) {
        /* Linux munmap accepts virtual address zero without dereferencing it. */
        // cppcheck-suppress [nullPointer,nullPointerRedundantCheck]
        if (munmap(g,(size_t)owner)!=0)
            return ria_fail(e,RIA_RESOURCE_LIMIT,"release unusable null graph owner mapping failed");
        return ria_fail(e,RIA_RESOURCE_LIMIT,"graph owner mapped an unusable null address");
    }
    g->owner_bytes=owner;g->state_bytes=state;g->host_bytes=total;
    if (madvise(g,(size_t)owner,MADV_DONTDUMP)!=0 || mlock(g,(size_t)owner)!=0) {
        (void)munmap(g,(size_t)owner);return ria_fail(e,RIA_RESOURCE_LIMIT,"protect/lock private graph owner failed");
    }
    g->owner_locked=true;
    g->store=store;g->options=*options;g->remote=remote;
    for (unsigned layer=0;layer<40;++layer) for (unsigned expert=0;expert<384;++expert) g->local_index[layer][expert]=-1;
    g->embedding=required(store,"embed.weight",e);g->norm=required(store,"norm.weight",e);
    const ria_tensor *head=required(store,"head.weight",e);
    if (!g->embedding || !g->norm || !head || !ria_tensor_matrix(store,head,&g->head,e)) goto bad;
    if (g->embedding->rank!=2 || g->embedding->shape[0]!=129280 || g->embedding->shape[1]!=5120 ||
        strcmp(g->embedding->dtype,"BF16") || g->norm->rank!=1 || g->norm->shape[0]!=5120 ||
        (strcmp(g->norm->dtype,"BF16") && strcmp(g->norm->dtype,"F32")) ||
        g->head.in_features!=5120 || g->head.out_features!=129280 ||
        (g->head.profile!=RIA_EXPERT_BF16 && g->head.profile!=RIA_EXPERT_F32)) {
        (void)ria_fail(e,RIA_INTEGRITY_ERROR,"global graph tensor precision/shape differs from source");
        goto bad;
    }
    for (unsigned i=0;i<40;++i) if (!bind_layer(g,i,e)) goto bad;
    ria_graph_options execution=*options;
    if (options->device_expert_budget>=options->device_budget) { (void)ria_fail(e,RIA_RESOURCE_LIMIT,"local device reservation leaves no graph execution budget");goto bad; }
    execution.device_budget-=options->device_expert_budget;
    if (!allocate_private(g,e) || !ria_graph_cuda_create(&execution,&g->cuda,e)) goto bad;
    if (options->max_image_patches) {
        uint64_t used=ria_graph_cuda_bytes(g->cuda);
        uint64_t pinned=ria_graph_cuda_pinned_bytes(g->cuda);
        if (used>=execution.device_budget || pinned>=options->pinned_budget ||
            !ria_vision_create_pooled(store,options->device,options->max_image_patches,execution.device_budget-used,
                                     options->pinned_budget-pinned,&g->vision,e)) goto bad;
        const char *names[3]={"image_start","image_end","image_newline"};
        for (unsigned i=0;i<3;++i) {
            const ria_tensor *t=g->image_delimiters[i]=required(store,names[i],e);
            if (!t || t->rank!=1 || t->shape[0]!=5120 || (strcmp(t->dtype,"BF16") && strcmp(t->dtype,"F32"))) {
                if (t) (void)ria_fail(e,RIA_INTEGRITY_ERROR,"image delimiter precision/shape differs from source");
                goto bad;
            }
        }
    }
    for (uint32_t i=0;i<options->local_expert_count;++i) {
        const ria_graph_local_expert *x=&options->local_experts[i];g->local_index[x->layer][x->expert_id]=(int32_t)i;
        if (x->tier==RIA_GRAPH_VRAM) {
            uint64_t used=ria_graph_cuda_bytes(g->cuda)+ria_vision_device_bytes(g->vision)+g->local_device_bytes;
            if (used>=options->device_budget || g->local_device_bytes>options->device_expert_budget ||
                !ria_graph_cuda_local_create(g->cuda,&x->expert,
                    options->device_budget-used<options->device_expert_budget-g->local_device_bytes ? options->device_budget-used : options->device_expert_budget-g->local_device_bytes,
                    &g->local_device[i],e)) goto bad;
            g->local_device_bytes+=ria_expert_cuda_resident_bytes(&g->local_device[i]);
        } else if (!ria_expert_validate(&x->expert,e)) goto bad;
    }
    if (!ria_graph_reset(g,1,1,e)) goto bad;
    *out=g;return true;
bad:{ria_error cleanup; (void)ria_graph_destroy(g,&cleanup);return false;}
}
bool ria_graph_destroy(ria_graph *g,ria_error *e) {
    if (!g) return true;
    bool ok=ria_vision_destroy(g->vision,e);
    for (uint32_t i=0;i<g->options.local_expert_count;++i) if (!ria_expert_cuda_resident_destroy(&g->local_device[i],ok ? e : NULL)) ok=false;
    if (!ria_graph_cuda_destroy(g->cuda,ok ? e : NULL)) ok=false;
    if (g->locked && munlock(g->host_allocation,(size_t)g->state_bytes)!=0) { if (ok) (void)ria_fail(e,RIA_INTERNAL_ERROR,"unlock graph backing failed");ok=false; }
    if (g->host_allocation && munmap(g->host_allocation,(size_t)g->state_bytes)!=0) { if (ok) (void)ria_fail(e,RIA_INTERNAL_ERROR,"release graph backing failed");ok=false; }
    size_t owner=(size_t)g->owner_bytes;
    if (g->owner_locked && munlock(g,owner)!=0) { if (ok) (void)ria_fail(e,RIA_INTERNAL_ERROR,"unlock graph owner failed");ok=false; }
    if (munmap(g,owner)!=0) { if (ok) (void)ria_fail(e,RIA_INTERNAL_ERROR,"release graph owner failed");ok=false; }
    return ok;
}
bool ria_graph_reset(ria_graph *g,uint64_t epoch,uint64_t generation,ria_error *e) {
    if (!g || !epoch || !generation) return ria_fail(e,RIA_INVALID_REQUEST,"invalid graph generation");
    if (!ria_graph_cuda_reset(g->cuda,e)) { g->poisoned=true;return false; }
    memset(g->host_allocation,0,(size_t)g->state_bytes);
    for (unsigned i=0;i<4;++i) g->history[i]=-1;
    g->position=0;g->epoch=epoch;g->generation=generation;g->selected_count=0;g->phase=RIA_GRAPH_PREFILL;g->poisoned=false;return true;
}
uint64_t ria_graph_position(const ria_graph *g) { return g ? g->position : 0; }
bool ria_graph_retag(ria_graph *g,uint64_t epoch,uint64_t generation,ria_error *e) {
    if (!g || g->poisoned || !epoch || !generation || epoch!=g->epoch || generation<=g->generation)
        return ria_fail(e,RIA_INVALID_REQUEST,"graph retag requires usable identical-prefix state and increasing same-epoch generation");
    if (!ria_graph_cuda_drain(g->cuda,e)) { g->poisoned=true;return false; }
    g->generation=generation;return true;
}
uint64_t ria_graph_host_bytes(const ria_graph *g) { return g ? g->host_bytes : 0; }
uint64_t ria_graph_device_bytes(const ria_graph *g) { return g ? ria_graph_cuda_bytes(g->cuda)+ria_vision_device_bytes(g->vision)+g->local_device_bytes : 0; }
bool ria_graph_set_phase(ria_graph *g,ria_graph_phase phase,ria_error *e) {
    if (!g || g->poisoned || phase<RIA_GRAPH_PREFILL || phase>RIA_GRAPH_CONTINUATION)
        return ria_fail(e,RIA_INVALID_REQUEST,"invalid/poisoned graph phase transition");
    g->phase=phase;return true;
}
uint64_t ria_graph_pinned_bytes(const ria_graph *g) { return g ? ria_graph_cuda_pinned_bytes(g->cuda)+ria_vision_pinned_bytes(g->vision) : 0; }
static float *buffer(ria_graph *g,unsigned id) { return ria_graph_cuda_buffer(g->cuda,id); }
static bool attention(ria_graph *g,unsigned layer,ria_error *e) {
    graph_layer *x=&g->layers[layer];ria_graph_cuda *c=g->cuda;uint64_t pos=g->position;
    float *input=buffer(g,RIA_G_INPUT),*qr=buffer(g,RIA_G_QR),*q=buffer(g,RIA_G_Q),*kv=buffer(g,RIA_G_KV),*latent=buffer(g,RIA_G_LATENT);
    if (!ria_graph_cuda_project(c,&x->wqa,input,qr,true,e) || !ria_graph_cuda_norm(c,qr,1,1280,x->q_norm,false,e) ||
        !ria_graph_cuda_project(c,&x->wqb,qr,q,true,e) || !ria_graph_cuda_rope(c,q,64,512,pos,x->ratio,false,e) ||
        !ria_graph_cuda_project(c,&x->wkv,input,kv,true,e) || !ria_graph_cuda_norm(c,kv,1,512,x->kv_norm,false,e) ||
        !ria_graph_cuda_rope(c,kv,1,512,pos,x->ratio,false,e) || !ria_graph_cuda_pack(c,kv,512,1,g->windows[layer]+(pos%128)*528,e)) return false;
    uint64_t compressed_count=x->ratio ? (pos+1)/x->ratio : 0;
    unsigned owner=source_slot(x->kv_source);bool produced=false;
    /* Full owns compressed KV and index keys. Reindex reads that immutable
     * published population and replaces query-dependent selected positions.
     * Reuse consumes the preceding index owner's selection unchanged. The
     * full-reference CED schedule also executes every decoder window/history,
     * including exact incomplete encoder pools across continued prefill. */
    if (x->ratio && x->kv_source==layer) {
        if (x->ratio==1) {
            if (!ria_graph_cuda_project(c,&x->compressor_kv,input,latent,true,e)) return false;
            produced=true;
        } else {
            float *pk=buffer(g,RIA_G_POOL_KV),*pg=buffer(g,RIA_G_POOL_GATE);unsigned slot=(unsigned)(pos%x->ratio);
            if (!ria_graph_cuda_upload(c,pk,g->pool_kv[owner],1024,e) || !ria_graph_cuda_upload(c,pg,g->pool_gate[owner],1024,e) ||
                !ria_graph_cuda_project(c,&x->compressor_kv,input,pk+slot*512,false,e) ||
                !ria_graph_cuda_project(c,&x->compressor_gate,input,pg+slot*512,false,e) ||
                !ria_graph_cuda_download(c,pk,g->pool_kv[owner],1024,e) || !ria_graph_cuda_download(c,pg,g->pool_gate[owner],1024,e)) return false;
            produced=(pos+1)%x->ratio==0;
            if (produced && !ria_graph_cuda_pool(c,pk,pg,x->ratio,latent,e)) return false;
        }
        if (produced) {
            float *key=buffer(g,RIA_G_KV);
            if (!ria_graph_cuda_norm(c,latent,1,512,x->compressor_norm,false,e) ||
                !ria_graph_cuda_project(c,&x->index_key,latent,key,true,e) || !ria_graph_cuda_norm(c,key,1,128,x->index_norm,false,e) ||
                !ria_graph_cuda_rope(c,key,1,128,pos+1-x->ratio,x->ratio,false,e) ||
                !ria_graph_cuda_pack(c,key,128,3,g->index_keys[owner]+(compressed_count-1)*68,e)) return false;
        }
    }
    if (x->ratio && x->index_source==layer) {
        g->selected_count=0;
        if (compressed_count) {
            float *iq=buffer(g,RIA_G_INDEX_Q),*weights=buffer(g,RIA_G_INDEX_WEIGHTS),*keys=buffer(g,RIA_G_INDEX_TILE);
            if (!ria_graph_cuda_project(c,&x->index_q,qr,iq,true,e) || !ria_graph_cuda_rope(c,iq,32,128,pos,x->ratio,false,e) ||
                !ria_graph_cuda_quantize_inplace(c,iq,4096,3,e) || !ria_graph_cuda_project(c,&x->index_weights,input,weights,true,e)) return false;
            for (uint64_t first=0;first<compressed_count;first+=g->options.state_tile_rows) {
                uint64_t count=compressed_count-first<g->options.state_tile_rows ? compressed_count-first : g->options.state_tile_rows;
                if (!ria_graph_cuda_unpack(c,g->index_keys[owner]+first*68,count,128,3,keys,e) ||
                    !ria_graph_cuda_index_scores(c,iq,weights,keys,first,count,e)) return false;
            }
            if (!ria_graph_cuda_select(c,compressed_count,layer==20,layer>20,g->selected,&g->selected_count,e)) return false;
        }
    }
    if (produced && (!ria_graph_cuda_rope(c,latent,1,512,pos+1-x->ratio,x->ratio,false,e) ||
        !ria_graph_cuda_pack(c,latent,512,2,g->compressed[owner]+(compressed_count-1)*288,e))) return false;
    uint64_t window_count=pos<128 ? pos+1 : 128,oldest=pos+1-window_count;
    float *attention_kv=buffer(g,RIA_G_ATTN_KV);
    for (uint64_t row=0;row<window_count;++row)
        memcpy(g->packed_window+row*528,g->windows[layer]+((oldest+row)%128)*528,528);
    if (!ria_graph_cuda_unpack(c,g->packed_window,window_count,512,1,attention_kv,e)) return false;
    uint32_t selected=x->ratio ? g->selected_count : 0;
    for (uint32_t row=0;row<selected;++row) {
        if (g->selected[row]>=compressed_count) return ria_fail(e,RIA_INTERNAL_ERROR,"index selected an unpublished compressed position");
        memcpy(g->packed_selected+(uint64_t)row*288,g->compressed[owner]+(uint64_t)g->selected[row]*288,288);
    }
    if (selected && !ria_graph_cuda_unpack(c,g->packed_selected,selected,512,2,attention_kv+window_count*512,e)) return false;
    float *o=buffer(g,RIA_G_ATTN_OUT),*low=buffer(g,RIA_G_OLOW);
    if (!ria_graph_cuda_attention(c,q,attention_kv,window_count+selected,x->sink,o,e) || !ria_graph_cuda_rope(c,o,64,512,pos,x->ratio,true,e)) return false;
    for (unsigned group=0;group<8;++group) {
        ria_expert_matrix matrix=x->woa;matrix.out_features=1024;matrix.values+=group*1024*matrix.value_row_stride;
        matrix.values_bytes-=group*1024*matrix.value_row_stride;
        if (!ria_graph_cuda_project(c,&matrix,o+group*4096,low+group*1024,true,e)) return false;
    }
    return ria_graph_cuda_project(c,&x->wob,low,buffer(g,RIA_G_OUTPUT),true,e);
}
static bool ffn(ria_graph *g,unsigned layer,bool image,ria_error *e) {
    graph_layer *x=&g->layers[layer];ria_graph_cuda *c=g->cuda;float *input=buffer(g,RIA_G_INPUT);
    uint16_t ids[6];float coefficients[6];
    if (!ria_graph_cuda_project(c,&x->router,input,buffer(g,RIA_G_ROUTER),false,e) ||
        !ria_graph_cuda_route(c,buffer(g,RIA_G_ROUTER),image ? x->bias_vl : x->bias,ids,coefficients,e)) return false;
    uint16_t remote_ids[6],remote_slots[6];float remote_coefficients[6];unsigned remote_count=0;
    for (unsigned slot=0;slot<6;++slot) {
        int32_t index=g->local_index[layer][ids[slot]];
        const ria_graph_local_expert *local=index>=0 ? &g->options.local_experts[index] : NULL;
        if (local && (local->phase_mask&(1u<<(unsigned)g->phase))) {
            const ria_expert_cuda_resident *resident=local->tier==RIA_GRAPH_VRAM ? &g->local_device[index] : NULL;
            if (!ria_graph_cuda_local_evaluate(c,&local->expert,resident,input,coefficients[slot],g->contributions+slot*5120,e)) return false;
        } else { remote_ids[remote_count]=ids[slot];remote_coefficients[remote_count]=coefficients[slot];remote_slots[remote_count++]=(uint16_t)slot; }
    }
    if (remote_count) {
        if (!ria_graph_cuda_download(c,input,g->input,5120,e) ||
            !g->remote.experts(g->remote.context,layer,g->input,remote_ids,remote_coefficients,remote_slots,remote_count,g->remote_contributions,e)) return false;
        for (unsigned i=0;i<remote_count;++i) memcpy(g->contributions+(uint64_t)remote_slots[i]*5120,g->remote_contributions+(uint64_t)i*5120,5120*sizeof(float));
    }
    for (unsigned slot=0;slot<6;++slot) for (unsigned k=0;k<5120;++k)
        { float value=g->contributions[slot*5120+k];uint32_t bits;memcpy(&bits,&value,sizeof(bits));
          if (!isfinite(value) || (bits&65535u)) return ria_fail(e,RIA_EXECUTOR_ERROR,"remote expert contribution is not finite BF16-rounded FP32"); }
    /* Shared branch exactly once. Its source mixed precision is independent
     * of routed-profile selection. Use the same CUDA expert graph boundary. */
    ria_expert shared={x->shared_gate,x->shared_up,x->shared_down,10};
    if (!ria_graph_cuda_shared(c,&shared,input,buffer(g,RIA_G_SHARED),e)) return false;
    return ria_graph_cuda_merge(c,ids,g->contributions,buffer(g,RIA_G_SHARED),buffer(g,RIA_G_OUTPUT),e);
}
bool ria_graph_step(ria_graph *g,uint32_t token,bool image,const float *embedding,float logits[RIA_GRAPH_VOCAB],ria_error *e) {
    if (!g || !logits || token>=129280 || g->poisoned || g->position>=g->options.max_tokens || (embedding && !image))
        return ria_fail(e,RIA_INVALID_REQUEST,"invalid/out-of-capacity/poisoned graph step");
    ria_graph_cuda *c=g->cuda;float *h=buffer(g,RIA_G_H),*residual=buffer(g,RIA_G_RESIDUAL),*input=buffer(g,RIA_G_INPUT);
    for (unsigned i=3;i>0;--i) g->history[i]=g->history[i-1];
    g->history[0]=image ? INT64_C(-1) : (int64_t)g->options.compressed_token_map[token];
    if (!(embedding ? ria_graph_cuda_upload(c,input,embedding,5120,e) :
                      ria_graph_cuda_tensor(c,g->embedding,(uint64_t)token*5120,input,5120,e)) || !ria_graph_cuda_expand(c,input,h,e)) goto bad;
    for (unsigned layer=0;layer<40;++layer) {
        graph_layer *x=&g->layers[layer];
        if ((layer==1 || layer==14) && !image) {
            uint64_t ids[24];
            if (!ria_graph_hash(&g->options,layer==14,g->history,ids,e) ||
                !ria_graph_rows_fetch(g->remote,layer,ids,24,g->engram_cache,g->engram_capacity,g->packed_rows,e) ||
                !ria_graph_cuda_engram(c,g->packed_rows,buffer(g,RIA_G_ENGRAM_INPUT),e) ||
                !ria_graph_cuda_project(c,&x->engram_kv,buffer(g,RIA_G_ENGRAM_INPUT),buffer(g,RIA_G_ENGRAM_KV),true,e) ||
                !ria_graph_cuda_engram_fuse(c,h,buffer(g,RIA_G_ENGRAM_KV),x->engram_q,x->engram_k,true,e)) goto bad;
        }
        if (!ria_graph_cuda_copy(c,residual,h,20480,e) ||
            !ria_graph_cuda_project(c,&x->hc_attn,h,buffer(g,RIA_G_MIX),false,e) ||
            !ria_graph_cuda_mhc(c,h,buffer(g,RIA_G_MIX),x->attn_scale,x->attn_base,
                                buffer(g,RIA_G_ATT_PRE),buffer(g,RIA_G_ATT_POST),buffer(g,RIA_G_ATT_COMB),e) ||
            !ria_graph_cuda_collapse(c,h,buffer(g,RIA_G_PRE),input,e) || !ria_graph_cuda_norm(c,input,1,5120,x->attn_norm,false,e) ||
            !attention(g,layer,e) || !ria_graph_cuda_post(c,buffer(g,RIA_G_OUTPUT),residual,buffer(g,RIA_G_ATT_POST),buffer(g,RIA_G_ATT_COMB),h,e) ||
            !ria_graph_cuda_copy(c,residual,h,20480,e) || !ria_graph_cuda_project(c,&x->hc_ffn,h,buffer(g,RIA_G_MIX),false,e) ||
            !ria_graph_cuda_mhc(c,h,buffer(g,RIA_G_MIX),x->ffn_scale,x->ffn_base,
                                buffer(g,RIA_G_FFN_PRE),buffer(g,RIA_G_FFN_POST),buffer(g,RIA_G_FFN_COMB),e) ||
            !ria_graph_cuda_collapse(c,h,buffer(g,RIA_G_ATT_PRE),input,e) || !ria_graph_cuda_norm(c,input,1,5120,x->ffn_norm,false,e) ||
            !ffn(g,layer,image,e) || !ria_graph_cuda_post(c,buffer(g,RIA_G_OUTPUT),residual,buffer(g,RIA_G_FFN_POST),buffer(g,RIA_G_FFN_COMB),h,e) ||
            !ria_graph_cuda_copy(c,buffer(g,RIA_G_PRE),buffer(g,RIA_G_FFN_PRE),4,e)) goto bad;
    }
    if (!ria_graph_cuda_collapse(c,h,buffer(g,RIA_G_PRE),input,e) || !ria_graph_cuda_norm(c,input,1,5120,g->norm,false,e) ||
        !ria_graph_cuda_project(c,&g->head,input,buffer(g,RIA_G_LOGITS),false,e) ||
        !ria_graph_cuda_download(c,buffer(g,RIA_G_LOGITS),logits,129280,e)) goto bad;
    for (unsigned i=0;i<RIA_GRAPH_VOCAB;++i) if (!isfinite(logits[i])) {
        (void)ria_fail(e,RIA_EXECUTOR_ERROR,"source graph produced nonfinite logits");
        goto bad;
    }
    ++g->position;return true;
bad:g->poisoned=true;(void)ria_graph_cuda_drain(c,NULL);return false;
}
bool ria_graph_prefill(ria_graph *g,const uint32_t *tokens,uint64_t count,float *last,float *all,ria_error *e) {
    if (!g || !tokens || !last || !count || count>g->options.max_tokens-g->position ||
        (all && count>SIZE_MAX/(129280*sizeof(float)))) return ria_fail(e,RIA_INVALID_REQUEST,"invalid bounded graph prefill");
    for (uint64_t row=0;row<count;++row) if (!ria_graph_step(g,tokens[row],false,NULL,all ? all+row*129280 : last,e)) return false;
    if (all) memcpy(last,all+(count-1)*129280,129280*sizeof(float));
    return true;
}
bool ria_graph_encode_image(ria_graph *g,const float *patches,uint32_t h,uint32_t w,float *output,uint64_t rows,ria_error *e) {
    if (!g || !g->vision || g->poisoned) return ria_fail(e,RIA_INVALID_REQUEST,"image graph not admitted/usable");
    if (!ria_vision_encode(g->vision,patches,h,w,output,rows,e)) { g->poisoned=true;return false; }return true;
}
bool ria_graph_image_delimiter(ria_graph *g,uint32_t type,float output[RIA_GRAPH_DIM],ria_error *e) {
    unsigned slot=type==0 ? 0 : type==3 ? 1 : type==2 ? 2 : 3;
    if (!g || g->poisoned || slot>=3 || !output || !g->image_delimiters[slot]) return ria_fail(e,RIA_INVALID_REQUEST,"invalid/unadmitted image delimiter");
    if (!ria_graph_cuda_tensor(g->cuda,g->image_delimiters[slot],0,buffer(g,RIA_G_INPUT),5120,e) ||
        !ria_graph_cuda_download(g->cuda,buffer(g,RIA_G_INPUT),output,5120,e)) {
        g->poisoned=true;(void)ria_graph_cuda_drain(g->cuda,NULL);return false;
    }
    return true;
}
