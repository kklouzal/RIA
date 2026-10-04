#define _GNU_SOURCE
/* Execute the production graph scheduler with independent host arithmetic and
 * transport association oracles. CUDA operations are recorded/replaced here;
 * no driver, learned checkpoint, GPU kernel, or performance claim is involved. */
#include <stdlib.h>
#include <string.h>
#include "../../ria/graph.c"

#define CHECK(x) do { if (!(x)) { fprintf(stderr,"prefill fixture %s:%d: %s\n",__FILE__,__LINE__,#x);abort(); } } while (0)

#define FIXTURE_TOKENS 192u
#define ACTIVE 8u
enum { OP_HC=1,OP_WQA,OP_WQB,OP_WKV,OP_WOA,OP_WOB,OP_ROUTER,
       OP_COMPRESS_KV,OP_COMPRESS_GATE,OP_INDEX_Q,OP_INDEX_WEIGHT,OP_INDEX_KEY,OP_ENGRAM,OP_HEAD };
struct ria_graph_cuda {
    ria_graph *graph;
    float *buffers[RIA_G_BUFFER_COUNT];
    float scores[FIXTURE_TOKENS];
    uint32_t candidates[2048],candidate_count,current_layer;
    uint32_t begins,drains,grouped_calls,multiple_rows,local_calls,stage;
    uint64_t observed_rows;
    float observed[FIXTURE_TOKENS][ACTIVE];
    bool expected_images[FIXTURE_TOKENS];
    bool expert_population[RIA_GRAPH_EXPERTS];
    bool fail_group,nonfinite_group,unrounded_group;
};
static uint32_t token_map[RIA_GRAPH_VOCAB];
static uint8_t matrix_storage[40][8192];
static ria_tensor biases[40][2];
static const uint64_t buffer_sizes[RIA_G_BUFFER_COUNT]={20480,20480,5120,5120,4,4,4,16,4,4,16,24,1280,32768,512,512,
    4096,32,640*512,32768,8192,384,6*5120,5120,6144,5120,129280,64*128,1024,1024};

/* Choose adjacent representable BF16 values by distance instead of using the
 * production bit-add rounding implementation. Tie selection is nearest even. */
static float rounded(float value) {
    uint32_t raw;memcpy(&raw,&value,4);
    uint32_t lower=raw&UINT32_C(0xffff0000),upper=lower+UINT32_C(0x10000);
    float lo,hi;memcpy(&lo,&lower,4);memcpy(&hi,&upper,4);
    double dl=fabs((double)value-lo),dh=fabs((double)hi-value);
    return dl<dh || (dl==dh && !(lower&UINT32_C(0x10000))) ? lo : hi;
}
static void clear_active(float *out,uint64_t count) { memset(out,0,(size_t)count*sizeof(float)); }
static float contribution(float input,uint16_t expert,float coefficient,unsigned column) {
    /* Per-row quantizer scope and coefficient placement are deliberately
     * observable: shared group maxima or slot/coefficient mixups change bits. */
    float scaled=rounded(input+(float)column/128.0f);
    return rounded((scaled/4.0f+(float)(expert+1)/512.0f)*coefficient);
}
float *ria_graph_cuda_buffer(ria_graph_cuda *c,unsigned id) { CHECK(id<RIA_G_BUFFER_COUNT);return c->buffers[id]; }
bool ria_graph_cuda_drain(ria_graph_cuda *c,ria_error *e) { (void)e;++c->drains;return true; }
bool ria_graph_cuda_begin_step(ria_graph_cuda *c,ria_error *e) {
    (void)e;++c->begins;c->stage=1;const float identity[4]={1,0,0,0};memcpy(c->buffers[RIA_G_PRE],identity,sizeof(identity));return true;
}
bool ria_graph_cuda_reset(ria_graph_cuda *c,ria_error *e) {
    for (unsigned i=0;i<RIA_G_BUFFER_COUNT;++i) memset(c->buffers[i],0,(size_t)buffer_sizes[i]*sizeof(float));
    c->candidate_count=0;c->fail_group=false;c->nonfinite_group=false;c->unrounded_group=false;
    return ria_graph_cuda_begin_step(c,e);
}
bool ria_graph_cuda_copy(ria_graph_cuda *c,float *to,const float *from,uint64_t count,ria_error *e) {
    (void)c;(void)e;memmove(to,from,(size_t)count*sizeof(float));return true;
}
bool ria_graph_cuda_upload(ria_graph_cuda *c,float *to,const float *from,uint64_t count,ria_error *e) {
    return ria_graph_cuda_copy(c,to,from,count,e);
}
bool ria_graph_cuda_download(ria_graph_cuda *c,const float *from,float *to,uint64_t count,ria_error *e) {
    if (from==c->buffers[RIA_G_LOGITS]) {
        uint64_t position=c->graph->position;CHECK(position<FIXTURE_TOKENS);
        memcpy(c->observed[position],from,ACTIVE*sizeof(float));++c->observed_rows;
    }
    return ria_graph_cuda_copy(c,to,from,count,e);
}
bool ria_graph_cuda_tensor(ria_graph_cuda *c,const ria_tensor *tensor,uint64_t first,float *out,uint64_t count,ria_error *e) {
    (void)c;(void)tensor;(void)e;clear_active(out,count);
    for (unsigned column=0;column<ACTIVE && column<count;++column)
        out[column]=rounded(0.5f+(float)(first/5120%19)/32.0f+(float)column/128.0f);
    return true;
}
bool ria_graph_cuda_expand(ria_graph_cuda *c,const float *input,float *h,ria_error *e) {
    (void)c;(void)e;for (unsigned stream=0;stream<4;++stream) memcpy(h+stream*5120,input,5120*sizeof(float));return true;
}
bool ria_graph_cuda_project(ria_graph_cuda *c,const ria_expert_matrix *matrix,const float *input,float *out,bool cast,ria_error *e) {
    (void)e;unsigned tag=(unsigned)matrix->activation_global_scale,op=tag%100;
    if (op==OP_HEAD) c->stage=5;
    c->current_layer=tag/100;float seed=input[0];clear_active(out,matrix->out_features);
    for (unsigned column=0;column<ACTIVE && column<matrix->out_features;++column) {
        float value;
        if (op==OP_COMPRESS_GATE) value=(float)(column%3)/16.0f;
        else if (op==OP_ROUTER) value=seed;
        else if (op==OP_HEAD) value=seed+(float)column/64.0f;
        else value=seed/2.0f+(float)op/128.0f+(float)column/1024.0f;
        out[column]=cast ? rounded(value) : value;
    }
    return true;
}
bool ria_graph_cuda_norm(ria_graph_cuda *c,float *values,uint64_t rows,uint64_t width,const ria_tensor *weight,bool layer_norm,ria_error *e) {
    (void)c;(void)weight;(void)layer_norm;(void)e;
    for (uint64_t row=0;row<rows;++row) for (unsigned column=0;column<ACTIVE && column<width;++column)
        values[row*width+column]=rounded(values[row*width+column]);
    return true;
}
bool ria_graph_cuda_mhc(ria_graph_cuda *c,const float *h,float *mix,const ria_tensor *scale,const ria_tensor *base,
                        float *pre,float *post,float *comb,ria_error *e) {
    (void)c;(void)h;(void)mix;(void)scale;(void)base;(void)e;
    const float weights[4]={0.25f,0.375f,0.125f,0.25f};memcpy(pre,weights,sizeof(weights));
    for (unsigned stream=0;stream<4;++stream) { post[stream]=(float)(stream+1)/128.0f;
        for (unsigned other=0;other<4;++other) comb[stream*4+other]=stream==other ? 0.96875f : 0; }
    return true;
}
bool ria_graph_cuda_collapse(ria_graph_cuda *c,const float *h,const float *pre,float *out,ria_error *e) {
    (void)c;(void)e;clear_active(out,5120);
    for (unsigned column=0;column<ACTIVE;++column) {
        float value=0;for (unsigned stream=0;stream<4;++stream) value+=h[stream*5120+column]*pre[stream];
        out[column]=rounded(value);
    }
    return true;
}
bool ria_graph_cuda_post(ria_graph_cuda *c,const float *value,const float *residual,const float *post,const float *comb,float *h,ria_error *e) {
    (void)c;(void)e;clear_active(h,20480);
    for (unsigned stream=0;stream<4;++stream) for (unsigned column=0;column<ACTIVE;++column) {
        float sum=0;for (unsigned other=0;other<4;++other) sum+=residual[other*5120+column]*comb[other*4+stream];
        h[stream*5120+column]=rounded(value[column]*post[stream]+sum);
    }
    return true;
}
bool ria_graph_cuda_rope(ria_graph_cuda *c,float *values,uint64_t rows,uint64_t width,uint64_t position,uint32_t ratio,bool inverse,ria_error *e) {
    (void)c;(void)ratio;(void)e;
    for (uint64_t row=0;row<rows;++row) values[row*width]=rounded(values[row*width]+(inverse ? -1 : 1)*(float)(position%17)/1024.0f);
    return true;
}
bool ria_graph_cuda_pack(ria_graph_cuda *c,const float *values,uint64_t width,uint32_t representation,uint8_t *host,ria_error *e) {
    (void)e;uint64_t stride=representation==1 ? width+width/32 : width/2+width/(representation==2 ? 16 : 32);
    memset(host,0,(size_t)stride);ria_write_f32(host,rounded(values[0]));ria_write_u32(host+4,(uint32_t)c->graph->position+1);return true;
}
bool ria_graph_cuda_unpack(ria_graph_cuda *c,const uint8_t *host,uint64_t rows,uint64_t width,uint32_t representation,float *out,ria_error *e) {
    (void)e;uint64_t stride=representation==1 ? width+width/32 : width/2+width/(representation==2 ? 16 : 32);
    clear_active(out,rows*width);
    if (representation==1) {
        uint64_t position=c->graph->position,extent=position ? 128 : 1,valid=position<128 ? position+1 : 128;
        CHECK(width==512 && rows==extent);
        /* Publisher sparse slot list: holes precede the oldest valid entry. */
        for (uint64_t row=0;row<extent;++row) {
            uint32_t expected=row<extent-valid ? 0 : (uint32_t)(position-valid+2+row-(extent-valid));
            CHECK(ria_read_u32(host+row*stride+4)==expected);
        }
    }
    for (uint64_t row=0;row<rows;++row) out[row*width]=ria_read_f32(host+row*stride);
    return true;
}
bool ria_graph_cuda_pool(ria_graph_cuda *c,float *kv,float *gates,uint32_t ratio,float *latent,ria_error *e) {
    (void)c;(void)e;CHECK(ratio==2);clear_active(latent,512);
    for (unsigned column=0;column<ACTIVE;++column) {
        double a=exp((double)gates[column]),b=exp((double)gates[512+column]);
        latent[column]=rounded((float)((a*kv[column]+b*kv[512+column])/(a+b)));
    }
    return true;
}
bool ria_graph_cuda_quantize_inplace(ria_graph_cuda *c,float *values,uint64_t count,uint32_t representation,ria_error *e) {
    (void)c;(void)representation;(void)e;for (uint64_t i=0;i<count;++i) values[i]=rounded(values[i]);return true;
}
bool ria_graph_cuda_index_scores(ria_graph_cuda *c,const float *q,const float *weights,const float *keys,uint64_t first,uint64_t count,ria_error *e) {
    (void)e;for (uint64_t row=0;row<count;++row) c->scores[first+row]=keys[row*128]+q[0]*weights[0]*(float)((first+row)%3+1);return true;
}
bool ria_graph_cuda_select(ria_graph_cuda *c,uint64_t count,bool source,bool uses,uint32_t *positions,uint32_t *chosen,ria_error *e) {
    (void)e;CHECK(count && count<=FIXTURE_TOKENS);
    if (source) { c->candidate_count=(uint32_t)((count+7)/8);for (uint32_t i=0;i<c->candidate_count;++i) c->candidates[i]=i; }
    if (uses) {
        /* At this bounded length the source selects every candidate block;
         * snapshot length still varies by causal row and must be restored. */
        CHECK(c->candidate_count==(count+7)/8);
        for (uint32_t i=0;i<c->candidate_count;++i) CHECK(c->candidates[i]==i);
    }
    uint32_t winner=0;for (uint32_t i=1;i<count;++i) if (c->scores[i]>c->scores[winner]) winner=i;
    positions[0]=winner;*chosen=1;return true;
}
bool ria_graph_cuda_attention(ria_graph_cuda *c,const float *q,const float *kv,uint64_t count,uint64_t masked_begin,uint64_t masked_end,
                              const ria_tensor *sink,float *out,ria_error *e) {
    (void)sink;(void)e;uint64_t position=c->graph->position,extent=position ? 128 : 1,valid=position<128 ? position+1 : 128;
    c->stage=2;
    CHECK(masked_begin==0 && masked_end==extent-valid && count>=extent && count<=extent+1);
    /* Sparse probability tile boundaries remain part of this oracle. */
    float aggregate=0;for (uint64_t row=masked_end;row<count;++row)
        aggregate=rounded(aggregate+kv[row*512]/(float)(1+row/64));
    clear_active(out,32768);for (unsigned head=0;head<64;++head) out[head*512]=rounded(q[head*512]/4+aggregate/(float)count);return true;
}
bool ria_graph_cuda_route(ria_graph_cuda *c,const float *scores,const ria_tensor *bias,uint16_t ids[6],float coefficients[6],ria_error *e) {
    (void)scores;(void)e;unsigned shift=(unsigned)(c->graph->position*3%384),image=bias==&biases[c->current_layer][1];
    const unsigned permutation[6]={5,1,3,0,4,2};
    for (unsigned slot=0;slot<6;++slot) { ids[slot]=(uint16_t)((permutation[slot]+shift+image)%384);
        coefficients[slot]=slot==1 ? 0 : (float)(slot+1)/16.0f; }
    return true;
}
bool ria_graph_cuda_shared(ria_graph_cuda *c,const ria_expert *expert,const float *input,float *out,ria_error *e) {
    (void)c;(void)expert;(void)e;clear_active(out,5120);for (unsigned column=0;column<ACTIVE;++column) out[column]=rounded(input[column]/8.0f);return true;
}
typedef struct { uint16_t expert,slot; } ordered_slot;
static int slot_order(const void *a,const void *b) {
    const ordered_slot *x=a,*y=b;return (x->expert>y->expert)-(x->expert<y->expert);
}
bool ria_graph_cuda_merge(ria_graph_cuda *c,const uint16_t ids[6],const float *results,const float *shared,float *out,ria_error *e) {
    (void)e;c->stage=4;clear_active(out,5120);
    /* Independent publisher MoE oracle: original global expert ID ascending,
     * followed by exactly one shared contribution, then the BF16 cast. */
    ordered_slot slots[6];for (uint16_t i=0;i<6;++i) slots[i]=(ordered_slot){ids[i],i};
    qsort(slots,6,sizeof(*slots),slot_order);
    for (unsigned column=0;column<ACTIVE;++column) {
        float sum=0;for (unsigned i=0;i<6;++i) sum+=results[slots[i].slot*5120+column];
        out[column]=rounded(sum+shared[column]);
    }
    return true;
}
bool ria_graph_cuda_local_evaluate(ria_graph_cuda *c,const ria_expert *expert,const ria_expert_cuda_resident *resident,
                                   const float *input,float coefficient,float *out,ria_error *e) {
    (void)resident;(void)e;++c->local_calls;uint16_t id=(uint16_t)expert->gate.weight_global_scale;
    clear_active(out,5120);for (unsigned column=0;column<ACTIVE;++column) out[column]=contribution(input[column],id,coefficient,column);return true;
}
bool ria_graph_cuda_engram(ria_graph_cuda *c,const uint8_t *rows,float *out,ria_error *e) {
    (void)c;(void)e;clear_active(out,6144);for (unsigned i=0;i<24;++i) out[i*256]=(float)rows[i*264]/65536.0f;return true;
}
bool ria_graph_cuda_engram_fuse(ria_graph_cuda *c,float *h,const float *kv,const ria_tensor *q,const ria_tensor *k,bool enabled,ria_error *e) {
    (void)c;(void)q;(void)k;(void)e;CHECK(enabled);
    for (unsigned stream=0;stream<4;++stream) h[stream*5120]=rounded(h[stream*5120]+kv[0]/64);
    return true;
}
static bool rows(void *context,uint32_t layer,const uint64_t *ids,uint32_t count,uint8_t *out,ria_error *e) {
    (void)context;(void)e;for (uint32_t row=0;row<count;++row) memset(out+row*264,(int)((ids[row]+layer)%251),264);return true;
}
static bool experts(void *context,uint32_t layer,const float *input,const uint16_t *ids,const float *coefficients,
                    const uint16_t *slots,uint32_t count,float *out,ria_error *e) {
    (void)context;(void)layer;(void)e;clear_active(out,(uint64_t)count*5120);
    for (uint32_t row=0;row<count;++row) { CHECK(slots[row]<6);
        for (unsigned column=0;column<ACTIVE;++column) out[row*5120+column]=contribution(input[column],ids[row],coefficients[row],column); }
    return true;
}
static bool expert_batch(void *context,uint32_t layer,uint16_t expert,uint32_t count,const uint64_t *row_ids,const float *input,
                         const float *coefficients,const uint16_t *slots,float *out,ria_error *e) {
    ria_graph_cuda *c=context;(void)layer;CHECK(count && count<=64 && expert<384);++c->grouped_calls;
    c->stage=3;
    c->expert_population[expert]=true;
    if (count>1) ++c->multiple_rows;
    if (c->fail_group && c->grouped_calls==3) return ria_fail(e,RIA_EXECUTOR_ERROR,"injected late expert subgroup failure");
    clear_active(out,(uint64_t)count*5120);
    for (uint32_t row=0;row<count;++row) {
        CHECK(row_ids[row]<FIXTURE_TOKENS && slots[row]<6);
        if (row) CHECK(row_ids[row]>row_ids[row-1]);
        unsigned expected=(unsigned)((const unsigned[6]){5,1,3,0,4,2}[slots[row]]+row_ids[row]*3+c->expected_images[row_ids[row]])%384;
        CHECK(expert==expected);
        CHECK(coefficients[row]==(slots[row]==1 ? 0 : (float)(slots[row]+1)/16.0f));
        for (unsigned column=0;column<ACTIVE;++column)
            out[row*5120+column]=contribution(input[row*5120+column],expert,coefficients[row],column);
    }
    if (c->nonfinite_group) out[0]=NAN;
    if (c->unrounded_group) out[0]=0.10001f;
    return true;
}

bool ria_graph_cuda_candidates_get(ria_graph_cuda *c,uint32_t *blocks,uint32_t *count,ria_error *e) {
    (void)e;memcpy(blocks,c->candidates,c->candidate_count*sizeof(*blocks));*count=c->candidate_count;return true;
}
bool ria_graph_cuda_candidates_set(ria_graph_cuda *c,const uint32_t *blocks,uint32_t count,ria_error *e) {
    (void)e;CHECK(count<=2048);memcpy(c->candidates,blocks,count*sizeof(*blocks));c->candidate_count=count;return true;
}
bool ria_graph_cuda_local_batch(ria_graph_cuda *c,const ria_expert *expert,const ria_expert_cuda_resident *resident,
                                const float *input,uint32_t count,const float *coefficients,float *out,ria_error *e) {
    CHECK(count && count<=64);
    for (uint32_t row=0;row<count;++row)
        if (!ria_graph_cuda_local_evaluate(c,expert,resident,input+row*5120,coefficients[row],out+row*5120,e)) return false;
    return true;
}
static ria_expert_matrix matrix(unsigned layer,unsigned op,uint64_t count,uint64_t width) {
    ria_expert_matrix m={0};m.profile=RIA_EXPERT_BF16;m.out_features=count;m.in_features=width;
    m.values=matrix_storage[layer];m.values_bytes=sizeof(matrix_storage[layer]);
    /* The fixture records operator identity without a learned tensor bank. */
    m.activation_global_scale=(float)(layer*100+op);return m;
}
static ria_graph *fixture(uint32_t chunk,bool locals) {
    unsigned local_count=locals ? 2 : 0;ria_graph *g=calloc(1,sizeof(*g)+local_count*sizeof(ria_expert_cuda_resident));CHECK(g);
    ria_graph_cuda *c=calloc(1,sizeof(*c));CHECK(c);c->graph=g;g->cuda=c;
    for (unsigned i=0;i<RIA_G_BUFFER_COUNT;++i) { c->buffers[i]=calloc((size_t)buffer_sizes[i],sizeof(float));CHECK(c->buffers[i]); }
    g->options.max_tokens=FIXTURE_TOKENS;g->options.state_tile_rows=64;g->options.projection_tile_rows=1;
    g->options.prefill_rows=chunk;g->options.compressed_token_map=token_map;g->options.compressed_vocab=99092;g->options.pad_compressed_id=2;
    for (unsigned i=0;i<4;++i) g->history[i]=-1;
    for (unsigned i=0;i<RIA_GRAPH_VOCAB;++i) token_map[i]=i%99092;
    for (unsigned table=0;table<2;++table) {
        for (unsigned i=0;i<4;++i) g->options.hash_multipliers[table][i]=(uint64_t)(i+1)*3;
        for (unsigned i=0;i<24;++i) g->options.hash_primes[table][i]=16000001;
    }
    g->remote=(ria_graph_remote){.context=c,.experts=experts,.engram=rows,.experts_batch=expert_batch};
    for (unsigned layer=0;layer<40;++layer) {
        graph_layer *x=&g->layers[layer];ria_error e={0};
        CHECK(ria_graph_dependencies(layer,&x->kv_source,&x->index_source,&x->ratio,&e));
        x->hc_attn=matrix(layer,OP_HC,24,20480);x->hc_ffn=x->hc_attn;
        x->wqa=matrix(layer,OP_WQA,1280,5120);x->wqb=matrix(layer,OP_WQB,32768,1280);
        x->wkv=matrix(layer,OP_WKV,512,5120);x->woa=matrix(layer,OP_WOA,8192,32768);x->wob=matrix(layer,OP_WOB,5120,8192);
        x->router=matrix(layer,OP_ROUTER,384,5120);x->compressor_kv=matrix(layer,OP_COMPRESS_KV,512,5120);
        x->compressor_gate=matrix(layer,OP_COMPRESS_GATE,512,5120);x->index_q=matrix(layer,OP_INDEX_Q,4096,1280);
        x->index_weights=matrix(layer,OP_INDEX_WEIGHT,32,5120);x->index_key=matrix(layer,OP_INDEX_KEY,128,512);
        x->engram_kv=matrix(layer,OP_ENGRAM,5120,6144);x->bias=&biases[layer][0];x->bias_vl=&biases[layer][1];
        for (unsigned id=0;id<384;++id) g->local_index[layer][id]=-1;
    }
    if (locals) {
        ria_graph_local_expert *entries=calloc(2,sizeof(*entries));CHECK(entries);
        entries[0]=(ria_graph_local_expert){.layer=0,.expert_id=1,.tier=RIA_GRAPH_HOST,.phase_mask=1};
        entries[1]=(ria_graph_local_expert){.layer=0,.expert_id=4,.tier=RIA_GRAPH_VRAM,.phase_mask=7};
        entries[0].expert.gate.weight_global_scale=1;entries[1].expert.gate.weight_global_scale=4;
        g->options.local_experts=entries;g->options.local_expert_count=2;g->local_index[0][1]=0;g->local_index[0][4]=1;
    }
    uint64_t owner;ria_error e={0};
    CHECK(private_sizes(FIXTURE_TOKENS,local_count,0,chunk,&owner,&g->state_bytes,&e));
    g->host_allocation=calloc(1,(size_t)g->state_bytes);CHECK(g->host_allocation);
    uint8_t *cursor=g->host_allocation;
    for (unsigned layer=0;layer<40;++layer) { g->windows[layer]=cursor;cursor+=128*528; }
    for (unsigned source=0;source<4;++source) {
        uint64_t rows=FIXTURE_TOKENS/(source==3 ? 1 : 2);
        g->compressed[source]=cursor;cursor+=rows*288;g->index_keys[source]=cursor;cursor+=rows*68;
        g->pool_kv[source]=(float *)cursor;cursor+=4096;g->pool_gate[source]=(float *)cursor;cursor+=4096;
    }
    CHECK(prefill_bind(g,&e));
    g->head=matrix(0,OP_HEAD,129280,5120);return g;
}
static void discard(ria_graph *g) {
    ria_graph_cuda *c=g->cuda;
    for (unsigned i=0;i<RIA_G_BUFFER_COUNT;++i) free(c->buffers[i]);
    free((void *)g->options.local_experts);free(g->host_allocation);free(c);free(g);
}
static void same_state(const ria_graph *expected,const ria_graph *actual) {
    CHECK(expected->position==actual->position && !expected->poisoned && !actual->poisoned);
    CHECK(!memcmp(expected->history,actual->history,sizeof(expected->history)));
    CHECK(expected->selected_count==actual->selected_count);
    CHECK(!memcmp(expected->selected,actual->selected,expected->selected_count*sizeof(uint32_t)));
    CHECK(expected->cuda->candidate_count==actual->cuda->candidate_count);
    CHECK(!memcmp(expected->cuda->candidates,actual->cuda->candidates,expected->cuda->candidate_count*sizeof(uint32_t)));
    for (unsigned layer=0;layer<40;++layer) CHECK(!memcmp(expected->windows[layer],actual->windows[layer],128*528));
    for (unsigned owner=0;owner<4;++owner) {
        size_t rows=FIXTURE_TOKENS/(owner==3 ? 1 : 2);
        CHECK(!memcmp(expected->compressed[owner],actual->compressed[owner],rows*288));
        CHECK(!memcmp(expected->index_keys[owner],actual->index_keys[owner],rows*68));
        CHECK(!memcmp(expected->pool_kv[owner],actual->pool_kv[owner],1024*sizeof(float)));
        CHECK(!memcmp(expected->pool_gate[owner],actual->pool_gate[owner],1024*sizeof(float)));
    }
}
static void check_text(void) {
    const uint32_t count=130;uint32_t tokens[130];for (unsigned i=0;i<count;++i) tokens[i]=(i*7+3)%19;
    float *expected=calloc((size_t)count*129280,sizeof(float)),*actual=calloc((size_t)count*129280,sizeof(float));
    float *last=calloc(129280,sizeof(float));CHECK(expected && actual && last);
    ria_graph *reference=fixture(1,true);ria_error e={0};
    for (unsigned row=0;row<count;++row) CHECK(ria_graph_step(reference,tokens[row],false,NULL,expected+(uint64_t)row*129280,&e));
    for (unsigned option=0;option<6;++option) {
        uint32_t chunk=(const uint32_t[]){1,2,3,8,17,64}[option];ria_graph *grouped=fixture(chunk,true);
        CHECK(ria_graph_prefill(grouped,tokens,count,last,actual,&e));
        CHECK(!memcmp(expected,actual,(size_t)count*129280*sizeof(float)));
        CHECK(!memcmp(last,expected+(uint64_t)(count-1)*129280,129280*sizeof(float)));same_state(reference,grouped);
        CHECK(grouped->cuda->grouped_calls && grouped->cuda->local_calls);
        for (unsigned expert=0;expert<384;++expert) CHECK(grouped->cuda->expert_population[expert]);
        if (chunk>1) CHECK(grouped->cuda->multiple_rows);
        /* The next independent decode consumes all retained causal state,
         * including the incomplete ratio2 compressor pool at odd boundaries. */
        CHECK(ria_graph_set_phase(grouped,RIA_GRAPH_DECODE,&e));
        CHECK(ria_graph_step(grouped,13,false,NULL,last,&e));
        ria_graph *replay=fixture(1,true);for (unsigned row=0;row<count;++row)
            CHECK(ria_graph_step(replay,tokens[row],false,NULL,actual,&e));
        CHECK(ria_graph_set_phase(replay,RIA_GRAPH_DECODE,&e));CHECK(ria_graph_step(replay,13,false,NULL,actual,&e));
        CHECK(!memcmp(last,actual,129280*sizeof(float)));same_state(replay,grouped);
        discard(replay);discard(grouped);
    }
    discard(reference);free(expected);free(actual);free(last);
}
static void check_continuation_images_and_last_head(void) {
    uint32_t tokens[19];for (unsigned i=0;i<19;++i) tokens[i]=i;
    float embeddings[2][5120]={{0}};for (unsigned k=0;k<ACTIVE;++k) { embeddings[0][k]=1.125f+(float)k/128;embeddings[1][k]=0.75f; }
    const float *images[19]={0};images[2]=embeddings[0];images[3]=embeddings[1];images[14]=embeddings[0];
    float *expected=calloc(19*129280,sizeof(float)),*actual=calloc(19*129280,sizeof(float)),*last=calloc(129280,sizeof(float));CHECK(expected && actual && last);
    ria_graph *reference=fixture(1,true),*grouped=fixture(8,true);ria_error e={0};
    for (unsigned i=0;i<19;++i) reference->cuda->expected_images[i]=grouped->cuda->expected_images[i]=images[i]!=NULL;
    for (unsigned i=0;i<19;++i) CHECK(ria_graph_step(reference,tokens[i],images[i]!=NULL,images[i],expected+(uint64_t)i*129280,&e));
    /* Split at odd compressor/history boundaries, including an image span. */
    const uint32_t sizes[]={3,2,8,6};unsigned first=0;
    for (unsigned i=0;i<4;++i) {
        CHECK(ria_graph_set_phase(grouped,i ? RIA_GRAPH_CONTINUATION : RIA_GRAPH_PREFILL,&e));
        CHECK(ria_graph_prefill_rows(grouped,tokens+first,images+first,sizes[i],last,actual+(uint64_t)first*129280,NULL,NULL,&e));first+=sizes[i];
    }
    CHECK(!memcmp(expected,actual,19*129280*sizeof(float)));same_state(reference,grouped);
    CHECK(!memcmp(last,expected+18*129280,129280*sizeof(float)));
    discard(reference);discard(grouped);
    grouped=fixture(64,false);CHECK(ria_graph_prefill(grouped,tokens,19,last,NULL,&e));CHECK(grouped->cuda->observed_rows==1);
    reference=fixture(1,false);for (unsigned i=0;i<19;++i) CHECK(ria_graph_step(reference,tokens[i],false,NULL,actual,&e));
    CHECK(!memcmp(last,actual,129280*sizeof(float)));same_state(reference,grouped);
    discard(reference);discard(grouped);free(expected);free(actual);free(last);
}
static void check_failures(void) {
    uint32_t tokens[8]={1,2,3,4,5,6,7,8};float *last=calloc(129280,sizeof(float));CHECK(last);ria_error e={0};
    for (unsigned kind=0;kind<3;++kind) {
        ria_graph *g=fixture(8,false);
        for (unsigned row=0;row<3;++row) CHECK(ria_graph_step(g,tokens[row],false,NULL,last,&e));
        g->cuda->fail_group=kind==0;g->cuda->nonfinite_group=kind==1;g->cuda->unrounded_group=kind==2;
        CHECK(!ria_graph_prefill_rows(g,tokens,NULL,8,last,NULL,NULL,NULL,&e));CHECK(e.code==RIA_EXECUTOR_ERROR);
        CHECK(g->poisoned && g->position==3 && g->cuda->drains);
        CHECK(!ria_graph_step(g,0,false,NULL,last,&e));
        CHECK(ria_graph_reset(g,2,1,&e));CHECK(!g->poisoned && g->position==0);
        CHECK(ria_graph_prefill_rows(g,tokens,NULL,8,last,NULL,NULL,NULL,&e));
        ria_graph *fresh=fixture(8,false);float *oracle_logits=calloc(129280,sizeof(float));CHECK(oracle_logits);
        CHECK(ria_graph_prefill_rows(fresh,tokens,NULL,8,oracle_logits,NULL,NULL,NULL,&e));
        CHECK(!memcmp(last,oracle_logits,129280*sizeof(float)));same_state(fresh,g);
        free(oracle_logits);discard(fresh);discard(g);
    }
    ria_graph *g=fixture(8,false);tokens[7]=RIA_GRAPH_VOCAB;
    CHECK(!ria_graph_prefill_rows(g,tokens,NULL,8,last,NULL,NULL,NULL,&e));CHECK(e.code==RIA_INVALID_REQUEST);
    CHECK(!g->poisoned && g->position==0 && !g->cuda->begins && !g->cuda->grouped_calls);
    CHECK(!ria_graph_prefill_rows(g,tokens,NULL,9,last,NULL,NULL,NULL,&e));
    CHECK(!ria_graph_prefill_rows(g,tokens,NULL,0,last,NULL,NULL,NULL,&e));
    uint32_t invalid_late[17]={0};invalid_late[16]=RIA_GRAPH_VOCAB;
    CHECK(!ria_graph_prefill(g,invalid_late,17,last,NULL,&e));
    CHECK(!g->poisoned && g->position==0 && !g->cuda->begins && !g->cuda->grouped_calls);
    tokens[7]=8;float image[5120]={0};image[5119]=NAN;const float *images[8]={0};images[7]=image;
    CHECK(!ria_graph_prefill_rows(g,tokens,images,8,last,NULL,NULL,NULL,&e));
    CHECK(!g->poisoned && g->position==0 && !g->cuda->begins && !g->cuda->grouped_calls);
    g->remote.experts_batch=NULL;CHECK(!ria_graph_prefill_rows(g,tokens,NULL,8,last,NULL,NULL,NULL,&e));
    discard(g);free(last);
}
typedef struct { ria_graph *graph;unsigned stage,checks; } cancellation;
static bool cancel_at_stage(void *context) {
    cancellation *c=context;++c->checks;return c->graph->cuda->stage==c->stage;
}
static void check_cancellation(void) {
    const uint32_t tokens[8]={1,2,3,4,5,6,7,8};float *last=calloc(129280,sizeof(float)),*expected=calloc(129280,sizeof(float));
    CHECK(last && expected);ria_error e={0};
    for (unsigned stage=1;stage<=5;++stage) {
        ria_graph *g=fixture(8,false);
        for (unsigned row=0;row<3;++row) CHECK(ria_graph_step(g,tokens[row],false,NULL,last,&e));
        g->cuda->stage=0;cancellation c={g,stage,0};
        CHECK(!ria_graph_prefill_rows(g,tokens,NULL,8,last,NULL,cancel_at_stage,&c,&e));
        CHECK(e.code==RIA_CANCELLED && c.checks && g->poisoned && g->position==3 && g->cuda->drains);
        CHECK(ria_graph_reset(g,2,1,&e));
        CHECK(ria_graph_prefill_rows(g,tokens,NULL,8,last,NULL,NULL,NULL,&e));
        ria_graph *fresh=fixture(8,false);CHECK(ria_graph_prefill_rows(fresh,tokens,NULL,8,expected,NULL,NULL,NULL,&e));
        CHECK(!memcmp(last,expected,129280*sizeof(float)));same_state(fresh,g);discard(fresh);discard(g);
    }
    free(last);free(expected);
}
static void check_logit_aliases(void) {
    const uint32_t tokens[17]={0};ria_error e={0};
    float *all=calloc(17*129280+1,sizeof(float)),*expected=calloc(17*129280,sizeof(float)),*last=calloc(129280,sizeof(float));
    CHECK(all && expected && last);ria_graph *reference=fixture(8,false);
    CHECK(ria_graph_prefill(reference,tokens,17,last,expected,&e));
    ria_graph *g=fixture(8,false);
    /* Exact final-row alias spans several admitted chunks. Earlier chunk
     * heads may update it transiently; final teacher-forced rows stay exact. */
    CHECK(ria_graph_prefill(g,tokens,17,all+16*129280,all,&e));
    CHECK(!memcmp(all,expected,17*129280*sizeof(float)));same_state(reference,g);discard(g);
    g=fixture(8,false);CHECK(ria_graph_prefill_rows(g,tokens,NULL,8,all+7*129280,all,NULL,NULL,&e));
    CHECK(!memcmp(all,expected,8*129280*sizeof(float)));discard(g);
    for (unsigned offset=0;offset<3;++offset) {
        uint64_t index=(const uint64_t[]){0,1,7*129280-1}[offset];
        g=fixture(8,false);
        CHECK(!ria_graph_prefill_rows(g,tokens,NULL,8,all+index,all,NULL,NULL,&e));
        CHECK(e.code==RIA_INVALID_REQUEST && !g->poisoned && !g->position && !g->cuda->begins);discard(g);
        g=fixture(8,false);
        CHECK(!ria_graph_prefill(g,tokens,17,all+index,all,&e));
        CHECK(e.code==RIA_INVALID_REQUEST && !g->poisoned && !g->position && !g->cuda->begins);discard(g);
    }
    discard(reference);free(all);free(expected);free(last);
}
int main(void) {
    check_text();check_continuation_images_and_last_head();check_failures();check_cancellation();check_logit_aliases();
    puts("grouped prompt: row/slot scatter, scalar parity, 130-token sparse rings, cache/candidate/history, continuation/images, failure/cancellation/reset: passed (no CUDA)");return 0;
}
