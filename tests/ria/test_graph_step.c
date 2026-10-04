#define _GNU_SOURCE
/* Execute the real per-token graph orchestrator with an offline transport and
 * CUDA command recorder. This checks source forward/lifetime semantics, not
 * numerical CUDA qualification. No model, driver, or GPU is used. */
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "../../ria/graph.c"

struct ria_graph_cuda {
    float *buffers[RIA_G_BUFFER_COUNT];
    unsigned begins,collapses,token_marker,window_extent;
    unsigned retained_candidate;
    uint8_t window_markers[130][128];
    unsigned window_extents[130];
};

float *ria_graph_cuda_buffer(ria_graph_cuda *c,unsigned id) { return c->buffers[id]; }
bool ria_graph_cuda_drain(ria_graph_cuda *c,ria_error *e) { (void)c;(void)e;return true; }
bool ria_graph_cuda_begin_step(ria_graph_cuda *c,ria_error *e) {
    (void)e;assert(c->retained_candidate==73);++c->begins;c->collapses=0;
    const float identity[4]={1,0,0,0};memcpy(c->buffers[RIA_G_PRE],identity,sizeof(identity));return true;
}
bool ria_graph_cuda_copy(ria_graph_cuda *c,float *to,const float *from,uint64_t n,ria_error *e) {
    (void)c;(void)e;memcpy(to,from,(size_t)n*sizeof(float));return true;
}
bool ria_graph_cuda_upload(ria_graph_cuda *c,float *to,const float *from,uint64_t n,ria_error *e) { return ria_graph_cuda_copy(c,to,from,n,e); }
bool ria_graph_cuda_download(ria_graph_cuda *c,const float *from,float *to,uint64_t n,ria_error *e) { return ria_graph_cuda_copy(c,to,from,n,e); }
bool ria_graph_cuda_tensor(ria_graph_cuda *c,const ria_tensor *t,uint64_t first,float *to,uint64_t n,ria_error *e) {
    (void)c;(void)t;(void)first;(void)e;for (uint64_t i=0;i<n;++i) to[i]=1;return true;
}
bool ria_graph_cuda_expand(ria_graph_cuda *c,const float *input,float *h,ria_error *e) {
    (void)c;(void)e;for (unsigned i=0;i<20480;++i) h[i]=input[i%5120];return true;
}
bool ria_graph_cuda_project(ria_graph_cuda *c,const ria_expert_matrix *m,const float *input,float *output,bool rounded,ria_error *e) {
    (void)c;(void)input;(void)rounded;(void)e;memset(output,0,(size_t)m->out_features*sizeof(float));return true;
}
bool ria_graph_cuda_norm(ria_graph_cuda *c,float *values,uint64_t rows,uint64_t width,const ria_tensor *weight,bool layer,ria_error *e) {
    (void)c;(void)values;(void)rows;(void)width;(void)weight;(void)layer;(void)e;return true;
}
bool ria_graph_cuda_mhc(ria_graph_cuda *c,const float *h,float *mix,const ria_tensor *scale,const ria_tensor *base,float *pre,float *post,float *comb,ria_error *e) {
    (void)c;(void)h;(void)mix;(void)scale;(void)base;(void)e;
    for (unsigned i=0;i<4;++i) { pre[i]=0.25f*(float)(i+1);post[i]=0; }
    memset(comb,0,16*sizeof(float));return true;
}
bool ria_graph_cuda_collapse(ria_graph_cuda *c,const float *h,const float *pre,float *out,ria_error *e) {
    (void)h;(void)e;
    if (!c->collapses++) { assert(pre==c->buffers[RIA_G_PRE]);assert(pre[0]==1 && pre[1]==0 && pre[2]==0 && pre[3]==0); }
    memset(out,0,5120*sizeof(float));return true;
}
bool ria_graph_cuda_post(ria_graph_cuda *c,const float *x,const float *r,const float *p,const float *comb,float *h,ria_error *e) {
    (void)c;(void)x;(void)p;(void)comb;(void)e;memcpy(h,r,20480*sizeof(float));return true;
}
bool ria_graph_cuda_rope(ria_graph_cuda *c,float *v,uint64_t rows,uint64_t width,uint64_t pos,uint32_t ratio,bool inverse,ria_error *e) {
    (void)c;(void)v;(void)rows;(void)width;(void)pos;(void)ratio;(void)inverse;(void)e;return true;
}
bool ria_graph_cuda_pack(ria_graph_cuda *c,const float *v,uint64_t width,uint32_t rep,uint8_t *host,ria_error *e) {
    (void)v;(void)e;uint64_t bytes=rep==1 ? width+width/32 : width/2+width/(rep==2 ? 16 : 32);
    memset(host,(int)c->token_marker,(size_t)bytes);return true;
}
bool ria_graph_cuda_unpack(ria_graph_cuda *c,const uint8_t *host,uint64_t rows,uint64_t width,uint32_t rep,float *out,ria_error *e) {
    (void)e;
    if (rep==1) {
        unsigned position=c->token_marker-1,extent=position ? 128 : 1;
        unsigned valid=position<128 ? position+1 : 128,invalid=extent-valid;
        assert(width==512 && rows==extent);c->window_extent=extent;
        c->window_extents[position]=extent;
        for (unsigned row=0;row<extent;++row) {
            unsigned marker=row<invalid ? 0 : c->token_marker-valid+1+row-invalid;
            assert(host[(uint64_t)row*528]==marker);
            c->window_markers[position][row]=host[(uint64_t)row*528];
        }
    } else if (rep==2) {
        /* Compressed operands follow the complete source sparse ring extent,
         * rather than the compact count of currently valid window rows. */
        assert(out==c->buffers[RIA_G_ATTN_KV]+c->window_extent*512);
    }
    memset(out,0,(size_t)(rows*width)*sizeof(float));return true;
}
bool ria_graph_cuda_pool(ria_graph_cuda *c,float *kv,float *scores,uint32_t ratio,float *latent,ria_error *e) {
    (void)c;(void)kv;(void)scores;(void)ratio;(void)e;memset(latent,0,512*sizeof(float));return true;
}
bool ria_graph_cuda_quantize_inplace(ria_graph_cuda *c,float *v,uint64_t n,uint32_t rep,ria_error *e) {
    (void)c;(void)v;(void)n;(void)rep;(void)e;return true;
}
bool ria_graph_cuda_index_scores(ria_graph_cuda *c,const float *q,const float *weights,const float *keys,uint64_t first,uint64_t count,ria_error *e) {
    (void)c;(void)q;(void)weights;(void)keys;(void)first;(void)count;(void)e;return true;
}
bool ria_graph_cuda_select(ria_graph_cuda *c,uint64_t count,bool source,bool candidates,uint32_t *positions,uint32_t *chosen,ria_error *e) {
    (void)c;(void)source;(void)candidates;(void)e;assert(count);positions[0]=0;*chosen=1;return true;
}
bool ria_graph_cuda_attention(ria_graph_cuda *c,const float *q,const float *kv,uint64_t count,uint64_t masked_begin,uint64_t masked_end,const ria_tensor *sink,float *out,ria_error *e) {
    (void)q;(void)kv;(void)sink;(void)e;
    unsigned position=c->token_marker-1,valid=position<128 ? position+1 : 128;
    assert(count>=c->window_extent && count<=c->window_extent+1);
    assert(masked_begin==0 && masked_end==c->window_extent-valid);
    memset(out,0,32768*sizeof(float));return true;
}
bool ria_graph_cuda_route(ria_graph_cuda *c,const float *scores,const ria_tensor *bias,uint16_t ids[6],float coefficients[6],ria_error *e) {
    (void)c;(void)scores;(void)bias;(void)e;for (unsigned i=0;i<6;++i) { ids[i]=(uint16_t)i;coefficients[i]=0.25f; }return true;
}
bool ria_graph_cuda_shared(ria_graph_cuda *c,const ria_expert *x,const float *input,float *out,ria_error *e) {
    (void)c;(void)x;(void)input;(void)e;memset(out,0,5120*sizeof(float));return true;
}
bool ria_graph_cuda_merge(ria_graph_cuda *c,const uint16_t ids[6],const float *results,const float *shared,float *out,ria_error *e) {
    (void)c;(void)ids;(void)results;(void)shared;(void)e;memset(out,0,5120*sizeof(float));return true;
}
bool ria_graph_cuda_local_evaluate(ria_graph_cuda *c,const ria_expert *x,const ria_expert_cuda_resident *resident,const float *input,float coefficient,float *out,ria_error *e) {
    (void)c;(void)x;(void)resident;(void)input;(void)coefficient;(void)out;return ria_fail(e,RIA_INTERNAL_ERROR,"unexpected local fixture branch");
}
bool ria_graph_cuda_engram(ria_graph_cuda *c,const uint8_t *rows,float *out,ria_error *e) {
    (void)c;(void)rows;(void)e;memset(out,0,6144*sizeof(float));return true;
}
bool ria_graph_cuda_engram_fuse(ria_graph_cuda *c,float *h,const float *kv,const ria_tensor *q,const ria_tensor *k,bool enabled,ria_error *e) {
    (void)c;(void)h;(void)kv;(void)q;(void)k;(void)enabled;(void)e;return true;
}
static bool rows(void *context,uint32_t layer,const uint64_t *ids,uint32_t count,uint8_t *out,ria_error *e) {
    (void)context;(void)layer;(void)ids;(void)e;memset(out,0,(size_t)count*264);return true;
}
static bool experts(void *context,uint32_t layer,const float *input,const uint16_t *ids,const float *weights,const uint16_t *slots,uint32_t count,float *out,ria_error *e) {
    (void)context;(void)layer;(void)input;(void)ids;(void)weights;(void)slots;(void)e;memset(out,0,(size_t)count*5120*sizeof(float));return true;
}
int main(int argc,char **argv) {
    ria_graph *g=calloc(1,sizeof(*g));ria_graph_cuda c={0};assert(g);
    for (unsigned i=0;i<RIA_G_BUFFER_COUNT;++i) { c.buffers[i]=calloc(129280,sizeof(float));assert(c.buffers[i]); }
    uint32_t map[2]={0,1};g->cuda=&c;g->options.max_tokens=130;g->options.state_tile_rows=1;g->options.prefill_rows=1;g->options.compressed_token_map=map;
    g->options.compressed_vocab=99092;g->options.pad_compressed_id=2;g->remote.engram=rows;g->remote.experts=experts;
    for (unsigned i=0;i<4;++i) g->history[i]=-1;
    for (unsigned layer=0;layer<2;++layer) {
        for (unsigned i=0;i<4;++i) g->options.hash_multipliers[layer][i]=1;
        for (unsigned i=0;i<24;++i) g->options.hash_primes[layer][i]=16000001;
    }
    for (unsigned layer=0;layer<40;++layer) {
        ria_error e={0};assert(ria_graph_dependencies(layer,&g->layers[layer].kv_source,&g->layers[layer].index_source,&g->layers[layer].ratio,&e));
        g->windows[layer]=calloc(128,528);assert(g->windows[layer]);
        for (unsigned expert=0;expert<384;++expert) g->local_index[layer][expert]=-1;
    }
    for (unsigned i=0;i<4;++i) {
        g->compressed[i]=calloc(130,288);g->index_keys[i]=calloc(130,68);g->pool_kv[i]=calloc(1024,sizeof(float));g->pool_gate[i]=calloc(1024,sizeof(float));
        assert(g->compressed[i] && g->index_keys[i] && g->pool_kv[i] && g->pool_gate[i]);
    }
    g->head.out_features=129280;c.retained_candidate=73;c.buffers[RIA_G_PRE][0]=1;
    float *logits=calloc(129280,sizeof(float));ria_error e={0};assert(logits);
    for (unsigned token=0;token<130;++token) {
        c.collapses=0;c.token_marker=token+1;
        assert(ria_graph_step(g,token%2,false,NULL,logits,&e));assert(c.collapses==81);
        /* A completed source forward leaves a deliberately nonidentity final
         * layer mix; the next forward must disregard it, without cache reset. */
        for (unsigned i=0;i<4;++i) assert(c.buffers[RIA_G_PRE][i]==0.25f*(float)(i+1));
        assert(g->position==token+1 && g->windows[0][(token%128)*528]==token+1 && c.retained_candidate==73);
    }
    assert(c.begins==130 && g->windows[0][0]==129 && g->windows[0][528]==130 && g->compressed[3][0]==1 && g->compressed[3][288]==2);
    free(logits);for (unsigned i=0;i<40;++i) free(g->windows[i]);
    for (unsigned i=0;i<4;++i) { free(g->compressed[i]);free(g->index_keys[i]);free(g->pool_kv[i]);free(g->pool_gate[i]); }
    for (unsigned i=0;i<RIA_G_BUFFER_COUNT;++i) free(c.buffers[i]);
    free(g);
    if (argc==2 && !strcmp(argv[1],"--report")) {
        putchar('[');
        for (unsigned position=0;position<130;++position) {
            if (position) putchar(',');
            putchar('[');
            for (unsigned row=0;row<c.window_extents[position];++row) {
                if (row) putchar(',');
                printf("%u",(unsigned)c.window_markers[position][row]);
            }
            putchar(']');
        }
        puts("]");return 0;
    }
    puts("130-token graph mix/ring/slot lifetime regression: passed (no CUDA execution)");return 0;
}
