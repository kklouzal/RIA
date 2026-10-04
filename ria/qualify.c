#define _POSIX_C_SOURCE 200809L
/* Standalone bounded operator qualification. No model/store loader. Results
 * are raw measurements, never acceptance: an external preregistered policy
 * and process supervisor own tolerances, physical admission and hard CUDA
 * deadlines. CPU oracles below deliberately do not use numeric.h or the
 * production quantizer/dot implementation. */
#include "expert.h"
#include "json.h"
#ifndef RIA_WITH_CUDA
#define RIA_QUALIFY_CPU_ONLY 1
#endif
#ifndef RIA_QUALIFY_CPU_ONLY
#include "expert_cuda.h"
#include "graph_cuda.h"
#include <cuda_runtime_api.h>
#endif
#include <errno.h>
#include <fenv.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <openssl/evp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#if defined(__linux__)
#include <sys/prctl.h>
#endif
#include <time.h>

#define Q_MAX_CASES 32u
#define Q_MAX_REPEATS 32u
#define Q_JSON_BYTES 16384u
typedef struct {
    const char *profile_name,*executor,*runner_executor,*role,*gpu_uuid,*shape;
    const char *identity[7];
    ria_expert_profile profile;
    uint64_t host_budget,device_budget,pinned_budget,deadline_ms,seed;
    unsigned repeats,warmup;
    double floor;
    uint8_t request_digest[32];
} q_request;
typedef struct {
    const char *id,*component,*path,*timing_group;
    uint64_t rows,input,mid,output,elements;
    uint8_t input_digest[32],oracle_digest[32],actual_digest[32];
    double absolute,relative,rms;
    uint64_t times[Q_MAX_REPEATS],host_bytes,device_bytes,pinned_bytes,startup_ns;
    unsigned samples,warmup;
    bool exact,repeat_identical,repeat_checked,requires_exact,placement_checked,placement_identical;
} q_result;
typedef struct {
    q_request request;
    q_result results[Q_MAX_CASES];
    unsigned count;
    uint64_t start_ns,host_current,host_peak,device_current,device_peak,pinned_current,pinned_peak,graph_startup_ns;
    ria_error error;
} q_context;
typedef struct { const void *data;size_t bytes; } q_part;
static uint64_t q_clock(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC,&t)!=0 || t.tv_sec<0) return 0;
    return (uint64_t)t.tv_sec*UINT64_C(1000000000)+(uint64_t)t.tv_nsec;
}
static bool q_alive(q_context *q) {
    uint64_t now=q_clock();
    return (now && now>=q->start_ns && (now-q->start_ns)/UINT64_C(1000000)<q->request.deadline_ms) ||
        ria_fail(&q->error,RIA_DEADLINE_EXCEEDED,"fixture monotonic deadline expired; CUDA calls require external process supervision");
}
static bool q_reserve(q_context *q,uint64_t bytes) {
    uint64_t next;
    if (!ria_u64_add(q->host_current,bytes,&next) || next>q->request.host_budget)
        return ria_fail(&q->error,RIA_RESOURCE_LIMIT,"fixture owned host allocation exceeds declared cap");
    q->host_current=next;if (next>q->host_peak) q->host_peak=next;return true;
}
static void *q_alloc(q_context *q,uint64_t bytes) {
    if (!bytes || bytes>SIZE_MAX || !q_reserve(q,bytes)) return NULL;
    void *p=calloc(1,(size_t)bytes);
    if (!p) { q->host_current-=bytes;ria_error_set(&q->error,RIA_RESOURCE_LIMIT,"fixture allocation failed"); }
    return p;
}
static void q_free(q_context *q,void *p,uint64_t bytes) { if (p) { free(p);q->host_current-=bytes; } }
static bool q_hash(q_context *q,const q_part *parts,unsigned count,uint8_t digest[32]) {
    EVP_MD_CTX *ctx=EVP_MD_CTX_new();unsigned length=0;
    bool ok=ctx && EVP_DigestInit_ex(ctx,EVP_sha256(),NULL)==1;
    for (unsigned i=0;ok && i<count;++i) if (parts[i].bytes) ok=EVP_DigestUpdate(ctx,parts[i].data,parts[i].bytes)==1;
    if (ok) ok=EVP_DigestFinal_ex(ctx,digest,&length)==1 && length==32;
    EVP_MD_CTX_free(ctx);
    return ok || ria_fail(&q->error,RIA_INTERNAL_ERROR,"hash qualification fixture");
}
static float o_bf16(float v) {
    uint32_t bits;memcpy(&bits,&v,4);
    if (!isfinite(v)) return v;
    uint32_t magnitude=bits&UINT32_C(0x7fffffff),low=magnitude>>16;
    uint32_t a_bits=low<<16,b_bits=(low+1)<<16;float a,b;
    memcpy(&a,&a_bits,4);memcpy(&b,&b_bits,4);
    double upper=isinf(b) ? ldexp(1.0,128) : (double)b;
    double da=fabs((double)fabsf(v)-(double)a),db=fabs(upper-(double)fabsf(v));
    float answer=db<da || (db==da && (low&1)) ? b : a;
    return copysignf(answer,v);
}
static float o_f4(unsigned code) {
    static const float palette[8]={0,.5f,1,1.5f,2,3,4,6};
    return copysignf(palette[code&7],code&8 ? -1.0f : 1.0f);
}
static float o_f8(unsigned code) {
    unsigned c=code&127,e=c/8,m=c%8;
    float v=e ? ldexpf(1+(float)m/8,(int)e-7) : (float)m/512;
    if (c==127) return NAN;
    return copysignf(v,code&128 ? -1.0f : 1.0f);
}
static uint8_t o_encode(float v,bool fp8) {
    unsigned count=fp8 ? 127 : 8,winner=0;double distance=INFINITY;
    for (unsigned code=0;code<count;++code) {
        double d=fabs((double)fabsf(v)-(double)(fp8 ? o_f8(code) : o_f4(code)));
        if (d<distance || (d==distance && !(code&1))) { winner=code;distance=d; }
    }
    return (uint8_t)(winner|(signbit(v) ? (fp8 ? 128u : 8u) : 0u));
}
static uint8_t o_power_scale(float target) {
    int exponent;float fraction=frexpf(target,&exponent);
    if (fraction==.5f) --exponent;
    return (uint8_t)(exponent+127);
}
static uint64_t q_groups(uint64_t n,uint64_t width) { return n/width+(n%width!=0); }
static void o_quant(const float *x,uint64_t n,ria_expert_profile p,float global,float *raw,float *scale) {
    uint64_t group=p==RIA_EXPERT_FP8 ? 32 : p==RIA_EXPERT_NVFP4 ? 16 : n;
    for (uint64_t first=0;first<n;first+=group) {
        uint64_t end=first+group<n ? first+group : n;float maximum=0;
        for (uint64_t k=first;k<end;++k) maximum=fmaxf(maximum,fabsf(o_bf16(x[k])));
        float factor=1;
        if (p==RIA_EXPERT_FP8) factor=ldexpf(1,(int)o_power_scale(fmaxf(maximum,1e-4f)*(1.0f/448))-127);
        else if (p==RIA_EXPERT_NVFP4) factor=o_f8(o_encode((maximum/6)/global,true));
        scale[first/group]=factor;
        for (uint64_t k=first;k<end;++k) {
            float v=o_bf16(x[k]);
            if (p==RIA_EXPERT_FP8) v=o_f8(o_encode(v/factor,true));
            else if (p==RIA_EXPERT_NVFP4) v=o_f4(o_encode(factor>0 ? v/(factor*global) : 0,false));
            raw[k]=v;
        }
    }
}
static float o_weight(const ria_expert_matrix *m,uint64_t n,uint64_t k) {
    const uint8_t *p=m->values+n*m->value_row_stride;
    if (m->profile==RIA_EXPERT_BF16) { uint32_t bits=(uint32_t)ria_read_u16(p+k*2)<<16;float v;memcpy(&v,&bits,4);return v; }
    return m->profile==RIA_EXPERT_FP8 ? o_f8(p[k]) : o_f4((unsigned)(p[k/2]>>(4*(k%2)))&15);
}
static void o_project(const ria_expert_matrix *m,const float *x,float *out,float *raw,float *factors) {
    uint64_t group=m->profile==RIA_EXPERT_FP8 ? 32 : m->profile==RIA_EXPERT_NVFP4 ? 16 : m->in_features;
    o_quant(x,m->in_features,m->profile,m->activation_global_scale,raw,factors);
    for (uint64_t n=0;n<m->out_features;++n) {
        float total=0;
        for (uint64_t base=0;base<m->in_features;base+=group) {
            uint64_t end=base+group<m->in_features ? base+group : m->in_features;float partial=0;
            for (uint64_t k=base;k<end;++k) partial=fmaf(o_weight(m,n,k),raw[k],partial);
            if (m->profile!=RIA_EXPERT_BF16) {
                uint8_t code=m->scales[(m->profile==RIA_EXPERT_FP8 ? n/32 : n)*m->scale_row_stride+base/group];
                float weight=m->profile==RIA_EXPERT_FP8 ? ldexpf(1,(int)code-127) : o_f8(code);
                partial=(partial*factors[base/group])*weight;
            }
            total+=partial;
        }
        if (m->profile==RIA_EXPERT_NVFP4) total*=m->weight_global_scale*m->activation_global_scale;
        out[n]=o_bf16(total);
    }
}
static void o_expert(const ria_expert *e,const float *x,uint64_t rows,const float *co,float *out,float *scratch) {
    uint64_t width=e->gate.in_features>e->gate.out_features ? e->gate.in_features : e->gate.out_features;
    float *raw=scratch,*factors=raw+width,*gate=factors+q_groups(width,16),*up=gate+e->gate.out_features,*mid=up+e->gate.out_features;
    for (uint64_t row=0;row<rows;++row) {
        o_project(&e->gate,x+row*e->gate.in_features,gate,raw,factors);
        o_project(&e->up,x+row*e->gate.in_features,up,raw,factors);
        for (uint64_t k=0;k<e->gate.out_features;++k) {
            float g=fminf(gate[k],10),u=fmaxf(-10,fminf(up[k],10));
            mid[k]=o_bf16(((g/(1+expf(-g)))*u)*co[row]);
        }
        o_project(&e->down,mid,out+row*e->down.out_features,raw,factors);
    }
}
/* A unit diagonal has no nontrivial dot reduction. This closed-form path
 * separately checks casts, per-domain factors and coefficient placement. */
static void o_unit_expert(const ria_expert *e,const float *x,uint64_t rows,const float *co,float *out,float *scratch) {
    uint64_t k=e->gate.in_features,m=e->gate.out_features,n=e->down.out_features;
    float *raw=scratch,*factors=raw+k,*mid=factors+q_groups(k,16);
    uint64_t group=e->gate.profile==RIA_EXPERT_FP8 ? 32 : e->gate.profile==RIA_EXPERT_NVFP4 ? 16 : k;
    for (uint64_t row=0;row<rows;++row) {
        o_quant(x+row*k,k,e->gate.profile,1,raw,factors);
        for (uint64_t i=0;i<m;++i) {
            float g=o_bf16(raw[i]*factors[i/group]),u=fmaxf(-10,fminf(g,10));g=fminf(g,10);
            mid[i]=o_bf16(((g/(1+expf(-g)))*u)*co[row]);
        }
        o_quant(mid,m,e->gate.profile,1,raw,factors);
        uint64_t down_group=e->gate.profile==RIA_EXPERT_BF16 ? m : group;
        for (uint64_t i=0;i<n;++i) out[row*n+i]=i<m ? o_bf16(raw[i]*factors[i/down_group]) : 0;
    }
}
static void o_packed(const float *x,uint64_t width,unsigned rep,uint8_t *packed,float *decoded) {
    uint64_t group=rep==2 ? 16 : 32;
    for (uint64_t base=0;base<width;base+=group) {
        float maximum=0;for (uint64_t k=base;k<base+group;++k) maximum=fmaxf(maximum,fabsf(o_bf16(x[k])));
        uint8_t code;float scale;
        if (rep==1) { code=o_power_scale(fmaxf(maximum,1e-4f)*(1.0f/448));scale=ldexpf(1,(int)code-127); }
        else if (rep==2) { code=o_encode(fmaxf(maximum,6*0x1p-9f)/6,true);scale=o_f8(code); }
        else { code=o_power_scale(fmaxf(maximum,6*0x1p-126f)*(1.0f/6));scale=ldexpf(1,(int)code-127); }
        packed[(rep==1 ? width : width/2)+base/group]=code;
        for (uint64_t k=base;k<base+group;++k) {
            uint8_t v=o_encode(o_bf16(x[k])/scale,rep==1);
            if (rep==1) packed[k]=v;else if (!(k%2)) packed[k/2]=v;else packed[k/2]|=(uint8_t)(v<<4);
            decoded[k]=o_bf16((rep==1 ? o_f8(v) : o_f4(v))*scale);
        }
    }
}
static void o_mhc(const float *h,const float *mix,const float *scale,const float *base,float *pre,float *post,float *comb) {
    float square=0;for (unsigned i=0;i<20480;++i) square=fmaf(h[i],h[i],square);
    float inverse=1/sqrtf(square/20480+1e-20f),normalized[24];
    for (unsigned i=0;i<24;++i) normalized[i]=mix[i]*inverse;
    for (unsigned i=0;i<4;++i) {
        pre[i]=1/(1+expf(-(normalized[i]*scale[0]+base[i])))+1e-6f;
        post[i]=2/(1+expf(-(normalized[i+4]*scale[1]+base[i+4])));
    }
    for (unsigned row=0;row<4;++row) {
        float peak=-INFINITY,sum=0;
        for (unsigned col=0;col<4;++col) { unsigned i=row*4+col;comb[i]=normalized[i+8]*scale[2]+base[i+8];peak=fmaxf(peak,comb[i]); }
        for (unsigned col=0;col<4;++col) { unsigned i=row*4+col;comb[i]=expf(comb[i]-peak);sum+=comb[i]; }
        for (unsigned col=0;col<4;++col) comb[row*4+col]=comb[row*4+col]/sum+1e-6f;
    }
    for (unsigned iter=0;iter<20;++iter) {
        if (iter) for (unsigned row=0;row<4;++row) {
            float sum=1e-6f;for (unsigned col=0;col<4;++col) sum+=comb[row*4+col];
            for (unsigned col=0;col<4;++col) comb[row*4+col]/=sum;
        }
        for (unsigned col=0;col<4;++col) {
            float sum=1e-6f;for (unsigned row=0;row<4;++row) sum+=comb[row*4+col];
            for (unsigned row=0;row<4;++row) comb[row*4+col]/=sum;
        }
    }
}
/* Independent slow scalar source algebra. Each 64-key softmax block retains
 * the source denominator precision and BF16 probability-before-value cast. */
static void o_attention(const float *query,const float *kv,unsigned count,const float *sink,float *out) {
    for (unsigned head=0;head<64;++head) {
        float sum=0,peak=-1e30f,accumulator[512]={0};
        for (unsigned first=0;first<count;first+=64) {
            unsigned last=first+64<count ? first+64 : count;float scores[64],prob[64],previous=peak;
            for (unsigned row=first;row<last;++row) {
                float dot=0;for (unsigned k=0;k<512;++k) dot=fmaf(query[head*512+k],kv[row*512+k],dot);
                scores[row-first]=dot*0.04419417382415922f;peak=fmaxf(peak,scores[row-first]);
            }
            float rescale=expf(previous-peak),block_sum=0;
            for (unsigned row=first;row<last;++row) { float p=expf(scores[row-first]-peak);block_sum+=p;prob[row-first]=o_bf16(p); }
            sum=sum*rescale+block_sum;
            for (unsigned k=0;k<512;++k) {
                accumulator[k]*=rescale;
                for (unsigned row=first;row<last;++row) accumulator[k]=fmaf(prob[row-first],kv[row*512+k],accumulator[k]);
            }
        }
        sum+=expf(sink[head]-peak);
        for (unsigned k=0;k<512;++k) out[head*512+k]=o_bf16(accumulator[k]/sum);
    }
}
static uint64_t q_mix(uint64_t n) {
    n^=n>>30;n*=UINT64_C(0xbf58476d1ce4e5b9);n^=n>>27;n*=UINT64_C(0x94d049bb133111eb);return n^(n>>31);
}
static bool q_matrix(q_context *q,uint64_t n,uint64_t k,unsigned projection,bool unit,ria_expert_matrix *m) {
    memset(m,0,sizeof(*m));m->profile=q->request.profile;m->out_features=n;m->in_features=k;
    m->weight_global_scale=unit ? 1 : .125f;m->activation_global_scale=unit ? 1 : .0625f;
    m->value_row_stride=m->profile==RIA_EXPERT_BF16 ? k*2 : m->profile==RIA_EXPERT_FP8 ? k : q_groups(k,2);
    m->values_bytes=n*m->value_row_stride;
    m->values=q_alloc(q,m->values_bytes);if (!m->values) return false;
    if (m->profile!=RIA_EXPERT_BF16) {
        m->scale_row_stride=q_groups(k,m->profile==RIA_EXPERT_FP8 ? 32 : 16);
        m->scales_bytes=q_groups(n,m->profile==RIA_EXPERT_FP8 ? 32 : 1)*m->scale_row_stride;
        m->scales=q_alloc(q,m->scales_bytes);if (!m->scales) return false;
        for (uint64_t i=0;i<m->scales_bytes;++i) ((uint8_t *)m->scales)[i]=unit ? (m->profile==RIA_EXPERT_FP8 ? 127 : 56) :
            (uint8_t)(m->profile==RIA_EXPERT_FP8 ? 123+q_mix(i+projection+q->request.seed)%7 : 40+q_mix(i+projection+q->request.seed)%32);
    }
    for (uint64_t row=0;row<n;++row) {
      if (!(row%32) && !q_alive(q)) return false;
      for (uint64_t col=0;col<k;++col) {
        uint64_t random=q_mix(q->request.seed+row*k+col+(uint64_t)projection*UINT64_C(0x100000001));
        uint8_t *p=(uint8_t *)m->values+row*m->value_row_stride;
        if (m->profile==RIA_EXPERT_BF16) {
            float v=unit ? (row==col ? 1 : 0) : (float)((int)(random%63)-31)/32;
            if (!unit && col==0 && row%3==0) v=32;
            uint32_t bits;memcpy(&bits,&v,4);ria_write_u16(p+col*2,(uint16_t)(bits>>16));
        } else {
            uint8_t code=unit ? (row==col ? (m->profile==RIA_EXPERT_FP8 ? 56 : 2) : 0) :
                (uint8_t)(m->profile==RIA_EXPERT_FP8 ? (random%127)|((random>>9)&128) : random%16);
            if (m->profile==RIA_EXPERT_FP8) p[col]=code;else p[col/2]|=(uint8_t)(code<<(4*(col%2)));
        }
      }
    }
    return true;
}
static void q_matrix_free(q_context *q,ria_expert_matrix *m) {
    q_free(q,(void *)m->values,m->values_bytes);q_free(q,(void *)m->scales,m->scales_bytes);memset(m,0,sizeof(*m));
}
static q_result *q_new(q_context *q,const char *id,const char *component,const char *path,uint64_t rows,uint64_t input,uint64_t mid,uint64_t output) {
    if (q->count==Q_MAX_CASES) { ria_error_set(&q->error,RIA_RESOURCE_LIMIT,"fixture result capacity exhausted");return NULL; }
    q_result *r=&q->results[q->count++];r->id=id;r->timing_group=id;r->component=component;r->path=path;
    r->rows=rows;r->input=input;r->mid=mid;r->output=output;r->exact=true;r->repeat_identical=true;return r;
}
static bool q_measure(q_context *q,q_result *r,const float *expected,const float *actual,uint64_t count) {
    double squares=0;
    for (uint64_t i=0;i<count;++i) {
        if (!isfinite(expected[i]) || !isfinite(actual[i])) return ria_fail(&q->error,RIA_EXECUTOR_ERROR,"nonfinite qualification result");
        double d=fabs((double)expected[i]-(double)actual[i]),relative=d/fmax(fabs(expected[i]),q->request.floor);
        if (!isfinite(relative)) return ria_fail(&q->error,RIA_INVALID_REQUEST,"explicit relative floor produces metric overflow");
        if (d>r->absolute) r->absolute=d;
        if (relative>r->relative) r->relative=relative;
        squares+=d*d;
    }
    r->elements=count;r->rms=sqrt(squares/(double)count);r->exact=memcmp(expected,actual,(size_t)count*4)==0;
    r->host_bytes=q->host_current;r->device_bytes=q->device_current;r->pinned_bytes=q->pinned_current;
    q_part a={expected,(size_t)count*4},b={actual,(size_t)count*4};
    return q_hash(q,&a,1,r->oracle_digest) && q_hash(q,&b,1,r->actual_digest);
}
static bool q_experts(q_context *q,bool unit) {
    uint64_t k=!strcmp(q->request.shape,"target") ? 5120 : 65,m=!strcmp(q->request.shape,"target") ? 2304 : 33,
        n=!strcmp(q->request.shape,"target") ? 5120 : 19,rows=!strcmp(q->request.shape,"target") ? 1 : 17;
    uint64_t xb=rows*k*4,yb=rows*n*4,cb=rows*4,sb=(k+q_groups(k,16)+3*m)*4;
    ria_expert e;memset(&e,0,sizeof(e));e.clamp=10;
    float *x=NULL,*co=NULL,*expected=NULL,*actual=NULL,*scratch=NULL;ria_expert_cpu *cpu=NULL;
#ifndef RIA_QUALIFY_CPU_ONLY
    ria_expert_cuda *gpu=NULL;ria_expert_cuda_resident resident;memset(&resident,0,sizeof(resident));float *di=NULL,*dout=NULL;
    uint64_t gpu_host=0,pinned=0,extra=0;
#endif
    bool ok=false;uint64_t cpu_bytes=0;
    if (!q_matrix(q,m,k,0,unit,&e.gate) || !q_matrix(q,m,k,1,unit,&e.up) || !q_matrix(q,n,m,2,unit,&e.down)) goto done;
    x=q_alloc(q,xb);co=q_alloc(q,cb);expected=q_alloc(q,yb);actual=q_alloc(q,yb);scratch=q_alloc(q,sb);
    if (!x || !co || !expected || !actual || !scratch || !ria_expert_validate(&e,&q->error)) goto done;
    for (uint64_t i=0;i<rows*k;++i) x[i]=unit ? (float)((int)(i%13)-6) : o_bf16((float)((int)(q_mix(i+q->request.seed)%127)-63)/16);
    for (uint64_t row=0;row<rows;++row) { static const float values[4]={0,.0625f,.5f,1.5f};co[row]=rows==1 ? 1.5f : values[row%4]; }
    o_expert(&e,x,rows,co,expected,scratch);
    if (unit) {
        o_unit_expert(&e,x,rows,co,actual,scratch);
        if (memcmp(expected,actual,(size_t)yb)) { ria_error_set(&q->error,RIA_INTERNAL_ERROR,"scalar oracle disagrees with unit-diagonal closed form");goto done; }
    }
    if (!q_alive(q)) goto done;
    float globals[6]={e.gate.weight_global_scale,e.gate.activation_global_scale,e.up.weight_global_scale,e.up.activation_global_scale,e.down.weight_global_scale,e.down.activation_global_scale};
    q_part inputs[9]={{x,(size_t)xb},{co,(size_t)cb},{e.gate.values,(size_t)e.gate.values_bytes},{e.up.values,(size_t)e.up.values_bytes},
        {e.down.values,(size_t)e.down.values_bytes},{e.gate.scales,(size_t)e.gate.scales_bytes},
        {e.up.scales,(size_t)e.up.scales_bytes},{e.down.scales,(size_t)e.down.scales_bytes},{globals,sizeof(globals)}};
    q_result *r=q_new(q,unit ? "expert_unit_palette" : "expert_dense_clamp_route", "experts","host",rows,k,m,n);
    if (!r || !q_hash(q,inputs,9,r->input_digest)) goto done;
    r->warmup=q->request.warmup;
    if (!strcmp(q->request.runner_executor,"cpu")) {
        uint64_t bytes,startup=q_clock();
        if (!ria_expert_cpu_required_bytes(k,m,n,&bytes,&q->error) || !ria_u64_add(bytes,ria_expert_cpu_metadata_bytes(),&cpu_bytes) ||
            !q_reserve(q,cpu_bytes) || !ria_expert_cpu_create(k,m,n,&cpu,&q->error)) goto done;
        r->startup_ns=q_clock()-startup;
        for (unsigned pass=0;pass<q->request.warmup+q->request.repeats;++pass) {
            uint64_t start=q_clock();
            if (!q_alive(q) || !ria_expert_cpu_evaluate(cpu,&e,x,rows,k,co,actual,n,&q->error)) goto done;
            uint64_t elapsed=q_clock()-start;
            if (pass>=q->request.warmup) {
                uint8_t digest[32];q_part part={actual,(size_t)yb};if (!q_hash(q,&part,1,digest)) goto done;
                if (r->samples) { r->repeat_checked=true;if (memcmp(digest,r->actual_digest,32)) r->repeat_identical=false; }
                memcpy(r->actual_digest,digest,32);r->times[r->samples++]=elapsed;
            }
        }
        if (!q_measure(q,r,expected,actual,rows*n)) goto done;
    } else {
#ifndef RIA_QUALIFY_CPU_ONLY
        uint64_t resident_bytes;
        if (!ria_expert_cuda_resident_required_bytes(&e,&resident_bytes,&q->error) ||
            !ria_expert_cuda_pinned_required_bytes(k,m,n,rows,32,&pinned,&q->error)) goto done;
        extra=(k+n)*4;gpu_host=ria_expert_cuda_metadata_bytes();
        if (pinned>q->request.pinned_budget || resident_bytes>q->request.device_budget || extra>q->request.device_budget-resident_bytes ||
            !q_reserve(q,gpu_host+pinned)) { ria_error_set(&q->error,RIA_RESOURCE_LIMIT,"expert fixture reservation exceeds caps");goto done; }
        uint64_t startup=q_clock();
        if (!ria_expert_cuda_create_pooled(0,k,m,n,rows,32,q->request.device_budget-resident_bytes-extra,q->request.pinned_budget,&gpu,&q->error)) goto done;
        r->startup_ns=q_clock()-startup;
        q->pinned_current=pinned;if (pinned>q->pinned_peak) q->pinned_peak=pinned;
        q->device_current=ria_expert_cuda_device_bytes(gpu);if (q->device_current>q->device_peak) q->device_peak=q->device_current;
        for (unsigned pass=0;pass<q->request.warmup+q->request.repeats;++pass) {
            uint64_t start=q_clock();
            if (!q_alive(q) || !ria_expert_cuda_evaluate(gpu,&e,x,rows,k,co,actual,n,&q->error)) goto done;
            uint64_t elapsed=q_clock()-start;
            if (pass>=q->request.warmup) {
                uint8_t digest[32];q_part part={actual,(size_t)yb};if (!q_hash(q,&part,1,digest)) goto done;
                if (r->samples) { r->repeat_checked=true;if (memcmp(digest,r->actual_digest,32)) r->repeat_identical=false; }
                memcpy(r->actual_digest,digest,32);r->times[r->samples++]=elapsed;
            }
        }
        if (!q_measure(q,r,expected,actual,rows*n) || !q_alive(q)) goto done;
        startup=q_clock();
        if (!ria_expert_cuda_resident_create(gpu,&e,resident_bytes,&resident,&q->error)) goto done;
        q->device_current+=resident_bytes;
        cudaError_t status=cudaMalloc((void **)&di,(size_t)k*4);
        if (status==cudaSuccess) status=cudaMalloc((void **)&dout,(size_t)n*4);
        if (status!=cudaSuccess) { ria_error_set(&q->error,RIA_RESOURCE_LIMIT,"allocate fixture device rows: %s",cudaGetErrorString(status));goto done; }
        q->device_current+=extra;if (q->device_current>q->device_peak) q->device_peak=q->device_current;
        q_result *v=q_new(q,unit ? "expert_resident_unit_palette" : "expert_resident_dense_clamp_route","experts","vram",rows,k,m,n);
        if (!v) goto done;
        v->warmup=q->request.warmup;
        v->startup_ns=q_clock()-startup;
        memcpy(v->input_digest,r->input_digest,32);
        for (unsigned pass=0;pass<q->request.warmup+q->request.repeats;++pass) {
            uint64_t start=q_clock();
            for (uint64_t row=0;row<rows;++row) if (!q_alive(q) ||
                !ria_expert_cuda_upload_bytes(gpu,di,x+row*k,k*4,&q->error) ||
                !ria_expert_cuda_evaluate_device(gpu,&e,&resident,di,co[row],dout,&q->error) ||
                !ria_expert_cuda_download_bytes(gpu,dout,actual+row*n,n*4,&q->error)) goto done;
            uint64_t elapsed=q_clock()-start;
            if (pass>=q->request.warmup) {
                uint8_t digest[32];q_part part={actual,(size_t)yb};if (!q_hash(q,&part,1,digest)) goto done;
                if (v->samples) { v->repeat_checked=true;if (memcmp(digest,v->actual_digest,32)) v->repeat_identical=false; }
                memcpy(v->actual_digest,digest,32);v->times[v->samples++]=elapsed;
            }
        }
        if (!q_measure(q,v,expected,actual,rows*n)) goto done;
        v->placement_checked=true;v->placement_identical=memcmp(v->actual_digest,r->actual_digest,32)==0;
#else
        ria_error_set(&q->error,RIA_UNSUPPORTED,"CPU qualifier has no CUDA runtime dependency");goto done;
#endif
    }
    ok=true;
done:
    ria_expert_cpu_destroy(cpu);if (cpu) q->host_current-=cpu_bytes;
#ifndef RIA_QUALIFY_CPU_ONLY
    { ria_error cleanup={0};bool clean=ria_expert_cuda_resident_destroy(&resident,&cleanup);
      if (di && cudaFree(di)!=cudaSuccess) { if (clean) ria_error_set(&cleanup,RIA_EXECUTOR_ERROR,"release qualification input row");clean=false; }
      if (dout && cudaFree(dout)!=cudaSuccess) { if (clean) ria_error_set(&cleanup,RIA_EXECUTOR_ERROR,"release qualification output row");clean=false; }
      if (!ria_expert_cuda_destroy(gpu,&cleanup)) clean=false;
      if (ok && !clean) { q->error=cleanup;ok=false; }
      if (gpu) q->host_current-=gpu_host+pinned;
      q->device_current=0;q->pinned_current=0; }
#endif
    q_matrix_free(q,&e.gate);q_matrix_free(q,&e.up);q_matrix_free(q,&e.down);
    q_free(q,x,xb);q_free(q,co,cb);q_free(q,expected,yb);q_free(q,actual,yb);q_free(q,scratch,sb);return ok;
}

#ifndef RIA_QUALIFY_CPU_ONLY
static bool q_transfers(q_context *q) {
    ria_expert_cuda *c=NULL;void *device=NULL;uint8_t *source=NULL,*actual=NULL;uint64_t pinned=0,bytes=0,metadata=ria_expert_cuda_metadata_bytes();
    bool ok=false;
    if (!ria_expert_cuda_pinned_required_bytes(65,33,19,1,32,&pinned,&q->error)) return false;
    bytes=pinned+137;
    source=q_alloc(q,bytes);actual=q_alloc(q,bytes);
    if (!source || !actual || pinned>q->request.pinned_budget || bytes>=q->request.device_budget || !q_reserve(q,metadata+pinned)) goto done;
    uint64_t startup=q_clock();
    if (!ria_expert_cuda_create_pooled(0,65,33,19,1,32,q->request.device_budget-bytes,q->request.pinned_budget,&c,&q->error)) goto done;
    cudaError_t status=cudaMalloc(&device,(size_t)bytes);
    if (status!=cudaSuccess) { ria_error_set(&q->error,RIA_RESOURCE_LIMIT,"allocate transfer fixture: %s",cudaGetErrorString(status));goto done; }
    q->device_current=ria_expert_cuda_device_bytes(c)+bytes;q->pinned_current=pinned;
    if (q->device_current>q->device_peak) q->device_peak=q->device_current;
    if (pinned>q->pinned_peak) q->pinned_peak=pinned;
    for (uint64_t i=0;i<bytes;++i) source[i]=(uint8_t)q_mix(i+q->request.seed);
    q_result *r=q_new(q,"pinned_multichunk_odd_bytes","transfers","pinned",1,bytes,0,bytes);
    if (!r) goto done;
    r->warmup=q->request.warmup;
    r->startup_ns=q_clock()-startup;
    q_part input={source,(size_t)bytes};if (!q_hash(q,&input,1,r->input_digest)) goto done;
    for (unsigned pass=0;pass<q->request.warmup+q->request.repeats;++pass) {
        uint64_t start=q_clock();
        if (!q_alive(q) || !ria_expert_cuda_upload_bytes(c,device,source,bytes,&q->error) ||
            !ria_expert_cuda_download_bytes(c,device,actual,bytes,&q->error)) goto done;
        if (pass>=q->request.warmup) {
            uint64_t elapsed=q_clock()-start;uint8_t digest[32];q_part current={actual,(size_t)bytes};
            if (!q_hash(q,&current,1,digest)) goto done;
            if (r->samples) { r->repeat_checked=true;if (memcmp(r->actual_digest,digest,32)) r->repeat_identical=false; }
            memcpy(r->actual_digest,digest,32);r->times[r->samples++]=elapsed;
        }
    }
    r->requires_exact=true;r->exact=memcmp(source,actual,(size_t)bytes)==0;r->elements=bytes;
    r->host_bytes=q->host_current;r->device_bytes=q->device_current;r->pinned_bytes=q->pinned_current;
    q_part output={actual,(size_t)bytes};memcpy(r->oracle_digest,r->input_digest,32);
    if (!q_hash(q,&output,1,r->actual_digest)) goto done;
    ok=true;
done:{ria_error cleanup;if (device && cudaFree(device)!=cudaSuccess && ok) { ria_error_set(&q->error,RIA_EXECUTOR_ERROR,"release transfer fixture");ok=false; }
      if (!ria_expert_cuda_destroy(c,&cleanup) && ok) { q->error=cleanup;ok=false; }
      if (c) q->host_current-=metadata+pinned;
      q->device_current=0;q->pinned_current=0;}
    q_free(q,source,bytes);q_free(q,actual,bytes);return ok;
}
static ria_tensor q_tensor(const float *values,uint64_t count) {
    ria_tensor t;memset(&t,0,sizeof(t));t.data=(const uint8_t *)values;t.dtype="F32";t.length=count*4;return t;
}
static bool q_packed_cases(q_context *q,ria_graph_cuda *g,float *arena) {
    float *x=arena,*expected=arena+512,*actual=arena+1024;uint8_t want[528],got[528];
    for (unsigned rep=1;rep<=3;++rep) {
        uint64_t width=rep==3 ? 128 : 512,bytes=ria_graph_packed_stride(rep,width);
        for (uint64_t i=0;i<width;++i) {
            float value=(float)((int)(i%31)-15)/8;
            if (i/32==0) value=0;else if (i/32==1) value=ldexpf(value,-10);else if (i/32==2) value*=128;
            x[i]=o_bf16(value);
        }
        memset(want,0,sizeof(want));o_packed(x,width,rep,want,expected);
        static const char *ids[3]={"window_fp8_group32","compressed_fp4_group16_e4m3","index_fp4_group32_ue8m0"};
        q_result *r=q_new(q,ids[rep-1],"graph_state","gpu",1,width,0,width);
        if (!r) return false;
        r->warmup=q->request.warmup;
        q_part p={x,(size_t)width*4};if (!q_hash(q,&p,1,r->input_digest)) return false;
        for (unsigned pass=0;pass<q->request.warmup+q->request.repeats;++pass) {
            uint64_t start=q_clock();
            if (!q_alive(q) || !ria_graph_cuda_upload(g,ria_graph_cuda_buffer(g,RIA_G_LATENT),x,width,&q->error) ||
                !ria_graph_cuda_pack(g,ria_graph_cuda_buffer(g,RIA_G_LATENT),width,rep,got,&q->error) ||
                !ria_graph_cuda_unpack(g,got,1,width,rep,ria_graph_cuda_buffer(g,RIA_G_KV),&q->error) ||
                !ria_graph_cuda_download(g,ria_graph_cuda_buffer(g,RIA_G_KV),actual,width,&q->error)) return false;
            if (pass>=q->request.warmup) {
                uint64_t elapsed=q_clock()-start;uint8_t digest[32];q_part current[2]={{got,(size_t)bytes},{actual,(size_t)width*4}};
                if (!q_hash(q,current,2,digest)) return false;
                if (r->samples) { r->repeat_checked=true;if (memcmp(r->actual_digest,digest,32)) r->repeat_identical=false; }
                memcpy(r->actual_digest,digest,32);r->times[r->samples++]=elapsed;
            }
        }
        if (!q_measure(q,r,expected,actual,width)) return false;
        r->requires_exact=true;r->exact=r->exact && memcmp(want,got,(size_t)bytes)==0;
        q_part wp={want,(size_t)bytes},gp={got,(size_t)bytes};
        if (!q_hash(q,&wp,1,r->oracle_digest) || !q_hash(q,&gp,1,r->actual_digest)) return false;
    }
    return true;
}
static bool q_mhc_case(q_context *q,ria_graph_cuda *g,float *arena) {
    float *h=arena,*expected=h+20480,*actual=expected+20480,*x=actual+20480;
    float mix[24],scale[3]={.25f,.5f,.125f},base[24],wp[4],wo[4],wc[16],ap[4],ao[4],ac[16];
    for (unsigned i=0;i<20480;++i) h[i]=o_bf16((float)((int)(i%29)-14)/8);
    for (unsigned i=0;i<5120;++i) x[i]=o_bf16((float)((int)(i%17)-8)/4);
    for (unsigned i=0;i<24;++i) { mix[i]=(float)((int)i-12)/8;base[i]=(float)((int)(i%7)-3)/4; }
    o_mhc(h,mix,scale,base,wp,wo,wc);
    for (unsigned stream=0;stream<4;++stream) for (unsigned k=0;k<5120;++k) {
        float sum=0;for (unsigned j=0;j<4;++j) sum+=wc[j*4+stream]*h[j*5120+k];
        expected[stream*5120+k]=o_bf16(wo[stream]*x[k]+sum);
    }
    ria_tensor st=q_tensor(scale,3),bt=q_tensor(base,24);
    q_result *r=q_new(q,"mhc_flatten_sinkhorn20_post_orientation","graph_state","gpu",4,5120,24,5120);
    q_part p[4]={{h,20480*4},{mix,sizeof(mix)},{scale,sizeof(scale)},{base,sizeof(base)}};
    if (!r || !q_hash(q,p,4,r->input_digest)) return false;
    for (unsigned pass=0;pass<q->request.warmup+q->request.repeats;++pass) {
        uint64_t start=q_clock();
        if (!q_alive(q) || !ria_graph_cuda_upload(g,ria_graph_cuda_buffer(g,RIA_G_H),h,20480,&q->error) ||
            !ria_graph_cuda_upload(g,ria_graph_cuda_buffer(g,RIA_G_MIX),mix,24,&q->error) ||
            !ria_graph_cuda_mhc(g,ria_graph_cuda_buffer(g,RIA_G_H),ria_graph_cuda_buffer(g,RIA_G_MIX),&st,&bt,
                ria_graph_cuda_buffer(g,RIA_G_ATT_PRE),ria_graph_cuda_buffer(g,RIA_G_ATT_POST),ria_graph_cuda_buffer(g,RIA_G_ATT_COMB),&q->error) ||
            !ria_graph_cuda_download(g,ria_graph_cuda_buffer(g,RIA_G_ATT_PRE),ap,4,&q->error) ||
            !ria_graph_cuda_download(g,ria_graph_cuda_buffer(g,RIA_G_ATT_POST),ao,4,&q->error) ||
            !ria_graph_cuda_download(g,ria_graph_cuda_buffer(g,RIA_G_ATT_COMB),ac,16,&q->error) ||
            !ria_graph_cuda_upload(g,ria_graph_cuda_buffer(g,RIA_G_INPUT),x,5120,&q->error) ||
            !ria_graph_cuda_post(g,ria_graph_cuda_buffer(g,RIA_G_INPUT),ria_graph_cuda_buffer(g,RIA_G_H),
                ria_graph_cuda_buffer(g,RIA_G_ATT_POST),ria_graph_cuda_buffer(g,RIA_G_ATT_COMB),ria_graph_cuda_buffer(g,RIA_G_RESIDUAL),&q->error) ||
            !ria_graph_cuda_download(g,ria_graph_cuda_buffer(g,RIA_G_RESIDUAL),actual,20480,&q->error)) return false;
        if (pass>=q->request.warmup) r->times[r->samples++]=q_clock()-start;
    }
    if (!q_measure(q,r,expected,actual,20480)) return false;
    q_result *small=q_new(q,"mhc_pre_post_comb_fp32","graph_state","gpu",1,20480,0,24);
    float a[24],b[24];memcpy(a,wp,sizeof(wp));memcpy(a+4,wo,sizeof(wo));memcpy(a+8,wc,sizeof(wc));
    memcpy(b,ap,sizeof(ap));memcpy(b+4,ao,sizeof(ao));memcpy(b+8,ac,sizeof(ac));
    if (!small) return false;
    small->timing_group=r->timing_group;small->samples=r->samples;
    memcpy(small->times,r->times,sizeof(small->times));
    memcpy(small->input_digest,r->input_digest,32);return q_measure(q,small,a,b,24);
}
static bool q_pool_rope_cases(q_context *q,ria_graph_cuda *g,float *arena) {
    float *kv=arena,*scores=arena+1024,*expected=arena+2048,*actual=arena+2560;
    for (unsigned i=0;i<1024;++i) { kv[i]=(float)((int)(i%53)-26)/16;scores[i]=(float)((int)(i%23)-11)/8; }
    for (unsigned k=0;k<512;++k) {
        float peak=fmaxf(scores[k],scores[k+512]),a=expf(scores[k]-peak),b=expf(scores[k+512]-peak);
        expected[k]=o_bf16(kv[k]*(a/(a+b))+kv[k+512]*(b/(a+b)));
    }
    q_result *r=q_new(q,"compressor_featurewise_ratio2_fp32_to_bf16","graph_state","gpu",2,512,0,512);
    q_part parts[2]={{kv,4096},{scores,4096}};if (!r || !q_hash(q,parts,2,r->input_digest)) return false;
    uint64_t start=q_clock();
    if (!q_alive(q) || !ria_graph_cuda_upload(g,ria_graph_cuda_buffer(g,RIA_G_POOL_KV),kv,1024,&q->error) ||
        !ria_graph_cuda_upload(g,ria_graph_cuda_buffer(g,RIA_G_POOL_GATE),scores,1024,&q->error) ||
        !ria_graph_cuda_pool(g,ria_graph_cuda_buffer(g,RIA_G_POOL_KV),ria_graph_cuda_buffer(g,RIA_G_POOL_GATE),2,ria_graph_cuda_buffer(g,RIA_G_LATENT),&q->error) ||
        !ria_graph_cuda_download(g,ria_graph_cuda_buffer(g,RIA_G_LATENT),actual,512,&q->error)) return false;
    r->times[r->samples++]=q_clock()-start;if (!q_measure(q,r,expected,actual,512)) return false;
    for (unsigned ratio=0;ratio<=2;ratio+=2) for (unsigned inverse=0;inverse<2;++inverse) {
        uint64_t position=257;
        for (unsigned i=0;i<512;++i) { kv[i]=o_bf16((float)((int)(i%29)-14)/8);expected[i]=kv[i]; }
        for (unsigned pair=0;pair<32;++pair) {
            float frequency=1/powf(ratio ? 160000.0f : 10000.0f,(float)(2*pair)/64);
            if (ratio) {
                double pi=3.14159265358979323846;
                int low=(int)floor(64*log(65536/(32*2*pi))/(2*log(160000))),high=(int)ceil(64*log(65536/(2*pi))/(2*log(160000)));
                float ramp=fminf(1,fmaxf(0,((float)pair-(float)low)/(float)(high-low)));
                frequency=frequency/16*ramp+frequency*(1-ramp);
            }
            float angle=(float)position*frequency,s=sinf(angle)*(inverse ? -1 : 1),c=cosf(angle),a=kv[448+2*pair],b=kv[449+2*pair];
            expected[448+2*pair]=o_bf16(a*c-b*s);expected[449+2*pair]=o_bf16(a*s+b*c);
        }
        static const char *ids[4]={"rope_window_forward","rope_window_inverse","rope_yarn_forward","rope_yarn_inverse"};
        r=q_new(q,ids[ratio+inverse],"graph_state","gpu",1,512,position,512);q_part p={kv,2048};
        if (!r || !q_hash(q,&p,1,r->input_digest)) return false;
        start=q_clock();
        if (!q_alive(q) || !ria_graph_cuda_upload(g,ria_graph_cuda_buffer(g,RIA_G_LATENT),kv,512,&q->error) ||
            !ria_graph_cuda_rope(g,ria_graph_cuda_buffer(g,RIA_G_LATENT),1,512,position,ratio,inverse!=0,&q->error) ||
            !ria_graph_cuda_download(g,ria_graph_cuda_buffer(g,RIA_G_LATENT),actual,512,&q->error)) return false;
        r->times[r->samples++]=q_clock()-start;if (!q_measure(q,r,expected,actual,512)) return false;
    }
    return true;
}
static bool q_attention_cases(q_context *q,ria_graph_cuda *g,float *arena) {
    float *query=arena,*kv=query+32768,*expected=kv+130*512,*actual=expected+32768,sinks[64];
    for (unsigned i=0;i<32768;++i) query[i]=i%512<4 ? o_bf16((float)((int)(i%11)-5)/64) : 0;
    for (unsigned i=0;i<130*512;++i) kv[i]=o_bf16((float)((int)(i%43)-21)/16);
    for (unsigned i=0;i<64;++i) sinks[i]=(float)((int)(i%7)-3)/4;
    ria_tensor sink=q_tensor(sinks,64);
    for (unsigned mode=0;mode<2;++mode) {
        unsigned count=mode ? 129 : 65;
        if (!mode) memset(query,0,32768*4);
        o_attention(query,kv,count,sinks,expected);
        q_result *r=q_new(q,mode ? "attention_ragged129_online_bf16_probabilities" : "attention_zero_query_causal65_sink", "graph_state","gpu",64,512,count,512);
        q_part p[3]={{query,32768*4},{kv,(size_t)count*512*4},{sinks,sizeof(sinks)}};
        if (!r || !q_hash(q,p,3,r->input_digest)) return false;
        uint64_t start=q_clock();
        if (!q_alive(q) || !ria_graph_cuda_upload(g,ria_graph_cuda_buffer(g,RIA_G_Q),query,32768,&q->error) ||
            !ria_graph_cuda_upload(g,ria_graph_cuda_buffer(g,RIA_G_ATTN_KV),kv,(uint64_t)(count+1)*512,&q->error) ||
            !ria_graph_cuda_attention(g,ria_graph_cuda_buffer(g,RIA_G_Q),ria_graph_cuda_buffer(g,RIA_G_ATTN_KV),count,&sink,ria_graph_cuda_buffer(g,RIA_G_ATTN_OUT),&q->error) ||
            !ria_graph_cuda_download(g,ria_graph_cuda_buffer(g,RIA_G_ATTN_OUT),actual,32768,&q->error)) return false;
        r->times[r->samples++]=q_clock()-start;if (!q_measure(q,r,expected,actual,32768)) return false;
        /* The extra materialized key is outside the authoritative causal
         * extent. Changing it must leave all returned bits unchanged. */
        float future[512];for (unsigned i=0;i<512;++i) future[i]=1024;
        if (!ria_graph_cuda_upload(g,ria_graph_cuda_buffer(g,RIA_G_ATTN_KV)+(uint64_t)count*512,future,512,&q->error) ||
            !ria_graph_cuda_attention(g,ria_graph_cuda_buffer(g,RIA_G_Q),ria_graph_cuda_buffer(g,RIA_G_ATTN_KV),count,&sink,ria_graph_cuda_buffer(g,RIA_G_ATTN_OUT),&q->error) ||
            !ria_graph_cuda_download(g,ria_graph_cuda_buffer(g,RIA_G_ATTN_OUT),expected,32768,&q->error)) return false;
        r->repeat_checked=true;r->repeat_identical=memcmp(expected,actual,32768*4)==0;
        for (unsigned i=0;i<32768;++i) query[i]=i%512<4 ? o_bf16((float)((int)(i%11)-5)/64) : 0;
    }
    return true;
}
typedef struct { float score;uint32_t id; } q_rank;
static int q_rank_compare(const void *a,const void *b) {
    const q_rank *x=a,*y=b;if (x->score>y->score) return -1;if (x->score<y->score) return 1;return x->id<y->id ? -1 : x->id>y->id;
}
static int q_u32_compare(const void *a,const void *b) { uint32_t x=*(const uint32_t *)a,y=*(const uint32_t *)b;return x<y ? -1 : x>y; }
static bool q_index_cases(q_context *q,ria_graph_cuda *g,float *arena) {
    const unsigned count=16409,tile=256,blocks=(count+7)/8;
    float *query=arena,*keys=query+4096,*weights=keys+tile*128,*expected=weights+32,*actual=expected+count;
    uint64_t rb=(uint64_t)count*sizeof(q_rank);q_rank *rank=q_alloc(q,rb);uint8_t *mask=q_alloc(q,blocks);
    bool ok=false;if (!rank || !mask) goto done;
    memset(query,0,4096*4);for (unsigned head=0;head<32;++head) { query[head*128]=1;weights[head]=1; }
    for (unsigned mode=0;mode<2;++mode) {
        uint64_t start=q_clock();
        if (!ria_graph_cuda_upload(g,ria_graph_cuda_buffer(g,RIA_G_INDEX_Q),query,4096,&q->error) ||
            !ria_graph_cuda_upload(g,ria_graph_cuda_buffer(g,RIA_G_INDEX_WEIGHTS),weights,32,&q->error)) goto done;
        for (unsigned first=0;first<count;first+=tile) {
            unsigned rows=count-first<tile ? count-first : tile;memset(keys,0,(size_t)rows*128*4);
            for (unsigned row=0;row<rows;++row) {
                float value=(float)((first+row)%200)/16;
                if (mode) value=(float)(199-(first+row)%200)/16;
                keys[row*128]=value;expected[first+row]=o_bf16(value*.5f);
            }
            if (!q_alive(q) || !ria_graph_cuda_upload(g,ria_graph_cuda_buffer(g,RIA_G_INDEX_TILE),keys,(uint64_t)rows*128,&q->error) ||
                !ria_graph_cuda_index_scores(g,ria_graph_cuda_buffer(g,RIA_G_INDEX_Q),ria_graph_cuda_buffer(g,RIA_G_INDEX_WEIGHTS),
                    ria_graph_cuda_buffer(g,RIA_G_INDEX_TILE),first,rows,&q->error)) goto done;
        }
        q_result *r=q_new(q,mode ? "index_reindex_closed_form_scores" : "index_full_closed_form_scores","graph_state","gpu",count,128,32,1);
        q_part p[3]={{query,4096*4},{weights,32*4},{expected,(size_t)count*4}};
        if (!r || !q_hash(q,p,3,r->input_digest) || !ria_graph_cuda_index_download(g,0,count,actual,&q->error) || !q_measure(q,r,expected,actual,count)) goto done;
        r->times[r->samples++]=q_clock()-start;
        if (!mode) {
            for (unsigned block=0;block<blocks;++block) {
                float peak=-INFINITY;for (unsigned i=block*8;i<count && i<block*8+8;++i) peak=fmaxf(peak,expected[i]);
                rank[block].id=block;rank[block].score=block==(count-1)/8 ? INFINITY : peak;
            }
            qsort(rank,blocks,sizeof(*rank),q_rank_compare);for (unsigned i=0;i<2048;++i) mask[rank[i].id]=1;
        }
        for (unsigned i=0;i<count;++i) { rank[i].id=i;rank[i].score=mode && !mask[i/8] ? -INFINITY : expected[i]; }
        qsort(rank,count,sizeof(*rank),q_rank_compare);uint32_t want[512],got[512],chosen=0;
        for (unsigned i=0;i<512;++i) want[i]=rank[i].id;
        qsort(want,512,sizeof(*want),q_u32_compare);
        start=q_clock();
        if (!ria_graph_cuda_select(g,count,!mode,mode!=0,got,&chosen,&q->error)) goto done;
        q_result *selection=q_new(q,mode ? "csa2_reindex_reuses_source_block_mask" : "csa2_full_newest_partial_block","graph_state","gpu",1,count,2048,512);
        if (!selection) goto done;
        selection->times[selection->samples++]=q_clock()-start;
        memcpy(selection->input_digest,r->input_digest,32);
        selection->requires_exact=true;selection->elements=512;selection->exact=chosen==512 && memcmp(want,got,sizeof(want))==0;
        selection->host_bytes=q->host_current;selection->device_bytes=q->device_current;selection->pinned_bytes=q->pinned_current;
        q_part w={want,sizeof(want)},a={got,sizeof(got)};
        if (!q_hash(q,&w,1,selection->oracle_digest) || !q_hash(q,&a,1,selection->actual_digest)) goto done;
    }
    ok=true;
done:q_free(q,rank,rb);q_free(q,mask,blocks);return ok;
}
static bool q_engram_cases(q_context *q,ria_graph_cuda *g,float *arena) {
    float *expected=arena,*actual=expected+20480,*h=actual+20480,*kv=h+20480,*qw=kv+25600,*kw=qw+20480;
    uint8_t packed[24*264];
    for (unsigned row=0;row<24;++row) {
        for (unsigned group=0;group<8;++group) packed[row*264+256+group]=(uint8_t)(118+(row+group)%18);
        for (unsigned k=0;k<256;++k) {
            uint8_t code=(uint8_t)(((row*256+k)%127)|((row+k)%2 ? 128 : 0));packed[row*264+k]=code;
            expected[row*256+k]=o_bf16(o_f8(code)*ldexpf(1,(int)packed[row*264+256+k/32]-127));
        }
    }
    q_result *r=q_new(q,"engram_e4m3_row256_scale32_decode","graph_state","gpu",24,256,8,256);
    q_part row_bytes={packed,sizeof(packed)};if (!r || !q_hash(q,&row_bytes,1,r->input_digest)) return false;
    uint64_t start=q_clock();
    if (!q_alive(q) || !ria_graph_cuda_engram(g,packed,ria_graph_cuda_buffer(g,RIA_G_ENGRAM_INPUT),&q->error) ||
        !ria_graph_cuda_download(g,ria_graph_cuda_buffer(g,RIA_G_ENGRAM_INPUT),actual,6144,&q->error)) return false;
    r->times[r->samples++]=q_clock()-start;r->requires_exact=true;
    if (!q_measure(q,r,expected,actual,6144)) return false;
    for (unsigned stream=0;stream<4;++stream) for (unsigned k=0;k<5120;++k) {
        unsigned i=stream*5120+k;float pattern=(float)((int)(k%29)-14)/8;
        h[i]=o_bf16((float)(stream+1)*pattern);kv[i]=o_bf16((stream%2 ? -1.0f : 1.0f)*(float)(4-stream)*pattern);
        qw[i]=.125f;kw[i]=.0625f;
    }
    for (unsigned k=0;k<5120;++k) kv[20480+k]=o_bf16((float)((int)(k%17)-8)/16);
    for (unsigned stream=0;stream<4;++stream) {
        float hs=0,ks=0,dot=0;
        for (unsigned k=0;k<5120;++k) {
            unsigned i=stream*5120+k;hs=fmaf(h[i],h[i],hs);ks=fmaf(kv[i],kv[i],ks);
            dot+=(h[i]*(qw[i]*kw[i]))*kv[i];
        }
        dot=(dot*((1/sqrtf(hs/5120+1e-20f))*(1/sqrtf(ks/5120+1e-20f))))*0.013975424859373685f;
        float gate=1/(1+expf(-copysignf(sqrtf(fmaxf(fabsf(dot),1e-6f)),dot)));
        for (unsigned k=0;k<5120;++k) expected[stream*5120+k]=o_bf16(h[stream*5120+k]+gate*kv[20480+k]);
    }
    ria_tensor qt=q_tensor(qw,20480),kt=q_tensor(kw,20480);
    r=q_new(q,"engram_per_stream_normalized_signed_sqrt_fusion","graph_state","gpu",4,5120,25600,5120);
    q_part parts[4]={{h,20480*4},{kv,25600*4},{qw,20480*4},{kw,20480*4}};
    if (!r || !q_hash(q,parts,4,r->input_digest)) return false;
    start=q_clock();
    if (!q_alive(q) || !ria_graph_cuda_upload(g,ria_graph_cuda_buffer(g,RIA_G_H),h,20480,&q->error) ||
        !ria_graph_cuda_upload(g,ria_graph_cuda_buffer(g,RIA_G_ENGRAM_KV),kv,25600,&q->error) ||
        !ria_graph_cuda_engram_fuse(g,ria_graph_cuda_buffer(g,RIA_G_H),ria_graph_cuda_buffer(g,RIA_G_ENGRAM_KV),&qt,&kt,true,&q->error) ||
        !ria_graph_cuda_download(g,ria_graph_cuda_buffer(g,RIA_G_H),actual,20480,&q->error)) return false;
    r->times[r->samples++]=q_clock()-start;if (!q_measure(q,r,expected,actual,20480)) return false;
    r=q_new(q,"engram_image_mask_passes_through_exact","graph_state","gpu",4,5120,25600,5120);
    if (!r || !q_hash(q,parts,4,r->input_digest)) return false;
    start=q_clock();
    if (!ria_graph_cuda_upload(g,ria_graph_cuda_buffer(g,RIA_G_H),h,20480,&q->error) ||
        !ria_graph_cuda_engram_fuse(g,ria_graph_cuda_buffer(g,RIA_G_H),ria_graph_cuda_buffer(g,RIA_G_ENGRAM_KV),&qt,&kt,false,&q->error) ||
        !ria_graph_cuda_download(g,ria_graph_cuda_buffer(g,RIA_G_H),actual,20480,&q->error)) return false;
    r->times[r->samples++]=q_clock()-start;r->requires_exact=true;return q_measure(q,r,h,actual,20480);
}
static bool q_router_merge_sensitive_cases(q_context *q,ria_graph_cuda *g,float *arena) {
    float scores[384],bias[384],transformed[384],coefficients[6],want_coeff[6];uint16_t ids[6],want_ids[6];bool chosen[384]={false};
    for (unsigned i=0;i<384;++i) { scores[i]=(float)((int)(i%31)-15)/8;bias[i]=i<6 ? 4 : 0;transformed[i]=sqrtf(log1pf(expf(scores[i]))); }
    float total=0;
    for (unsigned slot=0;slot<6;++slot) {
        unsigned winner=0;float best=-INFINITY;
        for (unsigned i=0;i<384;++i) if (!chosen[i] && transformed[i]+bias[i]>best) { best=transformed[i]+bias[i];winner=i; }
        chosen[winner]=true;want_ids[slot]=(uint16_t)winner;want_coeff[slot]=transformed[winner];total+=transformed[winner];
    }
    for (unsigned slot=0;slot<6;++slot) want_coeff[slot]=want_coeff[slot]/(total+1e-20f)*1.5f;
    ria_tensor bt=q_tensor(bias,384);q_result *r=q_new(q,"router_bias_selects_unbiased_coefficients","graph_state","gpu",1,384,6,6);
    q_part p[2]={{scores,sizeof(scores)},{bias,sizeof(bias)}};if (!r || !q_hash(q,p,2,r->input_digest)) return false;
    uint64_t start=q_clock();
    if (!q_alive(q) || !ria_graph_cuda_upload(g,ria_graph_cuda_buffer(g,RIA_G_ROUTER),scores,384,&q->error) ||
        !ria_graph_cuda_route(g,ria_graph_cuda_buffer(g,RIA_G_ROUTER),&bt,ids,coefficients,&q->error)) return false;
    r->times[r->samples++]=q_clock()-start;
    if (!q_measure(q,r,want_coeff,coefficients,6)) return false;
    /* Selection IDs are a separate exact requirement, independent of the
     * numeric coefficient comparison against a preregistered tolerance. */
    q_result *selected=q_new(q,"router_original_selected_ids","graph_state","gpu",1,384,0,6);
    if (!selected) return false;
    memcpy(selected->input_digest,r->input_digest,32);
    selected->requires_exact=true;selected->exact=memcmp(want_ids,ids,sizeof(ids))==0;selected->elements=6;
    selected->timing_group=r->timing_group;selected->samples=1;selected->times[0]=r->times[0];
    selected->host_bytes=q->host_current;selected->device_bytes=q->device_current;selected->pinned_bytes=q->pinned_current;
    q_part wi={want_ids,sizeof(want_ids)},ai={ids,sizeof(ids)};
    if (!q_hash(q,&wi,1,selected->oracle_digest) || !q_hash(q,&ai,1,selected->actual_digest)) return false;
    float *contributions=arena,*shared=arena+6*5120,*expected=shared+5120,*actual=expected+5120;
    const uint16_t order_ids[6]={383,1,30,7,0,42};const unsigned order[6]={4,1,3,2,5,0};
    const float values[6]={-.5f,1,-16777216,.5f,16777216,1.5f};
    for (unsigned k=0;k<5120;++k) {
        shared[k]=.25f;float sum=0;
        for (unsigned slot=0;slot<6;++slot) contributions[slot*5120+k]=values[slot]*(k%2 ? -1.0f : 1.0f);
        for (unsigned i=0;i<6;++i) sum+=contributions[order[i]*5120+k];
        expected[k]=o_bf16(sum+shared[k]);
    }
    r=q_new(q,"expert_original_id_order_shared_once_bf16","graph_state","gpu",6,5120,0,5120);
    q_part m[3]={{order_ids,sizeof(order_ids)},{contributions,6*5120*4},{shared,5120*4}};
    if (!r || !q_hash(q,m,3,r->input_digest)) return false;
    start=q_clock();
    if (!ria_graph_cuda_upload(g,ria_graph_cuda_buffer(g,RIA_G_SHARED),shared,5120,&q->error) ||
        !ria_graph_cuda_merge(g,order_ids,contributions,ria_graph_cuda_buffer(g,RIA_G_SHARED),ria_graph_cuda_buffer(g,RIA_G_OUTPUT),&q->error) ||
        !ria_graph_cuda_download(g,ria_graph_cuda_buffer(g,RIA_G_OUTPUT),actual,5120,&q->error)) return false;
    r->times[r->samples++]=q_clock()-start;r->requires_exact=true;if (!q_measure(q,r,expected,actual,5120)) return false;
    float *x=arena,*weights=x+65,*want=weights+65*19,*got=want+19;
    for (unsigned i=0;i<65;++i) x[i]=1+(float)i/1024;
    memset(weights,0,65*19*4);for (unsigned i=0;i<19;++i) { weights[i*65+i]=1;want[i]=x[i]; }
    ria_expert_matrix matrix;memset(&matrix,0,sizeof(matrix));matrix.profile=RIA_EXPERT_F32;
    matrix.in_features=65;matrix.out_features=19;matrix.values=(const uint8_t *)weights;matrix.values_bytes=65*19*4;matrix.value_row_stride=65*4;
    matrix.weight_global_scale=1;matrix.activation_global_scale=1;
    r=q_new(q,"sensitive_f32_projection_preserves_unrounded_input","graph_state","gpu",1,65,0,19);
    q_part f[2]={{x,65*4},{weights,65*19*4}};
    if (!r || !q_hash(q,f,2,r->input_digest)) return false;
    start=q_clock();
    if (!ria_graph_cuda_upload(g,ria_graph_cuda_buffer(g,RIA_G_INPUT),x,65,&q->error) ||
        !ria_graph_cuda_project(g,&matrix,ria_graph_cuda_buffer(g,RIA_G_INPUT),ria_graph_cuda_buffer(g,RIA_G_OUTPUT),false,&q->error) ||
        !ria_graph_cuda_download(g,ria_graph_cuda_buffer(g,RIA_G_OUTPUT),got,19,&q->error)) return false;
    r->times[r->samples++]=q_clock()-start;r->requires_exact=true;return q_measure(q,r,want,got,19);
}
static bool q_norm_case(q_context *q,ria_graph_cuda *g,float *arena) {
    float *x=arena,*weights=x+10240,*expected=weights+5120,*actual=expected+10240;
    float square=0;
    for (unsigned k=0;k<5120;++k) {
        x[k]=0;x[5120+k]=o_bf16((float)((int)(k%29)-14)/8);weights[k]=1;
        square=fmaf(x[5120+k],x[5120+k],square);
    }
    float inverse=1/sqrtf(square/5120+1e-20f);
    for (unsigned k=0;k<5120;++k) { expected[k]=0;expected[5120+k]=o_bf16(x[5120+k]*inverse); }
    ria_tensor weight=q_tensor(weights,5120);
    q_result *r=q_new(q,"source_rmsnorm5120_zero_and_nonzero_bf16","graph_state","gpu",2,5120,0,5120);
    q_part parts[2]={{x,10240*4},{weights,5120*4}};
    if (!r || !q_hash(q,parts,2,r->input_digest)) return false;
    uint64_t start=q_clock();
    if (!q_alive(q) || !ria_graph_cuda_upload(g,ria_graph_cuda_buffer(g,RIA_G_H),x,10240,&q->error) ||
        !ria_graph_cuda_norm(g,ria_graph_cuda_buffer(g,RIA_G_H),2,5120,&weight,false,&q->error) ||
        !ria_graph_cuda_download(g,ria_graph_cuda_buffer(g,RIA_G_H),actual,10240,&q->error)) return false;
    r->times[r->samples++]=q_clock()-start;return q_measure(q,r,expected,actual,10240);
}
static bool q_graph(q_context *q) {
    ria_graph_options o;memset(&o,0,sizeof(o));o.device=0;o.gpu_uuid=q->request.gpu_uuid;o.max_tokens=16409;
    o.projection_tile_rows=32;o.state_tile_rows=256;o.device_budget=q->request.device_budget;o.pinned_budget=q->request.pinned_budget;
    uint64_t pinned=0,metadata=ria_graph_cuda_metadata_bytes(),arena_bytes=UINT64_C(200000)*4;
    ria_graph_cuda *g=NULL;float *arena=NULL;bool ok=false;
    if (!ria_expert_cuda_pinned_required_bytes(32768,32768,129280,1,32,&pinned,&q->error) || pinned>q->request.pinned_budget ||
        !q_reserve(q,metadata+pinned)) return false;
    uint64_t startup=q_clock();
    if (!ria_graph_cuda_create(&o,&g,&q->error)) goto done;
    q->graph_startup_ns=q_clock()-startup;
    q->device_current=ria_graph_cuda_device_bytes(g);q->pinned_current=pinned;
    if (q->device_current>q->device_peak) q->device_peak=q->device_current;
    if (pinned>q->pinned_peak) q->pinned_peak=pinned;
    arena=q_alloc(q,arena_bytes);if (!arena) goto done;
    ok=q_packed_cases(q,g,arena) && q_mhc_case(q,g,arena) && q_pool_rope_cases(q,g,arena) &&
        q_attention_cases(q,g,arena) && q_index_cases(q,g,arena) && q_engram_cases(q,g,arena) &&
        q_router_merge_sensitive_cases(q,g,arena) && q_norm_case(q,g,arena) && q_alive(q);
done:{ria_error cleanup;if (!ria_graph_cuda_destroy(g,&cleanup) && ok) { q->error=cleanup;ok=false; }
      q->host_current-=metadata+pinned;q->device_current=0;q->pinned_current=0;}
    q_free(q,arena,arena_bytes);return ok;
}
#endif

static const char *const q_identity_names[7]={"logical_model_digest","source_lock_digest","environment_digest","build_digest","policy_digest","operator_contract_digest","preregistration_digest"};
static bool q_request_parse(const ria_json_doc *d,q_request *r,ria_error *e) {
    static const char *const names[]={"schema_revision","kind","profile","executor","runner_executor","role","gpu_uuid","expert_shape","logical_model_digest","source_lock_digest",
        "environment_digest","build_digest","policy_digest","operator_contract_digest","preregistration_digest","host_budget","device_budget","pinned_budget",
        "deadline_ms","repeats","warmup","fixture_seed","relative_floor"};
    memset(r,0,sizeof(*r));uint64_t revision,repeats,warmup;size_t length;const char *kind;
    if (!ria_json_fields(d,0,names,sizeof(names)/sizeof(*names),names,sizeof(names)/sizeof(*names),e) ||
        !ria_json_u64(d,ria_json_get(d,0,"schema_revision"),false,&revision,e) || revision!=1 ||
        !ria_json_string(d,ria_json_get(d,0,"kind"),&kind,&length,e) || strlen(kind)!=length || strcmp(kind,"native_fixture_request") ||
        !ria_json_string(d,ria_json_get(d,0,"profile"),&r->profile_name,&length,e) || strlen(r->profile_name)!=length ||
        !ria_json_string(d,ria_json_get(d,0,"executor"),&r->executor,&length,e) || strlen(r->executor)!=length ||
        !ria_json_string(d,ria_json_get(d,0,"runner_executor"),&r->runner_executor,&length,e) || strlen(r->runner_executor)!=length ||
        !ria_json_string(d,ria_json_get(d,0,"role"),&r->role,&length,e) || strlen(r->role)!=length ||
        !ria_json_string(d,ria_json_get(d,0,"expert_shape"),&r->shape,&length,e) || strlen(r->shape)!=length)
        return ria_fail(e,RIA_INVALID_REQUEST,"invalid fixture revision/kind/profile/executor/shape");
    r->profile=!strcmp(r->profile_name,"bf16") ? RIA_EXPERT_BF16 : !strcmp(r->profile_name,"fp8") ? RIA_EXPERT_FP8 : !strcmp(r->profile_name,"nvfp4") ? RIA_EXPERT_NVFP4 : 0;
    if (!r->profile || (strcmp(r->executor,"cpu") && strcmp(r->executor,"cuda")) ||
        (strcmp(r->runner_executor,"cpu") && strcmp(r->runner_executor,"cuda")) ||
        (strcmp(r->role,"client") && strcmp(r->role,"server")) ||
        (!strcmp(r->role,"client") ? strcmp(r->runner_executor,"cuda") : strcmp(r->runner_executor,r->executor)) ||
        (strcmp(r->shape,"ragged") && strcmp(r->shape,"target")))
        return ria_fail(e,RIA_INVALID_REQUEST,"unknown fixture profile/executor/shape");
    for (unsigned i=0;i<7;++i) {
        uint8_t digest[32];uint32_t index=ria_json_get(d,0,q_identity_names[i]);
        if (!ria_json_digest_field(d,index,digest,e) || !ria_json_string(d,index,&r->identity[i],&length,e)) return false;
        for (unsigned j=0;j<64;++j) if (!((r->identity[i][j]>='0' && r->identity[i][j]<='9') || (r->identity[i][j]>='a' && r->identity[i][j]<='f')))
            return ria_fail(e,RIA_INVALID_REQUEST,"fixture identities must be canonical lowercase SHA256");
    }
    const ria_json_node *uuid=ria_json_at(d,ria_json_get(d,0,"gpu_uuid"));
    if (!uuid) return ria_fail(e,RIA_INVALID_REQUEST,"missing fixture device identity");
    if (!strcmp(r->runner_executor,"cuda")) {
        if (!ria_json_string(d,ria_json_get(d,0,"gpu_uuid"),&r->gpu_uuid,&length,e) || length!=40 || strncmp(r->gpu_uuid,"GPU-",4))
            return ria_fail(e,RIA_INVALID_REQUEST,"CUDA fixture needs locked canonical UUID");
        for (unsigned i=4;i<40;++i) {
            char c=r->gpu_uuid[i];bool dash=i==12 || i==17 || i==22 || i==27;
            if (dash ? c!='-' : !((c>='0' && c<='9') || (c>='a' && c<='f')))
                return ria_fail(e,RIA_INVALID_REQUEST,"invalid canonical CUDA UUID");
        }
    } else if (uuid->type!=RIA_JSON_NULL) return ria_fail(e,RIA_INVALID_REQUEST,"CPU fixture gpu_uuid must be null");
    if (!ria_json_u64(d,ria_json_get(d,0,"host_budget"),true,&r->host_budget,e) ||
        !ria_json_u64(d,ria_json_get(d,0,"device_budget"),true,&r->device_budget,e) ||
        !ria_json_u64(d,ria_json_get(d,0,"pinned_budget"),true,&r->pinned_budget,e) ||
        !ria_json_u64(d,ria_json_get(d,0,"deadline_ms"),false,&r->deadline_ms,e) ||
        !ria_json_u64(d,ria_json_get(d,0,"repeats"),false,&repeats,e) ||
        !ria_json_u64(d,ria_json_get(d,0,"warmup"),false,&warmup,e) ||
        !ria_json_u64(d,ria_json_get(d,0,"fixture_seed"),true,&r->seed,e)) return false;
    const ria_json_node *floor=ria_json_at(d,ria_json_get(d,0,"relative_floor"));
    if (!floor || floor->type!=RIA_JSON_NUMBER || !isfinite(floor->number) || floor->number<=0 ||
        !r->host_budget || r->host_budget>RIA_JSON_SAFE_INTEGER || !r->deadline_ms || r->deadline_ms>600000 ||
        !repeats || repeats>Q_MAX_REPEATS || warmup>8 ||
        (!strcmp(r->runner_executor,"cpu") ? (r->device_budget || r->pinned_budget) : (!r->device_budget || !r->pinned_budget)))
        return ria_fail(e,RIA_INVALID_REQUEST,"fixture budgets/deadline/repetition/relative floor outside explicit bounds");
    r->repeats=(unsigned)repeats;r->warmup=(unsigned)warmup;r->floor=floor->number;
    return ria_json_sha256(d,false,r->request_digest,e);
}
static bool q_oracle_self_test(void) {
    for (unsigned i=0;i<127;++i) if (o_encode(o_f8(i),true)!=i) return false;
    for (unsigned i=0;i<16;++i) if (o_encode(o_f4(i),false)!=i) return false;
    if (o_bf16(1.00390625f)!=1 || o_bf16(1.01171875f)!=1.015625f || !signbit(o_bf16(-0.0f))) return false;
    float x[65],raw[65],scale[5];for (unsigned i=0;i<65;++i) x[i]=i==64 ? 96 : 6;
    o_quant(x,65,RIA_EXPERT_NVFP4,1,raw,scale);
    if (scale[0]!=1 || scale[4]!=16 || raw[0]!=6 || raw[64]!=6) return false;
    o_quant(x,65,RIA_EXPERT_FP8,1,raw,scale);
    if (scale[0]!=.015625f || scale[2]!=.25f || raw[0]!=384 || raw[64]!=384) return false;
    float values[512],decoded[512];uint8_t packed[528];
    for (unsigned i=0;i<512;++i) values[i]=6;
    for (unsigned rep=1;rep<=3;++rep) {
        o_packed(values,512,rep,packed,decoded);
        for (unsigned i=0;i<512;++i) if (decoded[i]!=6) return false;
        if (rep!=1) for (unsigned i=0;i<256;++i) if (packed[i]!=119) return false;
    }
    float *h=calloc(20480,sizeof(float)),*query=calloc(32768,sizeof(float)),*kv=calloc(65*512,sizeof(float)),*out=calloc(32768,sizeof(float));
    if (!h || !query || !kv || !out) { free(h);free(query);free(kv);free(out);return false; }
    float mix[24]={0},base[24]={0},s[3]={1,1,1},pre[4],post[4],comb[16],sink[64]={0};bool ok=true;
    for (unsigned i=0;i<20480;++i) h[i]=1;
    o_mhc(h,mix,s,base,pre,post,comb);
    for (unsigned i=0;i<4;++i) if (pre[i]!=(.5f+1e-6f) || post[i]!=1) ok=false;
    for (unsigned i=0;i<16;++i) if (comb[i]!=comb[0] || !(comb[i]>0 && comb[i]<.25f)) ok=false;
    for (unsigned i=0;i<65*512;++i) kv[i]=2;
    o_attention(query,kv,65,sink,out);
    for (unsigned i=0;i<32768;++i) if (out[i]!=o_bf16(130.0f/66)) ok=false;
    free(h);free(query);free(kv);free(out);if (!ok) return false;
    return true;
}
static bool q_emit(const q_context *q) {
    char digest[65];ria_hex_encode(q->request.request_digest,32,digest);
    printf("{\"schema_revision\":1,\"kind\":\"native_fixture_measurements\",\"qualified\":false,\"qualification_scope\":\"initial_fixture\",\"fixture_algorithm\":\"ria-native-fixtures-v1\",\"request_digest\":\"%s\",\"profile\":\"%s\",\"executor\":\"%s\",",digest,q->request.profile_name,q->request.executor);
    for (unsigned i=0;i<7;++i) printf("\"%s\":\"%s\",",q_identity_names[i],q->request.identity[i]);
    printf("\"runner_executor\":\"%s\",\"role\":\"%s\",",q->request.runner_executor,q->request.role);
    if (q->request.gpu_uuid) printf("\"gpu_uuid\":\"%s\",",q->request.gpu_uuid);else printf("\"gpu_uuid\":null,");
    struct rusage usage;bool measured=getrusage(RUSAGE_SELF,&usage)==0;
    printf("\"fixture_seed\":\"%" PRIu64 "\",\"relative_floor\":%.17g,\"owned_host_peak_bytes\":\"%" PRIu64 "\",\"device_peak_bytes\":\"%" PRIu64 "\",\"pinned_peak_bytes\":\"%" PRIu64 "\",\"process_peak_rss_bytes\":",
        q->request.seed,q->request.floor,q->host_peak,q->device_peak,q->pinned_peak);
    if (measured && usage.ru_maxrss>=0) printf("\"%" PRIu64 "\"",(uint64_t)usage.ru_maxrss*1024);else printf("null");
    printf(",\"warmup\":%u,\"repeats\":%u,\"elapsed_ns\":\"%" PRIu64 "\",\"graph_startup_ns\":\"%" PRIu64 "\",\"host_accounting_scope\":\"owned_heap_metadata_and_pinned_pools; process_RSS_also_observed\",\"latency_scope\":\"synchronous_H2D_operator_D2H; selected_experts_include_all_three_projections\",\"deadline_enforcement\":\"monotonic_control_points_and_required_external_process_supervisor\",\"full_model_graph_parity\":\"unexecuted\",\"model_quality\":\"unexecuted\",\"graph_gpu_fixtures\":\"%s\",\"transfer_gpu_fixtures\":\"%s\",\"cases\":[",
        q->request.warmup,q->request.repeats,q_clock()-q->start_ns,q->graph_startup_ns,
        !strcmp(q->request.runner_executor,"cuda") && !strcmp(q->request.role,"client") ? "executed" : "unexecuted",!strcmp(q->request.runner_executor,"cuda") ? "executed" : "unexecuted");
    for (unsigned i=0;i<q->count;++i) {
        const q_result *r=&q->results[i];char input[65],oracle[65],actual[65];ria_hex_encode(r->input_digest,32,input);ria_hex_encode(r->oracle_digest,32,oracle);ria_hex_encode(r->actual_digest,32,actual);
        printf("%s{\"id\":\"%s\",\"timing_group\":\"%s\",\"warmup\":%u,\"component\":\"%s\",\"path\":\"%s\",\"rows\":%" PRIu64 ",\"input_features\":%" PRIu64 ",\"intermediate_features\":%" PRIu64 ",\"output_features\":%" PRIu64 ",\"elements\":%" PRIu64 ",\"input_digest\":\"%s\",\"oracle_digest\":\"%s\",\"actual_digest\":\"%s\",\"max_abs_error\":%.17g,\"max_relative_error\":%.17g,\"max_rms_error\":%.17g,\"requires_exact_match\":%s,\"exact_match\":%s,\"repeat_identical\":%s,\"placement_identical\":%s,\"owned_host_bytes\":\"%" PRIu64 "\",\"device_bytes\":\"%" PRIu64 "\",\"pinned_bytes\":\"%" PRIu64 "\",\"startup_ns\":\"%" PRIu64 "\",\"latency_ns\":[",
            i ? "," : "",r->id,r->timing_group,r->warmup,r->component,r->path,r->rows,r->input,r->mid,r->output,r->elements,input,oracle,actual,r->absolute,r->relative,r->rms,
            r->requires_exact ? "true" : "false",r->exact ? "true" : "false",r->repeat_checked ? (r->repeat_identical ? "true" : "false") : "null",
            r->placement_checked ? (r->placement_identical ? "true" : "false") : "null",r->host_bytes,r->device_bytes,r->pinned_bytes,r->startup_ns);
        for (unsigned j=0;j<r->samples;++j) printf("%s\"%" PRIu64 "\"",j ? "," : "",r->times[j]);
        printf("]}");
    }
    printf("]}\n");return fflush(stdout)==0 && !ferror(stdout);
}
int main(int argc,char **argv) {
#if defined(__linux__)
    /* exec resets dumpability even when the outer supervisor disabled it. */
    if (prctl(PR_SET_DUMPABLE,0,0,0,0)!=0) {
        fprintf(stderr,"ria-qualify: cannot disable process dumpability\n");
        return 2;
    }
#endif
    if (argc==2 && !strcmp(argv[1],"--oracle-self-test")) {
        if (!q_oracle_self_test()) { fprintf(stderr,"independent fixture oracle self-test failed\n");return 1; }
        puts("{\"schema_revision\":1,\"oracle_self_test\":true,\"gpu_executed\":false}");return 0;
    }
    bool validate=argc==3 && !strcmp(argv[1],"--validate-request");
    if (argc!=2 && !validate) { fprintf(stderr,"usage: ds4-ria-qualify REQUEST.json | --validate-request REQUEST.json | --oracle-self-test\n");return 2; }
    if (sizeof(float)!=4 || FLT_RADIX!=2 || FLT_MANT_DIG!=24 || fegetround()!=FE_TONEAREST) { fprintf(stderr,"fixture requires IEEE binary32 nearest-even arithmetic\n");return 1; }
    ria_json_doc doc;ria_error error={0};q_request request;
    if (!ria_json_read(argv[validate ? 2 : 1],(ria_json_limits){Q_JSON_BYTES,256,8},&doc,&error)) { fprintf(stderr,"fixture request: %s\n",error.message);return 1; }
    bool parsed=q_request_parse(&doc,&request,&error);
    if (!parsed) { fprintf(stderr,"fixture request: %s\n",error.message);ria_json_free(&doc);return 1; }
    if (validate) { char hash[65];ria_hex_encode(request.request_digest,32,hash);printf("{\"schema_revision\":1,\"request_digest\":\"%s\",\"gpu_executed\":false}\n",hash);ria_json_free(&doc);return 0; }
    uint64_t minimum=doc.allocated_bytes+sizeof(q_context);
    if (request.host_budget<minimum) { fprintf(stderr,"fixture owner/request exceeds host cap\n");ria_json_free(&doc);return 1; }
    q_context *q=calloc(1,sizeof(*q));if (!q) { fprintf(stderr,"allocate fixture owner\n");ria_json_free(&doc);return 1; }
    q->request=request;q->host_current=minimum;q->host_peak=minimum;q->start_ns=q_clock();bool ok=q->start_ns!=0;
    if (!strcmp(request.runner_executor,"cuda")) {
#ifndef RIA_QUALIFY_CPU_ONLY
        ok=ok && ria_expert_cuda_device_require(0,request.gpu_uuid,!strcmp(request.role,"client") ? "NVIDIA GeForce RTX 5090" : NULL,&q->error);
#else
        ok=ria_fail(&q->error,RIA_UNSUPPORTED,"driver-free CPU qualifier cannot execute CUDA fixtures");
#endif
    }
    if (ok) ok=q_experts(q,false) && q_experts(q,true);
#ifndef RIA_QUALIFY_CPU_ONLY
    if (ok && !strcmp(request.runner_executor,"cuda")) ok=q_transfers(q) && (strcmp(request.role,"client") || q_graph(q));
#endif
    if (ok) ok=q_alive(q) && q_emit(q);
    if (!ok) fprintf(stderr,"qualification fixture failed (%d): %s\n",q->error.code,q->error.message[0] ? q->error.message : "oracle/clock/output invariant");
    free(q);ria_json_free(&doc);return ok ? 0 : 1;
}
