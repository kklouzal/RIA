#include "expert.h"
#include "expert_cuda.h"
#include "numeric.h"
#include <float.h>
#include <fenv.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif

typedef void (*ria_dot_four)(const ria_expert_matrix *,uint64_t,const float *,const float *,float *);
typedef struct { char padding;float value; } ria_float_alignment;
struct ria_expert_cpu {
    uint64_t max_input,max_intermediate,max_output,workspace_bytes,width;
    float *allocation,*logical,*quantized,*activation_scales,*gate,*up;
    ria_dot_four dot_four;
    const char *kernel;
    bool owns_allocation;
};

static bool fail(ria_error *error,int code,const char *message) {
    if (error) { error->code=code; (void)snprintf(error->message,sizeof(error->message),"%s",message); }
    return false;
}
static bool mul(uint64_t a,uint64_t b,uint64_t *result) {
    if (b && a>UINT64_MAX/b) return false;
    *result=a*b; return true;
}
static bool span(uint64_t rows,uint64_t stride,uint64_t width,uint64_t bytes) {
    uint64_t base;
    return rows && stride>=width && mul(rows-1,stride,&base) &&
        base<=UINT64_MAX-width && base+width<=bytes && bytes<=SIZE_MAX;
}
bool ria_expert_float_span_bytes(uint64_t rows,uint64_t stride,uint64_t width,uint64_t *bytes) {
    if (!bytes || !width || !span(rows,stride,width,SIZE_MAX/sizeof(float))) return false;
    *bytes=((rows-1)*stride+width)*sizeof(float);return true;
}
static uint64_t groups(uint64_t count,uint64_t group) { return count/group+(count%group!=0); }
bool ria_expert_cuda_pinned_required_bytes(uint64_t input,uint64_t intermediate,uint64_t output,
                                         uint64_t rows,uint64_t tile_rows,uint64_t *bytes,ria_error *e) {
    if (!bytes || !input || !intermediate || !output || !rows || rows>65535 || !tile_rows || tile_rows>4096 ||
        input>UINT32_MAX || intermediate>UINT32_MAX || output>UINT32_MAX)
        return fail(e,RIA_INVALID_REQUEST,"invalid pinned expert pool bounds");
    uint64_t width=input>intermediate ? input : intermediate,maximum=0,part;
    if (!mul(tile_rows,width,&part) || !mul(part,4,&maximum) ||
        !mul(rows,width>output ? width : output,&part) || !mul(part,4,&part))
        return fail(e,RIA_RESOURCE_LIMIT,"pinned expert pool size overflow");
    if (part>maximum) maximum=part;
    long page=sysconf(_SC_PAGESIZE);
    if (page<=0 || maximum>UINT64_MAX-(uint64_t)page+1)
        return fail(e,RIA_RESOURCE_LIMIT,"pinned expert pool rounding overflow");
    maximum=(maximum+(uint64_t)page-1)/(uint64_t)page*(uint64_t)page;
    if (maximum>SIZE_MAX) return fail(e,RIA_RESOURCE_LIMIT,"pinned expert pool exceeds address range");
    *bytes=maximum;return true;
}
bool ria_expert_ranges_disjoint(const void *a,uint64_t a_bytes,const void *b,uint64_t b_bytes) {
    uintptr_t aa=(uintptr_t)a,bb=(uintptr_t)b;
    if (a_bytes>UINTPTR_MAX-aa || b_bytes>UINTPTR_MAX-bb) return false;
    return aa+a_bytes<=bb || bb+b_bytes<=aa;
}
static bool math_mode(ria_error *error) {
    if (fegetround()!=FE_TONEAREST) return fail(error,RIA_UNSUPPORTED,"expert arithmetic requires round-to-nearest ties-to-even");
#if defined(__aarch64__)
    uint64_t control; __asm__ volatile("mrs %0, fpcr" : "=r"(control));
    if (control & (UINT64_C(1)<<24)) return fail(error,RIA_UNSUPPORTED,"expert arithmetic requires FP32 denormal preservation");
#elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if (_mm_getcsr() & ((1u<<15)|(1u<<6))) return fail(error,RIA_UNSUPPORTED,"expert arithmetic requires FP32 denormal preservation");
#endif
    return true;
}
static float read_bf16(const uint8_t *bytes) {
    uint32_t value=(uint32_t)bytes[0]|((uint32_t)bytes[1]<<8);
    return ria_num_float(value<<16);
}
static float read_float(const uint8_t *bytes) {
    return ria_num_float((uint32_t)bytes[0]|((uint32_t)bytes[1]<<8)|
                         ((uint32_t)bytes[2]<<16)|((uint32_t)bytes[3]<<24));
}
static bool matrix_shape(const ria_expert_matrix *m,ria_error *error) {
    if (!m || !m->values || !m->out_features || !m->in_features)
        return fail(error,RIA_INVALID_REQUEST,"empty expert matrix");
    uint64_t width=m->in_features,scale_rows=m->out_features,scale_width=0;
    if (m->profile==RIA_EXPERT_BF16 || m->profile==RIA_EXPERT_F32) {
        if (!mul(width,m->profile==RIA_EXPERT_BF16 ? 2 : 4,&width)) return fail(error,RIA_RESOURCE_LIMIT,"weight byte size overflow");
        if (m->scales || m->scales_bytes || m->scale_row_stride)
            return fail(error,RIA_INVALID_REQUEST,"unquantized matrix has quantized scales");
    } else if (m->profile==RIA_EXPERT_FP8) {
        scale_rows=groups(m->out_features,32); scale_width=groups(m->in_features,32);
    } else if (m->profile==RIA_EXPERT_NVFP4) {
        width=groups(width,2); scale_width=groups(m->in_features,16);
        if (!(m->weight_global_scale>0) || !isfinite(m->weight_global_scale) ||
            !(m->activation_global_scale>0) || !isfinite(m->activation_global_scale))
            return fail(error,RIA_INVALID_REQUEST,"NVFP4 requires finite positive frozen global scales");
    } else return fail(error,RIA_UNSUPPORTED,"unknown expert numerical profile");
    if (!span(m->out_features,m->value_row_stride,width,m->values_bytes))
        return fail(error,RIA_INVALID_REQUEST,"expert value dimensions/stride exceed byte view");
    if (scale_width && (!m->scales || !span(scale_rows,m->scale_row_stride,scale_width,m->scales_bytes)))
        return fail(error,RIA_INVALID_REQUEST,"expert scale dimensions/stride exceed byte view");
    return true;
}
bool ria_expert_matrix_validate(const ria_expert_matrix *m,ria_error *error) {
    if (!matrix_shape(m,error)) return false;
    uint64_t scale_rows=m->profile==RIA_EXPERT_FP8 ? groups(m->out_features,32) : m->out_features;
    uint64_t scale_width=m->profile==RIA_EXPERT_FP8 ? groups(m->in_features,32) : groups(m->in_features,16);
    if (m->profile==RIA_EXPERT_FP8 || m->profile==RIA_EXPERT_NVFP4) {
        for (uint64_t row=0;row<scale_rows;++row) for (uint64_t k=0;k<scale_width;++k) {
            uint8_t code=m->scales[row*m->scale_row_stride+k];
            if ((m->profile==RIA_EXPERT_FP8 && code==255) ||
                (m->profile==RIA_EXPERT_NVFP4 && (code & 128u || code==127)))
                return fail(error,RIA_INTEGRITY_ERROR,"expert scale contains exceptional/negative value");
        }
    }
    for (uint64_t row=0;row<m->out_features;++row) {
        const uint8_t *value=m->values+row*m->value_row_stride;
        for (uint64_t k=0;k<m->in_features;++k) {
            if (m->profile==RIA_EXPERT_BF16 && !isfinite(read_bf16(value+2*k)))
                return fail(error,RIA_INTEGRITY_ERROR,"BF16 expert contains nonfinite weight");
            if (m->profile==RIA_EXPERT_F32 && !isfinite(read_float(value+4*k)))
                return fail(error,RIA_INTEGRITY_ERROR,"FP32 matrix contains nonfinite weight");
            if (m->profile==RIA_EXPERT_FP8 && (value[k]&127u)==127u)
                return fail(error,RIA_INTEGRITY_ERROR,"FP8 expert contains NaN weight");
        }
        if (m->profile==RIA_EXPERT_NVFP4 && (m->in_features & 1u) && (value[m->in_features/2]&240u))
            return fail(error,RIA_INTEGRITY_ERROR,"NVFP4 odd-width padding nibble is nonzero");
    }
    return true;
}
bool ria_expert_matrix_descriptor_validate(const ria_expert_matrix *m,ria_error *error) {
    return matrix_shape(m,error);
}
static bool expert_shape(const ria_expert *e,ria_error *error) {
    if (!e || !matrix_shape(&e->gate,error) || !matrix_shape(&e->up,error) || !matrix_shape(&e->down,error)) return false;
    if (e->gate.profile==RIA_EXPERT_F32 || e->gate.profile!=e->up.profile || e->gate.profile!=e->down.profile ||
        e->gate.in_features!=e->up.in_features || e->gate.out_features!=e->up.out_features ||
        e->down.in_features!=e->gate.out_features || !isfinite(e->clamp) || e->clamp<0)
        return fail(error,RIA_INVALID_REQUEST,"expert projections/profile/clamp disagree");
    return true;
}
bool ria_expert_validate(const ria_expert *e,ria_error *error) {
    return expert_shape(e,error) && ria_expert_matrix_validate(&e->gate,error) &&
        ria_expert_matrix_validate(&e->up,error) && ria_expert_matrix_validate(&e->down,error);
}
bool ria_expert_cuda_resident_required_bytes(const ria_expert *e,uint64_t *bytes,ria_error *error) {
    if (!bytes || !expert_shape(e,error) || e->gate.profile==RIA_EXPERT_F32)
        return fail(error,RIA_INVALID_REQUEST,"invalid resident expert profile/result");
    const ria_expert_matrix *matrices[3]={&e->gate,&e->up,&e->down};uint64_t total=0;
    for (unsigned i=0;i<3;++i) {
        const ria_expert_matrix *m=matrices[i];uint64_t value_width=m->profile==RIA_EXPERT_NVFP4 ? groups(m->in_features,2) : m->in_features;
        uint64_t part;
        if (!matrix_shape(m,error)) return false;
        if (m->profile==RIA_EXPERT_BF16 && !mul(value_width,2,&value_width)) return fail(error,RIA_RESOURCE_LIMIT,"resident weight width overflow");
        if (!mul(value_width,m->out_features,&part) || part>UINT64_MAX-total) return fail(error,RIA_RESOURCE_LIMIT,"resident weight population overflow");
        total+=part;
        if (m->profile==RIA_EXPERT_FP8 || m->profile==RIA_EXPERT_NVFP4) {
            uint64_t scale_rows=m->profile==RIA_EXPERT_FP8 ? groups(m->out_features,32) : m->out_features;
            uint64_t scale_width=groups(m->in_features,m->profile==RIA_EXPERT_FP8 ? 32 : 16);
            if (!mul(scale_rows,scale_width,&part) || part>UINT64_MAX-total) return fail(error,RIA_RESOURCE_LIMIT,"resident scale population overflow");
            total+=part;
        }
    }
    *bytes=total;return true;
}
float ria_expert_bf16_round(float x) { return ria_num_bf16(x); }
float ria_expert_e4m3_decode(uint8_t x) { return ria_num_e4m3(x); }
uint8_t ria_expert_e4m3_encode(float x) { return ria_num_to_e4m3(x); }
float ria_expert_e2m1_decode(uint8_t x) { return ria_num_e2m1(x); }
uint8_t ria_expert_e2m1_encode(float x) { return ria_num_to_e2m1(x); }
float ria_expert_ue8m0_decode(uint8_t x) { return ria_num_ue8m0(x); }

static bool quantize(const float *logical,uint64_t count,ria_expert_profile profile,
                     float global,float *values,float *factors,uint8_t *codes,
                     uint8_t *scale_codes,ria_error *error) {
    if (profile==RIA_EXPERT_BF16 || profile==RIA_EXPERT_F32) {
        memcpy(values,logical,(size_t)count*sizeof(float)); return true;
    }
    uint64_t group=profile==RIA_EXPERT_FP8 ? 32 : 16;
    if (profile==RIA_EXPERT_NVFP4 && codes) memset(codes,0,(size_t)groups(count,2));
    for (uint64_t base=0;base<count;base+=group) {
        uint64_t end=count-base<group ? count : base+group;
        float maximum=0;
        for (uint64_t k=base;k<end;++k) maximum=fmaxf(maximum,fabsf(logical[k]));
        uint8_t code=profile==RIA_EXPERT_FP8 ? ria_num_fp8_scale(maximum) : ria_num_nvfp4_scale(maximum,global);
        float scale=profile==RIA_EXPERT_FP8 ? ria_num_ue8m0(code) : ria_num_e4m3(code);
        float factor=profile==RIA_EXPERT_FP8 ? scale : scale*global;
        if (!isfinite(factor)) return fail(error,RIA_EXECUTOR_ERROR,"activation scale overflow");
        factors[base/group]=scale;
        if (scale_codes) scale_codes[base/group]=code;
        for (uint64_t k=base;k<end;++k) {
            float normalized=factor>0 ? logical[k]/factor : 0;
            uint8_t value=profile==RIA_EXPERT_FP8 ? ria_num_to_e4m3(normalized) : ria_num_to_e2m1(normalized);
            values[k]=profile==RIA_EXPERT_FP8 ? ria_num_e4m3(value) : ria_num_e2m1(value);
            if (codes) {
                if (profile==RIA_EXPERT_FP8) codes[k]=value;
                else codes[k/2]|=(uint8_t)(value<<(4*(k&1u)));
            }
        }
    }
    return true;
}
bool ria_expert_quantize(const float *input,uint64_t count,ria_expert_profile profile,
                        float global,float *decoded,uint8_t *codes,uint8_t *scales,ria_error *error) {
    if (!input || !decoded || !count || count>SIZE_MAX/(2*sizeof(float)) ||
        profile<RIA_EXPERT_BF16 || profile>RIA_EXPERT_NVFP4 ||
        (profile==RIA_EXPERT_NVFP4 && (!(global>0) || !isfinite(global))))
        return fail(error,RIA_INVALID_REQUEST,"invalid activation quantizer input");
    if (!math_mode(error)) return false;
    float *scratch=malloc((size_t)(count+groups(count,16))*sizeof(float));
    if (!scratch) return fail(error,RIA_RESOURCE_LIMIT,"activation quantizer allocation failed");
    for (uint64_t k=0;k<count;++k) {
        scratch[k]=ria_num_bf16(input[k]);
        if (!isfinite(input[k]) || !isfinite(scratch[k])) {
            free(scratch); return fail(error,RIA_INVALID_REQUEST,"nonfinite logical BF16 activation");
        }
    }
    bool ok=quantize(scratch,count,profile,global,decoded,scratch+count,codes,scales,error);
    if (ok && profile!=RIA_EXPERT_BF16) {
        uint64_t group=profile==RIA_EXPERT_FP8 ? 32 : 16;
        for (uint64_t k=0;k<count;++k) {
            decoded[k]*=scratch[count+k/group];
            if (profile==RIA_EXPERT_NVFP4) decoded[k]*=global;
        }
    }
    free(scratch); return ok;
}
static float weight_value(const ria_expert_matrix *m,uint64_t row,uint64_t k) {
    const uint8_t *value=m->values+row*m->value_row_stride;
    if (m->profile==RIA_EXPERT_BF16) return read_bf16(value+2*k);
    if (m->profile==RIA_EXPERT_F32) return read_float(value+4*k);
    if (m->profile==RIA_EXPERT_FP8) return ria_num_e4m3(value[k]);
    return ria_num_e2m1((uint8_t)(value[k/2]>>(4*(k&1u))));
}
static float weight_scale(const ria_expert_matrix *m,uint64_t row,uint64_t block) {
    if (m->profile==RIA_EXPERT_FP8)
        return ria_num_ue8m0(m->scales[(row/32)*m->scale_row_stride+block]);
    return ria_num_e4m3(m->scales[row*m->scale_row_stride+block]);
}
static void dot_scalar_four(const ria_expert_matrix *m,uint64_t row,const float *x,const float *scales,float *result) {
    for (uint64_t lane=0;lane<4;++lane) {
        float total=0;
        uint64_t group=(m->profile==RIA_EXPERT_BF16 || m->profile==RIA_EXPERT_F32) ? m->in_features : m->profile==RIA_EXPERT_FP8 ? 32 : 16;
        for (uint64_t base=0;base<m->in_features;base+=group) {
            float partial=0;
            uint64_t end=m->in_features-base<group ? m->in_features : base+group;
            for (uint64_t k=base;k<end;++k) partial=fmaf(weight_value(m,row+lane,k),x[k],partial);
            if (m->profile==RIA_EXPERT_FP8 || m->profile==RIA_EXPERT_NVFP4) {
                partial=partial*scales[base/group];
                partial=partial*weight_scale(m,row+lane,base/group);
            }
            total=total+partial;
        }
        if (m->profile==RIA_EXPERT_NVFP4) total=total*(m->activation_global_scale*m->weight_global_scale);
        result[lane]=total;
    }
}
#if defined(__aarch64__)
static void dot_neon_four(const ria_expert_matrix *m,uint64_t row,const float *x,const float *scales,float *result) {
    float32x4_t total=vdupq_n_f32(0);
    uint64_t group=(m->profile==RIA_EXPERT_BF16 || m->profile==RIA_EXPERT_F32) ? m->in_features : m->profile==RIA_EXPERT_FP8 ? 32 : 16;
    for (uint64_t base=0;base<m->in_features;base+=group) {
        float32x4_t partial=vdupq_n_f32(0);
        uint64_t end=m->in_features-base<group ? m->in_features : base+group;
        for (uint64_t k=base;k<end;++k) {
            float w[4]; for (uint64_t lane=0;lane<4;++lane) w[lane]=weight_value(m,row+lane,k);
            partial=vfmaq_n_f32(partial,vld1q_f32(w),x[k]);
        }
        if (m->profile==RIA_EXPERT_FP8 || m->profile==RIA_EXPERT_NVFP4) {
            float s[4]; for (uint64_t lane=0;lane<4;++lane) s[lane]=weight_scale(m,row+lane,base/group);
            partial=vmulq_f32(vmulq_n_f32(partial,scales[base/group]),vld1q_f32(s));
        }
        total=vaddq_f32(total,partial);
    }
    if (m->profile==RIA_EXPERT_NVFP4) total=vmulq_n_f32(total,m->activation_global_scale*m->weight_global_scale);
    vst1q_f32(result,total);
}
#elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,fma")))
static void dot_avx_four(const ria_expert_matrix *m,uint64_t row,const float *x,const float *scales,float *result) {
    __m128 total=_mm_setzero_ps();
    uint64_t group=(m->profile==RIA_EXPERT_BF16 || m->profile==RIA_EXPERT_F32) ? m->in_features : m->profile==RIA_EXPERT_FP8 ? 32 : 16;
    for (uint64_t base=0;base<m->in_features;base+=group) {
        __m128 partial=_mm_setzero_ps();
        uint64_t end=m->in_features-base<group ? m->in_features : base+group;
        for (uint64_t k=base;k<end;++k) {
            float w[4]; for (uint64_t lane=0;lane<4;++lane) w[lane]=weight_value(m,row+lane,k);
            partial=_mm_fmadd_ps(_mm_loadu_ps(w),_mm_set1_ps(x[k]),partial);
        }
        if (m->profile==RIA_EXPERT_FP8 || m->profile==RIA_EXPERT_NVFP4) {
            float s[4]; for (uint64_t lane=0;lane<4;++lane) s[lane]=weight_scale(m,row+lane,base/group);
            partial=_mm_mul_ps(_mm_mul_ps(partial,_mm_set1_ps(scales[base/group])),_mm_loadu_ps(s));
        }
        total=_mm_add_ps(total,partial);
    }
    if (m->profile==RIA_EXPERT_NVFP4) total=_mm_mul_ps(total,_mm_set1_ps(m->activation_global_scale*m->weight_global_scale));
    _mm_storeu_ps(result,total);
}
#endif
static bool project(ria_expert_cpu *c,const ria_expert_matrix *m,const float *logical,float *output,bool round_bf16,ria_error *error) {
    if (!quantize(logical,m->in_features,m->profile,m->activation_global_scale,c->quantized,c->activation_scales,NULL,NULL,error)) return false;
    uint64_t row=0;
    for (;m->out_features-row>=4;row+=4) c->dot_four(m,row,c->quantized,c->activation_scales,output+row);
    for (;row<m->out_features;++row) {
        ria_expert_matrix one=*m; one.values=m->values+row*m->value_row_stride; one.value_row_stride=0;
        float total=0;
        uint64_t group=(m->profile==RIA_EXPERT_BF16 || m->profile==RIA_EXPERT_F32) ? m->in_features : m->profile==RIA_EXPERT_FP8 ? 32 : 16;
        for (uint64_t base=0;base<m->in_features;base+=group) {
            float partial=0;
            uint64_t end=m->in_features-base<group ? m->in_features : base+group;
            for (uint64_t k=base;k<end;++k) partial=fmaf(weight_value(&one,0,k),c->quantized[k],partial);
            if (m->profile==RIA_EXPERT_FP8 || m->profile==RIA_EXPERT_NVFP4) {
                partial=partial*c->activation_scales[base/group];
                partial=partial*weight_scale(m,row,base/group);
            }
            total=total+partial;
        }
        if (m->profile==RIA_EXPERT_NVFP4) total=total*(m->activation_global_scale*m->weight_global_scale);
        output[row]=total;
    }
    for (uint64_t n=0;n<m->out_features;++n) {
        if (round_bf16) output[n]=ria_num_bf16(output[n]);
        if (!isfinite(output[n])) return fail(error,RIA_EXECUTOR_ERROR,"expert projection produced nonfinite output");
    }
    return true;
}
bool ria_expert_cpu_required_bytes(uint64_t input,uint64_t intermediate,uint64_t output,uint64_t *result,ria_error *error) {
    if (!result) return fail(error,RIA_INVALID_REQUEST,"missing expert workspace size result");
    if (!input || !intermediate || !output) return fail(error,RIA_INVALID_REQUEST,"zero expert context dimensions");
    uint64_t width=input>intermediate ? input : intermediate, count,bytes;
    if (!mul(width,2,&count) || count>UINT64_MAX-groups(width,16) ||
        (count+=groups(width,16))>UINT64_MAX-intermediate ||
        (count+=intermediate)>UINT64_MAX-intermediate || !mul(count+intermediate,sizeof(float),&bytes) || bytes>SIZE_MAX-sizeof(ria_expert_cpu))
        return fail(error,RIA_RESOURCE_LIMIT,"expert workspace overflow");
    *result=bytes;return true;
}
bool ria_expert_cpu_create_in(uint64_t input,uint64_t intermediate,uint64_t output,void *workspace,uint64_t supplied,
                              ria_expert_cpu **out,ria_error *error) {
    if (!out) return fail(error,RIA_INVALID_REQUEST,"missing expert context result");
    *out=NULL;uint64_t bytes;
    if (!math_mode(error) || !ria_expert_cpu_required_bytes(input,intermediate,output,&bytes,error)) return false;
    if (!workspace || (uintptr_t)workspace%offsetof(ria_float_alignment,value) || supplied<bytes || supplied>SIZE_MAX)
        return fail(error,RIA_INVALID_REQUEST,"expert arena is missing, misaligned or smaller than admitted scratch");
    ria_expert_cpu *c=calloc(1,sizeof(*c));
    if (!c) return fail(error,RIA_RESOURCE_LIMIT,"expert context allocation failed");
    c->allocation=workspace;uint64_t width=input>intermediate ? input : intermediate;
    c->max_input=input; c->max_intermediate=intermediate; c->max_output=output;
    c->workspace_bytes=bytes+sizeof(*c); c->width=width;
    c->logical=c->allocation; c->quantized=c->logical+width;
    c->activation_scales=c->quantized+width; c->gate=c->activation_scales+groups(width,16); c->up=c->gate+intermediate;
    c->dot_four=dot_scalar_four; c->kernel="scalar-fma";
#if defined(__aarch64__)
    c->dot_four=dot_neon_four; c->kernel="aarch64-neon-fma4";
#elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) { c->dot_four=dot_avx_four; c->kernel="x86-avx2-fma4"; }
#endif
    *out=c; return true;
}
bool ria_expert_cpu_create(uint64_t input,uint64_t intermediate,uint64_t output,ria_expert_cpu **out,ria_error *error) {
    if (!out) return fail(error,RIA_INVALID_REQUEST,"missing expert context result");
    *out=NULL;uint64_t bytes;
    if (!math_mode(error) || !ria_expert_cpu_required_bytes(input,intermediate,output,&bytes,error)) return false;
    void *workspace=malloc((size_t)bytes);
    if (!workspace) return fail(error,RIA_RESOURCE_LIMIT,"expert workspace allocation failed");
    if (!ria_expert_cpu_create_in(input,intermediate,output,workspace,bytes,out,error)) { free(workspace);return false; }
    (*out)->owns_allocation=true;return true;
}
void ria_expert_cpu_destroy(ria_expert_cpu *c) { if (c) { if (c->owns_allocation) free(c->allocation);free(c); } }
const char *ria_expert_cpu_kernel(const ria_expert_cpu *c) { return c ? c->kernel : "uninitialized"; }
uint64_t ria_expert_cpu_workspace_bytes(const ria_expert_cpu *c) { return c ? c->workspace_bytes : 0; }
uint64_t ria_expert_cpu_metadata_bytes(void) { return sizeof(ria_expert_cpu); }
bool ria_expert_cpu_evaluate(ria_expert_cpu *c,const ria_expert *e,const float *input,uint64_t rows,
                            uint64_t input_stride,const float *coefficients,float *output,uint64_t output_stride,ria_error *error) {
    if (!c || !input || !output || !rows || !expert_shape(e,error)) return fail(error,RIA_INVALID_REQUEST,"invalid expert evaluation");
    if (!math_mode(error)) return false;
    uint64_t input_bytes,output_bytes;
    if (e->gate.in_features>c->max_input || e->gate.out_features>c->max_intermediate || e->down.out_features>c->max_output ||
        !ria_expert_float_span_bytes(rows,input_stride,e->gate.in_features,&input_bytes) ||
        !ria_expert_float_span_bytes(rows,output_stride,e->down.out_features,&output_bytes))
        return fail(error,RIA_RESOURCE_LIMIT,"expert evaluation exceeds admitted dimensions/strides");
    if (!ria_expert_ranges_disjoint(input,input_bytes,output,output_bytes) ||
        (coefficients && (!ria_expert_ranges_disjoint(coefficients,rows*sizeof(float),input,input_bytes) ||
                          !ria_expert_ranges_disjoint(coefficients,rows*sizeof(float),output,output_bytes))))
        return fail(error,RIA_INVALID_REQUEST,"expert input/coefficient/output ranges overlap");
    for (uint64_t row=0;row<rows;++row) {
        float coefficient=coefficients ? coefficients[row] : 1.0f;
        if (!isfinite(coefficient) || coefficient<0) return fail(error,RIA_INVALID_REQUEST,"invalid routing coefficient");
        for (uint64_t k=0;k<e->gate.in_features;++k) {
            c->logical[k]=ria_num_bf16(input[row*input_stride+k]);
            if (!isfinite(c->logical[k])) return fail(error,RIA_INVALID_REQUEST,"nonfinite logical BF16 expert input");
        }
        if (!project(c,&e->gate,c->logical,c->gate,true,error) || !project(c,&e->up,c->logical,c->up,true,error)) return false;
        for (uint64_t k=0;k<e->gate.out_features;++k) {
            float gate=c->gate[k],up=c->up[k];
            if (e->clamp>0) { gate=fminf(gate,e->clamp); up=fmaxf(-e->clamp,fminf(up,e->clamp)); }
            /* F.silu runs on the widened FP32 projections. The routing
             * coefficient is applied before the meaningful BF16/down A4/A8 boundary. */
            float silu=gate/(1.0f+expf(-gate));
            c->logical[k]=ria_num_bf16((silu*up)*coefficient);
            if (!isfinite(c->logical[k])) return fail(error,RIA_EXECUTOR_ERROR,"nonfinite expert gated intermediate");
        }
        if (!project(c,&e->down,c->logical,output+row*output_stride,true,error)) return false;
    }
    return true;
}
bool ria_expert_cpu_projection(ria_expert_cpu *c,const ria_expert_matrix *m,const float *input,
                              uint64_t rows,uint64_t input_stride,float *output,uint64_t output_stride,
                              bool round_bf16,ria_error *error) {
    if (!c || !input || !output || !rows || !matrix_shape(m,error)) return fail(error,RIA_INVALID_REQUEST,"invalid projection");
    if (!math_mode(error)) return false;
    uint64_t max_output=c->max_output>c->max_intermediate ? c->max_output : c->max_intermediate,input_bytes,output_bytes;
    if (m->in_features>c->width || m->out_features>max_output ||
        !ria_expert_float_span_bytes(rows,input_stride,m->in_features,&input_bytes) ||
        !ria_expert_float_span_bytes(rows,output_stride,m->out_features,&output_bytes))
        return fail(error,RIA_RESOURCE_LIMIT,"projection exceeds admitted dimensions/strides");
    if (!ria_expert_ranges_disjoint(input,input_bytes,output,output_bytes))
        return fail(error,RIA_INVALID_REQUEST,"projection input/output ranges overlap");
    for (uint64_t row=0;row<rows;++row) {
        for (uint64_t k=0;k<m->in_features;++k) {
            float value=input[row*input_stride+k];
            c->logical[k]=m->profile==RIA_EXPERT_F32 ? value : ria_num_bf16(value);
            if (!isfinite(c->logical[k])) return fail(error,RIA_INVALID_REQUEST,"nonfinite projection input");
        }
        if (!project(c,m,c->logical,output+row*output_stride,round_bf16,error)) return false;
    }
    return true;
}
