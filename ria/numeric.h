#ifndef RIA_NUMERIC_H
#define RIA_NUMERIC_H
#include <math.h>
#include <stdint.h>
#include <string.h>

#if defined(__FAST_MATH__)
#error "RIA numerical profiles require finite checks and strict floating-point compilation"
#endif

#if defined(__CUDACC__)
#define RIA_NUMERIC_INLINE static __host__ __device__ __forceinline__
#else
#define RIA_NUMERIC_INLINE static inline
#endif

RIA_NUMERIC_INLINE uint32_t ria_num_bits(float value) {
#ifdef __CUDA_ARCH__
    return __float_as_uint(value);
#else
    uint32_t bits; memcpy(&bits,&value,sizeof(bits)); return bits;
#endif
}
RIA_NUMERIC_INLINE float ria_num_float(uint32_t bits) {
#ifdef __CUDA_ARCH__
    return __uint_as_float(bits);
#else
    float value; memcpy(&value,&bits,sizeof(value)); return value;
#endif
}
RIA_NUMERIC_INLINE float ria_num_bf16(float value) {
    uint32_t bits=ria_num_bits(value);
    if ((bits & UINT32_C(0x7fffffff)) > UINT32_C(0x7f800000))
        return ria_num_float((bits & UINT32_C(0xffff0000)) | UINT32_C(0x00400000));
    bits += UINT32_C(0x7fff) + ((bits >> 16) & 1u);
    return ria_num_float(bits & UINT32_C(0xffff0000));
}
RIA_NUMERIC_INLINE float ria_num_e4m3(uint8_t code) {
    unsigned magnitude=code & 127u;
    if (magnitude==127u) return ria_num_float(UINT32_C(0x7fc00000));
    float value=magnitude<8u ? ldexpf((float)magnitude,-9) :
        ldexpf(1.0f+(float)(magnitude & 7u)*0.125f,(int)(magnitude>>3)-7);
    return (code & 128u) ? -value : value;
}
RIA_NUMERIC_INLINE unsigned ria_num_even(float value) {
    float integral=floorf(value), fraction=value-integral;
    unsigned result=(unsigned)integral;
    return result + (fraction>0.5f || (fraction==0.5f && (result & 1u)));
}
RIA_NUMERIC_INLINE uint8_t ria_num_to_e4m3(float value) {
    uint8_t sign=(uint8_t)((ria_num_bits(value)>>24)&128u);
    float magnitude=fabsf(value);
    if (isnan(value)) return 127u;
    if (magnitude>=448.0f) return (uint8_t)(sign|126u);
    if (magnitude<0.015625f) return (uint8_t)(sign|ria_num_even(magnitude*512.0f));
    int exponent;
    (void)frexpf(magnitude,&exponent);
    unsigned significand=ria_num_even(ldexpf(magnitude,4-exponent));
    return (uint8_t)(sign | ((unsigned)(exponent+5)*8u+significand));
}
RIA_NUMERIC_INLINE float ria_num_e2m1(uint8_t code) {
    const float values[8]={0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f};
    float value=values[code & 7u];
    return (code & 8u) ? -value : value;
}
RIA_NUMERIC_INLINE uint8_t ria_num_to_e2m1(float value) {
    const float boundaries[7]={0.25f,0.75f,1.25f,1.75f,2.5f,3.5f,5.0f};
    uint8_t sign=(uint8_t)((ria_num_bits(value)>>28)&8u), code=7u;
    float magnitude=fabsf(value);
    for (uint8_t i=0;i<7u;++i) {
        if (magnitude<boundaries[i] || (magnitude==boundaries[i] && !(i & 1u))) {
            code=i; break;
        }
    }
    return (uint8_t)(sign|code);
}
RIA_NUMERIC_INLINE float ria_num_ue8m0(uint8_t code) {
    return code==255u ? ria_num_float(UINT32_C(0x7fc00000)) : ldexpf(1.0f,(int)code-127);
}
RIA_NUMERIC_INLINE uint8_t ria_num_fp8_scale(float maximum) {
    /* The pinned native act_quant clamps amax to 1e-4, multiplies by
     * FP32(1/448), and rounds UP to a power of two by inspecting FP32 bits. */
    float bounded=maximum>1e-4f ? maximum : 1e-4f;
    uint32_t bits=ria_num_bits(bounded*(1.0f/448.0f));
    unsigned exponent=(bits>>23)&255u;
    return (uint8_t)(exponent + ((bits & UINT32_C(0x7fffff))!=0));
}
RIA_NUMERIC_INLINE uint8_t ria_num_nvfp4_scale(float maximum,float global) {
    return ria_num_to_e4m3((maximum / 6.0f) / global);
}
/* NVIDIA PTX ISA m16n8k16 BF16 / m16n8k32 E4M3 / m16n8k64 E2M1
 * register fragments. bits is the declared 16/8/4-bit operand width;
 * reg/element are within A[4], B[2], C[4], lane within one full warp.
 * Shared with host verification so layouts are checked without a GPU. */
RIA_NUMERIC_INLINE unsigned ria_num_mma_a_row(unsigned lane,unsigned reg) { return lane/4+(reg%2)*8; }
RIA_NUMERIC_INLINE unsigned ria_num_mma_a_col(unsigned lane,unsigned reg,unsigned element,unsigned bits) {
    unsigned packed=32/bits;return lane%4*packed+reg/2*(4*packed)+element;
}
RIA_NUMERIC_INLINE unsigned ria_num_mma_b_row(unsigned lane,unsigned reg,unsigned element,unsigned bits) {
    unsigned packed=32/bits;return lane%4*packed+reg*(4*packed)+element;
}
RIA_NUMERIC_INLINE unsigned ria_num_mma_c_row(unsigned lane,unsigned element) { return lane/4+(element/2)*8; }
RIA_NUMERIC_INLINE unsigned ria_num_mma_c_col(unsigned lane,unsigned element) { return lane%4*2+element%2; }
RIA_NUMERIC_INLINE uint32_t ria_num_mma_activation_fragment(const uint8_t *codes,uint64_t width,uint64_t rows,
    uint64_t row_base,uint64_t k_base,unsigned lane,unsigned reg,unsigned bits,unsigned valid_k) {
    uint64_t row=row_base+ria_num_mma_a_row(lane,reg);uint32_t result=0;
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
    for (unsigned element=0;element<32/bits;++element) {
        unsigned col=ria_num_mma_a_col(lane,reg,element,bits);uint64_t k=k_base+col;
        if (row<rows && col<valid_k && k<width) {
            uint64_t offset=row*width+k;uint32_t code=codes[offset*(bits==16 ? 2 : 1)];
            if (bits==16) code|=(uint32_t)codes[offset*2+1]<<8;
            result|=code<<(element*bits);
        }
    }
    return result;
}
RIA_NUMERIC_INLINE uint32_t ria_num_mma_weight_fragment(const uint8_t *values,uint64_t width,uint64_t count,
    uint64_t n_base,uint64_t k_base,uint64_t stride,unsigned lane,unsigned reg,unsigned bits,unsigned valid_k) {
    uint64_t n=n_base+lane/4;uint32_t result=0;
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
    for (unsigned element=0;element<32/bits;++element) {
        unsigned col=ria_num_mma_b_row(lane,reg,element,bits);uint64_t k=k_base+col;
        if (n<count && col<valid_k && k<width) {
            const uint8_t *row=values+n*stride;uint32_t code;
            if (bits==16) code=(uint32_t)row[k*2]|((uint32_t)row[k*2+1]<<8);
            else if (bits==8) code=row[k];
            else code=((uint32_t)row[k/2]>>(4*(k%2)))&15u;
            result|=code<<(element*bits);
        }
    }
    return result;
}
RIA_NUMERIC_INLINE uint32_t ria_num_mma_nvfp4_scale_a(const uint8_t *codes,uint64_t groups,uint64_t rows,
    uint64_t row_base,uint64_t group,unsigned lane) {
    uint64_t row=row_base+lane/4+(lane%4==1 ? 8 : 0);
    return lane%4<2 && row<rows ? codes[row*groups+group] : 0;
}
RIA_NUMERIC_INLINE uint32_t ria_num_mma_nvfp4_scale_b(const uint8_t *codes,uint64_t stride,uint64_t count,
    uint64_t n_base,uint64_t group,unsigned lane) {
    uint64_t n=n_base+lane/4;
    return lane%4==0 && n<count ? codes[n*stride+group] : 0;
}
#endif
