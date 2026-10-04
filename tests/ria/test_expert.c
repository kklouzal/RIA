#define _GNU_SOURCE
#include "../../ria/expert.h"
#include "../../ria/numeric.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fenv.h>
#if defined(__linux__)
#include <sys/mman.h>
#endif

static uint32_t bits(float value) { uint32_t result; memcpy(&result,&value,4); return result; }
static float floating(uint32_t value) { float result; memcpy(&result,&value,4); return result; }
static void bf16_bytes(uint8_t *destination,float value) {
    uint32_t representation=bits(ria_expert_bf16_round(value));
    destination[0]=(uint8_t)(representation>>16); destination[1]=(uint8_t)(representation>>24);
}
static ria_expert_matrix bf16_matrix(uint8_t *bytes,uint64_t n,uint64_t k) {
    ria_expert_matrix matrix={0}; matrix.profile=RIA_EXPERT_BF16;
    matrix.values=bytes; matrix.out_features=n; matrix.in_features=k;
    matrix.value_row_stride=k*2; matrix.values_bytes=n*k*2; return matrix;
}
static void numeric_codes(void) {
    static const float fp4[16]={0,.5f,1,1.5f,2,3,4,6,-0.0f,-.5f,-1,-1.5f,-2,-3,-4,-6};
    for (unsigned i=0;i<16;++i) {
        assert(bits(ria_expert_e2m1_decode((uint8_t)i))==bits(fp4[i]));
        assert(ria_expert_e2m1_encode(fp4[i])==i);
    }
    assert(ria_expert_e2m1_encode(.25f)==0);
    assert(ria_expert_e2m1_encode(.75f)==2);
    assert(ria_expert_e2m1_encode(1.25f)==2);
    assert(ria_expert_e2m1_encode(1.75f)==4);
    assert(ria_expert_e2m1_encode(2.5f)==4);
    assert(ria_expert_e2m1_encode(3.5f)==6);
    assert(ria_expert_e2m1_encode(5)==6);
    assert(ria_expert_e2m1_encode(INFINITY)==7);
    for (unsigned i=0;i<256;++i) {
        float x=ria_expert_e4m3_decode((uint8_t)i);
        if ((i&127)==127) assert(isnan(x));
        else assert(ria_expert_e4m3_encode(x)==i);
        float scale=ria_expert_ue8m0_decode((uint8_t)i);
        if (i==255) assert(isnan(scale));
        else assert(scale==ldexpf(1,(int)i-127));
    }
    assert(ria_expert_e4m3_encode(INFINITY)==126);
    assert(ria_expert_e4m3_encode(-INFINITY)==254);
    assert(ria_expert_e4m3_encode(1.0625f)==56);
    assert(ria_expert_e4m3_encode(1.1875f)==58);
    assert(ria_expert_e4m3_encode(ldexpf(1,-10))==0);
    assert(ria_expert_e4m3_decode(126)==448);
    assert(ria_expert_e4m3_decode(8)==ldexpf(1,-6));
    assert(ria_expert_e4m3_decode(1)==ldexpf(1,-9));
    assert(bits(ria_expert_bf16_round(floating(UINT32_C(0x3f808000))))==UINT32_C(0x3f800000));
    assert(bits(ria_expert_bf16_round(floating(UINT32_C(0x3f818000))))==UINT32_C(0x3f820000));
    assert(bits(ria_expert_bf16_round(-0.0f))==UINT32_C(0x80000000));
    assert(isnan(ria_expert_bf16_round(NAN)));
}
static void quantizers(void) {
    float input[65]={0},decoded[65],split[65]; uint8_t codes[65],scales[5]; ria_error error={0};
    assert(ria_expert_quantize(input,65,RIA_EXPERT_FP8,0,decoded,codes,scales,&error));
    assert(scales[0]==105 && scales[1]==105 && scales[2]==105);
    for (unsigned i=0;i<65;++i) assert(decoded[i]==0 && codes[i]==0);
    input[0]=1;input[31]=-2;input[32]=256;input[63]=448;input[64]=.015625f;
    assert(ria_expert_quantize(input,65,RIA_EXPERT_FP8,0,decoded,codes,scales,&error));
    assert(scales[0]==120 && scales[1]==127 && scales[2]==113);
    assert(ria_expert_quantize(input,32,RIA_EXPERT_FP8,0,split,NULL,NULL,&error));
    assert(ria_expert_quantize(input+32,32,RIA_EXPERT_FP8,0,split+32,NULL,NULL,&error));
    assert(ria_expert_quantize(input+64,1,RIA_EXPERT_FP8,0,split+64,NULL,NULL,&error));
    assert(memcmp(decoded,split,sizeof(decoded))==0);
    memset(input,0,sizeof(input));
    assert(ria_expert_quantize(input,65,RIA_EXPERT_NVFP4,1,decoded,codes,scales,&error));
    for (unsigned i=0;i<5;++i) assert(scales[i]==0);
    input[0]=6;input[1]=-.5f;input[16]=12;input[64]=3;
    assert(ria_expert_quantize(input,65,RIA_EXPERT_NVFP4,1,decoded,codes,scales,&error));
    assert(scales[0]==56 && scales[1]==64 && scales[4]==48);
    assert(codes[0]==UINT8_C(0x97)); assert((codes[32]&240)==0);
    assert(decoded[0]==6 && decoded[1]==-.5f && decoded[16]==12 && decoded[64]==3);
    input[3]=NAN;assert(!ria_expert_quantize(input,65,RIA_EXPERT_NVFP4,1,decoded,NULL,NULL,&error));
    input[3]=INFINITY;assert(!ria_expert_quantize(input,65,RIA_EXPERT_FP8,0,decoded,NULL,NULL,&error));
    assert(!ria_expert_quantize(input,65,RIA_EXPERT_NVFP4,0,decoded,NULL,NULL,&error));
}
static void projection_and_expert(void) {
    ria_error error={0}; ria_expert_cpu *context=NULL;
    assert(ria_expert_cpu_create(65,65,65,&context,&error));
    assert(ria_expert_cpu_workspace_bytes(context)<4096);
    uint8_t gate[12],up[12],down[8];
    const float gate_weights[6]={-30,0,30,0,0,1},up_weights[6]={30,0,-30,0,0,1};
    for (unsigned i=0;i<6;++i) { bf16_bytes(gate+2*i,gate_weights[i]);bf16_bytes(up+2*i,up_weights[i]); }
    for (unsigned i=0;i<4;++i) bf16_bytes(down+2*i,i==0 ? 1 : 0);
    ria_expert expert={bf16_matrix(gate,3,2),bf16_matrix(up,3,2),bf16_matrix(down,1,3),10};
    expert.down.values_bytes=6;
    assert(ria_expert_validate(&expert,&error));
    float input[6]={1,1,123,1,1,456},coefficients[2]={1,0},output[4]={999,888,777,666};
    assert(ria_expert_cpu_evaluate(context,&expert,input,2,3,coefficients,output,2,&error));
    /* An erroneous lower gate clamp gives ~-0.0045. Correct gate=-30
     * remains tiny after the upper-only clamp and positive up clamp. */
    assert(output[0]<0 && output[0]>-1e-9f); assert(output[2]==0);
    assert(output[1]==888 && output[3]==666);
    coefficients[0]=-1;assert(!ria_expert_cpu_evaluate(context,&expert,input,2,3,coefficients,output,2,&error));
    coefficients[0]=1; input[0]=NAN;
    assert(!ria_expert_cpu_evaluate(context,&expert,input,2,3,coefficients,output,2,&error));
    uint8_t values[65*33],scale[4]={127,128,126,129};
    memset(values,56,sizeof(values));
    ria_expert_matrix fp8={RIA_EXPERT_FP8,33,65,values,sizeof(values),65,scale,sizeof(scale),3,0,0};
    /* Three K-blocks and two N-blocks are required, reject short scales. */
    assert(!ria_expert_matrix_validate(&fp8,&error));
    uint8_t six_scales[6]={127,128,129,126,127,128};fp8.scales=six_scales;fp8.scales_bytes=6;
    assert(ria_expert_matrix_validate(&fp8,&error));
    float activation[65],result[33];for (unsigned i=0;i<65;++i) activation[i]=1;
    assert(ria_expert_cpu_projection(context,&fp8,activation,1,65,result,33,true,&error));
    assert(result[0]==100 && result[31]==100 && result[32]==50);
    six_scales[4]=255; assert(!ria_expert_matrix_validate(&fp8,&error));six_scales[4]=127;
    values[64]=127;assert(!ria_expert_matrix_validate(&fp8,&error));values[64]=56;
    fp8.value_row_stride=UINT64_MAX;assert(!ria_expert_matrix_descriptor_validate(&fp8,&error));
    uint8_t odd[2]={UINT8_C(0x22),UINT8_C(0xf2)},one_scale=56;
    ria_expert_matrix fp4={RIA_EXPERT_NVFP4,1,3,odd,2,2,&one_scale,1,1,1,1};
    assert(!ria_expert_matrix_validate(&fp4,&error));odd[1]=2;
    assert(ria_expert_matrix_validate(&fp4,&error));one_scale=128;assert(!ria_expert_matrix_validate(&fp4,&error));
    ria_expert_cpu_destroy(context);
    context=NULL;assert(!ria_expert_cpu_create(UINT64_MAX,1,1,&context,&error));assert(context==NULL);
    assert(fesetround(FE_DOWNWARD)==0);
    assert(!ria_expert_cpu_create(1,1,1,&context,&error));assert(context==NULL);
    assert(fesetround(FE_TONEAREST)==0);
}
static void wide_strides(void) {
#if defined(__linux__) && UINTPTR_MAX>UINT32_MAX
    /* Valid sparse VIRTUAL fixtures: two populated pages, no large model or
     * resident allocation. Exercise actual loads across 4 GiB and 1 TiB. */
    const uint64_t strides[2]={UINT64_C(1)<<32,UINT64_C(1)<<40};
    ria_error error={0};ria_expert_cpu *context=NULL;
    assert(ria_expert_cpu_create(1,1,2,&context,&error));
    for (unsigned i=0;i<2;++i) {
        size_t size=(size_t)strides[i]+4096;
        uint8_t *memory=mmap(NULL,size,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE,-1,0);
        assert(memory!=MAP_FAILED);
        bf16_bytes(memory,2);bf16_bytes(memory+strides[i],3);
        ria_expert_matrix matrix=bf16_matrix(memory,2,1);
        matrix.value_row_stride=strides[i];matrix.values_bytes=size;
        assert(ria_expert_matrix_validate(&matrix,&error));
        float input=4,result[2];
        assert(ria_expert_cpu_projection(context,&matrix,&input,1,1,result,2,true,&error));
        assert(result[0]==8 && result[1]==12);
        assert(munmap(memory,size)==0);
    }
    ria_expert_cpu_destroy(context);
#endif
}
static void fragment_coverage(void) {
    for (unsigned bits=4;bits<=16;bits*=2) {
        unsigned width=256/bits;bool a[16][64]={{false}},b[64][8]={{false}},c[16][8]={{false}};
        for (unsigned lane=0;lane<32;++lane) {
            for (unsigned reg=0;reg<4;++reg) for (unsigned element=0;element<32/bits;++element) {
                unsigned row=ria_num_mma_a_row(lane,reg),col=ria_num_mma_a_col(lane,reg,element,bits);
                assert(row<16 && col<width && !a[row][col]);a[row][col]=true;
            }
            for (unsigned reg=0;reg<2;++reg) for (unsigned element=0;element<32/bits;++element) {
                unsigned row=ria_num_mma_b_row(lane,reg,element,bits),col=lane/4;
                assert(row<width && col<8 && !b[row][col]);b[row][col]=true;
            }
            for (unsigned element=0;element<4;++element) {
                unsigned row=ria_num_mma_c_row(lane,element),col=ria_num_mma_c_col(lane,element);
                assert(row<16 && col<8 && !c[row][col]);c[row][col]=true;
            }
        }
        for (unsigned row=0;row<16;++row) for (unsigned col=0;col<width;++col) assert(a[row][col]);
        for (unsigned row=0;row<width;++row) for (unsigned col=0;col<8;++col) assert(b[row][col]);
        for (unsigned row=0;row<16;++row) for (unsigned col=0;col<8;++col) assert(c[row][col]);
    }
}
static void borrowed_workspace(void) {
    ria_error error={0};uint64_t bytes;ria_expert_cpu *context=NULL;
    assert(ria_expert_cpu_required_bytes(3,5,2,&bytes,&error));
    uint8_t *arena=malloc((size_t)bytes+1);assert(arena);memset(arena,0x55,(size_t)bytes+1);
    assert(!ria_expert_cpu_create_in(3,5,2,arena+1,bytes,&context,&error));assert(context==NULL);
    assert(!ria_expert_cpu_create_in(3,5,2,arena,bytes-1,&context,&error));assert(context==NULL);
    assert(ria_expert_cpu_create_in(3,5,2,arena,bytes,&context,&error));
    for (uint64_t i=0;i<bytes;++i) assert(arena[i]==0x55); /* no premature first touch */
    uint8_t weights[12];for (unsigned i=0;i<6;++i) bf16_bytes(weights+i*2,(float)i-2);
    ria_expert_matrix matrix=bf16_matrix(weights,2,3);float input[3]={1,2,3},output[2];
    assert(ria_expert_cpu_projection(context,&matrix,input,1,3,output,2,true,&error));assert(output[0]==-4 && output[1]==14);
    ria_expert_cpu_destroy(context);memset(arena,0xaa,(size_t)bytes);free(arena); /* caller still owns arena */
    assert(!ria_expert_cpu_required_bytes(UINT64_MAX,1,1,&bytes,&error));
}
int main(void) { numeric_codes();quantizers();projection_and_expert();wide_strides();fragment_coverage();borrowed_workspace();puts("RIA expert numeric/quantizer/graph boundary fixtures passed");return 0; }
