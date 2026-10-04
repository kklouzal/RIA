#include "expert_cuda.h"
#include <cuda_runtime.h>
#include "numeric.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

struct ria_expert_cuda {
    int device;
    uint64_t input,intermediate,output,rows,width,tile_rows,workspace_bytes,workspace_budget,max_pitch;
    cudaStream_t stream;
    float *input_values,*quantized,*factors,*gate,*up,*output_values,*coefficients;
    uint8_t *tile_values,*tile_scales,*activation_codes,*activation_scales;
    uint8_t *pinned;
    uint64_t pinned_bytes;
    int *device_error;
};
uint64_t ria_expert_cuda_metadata_bytes(void) { return sizeof(ria_expert_cuda); }
uint64_t ria_expert_cuda_device_bytes(const ria_expert_cuda *c) { return c ? c->workspace_bytes-sizeof(*c) : 0; }
static bool failure(ria_error *e,int code,const char *message) {
    if (e) { e->code=code; (void)snprintf(e->message,sizeof(e->message),"%s",message); }
    return false;
}
static bool cuda_failure(cudaError_t status,ria_error *e,const char *operation) {
    if (status==cudaSuccess) return true;
    if (e) { e->code=RIA_EXECUTOR_ERROR; (void)snprintf(e->message,sizeof(e->message),"%s: %s",operation,cudaGetErrorString(status)); }
    return false;
}
bool ria_expert_cuda_device_require(int device,const char *expected_uuid,const char *expected_name,ria_error *e) {
    if (device!=0 || !expected_uuid || strlen(expected_uuid)!=40 ||
        strncmp(expected_uuid,"GPU-",4)!=0 || (expected_name && !*expected_name))
        return failure(e,RIA_INVALID_REQUEST,"invalid locked CUDA device identity");
    for (unsigned i=4;i<40;++i) {
        bool hyphen=i==12 || i==17 || i==22 || i==27;
        char c=expected_uuid[i];
        if (hyphen ? c!='-' : !((c>='0' && c<='9') || (c>='a' && c<='f')))
            return failure(e,RIA_INVALID_REQUEST,"locked CUDA UUID must be canonical lowercase");
    }
    int count=0;cudaDeviceProp p;
    if (!cuda_failure(cudaGetDeviceCount(&count),e,"read visible CUDA device count") ||
        !cuda_failure(cudaGetDeviceProperties(&p,device),e,"read locked CUDA device properties")) return false;
    if (count!=1 || p.major!=12 || p.minor!=0)
        return failure(e,RIA_UNSUPPORTED,"CUDA executor requires exactly one visible physical SM120 device");
    char uuid[41];unsigned char *u=(unsigned char *)p.uuid.bytes;
    (void)snprintf(uuid,sizeof(uuid),"GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
        u[0],u[1],u[2],u[3],u[4],u[5],u[6],u[7],u[8],u[9],u[10],u[11],u[12],u[13],u[14],u[15]);
    if (strcmp(uuid,expected_uuid)!=0 || (expected_name && strcmp(p.name,expected_name)!=0))
        return failure(e,RIA_INVALID_REQUEST,"physical CUDA UUID/product differs from admitted inventory");
    return true;
}
static __host__ __device__ uint64_t ceil_groups(uint64_t n,uint64_t group) { return n/group+(n%group!=0); }
static bool product(uint64_t a,uint64_t b,uint64_t *result) {
    if (b && a>UINT64_MAX/b) return false;
    *result=a*b; return true;
}
static bool allocation(ria_expert_cuda *c,void **pointer,uint64_t count,uint64_t size,ria_error *e) {
    uint64_t bytes;
    if (!product(count,size,&bytes) || bytes>SIZE_MAX || bytes>UINT64_MAX-c->workspace_bytes ||
        c->workspace_bytes>c->workspace_budget || bytes>c->workspace_budget-c->workspace_bytes)
        return failure(e,RIA_RESOURCE_LIMIT,"CUDA expert allocation exceeds admitted workspace");
    if (!cuda_failure(cudaMalloc(pointer,(size_t)bytes),e,"allocate expert workspace")) return false;
    c->workspace_bytes+=bytes; return true;
}
bool ria_expert_cuda_upload_bytes(ria_expert_cuda *c,void *device,const void *host,uint64_t bytes,ria_error *e) {
    if (!c || !device || !host || !c->pinned || bytes>SIZE_MAX)
        return failure(e,RIA_INVALID_REQUEST,"invalid pinned CUDA upload");
    for (uint64_t offset=0;offset<bytes;) {
        uint64_t count=bytes-offset<c->pinned_bytes ? bytes-offset : c->pinned_bytes;
        memcpy(c->pinned,(const uint8_t *)host+offset,(size_t)count);
        if (!cuda_failure(cudaMemcpyAsync((uint8_t *)device+offset,c->pinned,(size_t)count,cudaMemcpyHostToDevice,c->stream),e,"upload reusable pinned bytes") ||
            !cuda_failure(cudaStreamSynchronize(c->stream),e,"complete pinned upload before pool reuse")) return false;
        offset+=count;
    }
    return true;
}
bool ria_expert_cuda_download_bytes(ria_expert_cuda *c,const void *device,void *host,uint64_t bytes,ria_error *e) {
    if (!c || !device || !host || !c->pinned || bytes>SIZE_MAX)
        return failure(e,RIA_INVALID_REQUEST,"invalid pinned CUDA download");
    for (uint64_t offset=0;offset<bytes;) {
        uint64_t count=bytes-offset<c->pinned_bytes ? bytes-offset : c->pinned_bytes;
        if (!cuda_failure(cudaMemcpyAsync(c->pinned,(const uint8_t *)device+offset,(size_t)count,cudaMemcpyDeviceToHost,c->stream),e,"download reusable pinned bytes") ||
            !cuda_failure(cudaStreamSynchronize(c->stream),e,"complete pinned download before pool reuse")) return false;
        memcpy((uint8_t *)host+offset,c->pinned,(size_t)count);
        offset+=count;
    }
    return true;
}
static bool copy_rows(ria_expert_cuda *c,void *destination,uint64_t destination_pitch,
                      const void *source,uint64_t source_pitch,uint64_t width,uint64_t rows,
                      cudaMemcpyKind direction,ria_error *e,const char *operation) {
    (void)operation;
    /* Gather bytewise into a fixed pinned pool. Logical row groups retain their
     * original quantizer domain; only transport chunks change. 64-bit pitches
     * remain valid even above cudaMemcpy2D's implementation pitch ceiling. */
    if (direction!=cudaMemcpyHostToDevice && direction!=cudaMemcpyDeviceToHost)
        return failure(e,RIA_INVALID_REQUEST,"invalid expert transfer direction");
    if (!width || width>c->pinned_bytes) return failure(e,RIA_RESOURCE_LIMIT,"expert row exceeds pinned pool");
    uint64_t batch=c->pinned_bytes/width;
    for (uint64_t first=0;first<rows;first+=batch) {
        uint64_t count=rows-first<batch ? rows-first : batch;
        if (direction==cudaMemcpyHostToDevice) {
            for (uint64_t row=0;row<count;++row) memcpy(c->pinned+row*width,(const uint8_t *)source+(first+row)*source_pitch,(size_t)width);
            if (!cuda_failure(cudaMemcpy2DAsync((uint8_t *)destination+first*destination_pitch,(size_t)destination_pitch,c->pinned,(size_t)width,
                (size_t)width,(size_t)count,direction,c->stream),e,"upload pinned gathered rows") ||
                !cuda_failure(cudaStreamSynchronize(c->stream),e,"complete pinned gathered rows")) return false;
        } else {
            if (!cuda_failure(cudaMemcpy2DAsync(c->pinned,(size_t)width,(const uint8_t *)source+first*source_pitch,(size_t)source_pitch,
                (size_t)width,(size_t)count,direction,c->stream),e,"download pinned gathered rows") ||
                !cuda_failure(cudaStreamSynchronize(c->stream),e,"complete pinned gathered rows")) return false;
            for (uint64_t row=0;row<count;++row) memcpy((uint8_t *)destination+(first+row)*destination_pitch,c->pinned+row*width,(size_t)width);
        }
    }
    return true;
}
static __global__ void quantize_kernel(const float *input,float *quantized,float *factors,uint8_t *codes,uint8_t *scale_codes,
                                      uint64_t width,uint64_t group,uint64_t groups_per_row,
                                      uint64_t rows,float global,int profile,int *error) {
    uint64_t block=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
    uint64_t row=block/groups_per_row,base=(block%groups_per_row)*group;
    if (row>=rows || base>=width) return;
    float maximum=0;
    uint64_t end=width-base<group ? width : base+group;
    for (uint64_t k=base;k<end;++k) {
        float value=profile==RIA_EXPERT_F32 ? input[row*width+k] : ria_num_bf16(input[row*width+k]);
        if (!isfinite(value)) { atomicExch(error,1); value=0; }
        maximum=fmaxf(maximum,fabsf(value));
    }
    float factor=1,scale=1;uint8_t scale_code=0;
    if (profile==RIA_EXPERT_FP8) { scale_code=ria_num_fp8_scale(maximum);factor=scale=ria_num_ue8m0(scale_code); }
    else if (profile==RIA_EXPERT_NVFP4) { scale_code=ria_num_nvfp4_scale(maximum,global);scale=ria_num_e4m3(scale_code);factor=__fmul_rn(scale,global); }
    if (!isfinite(factor)) { atomicExch(error,2); factor=0; }
    factors[block]=scale;
    scale_codes[block]=scale_code;
    for (uint64_t k=base;k<end;++k) {
        float value=profile==RIA_EXPERT_F32 ? input[row*width+k] : ria_num_bf16(input[row*width+k]);
        if (!isfinite(value)) value=0;
        if (profile==RIA_EXPERT_FP8) { uint8_t code=ria_num_to_e4m3(value/factor);codes[row*width+k]=code;value=ria_num_e4m3(code); }
        else if (profile==RIA_EXPERT_NVFP4) { uint8_t code=ria_num_to_e2m1(factor>0 ? value/factor : 0);codes[row*width+k]=code;value=ria_num_e2m1(code); }
        else if (profile==RIA_EXPERT_BF16) { uint32_t raw=ria_num_bits(value);codes[(row*width+k)*2]=(uint8_t)(raw>>16);codes[(row*width+k)*2+1]=(uint8_t)(raw>>24); }
        quantized[row*width+k]=value;
    }
}
/* Four independent full warps per block. Shape is 16 input rows by eight
 * output channels; one bounded host channel tile is resident. FP8 finishes
 * each K32 raw dot before its two explicit FP32 scale casts. NVFP4 executes
 * one K16 domain in a zero-padded K64 instruction with REAL E4M3 metadata;
 * E2M1 products and two E4M3 factors fit exactly within FP32, so its scaled
 * partial is exact before the ordered group addition and final global alpha.
 * This avoids silently combining four separately rounded quantizer domains.
 * PTX reference: docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16864 */
template<unsigned Bits> static __global__ void native_projection_kernel(const uint8_t *values,const uint8_t *scales,
    const uint8_t *activation_codes,const uint8_t *activation_scales,const float *factors,float *output,
    uint64_t width,uint64_t count,uint64_t stride,uint64_t scale_stride,uint64_t output_width,uint64_t output_base,
    uint64_t rows,float global,float activation_global,bool round_bf16,int *error) {
#if __CUDA_ARCH__ >= 1200
    unsigned lane=threadIdx.x%32,warp=threadIdx.x/32;
    uint64_t n_base=((uint64_t)blockIdx.x*4+warp)*8,row_base=(uint64_t)blockIdx.y*16;
    if (n_base>=count) return;float totals[4]={0,0,0,0};
    const unsigned group=Bits==8 ? 32 : 16;
    uint64_t groups=ceil_groups(width,group);
    for (uint64_t base=0;base<width;base+=group) {
        uint32_t a[4],b[2];
#pragma unroll
        for (unsigned reg=0;reg<4;++reg) a[reg]=ria_num_mma_activation_fragment(activation_codes,width,rows,row_base,base,lane,reg,Bits,group);
#pragma unroll
        for (unsigned reg=0;reg<2;++reg) b[reg]=ria_num_mma_weight_fragment(values,width,count,n_base,base,stride,lane,reg,Bits,group);
        float partial[4]={0,0,0,0};
        if (Bits==16) {
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                         "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                         : "+f"(totals[0]),"+f"(totals[1]),"+f"(totals[2]),"+f"(totals[3])
                         : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
        } else if (Bits==8) {
            uint32_t unity=127;
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
                         "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %10, {0,0};"
                         : "+f"(partial[0]),"+f"(partial[1]),"+f"(partial[2]),"+f"(partial[3])
                         : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(unity));
#pragma unroll
            for (unsigned element=0;element<4;++element) {
                uint64_t row=row_base+ria_num_mma_c_row(lane,element),n=n_base+ria_num_mma_c_col(lane,element);
                if (row<rows && n<count) {
                    float act=factors[row*groups+base/group];
                    uint8_t code=scales[((output_base+n)/32-output_base/32)*scale_stride+base/group];
                    partial[element]=__fmul_rn(__fmul_rn(partial[element],act),ria_num_ue8m0(code));
                    totals[element]=__fadd_rn(totals[element],partial[element]);
                }
            }
        } else {
            /* Selector A {0,0}: quad lane 0 supplies row groupID and lane 1
             * supplies groupID+8; selector B {0,0}: lane 0 supplies column
             * groupID. All four scale bytes are used, but only K[0,16) has
             * nonzero values. Padding scales are zero, never uninitialized. */
            uint32_t as=ria_num_mma_nvfp4_scale_a(activation_scales,groups,rows,row_base,base/group,lane);
            uint32_t bs=ria_num_mma_nvfp4_scale_b(scales,scale_stride,count,n_base,base/group,lane);
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                         "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};"
                         : "+f"(partial[0]),"+f"(partial[1]),"+f"(partial[2]),"+f"(partial[3])
                         : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(as),"r"(bs));
#pragma unroll
            for (unsigned element=0;element<4;++element) totals[element]=__fadd_rn(totals[element],partial[element]);
        }
    }
#pragma unroll
    for (unsigned element=0;element<4;++element) {
        uint64_t row=row_base+ria_num_mma_c_row(lane,element),n=n_base+ria_num_mma_c_col(lane,element);
        if (row<rows && n<count) {
            float total=totals[element];if (Bits==4) total=__fmul_rn(total,__fmul_rn(global,activation_global));
            if (round_bf16) total=ria_num_bf16(total);if (!isfinite(total)) atomicExch(error,2);
            output[row*output_width+output_base+n]=total;
        }
    }
#else
    if (threadIdx.x==0) atomicExch(error,2);
#endif
}
static __global__ void fp32_projection_kernel(const uint8_t *values,const float *input,float *output,
    uint64_t width,uint64_t count,uint64_t stride,uint64_t output_width,uint64_t output_base,uint64_t rows,
    bool round_bf16,int *error) {
    uint64_t position=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;if (position>=rows*count) return;
    uint64_t row=position/count,n=position%count;float total=0;
    for (uint64_t k=0;k<width;++k) {
        const uint8_t *p=values+n*stride+k*4;
        uint32_t bits=(uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);
        total=__fmaf_rn(ria_num_float(bits),input[row*width+k],total);
    }
    if (round_bf16) total=ria_num_bf16(total);if (!isfinite(total)) atomicExch(error,2);
    output[row*output_width+output_base+n]=total;
}
static __global__ void gated_kernel(const float *gate,const float *up,const float *coefficients,float *result,
                                   uint64_t width,uint64_t rows,float clamp,int *error) {
    uint64_t k=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if (k>=rows*width) return;
    float coefficient=coefficients ? coefficients[k/width] : 1;
    if (!isfinite(coefficient) || coefficient<0) { atomicExch(error,1); coefficient=0; }
    float g=gate[k],u=up[k];
    if (clamp>0) { g=fminf(g,clamp); u=fmaxf(-clamp,fminf(u,clamp)); }
    float silu=g/(1.0f+expf(-g));
    float value=ria_num_bf16(__fmul_rn(__fmul_rn(silu,u),coefficient));
    if (!isfinite(value)) atomicExch(error,2);
    result[k]=value;
}
static bool launch_ok(ria_error *e,const char *operation) { return cuda_failure(cudaGetLastError(),e,operation); }
static bool finish(ria_expert_cuda *c,ria_error *e) {
    int result=0;
    if (!ria_expert_cuda_download_bytes(c,c->device_error,&result,sizeof(result),e)) return false;
    if (result) return failure(e,result==1 ? RIA_INVALID_REQUEST : RIA_EXECUTOR_ERROR,
                               result==1 ? "nonfinite CUDA input/coefficient" : "nonfinite CUDA expert result/scale");
    return true;
}
static bool projection(ria_expert_cuda *c,const ria_expert_matrix *m,const float *input,uint64_t rows,
                       float *output,bool round_bf16,ria_error *e,bool resident=false) {
    if (!c || !m || !input || !output || !rows || rows>c->rows || m->in_features>c->width ||
        m->out_features>(c->output>c->intermediate ? c->output : c->intermediate))
        return failure(e,RIA_RESOURCE_LIMIT,"CUDA projection exceeds admitted dimensions");
    /* This immutable matrix was fully validated by TensorStore. Validate its
     * descriptor again, without rescan of a many-GiB resident bank. */
    if (!ria_expert_matrix_descriptor_validate(m,e)) return false;
    uint64_t row_bytes=m->profile==RIA_EXPERT_NVFP4 ? ceil_groups(m->in_features,2) :
        m->profile==RIA_EXPERT_BF16 ? m->in_features*2 : m->profile==RIA_EXPERT_F32 ? m->in_features*4 : m->in_features;
    if (!m->values || m->out_features==0 || m->in_features==0 || m->value_row_stride<row_bytes ||
        m->profile<RIA_EXPERT_BF16 || m->profile>RIA_EXPERT_F32)
        return failure(e,RIA_INVALID_REQUEST,"invalid CUDA projection descriptor");
    uint64_t group=(m->profile==RIA_EXPERT_BF16 || m->profile==RIA_EXPERT_F32) ? m->in_features : m->profile==RIA_EXPERT_FP8 ? 32 : 16;
    uint64_t ngroups=ceil_groups(m->in_features,group),blocks=rows*ngroups;
    quantize_kernel<<<(unsigned)ceil_groups(blocks,128),128,0,c->stream>>>(
        input,c->quantized,c->factors,c->activation_codes,c->activation_scales,m->in_features,group,ngroups,rows,m->activation_global_scale,m->profile,c->device_error);
    if (!launch_ok(e,"quantize expert activations")) return false;
    for (uint64_t base=0;base<m->out_features;base+=c->tile_rows) {
        uint64_t count=m->out_features-base<c->tile_rows ? m->out_features-base : c->tile_rows;
        if (!resident && !copy_rows(c,c->tile_values,row_bytes,m->values+base*m->value_row_stride,
                       m->value_row_stride,row_bytes,count,cudaMemcpyHostToDevice,e,"stage packed projection tile")) return false;
        uint64_t scale_width=m->profile==RIA_EXPERT_FP8 ? ceil_groups(m->in_features,32) : ceil_groups(m->in_features,16);
        const uint8_t *values=resident ? m->values+base*m->value_row_stride : c->tile_values;
        const uint8_t *scales=resident && m->scales ? m->scales+(m->profile==RIA_EXPERT_FP8 ? base/32 : base)*m->scale_row_stride : c->tile_scales;
        uint64_t value_stride=resident ? m->value_row_stride : row_bytes,scale_stride=resident ? m->scale_row_stride : scale_width;
        if (m->profile==RIA_EXPERT_FP8 || m->profile==RIA_EXPERT_NVFP4) {
            uint64_t scale_base=m->profile==RIA_EXPERT_FP8 ? base/32 : base;
            uint64_t scale_rows=m->profile==RIA_EXPERT_FP8 ? ceil_groups(base%32+count,32) : count;
            if (!m->scales || m->scale_row_stride<scale_width) return failure(e,RIA_INVALID_REQUEST,"invalid CUDA scale view");
            if (!resident && !copy_rows(c,c->tile_scales,scale_width,m->scales+scale_base*m->scale_row_stride,
                           m->scale_row_stride,scale_width,scale_rows,cudaMemcpyHostToDevice,e,"stage projection scale tile")) return false;
        }
        if (m->profile==RIA_EXPERT_F32) {
            fp32_projection_kernel<<<(unsigned)ceil_groups(rows*count,128),128,0,c->stream>>>(
                values,c->quantized,output,m->in_features,count,value_stride,m->out_features,base,rows,round_bf16,c->device_error);
        } else {
            dim3 grid((unsigned)ceil_groups(count,32),(unsigned)ceil_groups(rows,16));
#define NATIVE(Bits) native_projection_kernel<Bits><<<grid,128,0,c->stream>>>(values,scales,c->activation_codes,c->activation_scales,c->factors,output,m->in_features,count,value_stride,scale_stride,m->out_features,base,rows,m->weight_global_scale,m->activation_global_scale,round_bf16,c->device_error)
            if (m->profile==RIA_EXPERT_BF16) { NATIVE(16); }
            else if (m->profile==RIA_EXPERT_FP8) { NATIVE(8); }
            else { NATIVE(4); }
#undef NATIVE
        }
        if (!launch_ok(e,"execute native packed projection tile")) return false;
    }
    return true;
}
bool ria_expert_cuda_resident_create(ria_expert_cuda *c,const ria_expert *source,uint64_t budget,
                                    ria_expert_cuda_resident *view,ria_error *e) {
    if (!c || !view || view->owner || !source || !ria_expert_validate(source,e))
        return failure(e,RIA_INVALID_REQUEST,"invalid startup resident expert view");
    uint64_t bytes;
    if (!ria_expert_cuda_resident_required_bytes(source,&bytes,e)) return false;
    if (bytes>budget || bytes>SIZE_MAX) return failure(e,RIA_RESOURCE_LIMIT,"immutable local expert exceeds device cache budget");
    uint8_t *arena=NULL;
    if (!cuda_failure(cudaSetDevice(c->device),e,"select resident expert device") ||
        !cuda_failure(cudaMalloc((void **)&arena,(size_t)bytes),e,"allocate immutable local expert population")) return false;
    view->owner=c;view->bytes=bytes;view->expert=*source;uint8_t *cursor=arena;
    ria_expert_matrix *to[3]={&view->expert.gate,&view->expert.up,&view->expert.down};
    const ria_expert_matrix *from[3]={&source->gate,&source->up,&source->down};
    for (unsigned i=0;i<3;++i) {
        ria_expert_matrix *m=to[i];uint64_t stride=m->profile==RIA_EXPERT_NVFP4 ? ceil_groups(m->in_features,2) : m->in_features*(m->profile==RIA_EXPERT_BF16 ? 2 : 1);
        m->values=cursor;m->value_row_stride=stride;m->values_bytes=stride*m->out_features;cursor+=m->values_bytes;
        if (m->scales) { m->scale_row_stride=ceil_groups(m->in_features,m->profile==RIA_EXPERT_FP8 ? 32 : 16);
            m->scales_bytes=m->scale_row_stride*(m->profile==RIA_EXPERT_FP8 ? ceil_groups(m->out_features,32) : m->out_features);
            m->scales=cursor;cursor+=m->scales_bytes; }
    }
    bool ok=true;
    for (unsigned i=0;ok && i<3;++i) {
        ria_expert_matrix *m=to[i];const ria_expert_matrix *s=from[i];
        ok=copy_rows(c,(void *)m->values,m->value_row_stride,s->values,s->value_row_stride,m->value_row_stride,m->out_features,cudaMemcpyHostToDevice,e,"initialize local expert values");
        if (ok && m->scales) ok=copy_rows(c,(void *)m->scales,m->scale_row_stride,s->scales,s->scale_row_stride,m->scale_row_stride,
            m->profile==RIA_EXPERT_FP8 ? ceil_groups(m->out_features,32) : m->out_features,cudaMemcpyHostToDevice,e,"initialize local expert scales");
    }
    if (!ok) { ria_error cleanup={0,{0}};
        if (!ria_expert_cuda_resident_destroy(view,&cleanup)) {
            fprintf(stderr,"CUDA resident construction cleanup failed; retaining owned buffers (primary=%d cleanup=%d)\n",e ? e->code : 0,cleanup.code);
            _Exit(e && e->code ? e->code : cleanup.code ? cleanup.code : RIA_EXECUTOR_ERROR);
        }
        return false;
    }
    return true;
}
bool ria_expert_cuda_resident_destroy(ria_expert_cuda_resident *view,ria_error *e) {
    if (!view || !view->owner) return true;
    /* A failed CUDA wait is not a usable completion proof. Retain every
     * borrowed/owned address for the process owner's fail-stop policy. */
    if (!cuda_failure(cudaSetDevice(view->owner->device),e,"select local expert cleanup device") ||
        !cuda_failure(cudaStreamSynchronize(view->owner->stream),e,"drain local expert before release") ||
        !cuda_failure(cudaFree((void *)view->expert.gate.values),e,"release immutable local expert population")) return false;
    memset(view,0,sizeof(*view));return true;
}
uint64_t ria_expert_cuda_resident_bytes(const ria_expert_cuda_resident *view) { return view ? view->bytes : 0; }
bool ria_expert_cuda_evaluate_device(ria_expert_cuda *c,const ria_expert *source,const ria_expert_cuda_resident *resident,
                                    const float *input,float coefficient,float *output,ria_error *e) {
    const ria_expert *expert=resident ? &resident->expert : source;
    if (!c || !expert || !input || !output || (resident && resident->owner!=c) || !isfinite(coefficient) || coefficient<0 ||
        !expert->gate.in_features || !expert->gate.out_features || !expert->down.out_features ||
        expert->gate.in_features>c->input || expert->gate.out_features>c->intermediate || expert->down.out_features>c->output ||
        expert->gate.profile<RIA_EXPERT_BF16 || expert->gate.profile>RIA_EXPERT_NVFP4 || expert->gate.profile!=expert->up.profile || expert->gate.profile!=expert->down.profile ||
        expert->gate.in_features!=expert->up.in_features || expert->gate.out_features!=expert->up.out_features ||
        expert->gate.out_features!=expert->down.in_features ||
        !isfinite(expert->clamp) || expert->clamp<0)
        return failure(e,RIA_INVALID_REQUEST,"invalid local CUDA expert or coefficient");
    if (!cuda_failure(cudaSetDevice(c->device),e,"select local expert device") ||
        !cuda_failure(cudaMemsetAsync(c->device_error,0,sizeof(int),c->stream),e,"clear local expert status")) return false;
    bool ok=ria_expert_cuda_upload_bytes(c,c->coefficients,&coefficient,sizeof(coefficient),e) &&
        projection(c,&expert->gate,input,1,c->gate,true,e,resident!=NULL) && projection(c,&expert->up,input,1,c->up,true,e,resident!=NULL);
    if (ok) {
        gated_kernel<<<(unsigned)ceil_groups(expert->gate.out_features,128),128,0,c->stream>>>(c->gate,c->up,c->coefficients,c->input_values,
            expert->gate.out_features,1,expert->clamp,c->device_error);
        ok=launch_ok(e,"execute local coefficient before down quantization") && projection(c,&expert->down,c->input_values,1,output,true,e,resident!=NULL);
    }
    if (!ok) { ria_error drain;(void)finish(c,&drain);return false; }
    return finish(c,e);
}
bool ria_expert_cuda_create(int device,uint64_t input,uint64_t intermediate,uint64_t output,uint64_t rows,
                           uint64_t tile_rows,ria_expert_cuda **out,ria_error *e) {
    return ria_expert_cuda_create_limited(device,input,intermediate,output,rows,tile_rows,UINT64_MAX,out,e);
}
bool ria_expert_cuda_create_limited(int device,uint64_t input,uint64_t intermediate,uint64_t output,uint64_t rows,
                                   uint64_t tile_rows,uint64_t workspace_budget,ria_expert_cuda **out,ria_error *e) {
    return ria_expert_cuda_create_pooled(device,input,intermediate,output,rows,tile_rows,workspace_budget,UINT64_MAX,out,e);
}
bool ria_expert_cuda_create_pooled(int device,uint64_t input,uint64_t intermediate,uint64_t output,uint64_t rows,
                                  uint64_t tile_rows,uint64_t workspace_budget,uint64_t pinned_budget,ria_expert_cuda **out,ria_error *e) {
    if (!out) return failure(e,RIA_INVALID_REQUEST,"missing CUDA context result");
    *out=NULL;
    if (!input || !intermediate || !output || !rows || rows>65535 || !tile_rows || tile_rows>4096 ||
        input>UINT32_MAX || intermediate>UINT32_MAX || output>UINT32_MAX)
        return failure(e,RIA_INVALID_REQUEST,"invalid admitted CUDA expert bounds");
    if (workspace_budget<sizeof(ria_expert_cuda))
        return failure(e,RIA_RESOURCE_LIMIT,"CUDA expert owner exceeds admitted workspace");
    ria_expert_cuda *c=(ria_expert_cuda *)calloc(1,sizeof(*c));
    if (!c) return failure(e,RIA_RESOURCE_LIMIT,"CUDA expert context allocation failed");
    c->device=device; c->input=input; c->intermediate=intermediate; c->output=output;
    c->rows=rows; c->width=input>intermediate ? input : intermediate; c->tile_rows=tile_rows;
    c->workspace_bytes=sizeof(*c);c->workspace_budget=workspace_budget;
    if (!ria_expert_cuda_pinned_required_bytes(input,intermediate,output,rows,tile_rows,&c->pinned_bytes,e) || c->pinned_bytes>pinned_budget) {
        free(c);return failure(e,RIA_RESOURCE_LIMIT,"expert pinned pool exceeds admitted pinned budget");
    }
    cudaDeviceProp properties;
    bool ok=cuda_failure(cudaSetDevice(device),e,"select expert CUDA device") &&
        cuda_failure(cudaGetDeviceProperties(&properties,device),e,"read CUDA copy/grid limits");
    if (ok) {
        c->max_pitch=(uint64_t)properties.memPitch;
        if (properties.major!=12 || properties.minor!=0)
            ok=failure(e,RIA_UNSUPPORTED,"production native projection image requires SM120a/physical CC12.0");
        else if (ceil_groups(rows*ceil_groups(c->width,16),128)>(uint64_t)properties.maxGridSize[0])
            ok=failure(e,RIA_RESOURCE_LIMIT,"admitted CUDA quantizer exceeds grid limit");
    }
    ok=ok &&
        cuda_failure(cudaStreamCreateWithFlags(&c->stream,cudaStreamNonBlocking),e,"create expert stream") &&
        cuda_failure(cudaHostAlloc((void **)&c->pinned,(size_t)c->pinned_bytes,cudaHostAllocDefault),e,"allocate bounded pinned expert transfer pool");
    if (ok && madvise(c->pinned,(size_t)c->pinned_bytes,MADV_DONTDUMP)!=0)
        ok=failure(e,RIA_RESOURCE_LIMIT,"protect pinned expert transfer pool from core dumps");
    ok=ok &&
        allocation(c,(void **)&c->input_values,rows*c->width,sizeof(float),e) &&
        allocation(c,(void **)&c->quantized,rows*c->width,sizeof(float),e) &&
        allocation(c,(void **)&c->factors,rows*ceil_groups(c->width,16),sizeof(float),e) &&
        allocation(c,(void **)&c->activation_codes,rows*c->width,2,e) &&
        allocation(c,(void **)&c->activation_scales,rows*ceil_groups(c->width,16),1,e) &&
        allocation(c,(void **)&c->gate,rows*intermediate,sizeof(float),e) &&
        allocation(c,(void **)&c->up,rows*intermediate,sizeof(float),e) &&
        allocation(c,(void **)&c->output_values,rows*output>3 ? rows*output : 3,sizeof(float),e) &&
        allocation(c,(void **)&c->coefficients,rows,sizeof(float),e) &&
        allocation(c,(void **)&c->tile_values,tile_rows*c->width,4,e) &&
        allocation(c,(void **)&c->tile_scales,tile_rows*ceil_groups(c->width,16),1,e) &&
        allocation(c,(void **)&c->device_error,1,sizeof(int),e);
    if (!ok) { ria_error cleanup={0,{0}};
        if (!ria_expert_cuda_destroy(c,&cleanup)) {
            fprintf(stderr,"CUDA expert construction cleanup failed; retaining owned buffers (primary=%d cleanup=%d)\n",e ? e->code : 0,cleanup.code);
            _Exit(e && e->code ? e->code : cleanup.code ? cleanup.code : RIA_EXECUTOR_ERROR);
        }
        return false;
    }
    *out=c; return true;
}
bool ria_expert_cuda_destroy(ria_expert_cuda *c,ria_error *e) {
    if (!c) return true;
    if (!cuda_failure(cudaSetDevice(c->device),e,"select CUDA device for cleanup") ||
        (c->stream && !cuda_failure(cudaStreamSynchronize(c->stream),e,"drain expert stream"))) return false;
#define RELEASE(member) do { if (c->member && !cuda_failure(cudaFree(c->member),e,"release expert workspace")) return false;c->member=NULL; } while (0)
    RELEASE(input_values);RELEASE(quantized);RELEASE(factors);RELEASE(activation_codes);RELEASE(activation_scales);
    RELEASE(gate);RELEASE(up);RELEASE(output_values);RELEASE(coefficients);RELEASE(tile_values);RELEASE(tile_scales);RELEASE(device_error);
#undef RELEASE
    if (c->stream && !cuda_failure(cudaStreamDestroy(c->stream),e,"destroy expert stream")) return false;
    c->stream=NULL;
    if (c->pinned && !cuda_failure(cudaFreeHost(c->pinned),e,"release bounded pinned transfer pool")) return false;
    c->pinned=NULL;free(c);return true;
}
uint64_t ria_expert_cuda_workspace_bytes(const ria_expert_cuda *c) { return c ? c->workspace_bytes : 0; }
uint64_t ria_expert_cuda_pinned_bytes(const ria_expert_cuda *c) { return c ? c->pinned_bytes : 0; }
void *ria_expert_cuda_stream(ria_expert_cuda *c) { return c ? (void *)c->stream : NULL; }
bool ria_expert_cuda_projection(ria_expert_cuda *c,const ria_expert_matrix *m,const float *input,uint64_t rows,
                               float *output,bool round_bf16,ria_error *e) {
    if (!c) return failure(e,RIA_INVALID_REQUEST,"missing CUDA expert context");
    if (!cuda_failure(cudaSetDevice(c->device),e,"select projection device") ||
        !cuda_failure(cudaMemsetAsync(c->device_error,0,sizeof(int),c->stream),e,"clear projection status")) return false;
    bool ok=projection(c,m,input,rows,output,round_bf16,e);
    if (!ok) { ria_error drain; (void)finish(c,&drain); return false; }
    return finish(c,e);
}
bool ria_expert_cuda_evaluate(ria_expert_cuda *c,const ria_expert *expert,const float *input,uint64_t rows,
                             uint64_t input_stride,const float *coefficients,float *output,uint64_t output_stride,ria_error *e) {
    if (!c || !expert || !input || !output || !rows || rows>c->rows ||
        !expert->gate.in_features || !expert->gate.out_features || !expert->down.out_features || input_stride<expert->gate.in_features ||
        output_stride<expert->down.out_features || expert->gate.in_features>c->input || expert->gate.out_features>c->intermediate ||
        expert->down.out_features>c->output || expert->gate.profile<RIA_EXPERT_BF16 || expert->gate.profile>RIA_EXPERT_NVFP4 ||
        expert->up.profile!=expert->gate.profile || expert->down.profile!=expert->gate.profile ||
        expert->gate.out_features!=expert->up.out_features || expert->gate.in_features!=expert->up.in_features ||
        expert->gate.out_features!=expert->down.in_features || !isfinite(expert->clamp) || expert->clamp<0 ||
        input_stride>SIZE_MAX/sizeof(float) || output_stride>SIZE_MAX/sizeof(float) ||
        rows>SIZE_MAX/sizeof(float)/input_stride || rows>SIZE_MAX/sizeof(float)/output_stride)
        return failure(e,RIA_INVALID_REQUEST,"invalid CUDA selected-expert evaluation");
    if (!cuda_failure(cudaSetDevice(c->device),e,"select expert device") ||
        !cuda_failure(cudaMemsetAsync(c->device_error,0,sizeof(int),c->stream),e,"clear expert status")) return false;
    bool ok=copy_rows(c,c->input_values,expert->gate.in_features*sizeof(float),input,input_stride*sizeof(float),
                      expert->gate.in_features*sizeof(float),rows,cudaMemcpyHostToDevice,e,"upload expert input");
    if (ok && coefficients) ok=ria_expert_cuda_upload_bytes(c,c->coefficients,coefficients,rows*sizeof(float),e);
    if (ok) ok=projection(c,&expert->gate,c->input_values,rows,c->gate,true,e) &&
               projection(c,&expert->up,c->input_values,rows,c->up,true,e);
    if (ok) {
        gated_kernel<<<(unsigned)ceil_groups(rows*expert->gate.out_features,128),128,0,c->stream>>>(
            c->gate,c->up,coefficients ? c->coefficients : NULL,c->input_values,expert->gate.out_features,rows,expert->clamp,c->device_error);
        ok=launch_ok(e,"execute clamped expert activation") && projection(c,&expert->down,c->input_values,rows,c->output_values,true,e);
    }
    if (ok) ok=copy_rows(c,output,output_stride*sizeof(float),c->output_values,expert->down.out_features*sizeof(float),
                        expert->down.out_features*sizeof(float),rows,cudaMemcpyDeviceToHost,e,"download expert contributions");
    if (!ok) { ria_error drain; (void)finish(c,&drain); return false; }
    return finish(c,e);
}

static __global__ void native_probe_kernel(float *results) {
#if __CUDA_ARCH__ >= 1200
    float d0=0,d1=0,d2=0,d3=0;
    uint32_t one4=UINT32_C(0x22222222),one8=UINT32_C(0x38383838);
    uint32_t fp4_scales=UINT32_C(0x38383838),fp8_scales=UINT32_C(0x7f7f7f7f);
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,%1,%2,%3}, {%4,%4,%4,%4}, {%4,%4}, {%0,%1,%2,%3}, %5, {0,0}, %5, {0,0};"
                 : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3) : "r"(one4),"r"(fp4_scales));
    if (threadIdx.x==0) results[0]=d0;
    d0=0;d1=0;d2=0;d3=0;
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%4,%4,%4}, {%4,%4}, {%0,%1,%2,%3}, %5, {0,0}, %5, {0,0};"
                 : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3) : "r"(one8),"r"(fp8_scales));
    if (threadIdx.x==0) results[1]=d0;
    d0=0;d1=0;d2=0;d3=0;
    uint32_t one16=UINT32_C(0x3f803f80);
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%4,%4,%4}, {%4,%4}, {%0,%1,%2,%3};"
                 : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3) : "r"(one16));
    if (threadIdx.x==0) results[2]=d0;
#else
    if (threadIdx.x==0) { results[0]=NAN;results[1]=NAN;results[2]=NAN; }
#endif
}
bool ria_expert_cuda_native_probe(ria_expert_cuda *c,float results[3],ria_error *e) {
    if (!c || !results) return failure(e,RIA_INVALID_REQUEST,"missing native probe context/result");
    if (!cuda_failure(cudaSetDevice(c->device),e,"select probe device")) return false;
    cudaDeviceProp property;
    if (!cuda_failure(cudaGetDeviceProperties(&property,c->device),e,"read probe architecture")) return false;
    if (property.major!=12 || property.minor!=0) return failure(e,RIA_UNSUPPORTED,"native probe requires qualified SM120 image/device");
    native_probe_kernel<<<1,32,0,c->stream>>>(c->output_values);
    bool ok=launch_ok(e,"launch native low-bit/BF16 probe") &&
        ria_expert_cuda_download_bytes(c,c->output_values,results,3*sizeof(float),e);
    cudaError_t status=cudaStreamSynchronize(c->stream);
    if (!ok || !cuda_failure(status,e,"complete native probe")) return false;
    if (results[0]!=64.0f || results[1]!=32.0f || results[2]!=16.0f)
        return failure(e,RIA_EXECUTOR_ERROR,"native block-scale/BF16 probe arithmetic mismatch");
    return true;
}
