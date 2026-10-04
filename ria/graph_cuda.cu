#include "graph_cuda.h"
#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#include "numeric.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct ria_graph_cuda {
    ria_graph_options options;
    ria_expert_cuda *projection;
    cudaStream_t stream;
    float *buffers[RIA_G_BUFFER_COUNT],*weights,*frequencies,*scores,*sorted_scores,*block_scores;
    uint32_t *positions,*sorted_positions,*top_positions;
    uint16_t *route_ids;
    float *route_coefficients;
    uint8_t *packed,*candidate_blocks;
    int *error;
    void *sort_workspace;
    size_t sort_bytes;
    uint64_t bytes,packed_bytes;
};
uint64_t ria_graph_cuda_metadata_bytes(void) { return sizeof(ria_graph_cuda)+ria_expert_cuda_metadata_bytes(); }
uint64_t ria_graph_cuda_device_bytes(const ria_graph_cuda *c) { return c ? c->bytes-ria_expert_cuda_metadata_bytes() : 0; }
static bool check(cudaError_t status,ria_error *e,const char *operation) {
    return status==cudaSuccess || ria_fail(e,RIA_EXECUTOR_ERROR,"%s: %s",operation,cudaGetErrorString(status));
}
static uint64_t ceil_div(uint64_t value,uint64_t divisor) { return value/divisor+(value%divisor!=0); }
static bool allocate(ria_graph_cuda *c,void **out,uint64_t count,uint64_t size,ria_error *e) {
    uint64_t bytes,total;
    if (!ria_u64_mul(count,size,&bytes) || bytes>SIZE_MAX || !ria_u64_add(c->bytes,bytes,&total) || total>c->options.device_budget)
        return ria_fail(e,RIA_RESOURCE_LIMIT,"graph CUDA workspace exceeds admitted budget");
    if (!check(cudaMalloc(out,(size_t)bytes),e,"allocate graph CUDA workspace")) return false;
    c->bytes=total;return true;
}
static bool launched(ria_error *e,const char *op) { return check(cudaGetLastError(),e,op); }
static bool finish(ria_graph_cuda *c,ria_error *e) {
    int result=0;
    if (!ria_expert_cuda_download_bytes(c->projection,c->error,&result,sizeof(result),e)) return false;
    return !result || ria_fail(e,RIA_EXECUTOR_ERROR,"graph CUDA numerical/boundary invariant failed (%d)",result);
}
static __global__ void tensor_decode(const uint8_t *source,float *destination,uint64_t count,int dtype,int *error) {
    uint64_t i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;if (i>=count) return;
    const uint8_t *p=source+i*(dtype==1 ? 2 : 4);
    uint32_t bits=(uint32_t)p[0]|((uint32_t)p[1]<<8);
    bits=dtype==1 ? bits<<16 : bits|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);
    float value=ria_num_float(bits);if (!isfinite(value)) atomicExch(error,1);destination[i]=value;
}
static __global__ void norm_kernel(float *values,const float *weight,uint64_t width,bool layer_norm,int *error) {
    __shared__ float sums[256],squares[256],mean,inverse;
    uint64_t row=blockIdx.x;float sum=0,square=0;
    for (uint64_t k=threadIdx.x;k<width;k+=blockDim.x) { float value=values[row*width+k];sum=__fadd_rn(sum,value);square=__fmaf_rn(value,value,square); }
    sums[threadIdx.x]=sum;squares[threadIdx.x]=square;__syncthreads();
    for (unsigned offset=128;offset;offset>>=1) {
        if (threadIdx.x<offset) { sums[threadIdx.x]+=sums[threadIdx.x+offset];squares[threadIdx.x]+=squares[threadIdx.x+offset]; }__syncthreads();
    }
    if (threadIdx.x==0) {
        mean=layer_norm ? sums[0]/(float)width : 0;
        float variance=squares[0]/(float)width;
        if (layer_norm) variance=fmaxf(0,variance-mean*mean);
        inverse=rsqrtf(variance+(layer_norm ? 1e-6f : 1e-20f));
        if (!isfinite(inverse)) atomicExch(error,2);
    }__syncthreads();
    for (uint64_t k=threadIdx.x;k<width;k+=blockDim.x) {
        float value=ria_num_bf16(((values[row*width+k]-mean)*inverse)*weight[k]);
        if (!isfinite(value)) atomicExch(error,2);values[row*width+k]=value;
    }
}
static __global__ void expand_kernel(const float *embedding,float *h) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if (i<20480) h[i]=ria_num_bf16(embedding[i%5120]);
}
static __global__ void initial_pre(float *pre) { if (threadIdx.x<4) pre[threadIdx.x]=threadIdx.x==0 ? 1 : 0; }
static __global__ void mhc_kernel(const float *h,float *mix,const float *scale,const float *base,float *pre,float *post,float *comb,int *error) {
    if (threadIdx.x) return;
    float square=0;for (unsigned i=0;i<20480;++i) square=__fmaf_rn(h[i],h[i],square);
    float inverse=rsqrtf(square/20480.0f+1e-20f);
    for (unsigned i=0;i<24;++i) mix[i]=mix[i]*inverse;
    for (unsigned i=0;i<4;++i) { pre[i]=1/(1+expf(-(mix[i]*scale[0]+base[i])))+1e-6f;post[i]=2/(1+expf(-(mix[i+4]*scale[1]+base[i+4]))); }
    for (unsigned row=0;row<4;++row) {
        float maximum=-INFINITY,sum=0;
        for (unsigned column=0;column<4;++column) { unsigned i=row*4+column;comb[i]=mix[i+8]*scale[2]+base[i+8];maximum=fmaxf(maximum,comb[i]); }
        for (unsigned column=0;column<4;++column) { unsigned i=row*4+column;comb[i]=expf(comb[i]-maximum);sum+=comb[i]; }
        for (unsigned column=0;column<4;++column) comb[row*4+column]=comb[row*4+column]/sum+1e-6f;
    }
    for (unsigned iteration=0;iteration<20;++iteration) {
        if (iteration) for (unsigned row=0;row<4;++row) {
            float sum=1e-6f;for (unsigned column=0;column<4;++column) sum+=comb[row*4+column];
            for (unsigned column=0;column<4;++column) comb[row*4+column]/=sum;
        }
        for (unsigned column=0;column<4;++column) {
            float sum=1e-6f;for (unsigned row=0;row<4;++row) sum+=comb[row*4+column];
            for (unsigned row=0;row<4;++row) comb[row*4+column]/=sum;
        }
    }
    for (unsigned i=0;i<16;++i) if (!isfinite(comb[i])) atomicExch(error,2);
}
static __global__ void collapse_kernel(const float *h,const float *pre,float *output) {
    unsigned k=blockIdx.x*blockDim.x+threadIdx.x;if (k>=5120) return;float total=0;
    for (unsigned stream=0;stream<4;++stream) total=__fadd_rn(total,__fmul_rn(pre[stream],h[stream*5120+k]));output[k]=ria_num_bf16(total);
}
static __global__ void post_kernel(const float *x,const float *residual,const float *post,const float *comb,float *h,int *error) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if (i>=20480) return;unsigned stream=i/5120,k=i%5120;float sum=0;
    /* Source torch.sum(comb.unsqueeze(-1)*residual.unsqueeze(-2),dim=2)
     * contracts the INPUT stream j: comb[j,output_stream], not its transpose. */
    for (unsigned j=0;j<4;++j) sum=__fadd_rn(sum,__fmul_rn(comb[j*4+stream],residual[j*5120+k]));
    float value=ria_num_bf16(__fadd_rn(__fmul_rn(post[stream],x[k]),sum));
    if (!isfinite(value)) atomicExch(error,2);h[i]=value;
}
static __global__ void frequency_kernel(float *frequencies) {
    unsigned i=threadIdx.x;if (i>=32) return;
    frequencies[i]=1/powf(10000.0f,(float)(2*i)/64);
    float frequency=1/powf(160000.0f,(float)(2*i)/64);
    double pi=3.14159265358979323846;
    int low=(int)floor(64*log(65536/(32*2*pi))/(2*log(160000.0)));
    int high=(int)ceil(64*log(65536/(2*pi))/(2*log(160000.0)));
    float ramp=fminf(1,fmaxf(0,((float)i-(float)low)/(float)(high-low)));
    frequencies[32+i]=frequency/16*ramp+frequency*(1-ramp);
}
static __global__ void rope_kernel(float *values,uint64_t rows,uint64_t width,uint64_t position,const float *frequency,bool inverse) {
    uint64_t i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;if (i>=rows*32) return;
    uint64_t offset=(i/32)*width+width-64+2*(i%32);
    float angle=(float)position*frequency[i%32],sine,cosine;sincosf(angle,&sine,&cosine);if (inverse) sine=-sine;
    float a=values[offset],b=values[offset+1];
    values[offset]=ria_num_bf16(a*cosine-b*sine);values[offset+1]=ria_num_bf16(a*sine+b*cosine);
}
static __device__ uint64_t packed_stride(unsigned rep,uint64_t width) { return rep==1 ? width+width/32 : width/2+width/(rep==2 ? 16 : 32); }
static __global__ void pack_kernel(float *mutable_values,const float *values,uint8_t *output,uint64_t width,unsigned rep,bool inplace,int *error) {
    uint64_t group=rep==2 ? 16 : 32,block=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x,base=block*group;
    if (base>=width) return;float maximum=0;
    for (uint64_t k=base;k<base+group;++k) { float value=ria_num_bf16(values[k]);if (!isfinite(value)) atomicExch(error,1);maximum=fmaxf(maximum,fabsf(value)); }
    float scale;uint8_t code;
    if (rep==1) { code=ria_num_fp8_scale(maximum);scale=ria_num_ue8m0(code); }
    else if (rep==2) { code=ria_num_to_e4m3(fmaxf(maximum,6*0x1p-9f)/6);scale=ria_num_e4m3(code); }
    else {
        float bounded=fmaxf(maximum,6*0x1p-126f)*(1.0f/6);
        uint32_t bits=ria_num_bits(bounded);code=(uint8_t)(((bits>>23)&255)+((bits&0x7fffff)!=0));scale=ria_num_ue8m0(code);
    }
    if (!inplace) output[(rep==1 ? width : width/2)+block]=code;
    for (uint64_t k=base;k<base+group;++k) {
        float value=ria_num_bf16(values[k])/scale;
        uint8_t q=rep==1 ? ria_num_to_e4m3(value) : ria_num_to_e2m1(value);
        if (inplace) mutable_values[k]=ria_num_bf16((rep==1 ? ria_num_e4m3(q) : ria_num_e2m1(q))*scale);
        else if (rep==1) output[k]=q;
        else if (!(k&1)) {
            uint8_t next=ria_num_to_e2m1(ria_num_bf16(values[k+1])/scale);output[k/2]=(uint8_t)(q|(next<<4));
        }
    }
}
static __global__ void unpack_kernel(const uint8_t *source,float *destination,uint64_t rows,uint64_t width,unsigned rep,int *error) {
    uint64_t i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;if (i>=rows*width) return;
    uint64_t row=i/width,k=i%width,group=rep==2 ? 16 : 32;
    const uint8_t *p=source+row*packed_stride(rep,width);uint8_t scale_code=p[(rep==1 ? width : width/2)+k/group];
    float scale=rep==2 ? ria_num_e4m3(scale_code) : ria_num_ue8m0(scale_code);
    float value=rep==1 ? ria_num_e4m3(p[k]) : ria_num_e2m1((uint8_t)(p[k/2]>>(4*(k&1))));
    value=ria_num_bf16(value*scale);if (!isfinite(value) || scale<0) atomicExch(error,1);destination[i]=value;
}
static __global__ void pool_kernel(const float *values,const float *scores,unsigned ratio,float *latent) {
    unsigned k=blockIdx.x*blockDim.x+threadIdx.x;if (k>=512) return;float maximum=-INFINITY,total=0,sum=0;
    for (unsigned row=0;row<ratio;++row) maximum=fmaxf(maximum,scores[row*512+k]);
    for (unsigned row=0;row<ratio;++row) { float weight=expf(scores[row*512+k]-maximum);sum+=weight; }
    for (unsigned row=0;row<ratio;++row) total=__fadd_rn(total,__fmul_rn(values[row*512+k],expf(scores[row*512+k]-maximum)/sum));
    latent[k]=ria_num_bf16(total);
}
static __global__ void index_score_kernel(const float *q,const float *weights,const float *keys,float *scores,uint64_t first,uint64_t count,int *error) {
    uint64_t row=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;if (row>=count) return;float total=0;
    for (unsigned head=0;head<32;++head) {
        float dot=0;for (unsigned k=0;k<128;++k) dot=__fmaf_rn(q[head*128+k],keys[row*128+k],dot);
        dot=fmaxf(0,ria_num_bf16(dot));float weight=ria_num_bf16(weights[head]*0.015625f);
        total=__fadd_rn(total,ria_num_bf16(dot*weight));
    }
    total=ria_num_bf16(total);if (!isfinite(total)) atomicExch(error,2);scores[first+row]=total;
}
static __global__ void position_kernel(uint32_t *positions,uint64_t count) { uint64_t i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;if (i<count) positions[i]=(uint32_t)i; }
static __global__ void candidate_score_kernel(const float *scores,float *blocks,uint64_t count) {
    uint64_t block=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;if (block>=(count+7)/8) return;
    float value=-INFINITY;for (uint64_t i=block*8;i<count && i<block*8+8;++i) value=fmaxf(value,scores[i]);
    blocks[block]=block==(count-1)/8 ? INFINITY : value;
}
static __global__ void candidate_mark_kernel(uint8_t *mask,const uint32_t *positions,uint64_t chosen) {
    uint64_t i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;if (i<chosen) mask[positions[i]]=1;
}
static __global__ void candidate_mask_kernel(float *scores,const uint8_t *mask,uint64_t count) { uint64_t i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;if (i<count && !mask[i/8]) scores[i]=-INFINITY; }
static __global__ void sort_top_positions(const uint32_t *source,uint32_t *destination,unsigned count) {
    __shared__ uint32_t positions[512];unsigned i=threadIdx.x;positions[i]=i<count ? source[i] : UINT32_MAX;__syncthreads();
    for (unsigned k=2;k<=512;k<<=1) for (unsigned j=k>>1;j;j>>=1) {
        unsigned other=i^j;uint32_t value=positions[other];__syncthreads();
        if (((i&k)==0)==(i<other)) { if (positions[i]>value) positions[i]=value; }
        else if (positions[i]<value) positions[i]=value;__syncthreads();
    }
    if (i<count) destination[i]=positions[i];
}
static __global__ void attention_kernel(const float *q,const float *kv,uint64_t count,const float *sink,float *output,int *error) {
    __shared__ float scores[64],probabilities[64],maximum,denominator,rescale;
    unsigned head=blockIdx.x,lane=threadIdx.x;float accumulator[2]={0,0};
    if (lane==0) { maximum=-1e30f;denominator=0; }__syncthreads();
    for (uint64_t base=0;base<count;base+=64) {
        if (lane<64) {
            float dot=0;if (base+lane<count) for (unsigned k=0;k<512;++k) dot=__fmaf_rn(q[head*512+k],kv[(base+lane)*512+k],dot);
            scores[lane]=base+lane<count ? dot*0.04419417382415922f : -INFINITY;
        }__syncthreads();
        if (lane==0) {
            float previous=maximum,sum=0;for (unsigned i=0;i<64;++i) maximum=fmaxf(maximum,scores[i]);rescale=expf(previous-maximum);
            for (unsigned i=0;i<64;++i) { float value=expf(scores[i]-maximum);sum+=value;probabilities[i]=ria_num_bf16(value); }
            denominator=denominator*rescale+sum;
        }__syncthreads();
        for (unsigned entry=0;entry<2;++entry) {
            unsigned k=lane+entry*256;accumulator[entry]*=rescale;
            for (unsigned i=0;i<64 && base+i<count;++i) accumulator[entry]=__fmaf_rn(probabilities[i],kv[(base+i)*512+k],accumulator[entry]);
        }__syncthreads();
    }
    if (lane==0) denominator+=expf(sink[head]-maximum);__syncthreads();
    for (unsigned entry=0;entry<2;++entry) {
        float value=ria_num_bf16(accumulator[entry]/denominator);if (!isfinite(value)) atomicExch(error,2);output[head*512+lane+entry*256]=value;
    }
}
static __global__ void route_kernel(float *scores,const float *bias,uint16_t *ids,float *coefficients,int *error) {
    if (threadIdx.x) return;
    for (unsigned i=0;i<384;++i) { float v=scores[i];scores[i]=sqrtf(v>20 ? v : log1pf(expf(v)));if (!isfinite(scores[i])) atomicExch(error,2); }
    float sum=0;bool selected[384]={false};
    for (unsigned slot=0;slot<6;++slot) {
        unsigned winner=0;float best=-INFINITY;
        for (unsigned i=0;i<384;++i) if (!selected[i] && scores[i]+bias[i]>best) { winner=i;best=scores[i]+bias[i]; }
        selected[winner]=true;ids[slot]=(uint16_t)winner;coefficients[slot]=scores[winner];sum+=scores[winner];
    }
    for (unsigned slot=0;slot<6;++slot) coefficients[slot]=coefficients[slot]/(sum+1e-20f)*1.5f;
}
static __global__ void merge_kernel(const uint16_t *ids,const float *results,const float *shared,float *output,int *error) {
    unsigned k=blockIdx.x*blockDim.x+threadIdx.x;if (k>=5120) return;unsigned order[6]={0,1,2,3,4,5};
    for (unsigned i=1;i<6;++i) { unsigned value=order[i],j=i;while (j && ids[order[j-1]]>ids[value]) { order[j]=order[j-1];--j; }order[j]=value; }
    float total=0;for (unsigned i=0;i<6;++i) total=__fadd_rn(total,results[order[i]*5120+k]);
    total=ria_num_bf16(__fadd_rn(total,shared[k]));if (!isfinite(total)) atomicExch(error,2);output[k]=total;
}
static __global__ void engram_decode_kernel(const uint8_t *rows,float *features,int *error) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if (i>=6144) return;unsigned row=i/256,k=i%256;const uint8_t *p=rows+row*264;
    float value=ria_num_bf16(ria_num_e4m3(p[k])*ria_num_ue8m0(p[256+k/32]));if (!isfinite(value)) atomicExch(error,1);features[i]=value;
}
static __global__ void engram_fuse_kernel(float *h,const float *kv,const float *qweight,const float *kweight,int *error) {
    __shared__ float gate[4];unsigned i=threadIdx.x;
    if (i<4) {
        float hs=0,ks=0,dot=0;
        for (unsigned k=0;k<5120;++k) {
            float hv=h[i*5120+k],key=kv[i*5120+k];hs=__fmaf_rn(hv,hv,hs);ks=__fmaf_rn(key,key,ks);
            float product=qweight[i*5120+k]*kweight[i*5120+k];dot=__fadd_rn(dot,(hv*product)*key);
        }
        dot=(dot*(rsqrtf(hs/5120+1e-20f)*rsqrtf(ks/5120+1e-20f)))*0.013975424859373685f;
        float value=copysignf(sqrtf(fmaxf(fabsf(dot),1e-6f)),dot);gate[i]=1/(1+expf(-value));
    }__syncthreads();
    for (unsigned index=i;index<20480;index+=blockDim.x) {
        float value=ria_num_bf16(h[index]+gate[index/5120]*kv[20480+index%5120]);if (!isfinite(value)) atomicExch(error,2);h[index]=value;
    }
}
static __global__ void shared_mid_kernel(const float *gate,const float *up,float *mid,int *error) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if (i>=2304) return;float g=fminf(gate[i],10),u=fmaxf(-10,fminf(up[i],10));
    float value=ria_num_bf16((g/(1+expf(-g)))*u);if (!isfinite(value)) atomicExch(error,2);mid[i]=value;
}
bool ria_graph_cuda_create(const ria_graph_options *o,ria_graph_cuda **out,ria_error *e) {
    if (!out || !o || o->device!=0 || !o->gpu_uuid || !o->max_tokens || o->max_tokens>1048576 ||
        !o->projection_tile_rows || o->projection_tile_rows>4096 || !o->state_tile_rows || o->state_tile_rows>4096 ||
        !o->device_budget || !o->pinned_budget)
        return ria_fail(e,RIA_INVALID_REQUEST,"invalid bounded graph CUDA options/output");*out=NULL;
    ria_graph_cuda *c=(ria_graph_cuda *)calloc(1,sizeof(*c));if (!c) return ria_fail(e,RIA_RESOURCE_LIMIT,"graph CUDA owner allocation failed");c->options=*o;
    if (!check(cudaSetDevice(o->device),e,"select graph device")) goto bad;
    { int count=0;cudaDeviceProp p;
      if (!check(cudaGetDeviceCount(&count),e,"inspect client device count") ||
          !check(cudaGetDeviceProperties(&p,o->device),e,"inspect graph device")) goto bad;
      char text[41];const uint8_t *u=(const uint8_t *)p.uuid.bytes;
      (void)snprintf(text,sizeof(text),"GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",u[0],u[1],u[2],u[3],u[4],u[5],u[6],u[7],u[8],u[9],u[10],u[11],u[12],u[13],u[14],u[15]);
      if (!ria_graph_client_device_validate(count,o->device,p.major,p.minor,p.name,text,o->gpu_uuid,e)) goto bad;
    }
    if (!ria_expert_cuda_create_pooled(o->device,32768,32768,129280,1,o->projection_tile_rows,o->device_budget,o->pinned_budget,&c->projection,e)) goto bad;
    c->stream=(cudaStream_t)ria_expert_cuda_stream(c->projection);c->bytes=ria_expert_cuda_workspace_bytes(c->projection);
    {const uint64_t sizes[RIA_G_BUFFER_COUNT]={20480,20480,5120,5120,4,4,4,16,4,4,16,24,1280,32768,512,512,4096,32,640*512,32768,8192,384,6*5120,5120,6144,25600,129280,(uint64_t)o->state_tile_rows*128,1024,1024};
     for (unsigned i=0;i<RIA_G_BUFFER_COUNT;++i) if (!allocate(c,(void **)&c->buffers[i],sizes[i],sizeof(float),e)) goto bad;
    }
    c->packed_bytes=(uint64_t)o->state_tile_rows*528;if (c->packed_bytes<640*528) c->packed_bytes=640*528;
    if (!allocate(c,(void **)&c->weights,40960,sizeof(float),e) || !allocate(c,(void **)&c->frequencies,64,sizeof(float),e) ||
        !allocate(c,(void **)&c->scores,o->max_tokens,sizeof(float),e) || !allocate(c,(void **)&c->sorted_scores,o->max_tokens,sizeof(float),e) ||
        !allocate(c,(void **)&c->block_scores,ceil_div(o->max_tokens,8),sizeof(float),e) ||
        !allocate(c,(void **)&c->positions,o->max_tokens,sizeof(uint32_t),e) || !allocate(c,(void **)&c->sorted_positions,o->max_tokens,sizeof(uint32_t),e) ||
        !allocate(c,(void **)&c->top_positions,512,sizeof(uint32_t),e) || !allocate(c,(void **)&c->route_ids,6,sizeof(uint16_t),e) ||
        !allocate(c,(void **)&c->route_coefficients,6,sizeof(float),e) || !allocate(c,(void **)&c->packed,c->packed_bytes,1,e) ||
        !allocate(c,(void **)&c->candidate_blocks,ceil_div(o->max_tokens,8),1,e) || !allocate(c,(void **)&c->error,1,sizeof(int),e)) goto bad;
    if (!check(cub::DeviceRadixSort::SortPairsDescending(NULL,c->sort_bytes,c->scores,c->sorted_scores,c->positions,c->sorted_positions,
                   (int)o->max_tokens,0,32,c->stream),e,"size exact index sort workspace") ||
        !allocate(c,&c->sort_workspace,c->sort_bytes,1,e) || !ria_graph_cuda_reset(c,e)) goto bad;
    frequency_kernel<<<1,32,0,c->stream>>>(c->frequencies);
    if (!launched(e,"initialize static RoPE frequencies") || !finish(c,e)) goto bad;
    *out=c;return true;
bad:{ria_error cleanup;(void)ria_graph_cuda_destroy(c,&cleanup);return false;}
}
bool ria_graph_cuda_destroy(ria_graph_cuda *c,ria_error *e) {
    if (!c) return true;bool ok=check(cudaSetDevice(c->options.device),e,"select graph cleanup device");
    if (c->stream && !check(cudaStreamSynchronize(c->stream),ok ? e : NULL,"drain graph kernels")) ok=false;
    for (unsigned i=0;i<RIA_G_BUFFER_COUNT;++i) if (c->buffers[i] && !check(cudaFree(c->buffers[i]),ok ? e : NULL,"release graph buffer")) ok=false;
    void *owned[]={c->weights,c->frequencies,c->scores,c->sorted_scores,c->block_scores,c->positions,c->sorted_positions,c->top_positions,
                    c->route_ids,c->route_coefficients,c->packed,c->candidate_blocks,c->error,c->sort_workspace};
    for (unsigned i=0;i<sizeof(owned)/sizeof(owned[0]);++i) if (owned[i] && !check(cudaFree(owned[i]),ok ? e : NULL,"release graph scratch")) ok=false;
    if (!ria_expert_cuda_destroy(c->projection,ok ? e : NULL)) ok=false;free(c);return ok;
}
uint64_t ria_graph_cuda_bytes(const ria_graph_cuda *c) { return c ? c->bytes : 0; }
uint64_t ria_graph_cuda_pinned_bytes(const ria_graph_cuda *c) { return c ? ria_expert_cuda_pinned_bytes(c->projection) : 0; }
float *ria_graph_cuda_buffer(ria_graph_cuda *c,unsigned buffer) { return c && buffer<RIA_G_BUFFER_COUNT ? c->buffers[buffer] : NULL; }
void *ria_graph_cuda_stream(ria_graph_cuda *c) { return c ? (void *)c->stream : NULL; }
bool ria_graph_cuda_drain(ria_graph_cuda *c,ria_error *e) {
    return !c || check(cudaStreamSynchronize(c->stream),e,"drain failed graph operation");
}
bool ria_graph_cuda_reset(ria_graph_cuda *c,ria_error *e) {
    if (!c || !check(cudaStreamSynchronize(c->stream),e,"drain previous graph generation") ||
        !check(cudaMemsetAsync(c->error,0,sizeof(int),c->stream),e,"reset graph numerical status") ||
        !check(cudaMemsetAsync(c->candidate_blocks,0,(size_t)ceil_div(c->options.max_tokens,8),c->stream),e,"reset candidate cache")) return false;
    initial_pre<<<1,4,0,c->stream>>>(c->buffers[RIA_G_PRE]);return launched(e,"initialize one-hot graph mix") && finish(c,e);
}
bool ria_graph_cuda_copy(ria_graph_cuda *c,float *to,const float *from,uint64_t count,ria_error *e) { return check(cudaMemcpyAsync(to,from,(size_t)count*sizeof(float),cudaMemcpyDeviceToDevice,c->stream),e,"copy graph values"); }
bool ria_graph_cuda_upload(ria_graph_cuda *c,float *to,const float *from,uint64_t count,ria_error *e) { return ria_expert_cuda_upload_bytes(c->projection,to,from,count*sizeof(float),e); }
bool ria_graph_cuda_download(ria_graph_cuda *c,const float *from,float *to,uint64_t count,ria_error *e) {
    return ria_expert_cuda_download_bytes(c->projection,from,to,count*sizeof(float),e) && finish(c,e);
}
bool ria_graph_cuda_tensor(ria_graph_cuda *c,const ria_tensor *t,uint64_t first,float *out,uint64_t count,ria_error *e) {
    unsigned size=t && !strcmp(t->dtype,"BF16") ? 2 : t && !strcmp(t->dtype,"F32") ? 4 : 0;
    if (!size || count>40960 || first>t->length/size || count>t->length/size-first || count*size>c->packed_bytes)
        return ria_fail(e,RIA_INTEGRITY_ERROR,"graph tensor vector shape/precision invalid");
    if (!ria_expert_cuda_upload_bytes(c->projection,c->packed,t->data+first*size,count*size,e)) return false;
    tensor_decode<<<(unsigned)ceil_div(count,256),256,0,c->stream>>>(c->packed,out,count,size==2 ? 1 : 2,c->error);return launched(e,"decode graph vector");
}
bool ria_graph_cuda_project(ria_graph_cuda *c,const ria_expert_matrix *m,const float *input,float *output,bool bf16,ria_error *e) {
    return ria_expert_cuda_projection(c->projection,m,input,1,output,bf16,e);
}
bool ria_graph_cuda_norm(ria_graph_cuda *c,float *values,uint64_t rows,uint64_t width,const ria_tensor *weight,bool layer,ria_error *e) {
    if (!rows || !width || width>40960 || !ria_graph_cuda_tensor(c,weight,0,c->weights,width,e)) return false;
    norm_kernel<<<(unsigned)rows,256,0,c->stream>>>(values,c->weights,width,layer,c->error);return launched(e,"normalize graph values");
}
bool ria_graph_cuda_expand(ria_graph_cuda *c,const float *input,float *h,ria_error *e) { expand_kernel<<<80,256,0,c->stream>>>(input,h);return launched(e,"expand residual streams"); }
bool ria_graph_cuda_mhc(ria_graph_cuda *c,const float *h,float *mix,const ria_tensor *scale,const ria_tensor *base,float *pre,float *post,float *comb,ria_error *e) {
    if (!ria_graph_cuda_tensor(c,scale,0,c->weights,3,e) || !ria_graph_cuda_tensor(c,base,0,c->weights+3,24,e)) return false;
    mhc_kernel<<<1,1,0,c->stream>>>(h,mix,c->weights,c->weights+3,pre,post,comb,c->error);return launched(e,"compute flattened mHC/Sinkhorn");
}
bool ria_graph_cuda_collapse(ria_graph_cuda *c,const float *h,const float *pre,float *out,ria_error *e) { collapse_kernel<<<20,256,0,c->stream>>>(h,pre,out);return launched(e,"collapse residual streams"); }
bool ria_graph_cuda_post(ria_graph_cuda *c,const float *x,const float *residual,const float *post,const float *comb,float *h,ria_error *e) { post_kernel<<<80,256,0,c->stream>>>(x,residual,post,comb,h,c->error);return launched(e,"mix residual streams"); }
bool ria_graph_cuda_rope(ria_graph_cuda *c,float *values,uint64_t rows,uint64_t width,uint64_t position,uint32_t ratio,bool inverse,ria_error *e) {
    if (!rows || width<64 || position>=c->options.max_tokens) return ria_fail(e,RIA_INVALID_REQUEST,"RoPE extent outside admitted positions");
    rope_kernel<<<(unsigned)ceil_div(rows*32,256),256,0,c->stream>>>(values,rows,width,position,c->frequencies+(ratio ? 32 : 0),inverse);return launched(e,"rotate graph values");
}
uint64_t ria_graph_packed_stride(uint32_t rep,uint64_t width) { return rep==1 ? width+width/32 : rep==2 ? width/2+width/16 : rep==3 ? width/2+width/32 : 0; }
bool ria_graph_cuda_pack(ria_graph_cuda *c,const float *values,uint64_t width,uint32_t rep,uint8_t *host,ria_error *e) {
    uint64_t bytes=ria_graph_packed_stride(rep,width);if (!host || !bytes || bytes>c->packed_bytes || width%(rep==2 ? 16 : 32)) return ria_fail(e,RIA_INVALID_REQUEST,"invalid packed state vector");
    pack_kernel<<<(unsigned)ceil_div(width/(rep==2 ? 16 : 32),128),128,0,c->stream>>>(NULL,values,c->packed,width,rep,false,c->error);
    return launched(e,"quantize authoritative graph state") && ria_expert_cuda_download_bytes(c->projection,c->packed,host,bytes,e) && finish(c,e);
}
bool ria_graph_cuda_quantize_inplace(ria_graph_cuda *c,float *values,uint64_t width,uint32_t rep,ria_error *e) {
    if (rep<1 || rep>3 || !width || width%(rep==2 ? 16 : 32)) return ria_fail(e,RIA_INVALID_REQUEST,"invalid in-place state quantizer");
    pack_kernel<<<(unsigned)ceil_div(width/(rep==2 ? 16 : 32),128),128,0,c->stream>>>(values,values,NULL,width,rep,true,c->error);return launched(e,"quantize index query values");
}
bool ria_graph_cuda_unpack(ria_graph_cuda *c,const uint8_t *host,uint64_t rows,uint64_t width,uint32_t rep,float *out,ria_error *e) {
    uint64_t bytes=ria_graph_packed_stride(rep,width)*rows;
    if (!rows || !bytes || bytes>c->packed_bytes || width%(rep==2 ? 16 : 32)) return ria_fail(e,RIA_RESOURCE_LIMIT,"packed state group exceeds admitted tile");
    if (!ria_expert_cuda_upload_bytes(c->projection,c->packed,host,bytes,e)) return false;
    unpack_kernel<<<(unsigned)ceil_div(rows*width,256),256,0,c->stream>>>(c->packed,out,rows,width,rep,c->error);return launched(e,"decode exact graph state");
}
bool ria_graph_cuda_pool(ria_graph_cuda *c,float *kv,float *scores,uint32_t ratio,float *latent,ria_error *e) {
    if (ratio!=2) return ria_fail(e,RIA_INVALID_REQUEST,"compressor pool ratio differs from pinned graph");pool_kernel<<<2,256,0,c->stream>>>(kv,scores,ratio,latent);return launched(e,"pool completed compressor group");
}
bool ria_graph_cuda_index_scores(ria_graph_cuda *c,const float *q,const float *weights,const float *keys,uint64_t first,uint64_t count,ria_error *e) {
    if (!count || first>c->options.max_tokens || count>c->options.max_tokens-first) return ria_fail(e,RIA_INVALID_REQUEST,"index score tile outside admitted history");
    index_score_kernel<<<(unsigned)ceil_div(count,128),128,0,c->stream>>>(q,weights,keys,c->scores,first,count,c->error);return launched(e,"score exact index positions");
}
bool ria_graph_cuda_index_download(ria_graph_cuda *c,uint64_t first,uint64_t count,float *host,ria_error *e) {
    if (!c || !host || !count || first>c->options.max_tokens || count>c->options.max_tokens-first)
        return ria_fail(e,RIA_INVALID_REQUEST,"qualification index observation exceeds admitted history");
    return ria_expert_cuda_download_bytes(c->projection,c->scores+first,host,count*sizeof(float),e) && finish(c,e);
}
static bool sort_pairs(ria_graph_cuda *c,float *values,uint64_t count,ria_error *e) {
    position_kernel<<<(unsigned)ceil_div(count,256),256,0,c->stream>>>(c->positions,count);
    return launched(e,"initialize stable position IDs") && check(cub::DeviceRadixSort::SortPairsDescending(c->sort_workspace,c->sort_bytes,values,
            c->sorted_scores,c->positions,c->sorted_positions,(int)count,0,32,c->stream),e,"sort exact index/candidate scores");
}
bool ria_graph_cuda_select(ria_graph_cuda *c,uint64_t count,bool source,bool uses,uint32_t *selected,uint32_t *selected_count,ria_error *e) {
    if (!count || count>c->options.max_tokens || !selected || !selected_count) return ria_fail(e,RIA_INVALID_REQUEST,"invalid hierarchical candidate selection");
    if (source) {
        uint64_t blocks=ceil_div(count,8),chosen=blocks<2048 ? blocks : 2048;
        candidate_score_kernel<<<(unsigned)ceil_div(blocks,256),256,0,c->stream>>>(c->scores,c->block_scores,count);
        if (!launched(e,"score candidate blocks") || !sort_pairs(c,c->block_scores,blocks,e) ||
            !check(cudaMemsetAsync(c->candidate_blocks,0,(size_t)ceil_div(c->options.max_tokens,8),c->stream),e,"replace candidate mask")) return false;
        candidate_mark_kernel<<<(unsigned)ceil_div(chosen,256),256,0,c->stream>>>(c->candidate_blocks,c->sorted_positions,chosen);
        if (!launched(e,"publish candidate blocks")) return false;
    } else if (uses) { candidate_mask_kernel<<<(unsigned)ceil_div(count,256),256,0,c->stream>>>(c->scores,c->candidate_blocks,count);if (!launched(e,"apply shared candidate mask")) return false; }
    if (!sort_pairs(c,c->scores,count,e)) return false;
    unsigned chosen=count<512 ? (unsigned)count : 512;
    sort_top_positions<<<1,512,0,c->stream>>>(c->sorted_positions,c->top_positions,chosen);
    if (!launched(e,"order selected positions") || !ria_expert_cuda_download_bytes(c->projection,c->top_positions,selected,chosen*sizeof(uint32_t),e) || !finish(c,e)) return false;
    *selected_count=chosen;return true;
}
bool ria_graph_cuda_attention(ria_graph_cuda *c,const float *q,const float *kv,uint64_t count,const ria_tensor *sink,float *output,ria_error *e) {
    if (!count || count>640 || !ria_graph_cuda_tensor(c,sink,0,c->weights,64,e)) return false;
    attention_kernel<<<64,256,0,c->stream>>>(q,kv,count,c->weights,output,c->error);return launched(e,"source sparse attention/sink");
}
bool ria_graph_cuda_route(ria_graph_cuda *c,const float *scores,const ria_tensor *bias,uint16_t ids[6],float coefficients[6],ria_error *e) {
    if (!ria_graph_cuda_tensor(c,bias,0,c->weights,384,e)) return false;
    route_kernel<<<1,1,0,c->stream>>>((float *)scores,c->weights,c->route_ids,c->route_coefficients,c->error);
    if (!launched(e,"source sqrtsoftplus/VL routing") || !ria_expert_cuda_download_bytes(c->projection,c->route_ids,ids,6*sizeof(uint16_t),e) ||
        !ria_expert_cuda_download_bytes(c->projection,c->route_coefficients,coefficients,6*sizeof(float),e) || !finish(c,e)) return false;
    for (unsigned i=0;i<6;++i) { if (ids[i]>=384 || !isfinite(coefficients[i]) || coefficients[i]<0 || coefficients[i]>1.5f) return ria_fail(e,RIA_EXECUTOR_ERROR,"invalid CUDA routing result");for (unsigned j=0;j<i;++j) if (ids[i]==ids[j]) return ria_fail(e,RIA_EXECUTOR_ERROR,"duplicate selected expert"); }
    return true;
}
bool ria_graph_cuda_merge(ria_graph_cuda *c,const uint16_t ids[6],const float *contributions,const float *shared,float *out,ria_error *e) {
    if (!ria_graph_cuda_upload(c,c->buffers[RIA_G_EXPERT_RESULTS],contributions,6*5120,e) ||
        !ria_expert_cuda_upload_bytes(c->projection,c->route_ids,ids,6*sizeof(uint16_t),e)) return false;
    merge_kernel<<<20,256,0,c->stream>>>(c->route_ids,c->buffers[RIA_G_EXPERT_RESULTS],shared,out,c->error);
    /* ids is the caller's stack array: its host-copy lifetime ends on return. */
    return launched(e,"ordered routed/shared contribution sum") && finish(c,e);
}
bool ria_graph_cuda_engram(ria_graph_cuda *c,const uint8_t *rows,float *features,ria_error *e) {
    if (!ria_expert_cuda_upload_bytes(c->projection,c->packed,rows,24*264,e)) return false;
    engram_decode_kernel<<<24,256,0,c->stream>>>(c->packed,features,c->error);return launched(e,"decode source Engram rows");
}
bool ria_graph_cuda_engram_fuse(ria_graph_cuda *c,float *h,const float *kv,const ria_tensor *qw,const ria_tensor *kw,bool enabled,ria_error *e) {
    if (!enabled) return true;
    if (!ria_graph_cuda_tensor(c,qw,0,c->weights,20480,e) || !ria_graph_cuda_tensor(c,kw,0,c->weights+20480,20480,e)) return false;
    engram_fuse_kernel<<<1,256,0,c->stream>>>(h,kv,c->weights,c->weights+20480,c->error);return launched(e,"source normalized Engram fusion");
}
bool ria_graph_cuda_shared(ria_graph_cuda *c,const ria_expert *expert,const float *input,float *out,ria_error *e) {
    float *gate=c->buffers[RIA_G_ENGRAM_INPUT],*up=c->buffers[RIA_G_ENGRAM_INPUT]+2304,*mid=c->buffers[RIA_G_ENGRAM_KV];
    if (!ria_graph_cuda_project(c,&expert->gate,input,gate,true,e) || !ria_graph_cuda_project(c,&expert->up,input,up,true,e)) return false;
    shared_mid_kernel<<<9,256,0,c->stream>>>(gate,up,mid,c->error);
    return launched(e,"shared expert clamped activation") && ria_graph_cuda_project(c,&expert->down,mid,out,true,e);
}
bool ria_graph_cuda_local_create(ria_graph_cuda *c,const ria_expert *expert,uint64_t budget,ria_expert_cuda_resident *out,ria_error *e) {
    return ria_expert_cuda_resident_create(c->projection,expert,budget,out,e);
}
bool ria_graph_cuda_local_evaluate(ria_graph_cuda *c,const ria_expert *expert,const ria_expert_cuda_resident *resident,
                                  const float *input,float coefficient,float *host,ria_error *e) {
    float *output=c->buffers[RIA_G_EXPERT_RESULTS];
    return ria_expert_cuda_evaluate_device(c->projection,expert,resident,input,coefficient,output,e) &&
        ria_graph_cuda_download(c,output,host,5120,e);
}
