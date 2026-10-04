#include "vision.h"
#include "expert_cuda.h"
#include <cuda_runtime.h>
#include "numeric.h"
#include <stdlib.h>
#include <string.h>

struct ria_vision_cuda {
    const ria_vision_parameters *parameters;
    int device;
    uint32_t max_patches;
    uint64_t budget,bytes;
    ria_expert_cuda *projection;
    cudaStream_t stream;
    float *input,*h,*normalized,*qkv,*attention,*gate_up,*mid,*downsampled,*aligned,*output,*weights;
    uint8_t *packed;
    int *error;
};
uint64_t ria_vision_cuda_metadata_bytes(void) { return sizeof(ria_vision_cuda)+ria_expert_cuda_metadata_bytes(); }
static bool check(cudaError_t status,ria_error *e,const char *op) { return status==cudaSuccess || ria_fail(e,RIA_EXECUTOR_ERROR,"%s: %s",op,cudaGetErrorString(status)); }
static uint64_t divide(uint64_t a,uint64_t b) { return a/b+(a%b!=0); }
static bool allocate(ria_vision_cuda *v,void **out,uint64_t count,uint64_t element,ria_error *e) {
    uint64_t bytes,total;
    if (!ria_u64_mul(count,element,&bytes) || bytes>SIZE_MAX || !ria_u64_add(v->bytes,bytes,&total) || total>v->budget)
        return ria_fail(e,RIA_RESOURCE_LIMIT,"vision activation/projection workspace exceeds admission");
    if (!check(cudaMalloc(out,(size_t)bytes),e,"allocate bounded vision workspace")) return false;v->bytes=total;return true;
}
static bool launched(ria_error *e,const char *op) { return check(cudaGetLastError(),e,op); }
static bool finish(ria_vision_cuda *v,ria_error *e) {
    int status=0;if (!ria_expert_cuda_download_bytes(v->projection,v->error,&status,sizeof(status),e)) return false;
    return !status || ria_fail(e,RIA_EXECUTOR_ERROR,"vision numerical/input invariant failed (%d)",status);
}
static __global__ void decode_weight(const uint8_t *bytes,float *weights,unsigned count,unsigned size,int *error) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if (i>=count) return;const uint8_t *p=bytes+i*size;
    uint32_t bits=(uint32_t)p[0]|((uint32_t)p[1]<<8);bits=size==2 ? bits<<16 : bits|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);
    float value=ria_num_float(bits);if (!isfinite(value)) atomicExch(error,1);weights[i]=value;
}
static bool vector(ria_vision_cuda *v,const ria_tensor *t,unsigned count,ria_error *e) {
    unsigned size=!strcmp(t->dtype,"F32") ? 4 : !strcmp(t->dtype,"BF16") ? 2 : 0;
    if (!size || count>5120 || t->length!=(uint64_t)count*size) return ria_fail(e,RIA_INTEGRITY_ERROR,"vision vector has invalid byte shape");
    if (!ria_expert_cuda_upload_bytes(v->projection,v->packed,t->data,(uint64_t)count*size,e)) return false;
    decode_weight<<<(unsigned)divide(count,256),256,0,v->stream>>>(v->packed,v->weights,count,size,v->error);return launched(e,"decode vision vector");
}
static __global__ void bias_kernel(float *values,const float *bias,uint64_t rows,unsigned width,int *error) {
    uint64_t i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;if (i>=rows*width) return;
    float value=ria_num_bf16(values[i]+bias[i%width]);if (!isfinite(value)) atomicExch(error,2);values[i]=value;
}
static bool project(ria_vision_cuda *v,const ria_expert_matrix *m,const ria_tensor *bias,const float *input,uint64_t rows,float *output,ria_error *e) {
    /* Runtime rows are independent; quantizer reduction groups remain within
     * each row. This bounds the generic projection's allocation independently
     * of the entire image while retaining one image-wide attention domain. */
    for (uint64_t first=0;first<rows;first+=64) {
        uint64_t count=rows-first<64 ? rows-first : 64;
        if (!ria_expert_cuda_projection(v->projection,m,input+first*m->in_features,count,output+first*m->out_features,bias==NULL,e)) return false;
    }
    if (bias) {
        if (!vector(v,bias,(unsigned)m->out_features,e)) return false;
        bias_kernel<<<(unsigned)divide(rows*m->out_features,256),256,0,v->stream>>>(output,v->weights,rows,(unsigned)m->out_features,v->error);
        if (!launched(e,"vision bias before BF16 epilogue")) return false;
    }
    return true;
}
static __global__ void norm_kernel(const float *input,float *output,const float *weights,uint64_t rows,int *error) {
    __shared__ float sums[256],inverse;unsigned row=blockIdx.x;if (row>=rows) return;float square=0;
    for (unsigned k=threadIdx.x;k<1024;k+=256) square=__fmaf_rn(input[(uint64_t)row*1024+k],input[(uint64_t)row*1024+k],square);
    sums[threadIdx.x]=square;__syncthreads();
    for (unsigned size=128;size;size>>=1) { if (threadIdx.x<size) sums[threadIdx.x]+=sums[threadIdx.x+size];__syncthreads(); }
    if (!threadIdx.x) inverse=rsqrtf(sums[0]/1024+1e-6f);__syncthreads();
    for (unsigned k=threadIdx.x;k<1024;k+=256) {
        float value=ria_num_bf16((input[(uint64_t)row*1024+k]*inverse)*weights[k]);
        if (!isfinite(value)) atomicExch(error,2);output[(uint64_t)row*1024+k]=value;
    }
}
static bool norm(ria_vision_cuda *v,const float *input,float *out,uint64_t rows,const ria_tensor *weights,ria_error *e) {
    if (!vector(v,weights,1024,e)) return false;norm_kernel<<<(unsigned)rows,256,0,v->stream>>>(input,out,v->weights,rows,v->error);return launched(e,"vision RMSNorm");
}
static __global__ void rope_kernel(float *qkv,uint64_t count,unsigned grid_width) {
    uint64_t i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;if (i>=count*16*32) return;
    uint64_t row=i/(16*32),head=(i/32)%16,pair=i%32;
    unsigned axis=pair<16 ? (unsigned)(row/grid_width) : (unsigned)(row%grid_width);
    float frequency=1/powf(10000.0f,(float)(2*(pair%16))/32),sine,cosine;sincosf((float)axis*frequency,&sine,&cosine);
    for (unsigned which=0;which<2;++which) {
        uint64_t offset=row*3072+which*1024+head*64+pair;float a=qkv[offset],b=qkv[offset+32];
        qkv[offset]=ria_num_bf16(a*cosine-b*sine);qkv[offset+32]=ria_num_bf16(b*cosine+a*sine);
    }
}
static __global__ void attention_kernel(const float *qkv,uint64_t rows,float *output,int *error) {
    __shared__ float scores[64],probability[64],maximum,denominator,rescale;
    uint64_t query=blockIdx.x/16;unsigned head=blockIdx.x%16,k=threadIdx.x;float accumulator=0;
    if (!k) { maximum=-INFINITY;denominator=0; }__syncthreads();
    for (uint64_t first=0;first<rows;first+=64) {
        float dot=0;
        if (first+k<rows) for (unsigned d=0;d<64;++d) dot=__fmaf_rn(qkv[query*3072+head*64+d],qkv[(first+k)*3072+1024+head*64+d],dot);
        scores[k]=first+k<rows ? dot*0.125f : -INFINITY;__syncthreads();
        if (!k) {
            float previous=maximum,sum=0;for (unsigned j=0;j<64;++j) maximum=fmaxf(maximum,scores[j]);rescale=previous==-INFINITY ? 0 : expf(previous-maximum);
            for (unsigned j=0;j<64;++j) { float p=expf(scores[j]-maximum);sum+=p;probability[j]=ria_num_bf16(p); }denominator=denominator*rescale+sum;
        }__syncthreads();
        accumulator*=rescale;
        for (unsigned j=0;j<64 && first+j<rows;++j) accumulator=__fmaf_rn(probability[j],qkv[(first+j)*3072+2048+head*64+k],accumulator);
        __syncthreads();
    }
    float value=ria_num_bf16(accumulator/denominator);if (!isfinite(value)) atomicExch(error,2);output[query*1024+head*64+k]=value;
}
static __global__ void residual_kernel(float *h,const float *update,uint64_t count,int *error) {
    uint64_t i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;if (i>=count) return;float value=ria_num_bf16(h[i]+update[i]);if (!isfinite(value)) atomicExch(error,2);h[i]=value;
}
static __global__ void swiglu_kernel(const float *gate_up,float *mid,uint64_t rows,int *error) {
    uint64_t i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;if (i>=rows*2816) return;uint64_t row=i/2816,k=i%2816;
    float g=gate_up[row*5632+k],up=gate_up[row*5632+2816+k];float silu=ria_num_bf16(g/(1+expf(-g)));
    float value=ria_num_bf16(silu*up);if (!isfinite(value)) atomicExch(error,2);mid[i]=value;
}
static __global__ void unfold_kernel(const float *input,float *output,unsigned h,unsigned w,unsigned out_width,uint64_t rows) {
    uint64_t i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;if (i>=rows*9216) return;
    uint64_t row=i/9216,k=i%9216;unsigned channel=(unsigned)(k/9),dy=(unsigned)(k%9)/3,dx=(unsigned)(k%3);
    unsigned y=(unsigned)(row/out_width)*3+dy,x=(unsigned)(row%out_width)*3+dx;
    output[i]=y<h && x<w ? input[((uint64_t)y*w+x)*1024+channel] : 0;
}
static __global__ void gelu_kernel(float *input,uint64_t count,int *error) {
    uint64_t i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;if (i>=count) return;float x=input[i];
    float value=ria_num_bf16(0.5f*x*(1+erff(x*0.707106781186547524f)));if (!isfinite(value)) atomicExch(error,2);input[i]=value;
}
bool ria_vision_cuda_create(const ria_vision_parameters *p,int device,uint32_t patches,uint64_t budget,uint64_t pinned_budget,ria_vision_cuda **out,ria_error *e) {
    if (!out || !p || !patches || patches>9216) return ria_fail(e,RIA_INVALID_REQUEST,"invalid vision CUDA limits");*out=NULL;
    ria_vision_cuda *v=(ria_vision_cuda *)calloc(1,sizeof(*v));if (!v) return ria_fail(e,RIA_RESOURCE_LIMIT,"vision CUDA owner allocation failed");
    v->parameters=p;v->device=device;v->max_patches=patches;v->budget=budget;
    if (!check(cudaSetDevice(device),e,"select vision CUDA device") || !ria_expert_cuda_create_pooled(device,9216,5632,5120,64,128,budget,pinned_budget,&v->projection,e)) goto bad;
    v->stream=(cudaStream_t)ria_expert_cuda_stream(v->projection);v->bytes=ria_expert_cuda_workspace_bytes(v->projection);
    if (!allocate(v,(void **)&v->input,(uint64_t)patches*588,sizeof(float),e) || !allocate(v,(void **)&v->h,(uint64_t)patches*1024,sizeof(float),e) ||
        !allocate(v,(void **)&v->normalized,(uint64_t)patches*1024,sizeof(float),e) || !allocate(v,(void **)&v->qkv,(uint64_t)patches*3072,sizeof(float),e) ||
        !allocate(v,(void **)&v->attention,(uint64_t)patches*1024,sizeof(float),e) || !allocate(v,(void **)&v->gate_up,(uint64_t)patches*5632,sizeof(float),e) ||
        !allocate(v,(void **)&v->mid,(uint64_t)patches*2816,sizeof(float),e) || !allocate(v,(void **)&v->downsampled,1024*9216,sizeof(float),e) ||
        !allocate(v,(void **)&v->aligned,1024*5120,sizeof(float),e) || !allocate(v,(void **)&v->output,1024*5120,sizeof(float),e) ||
        !allocate(v,(void **)&v->weights,5120,sizeof(float),e) || !allocate(v,(void **)&v->packed,5120*4,1,e) ||
        !allocate(v,(void **)&v->error,1,sizeof(int),e)) goto bad;
    *out=v;return true;
bad:{ria_error cleanup;(void)ria_vision_cuda_destroy(v,&cleanup);return false;}
}
bool ria_vision_cuda_destroy(ria_vision_cuda *v,ria_error *e) {
    if (!v) return true;bool ok=check(cudaSetDevice(v->device),e,"select vision cleanup device");
    if (v->stream && !check(cudaStreamSynchronize(v->stream),ok ? e : NULL,"drain vision kernels")) ok=false;
    void *owned[]={v->input,v->h,v->normalized,v->qkv,v->attention,v->gate_up,v->mid,v->downsampled,v->aligned,v->output,v->weights,v->packed,v->error};
    for (unsigned i=0;i<sizeof(owned)/sizeof(owned[0]);++i) if (owned[i] && !check(cudaFree(owned[i]),ok ? e : NULL,"release vision workspace")) ok=false;
    if (!ria_expert_cuda_destroy(v->projection,ok ? e : NULL)) ok=false;free(v);return ok;
}
uint64_t ria_vision_cuda_bytes(const ria_vision_cuda *v) { return v ? v->bytes : 0; }
uint64_t ria_vision_cuda_pinned_bytes(const ria_vision_cuda *v) { return v ? ria_expert_cuda_pinned_bytes(v->projection) : 0; }
bool ria_vision_cuda_encode(ria_vision_cuda *v,const float *patches,uint32_t h,uint32_t w,float *output,uint64_t output_rows,ria_error *e) {
    uint64_t rows=(uint64_t)h*w,lh=divide(h,3),lw=divide(w,3),aligned_rows=lh*lw;
    if (!v || !patches || !output || !h || !w || rows>v->max_patches || lh*(lw+1)+2>1024 || output_rows!=aligned_rows)
        return ria_fail(e,RIA_INVALID_REQUEST,"image grid/token expansion exceeds admitted source limits");
    if (!check(cudaSetDevice(v->device),e,"select image device") || !check(cudaMemsetAsync(v->error,0,sizeof(int),v->stream),e,"clear image status") ||
        !ria_expert_cuda_upload_bytes(v->projection,v->input,patches,rows*588*sizeof(float),e) ||
        !project(v,&v->parameters->patch,v->parameters->patch_bias,v->input,rows,v->h,e)) goto failed;
    for (unsigned layer=0;layer<32;++layer) {
        const ria_vision_layer *x=&v->parameters->layers[layer];
        if (!norm(v,v->h,v->normalized,rows,x->norm1,e) || !project(v,&x->qkv,x->qkv_bias,v->normalized,rows,v->qkv,e)) goto failed;
        rope_kernel<<<(unsigned)divide(rows*16*32,256),256,0,v->stream>>>(v->qkv,rows,w);
        if (!launched(e,"apply source 2D vision RoPE")) goto failed;
        attention_kernel<<<(unsigned)(rows*16),64,0,v->stream>>>(v->qkv,rows,v->attention,v->error);
        if (!launched(e,"full bidirectional image attention") || !project(v,&x->output,x->output_bias,v->attention,rows,v->normalized,e)) goto failed;
        residual_kernel<<<(unsigned)divide(rows*1024,256),256,0,v->stream>>>(v->h,v->normalized,rows*1024,v->error);
        if (!launched(e,"vision attention residual") || !norm(v,v->h,v->normalized,rows,x->norm2,e) ||
            !project(v,&x->gate_up,NULL,v->normalized,rows,v->gate_up,e)) goto failed;
        swiglu_kernel<<<(unsigned)divide(rows*2816,256),256,0,v->stream>>>(v->gate_up,v->mid,rows,v->error);
        if (!launched(e,"source BF16 vision SwiGLU") || !project(v,&x->down,NULL,v->mid,rows,v->normalized,e)) goto failed;
        residual_kernel<<<(unsigned)divide(rows*1024,256),256,0,v->stream>>>(v->h,v->normalized,rows*1024,v->error);
        if (!launched(e,"vision MLP residual")) goto failed;
    }
    if (!norm(v,v->h,v->normalized,rows,v->parameters->norm,e)) goto failed;
    unfold_kernel<<<(unsigned)divide(aligned_rows*9216,256),256,0,v->stream>>>(v->normalized,v->downsampled,h,w,(unsigned)lw,aligned_rows);
    if (!launched(e,"source channel-major padded 3x3 unfold") || !project(v,&v->parameters->align1,v->parameters->align1_bias,v->downsampled,aligned_rows,v->aligned,e)) goto failed;
    gelu_kernel<<<(unsigned)divide(aligned_rows*5120,256),256,0,v->stream>>>(v->aligned,aligned_rows*5120,v->error);
    if (!launched(e,"source exact GELU aligner") || !project(v,&v->parameters->align2,v->parameters->align2_bias,v->aligned,aligned_rows,v->output,e) ||
        !ria_expert_cuda_download_bytes(v->projection,v->output,output,aligned_rows*5120*sizeof(float),e)) goto failed;
    return finish(v,e);
failed:{ria_error drain;(void)finish(v,&drain);return false;}
}
