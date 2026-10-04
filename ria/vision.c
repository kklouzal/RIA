#include "vision.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fenv.h>
#include <setjmp.h>
#include <png.h>
#include <jpeglib.h>
#include <jerror.h>
#include "numeric.h"

static uint32_t round_even_positive(double x) {
    double integral=floor(x),fraction=x-integral;uint32_t result=(uint32_t)integral;
    return result+(fraction>0.5 || (fraction==0.5 && (result&1u)));
}
static uint32_t div_up(uint32_t a,uint32_t b) { return a/b+(a%b!=0); }
bool ria_image_plan(uint32_t width,uint32_t height,uint32_t maximum,ria_image_grid *out,ria_error *e) {
    if (fegetround()!=FE_TONEAREST) return ria_fail(e,RIA_UNSUPPORTED,"image source profile requires nearest-even host floating point");
    if (!out || !width || !height || width>16384 || height>16384 || (uint64_t)width*height>67108864 || !maximum || maximum>9216)
        return ria_fail(e,RIA_INVALID_REQUEST,"invalid image dimensions/admitted patch bound");
    uint32_t w=width,h=height;
    if ((uint64_t)w*h<295936) { double factor=sqrt(295936.0/((double)w*h));w=(uint32_t)(w*factor);h=(uint32_t)(h*factor); }
    uint32_t bw=div_up(w,14)*14,bh=div_up(h,14)*14;
    uint32_t lh=div_up(bh/14,3),lw=div_up(bw/14,3);
    if ((uint64_t)lh*(lw+1)+2>1024) {
        double ratio=(double)h/w,mw=sqrt(1022/ratio+0.25)-0.5,mh=mw*ratio;
        if (mw<1) { bh=511*42;bw=42; }
        else if (mh<1) { bh=42;bw=1021*42; }
        else { double beta=fmin(floor(mw)*42/w,floor(mh)*42/h);bh=(uint32_t)floor(h*beta/14)*14;bw=(uint32_t)floor(w*beta/14)*14; }
        lh=div_up(bh/14,3);lw=div_up(bw/14,3);
    }
    uint64_t patches=(uint64_t)(bh/14)*(bw/14),tokens=(uint64_t)lh*(lw+1)+2;
    if (!bh || !bw || patches>maximum || tokens>1024)
        return ria_fail(e,RIA_RESOURCE_LIMIT,"source image plan exceeds admitted patches/token capacity");
    *out=(ria_image_grid){bh,bw,bh/14,bw/14,lh,lw,(uint32_t)tokens};return true;
}
typedef struct { uint32_t source,count;int32_t *weights; } resample_row;
typedef struct { resample_row *rows;int32_t *weights;uint32_t size,span; } resample_plan;
static double cubic_weight(double distance) {
    double x=fabs(distance);
    if (x<1) return ((1.5*x-2.5)*x)*x+1;
    if (x<2) return -0.5*(((x-5)*x+8)*x-4);
    return 0;
}
static void resample_free(resample_plan *p) { free(p->rows);free(p->weights);memset(p,0,sizeof(*p)); }
static bool resample_make(uint32_t source,uint32_t size,resample_plan *p,ria_error *e) {
    double scale=(double)source/size,filter=fmax(scale,1),support=2*filter;
    p->size=size;p->span=(uint32_t)ceil(support)*2+1;
    p->rows=calloc(size,sizeof(*p->rows));p->weights=calloc((size_t)size*p->span,sizeof(*p->weights));
    if (!p->rows || !p->weights) return ria_fail(e,RIA_RESOURCE_LIMIT,"allocate image resampler coefficients failed");
    for (uint32_t i=0;i<size;++i) {
        double center=(i+0.5)*scale;int low=(int)(center-support+0.5),high=(int)(center+support+0.5);
        if (low<0) low=0;
        if (high>(int)source) high=(int)source;
        resample_row *row=&p->rows[i];row->source=(uint32_t)low;row->count=(uint32_t)(high-low);row->weights=p->weights+(size_t)i*p->span;
        double sum=0;for (uint32_t j=0;j<row->count;++j) sum+=cubic_weight(((double)j+low-center+0.5)/filter);
        if (sum==0) return ria_fail(e,RIA_INTERNAL_ERROR,"image resampling has empty support");
        for (uint32_t j=0;j<row->count;++j) {
            double value=cubic_weight(((double)j+low-center+0.5)/filter)/sum*4194304;
            row->weights[j]=(int32_t)(value+(value<0 ? -0.5 : 0.5));
        }
    }
    return true;
}
static uint8_t resample_byte(int64_t accumulator) {
    if (accumulator<=0) return 0;
    uint64_t result=(uint64_t)accumulator/4194304;
    return result>255 ? 255 : (uint8_t)result;
}
void ria_image_input_free(ria_image_input *input) {
    if (!input) return;
    free(input->patches);free(input->types);memset(input,0,sizeof(*input));
}
bool ria_image_prepare_rgb(const uint8_t *rgb,uint64_t length,uint32_t width,uint32_t height,
                            uint32_t maximum,uint64_t budget,ria_image_input *out,ria_error *e) {
    if (!out) return ria_fail(e,RIA_INVALID_REQUEST,"missing prepared image result");
    memset(out,0,sizeof(*out));ria_image_grid grid;
    if (!rgb || length!=(uint64_t)width*height*3) return ria_fail(e,RIA_INVALID_REQUEST,"invalid RGB image byte shape");
    if (!ria_image_plan(width,height,maximum,&grid,e)) return false;
    uint32_t rw=grid.pixel_width,rh=grid.pixel_height;
    double source_ratio=(double)width/height,target_ratio=(double)rw/rh;
    if (source_ratio>target_ratio) rh=round_even_positive((double)height/width*rw);
    else if (source_ratio<target_ratio) rw=round_even_positive((double)width/height*rh);
    if (!rw || !rh) return ria_fail(e,RIA_INVALID_REQUEST,"source contain operation produces empty image dimension");
    uint32_t xoff=round_even_positive((grid.pixel_width-rw)*0.5),yoff=round_even_positive((grid.pixel_height-rh)*0.5);
    uint64_t values=(uint64_t)grid.vit_height*grid.vit_width*588,patch_bytes=values*sizeof(float),canvas_bytes=(uint64_t)grid.pixel_height*grid.pixel_width*3;
    uint64_t intermediate_bytes=(uint64_t)rw*height*3;
    uint64_t span_x=(uint64_t)ceil(2*fmax((double)width/rw,1))*2+1,span_y=(uint64_t)ceil(2*fmax((double)height/rh,1))*2+1;
    uint64_t coefficient_bytes=((uint64_t)rw+(uint64_t)rh)*sizeof(resample_row)+((uint64_t)rw*span_x+(uint64_t)rh*span_y)*sizeof(int32_t);
    uint64_t peak=patch_bytes+grid.token_count+canvas_bytes+intermediate_bytes+coefficient_bytes;
    if (!budget || peak>budget || patch_bytes>SIZE_MAX || intermediate_bytes>SIZE_MAX)
        return ria_fail(e,RIA_RESOURCE_LIMIT,"source image preprocessing peak exceeds host budget");
    uint8_t *horizontal=malloc((size_t)intermediate_bytes),*canvas=malloc((size_t)canvas_bytes);resample_plan xp={0},yp={0};
    out->patches=malloc((size_t)patch_bytes);out->types=malloc(grid.token_count);
    if (!horizontal || !canvas || !out->patches || !out->types) { (void)ria_fail(e,RIA_RESOURCE_LIMIT,"allocate source image preprocessing failed");goto bad; }
    if (!resample_make(width,rw,&xp,e) || !resample_make(height,rh,&yp,e)) goto bad;
    for (uint32_t y=0;y<height;++y) for (uint32_t x=0;x<rw;++x) for (unsigned channel=0;channel<3;++channel) {
        const resample_row *row=&xp.rows[x];int64_t sum=2097152;
        if (rw==width) horizontal[((uint64_t)y*rw+x)*3+channel]=rgb[((uint64_t)y*width+x)*3+channel];
        else {
            for (uint32_t k=0;k<row->count;++k) sum+=(int64_t)rgb[((uint64_t)y*width+row->source+k)*3+channel]*row->weights[k];
            horizontal[((uint64_t)y*rw+x)*3+channel]=resample_byte(sum);
        }
    }
    memset(canvas,127,(size_t)canvas_bytes);
    for (uint32_t y=0;y<rh;++y) for (uint32_t x=0;x<rw;++x) for (unsigned channel=0;channel<3;++channel) {
        const resample_row *row=&yp.rows[y];int64_t sum=2097152;uint8_t value;
        if (rh==height) value=horizontal[((uint64_t)y*rw+x)*3+channel];
        else {
            for (uint32_t k=0;k<row->count;++k) sum+=(int64_t)horizontal[((uint64_t)(row->source+k)*rw+x)*3+channel]*row->weights[k];
            value=resample_byte(sum);
        }
        canvas[((uint64_t)(y+yoff)*grid.pixel_width+x+xoff)*3+channel]=value;
    }
    uint64_t index=0;
    for (uint32_t py=0;py<grid.vit_height;++py) for (uint32_t px=0;px<grid.vit_width;++px)
        for (unsigned channel=0;channel<3;++channel) for (unsigned y=0;y<14;++y) for (unsigned x=0;x<14;++x) {
            float pixel=(float)canvas[((uint64_t)(py*14+y)*grid.pixel_width+px*14+x)*3+channel];
            out->patches[index++]=ria_num_bf16(((pixel/255.0f)-0.5f)/0.5f);
        }
    uint32_t token=0;out->types[token++]=0;
    for (uint32_t y=0;y<grid.llm_height;++y) { for (uint32_t x=0;x<grid.llm_width;++x) out->types[token++]=1;out->types[token++]=2; }
    out->types[token++]=3;out->grid=grid;out->host_bytes=patch_bytes+grid.token_count;
    free(horizontal);free(canvas);resample_free(&xp);resample_free(&yp);return true;
bad:free(horizontal);free(canvas);resample_free(&xp);resample_free(&yp);ria_image_input_free(out);return false;
}

typedef struct { const uint8_t *input;size_t length,offset;uint8_t *rgb;png_bytep *rows;int code; } png_input;
static void read_png(png_structp png,png_bytep destination,png_size_t count) {
    png_input *input=png_get_io_ptr(png);
    if (count>input->length-input->offset) png_error(png,"truncated input");
    memcpy(destination,input->input+input->offset,count);input->offset+=count;
}
static void quiet_png(png_structp png,png_const_charp text) { (void)text;png_longjmp(png,1); }
static void ignore_png_warning(png_structp png,png_const_charp text) { (void)png;(void)text; }
static bool decode_png(const uint8_t *bytes,size_t length,uint64_t budget,uint8_t **rgb,uint64_t *allocation,uint32_t *width,uint32_t *height,ria_error *e) {
    png_input *input=calloc(1,sizeof(*input));if (!input) return ria_fail(e,RIA_RESOURCE_LIMIT,"PNG owner allocation failed");
    png_structp png=png_create_read_struct(PNG_LIBPNG_VER_STRING,NULL,quiet_png,ignore_png_warning);
    png_infop info=png ? png_create_info_struct(png) : NULL;
    if (!png || !info) { png_destroy_read_struct(&png,&info,NULL);free(input);return ria_fail(e,RIA_RESOURCE_LIMIT,"PNG decoder allocation failed"); }
    if (setjmp(png_jmpbuf(png))) { int code=input->code;free(input->rgb);free(input->rows);free(input);png_destroy_read_struct(&png,&info,NULL);return ria_fail(e,code ? code : RIA_INVALID_REQUEST,"invalid/truncated/over-capacity PNG image"); }
    input->input=bytes;input->length=length;png_set_read_fn(png,input,read_png);png_set_user_limits(png,16384,16384);
    /* Metadata is irrelevant to source RGB conversion. Bound ancillary
     * decompression/retention as well as the raster, and skip unknown chunks. */
    png_set_chunk_malloc_max(png,1048576);png_set_chunk_cache_max(png,2);
    png_set_keep_unknown_chunks(png,PNG_HANDLE_CHUNK_NEVER,NULL,0);png_read_info(png,info);
    uint32_t w=png_get_image_width(png,info),h=png_get_image_height(png,info);int type=png_get_color_type(png,info),depth=png_get_bit_depth(png,info);
    bool gray16=type==PNG_COLOR_TYPE_GRAY && depth==16;
    uint64_t required=(uint64_t)w*h*(gray16 ? 5u : 3u)+(uint64_t)h*sizeof(png_bytep);
    if (!w || !h || (uint64_t)w*h>67108864 || required+2359296>budget) { input->code=RIA_RESOURCE_LIMIT;png_error(png,"PNG resource limit"); }
    if (type==PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (type==PNG_COLOR_TYPE_GRAY && depth<8) png_set_expand_gray_1_2_4_to_8(png);
    png_set_strip_alpha(png);
    if (depth==16 && !gray16) png_set_strip_16(png);
    if ((type==PNG_COLOR_TYPE_GRAY || type==PNG_COLOR_TYPE_GRAY_ALPHA) && !gray16) png_set_gray_to_rgb(png);
    (void)png_set_interlace_handling(png);png_read_update_info(png,info);
    size_t row_bytes=png_get_rowbytes(png,info);
    if (row_bytes!=(size_t)w*(gray16 ? 2u : 3u)) png_error(png,"unsupported PNG channel transform");
    input->rgb=malloc((size_t)required-(size_t)h*sizeof(png_bytep));input->rows=malloc((size_t)h*sizeof(png_bytep));
    if (!input->rgb || !input->rows) { input->code=RIA_RESOURCE_LIMIT;png_error(png,"PNG allocation failed"); }
    uint8_t *pixels=input->rgb+(gray16 ? (size_t)w*h*3 : 0);
    for (uint32_t y=0;y<h;++y) input->rows[y]=pixels+(size_t)y*row_bytes;
    png_read_image(png,input->rows);png_read_end(png,info);
    if (gray16) for (uint64_t i=0;i<(uint64_t)w*h;++i) { uint32_t value=(uint32_t)pixels[i*2]*256+pixels[i*2+1];uint8_t v=value>255 ? 255 : (uint8_t)value;input->rgb[i*3]=v;input->rgb[i*3+1]=v;input->rgb[i*3+2]=v; }
    *rgb=input->rgb;*allocation=(uint64_t)w*h*(gray16 ? 5u : 3u);*width=w;*height=h;free(input->rows);free(input);png_destroy_read_struct(&png,&info,NULL);return true;
}
typedef struct { struct jpeg_error_mgr standard;jmp_buf jump;int code; } jpeg_failure;
typedef struct { struct jpeg_decompress_struct decoder;jpeg_failure error;uint8_t *pixels;bool created; } jpeg_input;
static void jpeg_error(j_common_ptr common) {
    jpeg_failure *error=(jpeg_failure *)common->err;
    if (common->err->msg_code==JERR_OUT_OF_MEMORY) error->code=RIA_RESOURCE_LIMIT;
    longjmp(error->jump,1);
}
static void jpeg_quiet(j_common_ptr common) { (void)common; }
static bool decode_jpeg(const uint8_t *bytes,size_t length,uint64_t budget,uint8_t **rgb,uint64_t *allocation,uint32_t *width,uint32_t *height,ria_error *e) {
    jpeg_input *input=calloc(1,sizeof(*input));if (!input) return ria_fail(e,RIA_RESOURCE_LIMIT,"JPEG owner allocation failed");
    input->decoder.err=jpeg_std_error(&input->error.standard);input->error.standard.error_exit=jpeg_error;input->error.standard.output_message=jpeg_quiet;
    if (setjmp(input->error.jump)) { int code=input->error.code;if (input->created) jpeg_destroy_decompress(&input->decoder);free(input->pixels);free(input);return ria_fail(e,code ? code : RIA_INVALID_REQUEST,"invalid/truncated/over-capacity JPEG image"); }
    input->created=true;jpeg_create_decompress(&input->decoder);jpeg_mem_src(&input->decoder,bytes,(unsigned long)length);
    if (jpeg_read_header(&input->decoder,TRUE)!=JPEG_HEADER_OK) jpeg_error((j_common_ptr)&input->decoder);
    uint32_t w=input->decoder.image_width,h=input->decoder.image_height;
    if (!w || !h || w>16384 || h>16384 || (uint64_t)w*h>67108864 ||
        input->decoder.num_components<1 || input->decoder.num_components>4) jpeg_error((j_common_ptr)&input->decoder);
    bool cmyk=input->decoder.jpeg_color_space==JCS_CMYK || input->decoder.jpeg_color_space==JCS_YCCK;
    uint64_t required=(uint64_t)w*h*(cmyk ? 7u : 3u);
    uint64_t coefficient_reservation=(uint64_t)div_up(w,32)*32*div_up(h,32)*32*(uint64_t)input->decoder.num_components*2+4194304;
    if (required+coefficient_reservation>budget) { input->error.code=RIA_RESOURCE_LIMIT;jpeg_error((j_common_ptr)&input->decoder); }
    input->decoder.mem->max_memory_to_use=(long)coefficient_reservation;
    input->decoder.out_color_space=cmyk ? JCS_CMYK : JCS_RGB;jpeg_start_decompress(&input->decoder);
    input->pixels=malloc((size_t)required);if (!input->pixels) { input->error.code=RIA_RESOURCE_LIMIT;jpeg_error((j_common_ptr)&input->decoder); }
    uint8_t *pixels=input->pixels+(cmyk ? (size_t)w*h*3 : 0);
    while (input->decoder.output_scanline<h) { JSAMPROW row=pixels+(size_t)input->decoder.output_scanline*w*(cmyk ? 4u : 3u);(void)jpeg_read_scanlines(&input->decoder,&row,1); }
    jpeg_finish_decompress(&input->decoder);
    if (input->decoder.err->num_warnings) jpeg_error((j_common_ptr)&input->decoder);
    if (cmyk) for (uint64_t i=0;i<(uint64_t)w*h;++i) for (unsigned channel=0;channel<3;++channel) {
        /* Pillow's CMYK decoder inverts libjpeg channels before RGB conversion. */
        unsigned c=255u-pixels[i*4+channel],k=255u-pixels[i*4+3];unsigned value=(255u-c)*(255u-k);
        value=(value+128u+((value+128u)>>8))>>8;input->pixels[i*3+channel]=(uint8_t)value;
    }
    jpeg_destroy_decompress(&input->decoder);*rgb=input->pixels;*allocation=required;*width=w;*height=h;free(input);return true;
}
bool ria_image_prepare(const uint8_t *encoded,uint64_t length,uint32_t maximum,uint64_t budget,ria_image_input *out,ria_error *e) {
    if (!out) return ria_fail(e,RIA_INVALID_REQUEST,"missing prepared image result");
    memset(out,0,sizeof(*out));
    if (!encoded || !length || length>67108864 || !budget || maximum>9216 || !maximum)
        return ria_fail(e,RIA_INVALID_REQUEST,"invalid encoded image bounds/admission");
    uint8_t *rgb=NULL;uint64_t allocation=0;uint32_t w=0,h=0;bool decoded;
    if (length>=8 && !memcmp(encoded,"\211PNG\r\n\032\n",8)) decoded=decode_png(encoded,(size_t)length,budget,&rgb,&allocation,&w,&h,e);
    else if (length>=2 && encoded[0]==255 && encoded[1]==216) decoded=decode_jpeg(encoded,(size_t)length,budget,&rgb,&allocation,&w,&h,e);
    else return ria_fail(e,RIA_INVALID_REQUEST,"image encoding must be JPEG or PNG");
    if (!decoded) return false;
    uint64_t decoded_bytes=(uint64_t)w*h*3;
    bool ok=budget>allocation && ria_image_prepare_rgb(rgb,decoded_bytes,w,h,maximum,budget-allocation,out,e);
    if (budget<=allocation) (void)ria_fail(e,RIA_RESOURCE_LIMIT,"decoded image exhausts preprocessing host budget");
    free(rgb);return ok;
}
struct ria_vision { ria_vision_parameters parameters;ria_vision_cuda *cuda; };
uint64_t ria_vision_metadata_bytes(void) { return sizeof(ria_vision)+ria_vision_cuda_metadata_bytes(); }
static const ria_tensor *tensor(const ria_tensor_store *store,const char *name,ria_error *error) {
    const ria_tensor *t=ria_tensor_name(store,name);
    if (!t) (void)ria_fail(error,RIA_INTEGRITY_ERROR,"required vision tensor missing: %s",name);
    return t;
}
static bool matrix(const ria_tensor_store *store,const char *name,uint64_t n,uint64_t k,ria_expert_matrix *out,ria_error *e) {
    const ria_tensor *t=tensor(store,name,e);
    if (!t || !ria_tensor_matrix(store,t,out,e)) return false;
    if (out->out_features!=n || out->in_features!=k || out->profile!=RIA_EXPERT_BF16)
        return ria_fail(e,RIA_INTEGRITY_ERROR,"vision matrix differs from pinned BF16 shape: %s",name);
    return true;
}
static bool vector(const ria_tensor_store *store,const char *name,uint64_t count,const ria_tensor **out,ria_error *e) {
    const ria_tensor *t=tensor(store,name,e);
    if (!t || t->rank!=1 || t->shape[0]!=count || (strcmp(t->dtype,"F32") && strcmp(t->dtype,"BF16")))
        return ria_fail(e,RIA_INTEGRITY_ERROR,"vision vector differs from pinned shape: %s",name);
    *out=t;return true;
}
bool ria_vision_create(const ria_tensor_store *s,int device,uint32_t patches,uint64_t budget,ria_vision **out,ria_error *e) {
    return ria_vision_create_pooled(s,device,patches,budget,UINT64_MAX,out,e);
}
bool ria_vision_create_pooled(const ria_tensor_store *s,int device,uint32_t patches,uint64_t budget,uint64_t pinned_budget,ria_vision **out,ria_error *e) {
    if (!out || !s || !patches || patches>9216 || !budget) return ria_fail(e,RIA_INVALID_REQUEST,"invalid admitted vision options");
    *out=NULL;ria_vision *v=calloc(1,sizeof(*v));if (!v) return ria_fail(e,RIA_RESOURCE_LIMIT,"vision owner allocation failed");
    ria_vision_parameters *p=&v->parameters;
    if (!matrix(s,"vision.patch_embed.proj.weight",1024,588,&p->patch,e) || !vector(s,"vision.patch_embed.proj.bias",1024,&p->patch_bias,e) ||
        !vector(s,"vision.norm.weight",1024,&p->norm,e) || !matrix(s,"aligner.w1.weight",5120,9216,&p->align1,e) ||
        !matrix(s,"aligner.w2.weight",5120,5120,&p->align2,e) || !vector(s,"aligner.w1.bias",5120,&p->align1_bias,e) ||
        !vector(s,"aligner.w2.bias",5120,&p->align2_bias,e)) goto bad;
    for (unsigned layer=0;layer<32;++layer) {
        char name[160];ria_vision_layer *x=&p->layers[layer];
#define VM(member,suffix,n,k) do { (void)snprintf(name,sizeof(name),"vision.blocks.%u.%s",layer,suffix);if (!matrix(s,name,n,k,&x->member,e)) goto bad; } while (0)
#define VV(member,suffix,n) do { (void)snprintf(name,sizeof(name),"vision.blocks.%u.%s",layer,suffix);if (!vector(s,name,n,&x->member,e)) goto bad; } while (0)
        VM(qkv,"attn.wqkv.weight",3072,1024);VM(output,"attn.wo.weight",1024,1024);
        VM(gate_up,"mlp.w1.weight",5632,1024);VM(down,"mlp.w2.weight",1024,2816);
        VV(norm1,"norm1.weight",1024);VV(norm2,"norm2.weight",1024);
        VV(qkv_bias,"attn.wqkv.bias",3072);VV(output_bias,"attn.wo.bias",1024);
#undef VM
#undef VV
    }
    if (!ria_vision_cuda_create(p,device,patches,budget,pinned_budget,&v->cuda,e)) goto bad;
    *out=v;return true;
bad:{ria_error cleanup;(void)ria_vision_destroy(v,&cleanup);return false;}
}
bool ria_vision_destroy(ria_vision *v,ria_error *e) { if (!v) return true;bool ok=ria_vision_cuda_destroy(v->cuda,e);free(v);return ok; }
uint64_t ria_vision_device_bytes(const ria_vision *v) { return v ? ria_vision_cuda_bytes(v->cuda) : 0; }
uint64_t ria_vision_pinned_bytes(const ria_vision *v) { return v ? ria_vision_cuda_pinned_bytes(v->cuda) : 0; }
bool ria_vision_encode(ria_vision *v,const float *patches,uint32_t h,uint32_t w,float *output,uint64_t rows,ria_error *e) {
    if (!v) return ria_fail(e,RIA_INVALID_REQUEST,"missing vision context");
    return ria_vision_cuda_encode(v->cuda,patches,h,w,output,rows,e);
}
