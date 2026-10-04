#ifndef RIA_VISION_H
#define RIA_VISION_H
#include "graph.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ria_vision ria_vision;
typedef struct {
    uint32_t pixel_height,pixel_width,vit_height,vit_width,llm_height,llm_width,token_count;
} ria_image_grid;
typedef struct {
    ria_image_grid grid;
    float *patches; /* [vit_height*vit_width,3,14,14], BF16 widened FP32 */
    uint8_t *types; /* source IDs 0=start,1=image,2=newline,3=end */
    uint64_t host_bytes;
} ria_image_input;
/* Pinned source min_pixels=544^2, patch=14, downsample=3, token cap=1024.
 * JPEG/PNG are decoded without EXIF rotation or alpha compositing, as source
 * Image.open(...).convert("RGB"). Native bicubic implements Pillow 10.2's
 * separable 22-bit coefficient/8-bit epilogues, then centered gray-127 pad.
 * All work is bounded preprocessing; no CPU neural operation is performed. */
bool ria_image_plan(uint32_t width,uint32_t height,uint32_t max_patches,ria_image_grid *,ria_error *);
bool ria_image_prepare(const uint8_t *encoded,uint64_t length,uint32_t max_patches,
                        uint64_t host_budget,ria_image_input *,ria_error *);
bool ria_image_prepare_rgb(const uint8_t *rgb,uint64_t length,uint32_t width,uint32_t height,
                            uint32_t max_patches,uint64_t host_budget,ria_image_input *,ria_error *);
void ria_image_input_free(ria_image_input *);
bool ria_vision_create(const ria_tensor_store *,int device,uint32_t max_patches,uint64_t device_budget,
                       ria_vision **,ria_error *);
bool ria_vision_create_pooled(const ria_tensor_store *,int device,uint32_t max_patches,uint64_t device_budget,
                             uint64_t pinned_budget,ria_vision **,ria_error *);
bool ria_vision_destroy(ria_vision *,ria_error *);
uint64_t ria_vision_device_bytes(const ria_vision *);
uint64_t ria_vision_pinned_bytes(const ria_vision *);
/* Pure ordinary-host owner allocation, separate from the pinned pool. */
uint64_t ria_vision_metadata_bytes(void);
bool ria_vision_encode(ria_vision *,const float *,uint32_t height,uint32_t width,
                       float *output,uint64_t output_rows,ria_error *);
/* Native implementation boundary, immutable startup bindings. */
typedef struct {
    ria_expert_matrix qkv,output,gate_up,down;
    const ria_tensor *norm1,*norm2,*qkv_bias,*output_bias;
} ria_vision_layer;
typedef struct {
    ria_expert_matrix patch,align1,align2;
    const ria_tensor *patch_bias,*norm,*align1_bias,*align2_bias;
    ria_vision_layer layers[32];
} ria_vision_parameters;
typedef struct ria_vision_cuda ria_vision_cuda;
bool ria_vision_cuda_create(const ria_vision_parameters *,int,uint32_t,uint64_t,uint64_t,ria_vision_cuda **,ria_error *);
bool ria_vision_cuda_destroy(ria_vision_cuda *,ria_error *);
uint64_t ria_vision_cuda_bytes(const ria_vision_cuda *);
uint64_t ria_vision_cuda_pinned_bytes(const ria_vision_cuda *);
uint64_t ria_vision_cuda_metadata_bytes(void);
bool ria_vision_cuda_encode(ria_vision_cuda *,const float *,uint32_t,uint32_t,float *,uint64_t,ria_error *);
#ifdef __cplusplus
}
#endif
#endif
