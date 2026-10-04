#ifndef RIA_GRAPH_CUDA_H
#define RIA_GRAPH_CUDA_H
#include "graph.h"
#include "expert_cuda.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ria_graph_cuda ria_graph_cuda;
/* Pure startup predicate, also exercised by offline fixtures. The admitted
 * physical UUID comes from the inventory owner; CUDA reports one visible
 * device and its concrete product, architecture and UUID here. */
bool ria_graph_client_device_validate(int count,int device,int major,int minor,
                                     const char *name,const char *uuid,const char *expected_uuid,ria_error *);
enum {
 RIA_G_H,RIA_G_RESIDUAL,RIA_G_INPUT,RIA_G_OUTPUT,RIA_G_PRE,
 RIA_G_ATT_PRE,RIA_G_ATT_POST,RIA_G_ATT_COMB,RIA_G_FFN_PRE,RIA_G_FFN_POST,RIA_G_FFN_COMB,
 RIA_G_MIX,RIA_G_QR,RIA_G_Q,RIA_G_KV,RIA_G_LATENT,RIA_G_INDEX_Q,RIA_G_INDEX_WEIGHTS,
 RIA_G_ATTN_KV,RIA_G_ATTN_OUT,RIA_G_OLOW,RIA_G_ROUTER,RIA_G_EXPERT_RESULTS,RIA_G_SHARED,
 RIA_G_ENGRAM_INPUT,RIA_G_ENGRAM_KV,RIA_G_LOGITS,RIA_G_INDEX_TILE,RIA_G_POOL_KV,RIA_G_POOL_GATE,
 RIA_G_BUFFER_COUNT
};
bool ria_graph_cuda_create(const ria_graph_options *,ria_graph_cuda **,ria_error *);
bool ria_graph_cuda_destroy(ria_graph_cuda *,ria_error *);
uint64_t ria_graph_cuda_bytes(const ria_graph_cuda *);
uint64_t ria_graph_cuda_metadata_bytes(void);
uint64_t ria_graph_cuda_device_bytes(const ria_graph_cuda *);
uint64_t ria_graph_cuda_pinned_bytes(const ria_graph_cuda *);
float *ria_graph_cuda_buffer(ria_graph_cuda *,unsigned buffer);
void *ria_graph_cuda_stream(ria_graph_cuda *);
bool ria_graph_cuda_reset(ria_graph_cuda *,ria_error *);
/* Every source forward starts from an identity pre-mix. This resets only
 * per-token mHC coefficients, retaining attention/index/compressor history. */
bool ria_graph_cuda_begin_step(ria_graph_cuda *,ria_error *);
bool ria_graph_cuda_drain(ria_graph_cuda *,ria_error *);
bool ria_graph_cuda_copy(ria_graph_cuda *,float *destination,const float *source,uint64_t count,ria_error *);
bool ria_graph_cuda_upload(ria_graph_cuda *,float *destination,const float *source,uint64_t count,ria_error *);
bool ria_graph_cuda_download(ria_graph_cuda *,const float *source,float *destination,uint64_t count,ria_error *);
bool ria_graph_cuda_tensor(ria_graph_cuda *,const ria_tensor *,uint64_t first,float *,uint64_t count,ria_error *);
bool ria_graph_cuda_project(ria_graph_cuda *,const ria_expert_matrix *,const float *,float *,bool,ria_error *);
bool ria_graph_cuda_norm(ria_graph_cuda *,float *,uint64_t rows,uint64_t width,const ria_tensor *weight,bool layer_norm,ria_error *);
bool ria_graph_cuda_expand(ria_graph_cuda *,const float *,float *,ria_error *);
bool ria_graph_cuda_mhc(ria_graph_cuda *,const float *,float *,const ria_tensor *scale,const ria_tensor *base,
                        float *pre,float *post,float *comb,ria_error *);
bool ria_graph_cuda_collapse(ria_graph_cuda *,const float *,const float *pre,float *,ria_error *);
bool ria_graph_cuda_post(ria_graph_cuda *,const float *,const float *,const float *,const float *,float *,ria_error *);
bool ria_graph_cuda_rope(ria_graph_cuda *,float *,uint64_t rows,uint64_t width,uint64_t position,uint32_t ratio,bool inverse,ria_error *);
/* Representation 1: E4M3/UE8M0 act32 window; 2: E2M1/E4M3 group16
 * compressed KV; 3: E2M1/UE8M0 group32 index. Packed history is transferred
 * verbatim, decoded/rounded exclusively on CUDA. */
uint64_t ria_graph_packed_stride(uint32_t representation,uint64_t width);
bool ria_graph_cuda_pack(ria_graph_cuda *,const float *,uint64_t width,uint32_t representation,uint8_t *host,ria_error *);
bool ria_graph_cuda_quantize_inplace(ria_graph_cuda *,float *,uint64_t width,uint32_t representation,ria_error *);
bool ria_graph_cuda_unpack(ria_graph_cuda *,const uint8_t *host,uint64_t rows,uint64_t width,uint32_t representation,float *,ria_error *);
bool ria_graph_cuda_pool(ria_graph_cuda *,float *kv,float *scores,uint32_t ratio,float *latent,ria_error *);
bool ria_graph_cuda_index_scores(ria_graph_cuda *,const float *q,const float *weights,const float *keys,
                                 uint64_t first,uint64_t count,ria_error *);
/* Qualification-only observation of the actual learned index kernel output;
 * does not require a model loader or modify source candidate/state semantics. */
bool ria_graph_cuda_index_download(ria_graph_cuda *,uint64_t first,uint64_t count,float *host,ria_error *);
bool ria_graph_cuda_select(ria_graph_cuda *,uint64_t count,bool candidate_source,bool uses_candidates,
                           uint32_t *selected,uint32_t *selected_count,ria_error *);
/* Preserve source sparse-slot order, including [masked_begin,masked_end)
 * holes: its online-softmax BF16 boundary is scoped to each 64-slot tile. */
bool ria_graph_cuda_attention(ria_graph_cuda *,const float *q,const float *kv,uint64_t count,uint64_t masked_begin,uint64_t masked_end,
                              const ria_tensor *sink,float *output,ria_error *);
bool ria_graph_cuda_route(ria_graph_cuda *,const float *scores,const ria_tensor *bias,
                          uint16_t ids[6],float coefficients[6],ria_error *);
bool ria_graph_cuda_merge(ria_graph_cuda *,const uint16_t ids[6],const float *contributions,
                          const float *shared,float *output,ria_error *);
bool ria_graph_cuda_engram(ria_graph_cuda *,const uint8_t *rows,float *features,ria_error *);
bool ria_graph_cuda_engram_fuse(ria_graph_cuda *,float *h,const float *kv,const ria_tensor *qweight,
                                const ria_tensor *kweight,bool enabled,ria_error *);
bool ria_graph_cuda_shared(ria_graph_cuda *,const ria_expert *,const float *,float *,ria_error *);
bool ria_graph_cuda_local_create(ria_graph_cuda *,const ria_expert *,uint64_t budget,ria_expert_cuda_resident *,ria_error *);
bool ria_graph_cuda_local_evaluate(ria_graph_cuda *,const ria_expert *,const ria_expert_cuda_resident *,
                                  const float *,float coefficient,float *host_contribution,ria_error *);
#ifdef __cplusplus
}
#endif
#endif
