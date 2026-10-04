#ifndef RIA_GRAPH_H
#define RIA_GRAPH_H
#include "tensor.h"
#include "state.h"
#ifdef __cplusplus
extern "C" {
#endif
#define RIA_GRAPH_LAYERS 40u
#define RIA_GRAPH_DIM 5120u
#define RIA_GRAPH_VOCAB 129280u
#define RIA_GRAPH_EXPERTS 384u
#define RIA_GRAPH_SELECTED 6u
#define RIA_GRAPH_HASH_COLUMNS 24u
typedef enum { RIA_GRAPH_PREFILL=0,RIA_GRAPH_DECODE=1,RIA_GRAPH_CONTINUATION=2 } ria_graph_phase;
typedef enum { RIA_GRAPH_HOST=1,RIA_GRAPH_VRAM=2 } ria_graph_tier;
typedef enum { RIA_GRAPH_WINDOW=0,RIA_GRAPH_FULL=1,RIA_GRAPH_REINDEX=2,RIA_GRAPH_REUSE=3 } ria_graph_attention_kind;
typedef struct {
    uint32_t layer;
    uint16_t expert_id;
    uint8_t tier,phase_mask; /* bit 0=prefill, 1=decode, 2=continuation */
    ria_expert expert;      /* complete verified immutable host triplet */
} ria_graph_local_expert;

typedef struct {
    void *context;
    /* Ordered selected slots, original IDs/coefficients. Returns exactly
     * count BF16-rounded widened FP32 contributions in the same slot order.
     * This callback performs remote arithmetic or explicitly local CUDA work;
     * a client miss never fetches weights and never executes CPU experts. */
    bool (*experts)(void *,uint32_t layer,const float input[RIA_GRAPH_DIM],
                    const uint16_t *ids,const float *coefficients,const uint16_t *original_slots,uint32_t count,
                    float *contributions,ria_error *);
    /* Full lossless packed Engram row: 256 E4M3 bytes + 8 UE8M0 bytes.
     * IDs are original global table rows; response preserves association. */
    bool (*engram)(void *,uint32_t layer,const uint64_t *ids,uint32_t count,
                   uint8_t *packed_rows,ria_error *);
} ria_graph_remote;
typedef struct { uint64_t row;uint32_t layer;uint8_t packed[264];bool valid; } ria_graph_row_cache_entry;
/* Lossless immutable-model cache. Caller owns zeroed entries and serializes
 * access; cache must never be reused across model/table identities. Collisions
 * cause bounded replacement only. Returned bytes preserve request association. */
bool ria_graph_rows_fetch(ria_graph_remote,uint32_t layer,const uint64_t *ids,uint32_t count,
                          ria_graph_row_cache_entry *cache,uint64_t capacity,uint8_t *packed,ria_error *);

typedef struct {
    int device;
    const char *gpu_uuid;
    uint64_t max_tokens,host_state_budget,device_budget,pinned_budget;
    uint32_t projection_tile_rows,state_tile_rows,max_image_patches;
    /* Sorted unique (layer,expert_id) explicit membership. Caller-owned
     * descriptors/bytes outlive graph. No miss fetch or implicit promotion.
     * Host budget counts local logical values/scales; parent TensorStore also
     * accounts actual owned shard/metadata/overfetch bytes in its host cap. */
    const ria_graph_local_expert *local_experts;
    uint32_t local_expert_count;
    uint64_t host_expert_budget,device_expert_budget,engram_cache_budget;
    /* Exact prepared hash constants. Runtime does not approximate the
     * tokenizer's Unicode normalization or substitute another RNG. */
    const uint32_t *compressed_token_map;
    uint32_t compressed_vocab,pad_compressed_id;
    uint64_t hash_multipliers[2][4],hash_primes[2][RIA_GRAPH_HASH_COLUMNS];
} ria_graph_options;

typedef struct ria_graph ria_graph;
bool ria_graph_options_validate(const ria_graph_options *options,ria_error *error);
bool ria_graph_local_experts_validate(const ria_graph_options *,uint64_t *host_bytes,uint64_t *device_bytes,ria_error *);
/* Pure selected-slot partition; preserves original slot order. Local entries
 * disabled for this explicit phase use the remote owner. -1 means remote. */
bool ria_graph_partition(const ria_graph_options *,uint32_t layer,ria_graph_phase,const uint16_t *ids,uint32_t count,
                         int32_t *local_indices,uint8_t *remote_slots,uint32_t *remote_count,ria_error *);
/* Actual page-rounded private host allocation, including the session owner.
 * The store and caller-owned immutable Engram metadata are separate pools. */
bool ria_graph_host_state_required(uint64_t max_tokens,uint64_t *bytes,ria_error *error);
/* Includes current local-VRAM descriptor metadata and the complete optional
 * packed Engram-row cache; immutable local source bytes remain in TensorStore. */
bool ria_graph_host_required_bytes(const ria_graph_options *,uint64_t *bytes,ria_error *);
/* Extra page-locked CUDA transfer pools, separate from ordinary private-state
 * mlock. Reserve in both the host cap and the dedicated pinned cap. */
bool ria_graph_pinned_required_bytes(const ria_graph_options *,uint64_t *bytes,ria_error *);
/* Logical graph aliases: source arrays are immutable pinned configuration;
 * each consumer refers to the latest preceding source, never its own copy. */
bool ria_graph_dependencies(uint32_t layer,uint32_t *kv_source,uint32_t *index_source,
                             uint32_t *ratio,ria_error *error);
bool ria_graph_attention_kind_for_layer(uint32_t layer,ria_graph_attention_kind *,ria_error *);
bool ria_graph_hash(const ria_graph_options *options,uint32_t engram_index,
                     const int64_t history[4],uint64_t ids[RIA_GRAPH_HASH_COLUMNS],ria_error *error);
bool ria_graph_create(const ria_tensor_store *store,const ria_graph_options *options,
                       ria_graph_remote remote,ria_graph **out,ria_error *error);
/* On failed CUDA drain/release, retains the poisoned owner/backing. The
 * serving owner must terminate before freeing borrowed TensorStore operands. */
bool ria_graph_destroy(ria_graph *graph,ria_error *error);
/* Reset drains CUDA, clears all private source caches/partial groups/history;
 * no snapshot or truncated replay. Epoch/generation identify this session. */
bool ria_graph_reset(ria_graph *graph,uint64_t epoch,uint64_t generation,ria_error *error);
/* Caller proves identical incorporated token/image prefix and quiescent remote
 * requests before retagging. All packed history and incomplete pools survive. */
bool ria_graph_retag(ria_graph *graph,uint64_t epoch,uint64_t generation,ria_error *error);
/* Caller declares initial/continued prefill versus ordinary decoding. Changes
 * only placement decisions, never numerical/source state or membership. */
bool ria_graph_set_phase(ria_graph *,ria_graph_phase,ria_error *);
uint64_t ria_graph_position(const ria_graph *graph);
uint64_t ria_graph_host_bytes(const ria_graph *graph);
uint64_t ria_graph_device_bytes(const ria_graph *graph);
uint64_t ria_graph_pinned_bytes(const ria_graph *graph);
/* One authoritative causal position. logits is host FP32[vocab]. Optional
 * image_embedding is host BF16-logical [dim], from ria_graph_encode_image or a
 * learned delimiter row; image_span disables Engram and selects VL bias.
 * Successful return advances incorporated position. Failure poisons session. */
bool ria_graph_step(ria_graph *graph,uint32_t token,bool image_span,
                     const float *image_embedding,float logits[RIA_GRAPH_VOCAB],ria_error *error);
/* Exact causal token-at-a-time prefill; rows are bounded within one session.
 * Its operator schedule matches source seqlen=1 forwards: an initial one-slot
 * window, then decode-form 128 sparse ring slots with invalid holes retained
 * through the 64-slot BF16 attention probability boundary. Optional logits
 * receives every position for teacher-forced validation. This does not yet
 * implement the required bounded per-layer grouped-row prefill scheduler. */
bool ria_graph_prefill(ria_graph *graph,const uint32_t *tokens,uint64_t count,
                        float *last_logits,float *all_logits,ria_error *error);
/* patches are source-normalized [n_vit_h*n_vit_w,3*14*14] host floats.
 * output is row-major [ceil(h/3)*ceil(w/3),dim] host BF16-logical values. */
bool ria_graph_encode_image(ria_graph *graph,const float *patches,uint32_t n_vit_h,
                             uint32_t n_vit_w,float *output,uint64_t output_rows,ria_error *error);
/* Pinned source token types: 0=start, 2=newline, 3=end; 1 is an image row. */
bool ria_graph_image_delimiter(ria_graph *graph,uint32_t type,float output[RIA_GRAPH_DIM],ria_error *error);
#ifdef __cplusplus
}
#endif
#endif
