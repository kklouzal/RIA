#ifndef RIA_PREFILL_H
#define RIA_PREFILL_H
#include "common.h"
#ifdef __cplusplus
extern "C" {
#endif
#define RIA_GRAPH_PREFILL_MAX_ROWS 64u
#define RIA_GRAPH_CANDIDATE_BLOCKS 2048u
/* Protected host snapshots of independently causal rows. Arithmetic remains
 * on CUDA; these widened BF16/FP32 values are copied without requantization.
 * The layout is authoritative for both admission and the graph allocator. */
typedef struct {
    float h[20480],residual[20480],input[5120],shared[5120];
    float pre[4],ffn_pre[4],ffn_post[4],ffn_comb[16];
    float contributions[6*5120],coefficients[6];
    int64_t history[4];
    uint32_t selected[512],candidate_blocks[RIA_GRAPH_CANDIDATE_BLOCKS];
    uint32_t selected_count,candidate_count;
    uint16_t ids[6];
    bool image;
} ria_graph_prefill_row;
/* Exact extra page-rounded private host allocation; exact device increment
 * over a one-row projection context (including candidate-ID storage); total
 * graph pinned transfer pool. No runtime/device initialization occurs here.
 * Vision and existing graph/state/device pools remain separately accounted. */
bool ria_graph_prefill_required_bytes(uint32_t rows,uint32_t projection_tile_rows,
                                      uint64_t *host,uint64_t *device,uint64_t *pinned,ria_error *);
#ifdef __cplusplus
}
#endif
#endif
