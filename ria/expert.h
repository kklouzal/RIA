#ifndef RIA_EXPERT_H
#define RIA_EXPERT_H
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { RIA_EXPERT_BF16=1, RIA_EXPERT_FP8=2, RIA_EXPERT_NVFP4=3,
               RIA_EXPERT_F32=4 } ria_expert_profile;

/* Immutable host bytes, borrowed until all evaluation completes. Values are
 * row-major [out_features,in_features], little endian, untransposed. NVFP4
 * packs adjacent K values low nibble first; an odd final nibble is zero.
 * Byte strides support independent physical layouts and offsets beyond 4 GiB.
 * FP8 scale rows are N/32 blocks; NVFP4 scale rows are individual N rows.
 * Calibration/global factors are DEQUANT multipliers, frozen during preparation
 * over the manifest's full original population, never recomputed by this API. */
typedef struct {
    ria_expert_profile profile;
    uint64_t out_features, in_features;
    const uint8_t *values;
    uint64_t values_bytes, value_row_stride;
    const uint8_t *scales;
    uint64_t scales_bytes, scale_row_stride;
    float weight_global_scale;
    float activation_global_scale;
} ria_expert_matrix;

typedef struct {
    ria_expert_matrix gate, up, down;
    float clamp; /* zero disables; positive: gate <= clamp, up in [-clamp,clamp] */
} ria_expert;

typedef struct ria_expert_cpu ria_expert_cpu;

/* Full byte/code scan: run once when admitting immutable verified tensors.
 * Evaluation requires a successfully validated, unchanged descriptor. */
bool ria_expert_validate(const ria_expert *expert, ria_error *error);
bool ria_expert_matrix_validate(const ria_expert_matrix *matrix, ria_error *error);
bool ria_expert_matrix_descriptor_validate(const ria_expert_matrix *matrix, ria_error *error);
bool ria_expert_cpu_create(uint64_t max_input, uint64_t max_intermediate,
                          uint64_t max_output, ria_expert_cpu **out, ria_error *error);
/* Scratch byte requirement excludes the small context metadata. create_in
 * borrows a caller-owned float-aligned arena already bound/locked according to
 * server placement; no arena byte is observed before it is overwritten.
 * The arena outlives the nonreentrant context and destroy never frees it. */
bool ria_expert_cpu_required_bytes(uint64_t max_input,uint64_t max_intermediate,
                                    uint64_t max_output,uint64_t *bytes,ria_error *error);
bool ria_expert_cpu_create_in(uint64_t max_input,uint64_t max_intermediate,
                               uint64_t max_output,void *workspace,uint64_t bytes,
                               ria_expert_cpu **out,ria_error *error);
void ria_expert_cpu_destroy(ria_expert_cpu *context);
const char *ria_expert_cpu_kernel(const ria_expert_cpu *context);
uint64_t ria_expert_cpu_workspace_bytes(const ria_expert_cpu *context);
uint64_t ria_expert_cpu_metadata_bytes(void);
/* Pure nonempty byte-span predicate shared by host expert boundaries. Returns
 * false for overlap or unrepresentable address arithmetic; never accesses the
 * pointed-to storage. Callers prove each logical strided extent first. */
bool ria_expert_ranges_disjoint(const void *a,uint64_t a_bytes,const void *b,uint64_t b_bytes);

/* Context is caller-owned, nonreentrant; FP32 nearest-even/denormal preserving
 * math is required and checked at operation boundaries. Build without fast
 * math and with contraction disabled; explicitly requested dot FMA remains.
 * Reproducibility is repeated invocation in the same build/environment, with
 * numerical tolerances required across other runtimes/devices/tilings.
 * Input/output strides count float
 * elements. Wire FP32 inputs first round to logical BF16, projection outputs
 * round to BF16, nonlinear/clamp/coefficient work stays FP32, the down input
 * rounds to BF16 BEFORE the profile quantizer. Each result is BF16-rounded and
 * widened to FP32. NULL coefficients denotes the shared branch (weight 1).
 * Host input/coefficient/output ranges must be disjoint. On failure output is
 * incomplete and MUST NOT be published or accumulated. */
bool ria_expert_cpu_evaluate(ria_expert_cpu *context, const ria_expert *expert,
                             const float *input, uint64_t rows, uint64_t input_stride,
                             const float *coefficients, float *output,
                             uint64_t output_stride, ria_error *error);
/* Dense diagnostic/client-contract fixture operation. FP32 weight views retain
 * FP32 inputs/outputs when round_bf16=false; low-bit/BF16 inputs round to BF16.
 * Input/output are host pointers here. Context admits K <= max(input,mid),
 * N <= max(mid,output). */
bool ria_expert_cpu_projection(ria_expert_cpu *context, const ria_expert_matrix *matrix,
                               const float *input, uint64_t rows, uint64_t input_stride,
                               float *output, uint64_t output_stride,
                               bool round_bf16, ria_error *error);

/* Numeric helpers also used by preparation/oracle byte fixtures. Nonfinite
 * inputs are errors in evaluation; E4M3 encoding returns canonical NaN.
 * E2M1 has no NaN code and its encoding helper requires a non-NaN input.
 * E4M3FN/E2M1 use round-to-nearest ties-to-even with finite saturation. */
float ria_expert_bf16_round(float value);
float ria_expert_e4m3_decode(uint8_t code);
uint8_t ria_expert_e4m3_encode(float value);
float ria_expert_e2m1_decode(uint8_t code);
uint8_t ria_expert_e2m1_encode(float value);
float ria_expert_ue8m0_decode(uint8_t code);
/* Input first rounds to BF16. decoded has count floats; optional FP8 codes has
 * count bytes, NVFP4 codes ceil(count/2) bytes with zero odd-nibble padding;
 * optional scales has ceil(count/32) FP8 or ceil(count/16) NVFP4 bytes.
 * BF16 ignores codes/scales. Group boundaries start at input[0]; splitting
 * inside a group changes the contract and is forbidden by the caller. */
bool ria_expert_quantize(const float *input, uint64_t count, ria_expert_profile profile,
                          float activation_global_scale, float *decoded,
                          uint8_t *codes, uint8_t *scales, ria_error *error);

#ifdef __cplusplus
}
#endif
#endif
