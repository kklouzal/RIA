#ifndef RIA_EXPERT_CUDA_H
#define RIA_EXPERT_CUDA_H
#include "expert.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ria_expert_cuda ria_expert_cuda;
/* Startup identity gate, before any owned device/pinned allocation. Exactly
 * one visible device, device zero, physical SM120 and canonical locked UUID.
 * A nonnull product name additionally requires that exact inventory product;
 * qualified expert servers may admit another explicitly supported SM120 SKU.
 * This checks metadata only and never executes a capability kernel. */
bool ria_expert_cuda_device_require(int device,const char *expected_uuid,
                                    const char *expected_name,ria_error *);
/* Implementation-owned immutable view, stored in the graph's locked private
 * metadata arena. Caller zero-initializes and never changes these fields. */
typedef struct {
    ria_expert expert;
    ria_expert_cuda *owner;
    uint64_t bytes;
} ria_expert_cuda_resident;
/* Logical packed device population, without host source/pinned scratch. */
bool ria_expert_cuda_resident_required_bytes(const ria_expert *,uint64_t *bytes,ria_error *);
/* Immutable startup residency, caller-owned until destroy. Uses context's
 * pinned pool, no full-bank registration or mutable automatic eviction. */
bool ria_expert_cuda_resident_create(ria_expert_cuda *,const ria_expert *,uint64_t budget,
                                    ria_expert_cuda_resident *,ria_error *);
/* A failed wait/release preserves the view for process termination; callers
 * must not release borrowed host operands or attempt executor recovery. */
bool ria_expert_cuda_resident_destroy(ria_expert_cuda_resident *,ria_error *);
uint64_t ria_expert_cuda_resident_bytes(const ria_expert_cuda_resident *);
/* DEVICE input/output [5120] generic selected expert, coefficient host scalar.
 * A null resident uses the verified host triplet with bounded pinned staging.
 * Resident must belong to this context; all work completes before return. */
bool ria_expert_cuda_evaluate_device(ria_expert_cuda *,const ria_expert *,const ria_expert_cuda_resident *,
                                    const float *input,float coefficient,float *output,ria_error *);
/* Pure host arithmetic; no CUDA initialization. Page-rounded pinned bytes for
 * one reusable gather/transfer pool, counted separately from ordinary mlock. */
bool ria_expert_cuda_pinned_required_bytes(uint64_t max_input,uint64_t max_intermediate,
                                         uint64_t max_output,uint64_t max_rows,uint64_t tile_rows,
                                         uint64_t *bytes,ria_error *error);
/* Every allocation is fixed by these admitted bounds; only one output-channel
 * tile is staged, never an expert layer or whole bank. No CPU math fallback.
 * Production BF16/FP8/NVFP4 use native SM120a MMA; FP32-sensitive matrices
 * retain strict FP32 GPU FMA. FP8 scales apply at each K32 FP32 boundary;
 * native NVFP4 uses actual E4M3 metadata per independently accumulated K16.
 * This compiled instruction path still requires explicit physical-device
 * numerical/performance qualification; context creation never runs probes.
 * Context owns a nonblocking stream and is nonreentrant. */
bool ria_expert_cuda_create(int device,uint64_t max_input,uint64_t max_intermediate,
                            uint64_t max_output,uint64_t max_rows,uint64_t tile_rows,
                            ria_expert_cuda **out,ria_error *error);
/* Enforces the complete owned workspace reservation before each allocation,
 * including the small owner metadata, for graph/service hard admission caps. */
bool ria_expert_cuda_create_limited(int device,uint64_t max_input,uint64_t max_intermediate,
                                   uint64_t max_output,uint64_t max_rows,uint64_t tile_rows,
                                   uint64_t workspace_budget,ria_expert_cuda **out,ria_error *error);
bool ria_expert_cuda_create_pooled(int device,uint64_t max_input,uint64_t max_intermediate,
                                  uint64_t max_output,uint64_t max_rows,uint64_t tile_rows,
                                  uint64_t workspace_budget,uint64_t pinned_budget,
                                  ria_expert_cuda **out,ria_error *error);
/* A failed drain/release retains the context and live pinned/device owners.
 * The policy owner terminates the process before releasing source operands. */
bool ria_expert_cuda_destroy(ria_expert_cuda *context,ria_error *error);
uint64_t ria_expert_cuda_workspace_bytes(const ria_expert_cuda *context);
uint64_t ria_expert_cuda_metadata_bytes(void);
/* Physical cudaMalloc bytes; workspace_bytes also reserves owner metadata. */
uint64_t ria_expert_cuda_device_bytes(const ria_expert_cuda *context);
uint64_t ria_expert_cuda_pinned_bytes(const ria_expert_cuda *context);
/* Host/device bytes pass through the context's owned pinned pool. Each chunk
 * completes before reuse/return; ordinary host inputs/outputs may be pageable.
 * Context remains nonreentrant; caller has selected this context's device. */
bool ria_expert_cuda_upload_bytes(ria_expert_cuda *,void *device,const void *host,uint64_t bytes,ria_error *);
bool ria_expert_cuda_download_bytes(ria_expert_cuda *,const void *device,void *host,uint64_t bytes,ria_error *);
void *ria_expert_cuda_stream(ria_expert_cuda *context);
/* Host input/output API for the expert service, same semantics as CPU.
 * On return all stream work has completed, including failures. */
bool ria_expert_cuda_evaluate(ria_expert_cuda *context,const ria_expert *expert,
                              const float *input,uint64_t rows,uint64_t input_stride,
                              const float *coefficients,float *output,
                              uint64_t output_stride,ria_error *error);
/* Same host-row contract over one original expert. An explicitly admitted
 * resident view reuses device weights; a host triplet stages each bounded
 * weight tile once for the complete row group. Quantizers remain per-row.
 * Logical strided host input, coefficients and output spans must be disjoint,
 * including padding between rows; overlap is rejected before any CUDA call.
 * Used extents exclude final-row padding; a one-row stride is unused except
 * for the minimum logical width, exactly as on the CPU boundary.
 * On failure, output is unspecified and must not be published. */
bool ria_expert_cuda_evaluate_resident(ria_expert_cuda *,const ria_expert *,const ria_expert_cuda_resident *,
                                      const float *input,uint64_t rows,uint64_t input_stride,
                                      const float *coefficients,float *output,uint64_t output_stride,ria_error *);
/* Client graph API: input/output are contiguous CUDA DEVICE pointers, while
 * the immutable matrix descriptor borrows verified host bytes. Enqueue earlier
 * producers on ria_expert_cuda_stream(), or establish an event dependency.
 * This call synchronizes that stream before returning complete results. */
bool ria_expert_cuda_projection(ria_expert_cuda *context,const ria_expert_matrix *matrix,
                                const float *input,uint64_t rows,float *output,
                                bool round_bf16,ria_error *error);
/* Tiny native BF16 and block-scaled E4M3/UE8M0 and E2M1/E4M3 operations.
 * Compile for sm_120a; explicitly invoked hardware qualification, never startup
 * implicit execution. Compilation is not a passed capability/accuracy probe. */
bool ria_expert_cuda_native_probe(ria_expert_cuda *context,float results[3],ria_error *error);
#ifdef __cplusplus
}
#endif
#endif
