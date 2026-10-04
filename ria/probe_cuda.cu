#include "expert_cuda.h"
#include "probe.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>

static bool checked(cudaError_t status, const char *operation, ria_error *e) {
  if (status == cudaSuccess)
    return true;
  return ria_fail(e, RIA_EXECUTOR_ERROR, "%s: %s", operation,
                  cudaGetErrorString(status));
}
bool ria_probe_gpu(const ria_probe_config *c, ria_probe_gpu_result *r,
                   ria_error *e) {
  memset(r, 0, sizeof(*r));
  int count = 0;
  cudaDeviceProp property;
  size_t free_bytes = 0, total_bytes = 0;
  if (!checked(cudaGetDeviceCount(&count), "inspect selected devices", e) ||
      count != 1 ||
      !checked(cudaGetDeviceProperties(&property, 0),
               "inspect device capability", e) ||
      property.major != 12 || property.minor != 0)
    return e->code ? false
                   : ria_fail(e, RIA_UNSUPPORTED,
                              "CUDA image requires one physical "
                              "compute-capability-12.0 GPU");
  const unsigned char *u = (const unsigned char *)property.uuid.bytes;
  snprintf(r->uuid, sizeof(r->uuid),
           "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%"
           "02x%02x",
           u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10],
           u[11], u[12], u[13], u[14], u[15]);
  if (strcasecmp(r->uuid, c->gpu_uuid))
    return ria_fail(e, RIA_IDENTITY_MISMATCH,
                    "visible physical GPU UUID differs from the probe grant");
  if (!checked(cudaSetDevice(0), "select device", e) ||
      !checked(cudaDriverGetVersion(&r->driver_version),
               "inspect driver version", e) ||
      !checked(cudaRuntimeGetVersion(&r->runtime_version),
               "inspect runtime version", e) ||
      !checked(cudaMemGetInfo(&free_bytes, &total_bytes),
               "inspect available VRAM", e))
    return false;
  r->major = property.major;
  r->minor = property.minor;
  if (c->client && strcmp(property.name, "NVIDIA GeForce RTX 5090"))
    return ria_fail(e, RIA_UNSUPPORTED,
                    "client probe requires the declared physical RTX5090");
  if (c->device_test_bytes > free_bytes || c->device_test_bytes > SIZE_MAX ||
      c->pinned_test_bytes > SIZE_MAX)
    return ria_fail(
        e, RIA_RESOURCE_LIMIT,
        "bounded probe allocation exceeds physical/addressable resources");
  ria_expert_cuda *expert = NULL;
  uint64_t start = ria_monotonic_ms();
  if (!ria_expert_cuda_create_pooled(0, 64, 64, 64, 1, 32, c->device_test_bytes,
                                     c->pinned_test_bytes, &expert, e))
    return false;
  uint64_t workspace = ria_expert_cuda_workspace_bytes(expert);
  uint64_t pool_bytes = ria_expert_cuda_pinned_bytes(expert);
  bool ok = true;
  void *device = NULL, *pinned = NULL;
  if (workspace >= c->device_test_bytes || pool_bytes >= c->pinned_test_bytes)
    ok = ria_fail(e, RIA_RESOURCE_LIMIT,
                  "probe budgets must include native-kernel workspace and "
                  "its owned pinned transfer pool");
  if (ok)
    ok =
        checked(cudaMalloc(&device, (size_t)(c->device_test_bytes - workspace)),
                "bounded VRAM allocation", e);
  if (ok)
    ok = checked(cudaHostAlloc(&pinned,
                               (size_t)(c->pinned_test_bytes - pool_bytes),
                               cudaHostAllocDefault),
                 "bounded CUDA page-lock allocation", e);
  size_t transfer = (size_t)(c->device_test_bytes - workspace <
                                     c->pinned_test_bytes - pool_bytes
                                 ? c->device_test_bytes - workspace
                                 : c->pinned_test_bytes - pool_bytes);
  if (ok) {
    memset(pinned, 0, (size_t)(c->pinned_test_bytes - pool_bytes));
    ok = checked(
             cudaMemset(device, 0, (size_t)(c->device_test_bytes - workspace)),
             "initialize bounded VRAM", e) &&
         checked(cudaMemcpy(device, pinned, transfer, cudaMemcpyHostToDevice),
                 "bounded H2D transfer", e) &&
         checked(cudaMemcpy(pinned, device, transfer, cudaMemcpyDeviceToHost),
                 "bounded D2H transfer", e);
  }
  r->allocation_ms = ria_monotonic_ms() - start;
  start = ria_monotonic_ms();
  if (ok)
    ok = ria_expert_cuda_native_probe(expert, r->native_results, e);
  if (ok && (r->native_results[0] != 64 || r->native_results[1] != 32 ||
             r->native_results[2] != 16))
    ok = ria_fail(e, RIA_EXECUTOR_ERROR,
                  "native block-scaled FP4/FP8/BF16 kernel result differs from "
                  "the independent unit-product oracle");
  r->kernel_ms = ria_monotonic_ms() - start;
  ria_error cleanup{};
  if (pinned &&
      !checked(cudaFreeHost(pinned), "release bounded CUDA pinned allocation",
               &cleanup) &&
      ok) {
    *e = cleanup;
    ok = false;
  }
  if (device &&
      !checked(cudaFree(device), "release bounded VRAM allocation", &cleanup) &&
      ok) {
    *e = cleanup;
    ok = false;
  }
  if (!ria_expert_cuda_destroy(expert, &cleanup) && ok) {
    *e = cleanup;
    ok = false;
  }
  if (ok) {
    r->device_bytes = free_bytes;
    r->pinned_bytes = c->pinned_test_bytes;
  }
  return ok;
}
