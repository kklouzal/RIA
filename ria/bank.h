#ifndef RIA_BANK_H
#define RIA_BANK_H
#include "protocol.h"
#include "tensor.h"
#ifdef __cplusplus
extern "C" {
#endif
#define RIA_LAYERS 40u
#define RIA_EXPERTS 384u
#define RIA_WIDTH 5120u
#define RIA_INTERMEDIATE 2304u
typedef struct {
  ria_expert *experts;
  ria_expert shared[RIA_LAYERS];
  bool shared_present[RIA_LAYERS];
  ria_operation operations[RIA_LAYERS * 2];
  ria_table tables[2];
  const ria_tensor *engram[2];
  const ria_tensor_store *store;
} ria_bank;
/* Build immutable operation tables once. Handles 1..40 are routed FFNs;
 * 41..80 shared FFNs; tables 1,2 correspond to Engram layers 1,14. These
 * stable assignments are included in the bound operator contract. */
bool ria_bank_open(ria_bank *bank, const ria_tensor_store *store,
                   ria_error *error);
void ria_bank_close(ria_bank *bank);
const ria_operation *ria_bank_operation(const ria_bank *bank, uint64_t handle);
const ria_expert *ria_bank_expert(const ria_bank *bank, uint64_t handle,
                                  uint16_t expert);
const ria_table *ria_bank_table(const ria_bank *bank, uint64_t handle);
bool ria_bank_rows(const ria_bank *bank, const ria_row_request *request,
                   uint8_t *response, size_t length, ria_error *error);
#ifdef __cplusplus
}
#endif
#endif
