#include "ria/tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
  ria_error e = {0};
  uint8_t digest[32];
  ria_tensor_store s;
  bool validate_only = argc == 4 && !strcmp(argv[3], "--validate-only");
  if ((argc != 3 && !validate_only) || !ria_hex_decode(argv[2], strlen(argv[2]), digest, 32, &e))
    return 2;
  ria_tensor_load_options o = {.role = "server",
                               .expected_digest = digest,
                               .max_resident_bytes = 1 << 20,
                               .numa_node = -1};
  if (!ria_tensor_store_open(&s, argv[1], &o, &e)) {
    fprintf(stderr, "%d: %s\n", e.code, e.message);
    return 1;
  }
  if (validate_only) {
    ria_tensor_store_close(&s);
    return 0;
  }
  const ria_tensor *t = ria_tensor_name(&s, "layers.0.ffn.experts.0.w1.weight");
  ria_expert_matrix m;
  if (!t || !ria_tensor_matrix(&s, t, &m, &e) || m.in_features != 4 ||
      m.out_features != 2 || m.profile != RIA_EXPERT_BF16) {
    fprintf(stderr, "tensor fixture mismatch: %s\n", e.message);
    ria_tensor_store_close(&s);
    return 1;
  }
  const ria_shard *a = ria_shard_id(&s, t->shard);
  const uint8_t *bytes, *hash;
  uint32_t length;
  if (!a || !ria_tensor_chunk(&s, a->id, 0, &bytes, &length, &hash, &e) ||
      length != 64 || memcmp(hash, a->chunk_hashes, 32)) {
    ria_tensor_store_close(&s);
    return 1;
  }
  if (ria_tensor_chunk(&s, a->id, a->chunk_count, &bytes, &length, &hash, &e)) {
    ria_tensor_store_close(&s);
    return 1;
  }
  printf("{\"resident_bytes\":%llu,\"tensors\":%llu}\n",
         (unsigned long long)s.resident_bytes,
         (unsigned long long)s.tensor_count);
  ria_tensor_store_close(&s);
  return 0;
}
