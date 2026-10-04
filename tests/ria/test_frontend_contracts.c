#define _POSIX_C_SOURCE 200809L
#include "ria/engine.h"
#include "ria/prompt.h"
#include "ria/tokenizer.h"
#include <assert.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <openssl/bio.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
static char *input(size_t *length) {
  size_t cap = 1024, n = 0;
  char *s = malloc(cap);
  if (!s)
    return NULL;
  for (;;) {
    if (n == cap) {
      if (cap == 67108864) {
        free(s);
        return NULL;
      }
      cap *= 2;
      char *p = realloc(s, cap);
      if (!p) {
        free(s);
        return NULL;
      }
      s = p;
    }
    size_t got = fread(s + n, 1, cap - n, stdin);
    n += got;
    if (!got)
      break;
  }
  if (ferror(stdin)) {
    free(s);
    return NULL;
  }
  *length = n;
  return s;
}
int main(int argc, char **argv) {
  ria_error e = {0};
  size_t n = 0;
  char *bytes = input(&n);
  if (!bytes) {
    fprintf(stderr, "input read failed\n");
    return 1;
  }
  bool ok = false;
  char *out = NULL;
  size_t length = 0;
  if (argc == 4 && !strcmp(argv[1], "tokenize")) {
    uint8_t digest[32];
    ria_tokenizer *t = NULL;
    uint32_t *tokens = NULL;
    size_t count = 0;
    ok = ria_hex_decode(argv[3], strlen(argv[3]), digest, 32, &e) &&
         ria_tokenizer_runtime_begin(&e) &&
         ria_tokenizer_open(argv[2], digest, UINT64_C(268435456), &t, &e) &&
         ria_tokenizer_encode(t, bytes, n, true, 1048576, &tokens, &count, &e);
    if (ok) {
      printf("[");
      for (size_t i = 0; i < count; i++)
        printf("%s%u", i ? "," : "", tokens[i]);
      printf("]\n");
    }
    free(tokens);
    ria_tokenizer_close(t);
    ria_tokenizer_runtime_end();
  } else if (argc == 2 && !strcmp(argv[1], "process-policy")) {
    struct sigaction disposition;
    int sockets[2] = {-1, -1};
    ok = signal(SIGPIPE, SIG_DFL) != SIG_ERR &&
         ria_engine_process_policy(&e) &&
         sigaction(SIGPIPE, NULL, &disposition) == 0 &&
         disposition.sa_handler == SIG_IGN &&
         socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0;
    BIO *socket = NULL;
    if (ok) {
      ok = close(sockets[1]) == 0;
      sockets[1] = -1;
      socket = BIO_new_socket(sockets[0], BIO_NOCLOSE);
      ok = ok && socket && BIO_write(socket, "x", 1) < 0;
    }
    BIO_free(socket);
    for (unsigned i = 0; i < 2; i++)
      if (sockets[i] >= 0 && close(sockets[i]))
        ok = false;
    if (ok)
      puts("protected");
  } else if (argc == 2 && !strcmp(argv[1], "nll")) {
    float *logits = malloc(RIA_GRAPH_VOCAB * sizeof *logits);
    ok = logits != NULL;
    const float offsets[] = {0, 1e20f, -1e20f, FLT_MAX, -FLT_MAX};
    if (ok)
      putchar('[');
    for (unsigned i = 0; ok && i < sizeof offsets / sizeof offsets[0]; i++) {
      for (uint32_t j = 0; j < RIA_GRAPH_VOCAB; j++)
        logits[j] = offsets[i];
      double loss;
      ok = ria_logits_nll(logits, RIA_GRAPH_VOCAB - 1, &loss, &e);
      if (ok)
        printf("%s%.17g", i ? "," : "", loss);
    }
    if (ok) {
      logits[0] = FLT_MAX;
      double loss;
      ok = ria_logits_nll(logits, 1, &loss, &e);
      if (ok)
        printf(",%.17g]\n", loss);
      double preserved = 3;
      logits[0] = NAN;
      ok = ok && !ria_logits_nll(logits, 1, &preserved, &e) && preserved == 3;
      logits[0] = INFINITY;
      ok = ok && !ria_logits_nll(logits, 1, &preserved, &e) && preserved == 3;
      logits[0] = -INFINITY;
      ok = ok && !ria_logits_nll(logits, 1, &preserved, &e) && preserved == 3;
      ok = ok && !ria_logits_nll(logits, RIA_GRAPH_VOCAB, &preserved, &e) &&
           !ria_logits_nll(NULL, 0, &preserved, &e) &&
           !ria_logits_nll(logits, 0, NULL, &e);
    }
    free(logits);
  } else if (argc == 2 && !strcmp(argv[1], "render")) {
    ria_json_doc doc = {0};
    ok = ria_json_parse(bytes, n, (ria_json_limits){67108864, 200000, 64}, &doc,
                        &e);
    uint64_t effort = 75;
    bool thinking = true, drop = true;
    if (ok) {
      const ria_json_node *v =
          ria_json_at(&doc, ria_json_get(&doc, 0, "thinking"));
      if (v) {
        ok = v->type == RIA_JSON_BOOL;
        thinking = v->boolean;
      }
      v = ria_json_at(&doc, ria_json_get(&doc, 0, "drop"));
      if (v) {
        ok = ok && v->type == RIA_JSON_BOOL;
        drop = v->boolean;
      }
      uint32_t k = ria_json_get(&doc, 0, "effort");
      if (k != RIA_JSON_NONE)
        ok = ok && ria_json_u64(&doc, k, false, &effort, &e);
    }
    if (ok)
      ok = ria_prompt_render(
          &doc, ria_json_get(&doc, 0, "messages"),
          (ria_prompt_options){thinking, drop, (unsigned)effort, 67108864},
          &out, &length, &e);
    ria_json_free(&doc);
  } else if (argc == 2 && !strcmp(argv[1], "utf8")) {
    ria_utf8_decoder decoder = {{0}, 0};
    out = malloc(3 * n + 1);
    ok = out != NULL;
    for (size_t i = 0; ok && i < n; i++) {
      size_t k = 0;
      ok = ria_utf8_decode(&decoder, (const uint8_t *)bytes + i, 1, false,
                           out + length, 3 * n - length, &k, &e);
      length += k;
    }
    size_t k = 0;
    if (ok)
      ok = ria_utf8_decode(&decoder, NULL, 0, true, out + length,
                           3 * n - length, &k, &e);
    length += k;
  } else if (argc == 2 && !strcmp(argv[1], "header")) {
    ria_http_header h;
    ok = ria_api_header(bytes, n, "fixture-token", 13, 4096, &h, &e);
    if (ok)
      printf("%s %s %" PRIu64 "\n", h.method, h.path, h.body_bytes);
  } else if (argc == 2 && !strcmp(argv[1], "measurements")) {
    ria_json_doc doc = {0};
    bool enabled = false;
    ok = ria_json_parse(bytes, n, (ria_json_limits){4096, 64, 4}, &doc, &e);
    const ria_json_node *stream =
        ria_json_at(&doc, ria_json_get(&doc, 0, "stream"));
    if (ok)
      ok = (!stream || stream->type == RIA_JSON_BOOL) &&
           ria_api_measurements(&doc, stream && stream->boolean, &enabled, &e);
    if (ok)
      printf("%s\n", enabled ? "true" : "false");
    ria_json_free(&doc);
  } else if (argc == 3 && !strcmp(argv[1], "completion"))
    ok = ria_prompt_completion(bytes, n, !strcmp(argv[2], "thinking"), &out,
                               &length, &e);
  else {
    fprintf(stderr, "usage: test_frontend_contracts tokenize PATH SHA | render "
                    "| completion chat|thinking | header | utf8\n");
  }
  if (ok && out)
    ok = fwrite(out, 1, length, stdout) == length;
  free(out);
  free(bytes);
  if (!ok) {
    fprintf(stderr, "error %d: %s\n", e.code, e.message);
    return 1;
  }
  return 0;
}
