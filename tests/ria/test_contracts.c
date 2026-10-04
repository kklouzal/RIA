#define _GNU_SOURCE
#include "ria/admission.h"
#include "ria/protocol.h"
#include "ria/transport.h"
#include <arpa/inet.h>
#include <math.h>
#include <openssl/err.h>
#include <openssl/pem.h>
#include <openssl/x509v3.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#define CHECK(x)                                                               \
  do {                                                                         \
    if (!(x)) {                                                                \
      fprintf(stderr, "contract failure at %s:%d: %s\n", __FILE__, __LINE__,   \
              #x);                                                             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
static ria_json_doc document(const char *s) {
  ria_json_doc doc = {0};
  ria_error error = {0};
  ria_json_limits bounds = {65536, 8192, 64};
  if (!ria_json_parse(s, strlen(s), bounds, &doc, &error)) {
    fprintf(stderr, "fixture JSON: %s\n", error.message);
    exit(1);
  }
  return doc;
}
static void canonical(const char *source, const char *expected) {
  ria_json_doc doc = document(source);
  ria_error e = {0};
  char *actual = NULL;
  size_t length = 0;
  CHECK(ria_json_canonical(&doc, false, &actual, &length, &e));
  CHECK(length == strlen(expected));
  CHECK(!memcmp(actual, expected, length));
  free(actual);
  ria_json_free(&doc);
}
static void test_json(void) {
  ria_error e = {0};
  uint64_t u;
  CHECK(ria_parse_u64("18446744073709551615", 20, &u, &e) && u == UINT64_MAX);
  CHECK(!ria_parse_u64("18446744073709551616", 20, &u, &e));
  CHECK(!ria_parse_u64("00", 2, &u, &e));
  CHECK(!ria_u64_add(UINT64_MAX, 1, &u));
  CHECK(!ria_u64_mul(UINT64_C(1) << 63, 2, &u));
  CHECK(ria_u64_mul(UINT64_C(1) << 40, 17, &u) &&
        u == UINT64_C(18691697672192));
  CHECK(ria_u64_add(UINT64_C(0x7ffffff0), 32, &u) && u == UINT64_C(0x80000010));
  CHECK(ria_u64_add(UINT64_C(0xfffffff0), 32, &u) &&
        u == UINT64_C(0x100000010));
  CHECK(ria_u64_mul(UINT64_C(5368709120), 384, &u) &&
        u == UINT64_C(2061584302080));
  canonical("{\"b\":2, \"a\":1}", "{\"a\":1,\"b\":2}");
  canonical("[333333333.33333329,1E30,4.50,2e-3,0.000000000000000000000000001]",
            "[333333333.3333333,1e+30,4.5,0.002,1e-27]");
  canonical("[-0,1e-6,1e-7,1e20,1e21,5e-324]",
            "[0,0.000001,1e-7,100000000000000000000,1e+21,5e-324]");
  canonical(
      "{\"\\ufb33\":1,\"\\ud83d\\ude00\":2,\"\\u20ac\":3,\"\\u000d\":4,"
      "\"\\u00f6\":5,\"1\":"
      "6,\"\\u0080\":7}",
      "{\"\\r\":4,\"1\":6,\"\xc2\x80\":7,\"\xc3\xb6\":5,\"\xe2\x82\xac\":3,"
      "\"\xf0\x9f\x98\x80\":2,\"\xef\xac\xb3\":1}");
  canonical("\"\\u0000\\u0008\\u0009\\u000a\\u000c\\u000d\\u001f\\\"\\\\\\/\"",
            "\"\\u0000\\b\\t\\n\\f\\r\\u001f\\\"\\\\/\"");
  const char *bad[] = {"{\"x\":1,\"x\":2}",
                       "{\"a\":0,\"\\u0061\":1}",
                       "\"\\ud800\"",
                       "\"\\udc00\"",
                       "\"\xc0\xaf\"",
                       "\"\xed\xa0\x80\"",
                       "\"\xf4\x90\x80\x80\"",
                       "[1,]",
                       "{\"x\":1,}",
                       "01",
                       "+1",
                       "1.",
                       "1e",
                       "NaN",
                       "1e999",
                       "true false"};
  ria_json_limits bounds = {65536, 8192, 32};
  for (size_t i = 0; i < sizeof bad / sizeof *bad; i++) {
    ria_json_doc doc = {0};
    CHECK(!ria_json_parse(bad[i], strlen(bad[i]), bounds, &doc, &e));
  }
  ria_json_doc doc = document(
      "{\"safe\":9007199254740991,\"large\":\"18446744073709551615\","
      "\"wrong\":9007199254740991.1,\"zero\":1e-999,\"exact\":12.0e2}");
  CHECK(ria_json_u64(&doc, ria_json_get(&doc, 0, "safe"), false, &u, &e) &&
        u == RIA_JSON_SAFE_INTEGER);
  CHECK(ria_json_u64(&doc, ria_json_get(&doc, 0, "large"), true, &u, &e) &&
        u == UINT64_MAX);
  CHECK(!ria_json_u64(&doc, ria_json_get(&doc, 0, "wrong"), false, &u, &e));
  CHECK(!ria_json_u64(&doc, ria_json_get(&doc, 0, "zero"), false, &u, &e));
  CHECK(ria_json_u64(&doc, ria_json_get(&doc, 0, "exact"), false, &u, &e) &&
        u == 1200);
  ria_json_free(&doc);
  doc = document("{\"digest\":\"ignored\",\"signatures\":[],\"child\":{"
                 "\"digest\":\"covered\"}}");
  char *text;
  size_t n;
  CHECK(ria_json_canonical(&doc, true, &text, &n, &e));
  CHECK(!strcmp(text, "{\"child\":{\"digest\":\"covered\"}}"));
  free(text);
  ria_json_free(&doc);
  /* RFC8785 Appendix B numbers, expressed by independently specified IEEE bits.
   */
  const struct {
    uint64_t bits;
    const char *text;
  } samples[] = {{UINT64_C(0x0000000000000000), "0"},
                 {UINT64_C(0x8000000000000000), "0"},
                 {UINT64_C(0x0000000000000001), "5e-324"},
                 {UINT64_C(0x8000000000000001), "-5e-324"},
                 {UINT64_C(0x7fefffffffffffff), "1.7976931348623157e+308"},
                 {UINT64_C(0xffefffffffffffff), "-1.7976931348623157e+308"},
                 {UINT64_C(0x4340000000000000), "9007199254740992"},
                 {UINT64_C(0xc340000000000000), "-9007199254740992"},
                 {UINT64_C(0x4430000000000000), "295147905179352830000"},
                 {UINT64_C(0x44b52d02c7e14af5), "9.999999999999997e+22"},
                 {UINT64_C(0x44b52d02c7e14af6), "1e+23"},
                 {UINT64_C(0x44b52d02c7e14af7), "1.0000000000000001e+23"},
                 {UINT64_C(0x444b1ae4d6e2ef4e), "999999999999999700000"},
                 {UINT64_C(0x444b1ae4d6e2ef4f), "999999999999999900000"},
                 {UINT64_C(0x444b1ae4d6e2ef50), "1e+21"},
                 {UINT64_C(0x3eb0c6f7a0b5ed8c), "9.999999999999997e-7"},
                 {UINT64_C(0x3eb0c6f7a0b5ed8d), "0.000001"},
                 {UINT64_C(0x41b3de4355555553), "333333333.3333332"},
                 {UINT64_C(0x41b3de4355555554), "333333333.33333325"},
                 {UINT64_C(0x41b3de4355555555), "333333333.3333333"},
                 {UINT64_C(0x41b3de4355555556), "333333333.3333334"},
                 {UINT64_C(0x41b3de4355555557), "333333333.33333343"},
                 {UINT64_C(0xbecbf647612f3696), "-0.0000033333333333333333"},
                 {UINT64_C(0x43143ff3c1cb0959), "1424953923781206.2"}};
  for (size_t i = 0; i < sizeof samples / sizeof *samples; i++) {
    ria_json_node node = {0};
    node.type = RIA_JSON_NUMBER;
    memcpy(&node.number, &samples[i].bits, 8);
    ria_json_doc d = {.nodes = &node, .count = 1};
    CHECK(ria_json_canonical(&d, false, &text, &n, &e));
    if (strcmp(text, samples[i].text)) {
      fprintf(stderr, "JCS %016llx: %s expected %s\n",
              (unsigned long long)samples[i].bits, text, samples[i].text);
      exit(1);
    }
    free(text);
  }
  /* Deterministic random malformed strings exercise all parser boundary paths.
   */
  uint64_t seed = UINT64_C(0x827739df123abc);
  for (unsigned i = 0; i < 10000; i++) {
    unsigned char bytes[256];
    size_t length = i % sizeof bytes;
    for (size_t j = 0; j < length; j++) {
      seed ^= seed << 13;
      seed ^= seed >> 7;
      seed ^= seed << 17;
      bytes[j] = (unsigned char)seed;
    }
    ria_json_doc d = {0};
    if (ria_json_parse(bytes, length, bounds, &d, &e)) {
      CHECK(ria_json_canonical(&d, false, &text, &n, &e));
      ria_json_doc repeated = {0};
      CHECK(ria_json_parse(text, n, bounds, &repeated, &e));
      free(text);
      ria_json_free(&repeated);
    }
    ria_json_free(&d);
  }
}
static void test_json_files(void) {
  char directory[] = "/tmp/ria-json-contracts-XXXXXX";
  CHECK(mkdtemp(directory));
  char path[256], fifo[256];
  CHECK(snprintf(path, sizeof path, "%s/input.json", directory) > 0);
  CHECK(snprintf(fifo, sizeof fifo, "%s/fifo", directory) > 0);
  FILE *file = fopen(path, "wb");
  CHECK(file && fwrite("{}", 1, 2, file) == 2 && fclose(file) == 0);
  ria_error e = {0};
  ria_json_doc d = {0};
  CHECK(ria_json_read(path, (ria_json_limits){1048576, 8192, 8}, &d, &e));
  CHECK(d.allocated_bytes == 2 * sizeof(ria_json_node) + 3);
  ria_json_free(&d);
  CHECK(!ria_json_read(path, (ria_json_limits){1, 8192, 8}, &d, &e));
  CHECK(e.code == RIA_RESOURCE_LIMIT && !d.nodes && !d.strings);
  CHECK(mkfifo(fifo, 0600) == 0);
  memset(&d, 0xa5, sizeof d); /* An output never observes its old bytes. */
  alarm(2);
  CHECK(!ria_json_read(fifo, (ria_json_limits){1048576, 8192, 8}, &d, &e));
  alarm(0);
  CHECK(e.code == RIA_RESOURCE_LIMIT && !d.nodes && !d.strings);
  CHECK(unlink(fifo) == 0 && unlink(path) == 0 && rmdir(directory) == 0);
}
static ria_limits limits(void) {
  ria_limits l = {RIA_FRAME_MAX, RIA_BULK_MAX, 64,   2,   256,
                  67108864,      1000,         1000, 1000};
  return l;
}
static size_t expert_fixture(uint8_t p[256], bool shared) {
  memset(p, 0, 256);
  ria_write_u64(p, 7);
  ria_write_u64(p + 8, 9);
  ria_write_u32(p + 16, 2);
  ria_write_u32(p + 20, 3);
  ria_write_u32(p + 24, 2);
  ria_write_u32(p + 28, shared ? 2 : 3);
  ria_write_u64(p + 40, UINT64_C(1) << 40);
  ria_write_u64(p + 48, (UINT64_C(1) << 40) + 1);
  ria_write_u32(p + 56, 0);
  ria_write_u32(p + 60, shared ? 1 : 2);
  ria_write_u32(p + 64, shared ? 2 : 3);
  uint32_t entries = shared ? 2 : 3;
  for (uint32_t i = 0; i < entries; i++) {
    ria_write_u16(p + 68 + i * 8, shared ? 0 : (uint16_t)(i + 1));
    ria_write_u16(p + 70 + i * 8, shared ? 65535 : (uint16_t)(i % 2));
    ria_write_f32(p + 72 + i * 8, shared ? 1.0f : 0.5f);
  }
  for (unsigned i = 0; i < 6; i++)
    ria_write_f32(p + 68 + entries * 8 + i * 4, (float)i);
  return 68 + entries * 8 + 24;
}
static void test_wire(void) {
  ria_error e = {0};
  ria_limits l = limits();
  CHECK(ria_limits_validate(&l, &e));
  ria_header h = {RIA_EXPERT, 0, 0, 123, UINT64_MAX, UINT64_C(1) << 40, {0}};
  h.session[0] = 0xa5;
  uint8_t wire[64];
  CHECK(ria_header_encode(&h, wire, &e));
  CHECK(!memcmp(wire, "DSER\1\0\12\0", 8));
  CHECK(wire[32] == 0xa5 && ria_read_u64(wire + 24) == UINT64_MAX &&
        ria_read_u64(wire + 48) == h.epoch);
  ria_header d;
  CHECK(ria_header_decode(wire, RIA_FRAME_MAX, &d, &e) &&
        d.request_id == h.request_id);
  wire[56] = 1;
  CHECK(!ria_header_decode(wire, RIA_FRAME_MAX, &d, &e));
  wire[56] = 0;
  wire[4] = 0;
  wire[5] = 1;
  CHECK(!ria_header_decode(wire, RIA_FRAME_MAX, &d, &e));
  wire[4] = 1;
  wire[5] = 0;
  wire[8] = 2;
  CHECK(!ria_header_decode(wire, RIA_FRAME_MAX, &d, &e));
  wire[8] = 0;
  ria_write_u16(wire + 6, 99);
  CHECK(!ria_header_decode(wire, RIA_FRAME_MAX, &d, &e));
  uint64_t req, rep;
  CHECK(ria_expert_lengths(64, 384, 5120, 5120, 0, &req, &rep, &e) &&
        req == 1314604 && rep == 7864352);
  CHECK(!ria_expert_lengths(UINT64_MAX, 1, 2, 2, 0, &req, &rep, &e));
  ria_operation op = {7, 3, 2, 64, 0, 384, 6, 0, 2, false, true, true};
  uint8_t p[256];
  size_t n = expert_fixture(p, false);
  ria_expert_request request;
  CHECK(ria_expert_parse(p, n, RIA_EXPERT, &op, &l, &request, &e));
  CHECK(request.row_count == 2 && request.entry_count == 3 &&
        request.response_bytes == 56);
  CHECK(!ria_expert_parse(p, n - 1, RIA_EXPERT, &op, &l, &request, &e));
  CHECK(!ria_expert_parse(p, n + 1, RIA_EXPERT, &op, &l, &request, &e));
  ria_write_u16(p + 76, 1);
  CHECK(!ria_expert_parse(p, n, RIA_EXPERT, &op, &l, &request, &e));
  n = expert_fixture(p, false);
  ria_write_u16(p + 78, 0);
  CHECK(!ria_expert_parse(p, n, RIA_EXPERT, &op, &l, &request, &e));
  n = expert_fixture(p, false);
  ria_write_f32(p + 72, NAN);
  CHECK(!ria_expert_parse(p, n, RIA_EXPERT, &op, &l, &request, &e));
  n = expert_fixture(p, false);
  ria_write_f32(p + 92, 0.1f);
  CHECK(!ria_expert_parse(p, n, RIA_EXPERT, &op, &l, &request, &e));
  n = expert_fixture(p, false);
  ria_write_u32(p + 60, 4);
  CHECK(!ria_expert_parse(p, n, RIA_EXPERT, &op, &l, &request, &e));
  n = expert_fixture(p, false);
  ria_write_u64(p + 48, ria_read_u64(p + 40));
  CHECK(!ria_expert_parse(p, n, RIA_EXPERT, &op, &l, &request, &e));
  n = expert_fixture(p, false);
  ria_write_u32(p + 32, 4);
  CHECK(!ria_expert_parse(p, n, RIA_EXPERT, &op, &l, &request, &e));
  n = expert_fixture(p, false);
  CHECK(ria_expert_parse(p, n, RIA_EXPERT, &op, &l, &request, &e));
  uint8_t response[56] = {0};
  ria_write_u64(response, 7);
  ria_write_u64(response + 8, 9);
  ria_write_u32(response + 16, 2);
  ria_write_u32(response + 20, 2);
  ria_write_u32(response + 24, 3);
  for (unsigned i = 0; i < 6; i++)
    ria_write_f32(response + 32 + i * 4, (float)i);
  CHECK(ria_expert_response(response, sizeof response, &request, &op, &e));
  ria_write_u64(response + 8, 10);
  CHECK(!ria_expert_response(response, sizeof response, &request, &op, &e));
  op.shared = true;
  n = expert_fixture(p, true);
  CHECK(ria_expert_parse(p, n, RIA_SHARED, &op, &l, &request, &e));
  ria_write_u16(p + 70, 0);
  CHECK(!ria_expert_parse(p, n, RIA_SHARED, &op, &l, &request, &e));
  ria_table table = {99, UINT64_C(1) << 40, 8, 100};
  uint8_t rows[48] = {0};
  ria_write_u64(rows, 99);
  ria_write_u32(rows + 8, 2);
  ria_write_u64(rows + 16, (UINT64_C(1) << 40) - 1);
  ria_write_u64(rows + 24, UINT64_MAX);
  ria_write_u64(rows + 32, 7);
  ria_write_u64(rows + 40, 42);
  ria_row_request rr;
  CHECK(ria_rows_parse(rows, sizeof rows, &table, &l, &rr, &e));
  uint8_t rrep[72] = {0};
  ria_write_u64(rrep, 99);
  ria_write_u32(rrep + 8, 2);
  ria_write_u32(rrep + 12, 8);
  ria_write_u64(rrep + 16, 100);
  memcpy(rrep + 24, rows + 16, 32);
  CHECK(ria_rows_response(rrep, sizeof rrep, &rr, &table, &e));
  rrep[32] ^= 1;
  CHECK(!ria_rows_response(rrep, sizeof rrep, &rr, &table, &e));
  ria_write_u64(rows + 16, UINT64_C(1) << 40);
  CHECK(!ria_rows_parse(rows, sizeof rows, &table, &l, &rr, &e));
  uint8_t chunk[67] = {0}, hash[32];
  ria_write_u64(chunk, 3);
  ria_write_u64(chunk + 8, 2);
  ria_write_u64(chunk + 16, 8);
  ria_write_u32(chunk + 24, 3);
  memcpy(chunk + 64, "abc", 3);
  CHECK(ria_sha256("abc", 3, hash, &e));
  char hash_hex[65];
  ria_hex_encode(hash, 32, hash_hex);
  CHECK(!strcmp(
      hash_hex,
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"));
  memcpy(chunk + 32, hash, 32);
  const uint8_t *data;
  size_t data_length;
  CHECK(ria_chunk_response(chunk, sizeof chunk, 3, 2, 11, 4, hash, &data,
                           &data_length, &e) &&
        data_length == 3);
  chunk[66] ^= 1;
  CHECK(!ria_chunk_response(chunk, sizeof chunk, 3, 2, 11, 4, hash, &data,
                            &data_length, &e));
  CHECK(!ria_chunk_response(chunk, sizeof chunk, 3, UINT64_MAX, 11, 4, hash,
                            &data, &data_length, &e));
  uint8_t cancel[24] = {0};
  ria_write_u64(cancel, 5);
  ria_write_u64(cancel + 8, 3);
  uint64_t target;
  CHECK(ria_cancel_parse(cancel, 16, 6, 3, &target, &e) && target == 5);
  CHECK(!ria_cancel_parse(cancel, 16, 5, 3, &target, &e));
  uint32_t state;
  CHECK(ria_cancel_response(cancel, 24, 5, 3, &state, &e) && state == 0);
  ria_write_u32(cancel + 16, 3);
  CHECK(!ria_cancel_response(cancel, 24, 5, 3, &state, &e));
}
static void test_control(void) {
  ria_error e = {0};
  const char *digest =
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
  char input[4096];
  snprintf(input, sizeof input,
           "{\"role\":\"client\",\"logical_model_digest\":\"%s\",\"operator_"
           "contract_digest\":\"%"
           "s\",\"encoding_digest\":\"%s\",\"client_layout_digest\":\"%s\","
           "\"placement_plan_"
           "digest\":\"%s\",\"profile\":\"bf16\",\"server_executor\":\"cpu\","
           "\"limits\":{\"frame_"
           "payload_bytes\":16777216,\"bulk_data_bytes\":4194304,\"expert_"
           "rows\":64,\"expert_"
           "requests\":2,\"row_lookup_rows\":256,\"inflight_payload_bytes\":"
           "67108864,\"operation_"
           "timeout_ms\":1000,\"frame_io_timeout_ms\":1000,\"write_timeout_"
           "ms\":1000}}",
           digest, digest, digest, digest, digest);
  ria_json_doc d = document(input), parsed = {0};
  ria_limits l;
  CHECK(ria_bind_validate(&d, false, &l, &e));
  char *json = NULL;
  size_t n = 0;
  CHECK(ria_json_canonical(&d, false, &json, &n, &e));
  ria_header h = {RIA_BIND, 0, 0, n, 1, 0, {0}};
  CHECK(ria_control_json((uint8_t *)json, n, &h, &parsed, &e));
  ria_json_free(&parsed);
  free(json);
  h.payload_length = strlen(input);
  CHECK(!ria_control_json((uint8_t *)input, strlen(input), &h, &parsed, &e));
  uint8_t session[16] = {1}, capability[32] = {2}, layout[32] = {3};
  CHECK(ria_bind_response_json(&d, &l, layout, session, 3, capability, &json,
                               &n, &e));
  h.flags = 1;
  h.epoch = 3;
  h.payload_length = n;
  memcpy(h.session, session, 16);
  CHECK(ria_control_json((uint8_t *)json, n, &h, &parsed, &e));
  ria_json_free(&parsed);
  h.epoch = 4;
  CHECK(!ria_control_json((uint8_t *)json, n, &h, &parsed, &e));
  h.epoch = 3;
  free(json);
  ria_limits larger = l;
  larger.expert_requests = 3;
  CHECK(!ria_bind_response_json(&d, &larger, layout, session, 3, capability,
                                &json, &n, &e));
  ria_json_free(&d);
  char sid[33], cap[65];
  ria_hex_encode(session, 16, sid);
  ria_hex_encode(capability, 32, cap);
  snprintf(input, sizeof input,
           "{\"session_id\":\"%s\",\"epoch\":\"3\",\"logical_model_digest\":\"%"
           "s\",\"operator_"
           "contract_digest\":\"%s\",\"bulk_capability\":\"%s\"}",
           sid, digest, digest, cap);
  d = document(input);
  CHECK(ria_json_canonical(&d, false, &json, &n, &e));
  h.kind = RIA_BIND_BULK;
  h.flags = 0;
  h.payload_length = n;
  CHECK(ria_control_json((uint8_t *)json, n, &h, &parsed, &e));
  ria_json_free(&parsed);
  ria_json_free(&d);
  free(json);
  const char *bound = "{\"bound\":true}";
  h.flags = 1;
  h.payload_length = strlen(bound);
  CHECK(ria_control_json((uint8_t *)bound, strlen(bound), &h, &parsed, &e));
  ria_json_free(&parsed);
  h.kind = RIA_HEALTH;
  h.flags = 0;
  h.payload_length = 2;
  CHECK(ria_control_json((uint8_t *)"{}", 2, &h, &parsed, &e));
  ria_json_free(&parsed);
  const char *health =
      "{\"state\":\"ready\",\"ready\":true,\"counters\":{\"jobs\":2}}";
  h.flags = 1;
  h.payload_length = strlen(health);
  CHECK(ria_control_json((uint8_t *)health, strlen(health), &h, &parsed, &e));
  ria_json_free(&parsed);
  const char *error = "{\"code\":5,\"message\":\"credit unavailable\"}";
  h.status = 5;
  h.payload_length = strlen(error);
  CHECK(ria_control_json((uint8_t *)error, strlen(error), &h, &parsed, &e));
  ria_json_free(&parsed);
  h.status = 6;
  CHECK(!ria_control_json((uint8_t *)error, strlen(error), &h, &parsed, &e));
  h.kind = RIA_CLOSE;
  h.status = 0;
  h.payload_length = 0;
  uint8_t header[64];
  CHECK(ria_header_encode(&h, header, &e));
  ria_header decoded;
  CHECK(ria_header_decode(header, RIA_FRAME_MAX, &decoded, &e));
  h.payload_length = 1;
  CHECK(!ria_header_encode(&h, header, &e));
}
static void test_lifecycle(void) {
  ria_error e = {0};
  ria_limits l = limits();
  l.inflight_payload_bytes = 1000;
  ria_binding b;
  ria_binding_init(&b, &l);
  CHECK(ria_binding_protect(&b, 100, 100, 100, &e));
  uint8_t session[16] = {1}, cap[32] = {2};
  CHECK(ria_binding_install(&b, session, 3, cap, &e));
  CHECK(ria_binding_bulk(&b, cap, true, &e));
  CHECK(!ria_binding_bulk(&b, cap, true, &e));
  ria_header h = {RIA_EXPERT, 0, 0, 50, 2, 3, {0}};
  memcpy(h.session, session, 16);
  CHECK(ria_binding_receive(&b, &h, false, &e));
  CHECK(ria_binding_admit(&b, &h, 350, 100, &e));
  h.request_id = 3;
  CHECK(ria_binding_receive(&b, &h, false, &e));
  CHECK(ria_binding_admit(&b, &h, 350, 100, &e));
  h.request_id = 4;
  CHECK(!ria_binding_admit(&b, &h, 1, 100, &e));
  CHECK(b.reserved_bytes == 700);
  uint32_t state;
  CHECK(ria_binding_cancel(&b, 2, &state, &e) && state == 0 &&
        b.reserved_bytes == 700);
  CHECK(!ria_binding_terminal(&b, 2, false, &e));
  ria_header response = h;
  response.kind = RIA_EXPERT;
  response.flags = 1;
  response.request_id = 3;
  CHECK(ria_binding_response(&b, &response, &e));
  CHECK(ria_binding_terminal(&b, 3, true, &e));
  CHECK(!ria_binding_response(&b, &response, &e));
  CHECK(ria_binding_terminal(&b, 2, true, &e));
  CHECK(ria_binding_cancel(&b, 2, &state, &e) && state == 1);
  CHECK(ria_binding_cancel(&b, 999, &state, &e) && state == 2);
  /* Identical numeric IDs are legal on independent channels. */
  h.kind = RIA_EXPERT;
  h.request_id = 4;
  CHECK(ria_binding_admit(&b, &h, 100, 100, &e));
  ria_header bulk = h;
  bulk.kind = RIA_CHUNK;
  bulk.request_id = 4;
  bulk.payload_length = 16;
  CHECK(ria_binding_admit(&b, &bulk, 100, 100, &e));
  CHECK(ria_binding_terminal_channel(&b, 4, true, true, &e));
  CHECK(b.reserved_bytes == 100);
  CHECK(ria_binding_terminal(&b, 4, true, &e));
  for (uint64_t id = 5; id < 24; id++) {
    h.request_id = id;
    CHECK(ria_binding_admit(&b, &h, 100, 100, &e));
    CHECK(ria_binding_terminal(&b, id, true, &e));
  }
  CHECK(b.terminal_count == 16);
  CHECK(ria_binding_cancel(&b, 2, &state, &e) && state == 2);
  CHECK(ria_binding_cancel(&b, 23, &state, &e) && state == 1);
  h.request_id = 24;
  CHECK(ria_binding_admit(&b, &h, 100, 100, &e));
  uint64_t id;
  CHECK(!ria_binding_expired(&b, 1099, &id));
  CHECK(ria_binding_expired(&b, 1100, &id) && id == 24);
  ria_binding_invalidate(&b);
  CHECK(!b.valid && b.reserved_bytes == 0);
  CHECK(!ria_binding_admit(&b, &h, 100, 100, &e));
}
static ria_json_doc signed_document(const char *source) {
  ria_json_doc doc = document(source);
  ria_error e = {0};
  uint8_t hash[32];
  CHECK(ria_json_sha256(&doc, true, hash, &e));
  ria_json_free(&doc);
  char hex[65];
  ria_hex_encode(hash, 32, hex);
  size_t n = strlen(source);
  char *text = malloc(n + 80);
  CHECK(text);
  memcpy(text, source, n - 1);
  snprintf(text + n - 1, 81, ",\"digest\":\"%s\"}", hex);
  doc = document(text);
  free(text);
  return doc;
}
static void test_admission(void) {
  const char *logical =
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
  char request[1024], inventory[4096], calibration[1024], probe[1024];
  snprintf(request, sizeof request,
           "{\"schema_revision\":1,\"role\":\"expert\",\"executor\":\"cpu\","
           "\"profile\":\"bf16\","
           "\"logical_model_digest\":\"%s\",\"operator_contract_digest\":\"%"
           "s\",\"context_"
           "positions\":8,\"caps\":{\"host_bytes\":1000,\"device_bytes\":0,"
           "\"pinned_bytes\":0,"
           "\"numa\":[{\"node\":0,\"bytes\":1000}]}}",
           logical, logical);
  snprintf(inventory, sizeof inventory,
           "{\"schema_revision\":1,\"logical_model_digest\":\"%s\",\"operator_"
           "contract_digest\":"
           "\"%s\",\"semantic_max_positions\":4096,\"allocations\":[{\"id\":"
           "\"18446744073709551615\",\"name\":\"canonical model and "
           "state\",\"resource\":\"host\",\"base_bytes\":100,\"bytes_per_"
           "position\":10,\"numa_"
           "node\":0,\"pinned\":false,\"protected_progress\":false,\"phases\":["
           "\"startup\","
           "\"prefill\",\"decode\",\"continuation\",\"image\",\"drain\"]},{"
           "\"id\":\"2\",\"name\":"
           "\"progress\",\"resource\":\"host\",\"base_bytes\":64,\"bytes_per_"
           "position\":0,\"numa_"
           "node\":0,\"pinned\":false,\"protected_progress\":true,\"phases\":["
           "\"startup\","
           "\"prefill\",\"decode\",\"continuation\",\"image\",\"drain\"]}]}",
           logical, logical);
  snprintf(calibration, sizeof calibration,
           "{\"schema_revision\":1,\"profile\":\"bf16\",\"operator_contract_"
           "digest\":\"%s\","
           "\"executor\":\"cpu\",\"qualified\":true,\"environment_digest\":\"%"
           "s\",\"build_digest\":\"%s\",\"evidence_digest\":\"%s\",\"policy_"
           "digest\":\"%s\"}",
           logical, logical, logical, logical, logical);
  snprintf(
      probe, sizeof probe,
      "{\"schema_revision\":1,\"role\":\"expert\",\"executor\":\"cpu\",\"host_"
      "bytes\":1000,\"device_bytes\":0,\"pinned_bytes\":0,\"numa\":[{\"node\":"
      "0,\"bytes\":1000}],\"qualified\":true,\"environment_digest\":\"%s\","
      "\"build_digest\":\"%s\",\"evidence_digest\":\"%s\"}",
      logical, logical, logical);
  ria_json_doc r = document(request), i = document(inventory),
               p = signed_document(probe), c = signed_document(calibration);
  ria_error e = {0};
  ria_admission_plan plan;
  CHECK(ria_admission_compute(&r, &i, &p, &c, &plan, &e));
  CHECK(plan.peak.host == 244 && plan.peak.numa[0] == 244 &&
        plan.allocation_count == 2);
  char *json;
  size_t n;
  CHECK(ria_admission_json(&plan, &json, &n, &e));
  ria_json_doc out = document(json);
  uint8_t expected[32], actual[32];
  CHECK(ria_json_digest_field(&out, ria_json_get(&out, 0, "digest"), expected,
                              &e));
  CHECK(ria_json_sha256(&out, true, actual, &e));
  CHECK(!memcmp(expected, actual, 32));
  CHECK(ria_json_digest_field(&out, ria_json_get(&out, 0, "environment_digest"),
                              expected, &e));
  CHECK(!memcmp(expected, plan.environment_digest, 32));
  CHECK(ria_json_digest_field(&out, ria_json_get(&out, 0, "build_digest"),
                              expected, &e));
  CHECK(!memcmp(expected, plan.build_digest, 32));
  CHECK(ria_json_digest_field(&out, ria_json_get(&out, 0, "policy_digest"),
                              expected, &e));
  CHECK(!memcmp(expected, plan.policy_digest, 32));
  ria_json_free(&out);
  free(json);
  uint32_t environment = ria_json_get(&c, 0, "environment_digest");
  const char *old_environment = c.nodes[environment].text;
  c.nodes[environment].text =
      "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
  CHECK(!ria_admission_compute(&r, &i, &p, &c, &plan, &e) &&
        e.code == RIA_IDENTITY_MISMATCH);
  c.nodes[environment].text = old_environment;
  ria_json_node *node = &r.nodes[ria_json_get(&r, 0, "context_positions")];
  node->number = 1000;
  node->text = "1000";
  node->length = 4;
  CHECK(!ria_admission_compute(&r, &i, &p, &c, &plan, &e));
  node->number = 8;
  node->text = "8";
  node->length = 1;
  const ria_json_node *alloc = &i.nodes[ria_json_get(&i, 0, "allocations")];
  uint32_t progress = i.nodes[alloc->child].next;
  i.nodes[ria_json_get(&i, progress, "protected_progress")].boolean = false;
  CHECK(!ria_admission_compute(&r, &i, &p, &c, &plan, &e));
  i.nodes[ria_json_get(&i, progress, "protected_progress")].boolean = true;
  p.nodes[ria_json_get(&p, 0, "qualified")].boolean = false;
  CHECK(!ria_admission_compute(&r, &i, &p, &c, &plan, &e));
  p.nodes[ria_json_get(&p, 0, "qualified")].boolean = true;
  c.nodes[ria_json_get(&c, 0, "digest")].text = logical;
  CHECK(!ria_admission_compute(&r, &i, &p, &c, &plan, &e));
  ria_json_free(&r);
  ria_json_free(&i);
  ria_json_free(&p);
  ria_json_free(&c);
}
static void extension(X509 *cert, X509 *issuer, int nid, const char *value) {
  X509V3_CTX ctx;
  X509V3_set_ctx(&ctx, issuer, cert, NULL, NULL, 0);
  X509_EXTENSION *ext = X509V3_EXT_conf_nid(NULL, &ctx, nid, (char *)value);
  CHECK(ext && X509_add_ext(cert, ext, -1) == 1);
  X509_EXTENSION_free(ext);
}
static X509 *certificate(EVP_PKEY *key, X509 *issuer, EVP_PKEY *issuer_key,
                         const char *name, long serial, bool ca) {
  X509 *cert = X509_new();
  CHECK(cert && X509_set_version(cert, 2) == 1 &&
        ASN1_INTEGER_set(X509_get_serialNumber(cert), serial) == 1 &&
        X509_gmtime_adj(X509_getm_notBefore(cert), -60) &&
        X509_gmtime_adj(X509_getm_notAfter(cert), 3600) &&
        X509_set_pubkey(cert, key) == 1);
  X509_NAME *subject = X509_get_subject_name(cert);
  CHECK(X509_NAME_add_entry_by_txt(subject, "CN", MBSTRING_ASC,
                                   (const unsigned char *)name, -1, -1,
                                   0) == 1 &&
        X509_set_issuer_name(cert, issuer ? X509_get_subject_name(issuer)
                                          : subject) == 1);
  extension(cert, issuer ? issuer : cert, NID_basic_constraints,
            ca ? "critical,CA:TRUE" : "critical,CA:FALSE");
  extension(cert, issuer ? issuer : cert, NID_key_usage,
            ca ? "critical,keyCertSign,cRLSign" : "critical,digitalSignature");
  if (!ca) {
    char san[128];
    snprintf(san, sizeof san, "DNS:%s", name);
    extension(cert, issuer, NID_subject_alt_name, san);
    extension(cert, issuer, NID_ext_key_usage, "serverAuth,clientAuth");
  }
  CHECK(X509_sign(cert, issuer_key ? issuer_key : key, EVP_sha256()) > 0);
  return cert;
}
static void pem(const char *path, EVP_PKEY *key, X509 *cert) {
  FILE *f = fopen(path, "wb");
  CHECK(f);
  CHECK(fchmod(fileno(f), 0600) == 0);
  CHECK(key ? PEM_write_PrivateKey(f, key, NULL, NULL, 0, NULL, NULL) == 1
            : PEM_write_X509(f, cert) == 1);
  CHECK(fclose(f) == 0);
}
static void tls_exchange(ria_tls *server, ria_tls *client, unsigned mode) {
  int listener = socket(AF_INET, SOCK_STREAM, 0);
  CHECK(listener >= 0);
  struct sockaddr_in address = {0};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  CHECK(bind(listener, (struct sockaddr *)&address, sizeof address) == 0 &&
        listen(listener, 1) == 0);
  socklen_t length = sizeof address;
  CHECK(getsockname(listener, (struct sockaddr *)&address, &length) == 0);
  pid_t pid = fork();
  CHECK(pid >= 0);
  if (pid == 0) {
    int fd = accept(listener, NULL, NULL);
    close(listener);
    if (fd < 0)
      _exit(1);
    ria_transport t = {0};
    ria_error e = {0};
    bool ok = ria_transport_accept(&t, server, fd, &e);
    if (ok && mode == 1) {
      for (unsigned i = 0; i < 2; i++) {
        ria_header h;
        uint8_t *p = NULL;
        ok = ria_transport_frame(&t, ria_monotonic_ms() + 2000, 1000,
                                 RIA_FRAME_MAX, &h, &p, &e);
        if (!ok)
          break;
        ok = h.request_id == 10 + i && h.payload_length == 3 &&
             !memcmp(p, "abc", 3);
        free(p);
        if (!ok)
          break;
        uint8_t response[64000];
        memset(response, 0xa5, sizeof response);
        h.flags = 1;
        h.payload_length = sizeof response;
        ok =
            ria_transport_send(&t, &h, response, ria_monotonic_ms() + 2000, &e);
        if (!ok)
          break;
      }
    }
    if (ok && mode == 2) {
      ria_header h;
      uint8_t *payload = NULL;
      bool read = ria_transport_frame(&t, ria_monotonic_ms() + 2000, 80,
                                      RIA_FRAME_MAX, &h, &payload, &e);
      free(payload);
      ok = !read && e.code == RIA_DEADLINE_EXCEEDED && t.unusable;
      uint8_t byte = 0;
      if (ok)
        ok = !ria_transport_write(&t, &byte, 1, ria_monotonic_ms() + 1000,
                                  &e) &&
             e.code == RIA_NOT_READY;
    }
    if (t.ssl)
      ria_transport_close(&t);
    else
      close(fd);
    _exit(mode ? (ok ? 0 : 1) : 0);
  }
  close(listener);
  ria_transport t = {0};
  ria_error e = {0};
  bool ok = ria_transport_connect(&t, client, (struct sockaddr *)&address,
                                  sizeof address, 2000, &e);
  if (mode) {
    CHECK(ok);
    uint8_t certificate_hash[32];
    CHECK(ria_transport_peer_digest(&t, certificate_hash, &e));
    if (mode == 1) {
      ria_header h = {RIA_EXPERT, 0, 0, 3, 10, 1, {1}};
      uint8_t frame[67];
      CHECK(ria_header_encode(&h, frame, &e));
      memcpy(frame + 64, "abc", 3);
      ERR_raise(ERR_LIB_USER,
                7); /* Deliberately unrelated previous-library error. */
      CHECK(ria_transport_write(&t, frame, 1, ria_monotonic_ms() + 2000, &e));
      CHECK(ria_transport_write(&t, frame + 1, 66, ria_monotonic_ms() + 2000,
                                &e));
      h.request_id = 11;
      CHECK(ria_header_encode(&h, frame, &e));
      CHECK(ria_transport_write(&t, frame, sizeof frame,
                                ria_monotonic_ms() + 2000, &e));
    }
    if (mode == 2) {
      uint8_t first = 'D';
      CHECK(ria_transport_write(&t, &first, 1, ria_monotonic_ms() + 2000, &e));
      uint8_t received;
      CHECK(
          !ria_transport_read(&t, &received, 1, ria_monotonic_ms() + 2000, &e));
    } else
      for (unsigned i = 0; i < 2; i++) {
        ria_header response;
        uint8_t *payload = NULL;
        CHECK(ria_transport_frame(&t, ria_monotonic_ms() + 2000, 1000,
                                  RIA_FRAME_MAX, &response, &payload, &e));
        CHECK(response.request_id == 10 + i && response.flags == 1 &&
              response.payload_length == 64000);
        for (size_t j = 0; j < 64000; j++)
          CHECK(payload[j] == 0xa5);
        free(payload);
      }
  } else
    CHECK(!ok);
  if (t.ssl)
    ria_transport_close(&t);
  int status;
  CHECK(waitpid(pid, &status, 0) == pid);
  CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}
static void test_tls_configuration(void) {
  char directory[] = "/tmp/ria-contracts-XXXXXX";
  CHECK(mkdtemp(directory));
  char ca_path[256], key_path[256], server_path[256], client_path[256];
  snprintf(ca_path, sizeof ca_path, "%s/ca.pem", directory);
  snprintf(key_path, sizeof key_path, "%s/key.pem", directory);
  snprintf(server_path, sizeof server_path, "%s/server.pem", directory);
  snprintf(client_path, sizeof client_path, "%s/client.pem", directory);
  EVP_PKEY *key = EVP_PKEY_Q_keygen(NULL, NULL, "EC", "prime256v1");
  CHECK(key);
  X509 *ca = certificate(key, NULL, NULL, "fixture root", 1, true),
       *server = certificate(key, ca, key, "server.test", 2, false),
       *client = certificate(key, ca, key, "client.test", 3, false);
  pem(ca_path, NULL, ca);
  pem(key_path, key, NULL);
  pem(server_path, NULL, server);
  pem(client_path, NULL, client);
  ria_tls a = {0}, b = {0};
  ria_error e = {0};
  ria_tls_config sc = {ca_path,       server_path, key_path,
                       "client.test", true,        1000},
                 cc = {ca_path,       client_path, key_path,
                       "server.test", false,       1000};
  char fifo_path[256], encrypted_path[256];
  CHECK(snprintf(fifo_path, sizeof fifo_path, "%s/key.fifo", directory) > 0);
  CHECK(snprintf(encrypted_path, sizeof encrypted_path, "%s/encrypted.pem",
                 directory) > 0);
  CHECK(mkfifo(fifo_path, 0600) == 0);
  sc.private_key_file = fifo_path;
  alarm(2);
  CHECK(!ria_tls_create(&a, &sc, &e) && e.code == RIA_UNAUTHORIZED &&
        !a.context);
  alarm(0);
  FILE *encrypted = fopen(encrypted_path, "wb");
  unsigned char password[] = "fixture-pass";
  CHECK(encrypted &&
        PEM_write_PrivateKey(encrypted, key, EVP_aes_256_cbc(), password,
                             (int)sizeof password - 1, NULL, NULL) == 1 &&
        fclose(encrypted) == 0);
  sc.private_key_file = encrypted_path;
  alarm(2);
  CHECK(!ria_tls_create(&a, &sc, &e) && e.code == RIA_UNAUTHORIZED &&
        !a.context);
  alarm(0);
  sc.private_key_file = key_path;
  CHECK(unlink(fifo_path) == 0 && unlink(encrypted_path) == 0);
  CHECK(ria_tls_create(&a, &sc, &e));
  CHECK(ria_tls_create(&b, &cc, &e));
  tls_exchange(&a, &b, 1);
  tls_exchange(&a, &b, 2);
  ria_tls_destroy(&b);
  cc.expected_peer_name = "unauthorized.test";
  CHECK(ria_tls_create(&b, &cc, &e));
  tls_exchange(&a, &b, false);
  ria_tls_destroy(&a);
  ria_tls_destroy(&b);
  X509_free(ca);
  X509_free(server);
  X509_free(client);
  EVP_PKEY_free(key);
  CHECK(unlink(ca_path) == 0 && unlink(key_path) == 0 &&
        unlink(server_path) == 0 && unlink(client_path) == 0 &&
        rmdir(directory) == 0);
}
int main(void) {
  CHECK(signal(SIGPIPE, SIG_IGN) != SIG_ERR);
  test_json();
  test_json_files();
  test_wire();
  test_control();
  test_lifecycle();
  test_admission();
  test_tls_configuration();
  puts("RIA native contracts: JSON/JCS, wire, lifecycle, admission, TLS1.3 "
       "mTLS/partial "
       "IO/deadlines passed");
  return 0;
}
