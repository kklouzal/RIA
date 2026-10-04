#define _POSIX_C_SOURCE 200809L
#include "admission.h"
#include "probe.h"
#include "qualify_transport.h"
#include "service.h"
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  const char *key, *value;
} option;
static bool options(int argc, char **argv, option *o, size_t count,
                    ria_error *e) {
  for (int i = 2; i < argc; ++i) {
    size_t j = 0;
    for (; j < count; ++j)
      if (!strcmp(argv[i], o[j].key))
        break;
    if (j == count || o[j].value || i + 1 >= argc || !argv[i + 1][0])
      return ria_fail(e, RIA_INVALID_REQUEST,
                      "unknown, duplicate or missing command option");
    o[j].value = argv[++i];
  }
  return true;
}
static bool required(const option *o, size_t n, ria_error *e) {
  for (size_t i = 0; i < n; ++i)
    if (!o[i].value)
      return ria_fail(e, RIA_INVALID_REQUEST, "missing required option %s",
                      o[i].key);
  return true;
}
static void error_json(const ria_error *e) {
  /* Diagnostic strings contain no credentials; escape all JSON metacharacters.
   */
  fprintf(stderr, "{\"ok\":false,\"code\":%d,\"message\":\"", e->code);
  for (const unsigned char *p = (const unsigned char *)e->message; *p; ++p) {
    if (*p == '"' || *p == '\\')
      fprintf(stderr, "\\%c", *p);
    else if (*p < 32)
      fprintf(stderr, "\\u%04x", *p);
    else
      fputc(*p, stderr);
  }
  fputs("\"}\n", stderr);
}
static bool validate(const char *path, ria_error *e) {
  ria_service s = {0};
  if (!ria_service_read(&s, path, e))
    return false;
  ria_json_doc manifest = {0};
  uint8_t actual[32], claimed[32], logical[32], operator_digest[32];
  bool ok =
      ria_json_read(s.manifest_path, (ria_json_limits){256u << 10, 32768, 64},
                    &manifest, e) &&
      ria_json_digest_field(&manifest, ria_json_get(&manifest, 0, "digest"),
                            claimed, e) &&
      ria_json_sha256(&manifest, true, actual, e) &&
      !memcmp(actual, claimed, 32) && !memcmp(actual, s.manifest_digest, 32) &&
      ria_json_digest_field(&manifest,
                            ria_json_get(&manifest, 0, "logical_model_digest"),
                            logical, e) &&
      ria_json_digest_field(
          &manifest, ria_json_get(&manifest, 0, "operator_contract_digest"),
          operator_digest, e) &&
      !memcmp(logical, s.logical_model_digest, 32) &&
      !memcmp(operator_digest, s.operator_contract_digest, 32);
  const char *role = NULL, *profile = NULL;
  size_t n;
  if (ok)
    ok = ria_json_string(&manifest, ria_json_get(&manifest, 0, "role"), &role,
                         &n, e) &&
         strlen(role) == n &&
         !strcmp(role, !strcmp(s.role, "client") ? "client" : "server") &&
         ria_json_string(&manifest, ria_json_get(&manifest, 0, "profile"),
                         &profile, &n, e) &&
         strlen(profile) == n && !strcmp(profile, s.profile);
  if (!ok && !e->code)
    ria_error_set(e, RIA_INTEGRITY_ERROR,
                  "service package and trusted model identity differ");
  ria_json_free(&manifest);
  ria_service_free(&s);
  return ok;
}
int main(int argc, char **argv) {
  ria_error e = {0};
  if (!ria_disable_dumps(&e)) {
    error_json(&e);
    return e.code;
  }
  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sa.sa_handler = SIG_IGN;
  sigemptyset(&sa.sa_mask);
  if (sigaction(SIGPIPE, &sa, NULL)) {
    ria_error_set(&e, RIA_INTERNAL_ERROR, "cannot establish SIGPIPE policy");
    error_json(&e);
    return e.code;
  }
  bool ok = false;
  if (argc < 2)
    ria_error_set(
        &e, RIA_INVALID_REQUEST,
        "usage: ds4ctl validate|probe|plan|qualify-transport|health|drain [explicit options]");
  else if (!strcmp(argv[1], "qualify-transport")) {
    option o[] = {{"--config", NULL}, {"--request", NULL}};
    char *report = NULL;
    size_t length = 0;
    ok = options(argc, argv, o, 2, &e) && required(o, 2, &e) &&
         ria_qualify_transport(o[0].value, o[1].value, &report, &length, &e);
    if (ok && (fwrite(report, 1, length, stdout) != length || fputc('\n', stdout) == EOF || fflush(stdout)))
      ok = ria_fail(&e, RIA_INTERNAL_ERROR, "transport fixture report output failed");
    free(report);
    if (ok)
      return 0;
  }
  else if (!strcmp(argv[1], "plan")) {
    option o[] = {{"--request", NULL},
                  {"--inventory", NULL},
                  {"--probe", NULL},
                  {"--calibration", NULL},
                  {"--output", NULL}};
    ok = options(argc, argv, o, 5, &e) && required(o, 5, &e) &&
         ria_admission_files(o[0].value, o[1].value, o[2].value, o[3].value,
                             o[4].value, &e);
  } else if (!strcmp(argv[1], "validate")) {
    option o[] = {{"--config", NULL}};
    ok = options(argc, argv, o, 1, &e) && required(o, 1, &e) &&
         validate(o[0].value, &e);
  } else if (!strcmp(argv[1], "probe")) {
    option o[] = {{"--probe-config", NULL},
                  {"--output", NULL},
                  {"--details-output", NULL}};
    ria_probe_config c;
    char *report = NULL, *detail = NULL;
    size_t rn = 0, dn = 0;
    ok = options(argc, argv, o, 3, &e) && required(o, 2, &e) &&
         ria_probe_config_read(o[0].value, &c, &e);
    char *detail_path = NULL;
    if (ok && !o[2].value) {
      size_t n = strlen(o[1].value);
      detail_path = malloc(n + 14);
      if (!detail_path)
        ok = ria_fail(&e, RIA_RESOURCE_LIMIT, "detail path allocation failed");
      else {
        memcpy(detail_path, o[1].value, n);
        memcpy(detail_path + n, ".details.json", 14);
        o[2].value = detail_path;
      }
    }
    if (ok && (!o[1].value || !o[2].value))
      ok = ria_fail(&e, RIA_INVALID_REQUEST,
                    "probe evidence destinations are missing");
    if (ok && !strcmp(o[1].value, o[2].value))
      ok = ria_fail(&e, RIA_INVALID_REQUEST,
                    "compact and detailed evidence destinations must differ");
    if (ok)
      ok = ria_probe_run(&c, &report, &rn, &detail, &dn, &e);
    /* Compact report is the admission commit point. */
    if (ok)
      ok = ria_report_write(o[2].value, detail, dn, &e) &&
           ria_report_write(o[1].value, report, rn, &e);
    free(report);
    free(detail);
    free(detail_path);
  } else if (!strcmp(argv[1], "health") || !strcmp(argv[1], "drain")) {
    option o[] = {{"--socket", NULL}, {"--timeout-ms", NULL}};
    uint64_t timeout = 10000;
    ok = options(argc, argv, o, 2, &e);
    if (ok && o[1].value)
      ok = ria_parse_u64(o[1].value, strlen(o[1].value), &timeout, &e) &&
           timeout && timeout <= 2147483647;
    char *reply = NULL;
    size_t n = 0;
    if (ok)
      ok = ria_admin_request(o[0].value ? o[0].value
                                        : "/run/dwarfstar/admin.sock",
                             argv[1], timeout, &reply, &n, &e);
    if (ok) {
      ria_json_doc d = {0};
      ok = ria_json_parse(reply, n, (ria_json_limits){16384, 1024, 16}, &d, &e);
      const ria_json_node *status =
          ok ? ria_json_at(&d, ria_json_get(&d, 0, "ok")) : NULL;
      if (ok && (!status || status->type != RIA_JSON_BOOL || !status->boolean))
        ok = ria_fail(&e, RIA_NOT_READY,
                      "administrative operation reports failure");
      if (ok && !strcmp(argv[1], "health")) {
        const ria_json_node *ready =
            ria_json_at(&d, ria_json_get(&d, 0, "ready"));
        if (!ready || ready->type != RIA_JSON_BOOL || !ready->boolean)
          ok = ria_fail(&e, RIA_NOT_READY, "service is not ready");
      }
      ria_json_free(&d);
      if (ok && fwrite(reply, 1, n, stdout) != n)
        ok = ria_fail(&e, RIA_INTERNAL_ERROR,
                      "cannot write structured administrative result");
    }
    free(reply);
    if (ok)
      return fflush(stdout) == 0 ? 0 : RIA_INTERNAL_ERROR;
  } else
    ria_error_set(&e, RIA_INVALID_REQUEST, "unknown native control command");
  if (!ok) {
    if (!e.code)
      ria_error_set(&e, RIA_INVALID_REQUEST, "invalid command input");
    error_json(&e);
    return e.code;
  }
  if (fputs("{\"ok\":true}\n", stdout) == EOF || fflush(stdout))
    return RIA_INTERNAL_ERROR;
  return 0;
}
