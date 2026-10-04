#define _POSIX_C_SOURCE 200809L
#include "ria/probe.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static void write_bytes(const char *path, const char *data) {
  FILE *f = fopen(path, "wb");
  assert(f);
  assert(fwrite(data, 1, strlen(data), f) == strlen(data));
  assert(!fclose(f));
}
int main(void) {
  char dir[] = "/tmp/ria-probe-schema-XXXXXX";
  assert(mkdtemp(dir));
  char path[256];
  snprintf(path, sizeof(path), "%s/probe.json", dir);
  const char *valid =
      "{\"schema_revision\":1,\"role\":\"expert\",\"executor\":\"cpu\","
      "\"device_index\":null,\"expected_gpu_uuid\":null,\"numa_nodes\":[0,2],"
      "\"max_host_test_bytes\":8192,\"max_device_test_bytes\":0,\"max_pinned_"
      "test_bytes\":0,\"deadline_ms\":1000,\"disable_core_dumps\":true,"
      "\"environment_digest\":"
      "\"0000000000000000000000000000000000000000000000000000000000000000\","
      "\"build_digest\":"
      "\"1111111111111111111111111111111111111111111111111111111111111111\","
      "\"build_info_file\":\"/fixture/build-info.json\"}";
  write_bytes(path, valid);
  ria_probe_config c;
  ria_error e = {0};
  assert(ria_probe_config_read(path, &c, &e));
  assert(!c.cuda && !c.client && c.node_count == 2 && c.nodes[1] == 2 &&
         c.host_test_bytes == 8192);
  const char *bad[] = {"{}", "{\"schema_revision\":1,\"schema_revision\":1}",
                       "{\"schema_revision\":NaN}"};
  for (unsigned i = 0; i < sizeof(bad) / sizeof(*bad); ++i) {
    write_bytes(path, bad[i]);
    assert(!ria_probe_config_read(path, &c, &e));
  }
  char modified[2048];
  strcpy(modified, valid);
  char *p = strstr(modified, "[0,2]");
  assert(p);
  p[3] = '0';
  write_bytes(path, modified);
  assert(!ria_probe_config_read(path, &c, &e));
  strcpy(modified, valid);
  p = strstr(modified, "8192");
  assert(p);
  memcpy(p, "0000", 4);
  write_bytes(path, modified);
  assert(!ria_probe_config_read(path, &c, &e));
  assert(ria_report_write(path, "{\"fixture\":true}", 16, &e));
  struct stat st;
  assert(!stat(path, &st) && (st.st_mode & 0777) == 0600);
  assert(!unlink(path));
  assert(!rmdir(dir));
  puts("RIA bounded probe schema and atomic publication fixtures passed (no "
       "physical probe run)");
  return 0;
}
