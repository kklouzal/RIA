#include "server.h"
#include <stdio.h>
#include <string.h>
int main(int argc,char **argv) {
  ria_error error={0};
  if (argc!=3 || strcmp(argv[1],"--config") || !argv[2][0]) {
    fputs("usage: ds4-expert-server --config PATH\n",stderr); return RIA_INVALID_REQUEST;
  }
  if (!ria_server_run(argv[2],&error)) {
    fprintf(stderr,"expert service failed (%d): %s\n",error.code,error.message);
    return error.code ? error.code : RIA_INTERNAL_ERROR;
  }
  return 0;
}
