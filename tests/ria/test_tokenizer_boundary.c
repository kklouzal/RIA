#define _POSIX_C_SOURCE 200809L
#include "ria/tokenizer.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static const char *mutation_path;
static bool mutated;
#define REQUIRE(condition) do { if (!(condition)) abort(); } while (0)
ssize_t __real_read(int fd, void *bytes, size_t length);
ssize_t __wrap_read(int fd, void *bytes, size_t length);
ssize_t __real___read_chk(int fd, void *bytes, size_t length, size_t capacity);
ssize_t __wrap___read_chk(int fd, void *bytes, size_t length, size_t capacity);
/* Mutate and restore the exact first consumed byte and mtime. A stable source
 * hash alone would accept this concurrent rewrite; descriptor ctime must reject
 * it. The target is always the test's private copy, never pinned metadata. */
static ssize_t mutate_after_read(int fd, void *bytes, ssize_t count) {
  if (mutation_path && !mutated && count > 0) {
    struct stat before;
    REQUIRE(fstat(fd, &before) == 0);
    int writer = open(mutation_path, O_WRONLY | O_CLOEXEC | O_NOFOLLOW);
    REQUIRE(writer >= 0);
    unsigned char original = *(unsigned char *)bytes;
    unsigned char different = (unsigned char)(original ^ 1u);
    REQUIRE(pwrite(writer, &different, 1, 0) == 1);
    REQUIRE(pwrite(writer, &original, 1, 0) == 1);
    struct timespec times[2] = {before.st_atim, before.st_mtim};
    REQUIRE(futimens(writer, times) == 0);
    REQUIRE(close(writer) == 0);
    mutated = true;
  }
  return count;
}
ssize_t __wrap_read(int fd, void *bytes, size_t length) {
  return mutate_after_read(fd, bytes, __real_read(fd, bytes, length));
}
ssize_t __wrap___read_chk(int fd, void *bytes, size_t length, size_t capacity) {
  return mutate_after_read(fd, bytes, __real___read_chk(fd, bytes, length, capacity));
}

int main(int argc, char **argv) {
  if (argc < 3 || argc > 4)
    return 2;
  uint8_t expected[32];
  ria_error error = {0};
  if (!ria_hex_decode(argv[2], strlen(argv[2]), expected, sizeof expected, &error) ||
      !ria_tokenizer_runtime_begin(&error))
    return 2;
  if (argc == 4)
    mutation_path = argv[1];
  ria_tokenizer *tokenizer = (ria_tokenizer *)(uintptr_t)1;
  bool ok = ria_tokenizer_open(argv[1], expected, UINT64_C(268435456),
                               &tokenizer, &error);
  if (!ok && tokenizer != NULL)
    return 3;
  if (ok) {
    REQUIRE(ria_tokenizer_vocab(tokenizer) == 129280);
    puts("pinned tokenizer loaded");
    ria_tokenizer_close(tokenizer);
  } else {
    fprintf(stderr, "%d: %s\n", error.code, error.message);
  }
  ria_tokenizer_runtime_end();
  return ok ? 0 : 1;
}
