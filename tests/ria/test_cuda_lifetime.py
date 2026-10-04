"""Execute exact CUDA cleanup bodies with a driver-free failure recorder.

Only the owned host structs and cleanup functions are extracted. No kernel,
CUDA header, driver, device query or model data participates in this fixture.
The recorder verifies that failed completion cannot release owned resources.
"""
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


def body(path, pattern, *, structure=False):
    source = path.read_text()
    match = re.search(pattern, source)
    if not match:
        raise AssertionError(f"cleanup source definition missing: {pattern}")
    start = source.index("{", match.start())
    depth = 1
    end = start + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start():end] + (";" if structure else "")


RECORDER = r'''
#include "ria/graph_cuda.h"
#include "ria/vision.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef void *cudaStream_t;
typedef int cudaError_t;
#define cudaSuccess 0
static int select_failure,sync_failure,release_failure;
static unsigned device_frees,pinned_frees,stream_frees;
static cudaError_t cudaSetDevice(int device) { (void)device;return select_failure; }
static cudaError_t cudaStreamSynchronize(cudaStream_t stream) { assert(stream);return sync_failure; }
static cudaError_t cudaFree(void *pointer) { ++device_frees;if (release_failure) return release_failure;free(pointer);return 0; }
static cudaError_t cudaFreeHost(void *pointer) { ++pinned_frees;free(pointer);return 0; }
static cudaError_t cudaStreamDestroy(cudaStream_t stream) { assert(stream);++stream_frees;return 0; }
static const char *cudaGetErrorString(cudaError_t status) { (void)status;return "recorded CUDA failure"; }
'''

CASES = r'''
static ria_expert_cuda *expert(void) {
    ria_expert_cuda *c=calloc(1,sizeof(*c));assert(c);c->stream=(void *)1;
    c->input_values=malloc(8);c->pinned=malloc(8);assert(c->input_values && c->pinned);
    c->input_values[0]=42;c->pinned[0]=91;return c;
}
static void failure_mode(unsigned mode) {
    select_failure=mode==0;sync_failure=mode==1;release_failure=mode==2;
    device_frees=pinned_frees=stream_frees=0;
}
int main(void) {
    ria_error e={0};
    for (unsigned mode=0;mode<3;++mode) {
        ria_expert_cuda *c=expert();failure_mode(mode);
        assert(!ria_expert_cuda_destroy(c,&e));assert(e.code==RIA_EXECUTOR_ERROR);
        assert(device_frees==(mode==2));assert(!pinned_frees && !stream_frees);
        assert(c->input_values[0]==42 && c->pinned[0]==91);
        failure_mode(3);assert(ria_expert_cuda_destroy(c,&e));
        assert(device_frees==1 && pinned_frees==1 && stream_frees==1);
    }
    for (unsigned mode=0;mode<3;++mode) {
        ria_graph_cuda *g=calloc(1,sizeof(*g));assert(g);g->stream=(void *)1;g->projection=expert();
        g->buffers[0]=malloc(8);g->buffers[0][0]=43;failure_mode(mode);
        assert(!ria_graph_cuda_destroy(g,&e));assert(g->buffers[0][0]==43 && g->projection->pinned[0]==91);
        assert(device_frees==(mode==2));assert(!pinned_frees && !stream_frees);
        failure_mode(3);assert(ria_graph_cuda_destroy(g,&e));assert(device_frees==2 && pinned_frees==1 && stream_frees==1);
    }
    for (unsigned mode=0;mode<3;++mode) {
        ria_vision_cuda *v=calloc(1,sizeof(*v));assert(v);v->stream=(void *)1;v->projection=expert();
        v->input=malloc(8);v->input[0]=44;failure_mode(mode);
        assert(!ria_vision_cuda_destroy(v,&e));assert(v->input[0]==44 && v->projection->pinned[0]==91);
        assert(device_frees==(mode==2));assert(!pinned_frees && !stream_frees);
        failure_mode(3);assert(ria_vision_cuda_destroy(v,&e));assert(device_frees==2 && pinned_frees==1 && stream_frees==1);
    }
    for (unsigned mode=0;mode<3;++mode) {
        ria_expert_cuda *c=expert();ria_expert_cuda_resident view={0};view.owner=c;view.bytes=8;
        uint8_t *arena=malloc(8);assert(arena);arena[0]=45;view.expert.gate.values=arena;failure_mode(mode);
        assert(!ria_expert_cuda_resident_destroy(&view,&e));assert(view.owner==c && view.bytes==8 && arena[0]==45);
        assert(device_frees==(mode==2));assert(!pinned_frees && !stream_frees);
        failure_mode(3);assert(ria_expert_cuda_resident_destroy(&view,&e));assert(!view.owner && !view.bytes);
        assert(ria_expert_cuda_destroy(c,&e));assert(device_frees==2 && pinned_frees==1 && stream_frees==1);
    }
    puts("failed CUDA cleanup preserves owned resources: passed (no CUDA runtime)");return 0;
}
'''


class CudaLifetime(unittest.TestCase):
    def test_graph_engine_failed_drain_preserves_source_backing(self):
        with tempfile.TemporaryDirectory(prefix="ria-graph-cleanup-recorder-") as directory:
            binary = Path(directory) / "graph-cleanup"
            subprocess.run(["gcc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Wconversion", "-Werror",
                "-ffunction-sections", "-I", str(ROOT), str(ROOT / "tests/ria/test_graph_cleanup.c"),
                str(ROOT / "ria/common.c"), "-Wl,--gc-sections", "-lcrypto", "-lpthread", "-o", str(binary)], check=True)
            subprocess.run([binary], check=True)

    def test_exact_cleanup_bodies_preserve_failed_owners(self):
        expert = ROOT / "ria/expert_cuda.cu"
        graph = ROOT / "ria/graph_cuda.cu"
        vision = ROOT / "ria/vision_cuda.cu"
        definitions = [RECORDER]
        definitions += [body(path, rf"struct {name} \{{", structure=True) for path, name in (
            (expert, "ria_expert_cuda"), (graph, "ria_graph_cuda"), (vision, "ria_vision_cuda"))]
        definitions += [body(expert, r"static bool cuda_failure\("), body(graph, r"static bool check\(")]
        definitions += [body(path, rf"bool {name}\(") for path, name in (
            (expert, "ria_expert_cuda_destroy"), (expert, "ria_expert_cuda_resident_destroy"),
            (graph, "ria_graph_cuda_destroy"), (vision, "ria_vision_cuda_destroy"))]
        with tempfile.TemporaryDirectory(prefix="ria-cuda-cleanup-recorder-") as directory:
            source, binary = Path(directory) / "cleanup.c", Path(directory) / "cleanup"
            source.write_text("\n".join(definitions) + CASES)
            subprocess.run(["gcc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Wconversion", "-Werror",
                "-I", str(ROOT), str(source), str(ROOT / "ria/common.c"), "-lcrypto", "-o", str(binary)], check=True)
            subprocess.run([binary], check=True)


if __name__ == "__main__":
    unittest.main()
