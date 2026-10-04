"""Run the exact CUDA attention body on a bounded pthread barrier recorder.

CUDA builtins become ordinary host test operations; no CUDA runtime or GPU is
loaded. This proves mask/slot/cast control flow, not GPU numerical qualification.
"""
from pathlib import Path
import importlib.util
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("ria_cleanup_fixture", ROOT / "tests/ria/test_cuda_lifetime.py")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

RECORDER = r'''
#define _GNU_SOURCE
#include "ria/numeric.h"
#include <assert.h>
#include <stdbool.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#define __global__
#define __shared__ static
typedef struct { unsigned x; } dimension;
static _Thread_local dimension threadIdx;
static dimension blockIdx;
static pthread_barrier_t barrier;
static void __syncthreads(void) { int r=pthread_barrier_wait(&barrier);assert(!r || r==PTHREAD_BARRIER_SERIAL_THREAD); }
static float __fmaf_rn(float x,float y,float z) { return fmaf(x,y,z); }
static void atomicExch(int *p,int value) { __atomic_store_n(p,value,__ATOMIC_SEQ_CST); }
'''

CASES = r'''
typedef struct {
    const float *q,*kv,*sink;float *output;uint64_t count,begin,end;int *error;unsigned lane;
} work;
static void *run_lane(void *pointer) {
    work *w=pointer;threadIdx.x=w->lane;
    attention_kernel(w->q,w->kv,w->count,w->begin,w->end,w->sink,w->output,w->error);return NULL;
}
static void run(const float *q,const float *kv,uint64_t count,uint64_t begin,uint64_t end,const float *sink,float *out) {
    pthread_t threads[256];work jobs[256];int error=0;pthread_attr_t attr;
    assert(!pthread_barrier_init(&barrier,NULL,256));assert(!pthread_attr_init(&attr));
    long minimum=sysconf(_SC_THREAD_STACK_MIN);assert(minimum>0 && minimum<16777216);
    size_t stack=(size_t)minimum+65536;if (stack<262144) stack=262144;
    assert(!pthread_attr_setstacksize(&attr,stack));blockIdx.x=0;
    for (unsigned i=0;i<256;++i) {
        jobs[i]=(work){q,kv,sink,out,count,begin,end,&error,i};
        assert(!pthread_create(&threads[i],&attr,run_lane,&jobs[i]));
    }
    for (unsigned i=0;i<256;++i) assert(!pthread_join(threads[i],NULL));
    assert(!pthread_attr_destroy(&attr));assert(!pthread_barrier_destroy(&barrier));assert(!error);
    for (unsigned i=0;i<512;++i) assert(isfinite(out[i]));
}
int main(void) {
    float q[32768]={0},sink[64]={0},out[512],compact_out[512];
    float *kv=malloc(129*512*sizeof(float)),compact[3*512]={0};assert(kv);
    /* Hand-encoded, independently representable source caches: window1024
     * is E4M3 code0x78 times UE8M0 scale0x81; compressed [64,-384] is
     * E2M1 nibble pair0xf2 times E4M3 scale0x68. Q is BF16-logical. */
    q[0]=0.353515625f;
    for (unsigned i=0;i<129*512;++i) kv[i]=NAN;
    for (unsigned row=126;row<129;++row) for (unsigned k=0;k<512;++k) kv[row*512+k]=0;
    kv[126*512+1]=1024;kv[128*512]=64;kv[128*512+1]=-384;
    run(q,kv,129,0,126,sink,out);assert(out[1]==-3.453125f);
    compact[1]=1024;compact[2*512]=64;compact[2*512+1]=-384;
    run(q,compact,3,0,0,sink,compact_out);assert(compact_out[1]==-3.796875f && compact_out[1]!=out[1]);
    /* Invalid tiles after an already initialized aggregate contribute no
     * update; NaN/Inf payload in each masked position is never observed. */
    for (unsigned i=0;i<129*512;++i) kv[i]=i&1u ? INFINITY : NAN;
    for (unsigned row=0;row<2;++row) for (unsigned k=0;k<512;++k) kv[row*512+k]=compact[row*512+k];
    for (unsigned k=0;k<512;++k) kv[128*512+k]=compact[2*512+k];
    run(q,kv,129,2,128,sink,out);assert(out[1]==-3.453125f);
    run(q,kv,128,0,128,sink,out);for (unsigned i=0;i<512;++i) assert(out[i]==0);
    free(kv);puts("source sparse slots, masked payload and BF16 tile boundaries: passed (no CUDA)");return 0;
}
'''


class AttentionSlots(unittest.TestCase):
    def test_actual_kernel_mask_and_source_probability_boundaries(self):
        kernel = _module.body(ROOT / "ria/graph_cuda.cu", r"static __global__ void attention_kernel\(")
        with tempfile.TemporaryDirectory(prefix="ria-attention-slots-") as directory:
            source, binary = Path(directory) / "attention.c", Path(directory) / "attention"
            source.write_text(RECORDER + kernel + CASES)
            subprocess.run(["gcc", "-std=c11", "-O2", "-ffp-contract=off", "-fno-fast-math", "-Wall", "-Wextra", "-Wconversion",
                "-Werror", "-I", str(ROOT), str(source), "-lpthread", "-lm", "-o", str(binary)], check=True)
            subprocess.run([binary], check=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
