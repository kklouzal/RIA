"""Execute the actual CUDA host boundary prefix without a driver or device.

Only the code after the first CUDA call is replaced by an explicit sentinel.
The private context layout, argument/alias/coefficient checks and shared range
predicate come from production sources. Accepted inputs never execute CUDA.
"""

from pathlib import Path
import re
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


def source_block(text, signature):
    start = re.search(re.escape(signature), text)
    if start is None:
        raise AssertionError(f"production declaration missing: {signature}")
    opening = text.index("{", start.start())
    depth, end = 1, opening + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start.start():end]


class CudaHostBoundary(unittest.TestCase):
    def test_actual_prefix_matches_cpu_alias_contract_before_device_boundary(self):
        text = (ROOT / "ria/expert_cuda.cu").read_text()
        context = source_block(text, "struct ria_expert_cuda {") + ";"
        failure = source_block(text, "static bool failure(")
        operation = source_block(text, "bool ria_expert_cuda_evaluate_resident(")
        boundary = operation.index("    if (!cuda_failure(cudaSetDevice(")
        prefix = operation[:boundary] + '''
    return failure(e,RIA_NOT_READY,"fixture reached first CUDA boundary");
}
'''
        driver = r'''
#include "expert_cuda.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
typedef void *cudaStream_t;
''' + context + failure + prefix + r'''
static unsigned checks;
static void rejected(ria_expert_cuda *c,const ria_expert *expert,const ria_expert_cuda_resident *resident,
                     const float *input,uint64_t rows,uint64_t in_stride,const float *coefficients,
                     float *output,uint64_t out_stride) {
    ria_error error={0};
    assert(!ria_expert_cuda_evaluate_resident(c,expert,resident,input,rows,in_stride,coefficients,output,out_stride,&error));
    assert(error.code==RIA_INVALID_REQUEST);++checks;
}
static void accepted(ria_expert_cuda *c,const ria_expert *expert,const ria_expert_cuda_resident *resident,
                     const float *input,uint64_t rows,uint64_t in_stride,const float *coefficients,
                     float *output,uint64_t out_stride) {
    ria_error error={0};
    assert(!ria_expert_cuda_evaluate_resident(c,expert,resident,input,rows,in_stride,coefficients,output,out_stride,&error));
    assert(error.code==RIA_NOT_READY);++checks;
}
int main(void) {
    uint8_t zeros[12]={0};
    ria_expert expert={0};expert.clamp=10;
    expert.gate=(ria_expert_matrix){.profile=RIA_EXPERT_BF16,.out_features=2,.in_features=3,
        .values=zeros,.values_bytes=sizeof(zeros),.value_row_stride=6};
    expert.up=expert.gate;
    expert.down=(ria_expert_matrix){.profile=RIA_EXPERT_BF16,.out_features=3,.in_features=2,
        .values=zeros,.values_bytes=sizeof(zeros),.value_row_stride=4};
    ria_expert_cuda c={0};c.input=3;c.intermediate=2;c.output=3;c.rows=64;
    float input[32],output[32],coefficients[4]={1,1,1,1};
    for (unsigned i=0;i<32;i++) input[i]=output[i]=0.5f;
    ria_expert_cpu *cpu=NULL;ria_error error={0};
    assert(ria_expert_validate(&expert,&error));
    assert(ria_expert_cpu_create(3,2,3,&cpu,&error));
    accepted(&c,&expert,NULL,input,2,5,coefficients,output,5);
    assert(ria_expert_cpu_evaluate(cpu,&expert,input,2,5,coefficients,output,5,&error));
    rejected(&c,&expert,NULL,input,2,5,coefficients,input,5);
    assert(!ria_expert_cpu_evaluate(cpu,&expert,input,2,5,coefficients,input,5,&error));
    assert(error.code==RIA_INVALID_REQUEST);
    /* Stride holes are inside the protected logical span. */
    rejected(&c,&expert,NULL,input,2,5,coefficients,input+3,5);
    rejected(&c,&expert,NULL,input,2,5,input+3,output,5);
    rejected(&c,&expert,NULL,input,2,5,output+3,output,5);
    assert(!ria_expert_cpu_evaluate(cpu,&expert,input,2,5,input+3,output,5,&error));
    assert(!ria_expert_cpu_evaluate(cpu,&expert,input,2,5,output+3,output,5,&error));
    /* Minimal exact extent: (rows-1)*stride + width = eight floats.
     * Trailing row padding is outside the span and may hold another buffer. */
    accepted(&c,&expert,NULL,input,2,5,coefficients,input+8,5);
    assert(ria_expert_cpu_evaluate(cpu,&expert,input,2,5,coefficients,input+8,5,&error));
    accepted(&c,&expert,NULL,input,2,5,input+8,output,5);
    assert(ria_expert_cpu_evaluate(cpu,&expert,input,2,5,input+8,output,5,&error));
    accepted(&c,&expert,NULL,input,2,5,output+8,output,5);
    assert(ria_expert_cpu_evaluate(cpu,&expert,input,2,5,output+8,output,5,&error));
    accepted(&c,&expert,NULL,input,2,5,NULL,output,5);
    rejected(&c,&expert,NULL,input,2,UINT64_MAX,coefficients,output,5);
    rejected(&c,&expert,NULL,input,2,5,coefficients,output,UINT64_MAX);
    rejected(&c,&expert,NULL,input,UINT64_MAX,5,coefficients,output,5);
    rejected(&c,&expert,NULL,input,2,0,coefficients,output,5);
    coefficients[1]=NAN;rejected(&c,&expert,NULL,input,2,5,coefficients,output,5);
    coefficients[1]=-1;rejected(&c,&expert,NULL,input,2,5,coefficients,output,5);coefficients[1]=1;
    ria_expert_cuda other={0};ria_expert_cuda_resident resident={0};resident.expert=expert;resident.owner=&other;
    rejected(&c,&expert,&resident,input,2,5,coefficients,output,5);
    resident.owner=&c;accepted(&c,NULL,&resident,input,2,5,coefficients,output,5);
    assert(!ria_expert_ranges_disjoint((void *)(uintptr_t)(UINTPTR_MAX-3),8,input,sizeof(input)));
    assert(ria_expert_ranges_disjoint(input,8*sizeof(float),input+8,8*sizeof(float)));
    assert(!ria_expert_ranges_disjoint(input,8*sizeof(float),input+7,sizeof(float)));
    ria_expert_cpu_destroy(cpu);
    printf("CUDA host boundary: %u actual-prefix cases plus CPU/range comparisons passed (no CUDA)\n",checks);
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="ria-cuda-host-boundary-") as directory:
            source, executable = Path(directory) / "boundary.c", Path(directory) / "boundary"
            source.write_text(driver)
            subprocess.run([
                "cc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-Wformat=2", "-Wstrict-prototypes",
                "-fno-fast-math", "-ffp-contract=off", "-I", str(ROOT / "ria"), str(source),
                str(ROOT / "ria/expert.c"), str(ROOT / "ria/common.c"), "-lcrypto", "-lm", "-o", str(executable),
            ], check=True)
            completed = subprocess.run([str(executable)], capture_output=True, text=True, check=True)
            self.assertIn("17 actual-prefix cases", completed.stdout)


if __name__ == "__main__":
    unittest.main()
