#include "prefill.h"
#include "expert_cuda.h"
#include <unistd.h>
bool ria_graph_prefill_required_bytes(uint32_t rows,uint32_t tile,uint64_t *host,uint64_t *device,uint64_t *pinned,ria_error *e) {
    if (!rows || rows>RIA_GRAPH_PREFILL_MAX_ROWS || !tile || tile>4096 || !host || !device || !pinned)
        return ria_fail(e,RIA_INVALID_REQUEST,"invalid admitted prompt microbatch bounds/results");
    /* Group inputs/results plus stable original row and selected-slot IDs.
     * The reusable projection arena has one row of each listed array. Weight
     * tiles and owner metadata do not grow with the prompt microbatch. */
    uint64_t bytes,sum;
    const uint64_t staging=2*5120*sizeof(float)+sizeof(uint64_t)+sizeof(float)+2*sizeof(uint16_t);
    const uint64_t projection_row=32768*(2*sizeof(float)+2)+2048*(sizeof(float)+1)+
                                  2*32768*sizeof(float)+129280*sizeof(float)+sizeof(float);
    long page=sysconf(_SC_PAGESIZE);
    if (page<=0 || !ria_u64_mul(rows,sizeof(ria_graph_prefill_row)+staging,&bytes) ||
        !ria_u64_add(bytes,(uint64_t)page-1,&sum) || sum>SIZE_MAX ||
        !ria_u64_mul(rows-1,projection_row,device) ||
        !ria_u64_add(*device,RIA_GRAPH_CANDIDATE_BLOCKS*sizeof(uint32_t),device))
        return ria_fail(e,RIA_RESOURCE_LIMIT,"prompt microbatch workspace arithmetic overflow");
    *host=sum/(uint64_t)page*(uint64_t)page;
    return ria_expert_cuda_pinned_required_bytes(32768,32768,129280,rows,tile,pinned,e);
}
