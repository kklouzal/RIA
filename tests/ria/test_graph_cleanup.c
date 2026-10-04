#define _GNU_SOURCE
/* Real graph/engine cleanup callers, with failed CUDA completion recorded on
 * the host. No GPU/runtime/model is involved. */
#include <assert.h>
#include <sys/wait.h>
#include "../../ria/engine.c"
#define ria_graph_create fixture_discarded_graph_create
#include "../../ria/graph.c"
#undef ria_graph_create

static unsigned *releases;
static bool fail_graph_cuda=true;
bool ria_graph_cuda_destroy(ria_graph_cuda *c,ria_error *e) {
    (void)c;
    return !fail_graph_cuda || ria_fail(e,RIA_EXECUTOR_ERROR,"recorded failed CUDA completion");
}
bool ria_expert_cuda_resident_destroy(ria_expert_cuda_resident *v,ria_error *e) { (void)v;(void)e;return true; }
bool ria_vision_destroy(ria_vision *v,ria_error *e) { (void)v;(void)e;return true; }
bool ria_remote_close(ria_remote *r,ria_error *e) { (void)r;(void)e;++*releases;return true; }
bool ria_remote_open(ria_remote **r,const ria_remote_options *o,ria_error *e) { (void)r;(void)o;return ria_fail(e,RIA_INTERNAL_ERROR,"unexpected fixture remote reopen"); }
ria_graph_remote ria_remote_callbacks(ria_remote *r) { (void)r;return (ria_graph_remote){0}; }
bool ria_graph_create(const ria_tensor_store *s,const ria_graph_options *o,ria_graph_remote remote,ria_graph **out,ria_error *e) {
    (void)s;(void)o;(void)remote;(void)out;return ria_fail(e,RIA_INTERNAL_ERROR,"unexpected fixture graph reopen");
}
void ria_tokenizer_close(ria_tokenizer *t) { (void)t;++*releases; }
void ria_tokenizer_runtime_end(void) { ++*releases; }
void ria_tensor_store_close(ria_tensor_store *s) { (void)s;++*releases; }
void ria_service_free(ria_service *s) { (void)s;++*releases; }

static ria_engine *owner(void) {
    ria_engine *r=calloc(1,sizeof(*r));assert(r);
    long page=sysconf(_SC_PAGESIZE);assert(page>0);
    uint64_t bytes=(sizeof(ria_graph)+(uint64_t)page-1)&~((uint64_t)page-1);
    r->graph=mmap(NULL,(size_t)bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);assert(r->graph!=MAP_FAILED);
    r->graph->owner_bytes=bytes;r->graph->state_bytes=(uint64_t)page;
    r->graph->host_allocation=mmap(NULL,(size_t)page,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    assert(r->graph->host_allocation!=MAP_FAILED);((uint8_t *)r->graph->host_allocation)[0]=71;
    r->graph->cuda=(ria_graph_cuda *)1;r->valid=true;return r;
}
static void preserved(const ria_engine *r) {
    assert(!*releases && r->graph && r->graph->poisoned);
    assert(((uint8_t *)r->graph->host_allocation)[0]==71);
}
static void discard_fixture(ria_engine *r) {
    assert(!munmap(r->graph->host_allocation,(size_t)r->graph->state_bytes));
    assert(!munmap(r->graph,(size_t)r->graph->owner_bytes));free(r);
}
int main(void) {
    releases=mmap(NULL,4096,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANONYMOUS,-1,0);assert(releases!=MAP_FAILED);
    ria_engine *r=owner();ria_error e={0};ria_graph *original=r->graph;
    assert(!ria_engine_close(r,&e));assert(e.code==RIA_EXECUTOR_ERROR && r->fatal && r->graph==original);preserved(r);
    assert(!ria_engine_close(r,&e));preserved(r);discard_fixture(r);
    r=owner();pid_t child=fork();assert(child>=0);
    if (!child) { (void)reconnect(r,&e);_Exit(99); }
    int status=0;assert(waitpid(child,&status,0)==child);
    assert(WIFEXITED(status) && WEXITSTATUS(status)==RIA_EXECUTOR_ERROR && !*releases);
    /* The child terminates without invoking source/backing cleanup. */
    discard_fixture(r);assert(!munmap(releases,4096));
    puts("graph/engine failed-drain backing preservation and fail-stop: passed");return 0;
}
