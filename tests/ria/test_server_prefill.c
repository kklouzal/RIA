#define _GNU_SOURCE
/* Actual production worker loop and CPU executor, miniature verified BF16
 * tensors, and explicit NUMA lookup/affinity shims. This exercises grouped
 * ownership/quiescence without querying/binding physical NUMA or any GPU. */
#include "ria/expert.h"
#include "ria/numa_policy.h"
#include <math.h>
#include <time.h>
static bool recorded_evaluate(ria_expert_cpu *,const ria_expert *,const float *,uint64_t,uint64_t,const float *,float *,uint64_t,ria_error *);
static bool fixture_affinity(const ria_expert_node *,unsigned,ria_error *);
static const ria_expert *fixture_expert(const ria_numa *,unsigned,uint64_t,uint16_t);
#define ria_expert_cpu_evaluate recorded_evaluate
#define ria_numa_affinity fixture_affinity
#define ria_numa_expert fixture_expert
#include "ria/server.c"
#undef ria_expert_cpu_evaluate
#undef ria_numa_affinity
#undef ria_numa_expert

#define CHECK(x) do { if (!(x)) { fprintf(stderr,"prefill fixture %s:%d: %s\n",__FILE__,__LINE__,#x);abort(); } } while (0)

typedef struct fixture fixture;
typedef struct { fixture *context;worker *worker; } worker_argument;
struct fixture {
    server owner;
    ria_expert experts[2];uint8_t weights[2][3][2];
    uint64_t evaluated_rows;unsigned batches,largest;
    bool hold,entered,released,fail;
    worker_argument arguments[2];
};
static void wait_locked(fixture *,bool *);
/* Borrow one explicit argument until its worker returns; shims never inspect
 * another worker's unpublished initialization state or process globals. */
static _Thread_local worker_argument *current;
static bool fixture_affinity(const ria_expert_node *node,unsigned index,ria_error *e) {
    (void)node;(void)e;CHECK(current && current->worker && current->worker->index==index && index<2);
    fixture *f=current->context;
    if (f->owner.worker_count==2 && index==0) {
        /* Force worker1 to evaluate before worker0 initializes its capacity.
         * The old recorder's workers[0] predicate fails on this schedule. */
        CHECK(!pthread_mutex_lock(&f->owner.mutex));wait_locked(f,&f->entered);
        CHECK(!pthread_mutex_unlock(&f->owner.mutex));
    }
    return true;
}
static const ria_expert *fixture_expert(const ria_numa *local,unsigned node,uint64_t handle,uint16_t expert) {
    (void)local;CHECK(current && current->context && !node && handle>=11 && handle<=12 && expert>=1 && expert<=2);
    return &current->context->experts[expert-1];
}
static bool recorded_evaluate(ria_expert_cpu *context,const ria_expert *expert,const float *input,uint64_t rows,
                               uint64_t input_stride,const float *coefficients,float *output,uint64_t output_stride,ria_error *e) {
    CHECK(current && current->context && current->worker);
    fixture *f=current->context;worker *w=current->worker;
    CHECK(w->owner==&f->owner && context==w->cpu && rows && rows<=w->batch_rows);
    CHECK(!pthread_mutex_lock(&f->owner.mutex));
    if (!f->entered && f->owner.worker_count==2) CHECK(w->index==1);
    ++f->batches;f->evaluated_rows+=rows;
    if (rows>f->largest) f->largest=(unsigned)rows;
    f->entered=true;CHECK(!pthread_cond_broadcast(&f->owner.condition));
    while (f->hold && !f->released) CHECK(!pthread_cond_wait(&f->owner.condition,&f->owner.mutex));
    bool fail=f->fail;CHECK(!pthread_mutex_unlock(&f->owner.mutex));
    return fail ? ria_fail(e,RIA_EXECUTOR_ERROR,"injected grouped CPU executor failure") :
        ria_expert_cpu_evaluate(context,expert,input,rows,input_stride,coefficients,output,output_stride,e);
}
static void *run(void *pointer) {
    worker_argument *a=pointer;current=a;void *result=worker_main(a->worker);current=NULL;return result;
}
static float independent_bf16(float value) {
    uint32_t raw;memcpy(&raw,&value,4);uint32_t a=raw&UINT32_C(0xffff0000),b=a+UINT32_C(0x10000);
    float lo,hi;memcpy(&lo,&a,4);memcpy(&hi,&b,4);
    double dl=fabs((double)value-lo),dh=fabs((double)hi-value);
    return dl<dh || (dl==dh && !(a&UINT32_C(0x10000))) ? lo : hi;
}
static float oracle(float input,float coefficient,unsigned expert) {
    float logical=independent_bf16(input),silu=logical/(1+expf(-logical));
    float hidden=independent_bf16((silu*logical)*coefficient);
    return independent_bf16(hidden*(float)(expert+1));
}
static void wait_locked(fixture *f,bool *flag) {
    struct timespec deadline;CHECK(!clock_gettime(CLOCK_REALTIME,&deadline));deadline.tv_sec+=5;
    while (!*flag) CHECK(!pthread_cond_timedwait(&f->owner.condition,&f->owner.mutex,&deadline));
}
static void work_create(work *r,unsigned owner,uint32_t rows) {
    const uint32_t entries=rows*2;
    size_t bytes=(size_t)rows*8+(size_t)(rows+1)*4+(size_t)entries*8+(size_t)rows*4;
    r->input=calloc(1,bytes);CHECK(r->input);r->output=malloc(32+(size_t)entries*4);CHECK(r->output);
    memset(r->output,0xa5,32+(size_t)entries*4);r->used=true;r->remaining=entries;r->deadline=ria_monotonic_ms()+5000;
    r->header.kind=RIA_EXPERT;r->parsed=(ria_expert_request){.operation_handle=11+owner,.invocation_id=owner+1,
        .row_count=rows,.input_width=1,.output_width=1,.entry_count=entries,.response_bytes=32+(uint64_t)entries*4};
    uint8_t *cursor=r->input;r->parsed.row_ids=cursor;cursor+=(size_t)rows*8;
    r->parsed.offsets=cursor;cursor+=(size_t)(rows+1)*4;r->parsed.entries=cursor;cursor+=(size_t)entries*8;r->parsed.inputs=cursor;
    for (uint32_t row=0;row<rows;++row) {
        ria_write_u64((uint8_t *)r->parsed.row_ids+row*8,17*(uint64_t)row+3);
        ria_write_u32((uint8_t *)r->parsed.offsets+row*4,row*2);
        ria_write_f32((uint8_t *)r->parsed.inputs+row*4,0.25f+(float)row/64+(float)owner/8);
        for (unsigned slot=0;slot<2;++slot) {
            uint8_t *entry=(uint8_t *)r->parsed.entries+(row*2+slot)*8;
            ria_write_u16(entry,(uint16_t)(slot+1));ria_write_u16(entry+2,(uint16_t)slot);
            ria_write_f32(entry+4,(float)((row+slot)%7)/8);
        }
    }
    ria_write_u32((uint8_t *)r->parsed.offsets+rows*4,entries);
}
static fixture *create(bool hold,bool fail,unsigned workers) {
    fixture *f=calloc(1,sizeof(*f));CHECK(f);f->hold=hold;f->fail=fail;
    server *s=&f->owner;CHECK(!pthread_mutex_init(&s->mutex,NULL) && !pthread_cond_init(&s->condition,NULL));
    s->wake=eventfd(0,EFD_CLOEXEC|EFD_NONBLOCK);CHECK(s->wake>=0);s->service.executor="cpu";s->service.prefill_rows=64;
    CHECK(workers && workers<=2);s->worker_count=workers;
    for (unsigned i=0;i<workers;++i) {
        worker *w=&s->workers[i];w->owner=s;w->index=i;w->arena_bytes=WORKER_BYTES;w->arena=calloc(1,WORKER_BYTES);CHECK(w->arena);
        f->arguments[i]=(worker_argument){f,w};
    }
    ria_error e={0};
    for (unsigned expert=0;expert<2;++expert) {
        ria_expert_matrix matrices[3];
        for (unsigned projection=0;projection<3;++projection) {
            float value=projection==2 ? (float)(expert+1) : 1;
            uint32_t bits;memcpy(&bits,&value,4);ria_write_u16(f->weights[expert][projection],(uint16_t)(bits>>16));
            matrices[projection]=(ria_expert_matrix){.profile=RIA_EXPERT_BF16,.out_features=1,.in_features=1,
                .values=f->weights[expert][projection],.values_bytes=2,.value_row_stride=2};
        }
        f->experts[expert]=(ria_expert){matrices[0],matrices[1],matrices[2],10};CHECK(ria_expert_validate(&f->experts[expert],&e));
    }
    work_create(&s->requests[0],0,64);work_create(&s->requests[1],1,64);
    /* Interleave owners/experts; group gather may skip unrelated work but
     * never merge it or alter its original response entry. */
    for (unsigned row=0;row<64;++row) for (unsigned owner=0;owner<2;++owner) for (unsigned slot=0;slot<2;++slot)
        CHECK(ria_server_queue_push(&s->queue,(ria_server_job){&s->requests[owner],0,row,row*2+slot},&e));
    for (unsigned i=0;i<workers;++i) CHECK(!pthread_create(&s->workers[i].thread,NULL,run,&f->arguments[i]));
    return f;
}
static void finish(fixture *f) {
    server *s=&f->owner;CHECK(!pthread_mutex_lock(&s->mutex));
    wait_locked(f,&s->requests[0].completed);wait_locked(f,&s->requests[1].completed);
    CHECK(!s->queue.count && !s->requests[0].remaining && !s->requests[1].remaining);
    s->stopping=true;CHECK(!pthread_cond_broadcast(&s->condition));CHECK(!pthread_mutex_unlock(&s->mutex));
    for (unsigned i=0;i<s->worker_count;++i) { CHECK(!pthread_join(s->workers[i].thread,NULL));CHECK(s->workers[i].exited); }
    CHECK(!s->worker_error.code);
}
static void discard(fixture *f) {
    server *s=&f->owner;
    for (unsigned i=0;i<s->worker_count;++i) { ria_expert_cpu_destroy(s->workers[i].cpu);free(s->workers[i].arena); }
    for (unsigned i=0;i<2;++i) { free(s->requests[i].input);free(s->requests[i].output); }
    CHECK(!close(s->wake));CHECK(!pthread_cond_destroy(&s->condition));CHECK(!pthread_mutex_destroy(&s->mutex));free(f);
}
static void parity(unsigned workers) {
    fixture *f=create(false,false,workers);finish(f);CHECK(f->evaluated_rows==256 && f->largest>1 && f->batches<256);
    for (unsigned i=0;i<workers;++i)
        CHECK(f->largest<=f->owner.workers[i].batch_rows && f->owner.workers[i].batch_rows<64);
    for (unsigned owner=0;owner<2;++owner) {
        const work *r=&f->owner.requests[owner];CHECK(!r->error.code && !r->cancelled);
        for (unsigned row=0;row<64;++row) for (unsigned slot=0;slot<2;++slot) {
            unsigned entry=row*2+slot;
            float input=ria_read_f32(r->parsed.inputs+row*4),coefficient=ria_read_f32(r->parsed.entries+entry*8+4);
            CHECK(ria_read_f32(r->output+32+entry*4)==oracle(input,coefficient,slot));
        }
        for (unsigned byte=0;byte<32;++byte) CHECK(r->output[byte]==0xa5);
    }
    discard(f);
}
static void cancelled_and_failed(bool fail) {
    fixture *f=create(true,fail,1);server *s=&f->owner;
    CHECK(!pthread_mutex_lock(&s->mutex));wait_locked(f,&f->entered);
    /* The complete first subgroup remains outstanding while the actual
     * executor owns borrowed request input and separate response slots. */
    CHECK(s->requests[0].remaining==128 && !s->requests[0].completed && s->requests[0].input && s->requests[0].output);
    if (!fail) s->requests[0].cancelled=true;
    f->released=true;CHECK(!pthread_cond_broadcast(&s->condition));CHECK(!pthread_mutex_unlock(&s->mutex));finish(f);
    CHECK(f->evaluated_rows<256);
    if (fail) CHECK(s->requests[0].error.code==RIA_EXECUTOR_ERROR && s->requests[1].error.code==RIA_EXECUTOR_ERROR);
    else CHECK(s->requests[0].cancelled && !s->requests[1].cancelled && !s->requests[1].error.code);
    CHECK(s->requests[0].input && s->requests[0].output);discard(f);
}
int main(void) {
    parity(1);parity(2);cancelled_and_failed(false);cancelled_and_failed(true);
    puts("production grouped CPU worker: 64 rows, owner/expert isolation, exact scatter, bounded arena, cancellation/failure quiescence: passed (synthetic, no NUMA/GPU probes)");return 0;
}
