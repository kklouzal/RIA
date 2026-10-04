#include "ria/graph.h"
#include "ria/graph_cuda.h"
#include "ria/vision.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Host contracts only. Linking with function-section collection drops all
 * graph execution functions, so this fixture cannot initialize CUDA. */
static uint32_t token_map[RIA_GRAPH_VOCAB];
typedef struct { uint32_t calls,last_count;bool fail;uint64_t ids[24]; } row_fixture;
static bool fixture_rows(void *context,uint32_t layer,const uint64_t *ids,uint32_t count,uint8_t *packed,ria_error *e) {
    row_fixture *f=context;++f->calls;f->last_count=count;memcpy(f->ids,ids,count*sizeof(*ids));
    if (f->fail) return ria_fail(e,RIA_EXECUTOR_ERROR,"synthetic row failure");
    for (uint32_t row=0;row<count;++row) for (unsigned byte=0;byte<264;++byte) packed[(uint64_t)row*264+byte]=(uint8_t)(ids[row]+layer+byte);
    return true;
}
static void placement_and_cache(void) {
    ria_error e={0};ria_graph_options o={0};uint64_t length=UINT64_C(5120)*2304*2;
    uint8_t *zero=calloc(1,(size_t)length);assert(zero);
    ria_expert_matrix gate={RIA_EXPERT_BF16,2304,5120,zero,length,10240,NULL,0,0,1,1};
    ria_expert_matrix down=gate;down.in_features=2304;down.out_features=5120;down.value_row_stride=4608;
    ria_graph_local_expert entries[3]={{2,3,RIA_GRAPH_HOST,7,{gate,gate,down,10}},
                                     {2,9,RIA_GRAPH_VRAM,2,{gate,gate,down,10}},
                                     {5,1,RIA_GRAPH_HOST,4,{gate,gate,down,10}}};
    o.local_experts=entries;o.local_expert_count=3;o.host_expert_budget=length*9;o.device_expert_budget=length*3;
    uint64_t host,device;assert(ria_graph_local_experts_validate(&o,&host,&device,&e));assert(host==length*9 && device==length*3);
    uint16_t ids[6]={9,17,3,6,383,0};int32_t local[6];uint8_t remote[6];uint32_t count;
    assert(ria_graph_partition(&o,2,RIA_GRAPH_DECODE,ids,6,local,remote,&count,&e));assert(count==4 && local[0]==1 && local[2]==0);
    assert(remote[0]==1 && remote[1]==3 && remote[2]==4 && remote[3]==5);
    assert(ria_graph_partition(&o,2,RIA_GRAPH_PREFILL,ids,6,local,remote,&count,&e));assert(count==5 && local[0]==-1 && local[2]==0);
    assert(ria_graph_partition(&o,3,RIA_GRAPH_DECODE,ids,6,local,remote,&count,&e));assert(count==6);
    ids[5]=9;assert(!ria_graph_partition(&o,2,RIA_GRAPH_DECODE,ids,6,local,remote,&count,&e));ids[5]=0;
    o.device_expert_budget--;assert(!ria_graph_local_experts_validate(&o,&host,&device,&e));o.device_expert_budget++;
    entries[1].expert_id=3;assert(!ria_graph_local_experts_validate(&o,&host,&device,&e));entries[1].expert_id=9;
    entries[1].phase_mask=0;assert(!ria_graph_local_experts_validate(&o,&host,&device,&e));entries[1].phase_mask=2;free(zero);
    ria_graph_row_cache_entry cache[2]={0},before[2];row_fixture f={0};ria_graph_remote callback={.context=&f,.engram=fixture_rows};
    uint64_t rows[4]={7,4,7,9};uint8_t packed[4*264];assert(ria_graph_rows_fetch(callback,1,rows,4,cache,2,packed,&e));assert(f.last_count==4);
    assert(ria_graph_rows_fetch(callback,1,rows,4,cache,2,packed,&e));assert(f.calls==2 && f.last_count==2 && f.ids[0]==7 && f.ids[1]==7);
    for (unsigned row=0;row<4;++row) for (unsigned byte=0;byte<264;++byte) assert(packed[row*264+byte]==(uint8_t)(rows[row]+1+byte));
    assert(ria_graph_rows_fetch(callback,14,rows,1,cache,2,packed,&e));assert(f.last_count==1);
    memcpy(before,cache,sizeof(cache));f.fail=true;rows[0]=99;assert(!ria_graph_rows_fetch(callback,1,rows,1,cache,2,packed,&e));assert(!memcmp(before,cache,sizeof(cache)));
    rows[0]=384006168;assert(!ria_graph_rows_fetch(callback,1,rows,1,cache,2,packed,&e));
    assert(!ria_graph_rows_fetch(callback,2,rows,1,cache,2,packed,&e));
}
static ria_graph_options options(void) {
    ria_graph_options o={0};o.device=0;o.gpu_uuid="GPU-00112233-4455-6677-8899-aabbccddeeff";
    o.max_tokens=257;o.host_state_budget=UINT64_C(1)<<30;o.device_budget=UINT64_C(1)<<30;o.pinned_budget=UINT64_C(1)<<26;
    o.projection_tile_rows=64;o.state_tile_rows=32;o.max_image_patches=9216;o.prefill_rows=1;
    o.compressed_token_map=token_map;o.compressed_vocab=99092;o.pad_compressed_id=2;
    const uint64_t multipliers[2][4]={{76632096046245,4839876093313,35959672319349,73987337458391},
                                    {67716810739261,51510806800915,30921347202721,82619226485591}};
    memcpy(o.hash_multipliers,multipliers,sizeof(multipliers));
    /* Shape-valid synthetic buckets deliberately differ from publisher prime
     * constants; the independent Python fixture evaluates these exact inputs. */
    for (unsigned l=0;l<2;++l) { for (unsigned i=0;i<23;++i) o.hash_primes[l][i]=16000001;
        o.hash_primes[l][23]=(l ? UINT64_C(384016682) : UINT64_C(384006168))-23*UINT64_C(16000001); }
    return o;
}
static void report(void) {
    ria_graph_options o=options();ria_error e={0};
    const int64_t histories[6][4]={{7,6,5,4},{7,-1,5,4},{7,6,-1,4},{7,6,5,-1},{0,0,0,0},{99091,99090,99089,99088}};
    printf("{\"hashes\":[");
    for (unsigned l=0;l<2;++l) for (unsigned h=0;h<6;++h) {
        uint64_t ids[24];assert(ria_graph_hash(&o,l,histories[h],ids,&e));if (l || h) putchar(',');putchar('[');
        for (unsigned j=0;j<24;++j) { if (j) putchar(',');printf("%llu",(unsigned long long)ids[j]); }putchar(']');
    }
    printf("],\"dependencies\":[");
    for (uint32_t l=0;l<40;++l) { uint32_t kv,index,ratio;assert(ria_graph_dependencies(l,&kv,&index,&ratio,&e));
        if (l) putchar(',');
        printf("[%u,%u,%u]",kv,index,ratio); }
    printf("],\"modes\":[");
    for (uint32_t l=0;l<40;++l) { ria_graph_attention_kind kind;assert(ria_graph_attention_kind_for_layer(l,&kind,&e));if (l) putchar(',');printf("%u",(unsigned)kind); }
    printf("],\"grids\":[");const uint32_t shapes[][2]={{1,1},{800,600},{600,800},{16384,1},{1,16384},{544,544},{1024,1024},{1001,777}};
    for (unsigned i=0;i<sizeof(shapes)/sizeof(shapes[0]);++i) { ria_image_grid g;assert(ria_image_plan(shapes[i][0],shapes[i][1],9216,&g,&e));
        if (i) putchar(',');
        printf("[%u,%u,%u,%u,%u,%u,%u]",g.pixel_height,g.pixel_width,g.vit_height,g.vit_width,g.llm_height,g.llm_width,g.token_count); }
    puts("]}");
}
static void write_patches(const ria_image_input *input,const char *path) {
    FILE *f=fopen(path,"wb");assert(f);uint64_t count=(uint64_t)input->grid.vit_height*input->grid.vit_width*588;
    assert(fwrite(input->patches,sizeof(float),(size_t)count,f)==count);assert(fclose(f)==0);
    const ria_image_grid *g=&input->grid;
    printf("[%u,%u,%u,%u,%u,%u,%u]\n",g->pixel_height,g->pixel_width,g->vit_height,g->vit_width,g->llm_height,g->llm_width,g->token_count);
}
int main(int argc,char **argv) {
    ria_error e={0};ria_graph_options o=options();assert(ria_graph_options_validate(&o,&e));
    if (argc==2 && !strcmp(argv[1],"--report")) { report();return 0; }
    if (argc==4 && !strcmp(argv[1],"--image")) {
        FILE *f=fopen(argv[2],"rb");assert(f);assert(fseek(f,0,SEEK_END)==0);long n=ftell(f);assert(n>0 && n<=67108864);rewind(f);
        uint8_t *bytes=malloc((size_t)n);assert(bytes);assert(fread(bytes,1,(size_t)n,f)==(size_t)n);assert(fclose(f)==0);
        ria_image_input input;bool ok=ria_image_prepare(bytes,(uint64_t)n,9216,UINT64_C(1)<<30,&input,&e);free(bytes);
        if (!ok) { fprintf(stderr,"%d: %s\n",e.code,e.message);return 1; }
        write_patches(&input,argv[3]);ria_image_input_free(&input);return 0;
    }
    assert(argc==1);uint32_t kv,index,ratio;
    placement_and_cache();
    uint64_t pinned;assert(ria_graph_pinned_required_bytes(&o,&pinned,&e));assert(pinned==UINT64_C(13107200));
    ria_graph_options no_image=o;no_image.max_image_patches=0;assert(ria_graph_pinned_required_bytes(&no_image,&pinned,&e));assert(pinned==UINT64_C(8388608));
    no_image.pinned_budget=pinned-1;assert(!ria_graph_options_validate(&no_image,&e));
    assert(ria_graph_client_device_validate(1,0,12,0,"NVIDIA GeForce RTX 5090",o.gpu_uuid,o.gpu_uuid,&e));
    assert(!ria_graph_client_device_validate(2,0,12,0,"NVIDIA GeForce RTX 5090",o.gpu_uuid,o.gpu_uuid,&e));
    assert(!ria_graph_client_device_validate(1,1,12,0,"NVIDIA GeForce RTX 5090",o.gpu_uuid,o.gpu_uuid,&e));
    assert(!ria_graph_client_device_validate(1,0,12,1,"NVIDIA GeForce RTX 5090",o.gpu_uuid,o.gpu_uuid,&e));
    assert(!ria_graph_client_device_validate(1,0,12,0,"NVIDIA GeForce RTX 5090 D",o.gpu_uuid,o.gpu_uuid,&e));
    assert(!ria_graph_client_device_validate(1,0,12,0,NULL,o.gpu_uuid,o.gpu_uuid,&e));
    assert(!ria_graph_client_device_validate(1,0,12,0,"NVIDIA GeForce RTX 5090","GPU-other",o.gpu_uuid,&e));
    assert(!ria_graph_dependencies(40,&kv,&index,&ratio,&e));assert(!ria_graph_dependencies(0,NULL,&index,&ratio,&e));
    uint64_t bytes1,bytes2;assert(ria_graph_host_state_required(1,&bytes1,&e));assert(ria_graph_host_state_required(1048576,&bytes2,&e));
    long page=sysconf(_SC_PAGESIZE);assert(page>0 && bytes1%(uint64_t)page==0 && bytes2>bytes1);
    assert(!ria_graph_host_state_required(0,&bytes1,&e));assert(!ria_graph_host_state_required(UINT64_MAX,&bytes1,&e));
    assert(!ria_graph_host_state_required(1,NULL,&e));
    ria_graph_options bad=o;bad.gpu_uuid="GPU-00112233-4455-6677-8899-aabbccddeefZ";assert(!ria_graph_options_validate(&bad,&e));
    bad=o;bad.max_tokens=1048577;assert(!ria_graph_options_validate(&bad,&e));bad=o;bad.hash_multipliers[1][2]=2;assert(!ria_graph_options_validate(&bad,&e));
    bad=o;bad.prefill_rows=0;assert(!ria_graph_options_validate(&bad,&e));bad.prefill_rows=65;assert(!ria_graph_options_validate(&bad,&e));
    bad=o;bad.max_tokens=1;bad.prefill_rows=2;assert(!ria_graph_options_validate(&bad,&e));
    bad=o;bad.hash_primes[1][23]++;assert(!ria_graph_options_validate(&bad,&e));bad=o;bad.hash_multipliers[0][0]=UINT64_MAX;assert(!ria_graph_options_validate(&bad,&e));
    token_map[129279]=99092;assert(!ria_graph_options_validate(&o,&e));token_map[129279]=0;
    int64_t history[4]={99092,0,0,0};uint64_t ids[24];assert(!ria_graph_hash(&o,0,history,ids,&e));
    history[0]=0;assert(!ria_graph_hash(&o,2,history,ids,&e));bad=o;bad.hash_primes[0][0]=0;assert(!ria_graph_hash(&bad,0,history,ids,&e));
    ria_image_grid grid;assert(ria_image_plan(1,1,9216,&grid,&e));assert(grid.pixel_height==546 && grid.pixel_width==546 && grid.token_count==184);
    assert(!ria_image_plan(0,1,9216,&grid,&e));assert(!ria_image_plan(16384,16384,9216,&grid,&e));assert(!ria_image_plan(1,1,1,&grid,&e));
    ria_image_input image;uint8_t rgb[3]={0,127,255};assert(!ria_image_prepare_rgb(rgb,2,1,1,9216,UINT64_C(1)<<30,&image,&e));
    assert(!ria_image_prepare_rgb(rgb,3,1,1,9216,1,&image,&e));assert(ria_image_prepare_rgb(rgb,3,1,1,9216,UINT64_C(1)<<30,&image,&e));
    assert(image.types[0]==0 && image.types[image.grid.token_count-1]==3);
    for (uint32_t row=0;row<image.grid.llm_height;++row) { uint32_t begin=1+row*(image.grid.llm_width+1);assert(image.types[begin+image.grid.llm_width]==2); }
    assert(image.patches[0]==-1 && image.patches[392]==1);ria_image_input_free(&image);ria_image_input_free(&image);
    uint8_t invalid[9]={137,'P','N','G',13,10,26,10,0};assert(!ria_image_prepare(invalid,9,9216,UINT64_C(1)<<30,&image,&e));
    assert(!ria_image_prepare(invalid,UINT64_MAX,9216,UINT64_C(1)<<30,&image,&e));
    puts("graph host contracts: passed (no CUDA execution)");return 0;
}
