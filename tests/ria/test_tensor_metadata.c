/* Actual loader boundary: no tensor payload, model, NUMA operation or GPU. */
#define _GNU_SOURCE
#include "ria/tensor.c"

#define CHECK(condition) do { if (!(condition)) { fprintf(stderr,"metadata contract line %d: %s\n",__LINE__,#condition); return 1; } } while (0)

int main(void) {
  const size_t lengths[]={128,16384,262144};
  const ria_json_limits limits={262144,8192,32};
  const uint64_t retained=12345;
  for (unsigned test=0;test<sizeof(lengths)/sizeof(lengths[0]);test++) {
    size_t length=lengths[test];
    char *input=malloc(length);CHECK(input);
    memcpy(input,"{\"k\":\"",6);memset(input+6,'x',length-8);memcpy(input+length-2,"\"}",2);
    ria_error error={0};ria_json_doc d={0};uint32_t capacity;
    uint64_t owned,parse_keys,parse_peak,canonical_bytes,canonical_keys,canonical_peak;
    CHECK(ria_json_parse_required_bytes(length,limits,&capacity,&owned,&error));
    CHECK(ria_json_keys_required_bytes(capacity,&parse_keys,&error));
    parse_peak=retained+owned+parse_keys+length;
    CHECK(!metadata_peak(retained,length,limits,true,parse_peak-1,&error));
    CHECK(error.code==RIA_RESOURCE_LIMIT);
    error=(ria_error){0};
    CHECK(metadata_peak(retained,length,limits,true,parse_peak,&error));
    CHECK(metadata_peak(retained,length,limits,false,parse_peak-length,&error));
    CHECK(!metadata_peak(retained,length,limits,false,parse_peak-length-1,&error));
    error=(ria_error){0};
    CHECK(ria_json_parse(input,length,limits,&d,&error));
    CHECK(d.allocated_bytes==owned);
    CHECK(ria_json_canonical_required_bytes(d.string_bytes,d.count,&canonical_bytes,&error));
    CHECK(ria_json_keys_required_bytes((uint64_t)d.count+limits.max_depth,&canonical_keys,&error));
    canonical_peak=retained+owned+canonical_bytes+canonical_keys;
    CHECK(!metadata_canonical_peak(&d,retained,limits,canonical_peak-1,&error));
    error=(ria_error){0};
    CHECK(metadata_canonical_peak(&d,retained,limits,canonical_peak,&error));
    ria_json_free(&d);

    FILE *file=tmpfile();CHECK(file);
    CHECK(fwrite(input,1,length,file)==length && fflush(file)==0 && fseek(file,0,SEEK_SET)==0);
    uint64_t peak=parse_peak>canonical_peak ? parse_peak : canonical_peak;
    CHECK(read_document(dup(fileno(file)),&d,limits,retained,peak,&error));
    uint8_t hash[32];CHECK(ria_json_sha256(&d,true,hash,&error));ria_json_free(&d);
    CHECK(fseek(file,0,SEEK_SET)==0);
    error=(ria_error){0};
    /* Large strings make the canonical allocation larger than input/DOM.
     * Former accounting returned this document and then exceeded its cap. */
    if (length==262144) {
      CHECK(canonical_peak>parse_peak);
      CHECK(!read_document(dup(fileno(file)),&d,limits,retained,parse_peak,&error));
      CHECK(error.code==RIA_RESOURCE_LIMIT && !d.nodes && !d.strings && !d.allocated_bytes);
    }
    CHECK(fclose(file)==0);free(input);
  }
  puts("tensor metadata input/DOM/duplicate/canonical peaks passed");
  return 0;
}
