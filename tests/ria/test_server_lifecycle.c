#define _GNU_SOURCE
/* Exercise the production network owner without weights or executor startup. */
#include "ria/server.c"
#include <arpa/inet.h>
#include <openssl/pem.h>
#include <openssl/x509v3.h>
#include <sys/stat.h>
#include <sys/wait.h>

static pid_t fixture_child=-1;
#define CHECK(x) do { if (!(x)) { \
  fprintf(stderr,"lifecycle failure %s:%d: %s\n",__FILE__,__LINE__,#x); \
  if (fixture_child>0) { kill(fixture_child,SIGKILL); waitpid(fixture_child,NULL,0); } \
  exit(1); } } while (0)
static ria_json_doc doc(const char *text) {
  ria_json_doc d={0}; ria_error e={0}; ria_json_limits l={65536,8192,32};
  CHECK(ria_json_parse(text,strlen(text),l,&d,&e)); return d;
}
static void cert_extension(X509 *cert,X509 *issuer,int nid,const char *value) {
  X509V3_CTX c; X509V3_set_ctx(&c,issuer,cert,NULL,NULL,0);
  X509_EXTENSION *x=X509V3_EXT_conf_nid(NULL,&c,nid,(char *)value);
  CHECK(x && X509_add_ext(cert,x,-1)==1); X509_EXTENSION_free(x);
}
static X509 *cert(EVP_PKEY *key,X509 *issuer,const char *name,long serial) {
  X509 *x=X509_new(); CHECK(x && X509_set_version(x,2)==1 &&
    ASN1_INTEGER_set(X509_get_serialNumber(x),serial)==1 &&
    X509_gmtime_adj(X509_getm_notBefore(x),-60) && X509_gmtime_adj(X509_getm_notAfter(x),3600) &&
    X509_set_pubkey(x,key)==1);
  X509_NAME *n=X509_get_subject_name(x);
  CHECK(X509_NAME_add_entry_by_txt(n,"CN",MBSTRING_ASC,(const unsigned char *)name,-1,-1,0)==1 &&
    X509_set_issuer_name(x,issuer ? X509_get_subject_name(issuer) : n)==1);
  cert_extension(x,issuer?issuer:x,NID_basic_constraints,issuer?"critical,CA:FALSE":"critical,CA:TRUE");
  cert_extension(x,issuer?issuer:x,NID_key_usage,issuer?"critical,digitalSignature":"critical,keyCertSign,cRLSign");
  if (issuer) {
    char san[128]; CHECK(snprintf(san,sizeof(san),"DNS:%s",name)>0);
    cert_extension(x,issuer,NID_subject_alt_name,san);
    cert_extension(x,issuer,NID_ext_key_usage,"serverAuth,clientAuth");
  }
  CHECK(X509_sign(x,key,EVP_sha256())>0); return x;
}
static void write_pem(const char *path,EVP_PKEY *key,X509 *x) {
  FILE *f=fopen(path,"wb"); CHECK(f && fchmod(fileno(f),0600)==0);
  CHECK(key ? PEM_write_PrivateKey(f,key,NULL,NULL,0,NULL,NULL)==1 : PEM_write_X509(f,x)==1);
  CHECK(fclose(f)==0);
}
typedef struct { char directory[128],paths[4][256]; ria_tls client,tls; } credentials;
static void credentials_create(credentials *c) {
  strcpy(c->directory,"/tmp/ria-server-lifecycle-XXXXXX"); CHECK(mkdtemp(c->directory));
  const char *names[]={"ca.pem","key.pem","server.pem","client.pem"};
  for (unsigned i=0;i<4;i++) CHECK(snprintf(c->paths[i],256,"%s/%s",c->directory,names[i])>0);
  EVP_PKEY *key=EVP_PKEY_Q_keygen(NULL,NULL,"EC","prime256v1"); CHECK(key);
  X509 *ca=cert(key,NULL,"fixture root",1),*a=cert(key,ca,"server.test",2),*b=cert(key,ca,"client.test",3);
  write_pem(c->paths[0],NULL,ca); write_pem(c->paths[1],key,NULL);
  write_pem(c->paths[2],NULL,a); write_pem(c->paths[3],NULL,b);
  ria_tls_config sc={c->paths[0],c->paths[2],c->paths[1],"client.test",true,150},
    cc={c->paths[0],c->paths[3],c->paths[1],"server.test",false,1000}; ria_error e={0};
  CHECK(ria_tls_create(&c->tls,&sc,&e) && ria_tls_create(&c->client,&cc,&e));
  X509_free(ca); X509_free(a); X509_free(b); EVP_PKEY_free(key);
}
static void credentials_free(credentials *c) {
  ria_tls_destroy(&c->tls); ria_tls_destroy(&c->client);
  for (unsigned i=0;i<4;i++) CHECK(unlink(c->paths[i])==0);
  CHECK(rmdir(c->directory)==0);
}
static int listener(struct sockaddr_in *a) {
  int fd=socket(AF_INET,SOCK_STREAM|SOCK_CLOEXEC,0); CHECK(fd>=0);
  memset(a,0,sizeof(*a)); a->sin_family=AF_INET; a->sin_addr.s_addr=htonl(INADDR_LOOPBACK);
  CHECK(bind(fd,(struct sockaddr *)a,sizeof(*a))==0 && listen(fd,4)==0);
  socklen_t n=sizeof(*a); CHECK(getsockname(fd,(struct sockaddr *)a,&n)==0); return fd;
}
#define DIGEST "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
static server *fixture(credentials *c,struct sockaddr_in a[2]) {
  server *s=calloc(1,sizeof(*s)); CHECK(s);
  CHECK(pthread_mutex_init(&s->mutex,NULL)==0 && pthread_cond_init(&s->condition,NULL)==0);
  s->wake=eventfd(0,EFD_CLOEXEC|EFD_NONBLOCK); CHECK(s->wake>=0);
  sigset_t mask; sigemptyset(&mask); sigaddset(&mask,SIGTERM);
  CHECK(pthread_sigmask(SIG_BLOCK,&mask,NULL)==0);
  s->signals=signalfd(-1,&mask,SFD_CLOEXEC|SFD_NONBLOCK); CHECK(s->signals>=0);
  for (unsigned i=0;i<2;i++) { s->listeners[i]=listener(&a[i]); s->channels[i].transport.fd=-1; }
  s->tls=c->tls; s->service.tls.expected_peer_name="client.test";
  s->service.tls.handshake_timeout_ms=150; s->service.expert.drain_timeout_ms=120;
  s->service.limits=(ria_limits){65536,4096,1,1,1,33554432,200,100,100};
  s->service.executor="cpu"; s->ready=true; s->quiescent=true;
  s->service.expert.policy=RIA_NUMA_SHARDED; s->service.expert.node_count=1;
  s->bank.operations[0]=(ria_operation){1,1,1,1,0,1,1,0,1,false,false,false};
  s->network_arena=calloc(2,65536); CHECK(s->network_arena);
  s->channels[0].input=s->network_arena; s->channels[1].input=(uint8_t *)s->network_arena+65536;
  s->store.manifest=doc("{\"layout_digest\":\"" DIGEST "\"}");
  memset(s->store.logical_model_digest,0xaa,32); memset(s->store.operator_contract_digest,0xaa,32);
  s->service.grants=doc("{\"grants\":[{\"expected_peer_name\":\"client.test\","
    "\"logical_model_digest\":\"" DIGEST "\",\"operator_contract_digest\":\"" DIGEST "\","
    "\"encoding_digest\":\"" DIGEST "\",\"client_layout_digest\":\"" DIGEST "\","
    "\"placement_plan_digest\":\"" DIGEST "\",\"server_layout_digest\":\"" DIGEST "\","
    "\"profile\":\"bf16\",\"server_executor\":\"cpu\"}]}");
  ria_error e={0}; CHECK(server_minimum(s,&e));
  ria_binding_init(&s->binding,&s->service.limits); return s;
}
static void fixture_free(server *s) {
  invalidate(s);
  for (unsigned i=0;i<2;i++) { close(s->listeners[i]); free(s->requests[i].input); free(s->requests[i].output); }
  close(s->wake); close(s->signals); free(s->network_arena);
  ria_json_free(&s->store.manifest); ria_json_free(&s->service.grants);
  pthread_cond_destroy(&s->condition); pthread_mutex_destroy(&s->mutex); free(s);
}
static void launch(server *s) {
  fixture_child=fork(); CHECK(fixture_child>=0);
  if (!fixture_child) {
    ria_error e={0}; bool ok=event_loop(s,&e);
    if (!ok) fprintf(stderr,"production event loop: %s\n",e.message);
    fixture_free(s); _exit(ok?0:1);
  }
}
static int reap(bool stop,uint64_t duration) {
  if (stop) CHECK(kill(fixture_child,SIGTERM)==0);
  uint64_t deadline=ria_monotonic_ms()+duration; int status=0; pid_t result;
  do { result=waitpid(fixture_child,&status,WNOHANG); CHECK(result>=0); if (result) break;
    CHECK(poll(NULL,0,5)==0); } while (ria_monotonic_ms()<deadline);
  CHECK(result==fixture_child); fixture_child=-1; CHECK(WIFEXITED(status)); return WEXITSTATUS(status);
}
static ria_transport connect_client(credentials *c,const struct sockaddr_in *a) {
  ria_transport t={.fd=-1}; ria_error e={0};
  CHECK(ria_transport_connect(&t,&c->client,(const struct sockaddr *)a,sizeof(*a),1000,&e)); return t;
}
static void exchange(ria_transport *t,ria_header *h,const char *text,ria_header *response,ria_json_doc *out) {
  ria_json_doc d=doc(text); char *canonical=NULL; size_t length=0; ria_error e={0}; uint8_t *p=NULL;
  CHECK(ria_json_canonical(&d,false,&canonical,&length,&e)); ria_json_free(&d); h->payload_length=length;
  CHECK(ria_transport_send(t,h,canonical,ria_monotonic_ms()+1000,&e)); free(canonical);
  CHECK(ria_transport_frame(t,ria_monotonic_ms()+1000,100,RIA_CONTROL_MAX,response,&p,&e));
  CHECK(response->kind==h->kind && response->request_id==h->request_id && response->flags==1 && !response->status);
  if (out) { ria_json_limits l={65536,8192,32}; CHECK(ria_json_parse(p,(size_t)response->payload_length,l,out,&e)); }
  free(p);
}
static const char bind_request[]="{\"role\":\"client\",\"logical_model_digest\":\"" DIGEST "\","
    "\"operator_contract_digest\":\"" DIGEST "\",\"encoding_digest\":\"" DIGEST "\","
    "\"client_layout_digest\":\"" DIGEST "\",\"placement_plan_digest\":\"" DIGEST "\","
    "\"profile\":\"bf16\",\"server_executor\":\"cpu\",\"limits\":{\"frame_payload_bytes\":65536,"
    "\"bulk_data_bytes\":4096,\"expert_rows\":1,\"expert_requests\":1,\"row_lookup_rows\":1,"
    "\"inflight_payload_bytes\":33554432,\"operation_timeout_ms\":200,\"frame_io_timeout_ms\":100,\"write_timeout_ms\":100}}";
static ria_json_doc bind_control(ria_transport *t,ria_header *response) {
  ria_header h={.kind=RIA_BIND,.request_id=1}; ria_json_doc out={0};
  exchange(t,&h,bind_request,response,&out); return out;
}
static void expect_closed(ria_transport *t) {
  ria_error e={0}; uint8_t byte; uint64_t started=ria_monotonic_ms();
  CHECK(!ria_transport_read(t,&byte,1,started+1000,&e));
  CHECK(e.code!=RIA_DEADLINE_EXCEEDED && ria_monotonic_ms()-started<800);
  ria_transport_close(t);
}
static void test_initial_silence(credentials *c,bool control_bind) {
  struct sockaddr_in a[2]; server *s=fixture(c,a); launch(s);
  ria_transport control=connect_client(c,&a[0]);
  if (control_bind) { ria_header h; ria_json_doc d=bind_control(&control,&h); ria_json_free(&d); }
  expect_closed(&control); CHECK(reap(true,1000)==0); fixture_free(s);
}
static void bound_pair(credentials *c,const struct sockaddr_in a[2],ria_transport pair[2],ria_header *identity) {
  pair[0]=connect_client(c,&a[0]); ria_header response;
  ria_json_doc d=bind_control(&pair[0],&response); const char *session,*cap; ria_error e={0};
  CHECK(ria_json_string(&d,ria_json_get(&d,0,"session_id"),&session,NULL,&e) &&
    ria_json_string(&d,ria_json_get(&d,0,"bulk_capability"),&cap,NULL,&e));
  char text[1024]; CHECK(snprintf(text,sizeof(text),"{\"session_id\":\"%s\",\"epoch\":\"%llu\","
    "\"bulk_capability\":\"%s\",\"logical_model_digest\":\"" DIGEST "\",\"operator_contract_digest\":\"" DIGEST "\"}",
    session,(unsigned long long)response.epoch,cap)>0);
  pair[1]=connect_client(c,&a[1]); ria_header h=response; h.kind=RIA_BIND_BULK; h.flags=0; h.request_id=1;
  exchange(&pair[1],&h,text,&response,NULL); ria_json_free(&d); *identity=h;
}
static void test_bound_idle(credentials *c) {
  struct sockaddr_in a[2]; server *s=fixture(c,a); launch(s);
  ria_transport pair[2]; ria_header h,response; ria_error e={0}; bound_pair(c,a,pair,&h);
  CHECK(poll(NULL,0,500)==0); /* Idle pair outlives its initial setup budget. */
  h.kind=RIA_HEALTH; h.request_id=2; exchange(&pair[0],&h,"{}",&response,NULL);
  h.kind=RIA_CLOSE; h.request_id=3; h.payload_length=0;
  CHECK(ria_transport_send(&pair[0],&h,NULL,ria_monotonic_ms()+1000,&e));
  uint8_t *p=NULL; CHECK(ria_transport_frame(&pair[0],ria_monotonic_ms()+1000,100,65536,&response,&p,&e)); free(p);
  CHECK(response.kind==RIA_CLOSE && response.request_id==3 && !response.status);
  ria_transport_close(&pair[0]); ria_transport_close(&pair[1]);
  CHECK(reap(true,1000)==0); fixture_free(s);
}
static void test_late_completion(credentials *c) {
  struct sockaddr_in a[2]; server *s=fixture(c,a); ria_error e={0};
  CHECK(ria_binding_fresh(&s->binding,&e));
  work *w=&s->requests[0]; w->used=w->completed=w->cancelled=true;
  w->header=(ria_header){.kind=RIA_EXPERT,.request_id=2,.epoch=s->binding.epoch};
  memcpy(w->header.session,s->binding.session,16); w->input=malloc(1); w->output=calloc(1,32);
  CHECK(w->input && w->output); w->parsed.response_bytes=32; w->deadline=ria_monotonic_ms()+1000;
  ria_binding_init(&s->binding,&s->service.limits); CHECK(ria_binding_fresh(&s->binding,&e));
  CHECK(completed(s,&e)); CHECK(!w->used && s->channels[0].output_count==0); fixture_free(s);
}
static void test_retained_accept(credentials *c) {
  struct sockaddr_in a[2]; server *s=fixture(c,a); s->requests[0].used=s->requests[0].completed=true;
  int fd=socket(AF_INET,SOCK_STREAM,0); CHECK(fd>=0 && connect(fd,(struct sockaddr *)&a[0],sizeof(a[0]))==0);
  ria_error e={0}; CHECK(accept_channel(s,0,&e)); char byte;
  CHECK(read(fd,&byte,1)==0 && s->channels[0].transport.fd<0); close(fd); fixture_free(s);
}
static void test_disconnected_live_executor(credentials *c) {
  struct sockaddr_in a[2]; server *s=fixture(c,a); ria_error e={0};
  CHECK(ria_binding_fresh(&s->binding,&e));
  /* A retained executor-owned request may not be freed or detached on loss. */
  s->requests[0].used=true; invalidate(s); bool ready=true,active=false;
  CHECK(admin_health(s,&ready,&active,&e) && !ready); launch(s);
  CHECK(reap(false,1000)==RIA_DEADLINE_EXCEEDED); fixture_free(s);
}
static void test_admitted_connection_loss(credentials *c,unsigned lost_channel) {
  struct sockaddr_in a[2]; server *s=fixture(c,a); launch(s);
  ria_transport pair[2]; ria_header h,response; ria_error e={0}; bound_pair(c,a,pair,&h);
  uint8_t expert[68]={0};
  ria_write_u64(expert,1); ria_write_u64(expert+8,1);
  ria_write_u32(expert+16,1); ria_write_u32(expert+20,1); ria_write_u32(expert+24,1); ria_write_u32(expert+28,1);
  ria_write_u64(expert+40,1); ria_write_u32(expert+52,1); ria_write_f32(expert+60,1); ria_write_f32(expert+64,1);
  h.kind=RIA_EXPERT; h.request_id=2; h.payload_length=sizeof(expert);
  CHECK(ria_transport_send(&pair[0],&h,expert,ria_monotonic_ms()+1000,&e));
  /* The following ordered Health proves the real parser/admission path ran.
   * No numeric worker is started, so its borrowed request remains live. */
  h.kind=RIA_HEALTH; h.request_id=3; exchange(&pair[0],&h,"{}",&response,NULL);
  ria_transport_close(&pair[lost_channel]);
  CHECK(reap(false,1000)==RIA_DEADLINE_EXCEEDED);
  ria_transport_close(&pair[1-lost_channel]); fixture_free(s);
}
static void test_started_header_deadline(credentials *c) {
  struct sockaddr_in a[2]; server *s=fixture(c,a); launch(s);
  ria_transport t=connect_client(c,&a[0]); ria_error e={0}; uint8_t first='D';
  CHECK(ria_transport_write(&t,&first,1,ria_monotonic_ms()+1000,&e));
  uint64_t start=ria_monotonic_ms(); expect_closed(&t);
  CHECK(ria_monotonic_ms()-start<250); CHECK(reap(true,1000)==0); fixture_free(s);
}
static void test_bind_failure(credentials *c,bool malformed) {
  struct sockaddr_in a[2]; server *s=fixture(c,a);
  if (!malformed) { ria_json_free(&s->service.grants); s->service.grants=doc("{\"grants\":[]}"); }
  launch(s); ria_transport t=connect_client(c,&a[0]); ria_json_doc d=doc(malformed ? "{}" : bind_request);
  char *p=NULL; size_t length=0; ria_error e={0};
  CHECK(ria_json_canonical(&d,false,&p,&length,&e)); ria_json_free(&d);
  ria_header h={.kind=RIA_BIND,.request_id=1,.payload_length=length},response;
  CHECK(ria_transport_send(&t,&h,p,ria_monotonic_ms()+1000,&e)); free(p); uint8_t *reply=NULL;
  CHECK(ria_transport_frame(&t,ria_monotonic_ms()+1000,100,65536,&response,&reply,&e));
  CHECK(response.kind==RIA_BIND && response.flags==1 && response.request_id==1 && !response.epoch &&
    response.status==(unsigned)(malformed ? RIA_INVALID_REQUEST : RIA_UNAUTHORIZED));
  for (unsigned i=0;i<16;i++) CHECK(!response.session[i]);
  CHECK(ria_control_json(reply,(size_t)response.payload_length,&response,&d,&e)); free(reply); ria_json_free(&d);
  expect_closed(&t); CHECK(reap(true,1000)==0); fixture_free(s);
}
static void test_control_credit_refusal(credentials *c) {
  struct sockaddr_in a[2]; server *s=fixture(c,a); launch(s);
  ria_transport pair[2]; ria_header h,response; ria_error e={0}; bound_pair(c,a,pair,&h);
  uint8_t batch[5*66]; h.kind=RIA_HEALTH; h.payload_length=2;
  for (unsigned i=0;i<5;i++) {
    h.request_id=2+i; CHECK(ria_header_encode(&h,batch+i*66,&e)); memcpy(batch+i*66+64,"{}",2);
  }
  /* One TLS write exposes every request to the real parser before publication. */
  CHECK(ria_transport_write(&pair[0],batch,sizeof(batch),ria_monotonic_ms()+1000,&e));
  for (unsigned i=0;i<5;i++) {
    uint8_t *p=NULL; ria_json_doc d={0};
    CHECK(ria_transport_frame(&pair[0],ria_monotonic_ms()+1000,100,65536,&response,&p,&e));
    CHECK(response.kind==RIA_HEALTH && response.request_id==2+i && response.flags==1 &&
      response.status==(unsigned)(i==4 ? RIA_RESOURCE_LIMIT : RIA_OK) && response.epoch==h.epoch &&
      !memcmp(response.session,h.session,16));
    CHECK(ria_control_json(p,(size_t)response.payload_length,&response,&d,&e)); free(p); ria_json_free(&d);
  }
  h.request_id=7; exchange(&pair[0],&h,"{}",&response,NULL);
  ria_transport_close(&pair[0]); ria_transport_close(&pair[1]); CHECK(reap(true,1000)==0); fixture_free(s);
}
static void test_refusal_accounting(credentials *c) {
  struct sockaddr_in a[2]; server *s=fixture(c,a); ria_error e={0}; uint64_t control,row,bulk;
  CHECK(ria_progress_charges(&s->service.limits,&control,&row,&bulk,&e) &&
    ria_binding_protect(&s->binding,control,row,bulk,&e) && ria_binding_fresh(&s->binding,&e));
  ria_header h={.kind=RIA_HEALTH,.request_id=2,.epoch=s->binding.epoch,.payload_length=2};
  memcpy(h.session,s->binding.session,16);
  for (unsigned i=0;i<4;i++) { h.request_id=2+i; CHECK(dispatch(s,0,&h,(uint8_t *)"{}",ria_monotonic_ms(),&e)); }
  uint64_t reserved=s->binding.reserved_bytes; h.request_id=6;
  CHECK(dispatch(s,0,&h,(uint8_t *)"{}",ria_monotonic_ms(),&e));
  CHECK(s->binding.reserved_bytes==reserved && s->binding.last_admitted[0]==5 && s->binding.last_request[0]==6 &&
    s->channels[0].output_count==5 && !s->channels[0].output[4].terminal && !s->binding_retiring);
  for (unsigned i=0;i<9;i++) CHECK(!s->binding.pending[i].used || s->binding.pending[i].id!=6);
  fixture_free(s);
}
static void client_configuration(credentials *c,const server *s,const struct sockaddr_in a[2],
                                 ria_service *client,ria_tensor_store *store,ria_remote_options *options,
                                 char control[64],char bulk[64]) {
  CHECK(snprintf(control,64,"127.0.0.1:%u",ntohs(a[0].sin_port))>0 &&
    snprintf(bulk,64,"127.0.0.1:%u",ntohs(a[1].sin_port))>0);
  memset(client,0,sizeof(*client)); client->role="client"; client->server_executor="cpu";
  client->control_address=control; client->bulk_address=bulk; client->connect_timeout_ms=1000;
  client->limits=s->service.limits;
  client->tls=(ria_tls_config){c->paths[0],c->paths[3],c->paths[1],"server.test",false,1000};
  memset(store,0,sizeof(*store)); strcpy(store->role,"client"); strcpy(store->profile,"bf16");
  memset(store->logical_model_digest,0xaa,32); memset(store->operator_contract_digest,0xaa,32); memset(store->encoding_digest,0xaa,32);
  memcpy(client->logical_model_digest,store->logical_model_digest,32); memcpy(client->operator_contract_digest,store->operator_contract_digest,32);
  memset(options,0,sizeof(*options)); options->service=client; options->client_store=store;
  memset(options->client_layout_digest,0xaa,32); memset(options->server_layout_digest,0xaa,32); memset(options->placement_plan_digest,0xaa,32);
}
static void test_remote_bind_rejection(credentials *c,bool resource_limit) {
  struct sockaddr_in a[2]; server *s=fixture(c,a);
  if (!resource_limit) { ria_json_free(&s->service.grants); s->service.grants=doc("{\"grants\":[]}"); }
  launch(s); char control[64],bulk[64]; ria_service client; ria_tensor_store store; ria_remote_options options;
  client_configuration(c,s,a,&client,&store,&options,control,bulk);
  if (resource_limit) client.limits.inflight_payload_bytes=1;
  ria_remote *remote=NULL; ria_error e={0}; CHECK(!ria_remote_open(&remote,&options,&e) && !remote &&
    e.code==(resource_limit ? RIA_RESOURCE_LIMIT : RIA_UNAUTHORIZED));
  CHECK(reap(true,1000)==0); fixture_free(s);
}
static bool peer_flush(server *s,unsigned ch,ria_error *e) {
  uint64_t deadline=end_after(1000);
  while (s->channels[ch].output_count) {
    if (!channel_write(s,ch,e)) return false;
    if (ria_monotonic_ms()>=deadline) return false;
    struct pollfd p={s->channels[ch].transport.fd,s->channels[ch].write_event,0};
    if (s->channels[ch].output_count && poll(&p,1,10)<0) return false;
  }
  return true;
}
static bool peer_bind(server *s,unsigned ch,ria_error *e) {
  ria_header h; uint8_t *p=NULL;
  bool ok=accept_channel(s,ch,e) &&
    ria_transport_frame(&s->channels[ch].transport,end_after(1000),100,65536,&h,&p,e) &&
    dispatch(s,ch,&h,p,ria_monotonic_ms(),e) && peer_flush(s,ch,e);
  free(p); return ok;
}
/* The fault peer uses the production authenticated Bind handlers, then emits
 * one deliberately chosen response to the real client RPC implementation. */
static void test_remote_response_bound(credentials *c,bool chunk,unsigned mode) {
  struct sockaddr_in a[2]; server *s=fixture(c,a); fixture_child=fork(); CHECK(fixture_child>=0);
  if (!fixture_child) {
    ria_error e={0}; bool ok=peer_bind(s,0,&e) && peer_bind(s,1,&e); ria_header h; uint8_t *p=NULL;
    unsigned ch=chunk ? 1 : 0;
    if (ok) ok=ria_transport_frame(&s->channels[ch].transport,end_after(1000),100,65536,&h,&p,&e);
    if (ok) ok=h.kind==(chunk ? RIA_CHUNK : RIA_ROWS) && h.request_id==2;
    if (ok) {
      free(p); p=NULL; h.flags=1; h.payload_length=mode==0 ? 16385 : mode==1 ? 16384 : mode==3 ? 256 : chunk ? 65 : 304;
      h.status=mode==1 || mode==3 ? RIA_RESOURCE_LIMIT : RIA_OK;
      if (mode==0) {
        uint8_t header[64]; ok=ria_header_encode(&h,header,&e) &&
          ria_transport_write(&s->channels[ch].transport,header,64,end_after(1000),&e);
      } else {
        p=calloc(1,(size_t)h.payload_length); ok=p!=NULL;
        if (ok && (mode==1 || mode==3)) {
          memset(p,' ',(size_t)h.payload_length);
          if (mode==1) { const char *error="{\"code\":5,\"message\":\"denied\"}"; memcpy(p,error,strlen(error)); }
          else {
            const char *prefix="{\"code\":5,\"message\":\""; size_t n=strlen(prefix);
            memcpy(p,prefix,n); memset(p+n,'A',191); memcpy(p+n+191,"\xf0\x9f\x98\x80tail\"}",10);
          }
        } else if (ok && chunk) {
          ria_write_u64(p,1); ria_write_u32(p+24,1); p[64]='Z'; ok=ria_sha256(p+64,1,p+32,&e);
        } else if (ok) {
          ria_write_u64(p,1); ria_write_u32(p+8,1); ria_write_u32(p+12,264); ria_write_u64(p+16,1);
          ria_write_u64(p+32,7); memset(p+40,0x55,264);
        }
        if (ok) ok=ria_transport_send(&s->channels[ch].transport,&h,p,end_after(1000),&e);
      }
    }
    free(p); p=NULL;
    if (ok && mode==2) {
      ok=ria_transport_frame(&s->channels[0].transport,end_after(1000),100,65536,&h,&p,&e) &&
        h.kind==RIA_CLOSE && dispatch(s,0,&h,p,ria_monotonic_ms(),&e) && close_ready(s,&e) && peer_flush(s,0,&e);
    } else if (ok) {
      ok=!ria_transport_frame(&s->channels[0].transport,end_after(1000),100,65536,&h,&p,&e) && e.code!=RIA_DEADLINE_EXCEEDED;
    }
    free(p); fixture_free(s); _exit(ok?0:1);
  }
  char control[64],bulk[64]; ria_service client; ria_tensor_store store; ria_remote_options options;
  client_configuration(c,s,a,&client,&store,&options,control,bulk);
  ria_remote *remote=NULL; ria_error e={0}; CHECK(ria_remote_open(&remote,&options,&e));
  uint8_t output[264],hash[32],*bytes=NULL; size_t length=0; memset(output,0x77,sizeof(output));
  CHECK(ria_sha256("Z",1,hash,&e));
  ria_shard shard={.id=1,.length=1,.chunk_size=1,.chunk_count=1,.chunk_hashes=hash}; uint64_t row=0,association=7;
  bool ok=chunk ? ria_remote_chunk(remote,&shard,0,&bytes,&length,&e) :
    ria_remote_rows(remote,1,&row,&association,1,output,&e);
  if (mode==2) {
    CHECK(ok);
    if (chunk) CHECK(length==1 && bytes[0]=='Z'); else for (unsigned i=0;i<264;i++) CHECK(output[i]==0x55);
  } else {
    CHECK(!ok && e.code==(mode==0 ? RIA_INVALID_REQUEST : RIA_RESOURCE_LIMIT));
    CHECK(!bytes && !length); for (unsigned i=0;i<264;i++) CHECK(output[i]==0x77);
    if (mode==3) {
      const char *prefix="remote operation failed: "; size_t n=strlen(prefix);
      CHECK(strlen(e.message)==n+191 && !memcmp(e.message,prefix,n));
      for (unsigned i=0;i<191;i++) CHECK(e.message[n+i]=='A');
    }
  }
  free(bytes); CHECK(ria_remote_close(remote,&e)); CHECK(reap(false,1000)==0); fixture_free(s);
}
static void test_callback_groups(credentials *c,bool rows,bool fail_group) {
  struct sockaddr_in a[2]; server *s=fixture(c,a);
  s->bank.operations[0]=(ria_operation){1,5120,5120,1,0,384,6,0,1.5f,false,true,true};
  s->bank.tables[0]=(ria_table){1,384006168,264,1}; ria_error e={0}; CHECK(server_minimum(s,&e));
  fixture_child=fork(); CHECK(fixture_child>=0);
  const uint16_t experts[6]={7,2,6,1,9,3},slots[6]={5,0,3,1,4,2};
  if (!fixture_child) {
    bool ok=peer_bind(s,0,&e) && peer_bind(s,1,&e); unsigned groups=rows ? 24 : fail_group ? 2 : 4;
    for (unsigned group=0;ok && group<groups;group++) {
      ria_header h; uint8_t *p=NULL;
      ok=ria_transport_frame(&s->channels[0].transport,end_after(1000),100,65536,&h,&p,&e) &&
        h.kind==(rows ? RIA_ROWS : RIA_EXPERT) && h.request_id==group+2;
      uint64_t response_bytes=0; ria_expert_request expert={0}; ria_row_request row={0};
      if (ok && rows) ok=ria_rows_parse(p,(size_t)h.payload_length,&s->bank.tables[0],&s->binding.limits,&row,&e) &&
        row.row_count==1 && ria_read_u64(row.pairs)==(group/2)*31 && ria_read_u64(row.pairs+8)==group;
      else if (ok) {
        ok=ria_expert_parse(p,(size_t)h.payload_length,h.kind,&s->bank.operations[0],&s->binding.limits,&expert,&e) &&
          expert.row_count==1 && expert.entry_count==3 && expert.invocation_id==group+1 &&
          ria_read_u64(expert.row_ids)==(group/2)*2+1;
        for (unsigned i=0;ok && i<3;i++) ok=ria_read_u16(expert.entries+i*8)==experts[(group%2)*3+i] &&
          ria_read_u16(expert.entries+i*8+2)==slots[(group%2)*3+i] && ria_read_f32(expert.entries+i*8+4)==.25f;
        for (unsigned i=0;ok && i<5120;i++) ok=ria_read_f32(expert.inputs+i*4)==1;
      }
      if (ok) response_bytes=rows ? row.response_bytes : expert.response_bytes;
      bool reject=fail_group && group==1;
      if (ok && !reject) ok=ria_binding_receive(&s->binding,&h,false,&e) && admit(s,&h,response_bytes,ria_monotonic_ms(),&e);
      uint8_t *response=ok && !reject ? calloc(1,(size_t)response_bytes) : NULL;
      if (ok && !reject) ok=response!=NULL;
      if (ok && !reject && rows) {
        ria_write_u64(response,1); ria_write_u32(response+8,1); ria_write_u32(response+12,264); ria_write_u64(response+16,1);
        memcpy(response+24,row.pairs,16); memset(response+40,group+1,264);
      } else if (ok && !reject) {
        ria_write_u64(response,1); ria_write_u64(response+8,expert.invocation_id); ria_write_u32(response+16,1);
        ria_write_u32(response+20,5120); ria_write_u32(response+24,3);
        for (unsigned i=0;i<3;i++) for (unsigned j=0;j<5120;j++)
          ria_write_f32(response+32+((size_t)i*5120+j)*4,(float)(slots[(group%2)*3+i]+1));
      }
      free(p);
      if (ok && reject) ok=error_reply(s,0,&h,RIA_RESOURCE_LIMIT,false,false,end_after(200),&e) && peer_flush(s,0,&e);
      else if (ok) ok=queue_reply(s,0,&h,response,(size_t)response_bytes,0,true,false,end_after(200),&e) && peer_flush(s,0,&e);
      else free(response);
      if (reject) break;
    }
    ria_header close; uint8_t *p=NULL;
    if (ok && !fail_group) ok=ria_transport_frame(&s->channels[0].transport,end_after(1000),100,65536,&close,&p,&e) &&
      close.kind==RIA_CLOSE && dispatch(s,0,&close,p,ria_monotonic_ms(),&e) && close_ready(s,&e) && peer_flush(s,0,&e);
    else if (ok) ok=!ria_transport_frame(&s->channels[0].transport,end_after(1000),100,65536,&close,&p,&e) && e.code!=RIA_DEADLINE_EXCEEDED;
    free(p); fixture_free(s); _exit(ok?0:1);
  }
  char control[64],bulk[64]; ria_service client; ria_tensor_store store; ria_remote_options options;
  client_configuration(c,s,a,&client,&store,&options,control,bulk); ria_remote *remote=NULL;
  CHECK(ria_remote_open(&remote,&options,&e)); ria_graph_remote callbacks=ria_remote_callbacks(remote);
  float input[5120],weights[6],*output=malloc(6u*5120*sizeof(float)); uint64_t ids[24]; uint8_t packed[24*264]; CHECK(output);
  for (unsigned i=0;i<5120;i++) input[i]=1;
  for (unsigned i=0;i<6;i++) weights[i]=.25f;
  for (unsigned i=0;i<24;i++) ids[i]=(i/2)*31;
  unsigned calls=rows || fail_group ? 1 : 2; uint64_t epoch=0,generation=0;
  for (unsigned call=0;call<calls;call++) {
    uint64_t next_epoch,next_generation; CHECK(ria_remote_begin_generation(remote,&next_epoch,&next_generation,&e));
    CHECK(next_generation==generation+1 && (!epoch || next_epoch==epoch)); epoch=next_epoch; generation=next_generation;
    for (unsigned i=0;i<6u*5120;i++) output[i]=-7;
    memset(packed,0x77,sizeof(packed));
    bool ok=rows ? callbacks.engram(callbacks.context,1,ids,24,packed,&e) :
      callbacks.experts(callbacks.context,0,input,experts,weights,slots,6,output,&e);
    CHECK(ok!=fail_group);
    if (rows) for (unsigned i=0;i<24;i++) for (unsigned j=0;j<264;j++) CHECK(packed[i*264+j]==(fail_group ? 0x77 : i+1));
    else for (unsigned i=0;i<6;i++) for (unsigned j=0;j<5120;j++) CHECK(output[i*5120+j]==(fail_group ? -7 : slots[i]+1));
    if (fail_group) CHECK(e.code==RIA_RESOURCE_LIMIT);
  }
  free(output); CHECK(ria_remote_close(remote,&e)); CHECK(reap(false,1000)==0); fixture_free(s);
}
static void test_bind_minimum(credentials *c,unsigned kind) {
  struct sockaddr_in a[2]; server *s=fixture(c,a); ria_error e={0};
  ria_shard shard={.id=1,.length=4096,.chunk_size=4096,.chunk_count=1}; ria_chunk_grant grant={1,0,4096};
  if (kind==1) s->bank.operations[0]=(ria_operation){1,5120,5120,1,0,384,6,0,1.5f,false,true,true};
  if (kind==2) { s->store.shards=&shard; s->store.shard_count=1; s->grants.ranges=&grant; s->grants.count=1; }
  CHECK(server_minimum(s,&e)); launch(s);
  /* Parent/child own fixture-only borrowed metadata; prevent free of the stack. */
  s->grants.ranges=NULL; s->grants.count=0;
  char control[64],bulk[64]; ria_service client; ria_tensor_store store; ria_remote_options options;
  client_configuration(c,s,a,&client,&store,&options,control,bulk);
  if (kind==0) client.limits.frame_payload_bytes=16383;
  if (kind==1) client.limits.frame_payload_bytes=20543;
  if (kind==2) client.limits.bulk_data_bytes=2048;
  if (kind==3) {
    uint64_t control_bytes,row_bytes,bulk_bytes; CHECK(ria_progress_charges(&client.limits,&control_bytes,&row_bytes,&bulk_bytes,&e));
    client.limits.inflight_payload_bytes=control_bytes+row_bytes+bulk_bytes+s->minimum_expert_charge-1;
  }
  ria_remote *remote=NULL; CHECK(!ria_remote_open(&remote,&options,&e) && !remote && e.code==RIA_RESOURCE_LIMIT);
  CHECK(reap(true,1000)==0); fixture_free(s);
}
static void test_blocked_write_ownership(credentials *c) {
  struct sockaddr_in a[2]; server *s=fixture(c,a); int barrier[2]; CHECK(pipe(barrier)==0);
  fixture_child=fork(); CHECK(fixture_child>=0);
  if (!fixture_child) {
    close(barrier[0]); ria_error e={0}; bool ok=accept_channel(s,0,&e);
    int bytes=4096;
    if (ok) ok=setsockopt(s->channels[0].transport.fd,SOL_SOCKET,SO_SNDBUF,&bytes,sizeof(bytes))==0;
    uint8_t *payload=calloc(1,524288); ria_header h={.kind=RIA_EXPERT,.request_id=1};
    if (ok && payload) ok=queue_reply(s,0,&h,payload,524288,0,false,false,end_after(200),&e);
    else { free(payload); ok=false; }
    if (ok) ok=channel_write(s,0,&e) && s->channels[0].write_retry;
    if (ok) ok=write(barrier[1],"R",1)==1;
    close(barrier[1]);
    if (ok) ok=event_loop(s,&e) && !s->channels[0].frame_deadline;
    fixture_free(s); _exit(ok?0:1);
  }
  close(barrier[1]); ria_transport t=connect_client(c,&a[0]); char marker;
  CHECK(read(barrier[0],&marker,1)==1 && marker=='R'); close(barrier[0]);
  ria_error e={0}; uint8_t byte='D'; CHECK(ria_transport_write(&t,&byte,1,ria_monotonic_ms()+1000,&e));
  CHECK(poll(NULL,0,250)==0); CHECK(reap(true,1000)==0); ria_transport_close(&t); fixture_free(s);
}
static void test_fatal_ssl_marks_unusable(credentials *c) {
  struct sockaddr_in a; int listening=listener(&a); fixture_child=fork(); CHECK(fixture_child>=0);
  if (!fixture_child) {
    int fd=accept(listening,NULL,NULL); close(listening); ria_error e={0}; channel ch={0};
    bool ok=fd>=0 && ria_transport_accept(&ch.transport,&c->tls,fd,&e);
    if (ok) {
      struct pollfd p={fd,POLLIN,0}; ok=poll(&p,1,1000)>0;
      uint8_t byte; size_t received=0;
      if (ok) ok=!ssl_step(&ch,false,&byte,1,&received,&e) && ch.transport.unusable;
    }
    if (ch.transport.ssl) ria_transport_close(&ch.transport); else if (fd>=0) close(fd);
    _exit(ok?0:1);
  }
  ria_transport t=connect_client(c,&a); close(listening);
  CHECK(shutdown(t.fd,SHUT_RDWR)==0); t.unusable=true; ria_transport_close(&t); CHECK(reap(false,1000)==0);
}
static void test_ssl_error_queue(credentials *c) {
  struct sockaddr_in a; int listening=listener(&a); fixture_child=fork(); CHECK(fixture_child>=0);
  if (!fixture_child) {
    ria_error e={0}; channel ch={0}; int fd=accept(listening,NULL,NULL); close(listening);
    bool ok=fd>=0 && ria_transport_accept(&ch.transport,&c->tls,fd,&e);
    if (ok) {
      uint8_t byte; size_t received=0; ERR_raise(ERR_LIB_USER,7);
      ok=ssl_step(&ch,false,&byte,1,&received,&e) && !received && ch.read_event==POLLIN;
    }
    if (ch.transport.ssl) ria_transport_close(&ch.transport); else if (fd>=0) close(fd);
    _exit(ok?0:1);
  }
  ria_transport t=connect_client(c,&a); close(listening); CHECK(reap(false,1000)==0); ria_transport_close(&t);
}
int main(void) {
  CHECK(signal(SIGPIPE,SIG_IGN)!=SIG_ERR); alarm(20);
  credentials c={0}; credentials_create(&c);
  test_initial_silence(&c,false); test_initial_silence(&c,true); test_bound_idle(&c);
  test_late_completion(&c); test_retained_accept(&c); test_disconnected_live_executor(&c); test_ssl_error_queue(&c);
  test_admitted_connection_loss(&c,0); test_admitted_connection_loss(&c,1); test_started_header_deadline(&c);
  test_bind_failure(&c,false); test_bind_failure(&c,true); test_control_credit_refusal(&c); test_refusal_accounting(&c);
  test_remote_bind_rejection(&c,false); test_remote_bind_rejection(&c,true);
  test_blocked_write_ownership(&c); test_fatal_ssl_marks_unusable(&c);
  for (unsigned mode=0;mode<4;mode++) { test_remote_response_bound(&c,false,mode); test_remote_response_bound(&c,true,mode); }
  test_callback_groups(&c,false,false); test_callback_groups(&c,false,true);
  test_callback_groups(&c,true,false); test_callback_groups(&c,true,true);
  for (unsigned kind=0;kind<4;kind++) test_bind_minimum(&c,kind);
  credentials_free(&c); alarm(0);
  puts("RIA production server lifecycle: setup/frame deadlines, idle/rebind ownership, channel loss/drain, TLS retry/fatal rules, typed refusals, response caps, negotiated slot/row groups and atomic publication, operation/error/chunk Bind minima passed");
  return 0;
}
