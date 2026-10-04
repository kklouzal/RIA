#define _GNU_SOURCE
#define DS4_NO_GPU
#define DS4_RIA
#define DS4_SERVER_TEST
#define DS4_SERVER_TEST_NO_MAIN
#define ds4_engine_ria fixture_engine_ria
#define ria_engine_service fixture_engine_service
#define ria_engine_context fixture_engine_context
#define ria_engine_frontend_budget fixture_engine_frontend_budget
#define ria_engine_tokenizer fixture_engine_tokenizer
#define ria_tokenizer_max_token_bytes fixture_max_token
#define ria_admin_reserved_bytes fixture_admin_bytes
#include "../../ds4_server.c"
#include <sys/wait.h>
#undef ds4_engine_ria
#undef ria_engine_service
#undef ria_engine_context
#undef ria_engine_frontend_budget
#undef ria_engine_tokenizer
#undef ria_tokenizer_max_token_bytes
#undef ria_admin_reserved_bytes
#define REQUIRE(condition) do { if (!(condition)) { \
  fprintf(stderr,"fixture failure %s:%d: %s\n",__FILE__,__LINE__,#condition); \
  exit(1); } } while (0)
/* Explicit fixture-owned model metadata and capacity shims. They replace no
 * HTTP/policy/parser/credential/signal operation under review. */
static ria_service package;
static uint64_t frontend_budget=UINT64_C(1073741824);
struct ria_engine *fixture_engine_ria(ds4_engine *e) { (void)e; return (struct ria_engine *)&package; }
const ria_service *fixture_engine_service(const struct ria_engine *r) { (void)r; return &package; }
uint64_t fixture_engine_context(const struct ria_engine *r) { (void)r; return 64; }
uint64_t fixture_engine_frontend_budget(const struct ria_engine *r) { (void)r; return frontend_budget; }
ria_tokenizer *fixture_engine_tokenizer(struct ria_engine *r) { (void)r; return NULL; }
uint64_t fixture_max_token(const ria_tokenizer *t) { (void)t; return 8; }
uint64_t fixture_admin_bytes(void) { return 0; }
static bool policy(const char *path,char suffix,ria_error *e) {
  char json[4096];
  int n=snprintf(json,sizeof json,"{\"api\":{\"bind_address\":\"127.0.0.1:8080%s\",\"bearer_token_file\":\"%s%s\",\"max_body_bytes\":4096,\"header_timeout_ms\":100,\"body_timeout_ms\":100,\"stream_write_timeout_ms\":100,\"max_active_generations\":1,\"max_queued_generations\":0,\"allow_remote_image_urls\":false,\"cors_allowed_origins\":[],\"max_header_bytes\":128,\"max_json_depth\":4,\"max_json_nodes\":64,\"max_messages\":4,\"max_tools\":0,\"max_images\":0,\"max_encoded_image_bytes\":4096,\"max_decoded_image_bytes\":4096,\"max_http_connections\":1}}", suffix=='a'?"\\u0000discarded":"",path,suffix=='t'?"\\u0000discarded":"");
  REQUIRE(n>0 && (size_t)n<sizeof json && ria_json_parse(json,(size_t)n,(ria_json_limits){4096,256,8},&package.document,e));
  server s={0};server_config cfg={0};cfg.default_tokens=8;char host[256];
  bool ok=ria_server_policy(&s,&cfg,host,e);
  if(ok) REQUIRE(s.ria_bearer_length==13 && !strcmp(host,"127.0.0.1") && cfg.port==8080);
  ria_json_free(&package.document);return ok;
}
typedef struct {
  server owner;
  pthread_mutex_t mutex;
  pthread_cond_t condition;
  bool sampled;
} health_observer;
static void *observe_health(void *context) {
  health_observer *o=context;bool ready=false,active=false;ria_error e={0};
  REQUIRE(ria_client_health(&o->owner,&ready,&active,&e) && ready && active);
  pthread_mutex_lock(&o->mutex);o->sampled=true;
  pthread_cond_signal(&o->condition);pthread_mutex_unlock(&o->mutex);
  uint64_t deadline=ria_monotonic_ms()+1000;
  while (!ria_startup_cancelled(NULL) && ria_monotonic_ms()<deadline)
    REQUIRE(poll(NULL,0,1)==0 || errno==EINTR);
  REQUIRE(ria_startup_cancelled(NULL));
  REQUIRE(ria_client_health(&o->owner,&ready,&active,&e) && !ready && active);
  return NULL;
}
static void test_signal_health(void) {
  health_observer o={0};REQUIRE(pthread_mutex_init(&o.owner.mu,NULL)==0 &&
    pthread_mutex_init(&o.mutex,NULL)==0 && pthread_cond_init(&o.condition,NULL)==0);
  o.owner.ria_ready=true;o.owner.ria_generation_reserved=true;
  __atomic_store_n(&g_ria_signal_mode,1,__ATOMIC_RELAXED);
  pthread_t thread;REQUIRE(pthread_create(&thread,NULL,observe_health,&o)==0);
  pthread_mutex_lock(&o.mutex);
  while (!o.sampled) REQUIRE(pthread_cond_wait(&o.condition,&o.mutex)==0);
  pthread_mutex_unlock(&o.mutex);
  REQUIRE(raise(SIGTERM)==0 && pthread_join(thread,NULL)==0);
  /* A startup publisher racing cancellation cannot make health ready again. */
  pthread_mutex_lock(&o.owner.mu);o.owner.ria_ready=true;pthread_mutex_unlock(&o.owner.mu);
  bool ready=true,active=false;ria_error e={0};
  REQUIRE(ria_client_health(&o.owner,&ready,&active,&e) && !ready && active);
  REQUIRE(pthread_cond_destroy(&o.condition)==0 && pthread_mutex_destroy(&o.mutex)==0 &&
    pthread_mutex_destroy(&o.owner.mu)==0);
  __atomic_store_n(&g_stop_requested,0,__ATOMIC_RELAXED);
}
static void test_signal_listener(bool ria) {
  int fd=socket(AF_INET,SOCK_STREAM|SOCK_CLOEXEC,0);REQUIRE(fd>=0);
  __atomic_store_n(&g_ria_signal_mode,ria,__ATOMIC_RELAXED);
  __atomic_store_n(&g_listen_fd,fd,__ATOMIC_RELAXED);
  REQUIRE(raise(SIGTERM)==0 && server_stop_requested());
  int owned=(int)__atomic_exchange_n(&g_listen_fd,-1,__ATOMIC_RELAXED);
  if (ria) REQUIRE(owned==fd && fcntl(fd,F_GETFD)>=0 && close(owned)==0);
  else REQUIRE(owned==-1 && fcntl(fd,F_GETFD)==-1 && errno==EBADF);
  __atomic_store_n(&g_stop_requested,0,__ATOMIC_RELAXED);
}
static void test_second_signal_exit(void) {
  pid_t child=fork();REQUIRE(child>=0);
  if (!child) {
    __atomic_store_n(&g_ria_signal_mode,1,__ATOMIC_RELAXED);
    REQUIRE(raise(SIGTERM)==0);REQUIRE(raise(SIGTERM)==0);_exit(1);
  }
  int status=0;REQUIRE(waitpid(child,&status,0)==child && WIFEXITED(status) && WEXITSTATUS(status)==130);
}
int main(void) {
  alarm(5);
  char root[]="/tmp/ria-api-policy-XXXXXX";REQUIRE(mkdtemp(root));
  char token[256],fifo[256];REQUIRE(snprintf(token,sizeof token,"%s/api.token",root)>0);
  REQUIRE(snprintf(fifo,sizeof fifo,"%s/token.fifo",root)>0);
  int fd=open(token,O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);REQUIRE(fd>=0);
  REQUIRE(write(fd,"fixture-token",13)==13 && close(fd)==0 && mkfifo(fifo,0600)==0);
  ria_error e={0};REQUIRE(policy(token,0,&e));
  REQUIRE(!policy(token,'a',&e) && e.code==RIA_INVALID_REQUEST);
  REQUIRE(!policy(token,'t',&e) && e.code==RIA_INVALID_REQUEST);
  REQUIRE(!policy(fifo,0,&e) && e.code==RIA_UNAUTHORIZED);
  frontend_budget=0;REQUIRE(!policy(token,0,&e) && e.code==RIA_RESOURCE_LIMIT);
  frontend_budget=UINT64_C(1073741824);
  struct sigaction old,action={0};action.sa_handler=stop_signal_handler;sigemptyset(&action.sa_mask);
  REQUIRE(sigaction(SIGTERM,&action,&old)==0 && !ria_startup_cancelled(NULL));
  REQUIRE(raise(SIGTERM)==0 && ria_startup_cancelled(NULL));
  __atomic_store_n(&g_stop_requested,0,__ATOMIC_RELAXED);
  test_signal_health();test_signal_listener(false);test_signal_listener(true);test_second_signal_exit();
  __atomic_store_n(&g_ria_signal_mode,0,__ATOMIC_RELAXED);
  REQUIRE(sigaction(SIGTERM,&old,NULL)==0);
  REQUIRE(unlink(token)==0 && unlink(fifo)==0 && rmdir(root)==0);alarm(0);
  puts("RIA actual API policy: bounded bearer file, complete JSON address/path, capacity admission, startup cancellation, concurrent health readiness and signal listener ownership passed");return 0;
}
