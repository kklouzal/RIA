#define _POSIX_C_SOURCE 200809L
#include "admission.h"
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
const char *const ria_phase_names[RIA_PHASES] = {"startup",      "prefill", "decode",
                                                 "continuation", "image",   "drain"};
#define COUNT(a) (sizeof(a) / sizeof((a)[0]))
static bool exact(const ria_json_doc *d, uint32_t i, const char *value)
{
    const ria_json_node *n = ria_json_at(d, i);
    return n && n->type == RIA_JSON_STRING && n->length == strlen(value) &&
           !memcmp(n->text, value, n->length);
}
static bool revision(const ria_json_doc *d, ria_error *e)
{
    uint64_t rev;
    return ria_json_u64(d, ria_json_get(d, 0, "schema_revision"), false, &rev, e) &&
           (rev == 1 || ria_fail(e, RIA_UNSUPPORTED, "unsupported admission schema revision"));
}
static bool integer(const ria_json_doc *d, uint32_t obj, const char *name, uint64_t *out,
                    ria_error *e)
{
    return ria_json_u64(d, ria_json_get(d, obj, name), false, out, e);
}
static bool boolean(const ria_json_doc *d, uint32_t obj, const char *name, bool *out, ria_error *e)
{
    const ria_json_node *n = ria_json_at(d, ria_json_get(d, obj, name));
    if (!n || n->type != RIA_JSON_BOOL)
        return ria_fail(e, RIA_INVALID_REQUEST, "expected admission boolean: %s", name);
    *out = n->boolean;
    return true;
}
static bool identity(const ria_json_doc *d, uint8_t hash[32], bool required, ria_error *e)
{
    uint32_t i = ria_json_get(d, 0, "digest");
    if (i == RIA_JSON_NONE && required)
        return ria_fail(e, RIA_INTEGRITY_ERROR, "admission evidence digest missing");
    if (!ria_json_sha256(d, true, hash, e))
        return false;
    if (i != RIA_JSON_NONE) {
        uint8_t declared[32];
        if (!ria_json_digest_field(d, i, declared, e))
            return false;
        if (memcmp(hash, declared, 32))
            return ria_fail(e, RIA_INTEGRITY_ERROR, "admission evidence digest mismatch");
    }
    return true;
}
static bool equal_digest(const ria_json_doc *d, const char *name, const uint8_t expected[32],
                         ria_error *e)
{
    uint8_t actual[32];
    if (!ria_json_digest_field(d, ria_json_get(d, 0, name), actual, e))
        return false;
    return !memcmp(actual, expected, 32) ||
           ria_fail(e, RIA_IDENTITY_MISMATCH, "admission identity mismatch: %s", name);
}
static bool memory(const ria_json_doc *d, uint32_t object, ria_memory_peak *out, ria_error *e)
{
    static const char *const fields[] = {"host_bytes", "device_bytes", "pinned_bytes", "numa"};
    if (!ria_json_fields(d, object, fields, 4, fields, 4, e))
        return false;
    memset(out, 0, sizeof *out);
    if (!integer(d, object, "host_bytes", &out->host, e) ||
        !integer(d, object, "device_bytes", &out->device, e) ||
        !integer(d, object, "pinned_bytes", &out->pinned, e))
        return false;
    const ria_json_node *nodes = ria_json_at(d, ria_json_get(d, object, "numa"));
    if (!nodes || nodes->type != RIA_JSON_ARRAY)
        return ria_fail(e, RIA_INVALID_REQUEST, "NUMA capacities require an array");
    bool seen[RIA_NUMA_MAX] = {0};
    static const char *const node_fields[] = {"node", "bytes"};
    for (uint32_t i = nodes->child; i != RIA_JSON_NONE; i = d->nodes[i].next) {
        uint64_t node, bytes;
        if (!ria_json_fields(d, i, node_fields, 2, node_fields, 2, e) ||
            !integer(d, i, "node", &node, e) || !integer(d, i, "bytes", &bytes, e))
            return false;
        if (node >= RIA_NUMA_MAX || seen[node] || !bytes)
            return ria_fail(e, RIA_INVALID_REQUEST, "invalid or duplicate NUMA capacity");
        seen[node] = true;
        out->numa[node] = bytes;
    }
    return true;
}
static bool under(const ria_memory_peak *a, const ria_memory_peak *b, const char *what,
                  ria_error *e)
{
    if (a->host > b->host || a->device > b->device || a->pinned > b->pinned)
        return ria_fail(e, RIA_RESOURCE_LIMIT, "%s exceeds host/device/pinned capacity", what);
    for (unsigned i = 0; i < RIA_NUMA_MAX; i++)
        if (a->numa[i] > b->numa[i])
            return ria_fail(e, RIA_RESOURCE_LIMIT, "%s exceeds NUMA node %u capacity", what, i);
    return true;
}
static bool add(uint64_t *sum, uint64_t bytes, ria_error *e)
{
    if (!ria_u64_add(*sum, bytes, sum) || *sum > RIA_JSON_SAFE_INTEGER)
        return ria_fail(e, RIA_RESOURCE_LIMIT, "admission byte arithmetic overflow");
    return true;
}
bool ria_admission_compute(const ria_json_doc *request, const ria_json_doc *inventory,
                           const ria_json_doc *probe, const ria_json_doc *calibration,
                           ria_admission_plan *plan, ria_error *e)
{
    if (!plan)
        return ria_fail(e, RIA_INVALID_REQUEST, "missing admission output");
    memset(plan, 0, sizeof *plan);
    static const char *const req_fields[] = {"schema_revision",
                                             "role",
                                             "executor",
                                             "profile",
                                             "logical_model_digest",
                                             "operator_contract_digest",
                                             "context_positions",
                                             "prefill_rows",
                                             "caps",
                                             "digest"};
    static const char *const inv_fields[] = {
        "schema_revision",        "logical_model_digest", "operator_contract_digest",
        "semantic_max_positions", "allocations",          "digest", "derivation"};
    static const char *const probe_fields[] = {
        "schema_revision", "role", "executor",  "host_bytes", "device_bytes",
        "pinned_bytes",    "numa", "qualified", "environment_digest", "build_digest", "evidence_digest", "digest"};
    static const char *const cal_fields[] = {
        "schema_revision", "profile",   "operator_contract_digest",
        "executor",        "qualified", "environment_digest", "build_digest", "evidence_digest", "policy_digest", "digest"};
    if (!ria_json_fields(request, 0, req_fields, COUNT(req_fields), req_fields,
                         COUNT(req_fields) - 1, e) ||
        !ria_json_fields(inventory, 0, inv_fields, COUNT(inv_fields), inv_fields,
                         COUNT(inv_fields) - 2, e) ||
        !ria_json_fields(probe, 0, probe_fields, COUNT(probe_fields), probe_fields,
                         COUNT(probe_fields), e) ||
        !ria_json_fields(calibration, 0, cal_fields, COUNT(cal_fields), cal_fields,
                         COUNT(cal_fields), e) ||
        !revision(request, e) || !revision(inventory, e) || !revision(probe, e) ||
        !revision(calibration, e))
        return false;
    bool client = exact(request, ria_json_get(request, 0, "role"), "client"),
         expert = exact(request, ria_json_get(request, 0, "role"), "expert"),
         cpu = exact(request, ria_json_get(request, 0, "executor"), "cpu"),
         cuda = exact(request, ria_json_get(request, 0, "executor"), "cuda");
    if ((!client && !expert) || (!cpu && !cuda) || (client && !cuda))
        return ria_fail(e, RIA_INVALID_REQUEST, "invalid admission role/executor");
    strcpy(plan->role, client ? "client" : "expert");
    strcpy(plan->executor, cpu ? "cpu" : "cuda");
    const char *profile = NULL;
    size_t pn;
    if (!ria_json_string(request, ria_json_get(request, 0, "profile"), &profile, &pn, e))
        return false;
    if (!(exact(request, ria_json_get(request, 0, "profile"), "nvfp4") ||
          exact(request, ria_json_get(request, 0, "profile"), "fp8") ||
          exact(request, ria_json_get(request, 0, "profile"), "bf16")))
        return ria_fail(e, RIA_UNSUPPORTED, "unsupported admission profile");
    memcpy(plan->profile, profile, pn);
    plan->profile[pn] = 0;
    if (!ria_json_digest_field(request, ria_json_get(request, 0, "logical_model_digest"),
                               plan->logical_model_digest, e) ||
        !ria_json_digest_field(request, ria_json_get(request, 0, "operator_contract_digest"),
                               plan->operator_contract_digest, e) ||
        !equal_digest(inventory, "logical_model_digest", plan->logical_model_digest, e) ||
        !equal_digest(inventory, "operator_contract_digest", plan->operator_contract_digest, e) ||
        !equal_digest(calibration, "operator_contract_digest", plan->operator_contract_digest, e))
        return false;
    bool pq, cq;
    uint8_t evidence[32];
    if (!ria_json_digest_field(probe,ria_json_get(probe,0,"environment_digest"),plan->environment_digest,e) ||
        !ria_json_digest_field(probe,ria_json_get(probe,0,"build_digest"),plan->build_digest,e) ||
        !ria_json_digest_field(probe,ria_json_get(probe,0,"evidence_digest"),evidence,e) ||
        !equal_digest(calibration,"environment_digest",plan->environment_digest,e) ||
        !equal_digest(calibration,"build_digest",plan->build_digest,e) ||
        !ria_json_digest_field(calibration,ria_json_get(calibration,0,"evidence_digest"),evidence,e) ||
        !ria_json_digest_field(calibration,ria_json_get(calibration,0,"policy_digest"),plan->policy_digest,e))
        return false;
    if (!boolean(probe, 0, "qualified", &pq, e) || !boolean(calibration, 0, "qualified", &cq, e) ||
        !pq || !cq)
        return ria_fail(e, RIA_NOT_READY, "required probe or calibration is unqualified");
    if (!exact(probe, ria_json_get(probe, 0, "role"), plan->role) ||
        !exact(probe, ria_json_get(probe, 0, "executor"), plan->executor) ||
        !exact(calibration, ria_json_get(calibration, 0, "executor"), plan->executor) ||
        !exact(calibration, ria_json_get(calibration, 0, "profile"), plan->profile))
        return ria_fail(e, RIA_IDENTITY_MISMATCH,
                        "probe/calibration role, executor or profile mismatch");
    if (!identity(request, plan->request_digest, false, e) ||
        !identity(inventory, plan->inventory_digest, false, e) ||
        !identity(probe, plan->probe_digest, true, e) ||
        !identity(calibration, plan->calibration_digest, true, e))
        return false;
    if (!memory(request, ria_json_get(request, 0, "caps"), &plan->caps, e))
        return false;
    /* Probe capacities have additional evidence fields; read them explicitly. */
    ria_memory_peak available = {0};
    if (!integer(probe, 0, "host_bytes", &available.host, e) ||
        !integer(probe, 0, "device_bytes", &available.device, e) ||
        !integer(probe, 0, "pinned_bytes", &available.pinned, e))
        return false;
    const ria_json_node *nodes = ria_json_at(probe, ria_json_get(probe, 0, "numa"));
    if (!nodes || nodes->type != RIA_JSON_ARRAY)
        return ria_fail(e, RIA_INVALID_REQUEST, "probe NUMA capacities require an array");
    bool seen[RIA_NUMA_MAX] = {0};
    static const char *const node_fields[] = {"node", "bytes"};
    for (uint32_t i = nodes->child; i != RIA_JSON_NONE; i = probe->nodes[i].next) {
        uint64_t node, bytes;
        if (!ria_json_fields(probe, i, node_fields, 2, node_fields, 2, e) ||
            !integer(probe, i, "node", &node, e) || !integer(probe, i, "bytes", &bytes, e))
            return false;
        if (node >= RIA_NUMA_MAX || seen[node] || !bytes)
            return ria_fail(e, RIA_INVALID_REQUEST, "invalid probe NUMA capacity");
        seen[node] = true;
        available.numa[node] = bytes;
    }
    if (!plan->caps.host || (cuda && !plan->caps.device) ||
        (cpu && (plan->caps.device || plan->caps.pinned || available.device || available.pinned)) ||
        plan->caps.pinned > plan->caps.host)
        return ria_fail(e, RIA_INVALID_REQUEST, "invalid host/device/pinned cap contract");
    if (!under(&plan->caps, &available, "requested caps", e))
        return false;
    uint64_t semantic;
    if (!integer(request, 0, "context_positions", &plan->context_positions, e) ||
        !integer(request, 0, "prefill_rows", &plan->prefill_rows, e) ||
        !plan->prefill_rows || plan->prefill_rows > 64 || plan->prefill_rows > plan->context_positions ||
        !integer(inventory, 0, "semantic_max_positions", &semantic, e) ||
        !plan->context_positions || plan->context_positions > semantic)
        return ria_fail(e, RIA_RESOURCE_LIMIT, "context exceeds semantic position contract");
    uint32_t derivation=ria_json_get(inventory,0,"derivation");
    if (client && derivation==RIA_JSON_NONE)
        return ria_fail(e,RIA_INTEGRITY_ERROR,"client inventory must bind its context/prefill derivation");
    if (derivation!=RIA_JSON_NONE) {
        const char *const fields[]={"manifest_digest","runtime_policy_digest","request_digest","context_positions","prefill_rows"};
        uint8_t derived_digest[32];uint64_t derived_context,derived_rows;
        if (!ria_json_fields(inventory,derivation,fields,5,fields,5,e)) return false;
        for (unsigned i=0;i<3;i++)
            if (!ria_json_digest_field(inventory,ria_json_get(inventory,derivation,fields[i]),derived_digest,e)) return false;
        if (!integer(inventory,derivation,"context_positions",&derived_context,e) || derived_context!=semantic ||
            !integer(inventory,derivation,"prefill_rows",&derived_rows,e) || derived_rows!=plan->prefill_rows)
            return ria_fail(e,RIA_IDENTITY_MISMATCH,"inventory derivation context/prefill differs from workload bound");
    }
    const ria_json_node *allocations =
        ria_json_at(inventory, ria_json_get(inventory, 0, "allocations"));
    if (!allocations || allocations->type != RIA_JSON_ARRAY || allocations->child == RIA_JSON_NONE)
        return ria_fail(e, RIA_INVALID_REQUEST, "empty physical allocation inventory");
    static const char *const alloc_fields[] = {
        "id",        "name",   "resource",           "base_bytes", "bytes_per_position",
        "numa_node", "pinned", "protected_progress", "phases"};
    uint64_t ids[4096] = {0};
    bool progress_host[RIA_PHASES] = {0}, progress_device[RIA_PHASES] = {0};
    uint64_t nonprogress = 0;
    for (uint32_t i = allocations->child; i != RIA_JSON_NONE; i = inventory->nodes[i].next) {
        if (plan->allocation_count == COUNT(ids))
            return ria_fail(e, RIA_RESOURCE_LIMIT, "physical allocation count bound exceeded");
        uint64_t id, base, per_position, bytes;
        bool pinned, protected_progress;
        if (!ria_json_fields(inventory, i, alloc_fields, COUNT(alloc_fields), alloc_fields,
                             COUNT(alloc_fields), e) ||
            !ria_json_u64(inventory, ria_json_get(inventory, i, "id"), true, &id, e) || !id ||
            !integer(inventory, i, "base_bytes", &base, e) ||
            !integer(inventory, i, "bytes_per_position", &per_position, e) ||
            !boolean(inventory, i, "pinned", &pinned, e) ||
            !boolean(inventory, i, "protected_progress", &protected_progress, e))
            return ria_fail(e, RIA_INVALID_REQUEST, "invalid physical allocation descriptor");
        for (uint64_t j = 0; j < plan->allocation_count; j++)
            if (ids[j] == id)
                return ria_fail(e, RIA_INVALID_REQUEST, "duplicate physical allocation identity");
        ids[plan->allocation_count++] = id;
        const char *name;
        size_t name_length;
        if (!ria_json_string(inventory, ria_json_get(inventory, i, "name"), &name, &name_length,
                             e) ||
            !name_length || name_length > 256)
            return ria_fail(e, RIA_INVALID_REQUEST, "invalid allocation name");
        (void)name;
        bool host = exact(inventory, ria_json_get(inventory, i, "resource"), "host"),
             device = exact(inventory, ria_json_get(inventory, i, "resource"), "device");
        if ((!host && !device) || (device && pinned) || (cpu && device))
            return ria_fail(e, RIA_INVALID_REQUEST, "invalid allocation resource");
        if (!ria_u64_mul(per_position, plan->context_positions, &bytes) ||
            !ria_u64_add(base, bytes, &bytes) || !bytes || bytes > RIA_JSON_SAFE_INTEGER)
            return ria_fail(e, RIA_RESOURCE_LIMIT, "physical allocation size overflow or zero");
        size_t addressable;
        if (!ria_size(bytes, &addressable, e))
            return false;
        (void)addressable;
        int node = -1;
        uint32_t ni = ria_json_get(inventory, i, "numa_node");
        const ria_json_node *nn = ria_json_at(inventory, ni);
        if (!nn)
            return ria_fail(e, RIA_INVALID_REQUEST, "missing NUMA placement");
        if (nn->type != RIA_JSON_NULL) {
            uint64_t n;
            if (!host || !ria_json_u64(inventory, ni, false, &n, e) || n >= RIA_NUMA_MAX ||
                !plan->caps.numa[n])
                return ria_fail(e, RIA_INVALID_REQUEST, "unadmitted NUMA placement");
            node = (int)n;
        }
        const ria_json_node *phases = ria_json_at(inventory, ria_json_get(inventory, i, "phases"));
        if (!phases || phases->type != RIA_JSON_ARRAY || phases->child == RIA_JSON_NONE)
            return ria_fail(e, RIA_INVALID_REQUEST, "allocation must declare active phases");
        bool phase_seen[RIA_PHASES] = {0};
        for (uint32_t pi = phases->child; pi != RIA_JSON_NONE; pi = inventory->nodes[pi].next) {
            unsigned phase = RIA_PHASES;
            for (unsigned k = 0; k < RIA_PHASES; k++)
                if (exact(inventory, pi, ria_phase_names[k]))
                    phase = k;
            if (phase == RIA_PHASES || phase_seen[phase])
                return ria_fail(e, RIA_INVALID_REQUEST, "unknown or repeated allocation phase");
            phase_seen[phase] = true;
            ria_memory_peak *sum = &plan->phases[phase];
            if (!add(host ? &sum->host : &sum->device, bytes, e) ||
                (pinned && !add(&sum->pinned, bytes, e)) ||
                (node >= 0 && !add(&sum->numa[node], bytes, e)))
                return false;
            if (protected_progress) {
                if (host)
                    progress_host[phase] = true;
                else
                    progress_device[phase] = true;
            }
        }
        if (!protected_progress && !add(&nonprogress, bytes, e))
            return false;
    }
    if (!nonprogress)
        return ria_fail(e, RIA_INVALID_REQUEST,
                        "inventory contains no required execution allocations");
    for (unsigned k = 0; k < RIA_PHASES; k++) {
        ria_memory_peak *p = &plan->phases[k];
        if ((p->host && !progress_host[k]) || (p->device && !progress_device[k]))
            return ria_fail(e, RIA_RESOURCE_LIMIT, "phase %s has no protected progress allocation",
                            ria_phase_names[k]);
        if (!under(p, &plan->caps, ria_phase_names[k], e))
            return false;
        if (p->host > plan->peak.host)
            plan->peak.host = p->host;
        if (p->device > plan->peak.device)
            plan->peak.device = p->device;
        if (p->pinned > plan->peak.pinned)
            plan->peak.pinned = p->pinned;
        for (unsigned j = 0; j < RIA_NUMA_MAX; j++)
            if (p->numa[j] > plan->peak.numa[j])
                plan->peak.numa[j] = p->numa[j];
    }
    return true;
}
typedef struct {
    char *bytes;
    size_t used, cap;
    ria_error *error;
} output;
static bool print(output *o, const char *format, ...) RIA_PRINTF(2, 3);
static bool print(output *o, const char *format, ...)
{
    va_list args;
    va_start(args, format);
    int n = vsnprintf(o->bytes + o->used, o->cap - o->used, format, args);
    va_end(args);
    if (n < 0 || (size_t)n >= o->cap - o->used)
        return ria_fail(o->error, RIA_RESOURCE_LIMIT, "memory plan output bound exceeded");
    o->used += (size_t)n;
    return true;
}
static bool memory_json(output *o, const ria_memory_peak *p)
{
    if (!print(o, "{\"host_bytes\":%llu,\"device_bytes\":%llu,\"pinned_bytes\":%llu,\"numa\":[",
               (unsigned long long)p->host, (unsigned long long)p->device,
               (unsigned long long)p->pinned))
        return false;
    bool first = true;
    for (unsigned n = 0; n < RIA_NUMA_MAX; n++)
        if (p->numa[n]) {
            if (!print(o, "%s{\"node\":%u,\"bytes\":%llu}", first ? "" : ",", n,
                       (unsigned long long)p->numa[n]))
                return false;
            first = false;
        }
    return print(o, "]}");
}
bool ria_admission_json(const ria_admission_plan *p, char **json, size_t *length, ria_error *e)
{
    if (!p || !json || !length)
        return ria_fail(e, RIA_INVALID_REQUEST, "missing plan serialization target");
    output o = {malloc(65536), 0, 65536, e};
    if (!o.bytes)
        return ria_fail(e, RIA_RESOURCE_LIMIT, "plan output allocation failed");
    char logical[65], op[65], req[65], inv[65], probe[65], cal[65], environment[65],build[65],policy[65];
    ria_hex_encode(p->logical_model_digest, 32, logical);
    ria_hex_encode(p->operator_contract_digest, 32, op);
    ria_hex_encode(p->request_digest, 32, req);
    ria_hex_encode(p->inventory_digest, 32, inv);
    ria_hex_encode(p->probe_digest, 32, probe);
    ria_hex_encode(p->calibration_digest, 32, cal);
    ria_hex_encode(p->environment_digest, 32, environment);
    ria_hex_encode(p->build_digest, 32, build);
    ria_hex_encode(p->policy_digest, 32, policy);
    bool ok =
        print(&o,
              "{\"schema_revision\":1,\"admitted\":true,\"role\":\"%s\",\"executor\":\"%s\","
              "\"profile\":\"%s\",\"logical_model_digest\":\"%s\",\"operator_contract_digest\":\"%"
              "s\",\"request_digest\":\"%s\",\"inventory_digest\":\"%s\",\"probe_digest\":\"%s\","
              "\"calibration_digest\":\"%s\",\"context_positions\":%llu,\"prefill_rows\":%llu,\"allocation_count\":%llu,"
              "\"caps\":",
              p->role, p->executor, p->profile, logical, op, req, inv, probe, cal,
              (unsigned long long)p->context_positions, (unsigned long long)p->prefill_rows,
              (unsigned long long)p->allocation_count) &&
        memory_json(&o, &p->caps) && print(&o, ",\"peak\":") && memory_json(&o, &p->peak) &&
        print(&o, ",\"phases\":{");
    for (unsigned k = 0; k < RIA_PHASES && ok; k++)
        ok = print(&o, "%s\"%s\":", k ? "," : "", ria_phase_names[k]) &&
             memory_json(&o, &p->phases[k]);
    if (ok)
        ok = print(&o, "},\"environment_digest\":\"%s\",\"build_digest\":\"%s\",\"policy_digest\":\"%s\"}",environment,build,policy);
    ria_json_doc d = {0};
    ria_json_limits limits = {65536, 8192, 32};
    uint8_t hash[32];
    if (ok)
        ok = ria_json_parse(o.bytes, o.used, limits, &d, e) && ria_json_sha256(&d, true, hash, e);
    ria_json_free(&d);
    if (ok) {
        char digest[65];
        ria_hex_encode(hash, 32, digest);
        o.used--;
        ok = print(&o, ",\"digest\":\"%s\"}\n", digest);
    }
    if (!ok) {
        free(o.bytes);
        return false;
    }
    *json = o.bytes;
    *length = o.used;
    return true;
}
static bool atomic_file(const char *path, const void *bytes, size_t n, ria_error *e)
{
    size_t length = strlen(path);
    if (length > 4096)
        return ria_fail(e, RIA_INVALID_REQUEST, "output path exceeds bound");
    char *temporary = malloc(length + 16);
    char *directory = strdup(path);
    if (!temporary || !directory) {
        free(temporary);
        free(directory);
        return ria_fail(e, RIA_RESOURCE_LIMIT, "publication allocation failed");
    }
    snprintf(temporary, length + 16, "%s.tmp.XXXXXX", path);
    int fd = mkstemp(temporary);
    if (fd < 0) {
        free(temporary);
        free(directory);
        return ria_fail(e, RIA_INTERNAL_ERROR, "cannot create plan temporary file");
    }
    bool ok = fchmod(fd, 0644) == 0;
    size_t done = 0;
    while (ok && done < n) {
        ssize_t wrote = write(fd, (const char *)bytes + done, n - done);
        if (wrote < 0 && errno == EINTR)
            continue;
        if (wrote <= 0) {
            ok = false;
            break;
        }
        done += (size_t)wrote;
    }
    if (ok)
        ok = fsync(fd) == 0;
    if (close(fd) != 0)
        ok = false;
    if (ok)
        ok = rename(temporary, path) == 0;
    if (ok) {
        char *slash = strrchr(directory, '/');
        if (slash) {
            if (slash == directory)
                slash[1] = 0;
            else
                *slash = 0;
        } else
            strcpy(directory, ".");
        int dir = open(directory, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (dir < 0)
            ok = false;
        else {
            if (fsync(dir) != 0)
                ok = false;
            if (close(dir) != 0)
                ok = false;
        }
    }
    if (!ok)
        unlink(temporary);
    free(temporary);
    free(directory);
    if (!ok)
        return ria_fail(e, RIA_INTERNAL_ERROR, "memory plan publication failed");
    return true;
}
bool ria_admission_files(const char *request, const char *inventory, const char *probe,
                         const char *calibration, const char *path, ria_error *e)
{
    ria_json_doc r = {0}, i = {0}, p = {0}, c = {0};
    ria_json_limits limits = {16777216, 200000, 64};
    ria_admission_plan plan;
    char *json = NULL;
    size_t length = 0;
    bool ok = ria_json_read(request, limits, &r, e) && ria_json_read(inventory, limits, &i, e) &&
              ria_json_read(probe, limits, &p, e) && ria_json_read(calibration, limits, &c, e) &&
              ria_admission_compute(&r, &i, &p, &c, &plan, e) &&
              ria_admission_json(&plan, &json, &length, e) && atomic_file(path, json, length, e);
    free(json);
    ria_json_free(&r);
    ria_json_free(&i);
    ria_json_free(&p);
    ria_json_free(&c);
    return ok;
}
