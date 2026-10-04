#ifndef RIA_QUALIFY_TRANSPORT_H
#define RIA_QUALIFY_TRANSPORT_H
#include "common.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Model-free, bounded paired mTLS fixture. Both peers use separately authorized
 * requests sharing the logical/source/operator/policy/registration identities.
 * The owned canonical report is raw evidence with qualified=false. No model,
 * admitted memory plan or model service is read. Errors publish no success
 * report. This call owns all connections and completes their teardown. */
bool ria_qualify_transport(const char *config_path, const char *request_path,
                           char **report_json, size_t *report_length,
                           ria_error *error);
#ifdef __cplusplus
}
#endif
#endif
