#ifndef RIA_INVENTORY_H
#define RIA_INVENTORY_H
#include "tensor.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Metadata-only admission producer. Requests bind a trusted manifest and
 * explicit runtime reservations; populations, aliases, page extents and NUMA
 * replicas use the serving loader/accountant. No weights or CUDA are loaded.
 * The output is conservative for all phases and is not physical qualification. */
bool ria_inventory_files(const char *,const char *,const char *,ria_error *);
#ifdef __cplusplus
}
#endif
#endif
