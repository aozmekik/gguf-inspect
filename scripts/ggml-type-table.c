/* Prints ggml's own type table, one line per type ID:
 *
 *     <id> <name> <block size> <block bytes> <yes|no>   for types in use
 *     <id> removed                                        for removed IDs
 *
 * check-ggml-type-table.sh compiles this against include/ggml.h and
 * src/ggml-common.h at a ggml commit, with the type_traits initializer
 * copied out of src/ggml.c into traits.inc (without its function pointers).
 * IDs come from the designated initializers, so this checks the enum values,
 * names and sizes together. */
#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml.h"
#include <stdio.h>

static const struct ggml_type_traits type_traits[GGML_TYPE_COUNT] = {
#include "traits.inc"
};

int main(void) {
    for (int id = 0; id < GGML_TYPE_COUNT; id++) {
        const struct ggml_type_traits *t = &type_traits[id];
        if (t->blck_size == 0) {
            printf("%d removed\n", id);
        } else {
            printf("%d %s %lld %zu %s\n", id, t->type_name, (long long)t->blck_size,
                   t->type_size, t->is_quantized ? "yes" : "no");
        }
    }
    return 0;
}
