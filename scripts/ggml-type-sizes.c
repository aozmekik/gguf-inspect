/* Prints "name block_size type_size" for every quantized ggml block struct.
 * Compiled against ggml's src/ggml-common.h by check-ggml-type-sizes.sh. */
#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include <stdio.h>

#define P(name, blk, T) printf("%s %d %zu\n", name, (int)(blk), sizeof(T))

int main(void) {
    P("q4_0", QK4_0, block_q4_0);
    P("q4_1", QK4_1, block_q4_1);
    P("q5_0", QK5_0, block_q5_0);
    P("q5_1", QK5_1, block_q5_1);
    P("q8_0", QK8_0, block_q8_0);
    P("q8_1", QK8_1, block_q8_1);
    P("q2_K", QK_K, block_q2_K);
    P("q3_K", QK_K, block_q3_K);
    P("q4_K", QK_K, block_q4_K);
    P("q5_K", QK_K, block_q5_K);
    P("q6_K", QK_K, block_q6_K);
    P("q8_K", QK_K, block_q8_K);
    P("iq2_xxs", QK_K, block_iq2_xxs);
    P("iq2_xs", QK_K, block_iq2_xs);
    P("iq3_xxs", QK_K, block_iq3_xxs);
    P("iq1_s", QK_K, block_iq1_s);
    P("iq4_nl", QK4_NL, block_iq4_nl);
    P("iq3_s", QK_K, block_iq3_s);
    P("iq2_s", QK_K, block_iq2_s);
    P("iq4_xs", QK_K, block_iq4_xs);
    P("iq1_m", QK_K, block_iq1_m);
    P("tq1_0", QK_K, block_tq1_0);
    P("tq2_0", QK_K, block_tq2_0);
    P("mxfp4", QK_MXFP4, block_mxfp4);
    P("nvfp4", QK_NVFP4, block_nvfp4);
    P("q1_0", QK1_0, block_q1_0);
    P("q2_0", QK2_0, block_q2_0);
    return 0;
}
