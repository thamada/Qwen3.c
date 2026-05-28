// Host-only: compare compute_sf_index vs CuTe layout_SFA(r, k*16, 0)
#include <cstdio>
#include <cstdlib>

#include "cute/tensor.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"

using namespace cute;
using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<16>;
static constexpr int SF_VEC_SIZE = Sm1xxBlkScaledConfig::SFVecSize;

static int compute_sf_index(int r, int k_block, int /*rows*/, int nsb) {
    int r0 = r % 32;
    int r1 = (r / 32) % 4;
    int r2 = r / 128;
    int k1 = k_block % 4;
    int k2 = k_block / 4;
    int k_tiles = nsb / 4;
    int row_tile_stride = 512 * k_tiles;
    int k_tile_stride = 512;
    return r0 * 16 + r1 * 4 + r2 * row_tile_stride + k1 * 1 + k2 * k_tile_stride;
}

static int verify_case(int M, int N, int K, const char *label) {
    int nsb = K / SF_VEC_SIZE;
    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
    size_t sfa_elems = size(filter_zeros(layout_SFA));

    int mismatches = 0;
    int first_r = -1, first_sb = -1, first_a = -1, first_b = -1;

    for (int r = 0; r < M; r++) {
        for (int sb = 0; sb < nsb; sb++) {
            int k_start = sb * SF_VEC_SIZE;
            int a = compute_sf_index(r, sb, M, nsb);
            int b = static_cast<int>(layout_SFA(r, k_start, 0));
            if (a != b) {
                mismatches++;
                if (first_r < 0) {
                    first_r = r;
                    first_sb = sb;
                    first_a = a;
                    first_b = b;
                }
            }
            if (a < 0 || (size_t)a >= sfa_elems) {
                fprintf(stderr, "%s: OOB compute idx r=%d sb=%d -> %d (elems=%zu)\n",
                        label, r, sb, a, sfa_elems);
                return -1;
            }
            if (b < 0 || (size_t)b >= sfa_elems) {
                fprintf(stderr, "%s: OOB layout idx r=%d sb=%d -> %d (elems=%zu)\n",
                        label, r, sb, b, sfa_elems);
                return -1;
            }
        }
    }

    int *used = (int *)calloc(sfa_elems, sizeof(int));
    int collisions = 0;
    for (int r = 0; r < M; r++) {
        for (int sb = 0; sb < nsb; sb++) {
            int idx = compute_sf_index(r, sb, M, nsb);
            if (used[idx]++) collisions++;
        }
    }
    free(used);

    int expected = M * nsb;
    int unique_ok = (collisions == 0 && expected == (int)sfa_elems);

    printf("%s M=%d N=%d K=%d nsb=%d sfa_elems=%zu pairs=%d mismatches=%d collisions=%d %s\n",
           label, M, N, K, nsb, sfa_elems, expected, mismatches, collisions,
           mismatches == 0 ? "INDEX_OK" : "INDEX_FAIL");
    if (mismatches)
        printf("  first mismatch: r=%d sb=%d compute=%d layout=%d\n",
               first_r, first_sb, first_a, first_b);
    if (!unique_ok)
        printf("  WARNING: expected pairs=%d vs sfa_elems=%zu collisions=%d\n",
               expected, sfa_elems, collisions);

    return mismatches ? 1 : 0;
}

static int verify_sfb_case(int M_dummy, int N, int K, const char *label) {
    int nsb = K / SF_VEC_SIZE;
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M_dummy, N, K, 1));
    size_t sfb_elems = size(filter_zeros(layout_SFB));
    int mismatches = 0;
    for (int r = 0; r < N; r++) {
        for (int sb = 0; sb < nsb; sb++) {
            int k_start = sb * SF_VEC_SIZE;
            int a = compute_sf_index(r, sb, N, nsb);
            int b = static_cast<int>(layout_SFB(r, k_start, 0));
            if (a != b) mismatches++;
        }
    }
    printf("SFB %s M_dummy=%d N=%d K=%d sfb_elems=%zu pairs=%d mismatches=%d %s\n",
           label, M_dummy, N, K, sfb_elems, N * nsb, mismatches,
           mismatches == 0 ? "INDEX_OK" : "INDEX_FAIL");
    return mismatches ? 1 : 0;
}

int main() {
    struct Case { int M, N, K; const char *label; } cases[] = {
        {128, 128, 128, "square"},
        {128, 1024, 4096, "wk_decode"},
        {256, 4096, 4096, "wq_prefill_M256"},
        {256, 12288, 4096, "gate_prefill_M256"},
        {128, 4096, 4096, "wq_decode_M128"},
    };
    int fail = 0;
    printf("=== SFA (activation) ===\n");
    for (auto &c : cases)
        fail |= verify_case(c.M, c.N, c.K, c.label);
    printf("=== SFB (weight) ===\n");
    fail |= verify_sfb_case(128, 1024, 4096, "wk_pack_M128");
    fail |= verify_sfb_case(256, 1024, 4096, "wk_runtime_M256");
    fail |= verify_sfb_case(128, 4096, 4096, "wq_pack_M128");
    fail |= verify_sfb_case(256, 4096, 4096, "wq_runtime_M256");
    return fail ? 1 : 0;
}
