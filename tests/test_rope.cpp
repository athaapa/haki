#include "reference_helpers.h"

void test_rope_against_pytorch() {
    auto q = test_data::tensor<2>({3, 16}, fixtures::q);
    auto k = test_data::tensor<2>({3, 8}, fixtures::k);
    auto eq = test_data::tensor<2>({3, 16}, fixtures::rope_q);
    auto ek = test_data::tensor<2>({3, 8}, fixtures::rope_k);
    haki::Tensor<float, 2> oq({3, 16}), ok({3, 8});
    haki::rope(q.cview(), k.cview(), test_data::config(), oq.view(), ok.view());
    expect_close(oq.cview(), eq.cview(), 1e-5f, 1e-4f);
    expect_close(ok.cview(), ek.cview(), 1e-5f, 1e-4f);
}
