#include "reference_helpers.h"

// Contract: output contains only this head, [num_tokens, head_dim].
// Inputs contain all packed heads. Select nonzero head indices deliberately.
void test_attention_head_against_pytorch() {
    auto q = test_data::tensor<2>({3, 16}, fixtures::q);
    auto k = test_data::tensor<2>({3, 8}, fixtures::k);
    auto v = test_data::tensor<2>({3, 8}, fixtures::v);
    auto expected = test_data::tensor<2>({3, 4}, fixtures::head_expected);
    haki::Tensor<float, 2> out({3, 4});
    std::fill(out.storage.begin(), out.storage.end(), 99.0f);
    haki::self_attention_head(q.cview(), k.cview(), v.cview(), 3, 1, test_data::config(), out.view());
    expect_close(out.cview(), expected.cview(), 1e-5f, 1e-4f);
}

void test_attention_head_uniform_scores() {
    haki::Tensor<float, 2> q({3, 16}), k({3, 8}), out({3, 4});
    auto v = test_data::tensor<2>({3, 8}, fixtures::v);
    auto expected = test_data::tensor<2>({3, 4}, fixtures::uniform_expected);
    haki::self_attention_head(q.cview(), k.cview(), v.cview(), 3, 1, test_data::config(), out.view());
    expect_close(out.cview(), expected.cview(), 1e-5f, 1e-4f);
}
