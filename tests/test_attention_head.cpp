#include "reference_helpers.h"

// Contract: output contains only this head, [num_tokens, head_dim].
// Select nonzero heads from packed storage to exercise strided input views.
void test_attention_head_against_pytorch() {
    auto q = test_data::tensor<2>({ 3, 16 }, fixtures::q);
    auto k = test_data::tensor<2>({ 3, 8 }, fixtures::k);
    auto v = test_data::tensor<2>({ 3, 8 }, fixtures::v);
    auto expected = test_data::tensor<2>({ 3, 4 }, fixtures::head_expected);
    haki::Tensor<float, 2> out({ 3, 4 });
    std::fill(out.storage.begin(), out.storage.end(), 99.0f);
    haki::self_attention_head(q.cview().narrow(1, 12, 4), k.cview().narrow(1, 4, 4),
        v.cview().narrow(1, 4, 4), test_data::config(), out.view());
    expect_close(out.cview(), expected.cview(), 1e-5f, 1e-4f);
}

void test_attention_head_uniform_scores() {
    haki::Tensor<float, 2> q({ 3, 16 }), k({ 3, 8 }), out({ 3, 4 });
    auto v = test_data::tensor<2>({ 3, 8 }, fixtures::v);
    auto expected = test_data::tensor<2>({ 3, 4 }, fixtures::uniform_expected);
    haki::self_attention_head(q.cview().narrow(1, 12, 4), k.cview().narrow(1, 4, 4),
        v.cview().narrow(1, 4, 4), test_data::config(), out.view());
    expect_close(out.cview(), expected.cview(), 1e-5f, 1e-4f);
}

// Zero logits must give a uniform distribution over allowed source tokens only.
void test_attention_weights_causal_mask() {
    haki::Tensor<float, 2> q({ 3, 16 }), k({ 3, 8 }), out({ 3, 3 });
    haki::Tensor<float, 2> expected({ 3, 3 });
    expected.storage = { 1, 0, 0, 0.5f, 0.5f, 0, 1.0f / 3, 1.0f / 3, 1.0f / 3 };
    haki::compute_attention_weights(
        q.cview().narrow(1, 12, 4), k.cview().narrow(1, 4, 4), out.view());
    expect_close(out.cview(), expected.cview(), 1e-6f, 1e-5f);
}
