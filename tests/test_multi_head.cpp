#include "reference_helpers.h"

void test_multi_head_attention_against_pytorch() {
    auto input = test_data::tensor<2>({3, 6}, fixtures::input);
    auto expected = test_data::tensor<2>({3, 6}, fixtures::attention_expected);
    auto weights = test_data::attention_weights();
    haki::Tensor<float, 2> out({3, 6});
    haki::multi_head_attention(input.cview(), weights, test_data::config(), out.view());
    expect_close(out.cview(), expected.cview(), 1e-5f, 1e-4f);
}

void test_multi_head_attention_causality() {
    auto input = test_data::tensor<2>({3, 6}, fixtures::input);
    auto expected = test_data::tensor<2>({3, 6}, fixtures::attention_expected);
    auto weights = test_data::attention_weights();
    // Changing only the future token must preserve both earlier output rows.
    for (size_t d = 0; d < 6; ++d) input.storage[12 + d] += 5.0f;
    haki::Tensor<float, 2> out({3, 6});
    haki::multi_head_attention(input.cview(), weights, test_data::config(), out.view());
    for (size_t i = 0; i < 2; ++i)
        expect_close(out.cview().row(i), expected.cview().row(i), 1e-5f, 1e-4f);
}
