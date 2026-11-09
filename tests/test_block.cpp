#include "reference_helpers.h"

void test_transformer_block_against_pytorch() {
    auto input = test_data::tensor<2>({3, 6}, fixtures::input);
    auto expected = test_data::tensor<2>({3, 6}, fixtures::block_expected);
    auto model = test_data::model();
    haki::Tensor<float, 2> out({3, 6});
    haki::transformer_block(model, 0, input.cview(), out.view());
    expect_close(out.cview(), expected.cview(), 1e-5f, 1e-4f);
}

void test_transformer_block_zero_branches() {
    auto input = test_data::tensor<2>({3, 6}, fixtures::input);
    auto model = test_data::model();
    auto& w = model.weights.layers[0];
    std::fill(w.attention.o_proj.storage.begin(), w.attention.o_proj.storage.end(), 0.0f);
    std::fill(w.mlp.down_proj.storage.begin(), w.mlp.down_proj.storage.end(), 0.0f);
    haki::Tensor<float, 2> out({3, 6});
    std::fill(out.storage.begin(), out.storage.end(), 99.0f);
    haki::transformer_block(model, 0, input.cview(), out.view());
    expect_close(out.cview(), input.cview(), 1e-5f, 1e-4f);
}
