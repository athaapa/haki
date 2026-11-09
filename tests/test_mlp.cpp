#include "reference_helpers.h"

void test_swiglu_mlp_against_pytorch() {
    auto input = test_data::tensor<2>({3, 6}, fixtures::input);
    auto expected = test_data::tensor<2>({3, 6}, fixtures::mlp_expected);
    auto weights = test_data::mlp_weights();
    haki::Tensor<float, 2> out({3, 6});
    haki::swiglu_mlp(input.cview(), weights, out.view());
    expect_close(out.cview(), expected.cview(), 1e-5f, 1e-4f);
}

void test_swiglu_zero_gate() {
    auto input = test_data::tensor<2>({3, 6}, fixtures::input);
    auto weights = test_data::mlp_weights();
    std::fill(weights.gate_proj.storage.begin(), weights.gate_proj.storage.end(), 0.0f);
    haki::Tensor<float, 2> out({3, 6}), expected({3, 6});
    std::fill(out.storage.begin(), out.storage.end(), 99.0f);
    haki::swiglu_mlp(input.cview(), weights, out.view());
    expect_close(out.cview(), expected.cview(), 1e-5f, 1e-4f);
}
