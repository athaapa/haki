#include <exception>
#include <iostream>
#include <string_view>

void test_attention_head_against_pytorch();
void test_attention_head_uniform_scores();
void test_transformer_block_against_pytorch();
void test_transformer_block_zero_branches();
void test_mat_mul();
void test_mat_mul_transposed_rhs();
void test_silu();
void test_swiglu_mlp_against_pytorch();
void test_swiglu_zero_gate();
void test_multi_head_attention_against_pytorch();
void test_multi_head_attention_causality();
void test_rms_norm();
void test_softmax_small();
void test_softmax_large();
void test_dot();
void test_mat_vec();
void test_element_mul();
void test_rope_against_pytorch();

struct Test {
    std::string_view name;
    void (*run)();
};

int main() {
    const Test tests[] = {
        { "attention_head_against_pytorch", test_attention_head_against_pytorch },
        { "attention_head_uniform_scores", test_attention_head_uniform_scores },
        { "transformer_block_against_pytorch", test_transformer_block_against_pytorch },
        { "transformer_block_zero_branches", test_transformer_block_zero_branches },
        { "mat_mul", test_mat_mul },
        { "mat_mul_transposed_rhs", test_mat_mul_transposed_rhs },
        { "silu", test_silu },
        { "swiglu_mlp_against_pytorch", test_swiglu_mlp_against_pytorch },
        { "swiglu_zero_gate", test_swiglu_zero_gate },
        { "multi_head_attention_against_pytorch", test_multi_head_attention_against_pytorch },
        { "multi_head_attention_causality", test_multi_head_attention_causality },
        { "rms_norm", test_rms_norm },
        { "softmax_small", test_softmax_small },
        { "softmax_large", test_softmax_large },
        { "dot", test_dot },
        { "mat_vec", test_mat_vec },
        { "mul", test_element_mul },
        { "rope_against_pytorch", test_rope_against_pytorch },
    };

    int failures = 0;

    for (const auto& test : tests) {
        try {
            test.run();
            std::cout << "[PASS] " << test.name << '\n';
        } catch (const std::exception& e) {
            ++failures;
            std::cerr << "[FAIL] " << test.name << ": " << e.what() << '\n';
        } catch (...) {
            ++failures;
            std::cerr << "[FAIL] " << test.name << ": unknown exception\n";
        }
    }

    std::cout << failures << " test(s) failed\n";
    return failures == 0 ? 0 : 1;
}
