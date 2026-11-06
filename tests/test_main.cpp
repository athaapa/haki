#include <exception>
#include <iostream>

void test_mat_mul();
void test_mat_mul_transposed_rhs();
void test_rms_norm();
void test_softmax_small();
void test_softmax_large();
void test_silu();

struct Test {
    std::string_view name;
    void (*run)();
};

int main() {
    const Test tests[] = {
        { "mat_mul", test_mat_mul },
        { "mat_mul_transposed_rhs", test_mat_mul_transposed_rhs },
        { "rms_norm", test_rms_norm },
        { "softmax_small", test_softmax_small },
        { "softmax_large", test_softmax_large },
        { "silu", test_silu },
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
