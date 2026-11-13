#include "../inc/nn.h"
#include "test_helpers.h"

void test_dot() {
    haki::Tensor<float, 1> a({ 4 }), b({ 4 });
    a.storage = { 1, -2, 3, 0.5f };
    b.storage = { 4, 5, -1, 2 };
    if (haki::dot(a.cview(), b.cview()) != -8.0f)
        throw std::runtime_error("dot: expected -8");
}
void test_mat_vec() {
    haki::Tensor<float, 2> a({ 2, 3 });
    haki::Tensor<float, 1> b({ 3 }), out({ 2 }), expected({ 2 });
    a.storage = { 1, 2, 3, 4, 5, 6 };
    b.storage = { -1, 2, 0.5f };
    expected.storage = { 4.5f, 9 };
    haki::mat_mul(a.cview(), b.cview(), out.view());
    expect_close(out.cview(), expected.cview(), 1e-5f, 1e-4f);
}
void test_element_mul() {
    haki::Tensor<float, 1> a({ 3 }), b({ 3 }), out({ 3 }), expected({ 3 });
    a.storage = { 1, -2, 0.5f };
    b.storage = { 3, 2, -1 };
    expected.storage = { 3, -4, -0.5f };
    haki::element_mul(a.cview(), b.cview(), out.view());
    expect_close(out.cview(), expected.cview(), 1e-5f, 1e-4f);
}
