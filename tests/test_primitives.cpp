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
void test_add_in_place() {
    haki::Tensor<float, 1> a({ 3 }), b({ 3 }), expected({ 3 });
    a.storage = { 1, -2, 0.5f };
    b.storage = { 3, 2, -1 };
    expected.storage = { 4, 0, -0.5f };
    haki::add(a.cview(), b.cview(), a.view());
    expect_close(a.cview(), expected.cview(), 1e-5f, 1e-4f);
}
void test_scale_in_place() {
    haki::Tensor<float, 1> a({ 3 }), expected({ 3 });
    a.storage = { 1, -2, 0.5f };
    expected.storage = { -2, 4, -1 };
    haki::scale(a.cview(), -2.0f, a.view());
    expect_close(a.cview(), expected.cview(), 1e-5f, 1e-4f);
}
void test_element_mul() {
    haki::Tensor<float, 1> a({ 3 }), b({ 3 }), out({ 3 }), expected({ 3 });
    a.storage = { 1, -2, 0.5f };
    b.storage = { 3, 2, -1 };
    expected.storage = { 3, -4, -0.5f };
    haki::element_mul(a.cview(), b.cview(), out.view());
    expect_close(out.cview(), expected.cview(), 1e-5f, 1e-4f);
}
void test_rotate_in_place() {
    haki::Tensor<float, 1> pair({ 2 }), expected({ 2 });
    pair.storage = { 2, 3 };
    expected.storage = { -3, 2 };
    haki::rotate_vector(pair.cview(), std::acos(-1.0f) / 2, pair.view());
    expect_close(pair.cview(), expected.cview(), 1e-5f, 1e-4f);
}
