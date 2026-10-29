#include "../inc/haki.h"

void test_softmax() {
    haki::Tensor<double, 1> logits({ 3 });
    logits.storage = { 2.0, 1.0, 0.1 };

    haki::Tensor<double, 1> probabilities({ 3 });
    probabilities.storage = { 0, 0, 0 };

    auto a = logits.cview();
    auto out = probabilities.view();

    haki::softmax(a, out);
    for (auto& x : probabilities.storage) {
        std::cout << x << " ";
    }
    std::cout << '\n';
}

void test_mat_mul() {
    haki::Tensor<double, 2> a({ 3, 3 });
    a.storage = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };

    haki::Tensor<double, 2> b({ 3, 3 });
    b.storage = { 2, 3, 4, 5, 6, 7, 8, 9, 10 };

    haki::Tensor<double, 2> c({ 3, 3 });

    haki::mat_mul(a.cview(), b.cview(), c.view());

    for (int i = 0; i < 9; i++) {
        std::cout << c.storage[i] << " ";
    }
    std::cout << '\n';
}

int main() {
    // test_softmax();
    // test_mat_mul();
    return 0;
}
