#include "../inc/nn.h"
#include "test_helpers.h"

void test_rms_norm() {
    haki::Tensor<float, 1> x { { 4 } };
    haki::Tensor<float, 1> weight { { 4 } };

    haki::Tensor<float, 1> expected { { 4 } };
    haki::Tensor<float, 1> out { { 4 } };

    x.storage = { 3.366903663e-01f, 1.288094074e-01f, 2.344623655e-01f, 2.303330302e-01f };
    weight.storage = { 5.000000000e-01f, 1.000000000e+00f, 1.500000000e+00f, 2.000000000e+00f };
    expected.storage = { 6.901738048e-01f, 5.280868411e-01f, 1.441856980e+00f, 1.888617516e+00f };

    haki::rms_norm(x.cview(), 1e-6f, weight.cview(), out.view());

    expect_close(out.cview(), expected.cview(), 1e-5f, 1e-4f);
}

void test_softmax_small() {
    haki::Tensor<float, 1> x { { 3 } };

    haki::Tensor<float, 1> expected { { 3 } };
    haki::Tensor<float, 1> out { { 3 } };

    x.storage = { 2.000000000e+00f, 1.000000000e+00f, 1.000000015e-01f };
    expected.storage = { 6.590011716e-01f, 2.424329817e-01f, 9.856589139e-02f };

    haki::softmax(x.cview(), out.view());
    expect_close(out.cview(), expected.cview(), 1e-5f, 1e-4f);
}

void test_softmax_large() {
    haki::Tensor<float, 1> x { { 3 } };

    haki::Tensor<float, 1> expected { { 3 } };
    haki::Tensor<float, 1> out { { 3 } };

    x.storage = { 1.000000000e+03f, 1.001000000e+03f, 1.002000000e+03f };
    expected.storage = { 9.003057331e-02f, 2.447284758e-01f, 6.652409434e-01f };

    haki::softmax(x.cview(), out.view());
    expect_close(out.cview(), expected.cview(), 1e-5f, 1e-4f);
}
