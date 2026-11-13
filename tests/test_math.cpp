#include "../inc/tensor.h"
#include "../inc/tmath.h"
#include "test_helpers.h"

void test_mat_mul() {
    haki::Tensor<float, 2> a { { 2, 3 } };
    haki::Tensor<float, 2> b { { 3, 4 } };

    haki::Tensor<float, 2> expected({ 2, 4 });
    haki::Tensor<float, 2> out { { 2, 4 } };

    a.storage = { 3.366903663e-01f, 1.288094074e-01f, 2.344623655e-01f, 2.303330302e-01f,
        -1.122856379e+00f, -1.863282919e-01f };
    b.storage = { 2.208201408e+00f, -6.379970312e-01f, 4.616572261e-01f, 2.673508823e-01f,
        5.349046588e-01f, 8.093572259e-01f, 1.110290289e+00f, -1.689798951e+00f, -9.889599085e-01f,
        9.579718113e-01f, 1.322135091e+00f, 8.171897531e-01f };
    expected.storage = { 5.805070400e-01f, 1.140537038e-01f, 6.084423065e-01f, 6.395269930e-02f,
        9.227180481e-02f, -1.234241009e+00f, -1.386712790e+00f, 1.806715727e+00f };

    haki::mat_mul(a.cview(), b.cview(), out.view());

    expect_close(out.cview(), expected.cview(), 1e-5f, 1e-4f);
}

void test_mat_mul_transposed_rhs() {
    haki::Tensor<float, 2> a { { 2, 3 } };
    haki::Tensor<float, 2> b { { 4, 3 } };

    haki::Tensor<float, 2> expected({ 2, 4 });
    haki::Tensor<float, 2> out { { 2, 4 } };

    a.storage = { 3.366903663e-01f, 1.288094074e-01f, 2.344623655e-01f, 2.303330302e-01f,
        -1.122856379e+00f, -1.863282919e-01f };
    b.storage = { 2.208201408e+00f, -6.379970312e-01f, 4.616572261e-01f, 2.673508823e-01f,
        5.349046588e-01f, 8.093572259e-01f, 1.110290289e+00f, -1.689798951e+00f, -9.889599085e-01f,
        9.579718113e-01f, 1.322135091e+00f, 8.171897531e-01f };
    expected.storage = { 7.695413828e-01f, 3.486790359e-01f, -7.571186125e-02f, 6.844435334e-01f,
        1.138980985e+00f, -6.898474693e-01f, 2.337409496e+00f, -1.416180730e+00f };

    haki::mat_mul_transposed_rhs(a.cview(), b.cview(), out.view());

    expect_close(out.cview(), expected.cview(), 1e-5f, 1e-4f);
}

void test_silu() {
    haki::Tensor<float, 1> x { { 5 } };

    haki::Tensor<float, 1> expected({ 5 });
    haki::Tensor<float, 1> out { { 5 } };

    x.storage = { -1.000000000e+01f, -1.000000000e+00f, 0.000000000e+00f, 1.000000000e+00f,
        1.000000000e+01f };
    expected.storage = { -4.539787187e-04f, -2.689414322e-01f, 0.000000000e+00f, 7.310585976e-01f,
        9.999546051e+00f };

    haki::silu(x.cview(), out.view());
    expect_close(out.cview(), expected.cview(), 1e-5f, 1e-4f);

    // in place
    haki::silu(x.cview(), x.view());
    expect_close(x.cview(), expected.cview(), 1e-5f, 1e-4f);
}
