#pragma once
#include "tensor.h"
#include <algorithm>
#include <cmath>
#include <concepts>

namespace haki {

    template <std::floating_point T> T sigmoid(T x) { return 1 / (1 + std::exp(-x)); }

    template <std::floating_point T> T silu(T x) { return x * sigmoid(x); }

    template <std::floating_point T> T silu(ConstVectorView<T> x, TensorView<T, 1>& out) {
        assert(x.shape == out.shape);

        std::transform(x.data, x.data + x.size(), out.data, [](T x) { return silu(x); });
    }

}
