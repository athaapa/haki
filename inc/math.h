#pragma once
#include "tensor.h"
#include <cmath>
#include <concepts>

namespace haki::math {

    template <std::floating_point T> T rms(haki::tensor::ConstVectorView<T> x, T epsilon) {
        T total = 0;
        int n = x.size();
        for (size_t i = 0; i < n; ++i) {
            total += (x[i] * x[i]);
        }

        return std::sqrt(epsilon + total / n);
    }

    template <std::floating_point T> T sigmoid(T x) { return 1 / (1 + std::exp(-x)); }

    template <std::floating_point T> T silu(T x, T beta) { return x * sigmoid(beta * x); }

    template <std::floating_point T>
    T rms_norm(haki::tensor::ConstVectorView<T> x, T epsilon,
        haki::tensor::ConstVectorView<T> parameters, const haki::tensor::TensorView<T, 1> out) {
        int n = out.size();
        T res = rms(x, epsilon);
        for (int i = 0; i < n; ++i) {
            out.data[i] = parameters[i] * (x.data[i] / res);
        }
    }
}
