#pragma once
#include "../inc/tensor.h"
#include <concepts>
#include <cstdlib>
#include <sstream>
#include <stdexcept>

template <std::floating_point T, size_t Rank>
void expect_close(haki::TensorView<const T, Rank> actual, haki::TensorView<const T, Rank> expected,
    T atol, T rtol) {
    if (actual.shape != expected.shape) {
        throw std::runtime_error("Tensor shapes differ");
    }

    for (size_t i = 0; i < actual.size(); ++i) {
        const T a = actual.data[i];
        const T e = expected.data[i];
        const T error = std::abs(a - e);
        const T tolerance = atol + rtol * std::abs(e);

        if (!std::isfinite(a) || !std::isfinite(e) || error > tolerance) {
            std::ostringstream message;
            message << "Mismatch at flat index " << i << ": actual=" << a << ", expected=" << e
                    << ", error=" << error << ", tolerance=" << tolerance;
            throw std::runtime_error(message.str());
        }
    }
}
