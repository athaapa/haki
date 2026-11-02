#pragma once
#include "tensor.h"
#include <cmath>
#include <concepts>

namespace haki {

    template <std::floating_point T> T sigmoid(T x) { return 1 / (1 + std::exp(-x)); }

    template <std::floating_point T> T silu(T x, T beta) { return x * sigmoid(beta * x); }

}
