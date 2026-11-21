#pragma once

#include "tensor.h"

namespace haki {
    template <std::floating_point T> struct LayerKVCache {
        Tensor<T, 2> keys;
        Tensor<T, 2> values;
    };

    template <std::floating_point T> struct KVCache {
        std::vector<LayerKVCache<T>> layers;
        size_t len = 0;
        size_t capacity;
    };

}
