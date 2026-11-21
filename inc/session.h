#pragma once

#include "kv_cache.h"
#include "model.h"
#include "nn.h"
#include "tensor.h"
#include <concepts>
#include <span>

namespace haki {

    template <std::floating_point T> class InferenceSession {
    public:
        InferenceSession(const Qwen3<T>& model, size_t capacity)
            : model { model }
            , cache { .layers = { }, .len = 0, .capacity = capacity } {
            const auto& config = model.config;
            const size_t kv_width = config.num_key_value_heads * config.head_dim;

            cache.layers.reserve(config.num_hidden_layers);

            for (size_t i = 0; i < config.num_hidden_layers; ++i) {
                cache.layers.push_back({
                    .keys = Tensor<T, 2>({ capacity, kv_width }),
                    .values = Tensor<T, 2>({ capacity, kv_width }),
                });
            }
        }

        void reset() { cache.len = 0; }

        [[nodiscard]] size_t size() const { return cache.len; }

        void compute_logits(std::span<const int>& input, TensorView<T, 1>& out) {
            compute_logits_with_cache(model, input, cache, out);
        }

    private:
        const Qwen3<T>& model;
        KVCache<T> cache;
    };
}
