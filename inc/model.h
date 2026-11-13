#pragma once
#include "tensor.h"
#include <concepts>
#include <vector>

namespace haki {

    template <std::floating_point T> struct AttentionWeights {
        Tensor<T, 2> q_proj, k_proj, v_proj, o_proj;
        Tensor<T, 1> q_norm, k_norm;
    };

    template <std::floating_point T> struct MlpWeights {
        Tensor<T, 2> gate_proj, up_proj, down_proj;
    };

    template <std::floating_point T> struct Qwen3LayerWeights {
        Tensor<T, 1> pre_attention_norm;
        Tensor<T, 1> pre_mlp_norm;

        AttentionWeights<T> attention;
        MlpWeights<T> mlp;
    };

    template <std::floating_point T> struct Qwen3Weights {
        Tensor<T, 2> token_embedding;
        std::vector<Qwen3LayerWeights<T>> layers;
        Tensor<T, 1> final_norm;
    };

    struct Qwen3Config {
        size_t hidden_dim;
        size_t intermediate_dim;
        size_t head_dim;
        size_t num_attention_heads;
        size_t num_key_value_heads;
        size_t num_hidden_layers;
        double rms_norm_eps;
        double rope_theta;
    };

    template <std::floating_point T> struct Qwen3 {
        Qwen3Config config;
        Qwen3Weights<T> weights;
    };
}
