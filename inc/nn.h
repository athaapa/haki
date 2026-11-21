#pragma once

#include "kv_cache.h"
#include "model.h"
#include "tensor.h"
#include "tmath.h"
#include <algorithm>
#include <cmath>
#include <concepts>
#include <functional>
#include <limits>
#include <span>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace haki {
    template <std::floating_point T> T rms(ConstVectorView<T> x, T epsilon) {
        assert(x.size() > 0);
        assert(std::isfinite(epsilon) && epsilon >= T { 0 });
        T total = 0;
        size_t n = x.size();
        for (size_t i = 0; i < n; ++i) {
            total += (x[i] * x[i]);
        }

        return std::sqrt(epsilon + (total / n));
    }

    template <std::floating_point T, size_t Rank>
    void rms_norm(TensorView<const T, Rank> x, T epsilon, TensorView<const T, Rank> parameters,
        const TensorView<T, Rank> out) {
        assert(x.shape == parameters.shape);
        assert(x.shape == out.shape);
        assert(std::isfinite(epsilon) && epsilon > T { 0 });
        size_t n = out.size();
        T res = rms(x, epsilon);
        assert(std::isfinite(res) && res > T { 0 });
        for (size_t i = 0; i < n; ++i) {
            out[i] = parameters[i] * (x[i] / res);
        }
    }

    template <std::floating_point T>
    void compute_attention_weights(ConstMatrixView<T> queries, ConstMatrixView<T> keys,
        const TensorView<T, 2> out, size_t kv_len = 0) {
        const size_t new_tokens = queries.shape[0];
        const size_t total_tokens = keys.shape[0];
        const size_t head_dim = queries.shape[1];

        assert(head_dim > 0);
        assert(keys.shape[1] == head_dim);
        assert(total_tokens == kv_len + new_tokens);
        assert(out.shape[0] == new_tokens);
        assert(out.shape[1] == total_tokens);

        const T scale = T { 1 } / std::sqrt(static_cast<T>(head_dim));
        Tensor<T, 1> scores({ total_tokens });

        for (size_t i = 0; i < new_tokens; ++i) {
            for (size_t j = 0; j < total_tokens; ++j) {
                scores.view()[j] = j <= kv_len + i ? dot(queries.row(i), keys.row(j)) * scale
                                                   : -std::numeric_limits<T>::infinity();
            }

            softmax(scores.cview(), out.row(i));
        }
    }

    template <std::floating_point T>
    void self_attention_head(ConstMatrixView<T> queries, ConstMatrixView<T> keys,
        ConstMatrixView<T> values, Qwen3Config config, const TensorView<T, 2> out) {
        assert(queries.shape == keys.shape);
        assert(queries.shape[1] > 0);
        const size_t num_tokens = queries.shape[0];
        assert(values.shape[0] == num_tokens);
        assert(out.shape == values.shape);

        Tensor<T, 2> weights({ num_tokens, num_tokens });
        compute_attention_weights(queries, keys, weights.view());

        for (size_t i = 0; i < num_tokens; ++i) {
            for (size_t d = 0; d < values.shape[1]; ++d) {
                T sum = 0;
                for (size_t j = 0; j <= i; ++j) {
                    sum += weights.cview()[i, j] * values[j, d];
                }
                out[i, d] = sum;
            }
        }
    }

    template <std::floating_point T>
    void self_attention_head_with_cache(ConstMatrixView<T> queries, ConstMatrixView<T> keys,
        ConstMatrixView<T> values, size_t kv_len, Qwen3Config config, const TensorView<T, 2> out) {
        assert(keys.shape[0] == kv_len + queries.shape[0]);
        assert(keys.shape[1] == queries.shape[1]);
        assert(out.shape == queries.shape);
        assert(values.shape == keys.shape);

        const size_t num_tokens = queries.shape[0];

        Tensor<T, 2> weights({ num_tokens, num_tokens + kv_len });
        compute_attention_weights(queries, keys, weights.view(), kv_len);

        for (size_t i = 0; i < num_tokens; ++i) {
            for (size_t d = 0; d < values.shape[1]; ++d) {
                T sum = 0;
                for (size_t j = 0; j <= kv_len + i; ++j) {
                    sum += weights.cview()[i, j] * values[j, d];
                }
                out[i, d] = sum;
            }
        }
    }

    template <std::floating_point T>
    void rotate_vector(ConstVectorView<T> a, T theta, const TensorView<T, 1> out) {
        assert(a.size() == 2);
        assert(out.size() == 2);
        assert(std::isfinite(theta));

        const T x = a[0];
        const T y = a[1];

        const T cos_theta = std::cos(theta);
        const T sin_theta = std::sin(theta);

        out[0] = (cos_theta * x) - (sin_theta * y);
        out[1] = (sin_theta * x) + (cos_theta * y);
    }

    template <std::floating_point T>
    void rotate_pairs(ConstMatrixView<T> vectors, size_t head_dim, T theta,
        const TensorView<T, 2> out, size_t start_position = 0) {
        assert(vectors.shape == out.shape);
        assert(head_dim > 0 && head_dim % 2 == 0);
        assert(vectors.shape[1] % head_dim == 0);
        assert(std::isfinite(theta) && theta > T { 0 });
        const size_t num_tokens = vectors.shape[0];
        const size_t num_heads = vectors.shape[1] / head_dim;

        for (size_t i = 0; i < num_tokens; ++i) {
            for (size_t h = 0; h < num_heads; ++h) {
                const size_t offset = h * head_dim;

                for (size_t k = 0; k < head_dim / 2; ++k) {

                    const T freq = static_cast<T>(start_position + i)
                        * std::pow(theta, -2 * static_cast<T>(k) / static_cast<T>(head_dim));

                    std::array rotation_pair { vectors[i, offset + k],
                        vectors[i, offset + k + (head_dim / 2)] };

                    TensorView<T, 1> query_pair { rotation_pair.data(), { 2 } };
                    TensorView<const T, 1> const_query_pair { rotation_pair.data(), { 2 } };

                    rotate_vector(const_query_pair, freq, query_pair);

                    out[i, offset + k] = rotation_pair[0];
                    out[i, offset + k + (head_dim / 2)] = rotation_pair[1];
                }
            }
        }
    }

    template <std::floating_point T>
    void rope(ConstMatrixView<T> query_heads, ConstMatrixView<T> key_heads, Qwen3Config config,
        const TensorView<T, 2> query_out, const TensorView<T, 2> key_out,
        size_t start_position = 0) {
        assert(query_heads.shape[0] == key_heads.shape[0]);
        assert(query_heads.shape == query_out.shape);
        assert(key_heads.shape == key_out.shape);
        assert(config.head_dim > 0 && config.head_dim % 2 == 0);
        assert(query_heads.shape[1] % config.head_dim == 0);
        assert(key_heads.shape[1] % config.head_dim == 0);
        assert(query_heads.shape[1] / config.head_dim == config.num_attention_heads);
        assert(key_heads.shape[1] / config.head_dim == config.num_key_value_heads);
        const size_t num_tokens = query_heads.shape[0];
        const size_t head_dim = config.head_dim;
        const T theta = config.rope_theta;

        rotate_pairs(query_heads, head_dim, theta, query_out, start_position);
        rotate_pairs(key_heads, head_dim, theta, key_out, start_position);
    }

    template <std::floating_point T>
    void qk_norm(ConstMatrixView<T> vectors, ConstVectorView<T> norm_weights, T rms_norm_eps,

        const TensorView<T, 2> out) {
        assert(vectors.shape == out.shape);
        assert(std::isfinite(rms_norm_eps) && rms_norm_eps > T { 0 });
        const size_t head_dim = norm_weights.size();
        assert(head_dim > 0);
        assert(vectors.shape[1] % head_dim == 0);

        const size_t num_heads = vectors.shape[1] / head_dim;
        const size_t num_tokens = vectors.shape[0];

        for (size_t i = 0; i < num_tokens; ++i) {
            for (size_t h = 0; h < num_heads; ++h) {
                const size_t offset = h * head_dim;

                auto vector_norm = vectors.row(i).narrow(0, offset, head_dim);
                auto out_row = out.row(i).narrow(0, offset, head_dim);
                auto vector = vector_norm.cview();

                rms_norm(vector, rms_norm_eps, norm_weights, out_row);
            }
        }
    }

    template <std::floating_point T>
    void multi_head_attention(ConstMatrixView<T> input, const AttentionWeights<T>& weights,
        const Qwen3Config config, const TensorView<T, 2> out) {
        const size_t num_tokens = input.shape[0];

        const size_t num_attention_heads = config.num_attention_heads;
        const size_t head_dim = config.head_dim;
        const size_t num_kv_heads = config.num_key_value_heads;

        assert(input.shape == out.shape);
        assert(input.shape[1] == config.hidden_dim && config.hidden_dim > 0);
        assert(head_dim > 0 && head_dim % 2 == 0);
        assert(num_attention_heads > 0 && num_kv_heads > 0);
        assert(num_attention_heads % num_kv_heads == 0);
        assert(std::isfinite(config.rms_norm_eps) && config.rms_norm_eps > 0);
        assert(std::isfinite(config.rope_theta) && config.rope_theta > 0);
        assert(head_dim <= std::numeric_limits<size_t>::max() / num_attention_heads);
        assert(head_dim <= std::numeric_limits<size_t>::max() / num_kv_heads);
        const size_t q_heads_size = head_dim * num_attention_heads;
        const size_t kv_heads_size = head_dim * num_kv_heads;

        assert(weights.q_proj.shape[0] == q_heads_size);
        assert(weights.k_proj.shape[0] == kv_heads_size);
        assert(weights.v_proj.shape == weights.k_proj.shape);
        assert(weights.q_proj.shape[1] == input.shape[1]);
        assert(weights.k_proj.shape[1] == input.shape[1]);
        assert(weights.o_proj.shape[0] == input.shape[1]);
        assert(weights.o_proj.shape[1] == q_heads_size);
        assert(weights.q_norm.shape[0] == head_dim);
        assert(weights.k_norm.shape[0] == head_dim);

        Tensor<T, 2> query_heads { { num_tokens, q_heads_size } };
        Tensor<T, 2> key_heads { { num_tokens, kv_heads_size } };
        Tensor<T, 2> value_heads { { num_tokens, kv_heads_size } };
        mat_mul_transposed_rhs(input, weights.q_proj.cview(), query_heads.view());
        mat_mul_transposed_rhs(input, weights.k_proj.cview(), key_heads.view());
        mat_mul_transposed_rhs(input, weights.v_proj.cview(), value_heads.view());

        T rms_norm_eps = static_cast<T>(config.rms_norm_eps);

        qk_norm(query_heads.cview(), weights.q_norm.cview(), rms_norm_eps, query_heads.view());
        qk_norm(key_heads.cview(), weights.k_norm.cview(), rms_norm_eps, key_heads.view());

        rope(query_heads.cview(), key_heads.cview(), config, query_heads.view(), key_heads.view());

        Tensor<T, 2> attention_out { { num_tokens, num_attention_heads * head_dim } };

        for (size_t h = 0; h < num_attention_heads; ++h) {
            const size_t kv_h = h / (num_attention_heads / num_kv_heads);

            auto head_out = attention_out.view().narrow(1, h * head_dim, head_dim);
            auto queries = query_heads.cview().narrow(1, h * head_dim, head_dim);
            auto keys = key_heads.cview().narrow(1, kv_h * head_dim, head_dim);
            auto values = value_heads.cview().narrow(1, kv_h * head_dim, head_dim);

            self_attention_head(queries, keys, values, config, head_out);
        }

        mat_mul_transposed_rhs(attention_out.cview(), weights.o_proj.cview(), out);
    }

    template <std::floating_point T>
    void multi_head_attention_with_cache(ConstMatrixView<T> input,
        const AttentionWeights<T>& weights, size_t kv_len, LayerKVCache<T>& layer_kv_cache,
        const Qwen3Config config, const TensorView<T, 2> out) {
        const size_t num_tokens = input.shape[0];

        const size_t num_attention_heads = config.num_attention_heads;
        const size_t head_dim = config.head_dim;
        const size_t num_kv_heads = config.num_key_value_heads;

        assert(input.shape == out.shape);
        assert(input.shape[1] == config.hidden_dim && config.hidden_dim > 0);
        assert(head_dim > 0 && head_dim % 2 == 0);
        assert(num_attention_heads > 0 && num_kv_heads > 0);
        assert(num_attention_heads % num_kv_heads == 0);
        assert(std::isfinite(config.rms_norm_eps) && config.rms_norm_eps > 0);
        assert(std::isfinite(config.rope_theta) && config.rope_theta > 0);
        assert(head_dim <= std::numeric_limits<size_t>::max() / num_attention_heads);
        assert(head_dim <= std::numeric_limits<size_t>::max() / num_kv_heads);
        const size_t q_heads_size = head_dim * num_attention_heads;
        const size_t kv_heads_size = head_dim * num_kv_heads;

        assert(weights.q_proj.shape[0] == q_heads_size);
        assert(weights.k_proj.shape[0] == kv_heads_size);
        assert(weights.v_proj.shape == weights.k_proj.shape);
        assert(weights.q_proj.shape[1] == input.shape[1]);
        assert(weights.k_proj.shape[1] == input.shape[1]);
        assert(weights.o_proj.shape[0] == input.shape[1]);
        assert(weights.o_proj.shape[1] == q_heads_size);
        assert(weights.q_norm.shape[0] == head_dim);
        assert(weights.k_norm.shape[0] == head_dim);

        size_t total_tokens = num_tokens + kv_len;

        Tensor<T, 2> query_heads { { num_tokens, q_heads_size } };

        mat_mul_transposed_rhs(input, weights.q_proj.cview(), query_heads.view());

        auto new_keys = layer_kv_cache.keys.view().narrow(0, kv_len, num_tokens);
        auto new_values = layer_kv_cache.values.view().narrow(0, kv_len, num_tokens);

        mat_mul_transposed_rhs(input, weights.k_proj.cview(), new_keys);
        mat_mul_transposed_rhs(input, weights.v_proj.cview(), new_values);

        T rms_norm_eps = static_cast<T>(config.rms_norm_eps);

        qk_norm(query_heads.cview(), weights.q_norm.cview(), rms_norm_eps, query_heads.view());
        qk_norm(new_keys.cview(), weights.k_norm.cview(), rms_norm_eps, new_keys);

        rope(query_heads.cview(), new_keys.cview(), config, query_heads.view(), new_keys, kv_len);

        auto all_keys = layer_kv_cache.keys.cview().narrow(0, 0, total_tokens);
        auto all_values = layer_kv_cache.values.cview().narrow(0, 0, total_tokens);

        Tensor<T, 2> attention_out { { num_tokens, num_attention_heads * head_dim } };

        for (size_t h = 0; h < num_attention_heads; ++h) {
            const size_t kv_h = h / (num_attention_heads / num_kv_heads);

            auto head_out = attention_out.view().narrow(1, h * head_dim, head_dim);
            auto queries = query_heads.cview().narrow(1, h * head_dim, head_dim);
            auto keys = all_keys.narrow(1, kv_h * head_dim, head_dim);
            auto values = all_values.narrow(1, kv_h * head_dim, head_dim);

            self_attention_head_with_cache(queries, keys, values, kv_len, config, head_out);
        }

        mat_mul_transposed_rhs(attention_out.cview(), weights.o_proj.cview(), out);
    }

    template <std::floating_point T>
    void swiglu_mlp(
        ConstMatrixView<T> input, const MlpWeights<T>& weights, const TensorView<T, 2> out) {
        const size_t num_tokens = input.shape[0];
        assert(input.shape == out.shape);
        assert(weights.up_proj.shape[1] == input.shape[1]);
        assert(weights.gate_proj.shape == weights.up_proj.shape);
        assert(weights.down_proj.shape[0] == input.shape[1]);
        assert(weights.down_proj.shape[1] == weights.up_proj.shape[0]);
        const size_t intermediate_dim = weights.up_proj.shape[0];
        assert(intermediate_dim > 0);

        Tensor<T, 2> up { { num_tokens, intermediate_dim } };
        Tensor<T, 2> gate { { num_tokens, intermediate_dim } };

        mat_mul_transposed_rhs(input, weights.up_proj.cview(), up.view());
        mat_mul_transposed_rhs(input, weights.gate_proj.cview(), gate.view());

        silu(gate.cview(), gate.view());

        Tensor<T, 2> hidden { { num_tokens, intermediate_dim } };
        element_mul(up.cview(), gate.cview(), hidden.view());

        mat_mul_transposed_rhs(hidden.cview(), weights.down_proj.cview(), out);
    }

    template <std::floating_point T>
    void id_to_embedding(
        std::span<const int> ids, ConstMatrixView<T> embeddings, const TensorView<T, 2> out) {
        assert(out.shape[0] == ids.size());
        assert(out.shape[1] == embeddings.shape[1]);

        const size_t hidden_dim = embeddings.shape[1];

        for (size_t i = 0; i < ids.size(); ++i) {
            assert(ids[i] >= 0);
            assert(static_cast<size_t>(ids[i]) < embeddings.shape[0]);

            const auto token_vector = embeddings.row(ids[i]);
            const auto out_vector = out.row(i);

            for (size_t j = 0; j < hidden_dim; ++j) {
                out_vector[j] = token_vector[j];
            }
        }
    }

    template <std::floating_point T>
    void transformer_block(const Qwen3<T>& qwen, size_t layer_num, ConstMatrixView<T> input,
        const TensorView<T, 2> out) {
        const auto& weights = qwen.weights;
        const auto& config = qwen.config;
        const auto& layers = weights.layers;
        const T epsilon = static_cast<T>(qwen.config.rms_norm_eps);
        assert(layer_num < layers.size());
        assert(layer_num < config.num_hidden_layers);
        assert(input.shape == out.shape);
        assert(input.shape[1] == config.hidden_dim);
        const auto& layer_weights = layers[layer_num];
        assert(layer_weights.pre_attention_norm.shape[0] == input.shape[1]);
        assert(layer_weights.pre_mlp_norm.shape[0] == input.shape[1]);

        Tensor<T, 2> temp { input.shape };
        Tensor<T, 2> att_res { input.shape };
        Tensor<T, 2> mlp_res { input.shape };

        for (size_t i = 0; i < input.size(); ++i) {
            temp.view().flat(i) = input.flat(i);
        }

        for (size_t i = 0; i < out.shape[0]; ++i) {
            rms_norm(input.row(i), epsilon, layer_weights.pre_attention_norm.cview(),
                temp.view().row(i));
        }

        multi_head_attention(temp.cview(), layer_weights.attention, config, att_res.view());
        add(input, att_res.cview(), temp.view());

        for (size_t i = 0; i < out.shape[0]; ++i) {
            rms_norm(temp.cview().row(i), epsilon, layer_weights.pre_mlp_norm.cview(),
                temp.view().row(i));
        }

        swiglu_mlp(temp.cview(), layer_weights.mlp, mlp_res.view());

        add(mlp_res.cview(), att_res.cview(), temp.view());
        add(input, temp.cview(), out);
    }

    template <std::floating_point T>
    void transformer_block_with_cache(const Qwen3<T>& qwen, size_t layer_num,
        ConstMatrixView<T> input, size_t kv_len, LayerKVCache<T>& layer_kv_cache,
        const TensorView<T, 2> out) {
        const auto& weights = qwen.weights;
        const auto& config = qwen.config;
        const auto& layers = weights.layers;
        const T epsilon = static_cast<T>(qwen.config.rms_norm_eps);
        assert(layer_num < layers.size());
        assert(layer_num < config.num_hidden_layers);
        assert(input.shape == out.shape);
        assert(input.shape[1] == config.hidden_dim);
        const auto& layer_weights = layers[layer_num];
        assert(layer_weights.pre_attention_norm.shape[0] == input.shape[1]);
        assert(layer_weights.pre_mlp_norm.shape[0] == input.shape[1]);

        Tensor<T, 2> temp { input.shape };
        Tensor<T, 2> att_res { input.shape };
        Tensor<T, 2> mlp_res { input.shape };

        for (size_t i = 0; i < input.size(); ++i) {
            temp.view().flat(i) = input.flat(i);
        }

        for (size_t i = 0; i < out.shape[0]; ++i) {
            rms_norm(input.row(i), epsilon, layer_weights.pre_attention_norm.cview(),
                temp.view().row(i));
        }

        multi_head_attention_with_cache(
            temp.cview(), layer_weights.attention, kv_len, layer_kv_cache, config, att_res.view());
        add(input, att_res.cview(), temp.view());

        for (size_t i = 0; i < out.shape[0]; ++i) {
            rms_norm(temp.cview().row(i), epsilon, layer_weights.pre_mlp_norm.cview(),
                temp.view().row(i));
        }

        swiglu_mlp(temp.cview(), layer_weights.mlp, mlp_res.view());

        add(mlp_res.cview(), att_res.cview(), temp.view());
        add(input, temp.cview(), out);
    }

    template <std::floating_point T>
    void compute_logits(
        const Qwen3<T>& qwen, std::span<const int> tokens, const TensorView<T, 1> out) {
        const auto& config = qwen.config;
        const size_t num_tokens = tokens.size();
        const size_t hidden_dim = config.hidden_dim;
        const size_t hidden_layers = config.num_hidden_layers;
        const auto embeddings = qwen.weights.token_embedding.cview();

        Tensor<T, 2> input { { num_tokens, hidden_dim } };

        id_to_embedding(tokens, embeddings, input.view());

        for (size_t layer = 0; layer < hidden_layers; ++layer) {
            transformer_block(qwen, layer, input.cview(), input.view());
        }

        rms_norm(input.cview().row(num_tokens - 1), static_cast<T>(config.rms_norm_eps),
            qwen.weights.final_norm.cview(), input.view().row(num_tokens - 1));

        // Tensor<T, 1> logits { { embeddings.shape[0] } };
        mat_mul(embeddings, input.cview().row(num_tokens - 1), out);
    }

    template <std::floating_point T>
    void compute_logits_with_cache(const Qwen3<T>& qwen, std::span<const int> tokens,
        KVCache<T>& kv_cache, const TensorView<T, 1> out) {
        const auto& config = qwen.config;
        const size_t num_tokens = tokens.size();
        const size_t hidden_dim = config.hidden_dim;
        const size_t hidden_layers = config.num_hidden_layers;
        const auto embeddings = qwen.weights.token_embedding.cview();

        Tensor<T, 2> input { { num_tokens, hidden_dim } };

        id_to_embedding(tokens, embeddings, input.view());

        for (size_t layer = 0; layer < hidden_layers; ++layer) {
            transformer_block_with_cache(
                qwen, layer, input.cview(), kv_cache.len, kv_cache.layers[layer], input.view());
        }

        rms_norm(input.cview().row(num_tokens - 1), static_cast<T>(config.rms_norm_eps),
            qwen.weights.final_norm.cview(), input.view().row(num_tokens - 1));

        mat_mul(embeddings, input.cview().row(num_tokens - 1), out);

        kv_cache.len += num_tokens;
    }
}
