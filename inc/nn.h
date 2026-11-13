#pragma once

#include "model.h"
#include "tensor.h"
#include "tmath.h"
#include <algorithm>
#include <cmath>
#include <concepts>
#include <functional>
#include <limits>
#include <vector>

namespace haki {
    template <std::floating_point T> T rms(ConstVectorView<T> x, T epsilon) {
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
        size_t n = out.size();
        T res = rms(x, epsilon);
        for (size_t i = 0; i < n; ++i) {
            out[i] = parameters[i] * (x[i] / res);
        }
    }

    template <std::floating_point T>
    void compute_attention_weights(ConstMatrixView<T> queries, ConstMatrixView<T> keys,
        size_t head_dim, size_t h, size_t kv_h, const TensorView<T, 2> out) {
        const size_t num_tokens = queries.shape[0];
        const T dim_inv_root = T { 1 } / std::sqrt(static_cast<T>(head_dim));

        for (size_t i = 0; i < num_tokens; ++i) {
            auto query = queries.row(i).narrow(0, h * head_dim, head_dim);

            Tensor<T, 1> row_score({ num_tokens });
            std::fill_n(
                row_score.view().data, num_tokens, T { -std::numeric_limits<T>::infinity() });

            for (size_t j = 0; j <= i; ++j) {
                auto key = keys.row(j).narrow(0, kv_h * head_dim, head_dim);

                row_score.storage[j] = dot(query, key) * dim_inv_root;
            }

            softmax(row_score.cview(), out.row(i));
        }
    }
    template <std::floating_point T>
    void self_attention_head(ConstMatrixView<T> q_vectors, ConstMatrixView<T> k_vectors,
        ConstMatrixView<T> v_vectors, size_t h, size_t kv_h, Qwen3Config config,
        const TensorView<T, 2> head_output) {
        const size_t head_dim = config.head_dim;
        const size_t num_tokens = q_vectors.shape[0];
        const T dim_inv_root = T { 1 } / std::sqrt(static_cast<T>(head_dim));

        Tensor<T, 2> scores({ num_tokens, num_tokens });

        compute_attention_weights(q_vectors, k_vectors, head_dim, h, kv_h, scores.view());

        for (size_t i = 0; i < head_output.size(); ++i)
            head_output.flat(i) = T { 0 };

        // multiply each value vector by each attention score
        for (size_t i = 0; i < num_tokens; ++i) {
            auto out = head_output.row(i);

            for (size_t j = 0; j <= i; ++j) {
                auto value = v_vectors.row(j).narrow(0, kv_h * head_dim, head_dim);

                const T weight = scores.storage[(i * num_tokens) + j];

                for (size_t d = 0; d < head_dim; ++d) {
                    out[d] += weight * value[d];
                }
            }
        }
    }

    template <std::floating_point T>
    void rotate_vector(ConstVectorView<T> a, T theta, const TensorView<T, 1> out) {
        assert(a.size() == 2);
        assert(out.size() == 2);

        const T x = a[0];
        const T y = a[1];

        const T cos_theta = std::cos(theta);
        const T sin_theta = std::sin(theta);

        out[0] = (cos_theta * x) - (sin_theta * y);
        out[1] = (sin_theta * x) + (cos_theta * y);
    }

    template <std::floating_point T>
    void rotate_pairs(
        ConstMatrixView<T> vectors, size_t head_dim, T theta, const TensorView<T, 2> out) {
        const size_t num_tokens = vectors.shape[0];
        const size_t num_heads = vectors.shape[1] / head_dim;

        for (size_t i = 0; i < num_tokens; ++i) {
            for (size_t h = 0; h < num_heads; ++h) {
                const size_t offset = h * head_dim;

                for (size_t k = 0; k < head_dim / 2; ++k) {

                    const T freq = static_cast<T>(i)
                        * std::pow(theta, -2 * static_cast<T>(k) / static_cast<T>(head_dim));

                    T pair1 = vectors[i, offset + k];
                    T pair2 = vectors[i, offset + k + (head_dim / 2)];
                    std::array pair { pair1, pair2 };

                    TensorView<T, 1> q_pair { pair.data(), { 2 } };
                    TensorView<const T, 1> const_q_pair { pair.data(), { 2 } };

                    rotate_vector(const_q_pair, freq, q_pair);

                    out[i, offset + k] = pair[0];
                    out[i, offset + k + (head_dim / 2)] = pair[1];
                }
            }
        }
    }

    template <std::floating_point T>
    void rope(ConstMatrixView<T> q_vectors, ConstMatrixView<T> k_vectors, Qwen3Config config,
        const TensorView<T, 2> q_out, const TensorView<T, 2> k_out) {
        const size_t num_tokens = q_vectors.shape[0];
        const size_t head_dim = config.head_dim;
        const T theta = config.rope_theta;

        rotate_pairs(q_vectors, head_dim, theta, q_out);
        rotate_pairs(k_vectors, head_dim, theta, k_out);
    }

    template <std::floating_point T>
    void qk_norm(const TensorView<T, 2> vectors, ConstVectorView<T> norm_weights, T rms_norm_eps) {
        const size_t head_dim = norm_weights.size();
        assert(head_dim > 0);
        assert(vectors.shape[1] % head_dim == 0);

        const size_t num_heads = vectors.shape[1] / head_dim;
        const size_t num_tokens = vectors.shape[0];

        for (size_t i = 0; i < num_tokens; ++i) {
            for (size_t h = 0; h < num_heads; ++h) {
                const size_t offset = h * head_dim;

                auto vector_norm = vectors.row(i).narrow(0, offset, head_dim);
                auto vector = vector_norm.cview();

                rms_norm(vector, rms_norm_eps, norm_weights, vector_norm);
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

        const size_t q_heads_size = head_dim * num_attention_heads;
        const size_t kv_heads_size = head_dim * num_kv_heads;

        Tensor<T, 2> q_vectors { { num_tokens, q_heads_size } };
        Tensor<T, 2> k_vectors { { num_tokens, kv_heads_size } };
        Tensor<T, 2> v_vectors { { num_tokens, kv_heads_size } };
        mat_mul_transposed_rhs(input, weights.q_proj.cview(), q_vectors.view());
        mat_mul_transposed_rhs(input, weights.k_proj.cview(), k_vectors.view());
        mat_mul_transposed_rhs(input, weights.v_proj.cview(), v_vectors.view());

        T rms_norm_eps = static_cast<T>(config.rms_norm_eps);

        // TODO: In place?
        qk_norm(q_vectors.view(), weights.q_norm.cview(), rms_norm_eps);
        qk_norm(k_vectors.view(), weights.k_norm.cview(), rms_norm_eps);

        rope(q_vectors.cview(), k_vectors.cview(), config, q_vectors.view(), k_vectors.view());

        Tensor<T, 2> attention_out { { num_tokens, num_attention_heads * head_dim } };

        for (size_t h = 0; h < num_attention_heads; ++h) {
            const size_t kv_h = h / (num_attention_heads / num_kv_heads);

            auto head_out = attention_out.view().narrow(1, h * head_dim, head_dim);
            self_attention_head(
                q_vectors.cview(), k_vectors.cview(), v_vectors.cview(), h, kv_h, config, head_out);
        }

        mat_mul_transposed_rhs(attention_out.cview(), weights.o_proj.cview(), out);
    }

    template <std::floating_point T>
    void swiglu_mlp(
        ConstMatrixView<T> input, const MlpWeights<T>& weights, const TensorView<T, 2> out) {
        const size_t num_tokens = input.shape[0];
        const size_t intermediate_dim = weights.up_proj.shape[0];

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
    void transformer_block(const Qwen3<T>& qwen, size_t layer_num, ConstMatrixView<T> input,
        const TensorView<T, 2> out) {
        const Qwen3Weights<T>& weights = qwen.weights;
        const Qwen3Config& config = qwen.config;
        const std::vector<Qwen3LayerWeights<T>>& layers = weights.layers;
        const T epsilon = static_cast<T>(qwen.config.rms_norm_eps);
        const Qwen3LayerWeights<T>& layer_weights = layers[layer_num];

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
}
