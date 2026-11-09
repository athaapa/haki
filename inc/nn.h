#pragma once

#include "math.h"
#include "model.h"
#include "tensor.h"
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
            total += (x.data[i] * x.data[i]);
        }

        return std::sqrt(epsilon + total / n);
    }

    // TODO: add assertions
    template <std::floating_point T, size_t Rank>
    void rms_norm(TensorView<const T, Rank> x, T epsilon, TensorView<const T, Rank> parameters,
        const TensorView<T, Rank> out) {
        size_t n = out.size();
        T res = rms(x, epsilon);
        for (size_t i = 0; i < n; ++i) {
            out.data[i] = parameters.data[i] * (x.data[i] / res);
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

        for (size_t i = 0; i < num_tokens; ++i) {
            const size_t q_offset = i * q_vectors.shape[1] + h * head_dim;
            ConstVectorView<T> query { q_vectors.data + q_offset, { head_dim } };

            Tensor<T, 1> row_score({ num_tokens });
            std::fill_n(
                row_score.view().data, num_tokens, T { -std::numeric_limits<T>::infinity() });

            for (size_t j = 0; j <= i; ++j) {
                // TODO: hoist q_vectors.shape[1]
                const size_t kv_offset = j * k_vectors.shape[1] + kv_h * head_dim;
                ConstVectorView<T> key { k_vectors.data + kv_offset, { head_dim } };

                row_score.storage[j] = dot(query, key) * dim_inv_root;
            }

            softmax(row_score.cview(), scores.view().row(i));
        }

        std::fill_n(head_output.data, head_output.size(), T { 0 });

        // multiply each value vector by each attention score
        for (size_t i = 0; i < num_tokens; ++i) {
            auto out = head_output.row(i);

            for (size_t j = 0; j <= i; ++j) {
                const size_t offset = j * v_vectors.shape[1] + kv_h * head_dim;

                ConstVectorView<T> value { v_vectors.data + offset, { head_dim } };

                const T weight = scores.storage[i * num_tokens + j];

                for (size_t d = 0; d < head_dim; ++d) {
                    out.data[d] += weight * value.data[d];
                }
            }
        }
    }

    template <std::floating_point T>
    void rotate_vector(ConstVectorView<T> a, T theta, const TensorView<T, 1> out) {
        assert(a.size() == 2);
        assert(out.size() == 2);

        const T x = a.data[0];
        const T y = a.data[1];

        const T c = std::cos(theta);
        const T s = std::sin(theta);

        out.data[0] = c * x - s * y;
        out.data[1] = s * x + c * y;
    }

    template <std::floating_point T>
    void rope(ConstMatrixView<T> q_vectors, ConstMatrixView<T> k_vectors, Qwen3Config config,
        const TensorView<T, 2> q_out, const TensorView<T, 2> k_out) {
        const size_t num_tokens = q_vectors.shape[0];
        const size_t num_attention_heads = config.num_attention_heads;
        const size_t head_dim = config.head_dim;
        const size_t num_kv_heads = config.num_key_value_heads;
        const T theta = config.rope_theta;

        // queries
        for (size_t i = 0; i < num_tokens; ++i) {
            for (size_t h = 0; h < num_attention_heads; ++h) {
                const size_t q_offset = i * q_vectors.shape[1] + h * head_dim;

                for (size_t k = 0; k < head_dim / 2; ++k) {

                    const T freq = static_cast<T>(i)
                        * std::pow(theta, -2 * static_cast<T>(k) / static_cast<T>(head_dim));

                    T pair1 = *(q_vectors.data + q_offset + k);
                    T pair2 = *(q_vectors.data + q_offset + k + head_dim / 2);
                    T pair[2] { pair1, pair2 };

                    TensorView<T, 1> q_pair { pair, { 2 } };
                    TensorView<const T, 1> const_q_pair { pair, { 2 } };

                    rotate_vector(const_q_pair, freq, q_pair);

                    *(q_out.data + q_offset + k) = pair[0];
                    *(q_out.data + q_offset + k + head_dim / 2) = pair[1];
                }
            }
        }

        // keys
        for (size_t i = 0; i < num_tokens; ++i) {
            for (size_t h = 0; h < num_kv_heads; ++h) {
                const size_t k_offset = i * k_vectors.shape[1] + h * head_dim;

                for (size_t k = 0; k < head_dim / 2; ++k) {

                    const T freq = static_cast<T>(i)
                        * std::pow(theta, -2 * static_cast<T>(k) / static_cast<T>(head_dim));

                    T pair1 = *(k_vectors.data + k_offset + k);
                    T pair2 = *(k_vectors.data + k_offset + k + head_dim / 2);
                    T pair[2] { pair1, pair2 };

                    TensorView<T, 1> k_pair { pair, { 2 } };
                    TensorView<const T, 1> const_k_pair { pair, { 2 } };

                    rotate_vector(const_k_pair, freq, k_pair);

                    k_out.data[k_offset + k] = pair[0];
                    k_out.data[k_offset + k + head_dim / 2] = pair[1];
                }
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

        // get q, k, and v by multiplying embeddings by projections
        //
        // [num_tokens, hidden_size] * [num_attention_head * head_dim, hidden_size] (tranposed)
        Tensor<T, 2> q_vectors { { num_tokens, q_heads_size } };
        Tensor<T, 2> k_vectors { { num_tokens, kv_heads_size } };
        Tensor<T, 2> v_vectors { { num_tokens, kv_heads_size } };
        mat_mul_transposed_rhs(input, weights.q_proj.cview(), q_vectors.view());
        mat_mul_transposed_rhs(input, weights.k_proj.cview(), k_vectors.view());
        mat_mul_transposed_rhs(input, weights.v_proj.cview(), v_vectors.view());

        // QK-norm
        // NOTE: factor this out as its own function?
        for (size_t i = 0; i < num_tokens; ++i) {
            for (size_t h = 0; h < num_attention_heads; ++h) {

                const size_t q_offset = i * q_vectors.shape[1] + h * head_dim;

                ConstVectorView<T> q_vector { q_vectors.storage.data() + q_offset, { head_dim } };
                TensorView<T, 1> q_vector_norm { q_vectors.storage.data() + q_offset,
                    { head_dim } };

                rms_norm(q_vector, static_cast<T>(config.rms_norm_eps), weights.q_norm.cview(),
                    q_vector_norm);
            }
        }

        for (size_t i = 0; i < num_tokens; ++i) {
            for (size_t h = 0; h < num_kv_heads; ++h) {
                const size_t k_offset = i * k_vectors.shape[1] + h * head_dim;

                ConstVectorView<T> k_vector { k_vectors.storage.data() + k_offset, { head_dim } };
                TensorView<T, 1> k_vector_norm { k_vectors.storage.data() + k_offset,
                    { head_dim } };

                rms_norm(k_vector, static_cast<T>(config.rms_norm_eps), weights.k_norm.cview(),
                    k_vector_norm);
            }
        }

        // apply RoPE
        rope(q_vectors.cview(), k_vectors.cview(), config, q_vectors.view(), k_vectors.view());

        Tensor<T, 2> attention_out { { num_tokens, num_attention_heads * head_dim } };

        // split q, k, and v into views of length head_dim and run attention on each
        // concatenate results
        // qwen3 uses GQA so each each query shares a KV with another query

        for (size_t h = 0; h < num_attention_heads; ++h) {
            const size_t kv_h = h / (num_attention_heads / num_kv_heads);

            Tensor<T, 2> head_out { { num_tokens, head_dim } };

            self_attention_head(q_vectors.cview(), k_vectors.cview(), v_vectors.cview(), h, kv_h,
                config, head_out.view());

            for (size_t i = 0; i < num_tokens; ++i) {
                for (size_t j = 0; j < head_dim; ++j) {
                    attention_out.view()
                        .data[i * (num_attention_heads * head_dim) + (h * head_dim) + j]
                        = head_out.view().data[i * head_dim + j];
                }
            }
        }

        // apply output projection
        mat_mul_transposed_rhs(attention_out.cview(), weights.o_proj.cview(), out);
    }

    template <std::floating_point T>
    void swiglu_mlp(
        ConstMatrixView<T> input, const MlpWeights<T>& weights, const TensorView<T, 2> out) {
        const size_t num_tokens = input.shape[0];
        const size_t intermediate_dim = weights.up_proj.shape[0];

        // apply gated projection and up projection
        Tensor<T, 2> up { { num_tokens, intermediate_dim } };
        Tensor<T, 2> gate { { num_tokens, intermediate_dim } };

        // [num_tokens, hidden_size] *  [intermediate_dim, hidden_size] (transposed)

        mat_mul_transposed_rhs(input, weights.up_proj.cview(), up.view());
        mat_mul_transposed_rhs(input, weights.gate_proj.cview(), gate.view());

        // apply silu to gated projection
        silu(gate.cview(), gate.view());

        // element-wise multiply gated and up
        Tensor<T, 2> hidden { { num_tokens, intermediate_dim } };
        mul(up.cview(), gate.cview(), hidden.view());

        // down project
        // [hidden_size, intermediate_dim] * [num_tokens, intermediate_dim] (transposed)
        mat_mul_transposed_rhs(hidden.cview(), weights.down_proj.cview(), out);
    }

    template <std::floating_point T>
    void transformer_block(
        const Qwen3<T>& qwen, size_t l, ConstMatrixView<T> input, const TensorView<T, 2> out) {
        const Qwen3Weights<T>& weights = qwen.weights;
        const Qwen3Config& config = qwen.config;
        const std::vector<Qwen3LayerWeights<T>>& layers = weights.layers;
        const T epsilon = static_cast<T>(qwen.config.rms_norm_eps);
        const Qwen3LayerWeights<T>& layer_weights = layers[l];

        Tensor<T, 2> temp { input.shape };
        Tensor<T, 2> att_res { input.shape };
        Tensor<T, 2> mlp_res { input.shape };

        std::copy_n(input.data, input.size(), temp.view().data);

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
