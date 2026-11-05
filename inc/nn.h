#pragma once

#include "math.h"
#include "model.h"
#include "tensor.h"
#include <algorithm>
#include <cmath>
#include <concepts>
#include <functional>
#include <vector>

namespace haki {
    /*
     * @brief matrix multiplication with matrices a and b
     */
    template <std::floating_point T>
    void mat_mul(ConstMatrixView<T> a, ConstMatrixView<T> b, const TensorView<T, 2> out) {
        assert(a.shape[1] == b.shape[0]);

        assert(a.shape[0] == out.shape[0]);
        assert(b.shape[1] == out.shape[1]);

        int a_n = a.shape[0];
        int b_n = b.shape[0];

        int a_m = a.shape[1];
        int b_m = b.shape[1];

        for (int i = 0; i < a_n; ++i) { // row
            for (int j = 0; j < b_m; ++j) {
                T sum = 0;
                for (int k = 0; k < b_n; ++k) {
                    sum += a.data[i * a_m + k] * b.data[k * b_m + j];
                }
                out.data[i * b_m + j] = sum;
            }
        }
    }

    /*
     * @brief element-wise matrix multiplication
     */
    template <std::floating_point T, size_t Rank>
    void mul(TensorView<T, Rank> a, TensorView<T, Rank> b, const TensorView<T, Rank> out) {
        assert(a.size() == b.size());
        assert(a.size() == out.size());

        std::transform(a.data, a.data + a.size(), b.data, out.data, std::multiplies<T>());
    }

    /*
     * @brief matrix multiplication with matrix a and vector b
     * a: [m, k]
     * b: [k, n]
     * out: [m, n]
     */
    template <std::floating_point T>
    void mat_mul(ConstMatrixView<T> a, ConstVectorView<T> b, const TensorView<T, 1> out) {
        assert(a.shape[1] == b.shape[0]);
        assert(a.shape[0] == out.shape[0]);

        const size_t cols = a.shape[1];

        for (size_t i = 0; i < a.shape[0]; ++i) {
            const T* row = a.data + i * cols;
            out.data[i] = std::inner_product(row, row + cols, b.data, T { 0 });
        }
    }

    /*
     * @brief matrix multiplication with matrix a and matrix b^T
     *
     * a: [m, k]
     * b: [n, k]
     * out: [m, n]
     */
    template <std::floating_point T>
    void mat_mul_transposed_rhs(ConstMatrixView<T> a, ConstMatrixView<T> b, TensorView<T, 2> out) {
        assert(a.shape[1] == b.shape[1]);
        assert(out.shape[0] == a.shape[0]);
        assert(out.shape[1] == b.shape[0]);

        for (size_t i = 0; i < a.shape[0]; ++i) {
            for (size_t j = 0; j < b.shape[0]; ++j) {
                out.data[i * out.shape[1] + j] = dot(a.row(i), b.row(j));
            }
        }
    }

    /*
     * @brief tensor addition
     */
    template <std::floating_point T, size_t Rank>
    void add(const TensorView<const T, Rank> a, const TensorView<const T, Rank> b,
        const TensorView<T, Rank> out) {
        assert(a.shape == b.shape);
        assert(a.shape == out.shape);

        const size_t n = a.size();

        std::transform(a.data, a.data + n, b.data, out.data, std::plus<> { });
    }

    /*
     * @brief tensor scalar multiplication
     */
    template <std::floating_point T, size_t Rank>
    void scale(const TensorView<const T, Rank> a, T scalar, const TensorView<T, Rank> out) {
        assert(a.shape == out.shape);

        const size_t n = a.size();

        std::transform(a.data, a.data + n, out.data, [scalar](T x) { return x * scalar; });
    }

    // NOTE: can overflow
    template <std::floating_point T>
    void softmax(ConstVectorView<T> a, const TensorView<T, 1> out) {
        assert(a.shape == out.shape);

        size_t n = a.shape[0];

        T sum = 0;

        for (size_t i = 0; i < n; ++i) {
            T p = std::exp(a.data[i]);
            out.data[i] = p;
            sum += p;
        }

        for (size_t i = 0; i < n; ++i)
            out.data[i] /= sum;
    }

    template <std::floating_point T> T dot(ConstVectorView<T> a, ConstVectorView<T> b) {
        return std::inner_product(a.data, a.data + a.size(), b.data, T { 0 });
    }

    template <std::floating_point T> T rms(ConstVectorView<T> x, T epsilon) {
        T total = 0;
        int n = x.size();
        for (size_t i = 0; i < n; ++i) {
            total += (x[i] * x[i]);
        }

        return std::sqrt(epsilon + total / n);
    }

    template <std::floating_point T>
    void rms_norm(ConstVectorView<T> x, T epsilon, ConstVectorView<T> parameters,
        const TensorView<T, 1> out) {
        int n = out.size();
        T res = rms(x, epsilon);
        for (int i = 0; i < n; ++i) {
            out.data[i] = parameters[i] * (x.data[i] / res);
        }
    }

    template <std::floating_point T>
    void self_attention_head(ConstMatrixView<T> q_vectors, ConstMatrixView<T> k_vectors,
        ConstMatrixView<T> v_vectors, size_t h, size_t kv_h, Qwen3Config config,
        const TensorView<T, 2> head_output) {
        const size_t head_dim = config.head_dim;
        const size_t num_attention_heads = config.num_attention_heads;
        const size_t num_kv_heads = config.num_key_value_heads;
        const size_t num_tokens = q_vectors.shape[0];
        const T dim_inv_root = T { 1 } / std::sqrt(static_cast<T>(head_dim));

        Tensor<T, 2> scores({ num_tokens, num_tokens });

        for (size_t i = 0; i < num_tokens; ++i) {
            const size_t q_offset = i * q_vectors.shape[1] + h * head_dim;
            ConstVectorView<T> query { q_vectors.data + q_offset, { head_dim } };

            Tensor<T, 1> scores({ i + 1 });

            for (size_t j = 0; j <= i; ++j) {
                // TODO: hoist q_vectors.shape[1]
                const size_t kv_offset = j * k_vectors.shape[1] + kv_h * head_dim;
                ConstVectorView<T> key { k_vectors.data + kv_offset, { head_dim } };

                scores.storage[i * num_tokens + j] = dot(query, key) * dim_inv_root;
            }

            softmax(scores.cview().row(i), scores.view().row(i));
        }

        // multiply each value vector by each attention score
        for (size_t i = 0; i < num_tokens; ++i) {
            auto out = head_output.row(i);
            std::fill_n(out.data, head_dim, T { 0 });

            for (size_t j = 0; j <= i; ++j) {
                const size_t offset = j * v_vectors.shape[0] * kv_h * head_dim;

                ConstVectorView<T> value { v_vectors.data + offset, { head_dim } };

                const T weight = scores.storage[i * num_tokens + j];

                for (size_t d = 0; d < head_dim; ++d) {
                    out.data[d] += weight * value.data[d];
                }
            }
        }
    }

    template <std::floating_point T>
    void rotate_matrix(ConstMatrixView<T> a, T theta, const TensorView<T, 2> out) {
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

                    rotate_matrix(q_pair, freq, q_pair);

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

                    rotate_matrix(k_pair, freq, k_pair);

                    *(k_out.data + k_offset + k) = pair[0];
                    *(k_out.data + k_offset + k + head_dim / 2) = pair[1];
                }
            }
        }
    }

    template <std::floating_point T>
    void multi_head_attention(ConstMatrixView<T> input, AttentionWeights<T>& weights,
        Qwen3Config config, const TensorView<T, 2> out) {
        const size_t num_tokens = input.shape[0];

        const size_t num_attention_heads = config.num_attention_heads;
        const size_t head_dim = config.head_dim;
        const size_t num_kv_heads = config.num_key_value_heads;

        const size_t q_heads_size = head_dim * num_attention_heads;
        const size_t kv_heads_size = head_dim * num_kv_heads;

        // get q, k, and v by multiplying embeddings by projections
        //
        // [num_tokens, hidden_size] * [num_attention_head * head_dim, hidden_size]
        Tensor<T, 2> q_vectors { { num_tokens, q_heads_size } };
        Tensor<T, 2> k_vectors { { num_tokens, kv_heads_size } };
        Tensor<T, 2> v_vectors { { num_tokens, kv_heads_size } };
        mat_mul_transposed_rhs(input, weights.q_proj.cview(), q_vectors);
        mat_mul_transposed_rhs(input, weights.k_proj.cview(), k_vectors);
        mat_mul_transposed_rhs(input, weights.v_proj.cview(), v_vectors);

        // QK-norm
        // NOTE: factor this out as its own function?
        for (size_t i = 0; i < num_tokens; ++i) {
            for (size_t h = 0; h < num_attention_heads; ++h) {
                size_t kv_h = h / (num_attention_heads / num_kv_heads);

                const size_t q_offset = i * q_vectors.shape[1] + h * head_dim;
                const size_t k_offset = i * k_vectors.shape[1] + kv_h * head_dim;

                ConstVectorView<T> q_vector { q_vectors.storage.data() + q_offset, { head_dim } };
                TensorView<T, 1> q_vector_norm { q_vectors.storage.data() + q_offset,
                    { head_dim } };

                ConstVectorView<T> k_vector { k_vectors.storage.data() + k_offset, { head_dim } };
                TensorView<T, 1> k_vector_norm { k_vectors.storage.data() + k_offset,
                    { head_dim } };

                rms_norm(q_vector, config.rms_norm_eps, weights.q_norm.cview(), q_vector_norm);
                rms_norm(k_vector, config.rms_norm_eps, weights.k_norm.cview(), k_vector_norm);
            }
        }

        // apply RoPE
        rope(q_vectors.cview(), k_vectors.cview(), q_vectors.view(), k_vectors.view());

        Tensor<T, 2> attention_out { { num_tokens, num_attention_heads * head_dim } };

        // split q, k, and v into views of length head_dim and run attention on each
        // concatenate results
        // qwen3 uses GQA so each each query shares a KV with another query

        for (size_t h = 0; h < num_attention_heads; ++h) {
            const size_t kv_h = h / (num_attention_heads / num_kv_heads);

            self_attention_head(k_vectors.cview(), k_vectors.cview(), v_vectors.cview(), h, kv_h,
                attention_out.view());
        }

        // apply output projection
        mat_mul_transposed_rhs(attention_out.cview(), weights.o_proj.cview(), out);
    }

    template <std::floating_point T>
    void swiglu_mlp(ConstMatrixView<T> input, MlpWeights<T>& weights, const TensorView<T, 2> out) {
        const size_t num_tokens = input.shape[0];
        const size_t intermediate_dim = weights.up_proj.shape[0];

        // apply gated projection and up projection
        Tensor<T, 2> up { { num_tokens, intermediate_dim } };
        Tensor<T, 2> gate { { num_tokens, intermediate_dim } };

        // [intermediate_dim, hidden_size] * num_tokens, hidden_size] (transposed)

        mat_mul_transposed_rhs(input, weights.up_proj.cview(), up.view());
        mat_mul_transposed_rhs(input, weights.gate_proj.cview(), gate.view());

        // apply silu to gated projection
        silu(gate.cview(), gate.view());

        // element-wise multiply gated and up
        Tensor<T, 2> hidden { { num_tokens, intermediate_dim } };
        mul(up.cview(), gate.cview(), hidden);

        // down project
        // [hidden_size, intermediate_dim] * [num_tokens, intermediate_dim] (transposed)
        mat_mul_transposed_rhs(hidden.cview(), weights.down_proj.cview(), out);
    }

    template <std::floating_point T>
    void transformer_block(
        const Qwen3<T>& qwen, size_t l, ConstMatrixView<T> input, const TensorView<T, 2> out) {
        Qwen3Weights<T>& weights = qwen.weights;
        Qwen3Config& config = qwen.config;
        std::vector<Qwen3LayerWeights<T>>& layers = weights.layers;
        const T epsilon = static_cast<T>(qwen.config.rms_norm_eps);
        const Qwen3LayerWeights<T>& layer_weights = layers[l];

        std::copy_n(input.data, input.size(), out.data);

        ConstMatrixView<T> current { out.data, out.shape };

        Tensor<T, 2> input_norm { input.shape };
        Tensor<T, 2> residual { input.shape };

        rms_norm(current, epsilon, layer_weights.pre_attention_norm.cview(), input_norm.view());
        multi_head_attention(input_norm.cview(), layer_weights.attention, config, residual.view());
        add(current, residual.cview(), current);

        rms_norm(current, epsilon, layer_weights.pre_mlp_norm, input_norm.view());
        swiglu_mlp(input_norm.cview(), layer_weights.mlp, residual.cview());
        add(current, residual.cview(), current);
    }
}
