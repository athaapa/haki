#pragma once

#include "tensor.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace haki::transformer {
    template <std::floating_point T>
    void self_attention_head(const std::vector<tensor::ConstVectorView<T>>& embeddings,
        tensor::ConstMatrixView<T> w_q, tensor::ConstMatrixView<T> w_k,
        tensor::ConstMatrixView<T> w_v, std::vector<tensor::TensorView<T, 1>>& out) {
        size_t n = embeddings.size();

        // compute queries, keys, and values by multiplying each embedding with the query
        // projection matrix
        std::vector<tensor::Tensor<T, 1>> queries;
        std::vector<tensor::Tensor<T, 1>> keys;
        std::vector<tensor::Tensor<T, 1>> values;
        queries.reserve(n);
        keys.reserve(n);
        values.reserve(n);

        for (size_t i = 0; i < n; ++i) {
            queries.emplace_back(std::array<size_t, 1> { w_q.shape[0] });
            keys.emplace_back(std::array<size_t, 1> { w_k.shape[0] });
            values.emplace_back(std::array<size_t, 1> { w_v.shape[0] });
        }

        tensor::batched_mat_vec(w_q, embeddings, queries);
        tensor::batched_mat_vec(w_k, embeddings, keys);
        tensor::batched_mat_vec(w_v, embeddings, values);

        // compute dot product of every query, key pair
        tensor::Tensor<T, 2> scores({ n, n });

        T dim_root = std::sqrt(static_cast<T>(w_q.shape[0]));

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                scores.storage[i * n + j] = j > i
                    ? -std::numeric_limits<T>::infinity()
                    : dot(queries[i].cview(), keys[j].cview()) / dim_root;
            }
        }

        // compute softmax of rows
        tensor::Tensor<T, 2> weights({ n, n });

        for (int i = 0; i < n; ++i) {
            softmax(scores.cview().row(i), weights.view().row(i));
        }

        const size_t value_dim = values[0].shape[0];
        tensor::Tensor<T, 1> scaled_value_vector({ value_dim });

        for (int i = 0; i < n; ++i) {
            std::fill_n(out[i].data, out[i].size(), T { 0 });
            for (int j = 0; j < n; ++j) {
                tensor::scale(
                    values[j].cview(), weights.storage[i * n + j], scaled_value_vector.view());
                tensor::add(out[i], scaled_value_vector.cview(), out[i]);
            }
        }
    }
}
