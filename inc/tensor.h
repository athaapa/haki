#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <concepts>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace haki::tensor {
    template <std::floating_point T, size_t Rank> struct TensorView {
        T* data;
        std::array<size_t, Rank> shape;

        size_t size() const {
            const size_t n
                = std::accumulate(shape.begin(), shape.end(), size_t { 1 }, std::multiplies<> { });
            return n;
        }

        TensorView<T, 1> row(size_t i) const
            requires(Rank == 2)
        {
            assert(i < shape[0]);
            return { data + i * shape[1], { shape[1] } };
        }
    };

    template <std::floating_point T, size_t Rank> struct Tensor {
        std::vector<T> storage;
        std::array<size_t, Rank> shape;

        explicit Tensor(std::array<size_t, Rank> dims)
            : storage(
                  std::accumulate(dims.begin(), dims.end(), size_t { 1 }), std::multiplies<> { })
            , shape(dims) { }

        TensorView<T, Rank> view() { return { storage.data(), shape }; }
        TensorView<const T, Rank> view() const { return { storage.data(), shape }; }
        TensorView<const T, Rank> cview() const { return view(); }
    };

    template <std::floating_point T> using ConstVectorView = TensorView<const T, 1>;

    template <std::floating_point T> using ConstMatrixView = TensorView<const T, 2>;

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

    template <std::floating_point T, size_t Rank>
    void add(const TensorView<const T, Rank> a, const TensorView<const T, Rank> b,
        const TensorView<T, Rank> out) {
        assert(a.shape == b.shape);
        assert(a.shape == out.shape);

        const size_t n = a.size();

        std::transform(a.data, a.data + n, b.data, out.data, std::plus<> { });
    }

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

    template <std::floating_point T>
    void batched_mat_vec(ConstMatrixView<T> weights, const std::vector<ConstVectorView<T>>& inputs,
        const std::vector<TensorView<T, 1>>& outputs) {

        assert(inputs.size() == outputs.size());

        for (size_t i = 0; i < inputs.size(); ++i) {
            mat_mul(weights, inputs[i], outputs[i]);
        }
    }
    template <std::floating_point T> T dot(ConstVectorView<T> a, ConstVectorView<T> b) {
        return std::inner_product(a.data, a.data + a.size(), b.data, T { 0 });
    }

}
