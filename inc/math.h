#pragma once
#include "tensor.h"
#include <algorithm>
#include <cmath>
#include <concepts>

namespace haki {

    template <std::floating_point T> T sigmoid(T x) { return 1 / (1 + std::exp(-x)); }

    template <std::floating_point T> T silu(T x) { return x * sigmoid(x); }

    template <std::floating_point T> void silu(ConstVectorView<T> x, const TensorView<T, 1>& out) {
        assert(x.shape == out.shape);

        std::transform(x.data, x.data + x.size(), out.data, [](T x) { return silu(x); });
    }

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

        T mx = *std::max_element(a.data, a.data + a.size());

        for (size_t i = 0; i < n; ++i) {
            T p = std::exp(a.data[i] - mx);

            out.data[i] = p;
            sum += p;
        }

        for (size_t i = 0; i < n; ++i)
            out.data[i] /= sum;
    }

    template <std::floating_point T> T dot(ConstVectorView<T> a, ConstVectorView<T> b) {
        return std::inner_product(a.data, a.data + a.size(), b.data, T { 0 });
    }

}
