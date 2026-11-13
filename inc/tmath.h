#pragma once
#include "tensor.h"
#include <algorithm>
#include <cmath>
#include <concepts>

namespace haki {

    template <std::floating_point T> T sigmoid(T x) { return 1 / (1 + std::exp(-x)); }

    template <std::floating_point T> T silu(T x) { return x * sigmoid(x); }

    template <std::floating_point T, size_t Rank>
    void silu(const TensorView<const T, Rank> x, const TensorView<T, Rank>& out) {
        assert(x.shape == out.shape);

        for (size_t i = 0; i < x.size(); ++i) {
            out.flat(i) = silu(x.flat(i));
        }
    }

    /*
     * @brief matrix multiplication with matrices a and b
     */
    template <std::floating_point T>
    void mat_mul(ConstMatrixView<T> a, ConstMatrixView<T> b, const TensorView<T, 2> out) {
        assert(a.shape[1] == b.shape[0]);

        assert(a.shape[0] == out.shape[0]);
        assert(b.shape[1] == out.shape[1]);

        size_t a_n = a.shape[0];
        size_t b_n = b.shape[0];

        size_t b_m = b.shape[1];

        for (size_t i = 0; i < a_n; ++i) { // row
            for (size_t j = 0; j < b_m; ++j) {
                T sum = 0;
                for (size_t k = 0; k < b_n; ++k) {
                    sum += a[i, k] * b[k, j];
                }
                out[i, j] = sum;
            }
        }
    }

    /*
     * @brief element-wise matrix multiplication
     */
    template <std::floating_point T, size_t Rank>
    void element_mul(
        TensorView<const T, Rank> a, TensorView<const T, Rank> b, const TensorView<T, Rank> out) {
        assert(a.shape == b.shape);
        assert(a.shape == out.shape);

        for (size_t i = 0; i < a.size(); ++i)
            out.flat(i) = a.flat(i) * b.flat(i);
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
            T sum = 0;
            for (size_t k = 0; k < cols; ++k)
                sum += a[i, k] * b[k];
            out[i] = sum;
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
                out[i, j] = dot(a.row(i), b.row(j));
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

        for (size_t i = 0; i < n; ++i)
            out.flat(i) = a.flat(i) + b.flat(i);
    }

    /*
     * @brief tensor scalar multiplication
     */
    template <std::floating_point T, size_t Rank>
    void scale(const TensorView<const T, Rank> a, T scalar, const TensorView<T, Rank> out) {
        assert(a.shape == out.shape);

        const size_t n = a.size();

        for (size_t i = 0; i < n; ++i)
            out.flat(i) = a.flat(i) * scalar;
    }

    // Subtract the maximum before exponentiation.
    template <std::floating_point T>
    void softmax(ConstVectorView<T> a, const TensorView<T, 1> out) {
        assert(a.shape == out.shape);

        size_t n = a.shape[0];

        T sum = 0;

        if (n == 0)
            return;
        T mx = a[0];
        for (size_t i = 1; i < n; ++i)
            mx = std::max(mx, a[i]);

        for (size_t i = 0; i < n; ++i) {
            T p = std::exp(a[i] - mx);

            out[i] = p;
            sum += p;
        }

        for (size_t i = 0; i < n; ++i)
            out[i] /= sum;
    }

    template <std::floating_point T> T dot(ConstVectorView<T> a, ConstVectorView<T> b) {
        assert(a.shape == b.shape);
        T sum = 0;
        for (size_t i = 0; i < a.size(); ++i)
            sum += a[i] * b[i];
        return sum;
    }

}
