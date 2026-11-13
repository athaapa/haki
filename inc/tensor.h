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

namespace haki {
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
            return { data + (i * shape[1]), { shape[1] } };
        }
    };

    template <std::floating_point T, size_t Rank> struct Tensor {
        std::vector<T> storage;
        std::array<size_t, Rank> shape;

        explicit Tensor(std::array<size_t, Rank> dims)
            : storage(
                  std::accumulate(dims.begin(), dims.end(), size_t { 1 }, std::multiplies<> { }))
            , shape(dims) { }

        TensorView<T, Rank> view() { return { storage.data(), shape }; }
        TensorView<const T, Rank> view() const { return { storage.data(), shape }; }
        TensorView<const T, Rank> cview() const { return view(); }
    };

    template <std::floating_point T> using ConstVectorView = TensorView<const T, 1>;

    template <std::floating_point T> using ConstMatrixView = TensorView<const T, 2>;
}
