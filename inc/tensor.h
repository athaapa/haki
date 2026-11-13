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
        std::array<size_t, Rank> strides;

        TensorView(T* pointer, std::array<size_t, Rank> dimensions)
            : data(pointer)
            , shape(dimensions) {
            size_t stride = 1;
            for (size_t axis = Rank; axis-- > 0;) {
                strides[axis] = stride;
                assert(
                    shape[axis] == 0 || stride <= std::numeric_limits<size_t>::max() / shape[axis]);
                stride *= shape[axis];
            }
            assert(data != nullptr || size() == 0);
        }

        TensorView(T* pointer, std::array<size_t, Rank> dimensions,
            std::array<size_t, Rank> element_strides)
            : data(pointer)
            , shape(dimensions)
            , strides(element_strides) {
            assert(data != nullptr || size() == 0);
        }

        template <std::integral... Indices>
            requires(sizeof...(Indices) == Rank)
        T& operator[](Indices... indices) const {
            const std::array<size_t, Rank> coordinates { static_cast<size_t>(indices)... };
            size_t offset = 0;
            for (size_t axis = 0; axis < Rank; ++axis) {
                assert(coordinates[axis] < shape[axis]);
                offset += coordinates[axis] * strides[axis];
            }
            assert(data != nullptr);
            return data[offset];
        }

        // Logical row-major traversal, including noncontiguous views.
        T& flat(size_t index) const {
            assert(index < size());
            size_t offset = 0;
            for (size_t axis = Rank; axis-- > 0;) {
                offset += (index % shape[axis]) * strides[axis];
                index /= shape[axis];
            }
            assert(data != nullptr);
            return data[offset];
        }

        [[nodiscard]] bool is_contiguous() const {
            if (size() == 0) {
                return true;
            }
            size_t expected = 1;
            for (size_t axis = Rank; axis-- > 0;) {
                if (shape[axis] > 1 && strides[axis] != expected) {
                    return false;
                }
                expected *= shape[axis];
            }
            return true;
        }

        TensorView narrow(size_t axis, size_t start, size_t length) const {
            assert(axis < Rank);
            assert(start <= shape[axis]);
            assert(length <= shape[axis] - start);
            auto result = *this;
            result.shape[axis] = length;
            // Empty views need no pointer adjustment (possibly beyond one-past).
            if (result.size() != 0) {
                result.data += start * strides[axis];
            }
            return result;
        }

        TensorView<const std::remove_const_t<T>, Rank> cview() const {
            return { data, shape, strides };
        }

        [[nodiscard]] size_t size() const {
            const size_t n
                = std::accumulate(shape.begin(), shape.end(), size_t { 1 }, std::multiplies<> {});
            return n;
        }

        TensorView<T, 1> row(size_t i) const
            requires(Rank == 2)
        {
            assert(i < shape[0]);
            return { shape[1] == 0 ? data : data + (i * strides[0]), { shape[1] }, { strides[1] } };
        }
    };

    template <std::floating_point T, size_t Rank> struct Tensor {
        std::vector<T> storage;
        std::array<size_t, Rank> shape;

        explicit Tensor(std::array<size_t, Rank> dims)
            : storage(std::accumulate(dims.begin(), dims.end(), size_t { 1 }, std::multiplies<> {}))
            , shape(dims) { }

        TensorView<T, Rank> view() {
            TensorView<T, Rank> result { storage.data(), shape };
            assert(result.size() == storage.size());
            return result;
        }
        TensorView<const T, Rank> view() const {
            TensorView<const T, Rank> result { storage.data(), shape };
            assert(result.size() == storage.size());
            return result;
        }
        TensorView<const T, Rank> cview() const { return view(); }
    };

    template <std::floating_point T> using ConstVectorView = TensorView<const T, 1>;

    template <std::floating_point T> using ConstMatrixView = TensorView<const T, 2>;
}
