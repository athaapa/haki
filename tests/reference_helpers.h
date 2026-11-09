#pragma once
#include "../inc/nn.h"
#include "fixtures/reference.h"
#include "test_helpers.h"

namespace test_data {
    inline haki::Qwen3Config config() { return { 6, 7, 4, 4, 2, 1, 1e-6, 10000.0 }; }
    template <size_t Rank>
    haki::Tensor<float, Rank> tensor(
        std::array<size_t, Rank> shape, const std::vector<float>& data) {
        haki::Tensor<float, Rank> result(shape);
        if (result.storage.size() != data.size())
            throw std::runtime_error("Fixture shape mismatch");
        result.storage = data;
        return result;
    }
    inline auto attention_weights() {
        return haki::AttentionWeights<float> { tensor<2>({ 16, 6 }, fixtures::wq),
            tensor<2>({ 8, 6 }, fixtures::wk), tensor<2>({ 8, 6 }, fixtures::wv),
            tensor<2>({ 6, 16 }, fixtures::wo), tensor<1>({ 4 }, fixtures::qn),
            tensor<1>({ 4 }, fixtures::kn) };
    }
    inline auto mlp_weights() {
        return haki::MlpWeights<float> { tensor<2>({ 7, 6 }, fixtures::gate),
            tensor<2>({ 7, 6 }, fixtures::up), tensor<2>({ 6, 7 }, fixtures::down) };
    }
    inline auto model() {
        haki::Qwen3LayerWeights<float> layer { tensor<1>({ 6 }, fixtures::pre_attn),
            tensor<1>({ 6 }, fixtures::pre_mlp), attention_weights(), mlp_weights() };
        return haki::Qwen3<float> { config(),
            { haki::Tensor<float, 2>({ 1, 6 }), { std::move(layer) },
                haki::Tensor<float, 1>({ 6 }) } };
    }
}
