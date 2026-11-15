// One invocation runs one case. Tensor records contain Rank uint32 extents and
// contiguous float32 data, both in host byte order (the runner is local).
#include "nn.h"
#include <cstdint>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

using namespace haki;
using Matrix = Tensor<float, 2>;
using Vector = Tensor<float, 1>;

template <class T> void read_exact(T* data, size_t count) {
    if (!std::cin.read(reinterpret_cast<char*>(data), count * sizeof(T))) {
        throw std::runtime_error("incomplete input tensor");
    }
}

template <size_t Rank> Tensor<float, Rank> read_tensor() {
    std::array<size_t, Rank> shape;
    for (auto& dim : shape) {
        uint32_t extent;
        read_exact(&extent, 1);
        dim = extent;
    }
    Tensor<float, Rank> tensor(shape);
    read_exact(tensor.storage.data(), tensor.storage.size());
    return tensor;
}

template <size_t Rank> void emit(const Tensor<float, Rank>& tensor) {
    for (uint32_t dim : tensor.shape) {
        std::cout.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    }
    std::cout.write(reinterpret_cast<const char*>(tensor.storage.data()),
        tensor.storage.size() * sizeof(float));
}

template <size_t Rank> Tensor<float, Rank> destination(std::array<size_t, Rank> shape) {
    Tensor<float, Rank> out(shape);
    std::fill(out.storage.begin(), out.storage.end(), std::numeric_limits<float>::quiet_NaN());
    return out;
}

void scalar(float value) {
    Vector out({ 1 });
    out.storage[0] = value;
    emit(out);
}

AttentionWeights<float> read_attention() {
    return { .q_proj = read_tensor<2>(),
        .k_proj = read_tensor<2>(),
        .v_proj = read_tensor<2>(),
        .o_proj = read_tensor<2>(),
        .q_norm = read_tensor<1>(),
        .k_norm = read_tensor<1>() };
}

MlpWeights<float> read_mlp() {
    return {
        .gate_proj = read_tensor<2>(), .up_proj = read_tensor<2>(), .down_proj = read_tensor<2>()
    };
}

std::vector<int> read_ids() {
    uint32_t count;
    read_exact(&count, 1);
    std::vector<int32_t> ids(count);
    read_exact(ids.data(), count);
    return { ids.begin(), ids.end() };
}

Qwen3LayerWeights<float> read_layer() {
    return { .pre_attention_norm = read_tensor<1>(),
        .pre_mlp_norm = read_tensor<1>(),
        .attention = read_attention(),
        .mlp = read_mlp() };
}

void run_add(const Qwen3Config& /*unused*/) {
    auto x = read_tensor<2>();
    auto b = read_tensor<2>();
    auto out = destination<2>(x.shape);
    add(x.cview(), b.cview(), out.view());
    emit(out);
}

void run_element_mul(const Qwen3Config& /*unused*/) {
    auto x = read_tensor<2>();
    auto b = read_tensor<2>();
    auto out = destination<2>(x.shape);
    element_mul(x.cview(), b.cview(), out.view());
    emit(out);
}

void run_silu(const Qwen3Config& /*unused*/) {
    auto x = read_tensor<2>();
    auto out = destination<2>(x.shape);
    silu(x.cview(), out.view());
    emit(out);
}

void run_softmax(const Qwen3Config& /*unused*/) {
    auto x = read_tensor<1>();
    auto out = destination<1>(x.shape);
    softmax(x.cview(), out.view());
    emit(out);
}

void run_scale(const Qwen3Config& /*unused*/) {
    auto x = read_tensor<2>();
    auto scalar = read_tensor<1>();
    auto out = destination<2>(x.shape);
    scale(x.cview(), scalar.storage[0], out.view());
    emit(out);
}

void run_rotate_vector(const Qwen3Config& /*unused*/) {
    auto x = read_tensor<1>();
    auto angle = read_tensor<1>();
    auto out = destination<1>(x.shape);
    rotate_vector(x.cview(), angle.storage[0], out.view());
    emit(out);
}

void run_rms_norm(const Qwen3Config& config) {
    auto x = read_tensor<1>();
    auto weights = read_tensor<1>();
    auto out = destination<1>(x.shape);
    rms_norm(x.cview(), static_cast<float>(config.rms_norm_eps), weights.cview(), out.view());
    emit(out);
}

void run_qk_norm(const Qwen3Config& config) {
    auto x = read_tensor<2>();
    auto weights = read_tensor<1>();
    auto out = destination<2>(x.shape);
    qk_norm(x.cview(), weights.cview(), static_cast<float>(config.rms_norm_eps), out.view());
    emit(out);
}

void run_rotate_pairs(const Qwen3Config& config) {
    auto x = read_tensor<2>();
    auto out = destination<2>(x.shape);
    rotate_pairs(x.cview(), config.head_dim, static_cast<float>(config.rope_theta), out.view());
    emit(out);
}

void run_sigmoid(const Qwen3Config& /*unused*/) {
    auto x = read_tensor<1>();
    auto out = destination<1>(x.shape);
    for (size_t i = 0; i < x.storage.size(); ++i) {
        out.storage[i] = sigmoid(x.storage[i]);
    }
    emit(out);
}

void run_dot(const Qwen3Config& /*unused*/) {
    auto x = read_tensor<1>();
    auto b = read_tensor<1>();
    scalar(dot(x.cview(), b.cview()));
}

void run_rms(const Qwen3Config& config) {
    auto x = read_tensor<1>();
    scalar(rms(x.cview(), static_cast<float>(config.rms_norm_eps)));
}

void run_mat_vec(const Qwen3Config& /*unused*/) {
    auto a = read_tensor<2>();
    auto b = read_tensor<1>();
    auto out = destination<1>({ a.shape[0] });
    mat_mul(a.cview(), b.cview(), out.view());
    emit(out);
}

void run_mat_mul(const Qwen3Config& /*unused*/) {
    auto a = read_tensor<2>();
    auto b = read_tensor<2>();
    auto out = destination<2>({ a.shape[0], b.shape[1] });
    mat_mul(a.cview(), b.cview(), out.view());
    emit(out);
}

void run_mat_mul_transposed_rhs(const Qwen3Config& /*unused*/) {
    auto a = read_tensor<2>();
    auto b = read_tensor<2>();
    auto out = destination<2>({ a.shape[0], b.shape[0] });
    mat_mul_transposed_rhs(a.cview(), b.cview(), out.view());
    emit(out);
}

void run_attention_weights(const Qwen3Config& /*unused*/) {

    auto q = read_tensor<2>();
    auto k = read_tensor<2>();
    auto out = destination<2>({ q.shape[0], q.shape[0] });
    compute_attention_weights(q.cview(), k.cview(), out.view());
    emit(out);
}

void run_attention_head(const Qwen3Config& config) {
    auto q = read_tensor<2>();
    auto k = read_tensor<2>();
    auto v = read_tensor<2>();
    auto out = destination<2>(v.shape);
    self_attention_head(q.cview(), k.cview(), v.cview(), config, out.view());
    emit(out);
}

void run_rope(const Qwen3Config& config) {
    auto q = read_tensor<2>();
    auto k = read_tensor<2>();
    auto q_out = destination<2>(q.shape);
    auto k_out = destination<2>(k.shape);
    rope(q.cview(), k.cview(), config, q_out.view(), k_out.view());
    emit(q_out);
    emit(k_out);
}

void run_multi_head(const Qwen3Config& config) {
    auto x = read_tensor<2>();
    auto out = destination<2>(x.shape);
    auto weights = read_attention();
    multi_head_attention(x.cview(), weights, config, out.view());
    emit(out);
}

void run_mlp(const Qwen3Config& /*unused*/) {
    auto x = read_tensor<2>();
    auto out = destination<2>(x.shape);
    auto weights = read_mlp();
    swiglu_mlp(x.cview(), weights, out.view());
    emit(out);
}

void run_block(const Qwen3Config& config) {
    auto x = read_tensor<2>();
    auto out = destination<2>(x.shape);
    auto layer = read_layer();
    Qwen3<float> model { .config = config,
        .weights = { .token_embedding = Matrix({ 0, x.shape[1] }),
            .layers = { std::move(layer) },
            .final_norm = Vector({ x.shape[1] }) } };
    transformer_block(model, 0, x.cview(), out.view());
    emit(out);
}

void run_embedding(const Qwen3Config& config) {
    auto ids = read_ids();
    auto table = read_tensor<2>();
    auto out = destination<2>({ ids.size(), config.hidden_dim });
    id_to_embedding(ids, table.cview(), out.view());
    emit(out);
}

void run_compute_logits(const Qwen3Config& config) {
    auto ids = read_ids();
    auto embedding = read_tensor<2>();
    std::vector<Qwen3LayerWeights<float>> layers;
    layers.reserve(config.num_hidden_layers);
    for (size_t i = 0; i < config.num_hidden_layers; ++i) {
        layers.push_back(read_layer());
    }
    Qwen3<float> model { .config = config,
        .weights = { .token_embedding = std::move(embedding),
            .layers = std::move(layers),
            .final_norm = read_tensor<1>() } };
    auto logits = destination<1>({ model.weights.token_embedding.shape[0] });
    compute_logits(model, ids, logits.view());
    emit(logits);
}

using Handler = void (*)(const Qwen3Config&);
const std::map<std::string, Handler> handlers {
    { "add", run_add },
    { "element_mul", run_element_mul },
    { "silu", run_silu },
    { "softmax", run_softmax },
    { "scale", run_scale },
    { "rotate_vector", run_rotate_vector },
    { "rms_norm", run_rms_norm },
    { "qk_norm", run_qk_norm },
    { "rotate_pairs", run_rotate_pairs },
    { "sigmoid", run_sigmoid },
    { "dot", run_dot },
    { "rms", run_rms },
    { "mat_vec", run_mat_vec },
    { "mat_mul", run_mat_mul },
    { "mat_mul_transposed_rhs", run_mat_mul_transposed_rhs },
    { "attention_weights", run_attention_weights },
    { "attention_head", run_attention_head },
    { "rope", run_rope },
    { "multi_head", run_multi_head },
    { "mlp", run_mlp },
    { "block", run_block },
    { "embedding", run_embedding },
    { "compute_logits", run_compute_logits },
};

int main(int argc, char** argv) {
    // NOLINTBEGIN(readability-magic-numbers)
    if (argc != 10) {
        return 2;
    }
    try {
        Qwen3Config c { .hidden_dim = std::stoul(argv[2]),
            .intermediate_dim = std::stoul(argv[3]),
            .head_dim = std::stoul(argv[4]),
            .num_attention_heads = std::stoul(argv[5]),
            .num_key_value_heads = std::stoul(argv[6]),
            .num_hidden_layers = std::stoul(argv[7]),
            .rms_norm_eps = std::stod(argv[8]),
            .rope_theta = std::stod(argv[9]) };
        handlers.at(argv[1])(c);
        if (std::cin.peek() != EOF) {
            throw std::runtime_error("unused input bytes");
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    // NOLINTEND(readability-magic-numbers)
}
