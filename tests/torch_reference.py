"""PyTorch forward operations used by the differential tests."""
import math

import torch
import torch.nn.functional as F


def rms_norm(x, weight, epsilon):
    return F.rms_norm(x, (weight.numel(),), weight, epsilon)


def rotate_pairs(x, head_dim, theta):
    tokens = x.shape[0]
    heads = x.reshape(tokens, -1, head_dim)
    frequencies = theta ** (-torch.arange(0, head_dim, 2, dtype=x.dtype) / head_dim)
    positions = torch.arange(tokens, dtype=x.dtype)
    angles = torch.outer(positions, frequencies)[:, None, :]
    first, second = heads.chunk(2, dim=-1)
    real = first * angles.cos() - second * angles.sin()
    imag = second * angles.cos() + first * angles.sin()
    return torch.cat((real, imag), dim=-1).reshape_as(x)


def attention_weights(q, k):
    scores = q @ k.T / math.sqrt(q.shape[-1])
    future_tokens = torch.ones_like(scores, dtype=torch.bool).triu(1)
    return scores.masked_fill(future_tokens, -torch.inf).softmax(-1)


def multi_head(x, weights, config):
    tokens = x.shape[0]
    dim = config.head_dim
    queries = F.linear(x, weights["wq"]).reshape(tokens, config.heads, dim)
    keys = F.linear(x, weights["wk"]).reshape(tokens, config.kv_heads, dim)
    values = F.linear(x, weights["wv"]).reshape_as(keys)

    queries = rotate_pairs(rms_norm(queries, weights["qn"], config.epsilon), dim, config.theta)
    keys = rotate_pairs(rms_norm(keys, weights["kn"], config.epsilon), dim, config.theta)

    # PyTorch attention expects [batch, head, token, feature].
    group_size = config.heads // config.kv_heads
    queries = queries.transpose(0, 1).unsqueeze(0)
    keys = keys.transpose(0, 1).unsqueeze(0).repeat_interleave(group_size, dim=1)
    values = values.transpose(0, 1).unsqueeze(0).repeat_interleave(group_size, dim=1)
    heads = F.scaled_dot_product_attention(queries, keys, values, is_causal=True)
    concatenated = heads.squeeze(0).transpose(0, 1).reshape(tokens, -1)
    return F.linear(concatenated, weights["wo"])


def mlp(x, weights):
    gate = F.silu(F.linear(x, weights["gate"]))
    up = F.linear(x, weights["up"])
    return F.linear(gate * up, weights["down"])


def block(x, weights, config):
    normalized = rms_norm(x, weights["pre_attn"], config.epsilon)
    x = x + multi_head(normalized, weights, config)
    normalized = rms_norm(x, weights["pre_mlp"], config.epsilon)
    return x + mlp(normalized, weights)
