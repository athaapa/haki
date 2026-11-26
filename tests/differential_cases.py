"""Seeded inputs for each public operation. Inputs are sent in insertion order."""
import random
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F

import torch_reference as ref


class Config(NamedTuple):
    hidden: int
    intermediate: int
    head_dim: int
    heads: int
    kv_heads: int
    layers: int
    epsilon: float
    theta: float


@dataclass
class Case:
    config: Config
    inputs: dict[str, torch.Tensor]
    expected: dict[str, torch.Tensor]


class Trial:
    def __init__(self, component, seed):
        self.random = rng = random.Random(seed)
        self.generator = torch.Generator().manual_seed(seed)
        self.tokens = rng.randint(1, 9)
        hidden = rng.randint(1, 13)
        intermediate = rng.randint(1, 17)
        head_dim = rng.choice((2, 4, 6, 8))
        kv_heads = rng.randint(1, 3)
        heads = kv_heads * rng.randint(1, 4)
        layers = rng.randint(1, 3) if component in ("compute_logits", "cached_logits") else 1
        self.config = Config(hidden, intermediate, head_dim, heads, kv_heads, layers,
                             rng.choice((1e-6, 1e-5, 1e-4)), rng.choice((100., 10000., 1e6)))
        self.inputs = {}
        self.zero_weights = seed % 10 == 0

    def draw(self, name, *shape, scale=1.0):
        tensor = torch.randn(shape, generator=self.generator, dtype=torch.float32) * scale
        self.inputs[name] = tensor
        return tensor

    def projection(self, name, rows, cols):
        return self.draw(name, rows, cols, scale=cols ** -0.5)

    def attention_weights(self, prefix=""):
        c = self.config
        q_width = c.heads * c.head_dim
        kv_width = c.kv_heads * c.head_dim
        return {
            "wq": self.projection(prefix + "wq", q_width, c.hidden),
            "wk": self.projection(prefix + "wk", kv_width, c.hidden),
            "wv": self.projection(prefix + "wv", kv_width, c.hidden),
            "wo": self.projection(prefix + "wo", c.hidden, q_width),
            "qn": self.draw(prefix + "qn", c.head_dim),
            "kn": self.draw(prefix + "kn", c.head_dim),
        }

    def mlp_weights(self, prefix=""):
        c = self.config
        return {
            "gate": self.projection(prefix + "gate", c.intermediate, c.hidden),
            "up": self.projection(prefix + "up", c.intermediate, c.hidden),
            "down": self.projection(prefix + "down", c.hidden, c.intermediate),
        }

    def layer_weights(self, prefix=""):
        return {
            "pre_attn": self.draw(prefix + "pre_attn", self.config.hidden),
            "pre_mlp": self.draw(prefix + "pre_mlp", self.config.hidden),
            **self.attention_weights(prefix),
            **self.mlp_weights(prefix),
        }


def sigmoid(trial):
    x = trial.draw("x", trial.config.hidden)
    return x.sigmoid()


def softmax(trial):
    x = trial.draw("x", trial.config.hidden)
    x.mul_(trial.random.choice((1., 100.)))
    return x.softmax(0)


def dot(trial):
    x = trial.draw("x", trial.config.hidden)
    b = trial.draw("b", *x.shape)
    return torch.dot(x, b)


def rms(trial):
    x = trial.draw("x", trial.config.hidden)
    return (x.square().mean() + trial.config.epsilon).sqrt()


def rms_norm(trial):
    x = trial.draw("x", trial.config.hidden)
    weights = trial.draw("norm", *x.shape)
    return ref.rms_norm(x, weights, trial.config.epsilon)


def rotate_vector(trial):
    a, b = trial.draw("x", 2).unbind()
    angle = trial.draw("angle", 1)[0]
    return torch.stack((angle.cos() * a - angle.sin() * b,
                        angle.sin() * a + angle.cos() * b))


def silu(trial):
    x = trial.draw("x", trial.tokens, trial.config.hidden)
    return F.silu(x)


def add(trial):
    x = trial.draw("x", trial.tokens, trial.config.hidden)
    b = trial.draw("b", *x.shape)
    return x + b


def element_mul(trial):
    x = trial.draw("x", trial.tokens, trial.config.hidden)
    b = trial.draw("b", *x.shape)
    return x * b


def scale(trial):
    x = trial.draw("x", trial.tokens, trial.config.hidden)
    scalar = trial.draw("scalar", 1)
    return x * scalar


def mat_vec(trial):
    matrix = trial.draw("x", trial.tokens, trial.config.hidden)
    vector = trial.draw("b", trial.config.hidden)
    return matrix @ vector


def mat_mul(trial):
    c = trial.config
    a = trial.draw("x", trial.tokens, c.hidden)
    b = trial.draw("b", c.hidden, c.intermediate)
    return a @ b


def mat_mul_transposed_rhs(trial):
    c = trial.config
    a = trial.draw("x", trial.tokens, c.hidden)
    b = trial.draw("b", c.intermediate, c.hidden)
    return F.linear(a, b)


def multi_head(trial):
    x = trial.draw("x", trial.tokens, trial.config.hidden)
    return ref.multi_head(x, trial.attention_weights(), trial.config)


def mlp(trial):
    x = trial.draw("x", trial.tokens, trial.config.hidden)
    weights = trial.mlp_weights()
    if trial.zero_weights:
        weights["gate"].zero_()
    return ref.mlp(x, weights)


def transformer_block(trial):
    x = trial.draw("x", trial.tokens, trial.config.hidden)
    weights = trial.layer_weights()
    if trial.zero_weights:
        weights["wo"].zero_()
        weights["down"].zero_()
    return ref.block(x, weights, trial.config)


def qk_norm(trial):
    c = trial.config
    q = trial.draw("q", trial.tokens, c.heads * c.head_dim)
    heads = q.reshape(trial.tokens, c.heads, c.head_dim)
    norm = trial.draw("norm", c.head_dim)
    return ref.rms_norm(heads, norm, c.epsilon).reshape_as(q)


def rotate_pairs(trial):
    c = trial.config
    q = trial.draw("q", trial.tokens, c.heads * c.head_dim)
    return ref.rotate_pairs(q, c.head_dim, c.theta)


def rope(trial):
    c = trial.config
    q = trial.draw("q", trial.tokens, c.heads * c.head_dim)
    k = trial.draw("k", trial.tokens, c.kv_heads * c.head_dim)
    return {"q": ref.rotate_pairs(q, c.head_dim, c.theta),
            "k": ref.rotate_pairs(k, c.head_dim, c.theta)}


def attention_weights(trial):
    q = trial.draw("q", trial.tokens, trial.config.head_dim)
    k = trial.draw("k", *q.shape)
    if trial.zero_weights:
        q.zero_()  # Equal scores over the visible prefix.
    return ref.attention_weights(q, k)


def attention_head(trial):
    weights = attention_weights(trial)
    values = trial.draw("v", trial.tokens, trial.config.hidden)
    return weights @ values


def token_inputs(trial):
    vocab = trial.random.randint(2, 23)
    ids = torch.randint(vocab, (trial.tokens,), generator=trial.generator, dtype=torch.int32)
    trial.inputs["ids"] = ids
    return ids, trial.draw("embedding", vocab, trial.config.hidden)


def embedding(trial):
    ids, table = token_inputs(trial)
    return F.embedding(ids, table)


def compute_logits(trial):
    ids, table = token_inputs(trial)
    x = F.embedding(ids, table)
    c = trial.config
    for i in range(c.layers):
        x = ref.block(x, trial.layer_weights(f"layer{i}."), c)
    final = ref.rms_norm(x[-1], trial.draw("final_norm", c.hidden), c.epsilon)
    return F.linear(final, table)


def cached_logits(trial):
    ids, table = token_inputs(trial)
    c = trial.config
    layers = [trial.layer_weights(f"layer{i}.") for i in range(c.layers)]
    norm = trial.draw("final_norm", c.hidden)

    def full_prefix(tokens):
        x = F.embedding(tokens, table)
        for weights in layers:
            x = ref.block(x, weights, c)
        return F.linear(ref.rms_norm(x[-1], norm, c.epsilon), table)

    prefixes = torch.stack([full_prefix(ids[:end]) for end in range(1, len(ids) + 1)])
    chunk_ends = list(range(2, len(ids) + 1, 2))
    if len(ids) % 2:
        chunk_ends.append(len(ids))
    return {
        "single_token_steps": prefixes,
        "two_token_chunks": prefixes[[end - 1 for end in chunk_ends]],
        "full_prefill": prefixes[-1],
        "reset_new_sequence": full_prefix(ids.flip(0)),
    }


COMPONENTS = {fn.__name__: fn for fn in (
    sigmoid, softmax, dot, rms, rms_norm, rotate_vector, silu, add, element_mul,
    scale, mat_vec, mat_mul, mat_mul_transposed_rhs, multi_head, mlp, transformer_block,
    rotate_pairs, rope, qk_norm, attention_weights, attention_head, embedding, compute_logits, cached_logits,
)}


def make_case(reference, seed):
    trial = Trial(reference.__name__, seed)
    expected = reference(trial)
    if isinstance(expected, torch.Tensor):
        expected = {"output": torch.atleast_1d(expected)}
    return Case(trial.config, trial.inputs, expected)
