"""Regenerate small CPU float32 reference fixtures; requires PyTorch."""
from pathlib import Path
import json
from fixtures_to_header import generate_header
import torch
import torch.nn.functional as F

torch.manual_seed(2718)
torch.set_num_threads(1)
N, H, HQ, HK, D, I = 3, 6, 4, 2, 4, 7
EPS, THETA = 1e-6, 10000.0
fixtures = {}
def save(name, x):
    fixtures[name] = {'shape': list(x.shape), 'dtype': 'float32',
                      'data': x.detach().contiguous().flatten().tolist()}
    return x

def rms(x, w):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS) * w

def rope(x):
    angles = torch.arange(N, dtype=torch.float32)[:, None] * THETA ** (-torch.arange(0, D, 2, dtype=torch.float32) / D)
    angles = torch.cat((angles, angles), -1)[:, None, :]
    a, b = x.chunk(2, -1)
    return x * angles.cos() + torch.cat((-b, a), -1) * angles.sin()

x = save('input', torch.randn(N, H) * 0.4)
wq = save('wq', torch.randn(HQ * D, H) * 0.3)
wk = save('wk', torch.randn(HK * D, H) * 0.3)
wv = save('wv', torch.randn(HK * D, H) * 0.3)
wo = save('wo', torch.randn(H, HQ * D) * 0.3)
qn = save('qn', torch.tensor([0.7, 1.2, 0.8, 1.4]))
kn = save('kn', torch.tensor([1.3, 0.6, 1.1, 0.9]))
gate = save('gate', torch.randn(I, H) * 0.3)
up = save('up', torch.randn(I, H) * 0.3)
down = save('down', torch.randn(H, I) * 0.3)
pre_attn = save('pre_attn', torch.linspace(0.6, 1.4, H))
pre_mlp = save('pre_mlp', torch.linspace(1.3, 0.7, H))
q = torch.randn(N, HQ, D)
k = torch.randn(N, HK, D)
v = torch.randn(N, HK, D)
save('q', q); save('k', k); save('v', v)
save('rope_q', rope(q)); save('rope_k', rope(k))


def head(q, k, v):
    scores = q @ k.T / D ** 0.5
    scores = scores.masked_fill(torch.ones(N, N, dtype=torch.bool).triu(1), -torch.inf)
    return scores.softmax(-1) @ v

# Nonzero query/KV head indices expose incorrect packed-row offsets.
save('head_expected', head(q[:, 3], k[:, 1], v[:, 1]))
save('uniform_expected', head(torch.zeros_like(q[:, 3]), torch.zeros_like(k[:, 1]), v[:, 1]))

def attention(x):
    q = rope(rms((x @ wq.T).reshape(N, HQ, D), qn))
    k = rope(rms((x @ wk.T).reshape(N, HK, D), kn))
    v = (x @ wv.T).reshape(N, HK, D)
    return torch.stack([head(q[:, h], k[:, h // (HQ // HK)], v[:, h // (HQ // HK)]) for h in range(HQ)], 1).reshape(N, HQ * D) @ wo.T

def mlp(x):
    return (F.silu(x @ gate.T) * (x @ up.T)) @ down.T

save('attention_expected', attention(x))
save('mlp_expected', mlp(x))
y = x + attention(rms(x, pre_attn))
save('block_expected', y + mlp(rms(y, pre_mlp)))
document = {
    'config': {
        'num_tokens': N, 'hidden_size': H, 'num_attention_heads': HQ,
        'num_key_value_heads': HK, 'head_dim': D, 'intermediate_size': I,
        'num_hidden_layers': 1, 'rms_norm_eps': EPS, 'rope_theta': THETA,
    },
    'attention_test': {'query_head': 3, 'kv_head': 1},
    'seed': 2718,
    'torch_version': torch.__version__,
    'tensors': fixtures,
}
fixture_dir = Path(__file__).resolve().parent / 'fixtures'
fixture_dir.mkdir(exist_ok=True)
source = fixture_dir / 'reference.json'
source.write_text(json.dumps(document, indent=2, allow_nan=False) + '\n')
generate_header(source, fixture_dir / 'reference.h')
