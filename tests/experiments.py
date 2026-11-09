"""Inspect a full transformer block using the existing C++ test fixture."""
import torch
import torch.nn.functional as F
from fixture_data import load_fixtures

metadata, tensors = load_fixtures()
config = metadata['config']
x = tensors['input']
n = x.shape[0]
num_heads = config['num_attention_heads']
num_kv_heads = config['num_key_value_heads']
head_dim = config['head_dim']
epsilon = config['rms_norm_eps']
theta = config['rope_theta']

torch.set_printoptions(precision=7, linewidth=140, threshold=10000)


def show(name, tensor):
    print(f'\n{name} — shape {list(tensor.shape)}\n{tensor}')


def rms_norm(vectors, parameters):
    # Normalize the last axis: hidden_size for tokens, head_dim for heads.
    return vectors * torch.rsqrt(vectors.square().mean(-1, keepdim=True) + epsilon) * parameters


def rope(vectors, angles):
    first, second = vectors.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return vectors * angles.cos() + rotated * angles.sin()


# show('block_input', x)
input_norm = rms_norm(x, tensors['pre_attn'])
# show('pre_attention_norm', input_norm)

q_projected = input_norm @ tensors['wq'].T
k_projected = input_norm @ tensors['wk'].T
v_projected = input_norm @ tensors['wv'].T
# show('k_projected', k_projected.reshape(n, -1))
# show('kn', tensors['kn'])

q_normalized = rms_norm(q_projected.reshape(n, num_heads, head_dim), tensors['qn'])
k_normalized = rms_norm(k_projected.reshape(n, num_kv_heads, head_dim), tensors['kn'])
# show('q_normalized (packed)', q_normalized.reshape(n, -1))
# show('k_normalized (packed)', k_normalized.reshape(n, -1))

positions = torch.arange(n, dtype=torch.float32)
frequencies = theta ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
angles = positions[:, None] * frequencies[None, :]
#show('RoPE pair angles [token, pair]', angles)
angles = torch.cat((angles, angles), dim=-1)[:, None, :]
q = rope(q_normalized, angles)
k = rope(k_normalized, angles)
v = v_projected.reshape(n, num_kv_heads, head_dim)
# show('q_after_rope (packed)', q.reshape(n, -1))
# show('k_after_rope (packed)', k.reshape(n, -1))

mask = torch.ones(n, n, dtype=torch.bool).triu(1)
head_outputs = []
for h in range(num_heads):
    kv_h = h // (num_heads // num_kv_heads)
    query, key, value = q[:, h, :], k[:, kv_h, :], v[:, kv_h, :]
    dots = query @ key.T
    scores = dots / head_dim ** 0.5
    masked_scores = scores.masked_fill(mask, -torch.inf)
    weights = masked_scores.softmax(dim=-1)
    head_output = weights @ value
    head_outputs.append(head_output)

concatenated = torch.stack(head_outputs, dim=1).reshape(n, num_heads * head_dim)
attention_output = concatenated @ tensors['wo'].T
# show('concatenated_heads (packed)', concatenated)
# show('attention_output_after_o_proj', attention_output)

after_attention = x + attention_output
# show('after_attention_residual', after_attention)

mlp_input = rms_norm(after_attention, tensors['pre_mlp'])
# show('pre_mlp_norm', mlp_input)
gate_projected = mlp_input @ tensors['gate'].T
up_projected = mlp_input @ tensors['up'].T
# show('gate_projected', gate_projected)
# show('up_projected', up_projected)
gate_activated = F.silu(gate_projected)
# show('gate_after_silu', gate_activated)
hidden = gate_activated * up_projected
# show('mlp_hidden', hidden)
mlp_output = hidden @ tensors['down'].T
show('mlp_output_after_down_proj', mlp_output)

output = after_attention + mlp_output
# show('block_output', output)
