# haki

haki is a C++23 implementation of the Qwen3-0.6B forward pass on CPU. It takes token IDs as input and computes next-token logits.

## Build

From the repository root:

```sh
mkdir -p build
clang++ -std=c++23 -O3 src/main.cpp -o build/haki
```

## Run the tests

With [uv](https://docs.astral.sh/uv/) installed, compare the forward pass against PyTorch on one seeded input:

```sh
uv run pytest tests -q --component compute_logits --trials 1
```

## Next steps

- Add a per-layer KV cache and check that decoding one token at a time produces the same logits, within tolerance, as a full forward pass.
- Compare decode throughput and combined prefill and decode time against llama.cpp.
