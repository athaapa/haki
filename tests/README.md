# Shared reference fixtures

Activate the project environment:

```sh
source .venv/bin/activate
```

`fixtures/reference.json` stores tensor shapes, float32 data, configuration, and selected head indices. Python reads it directly; C++ uses the generated `fixtures/reference.h` with identical flattened data. Q/K/V retain `[tokens, heads, head_dim]` shapes in JSON; C++ views pack the last two dimensions into columns without changing element order.

Explore attention interactively:

```sh
python -i tests/inspect_attention.py
```

For example, enter `scores`, `weights`, or `q[1] * k[0]`. `scores` is scaled but unmasked; `dots` is unscaled. Rows are query tokens and columns are source tokens. No fixture regeneration occurs during inspection.

Regenerate reference data with PyTorch and update the header:

```sh
python tests/generate_fixtures.py
```

After manually editing JSON, update only the header (standard library only):

```sh
python tests/fixtures_to_header.py
```

Run C++ tests:

```sh
python tests/run_tests.py
python tests/run_tests.py attention_head
```

The runner compiles groups separately and reports compilation errors as failures. Add `--sanitize` for AddressSanitizer/UndefinedBehaviorSanitizer.
