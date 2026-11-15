# haki

## Differential tests

Run from the repository root with a C++23 compiler and a Python virtual environment.

```sh
python -m pip install -r tests/requirements.txt
python -m pytest tests -q
```

Before running any tests, pytest builds haki and its C++ test driver using
`clang++` (override with `CXX`). A build failure stops the session and prints the
compiler diagnostic. Each test generates CPU float32 tensors and weights from its seed and sends
their exact bytes to the driver. PyTorch computes the reference from the same tensors.
The default suite runs seeds 0–99 independently for each component. To replay or
customize a run:

```sh
python -m pytest tests -q --component multi_head --seed 17 --trials 1
python -m pytest tests -q --component block --trials 50 --atol 1e-5 --rtol 1e-4
```

Every element must satisfy `abs(actual - expected) <= atol + rtol * abs(expected)`.
The defaults are `atol=1e-5` and `rtol=1e-4`. Either side being NaN or infinite
fails the comparison. The logits test compares every vocabulary entry. Failures identify the output
coordinate with the largest tolerance overrun. The summary reports the mean,
median, and maximum of each trial's largest absolute error, grouped by component.
Use `--sanitize` to run the driver with AddressSanitizer and UndefinedBehaviorSanitizer.
