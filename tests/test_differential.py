import math
import struct
import subprocess

import numpy as np
import pytest

from differential_cases import make_case


def encode(inputs):
    chunks = []
    for tensor in inputs.values():
        chunks.append(struct.pack(f"={tensor.ndim}I", *tensor.shape))
        chunks.append(tensor.contiguous().numpy().tobytes())
    return b"".join(chunks)


def decode(data, expected):
    outputs = {}
    offset = 0
    for name, tensor in expected.items():
        rank = tensor.ndim
        shape = struct.unpack_from(f"={rank}I", data, offset)
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} expected shape {tuple(tensor.shape)}, actual shape {shape}")
        offset += rank * 4
        array = np.frombuffer(data, dtype=np.float32, count=math.prod(shape), offset=offset)
        outputs[name] = array.reshape(shape)
        offset += array.nbytes
    if offset != len(data):
        raise ValueError("extra output bytes")
    return outputs


def compare(actual, expected, atol, rtol):
    # Compare in float64 to avoid rounding the tolerance calculation to float32.
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    finite = np.isfinite(actual) & np.isfinite(expected)
    limit = atol + rtol * np.abs(expected.astype(np.float64))
    error[~finite] = np.inf
    excess = error - np.where(finite, limit, 0.)
    index = np.unravel_index(excess.argmax(), excess.shape)
    violation = None
    if excess[index] > 0:
        violation = (float(excess[index]), f"index={tuple(map(int, index))} expected={expected[index]!s} "
                     f"actual={actual[index]!s} abs_error={error[index]:g} limit={limit[index]:g}")
    return float(error.max()), violation


def test_against_pytorch(reference, seed, driver, pytestconfig, trial_errors):
    component = reference.__name__
    case = make_case(reference, seed)
    shapes = {name: tuple(t.shape) for name, t in case.inputs.items()}
    context = f"component={component} seed={seed} config={case.config} input_shapes={shapes}"
    command = [driver, component, *map(str, case.config)]
    try:
        result = subprocess.run(command, input=encode(case.inputs), capture_output=True, timeout=10)
        if result.returncode:
            pytest.fail(f"{context}\nHaki exited {result.returncode}: {result.stderr.decode(errors='replace')}",
                        pytrace=False)
        actual = decode(result.stdout, case.expected)
    except (ValueError, struct.error, subprocess.TimeoutExpired) as error:
        pytest.fail(f"{context}\n{error}", pytrace=False)

    atol, rtol = pytestconfig.getoption("atol"), pytestconfig.getoption("rtol")
    maximum = 0.
    failures = []
    for name, tensor in case.expected.items():
        with np.errstate(invalid="ignore"):
            error, failure = compare(actual[name], tensor.numpy(), atol, rtol)
        maximum = max(maximum, error)
        if failure:
            excess, detail = failure
            failures.append((excess, f"{name} shape={tuple(tensor.shape)} worst {detail}"))
    trial_errors.setdefault(component, []).append(maximum)
    if failures:
        pytest.fail(f"{context}\n{max(failures)[1]}", pytrace=False)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("side", ["actual", "expected", "both"])
def test_reject_nonfinite(bad, side):
    expected = np.array([1., 2.])
    actual = expected.copy()
    if side in ("expected", "both"):
        expected[1] = bad
    if side in ("actual", "both"):
        actual[1] = bad
    with np.errstate(invalid="ignore"):
        error, failure = compare(actual, expected, 1e-5, 1e-4)
    assert error == np.inf and failure is not None
    assert "index=(1,)" in failure[1]


def test_one_element_over_tolerance():
    # The largest absolute error passes its relative tolerance. The smaller error fails.
    expected = np.array([1000., 0.])
    actual = np.array([1001., 0.125])
    assert compare(actual, expected, atol=0.125, rtol=0.001)[1] is None
    _, failure = compare(actual, expected, atol=0.1, rtol=0.001)
    assert failure is not None and "index=(1,)" in failure[1]
