import math
import os
from pathlib import Path
import statistics
import subprocess
from tempfile import TemporaryDirectory

import pytest

from differential_cases import COMPONENTS


def pytest_addoption(parser):
    group = parser.getgroup("haki differential tests")
    group.addoption("--component", choices=COMPONENTS, action="append",
                    help="Run only this component (repeatable)")
    group.addoption("--trials", type=int, default=100, help="Trials per component")
    group.addoption("--seed", type=int, default=0, help="First trial seed")
    group.addoption("--atol", type=float, default=1e-5)
    group.addoption("--rtol", type=float, default=1e-4)
    group.addoption("--sanitize", action="store_true", help="Enable C++ ASan and UBSan")


def pytest_configure(config):
    trials, seed = config.getoption("trials"), config.getoption("seed")
    if trials < 1 or seed < 0 or seed + trials > 2**63:
        raise pytest.UsageError("Require positive trials and trial seeds in [0, 2^63)")
    for key in ("atol", "rtol"):
        value = config.getoption(key)
        if not math.isfinite(value) or value < 0:
            raise pytest.UsageError(f"--{key} must be finite and nonnegative")


def pytest_generate_tests(metafunc):
    if "reference" in metafunc.fixturenames:
        config = metafunc.config
        first = config.getoption("seed")
        seeds = range(first, first + config.getoption("trials"))
        ops = config.getoption("component") or COMPONENTS
        metafunc.parametrize(("reference", "seed"), [
            pytest.param(COMPONENTS[op], seed, id=f"{op}-seed{seed}") for op in ops for seed in seeds
        ])


DRIVER = pytest.StashKey[Path]()


@pytest.hookimpl(tryfirst=True)
def pytest_runtestloop(session):
    config = session.config
    if config.option.collectonly or session.testsfailed or not session.items:
        return

    root = Path(__file__).resolve().parent.parent
    temporary = TemporaryDirectory(prefix="haki-tests-")
    config.add_cleanup(temporary.cleanup)
    build = Path(temporary.name)
    compiler = [os.environ.get("CXX", "clang++"), "-std=c++23", "-O1", "-iquote", str(root / "inc")]
    if config.getoption("sanitize"):
        compiler.extend(["-fsanitize=address,undefined", "-fno-sanitize-recover=all"])

    for source, target in [("src/main.cpp", "haki"), ("tests/differential_driver.cpp", "driver")]:
        try:
            result = subprocess.run([*compiler, root / source, "-o", build / target],
                                    stderr=subprocess.PIPE, text=True)
        except OSError as error:
            pytest.exit(f"Cannot run {compiler[0]}: {error}", returncode=1)
        if result.returncode:
            pytest.exit(f"Failed to build {target} from {source}:\n{result.stderr}", returncode=1)
    config.stash[DRIVER] = build / "driver"


@pytest.fixture(scope="session")
def driver(pytestconfig):
    return pytestconfig.stash[DRIVER]


ERRORS = pytest.StashKey[dict]()


@pytest.fixture(scope="session")
def trial_errors(pytestconfig):
    import torch
    torch.set_num_threads(1)
    pytestconfig.stash[ERRORS] = {}
    return pytestconfig.stash[ERRORS]


def pytest_terminal_summary(terminalreporter):
    errors = terminalreporter.config.stash.get(ERRORS, {})
    if errors:
        terminalreporter.write_sep("-", "absolute error across trials (one maximum per trial)")
        terminalreporter.write_line("component                  trials        mean      median         max")
        for op, values in errors.items():
            terminalreporter.write_line(f"{op:26} {len(values):6}  {statistics.mean(values):10.3g}  "
                                        f"{statistics.median(values):10.3g}  {max(values):10.3g}")
