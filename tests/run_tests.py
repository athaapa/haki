"""Compatibility entry point for the pytest differential suite."""
from pathlib import Path
import sys

import pytest

groups = {
    "math": ["mat_mul", "mat_mul_transposed_rhs", "silu"],
    "nn": ["rms_norm", "softmax"],
    "primitives": ["dot", "mat_vec", "element_mul"],
    "attention_head": ["attention_head", "attention_weights"],
}
if __name__ == "__main__":
    args = sys.argv[1:]
    if args and not args[0].startswith("-"):
        group = args.pop(0)
        args = [arg for op in groups.get(group, [group]) for arg in ("--component", op)] + args
    sys.exit(pytest.main([str(Path(__file__).with_name("test_differential.py")), *args]))
