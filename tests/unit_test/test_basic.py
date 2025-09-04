import math

import pytest


def add(a, b):
    return a + b


def test_add_works():
    assert add(1, 2) == 3


@pytest.mark.parametrize("x,expected", [(0, 0.0), (math.pi, 0.0)])
def test_sin(x, expected):
    assert math.isclose(math.sin(x), expected, abs_tol=1e-9)


@pytest.mark.slow
def test_slow_example():
    assert sum(range(10000)) > 0


@pytest.mark.gpu
def test_gpu_feature():
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("No CUDA available")
    x = torch.tensor([1.0], device="cuda")
    assert float(x.item()) == 1.0
