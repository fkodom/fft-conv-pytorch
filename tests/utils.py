import torch
from torch import Tensor


def _assert_almost_equal(x: Tensor, y: Tensor) -> bool:
    abs_error = torch.abs(x - y)
    assert abs_error.mean().item() < 5e-5
    assert abs_error.max().item() < 1e-4
    return True


def _gcd(x: int, y: int) -> int:
    while y:
        x, y = y, x % y
    return x
