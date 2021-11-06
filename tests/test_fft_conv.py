from typing import Iterable, Union

import pytest
import torch
import torch.nn.functional as f
from torch import Tensor

from fft_conv_pytorch.fft_conv import _FFTConv, fft_conv


def _assert_almost_equal(x: Tensor, y: Tensor) -> bool:
    abs_error = torch.abs(x - y)
    assert abs_error.mean().item() < 1e-5
    assert abs_error.max().item() < 1e-4
    return True


def _gcd(x: int, y: int) -> int:
    while y:
        x, y = y, x % y
    return x


@pytest.mark.parametrize("in_channels", [1, 2, 3])
@pytest.mark.parametrize("out_channels", [1, 2, 3])
@pytest.mark.parametrize("groups", [1, 2, 3])
@pytest.mark.parametrize("kernel_size", [1, 2, 3])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("input_size", [7, 8])
def test_fft_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Iterable[int]],
    padding: Union[int, Iterable[int]],
    stride: Union[int, Iterable[int]],
    groups: int,
    bias: bool,
    ndim: int,
    input_size: int,
):
    torch_conv = getattr(f, f"conv{ndim}d")
    groups = _gcd(in_channels, _gcd(out_channels, groups))
    fft_conv_layer = _FFTConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=padding,
        stride=stride,
        groups=groups,
        bias=bias,
        ndim=ndim,
    )

    batch_size = 2  # TODO: Make this non-constant?
    dims = ndim * [input_size]
    signal = torch.randn(batch_size, in_channels, *dims)
    kwargs = dict(
        bias=fft_conv_layer.bias,
        padding=padding,
        stride=stride,
        groups=groups,
    )

    y0 = fft_conv_layer(signal)
    y1 = fft_conv(signal, fft_conv_layer.weight, **kwargs)
    y2 = torch_conv(signal, fft_conv_layer.weight, **kwargs)

    _assert_almost_equal(y0, y1)
    _assert_almost_equal(y0, y2)
    _assert_almost_equal(y1, y2)
