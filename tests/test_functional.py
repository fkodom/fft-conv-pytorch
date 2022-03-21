from typing import Iterable, Union

import pytest
import torch
import torch.nn.functional as f

from fft_conv_pytorch.fft_conv import fft_conv, to_ntuple
from fft_conv_pytorch.utils import _assert_almost_equal, _gcd


@pytest.mark.parametrize("in_channels", [2, 3])
@pytest.mark.parametrize("out_channels", [2, 3])
@pytest.mark.parametrize("groups", [1, 2, 3])
@pytest.mark.parametrize("kernel_size", [2, 3])
@pytest.mark.parametrize("padding", [0, 1, "same"])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("bias", [True])
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("input_size", [7, 8])
def test_fft_conv_functional(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Iterable[int]],
    padding: Union[int, Iterable[int]],
    stride: Union[int, Iterable[int]],
    dilation: Union[int, Iterable[int]],
    groups: int,
    bias: bool,
    ndim: int,
    input_size: int,
):
    if padding == "same" and (stride != 1 or dilation != 1):
        # padding='same' is not compatible with strided convolutions
        return

    torch_conv = getattr(f, f"conv{ndim}d")
    groups = _gcd(in_channels, _gcd(out_channels, groups))

    batch_size = 2  # TODO: Make this non-constant?
    dims = ndim * [input_size]
    signal = torch.randn(batch_size, in_channels, *dims)
    kwargs = dict(
        bias=torch.randn(out_channels) if bias else None,
        padding=padding,
        stride=stride,
        dilation=dilation,
        groups=groups,
    )

    kernel_size = to_ntuple(kernel_size, n=signal.ndim - 2)
    w0 = torch.randn(
        out_channels, in_channels // groups, *kernel_size, requires_grad=True
    )
    w1 = w0.detach().clone().requires_grad_()

    b0 = torch.randn(out_channels, requires_grad=True) if bias else None
    b1 = b0.detach().clone().requires_grad_() if bias else None

    kwargs = dict(
        padding=padding,
        stride=stride,
        dilation=dilation,
        groups=groups,
    )

    y0 = fft_conv(signal, w0, bias=b0, **kwargs)
    y1 = torch_conv(signal, w1, bias=b1, **kwargs)

    _assert_almost_equal(y0, y1)


@pytest.mark.parametrize("in_channels", [2, 3])
@pytest.mark.parametrize("out_channels", [2, 3])
@pytest.mark.parametrize("groups", [1, 2, 3])
@pytest.mark.parametrize("kernel_size", [2, 3])
@pytest.mark.parametrize("padding", [0, 1, "same"])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("bias", [True])
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("input_size", [7, 8])
def test_fft_conv_backward_functional(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Iterable[int]],
    padding: Union[int, Iterable[int]],
    stride: Union[int, Iterable[int]],
    dilation: Union[int, Iterable[int]],
    groups: int,
    bias: bool,
    ndim: int,
    input_size: int,
):
    if padding == "same" and (stride != 1 or dilation != 1):
        # padding='same' is not compatible with strided convolutions
        return

    torch_conv = getattr(f, f"conv{ndim}d")
    groups = _gcd(in_channels, _gcd(out_channels, groups))

    batch_size = 2  # TODO: Make this non-constant?
    dims = ndim * [input_size]
    signal = torch.randn(batch_size, in_channels, *dims)

    kernel_size = to_ntuple(kernel_size, n=signal.ndim - 2)
    w0 = torch.randn(
        out_channels, in_channels // groups, *kernel_size, requires_grad=True
    )
    w1 = w0.detach().clone().requires_grad_()

    b0 = torch.randn(out_channels, requires_grad=True) if bias else None
    b1 = b0.detach().clone().requires_grad_() if bias else None

    kwargs = dict(
        padding=padding,
        stride=stride,
        dilation=dilation,
        groups=groups,
    )

    y0 = fft_conv(signal, w0, bias=b0, **kwargs)
    y1 = torch_conv(signal, w1, bias=b1, **kwargs)

    # Compute pseudo-loss and gradient
    y0.sum().backward()
    y1.sum().backward()

    _assert_almost_equal(w0.grad, w1.grad)

    if bias:
        _assert_almost_equal(b0.grad, b1.grad)
