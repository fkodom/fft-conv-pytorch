from typing import Union, Iterable

import pytest
import torch
from torch import Tensor
import torch.nn.functional as f

from fft_conv_pytorch.fft_conv import fft_conv, _FFTConv


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


# def benchmark_conv(
#     signal: Tensor,
#     kernel: Tensor,
#     bias: Tensor,
#     padding: Union[int, Iterable[int]] = 0,
#     stride: Union[int, Iterable[int]] = 1,
#     groups: int = 1,
# ):
#     print(f"Signal size: {signal.shape}")
#     print(f"Kernel size: {kernel.shape}")

#     torch_conv = {1: f.conv1d, 2: f.conv2d, 3: f.conv3d}[signal.ndim - 2]
#     direct_time = benchmark(
#         torch_conv,
#         signal,
#         kernel,
#         bias=bias,
#         padding=padding,
#         stride=stride,
#         groups=groups,
#     )
#     fourier_time = benchmark(
#         fft_conv,
#         signal,
#         kernel,
#         bias=bias,
#         padding=padding,
#         stride=stride,
#         groups=groups,
#     )
#     print(f"Direct time: {direct_time}")
#     print(f"Fourier time: {fourier_time}")

#     y0 = torch_conv(
#         signal, kernel, bias=bias, padding=padding, stride=stride, groups=groups
#     )
#     y1 = fft_conv(
#         signal, kernel, bias=bias, padding=padding, stride=stride, groups=groups
#     )
#     abs_error = torch.abs(y0 - y1)
#     print(f"Output size: {y0.size()}")
#     print(f"Abs Error Mean: {abs_error.mean():.3E}")
#     print(f"Abs Error Std Dev: {abs_error.std():.3E}")


# print("\n--- 1D Convolution ---")
# benchmark_conv(
#     signal=torch.randn(4, 4, 4091),
#     kernel=torch.randn(6, 2, 1025),
#     bias=torch.randn(6),
#     padding=512,
#     stride=3,
#     groups=2,
# )

# print("\n--- 2D Convolution ---")
# benchmark_conv(
#     signal=torch.randn(4, 4, 256, 235),
#     kernel=torch.randn(6, 2, 22, 21),
#     bias=torch.randn(6),
#     padding=(9, 10),
#     stride=(2, 3),
#     groups=2,
# )

# print("\n--- 3D Convolution ---")
# benchmark_conv(
#     signal=torch.randn(4, 4, 96, 72, 61),
#     kernel=torch.randn(6, 2, 12, 7, 9),
#     bias=torch.randn(6),
#     padding=(2, 3, 4),
#     stride=(1, 2, 3),
#     groups=2,
# )
