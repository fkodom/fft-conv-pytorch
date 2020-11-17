from functools import partial

import torch
from torch import nn, Tensor
from torch.fft import rfftn, irfftn
import torch.nn.functional as f


def complex_matmul(a: Tensor, b: Tensor) -> Tensor:
    """Multiplies two complex-valued tensors."""
    # Scalar matrix multiplication of two tensors, over only the first two dimensions.
    # Dimensions 3 and higher will have the same shape after multiplication.
    scalar_matmul = partial(torch.einsum, "ab..., cb... -> ac...")

    # Compute the real and imaginary parts independently, then manually insert them
    # into the output Tensor.  This is fairly hacky but necessary for PyTorch 1.7.0,
    # because Autograd is not enabled for complex matrix operations yet.  Not exactly
    # idiomatic PyTorch code, but it should work for all future versions (>= 1.7.0).
    real = scalar_matmul(a.real, b.real) - scalar_matmul(a.imag, b.imag)
    imag = scalar_matmul(a.imag, b.real) + scalar_matmul(a.real, b.imag)
    c = torch.zeros(real.shape, dtype=torch.complex64)
    c.real, c.imag = real, imag

    return c


def fft_conv_1d(
    signal: Tensor, kernel: Tensor, bias: Tensor = None, padding: int = 0,
) -> Tensor:
    """
    Args:
        signal: (Tensor) Input tensor to be convolved with the kernel.
        kernel: (Tensor) Convolution kernel.
        bias: (Optional, Tensor) Bias tensor to add to the output.
        padding: (int) Number of zero samples to pad the input on the last dimension.

    Returns:
        (Tensor) Convolved tensor
    """
    # 1. Pad the input signal & kernel tensors
    signal = f.pad(signal, [padding, padding])
    kernel_padding = [0, signal.size(-1) - kernel.size(-1)]
    padded_kernel = f.pad(kernel, kernel_padding)

    # 2. Perform fourier convolution
    signal_fr = rfftn(signal, dim=-1)
    kernel_fr = rfftn(padded_kernel, dim=-1)

    # 3. Multiply the transformed matrices
    kernel_fr.imag *= -1
    output_fr = complex_matmul(signal_fr, kernel_fr)

    # 4. Compute inverse FFT, and remove extra padded values
    output = irfftn(output_fr, dim=-1)
    output = output[:, :, :signal.size(-1) - kernel.size(-1)]

    # Optionally, add a bias term before returning.
    if bias is not None:
        output += bias.view(1, -1, 1)

    return output


def fft_conv(
    signal: Tensor, kernel: Tensor, bias: Tensor = None, padding: int = 0,
) -> Tensor:
    """Performs N-d convolution of Tensors using a fast fourier transform, which
    is very fast for large kernel sizes. Also, optionally adds a bias Tensor after
    the convolution (in order ot mimic the PyTorch direct convolution).

    Args:
        signal: (Tensor) Input tensor to be convolved with the kernel.
        kernel: (Tensor) Convolution kernel.
        bias: (Optional, Tensor) Bias tensor to add to the output.
        padding: (int) Number of zero samples to pad the input on the last dimension.

    Returns:
        (Tensor) Convolved tensor
    """
    # Pad the input signal & kernel tensors
    signal_padding = (signal.ndim - 2) * [padding, padding]
    signal = f.pad(signal, signal_padding)
    kernel_padding = [
        pad for i in reversed(range(2, signal.ndim))
        for pad in [0, signal.size(i) - kernel.size(i)]
    ]
    padded_kernel = f.pad(kernel, kernel_padding)

    # Perform fourier convolution -- FFT, matrix multiply, then IFFT
    signal_fr = rfftn(signal, dim=tuple(range(2, signal.ndim)))
    kernel_fr = rfftn(padded_kernel, dim=tuple(range(2, signal.ndim)))

    kernel_fr.imag *= -1
    output_fr = complex_matmul(signal_fr, kernel_fr)
    output = irfftn(output_fr, dim=tuple(range(2, signal.ndim)))

    # Remove extra padded values
    crop_slices = [slice(0, output.shape[0]), slice(0, output.shape[1])] + [
        slice(0, (signal.size(i) - kernel.size(i))) for i in range(2, signal.ndim)
    ]
    output = output[crop_slices].contiguous()

    # Optionally, add a bias term before returning.
    if bias is not None:
        bias_shape = tuple([1, -1] + (signal.ndim - 2) * [1])
        output += bias.view(bias_shape)

    return output


class _FFTConv(nn.Module):
    """Base class for PyTorch FFT convolution layers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        bias: bool = True,
    ):
        """
        Args:
            in_channels: (int) Number of channels in input tensors
            out_channels: (int) Number of channels in output tensors
            kernel_size: (int) Square radius of the convolution kernel
            padding: (int) Amount of zero-padding to add to the input tensor
            bias: (bool) If True, includes bias, which is added after convolution
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.use_bias = bias

        self.weight = torch.empty(0)
        self.bias = nn.Parameter(torch.randn(out_channels,)) if bias else None

    def forward(self, signal):
        return fft_conv(
            signal,
            self.weight,
            bias=self.bias,
            padding=self.padding,
        )


class FFTConv1d(_FFTConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            bias=bias,
        )
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))


class FFTConv2d(_FFTConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            bias=bias,
        )
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )


class FFTConv3d(_FFTConv):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            bias=bias,
        )
        self.weight = nn.Parameter(
            torch.randn(
                out_channels, in_channels, kernel_size, kernel_size, kernel_size
            )
        )
