from functools import partial

import torch
from torch import nn, Tensor
from torch.fft import rfftn, irfftn
import torch.nn.functional as f


def complex_matmul(a: Tensor, b: Tensor) -> Tensor:
    """Multiplies two complex-valued tensors, where the last dimension contains the real and imaginary parts."""
    # Scalar matrix multiplication of two tensors, over only the first two dimensions.
    # Dimensions 3 and higher will have the same shape after multiplication.
    scalar_matmul = partial(torch.einsum, "ab..., cb... -> ac...")
    real = scalar_matmul(a.real, b.real) - scalar_matmul(a.imag, b.imag)
    imag = scalar_matmul(a.imag, b.real) + scalar_matmul(a.real, b.imag)

    c = torch.zeros(real.shape, dtype=torch.complex64)
    c.real, c.imag = real, imag
    return c


def fft_conv(
    signal: Tensor,
    kernel: Tensor,
    bias: Tensor = None,
    padding: int = 0,
    stride: int = 1,
) -> Tensor:
    """Performs N-d convolution of Tensors using a fast fourier transform, which is very fast for large kernel sizes.
    Also, optionally adds a bias Tensor after the convolution (in order ot mimic the PyTorch direct convolution).

    :param signal: Input tensor to be convolved with the kernel.  Shape: (batch, nchan, nsamples)
    :param kernel: Convolution kernel.  Shape: (channels_out, channels_in, kernel_size)
    :param bias: (Optional) bias tensor to add to the output.  Shape:  (channels_out, )
    :param padding: Number of zero samples to pad the input on the last dimension.
    :param stride: Convolution stride length
    :return: Convolved tensor
    """
    ndims = signal.ndim
    conv_dims = ndims - 2

    # Pad the input signal & kernel tensors
    signal_padding = conv_dims * [padding, padding]
    signal = f.pad(signal, signal_padding)
    kernel_padding = (
        torch.tensor(
            [[0, signal.size(i) - kernel.size(i)] for i in range(ndims - 1, 1, -1)]
        )
        .flatten()
        .tolist()
    )
    padded = f.pad(kernel, kernel_padding)

    # Perform fourier convolution -- FFT, matrix multiply, then IFFT
    signal_fr = rfftn(signal, dim=tuple(range(2, ndims)))
    kernel_fr = rfftn(padded, dim=tuple(range(2, ndims)))

    kernel_fr.imag *= -1
    output_fr = complex_matmul(signal_fr, kernel_fr)
    output = irfftn(output_fr, dim=tuple(range(2, ndims)))

    # Keep outputs at strided intervals, then remove extra padded values
    stride_slices = [slice(0, output.shape[0]), slice(0, output.shape[1])] + [
        slice(0, output.shape[i], stride) for i in range(2, ndims)
    ]
    crop_slices = [slice(0, output.shape[0]), slice(0, output.shape[1])] + [
        slice(0, (signal.size(i) - kernel.size(i)) // stride + 1)
        for i in range(2, ndims)
    ]
    output = output[stride_slices]
    output = output[crop_slices].contiguous()

    # Optionally, add a bias term before returning.
    if bias is not None:
        bias_shape = tuple([1, -1] + conv_dims * [1])
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
        stride: int = 1,
        bias: bool = True,
    ):
        """
        :param in_channels: Number of channels in input tensors
        :param out_channels: Number of channels in output tensors
        :param kernel_size: Size of the 2D convolution kernel.  (i.e. kernel_size=3 gives a 3x3 kernel)
        :param padding: Amount of zero-padding to add to the input tensor
        :param stride: Convolution stride length. Defaults to 1, as in standard convolution
        :param bias: If True, includes an additional bias term, which is added to the output after convolution
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.use_bias = bias

        self.weight = None
        self.bias = None

    def forward(self, signal):
        return fft_conv(
            signal,
            self.weight,
            bias=self.bias,
            padding=self.padding,
            stride=self.stride,
        )


class FFTConv1d(_FFTConv):
    """PyTorch 1D convoluton layer based on FFT."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        bias: bool = True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            stride=stride,
            bias=bias,
        )
        self.bias = nn.Parameter(torch.randn(out_channels,)) if bias else None
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))


class FFTConv2d(_FFTConv):
    """PyTorch 2D convoluton layer based on FFT."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        bias: bool = True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            stride=stride,
            bias=bias,
        )
        self.bias = nn.Parameter(torch.randn(out_channels,)) if bias else None
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )


class FFTConv3d(_FFTConv):
    """PyTorch 3D convoluton layer based on FFT."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        bias: bool = True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            stride=stride,
            bias=bias,
        )
        self.bias = nn.Parameter(torch.randn(out_channels,)) if bias else None
        self.weight = nn.Parameter(
            torch.randn(
                out_channels, in_channels, kernel_size, kernel_size, kernel_size
            )
        )
