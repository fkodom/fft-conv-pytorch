from functools import partial

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as f


def complex_matmul(a: Tensor, b: Tensor) -> Tensor:
    """Multiplies two complex-valued tensors, where the last dimension contains the real and imaginary parts."""
    # Scalar matrix multiplication of two tensors, over only the first two dimensions.
    # Dimensions 3 and higher will have the same shape after multiplication.
    num_dims = len(a.shape)
    if num_dims == 4:
        scalar_matmul = partial(torch.einsum, 'abd, cbd -> acd')
    elif num_dims == 5:
        scalar_matmul = partial(torch.einsum, 'abde, cbde -> acde')
    elif num_dims == 6:
        scalar_matmul = partial(torch.einsum, 'abdef, cbdef -> acdef')
    else:
        raise ValueError(f'Not supported for {num_dims} dimensions.  Supported dimensions: 3, 4, 5')

    return torch.stack([scalar_matmul(a[..., 0], b[..., 0]) + scalar_matmul(a[..., 1], b[..., 1]),
                        scalar_matmul(a[..., 1], b[..., 0]) - scalar_matmul(a[..., 0], b[..., 1])], dim=-1)


def fft_conv_nd(signal: Tensor, kernel: Tensor, bias: Tensor = None, padding: int = 0, stride: int = 1) -> Tensor:
    """Performs N-d convolution of Tensors using a fast fourier transform, which is very fast for large kernel sizes.
    Also, optionally adds a bias Tensor after the convolution (in order ot mimic the PyTorch direct convolution).

    :param signal: Input tensor to be convolved with the kernel.  Shape: (batch, nchan, nsamples)
    :param kernel: Convolution kernel.  Shape: (channels_out, channels_in, kernel_size)
    :param bias: (Optional) bias tensor to add to the output.  Shape:  (channels_out, )
    :param padding: Number of zero samples to pad the input on the last dimension.
    :param stride: Convolution stride length
    :return: Convolved tensor
    """
    ndims = len(signal.shape)
    conv_dims = ndims - 2

    # Pad the input signal & kernel tensors
    signal_padding = conv_dims * [padding, padding]
    signal = f.pad(signal, signal_padding)
    kernel_padding = torch.tensor(
        [[0, signal.size(i) - kernel.size(i)] for i in range(ndims - 1, 1, -1)]).flatten().tolist()
    padded = f.pad(kernel, kernel_padding)

    # Perform fourier convolution -- FFT, matrix multiply, then IFFT
    signal_fr = torch.rfft(signal, conv_dims)
    padded_fr = torch.rfft(padded, conv_dims)
    output_fr = complex_matmul(signal_fr, padded_fr)
    signal_sizes = [signal.size(i) for i in range(2, ndims)]
    output = torch.irfft(output_fr, conv_dims, signal_sizes=signal_sizes)

    # Keep outputs at strided intervals, then remove extra padded values
    stride_slices = [slice(0, output.shape[0]), slice(0, output.shape[1])] + \
                    [slice(0, output.shape[i], stride) for i in range(2, ndims)]
    crop_slices = [slice(0, output.shape[0]), slice(0, output.shape[1])] + \
                  [slice(0, (signal.size(i) - kernel.size(i)) // stride + 1) for i in range(2, ndims)]
    output = output[stride_slices]
    output = output[crop_slices].contiguous()

    # Optionally, add a bias term before returning.
    if bias is not None:
        bias_shape = tuple([1, -1] + conv_dims * [1])
        output += bias.view(bias_shape)

    return output


class _FFTConv(nn.Module):
    """Base class for PyTorch FFT convolution layers."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0, stride: int = 1,
                 bias: bool = True):
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
        return fft_conv_nd(signal, self.weight, bias=self.bias, padding=self.padding, stride=self.stride)


class FFTConv1d(_FFTConv):
    """PyTorch 1D convoluton layer based on FFT."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0, stride: int = 1,
                 bias: bool = True):
        super().__init__(in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=bias)
        self.bias = nn.Parameter(torch.randn(out_channels, )) if bias else None
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))


class FFTConv2d(_FFTConv):
    """PyTorch 2D convoluton layer based on FFT."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0, stride: int = 1,
                 bias: bool = True):
        super().__init__(in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=bias)
        self.bias = nn.Parameter(torch.randn(out_channels, )) if bias else None
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))


class FFTConv3d(_FFTConv):
    """PyTorch 3D convoluton layer based on FFT."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0, stride: int = 1,
                 bias: bool = True):
        super().__init__(in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=bias)
        self.bias = nn.Parameter(torch.randn(out_channels, )) if bias else None
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
