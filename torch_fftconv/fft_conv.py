from functools import partial
from typing import Tuple, Union, Iterable

import torch
from torch import nn, Tensor
from torch.fft import rfftn, ifftn, irfft, ifft
import torch.nn.functional as f
from torch.nn.modules.utils import _reverse_repeat_tuple


def complex_matmul(a: Tensor, b: Tensor, groups: int = 1) -> Tensor:
    """Multiplies two complex-valued tensors."""
    # Scalar matrix multiplication of two tensors, over only the first channel
    # dimensions. Dimensions 3 and higher will have the same shape after multiplication.
    # We also allow for "grouped" multiplications, where multiple sections of channels
    # are multiplied independently of one another (required for group convolutions).
    scalar_matmul = partial(torch.einsum, "agc..., gbc... -> agb...")
    a = a.view(a.size(0), groups, -1, *a.shape[2:])
    b = b.view(groups, -1, *b.shape[1:])

    # Compute the real and imaginary parts independently, then manually insert them
    # into the output Tensor.  This is fairly hacky but necessary for PyTorch 1.7.0,
    # because Autograd is not enabled for complex matrix operations yet.  Not exactly
    # idiomatic PyTorch code, but it should work for all future versions (>= 1.7.0).
    real = scalar_matmul(a.real, b.real) - scalar_matmul(a.imag, b.imag)
    imag = scalar_matmul(a.imag, b.real) + scalar_matmul(a.real, b.imag)

    c = torch.view_as_complex(torch.stack((real, imag), -1))

    return c.view(c.size(0), -1, *c.shape[3:])


def to_ntuple(val: Union[int, Iterable[int]], n: int) -> Tuple[int, ...]:
    """Casts to a tuple with length 'n'.  Useful for automatically computing the
    padding and stride for convolutions, where users may only provide an integer.

    Args:
        val: (Union[int, Iterable[int]]) Value to cast into a tuple.
        n: (int) Desired length of the tuple

    Returns:
        (Tuple[int, ...]) Tuple of length 'n'
    """
    if isinstance(val, Iterable):
        out = tuple(val)
        if len(out) == n:
            return out
        else:
            raise ValueError(
                f"Cannot cast tuple of length {len(out)} to length {n}.")
    else:
        return n * (val,)


def fft_conv(
    signal: Tensor,
    kernel: Tensor,
    bias: Tensor = None,
    padding: Union[int, Iterable[int]] = 0,
    stride: Union[int, Iterable[int]] = 1,
    groups: int = 1,
) -> Tensor:
    """Performs N-d convolution of Tensors using a fast fourier transform, which
    is very fast for large kernel sizes. Also, optionally adds a bias Tensor after
    the convolution (in order ot mimic the PyTorch direct convolution).

    Args:
        signal: (Tensor) Input tensor to be convolved with the kernel.
        kernel: (Tensor) Convolution kernel.
        bias: (Tensor) Bias tensor to add to the output.
        padding: (Union[int, Iterable[int]) Number of zero samples to pad the
            input on the last dimension.
        stride: (Union[int, Iterable[int]) Stride size for computing output values.

    Returns:
        (Tensor) Convolved tensor
    """
    # Cast padding & stride to tuples.
    padding_ = to_ntuple(padding, n=signal.ndim - 2)
    stride_ = to_ntuple(stride, n=signal.ndim - 2)

    # Pad the input signal & kernel tensors
    signal_padding = _reverse_repeat_tuple(padding_, 2)
    signal = f.pad(signal, signal_padding)

    # Because PyTorch computes a *one-sided* FFT, we need the final dimension to
    # have *even* length.  Just pad with one more zero if the final dimension is odd.

    extra_padding = []
    for i in range(signal.ndim - 1, 1, -1):
        factor = stride_[i - 2]
        if not len(extra_padding):
            if signal.size(i) % 2 and factor % 2:
                factor *= 2
        offset = signal.size(i) % factor
        extra_padding += [0, factor - offset if offset else 0]

    if sum(extra_padding):
        signal_ = f.pad(signal, extra_padding)
    else:
        signal_ = signal

    # Perform fourier convolution -- FFT, matrix multiply, then IFFT
    # signal_ = signal_.reshape(signal_.size(0), groups, -1, *signal_.shape[2:])
    signal_fr = rfftn(signal_, dim=tuple(range(2, signal.ndim)))
    kernel_fr = rfftn(
        kernel, s=signal_.shape[2:], dim=tuple(range(2, signal.ndim)))

    output_fr = complex_matmul(signal_fr, kernel_fr.conj(), groups=groups)

    if signal.ndim > 3:
        unfold_shape = [output_fr.size(0), output_fr.size(1)]
        sum_dims = []
        for i in range(2, signal.ndim - 1):
            if stride_[i - 2] == 1:
                unfold_shape.append(output_fr.size(i))
                continue
            step = output_fr.size(i) // stride_[i - 2]
            unfold_shape += [stride_[i - 2], step]
            sum_dims.append(len(unfold_shape) - 2)

        unfold_shape.append(-1)
        if len(sum_dims):
            output_fr = output_fr.view(*unfold_shape).mean(sum_dims)
        output_fr = ifftn(output_fr, dim=tuple(range(2, signal.ndim - 1)))

    if stride_[-1] != 1:
        output_fr = torch.cat(
            (output_fr, output_fr.flip(-1)[..., 1:-1].conj()), dim=-1)
        output = ifft(output_fr.view(
            *output_fr.shape[:-1], stride_[-1], -1).mean(-2)).real
    else:
        output = irfft(output_fr)

    # Remove extra padded values
    new_size = [output.shape[0], output.shape[1]]
    for i in range(2, signal.ndim):
        new_size.append((signal.size(i) - kernel.size(i)) //
                        stride_[i - 2] + 1)

    output = output.as_strided(new_size, output.stride()).contiguous()

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
        kernel_size: Union[int, Iterable[int]],
        padding: Union[int, Iterable[int]] = 0,
        stride: Union[int, Iterable[int]] = 1,
        groups: int = 1,
        bias: bool = True,
        ndim: int = 1,
    ):
        """
        Args:
            in_channels: (int) Number of channels in input tensors
            out_channels: (int) Number of channels in output tensors
            kernel_size: (Union[int, Iterable[int]) Square radius of the kernel
            padding: (Union[int, Iterable[int]) Number of zero samples to pad the
                input on the last dimension.
            stride: (Union[int, Iterable[int]) Stride size for computing output values.
            bias: (bool) If True, includes bias, which is added after convolution
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.groups = groups
        self.use_bias = bias

        if in_channels % groups != 0:
            raise ValueError(
                "'in_channels' must be divisible by 'groups'."
                f"Found: in_channels={in_channels}, groups={groups}."
            )
        if out_channels % groups != 0:
            raise ValueError(
                "'out_channels' must be divisible by 'groups'."
                f"Found: out_channels={out_channels}, groups={groups}."
            )

        kernel_size = to_ntuple(kernel_size, ndim)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels // groups, *kernel_size)
        )
        self.bias = nn.Parameter(torch.randn(out_channels,)) if bias else None

    def forward(self, signal):
        return fft_conv(
            signal,
            self.weight,
            bias=self.bias,
            padding=self.padding,
            stride=self.stride,
            groups=self.groups,
        )


#FFTConv1d = partial(_FFTConv, ndim=1)
#FFTConv2d = partial(_FFTConv, ndim=2)
#FFTConv3d = partial(_FFTConv, ndim=3)
