import torch
from torch import Tensor
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
import torch.nn.functional as F
from torch.fft import rfft, fftn, rfftn, ifft, irfft, ifftn

from typing import List, Optional, Union, Tuple
from math import gcd

__all__ = [
    'fft_conv1d', 'fft_conv2d', 'fft_conv3d',
]


def _lcm(x: int, y: int):
    return abs(x * y) // gcd(x, y)


def _complex_matmul(a: Tensor, b: Tensor, groups: int = 1) -> Tensor:
    """Multiplies two complex-valued tensors."""
    # Scalar matrix multiplication of two tensors, over only the first channel
    # dimensions. Dimensions 3 and higher will have the same shape after multiplication.
    # We also allow for "grouped" multiplications, where multiple sections of channels
    # are multiplied independently of one another (required for group convolutions).
    a = a.view(a.size(0), groups, -1, *a.shape[2:])
    b = b.view(groups, -1, *b.shape[1:])

    # Compute the real and imaginary parts independently, then manually insert them
    # into the output Tensor.  This is fairly hacky but necessary for PyTorch 1.7.0,
    # because Autograd is not enabled for complex matrix operations yet.  Not exactly
    # idiomatic PyTorch code, but it should work for all future versions (>= 1.7.0).
    real = torch.einsum("agc..., gbc... -> agb...", a.real, b.real) - \
        torch.einsum("agc..., gbc... -> agb...", a.imag, b.imag)
    imag = torch.einsum("agc..., gbc... -> agb...", a.imag, b.real) + \
        torch.einsum("agc..., gbc... -> agb...", a.real, b.imag)

    c = torch.view_as_complex(torch.stack((real, imag), -1))

    return c.view(c.size(0), -1, *c.shape[3:])


def _conv_shape(L_in: Tuple[int],
                kernel: Tuple[int],
                stride: Tuple[int],
                padding: Tuple[int],
                dilation: Tuple[int]) -> Tuple[int]:

    L_out: List[int] = []
    for l, k, s, p, d in zip(L_in, kernel, stride, padding, dilation):
        out = (l + 2 * p - d * (k - 1) - 1) // s + 1
        assert out > 0, "Kernel size can't be greater than input."
        L_out.append(out)

    return tuple(L_out)


def _fft_convnd(input: Tensor,
                weight: Tensor,
                bias: Optional[Tensor],
                stride: Tuple[int],
                padding: Tuple[int],
                dilation: Tuple[int],
                groups: int) -> Tensor:

    output_size = _conv_shape(input.shape[2:], weight.shape[2:],
                              stride, padding, dilation)
    reversed_padding_repeated_twice = _reverse_repeat_tuple(padding, 2)
    padded_input = F.pad(input, reversed_padding_repeated_twice)

    s: List[int] = []
    weight_s: List[int] = []
    for i, (x_size, w_size, d, st) in enumerate(zip(padded_input.shape[2:], weight.shape[2:], dilation, stride)):
        s_size = max(x_size, w_size * d)

        # find s size that can be divided by stride and dilation
        rfft_even = 2 if i == len(stride) - 1 else 1
        factor = _lcm(st, rfft_even)
        if d > 1:
            factor = _lcm(factor, d * rfft_even)

        offset = s_size % factor
        if offset:
            s_size += factor - offset
        s.append(s_size)
        weight_s.append(s_size // d)

    X = rfftn(padded_input, s=s)

    W = rfft(weight, n=weight_s[-1])
    # handle dilation
    # handle dilation for last dim
    if dilation[-1] > 1:
        W_neg_freq = W.flip(-1).conj()[..., 1:]
        tmp = [W]
        for i in range(1, dilation[-1]):
            if i % 2:
                tmp.append(W_neg_freq)
            else:
                tmp.append(W[..., 1:])

        W = torch.cat(tmp, -1)

    if len(weight_s) > 1:
        W = fftn(W, s=weight_s[:-1], dim=tuple(range(2, W.ndim - 1)))
        repeats = (1, 1) + dilation[:-1] + (1,)
        if sum(repeats) > W.ndim:
            W = W.repeat(*repeats)

    Y = _complex_matmul(X, W.conj(), groups)

    # handle stride
    if len(stride) > 1:
        unfold_shape = [Y.size(0), Y.size(1)]
        sum_dims: List[int] = []
        for i, st in enumerate(stride[:-1]):
            if st == 1:
                unfold_shape.append(Y.size(i + 2))
                continue
            step = Y.size(i + 2) // st
            unfold_shape += [st, step]
            sum_dims.append(len(unfold_shape) - 2)

        unfold_shape.append(-1)
        if len(sum_dims):
            Y = Y.view(*unfold_shape).mean(sum_dims)
        Y = ifftn(Y, dim=tuple(range(2, Y.ndim - 1)))

    if stride[-1] > 1:
        Y = torch.cat((Y, Y.flip(-1)[..., 1:-1].conj()), dim=-1)
        Y_strided = Y.view(*Y.shape[:-1], stride[-1], -1).mean(-2)
        output = ifft(Y_strided).real
    else:
        output = irfft(Y)

    # Remove extra padded values
    index = (slice(None),) * 2 + tuple(slice(l) for l in output_size)
    output = output[index].contiguous()

    # Optionally, add a bias term before returning.
    if bias is not None:
        output += bias[(slice(None),) + (None,) * (output.ndim - 2)]

    return output


def fft_conv1d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
               stride: Union[int, Tuple[int]] = 1,
               padding: Union[int, Tuple[int]] = 0,
               dilation: Union[int, Tuple[int]] = 1,
               groups: int = 1) -> Tensor:
    r"""
    """
    stride = _single(stride)
    padding = _single(padding)
    dilation = _single(dilation)

    return _fft_convnd(input, weight, bias, stride, padding, dilation, groups)


def fft_conv2d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
               stride: Union[int, Tuple[int]] = 1,
               padding: Union[int, Tuple[int]] = 0,
               dilation: Union[int, Tuple[int]] = 1,
               groups: int = 1) -> Tensor:
    r"""
    """
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    return _fft_convnd(input, weight, bias, stride, padding, dilation, groups)


def fft_conv3d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
               stride: Union[int, Tuple[int]] = 1,
               padding: Union[int, Tuple[int]] = 0,
               dilation: Union[int, Tuple[int]] = 1,
               groups: int = 1) -> Tensor:
    r"""
    """
    stride = _triple(stride)
    padding = _triple(padding)
    dilation = _triple(dilation)

    return _fft_convnd(input, weight, bias, stride, padding, dilation, groups)
