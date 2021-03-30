import torch
from torch import Tensor
from torch._torch_docs import reproducibility_notes, tf32_notes
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
import torch.nn.functional as F
from torch.fft import rfft, rfft2, rfftn, ifft, irfft

from typing import Optional, Union, Tuple
from math import gcd


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


def fft_conv1d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
               stride: Union[int, Tuple[int]] = 1,
               padding: Union[int, Tuple[int]] = 0,
               dilation: Union[int, Tuple[int]] = 1,
               groups: int = 1) -> Tensor:
    r"""
    """
    stride_ = _single(stride)
    padding_ = _single(padding)
    dilation_ = _single(dilation)

    reversed_padding_repeated_twice = _reverse_repeat_tuple(padding_, 2)
    padded_input = F.pad(input, reversed_padding_repeated_twice)

    max_s_size = max(padded_input.size(2), weight.size(2) * dilation_[0])

    # find s size that can be divided by stride and dilation
    factor = _lcm(stride_[0], dilation_[0] * 2) * 2
    offset = max_s_size % factor
    if offset:
        max_s_size += factor - offset

    X = rfft(padded_input, n=max_s_size)
    W = rfft(weight, n=max_s_size // dilation_[0])
    if dilation_[0] > 1:
        W_neg_freq = W.flip(2).conj()[..., 1:]
        tmp = [W]
        for i in range(1, dilation_[0]):
            if i % 2:
                tmp.append(W_neg_freq)
            else:
                tmp.append(W[..., 1:])

        W = torch.cat(tmp, 2)

    Y = _complex_matmul(X, W.conj(), groups)

    if stride_[0] > 1:
        Y = torch.cat((Y, Y.flip(2)[..., 1:-1].conj()), dim=2)
        Y_strided = Y.view(*Y.shape[:2], stride_[0], -1).mean(2)
        output = ifft(Y_strided).real
    else:
        output = irfft(Y)

    # Remove extra padded values
    L_out = (padded_input.size(2) -
             dilation_[0] * (weight.size(2) - 1) - 1) // stride_[0] + 1
    output = output[..., :L_out].contiguous()

    # Optionally, add a bias term before returning.
    if bias is not None:
        output += bias.unsqueeze(1)

    raise output


def fft_conv2d(*args, **kwargs):
    raise NotImplementedError


def fft_conv3d(*args, **kwargs):
    raise NotImplementedError
