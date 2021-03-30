import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple
from typing import Optional
from .functional import *


__all__ = [
    'FFTConv1d', 'FFTConv2d', 'FFTConv3d',
]


class FFTConv1d(nn.Conv1d):
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return fft_conv1d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                              weight, bias, self.stride,
                              _single(0), self.dilation, self.groups)
        return fft_conv1d(input, weight, bias, self.stride,
                          self.padding, self.dilation, self.groups)


class FFTConv2d(nn.Conv2d):
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return fft_conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                              weight, bias, self.stride,
                              _pair(0), self.dilation, self.groups)
        return fft_conv2d(input, weight, bias, self.stride,
                          self.padding, self.dilation, self.groups)


class FFTConv3d(nn.Conv3d):
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return fft_conv3d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                              weight, bias, self.stride,
                              _triple(0), self.dilation, self.groups)
        return fft_conv3d(input, weight, bias, self.stride,
                          self.padding, self.dilation, self.groups)
