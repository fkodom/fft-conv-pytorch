import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple

from .functional import *


class FFTConv1d(nn.Conv1d):
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return fft_conv1d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                              weight, bias, self.stride,
                              _single(0), self.dilation, self.groups)
        return fft_conv1d(input, weight, bias, self.stride,
                          self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)

    