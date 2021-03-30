import torch
from torch.nn import Conv1d
from torch_fftconv.modules import FFTConv1d

import pytest


@pytest.mark.parametrize('batch', [1, 4])
@pytest.mark.parametrize('in_channels', [16, 64])
@pytest.mark.parametrize('out_channels', [8, 32])
@pytest.mark.parametrize('length', [3571])
@pytest.mark.parametrize('kernel_size', [128, 256])
@pytest.mark.parametrize('stride', [1, 2, 3])
@pytest.mark.parametrize('dilation', [1, 2, 4])
@pytest.mark.parametrize('padding', [0, 3])
@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('groups', [1, 4])
@pytest.mark.parametrize('padding_mode', ['zeros', 'reflect'])
def test_conv1d(batch, length,
                in_channels, out_channels,
                kernel_size, stride, padding, dilation, groups, bias, padding_mode):

    x = torch.randn(batch, in_channels, length)
    conv = Conv1d(in_channels, out_channels, kernel_size, stride,
                  padding, dilation, groups, bias, padding_mode)
    fft_conv = FFTConv1d(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias, padding_mode)
    fft_conv.load_state_dict(conv.state_dict())

    y1 = conv(x)
    y2 = fft_conv(x)
    assert torch.allclose(y1, y2, atol=1e-5, rtol=1e-4), torch.abs(y1 - y2).max().item()
