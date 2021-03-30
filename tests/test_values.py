import torch
from torch.nn import Conv1d, Conv2d, Conv3d
from torch_fftconv.modules import FFTConv1d, FFTConv2d, FFTConv3d

import pytest

device = 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.mark.parametrize('batch', [1, 4])
@pytest.mark.parametrize('in_channels', [16, 64])
@pytest.mark.parametrize('out_channels', [8, 32])
@pytest.mark.parametrize('length', [2557])
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

    x = torch.randn(batch, in_channels, length,
                    requires_grad=True, device=device)
    conv = Conv1d(in_channels, out_channels, kernel_size, stride,
                  padding, dilation, groups, bias, padding_mode).to(device)
    fft_conv = FFTConv1d(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias, padding_mode).to(device)
    fft_conv.load_state_dict(conv.state_dict())

    y1 = conv(x)
    y2 = fft_conv(x)
    assert torch.allclose(
        y1, y2, atol=1e-5, rtol=1e-5), torch.abs(y1 - y2).max().item()
    y2.sum().backward()


@pytest.mark.parametrize('batch', [2])
@pytest.mark.parametrize('in_channels', [8, 32])
@pytest.mark.parametrize('out_channels', [4, 16])
@pytest.mark.parametrize('length', [(101, 101)])
@pytest.mark.parametrize('kernel_size', [17, 23])
@pytest.mark.parametrize('stride', [1, 2, 3])
@pytest.mark.parametrize('dilation', [1, 2, 3])
@pytest.mark.parametrize('padding', [0, 7])
@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('groups', [1, 2])
@pytest.mark.parametrize('padding_mode', ['zeros', 'reflect'])
def test_conv2d(batch, length,
                in_channels, out_channels,
                kernel_size, stride, padding, dilation, groups, bias, padding_mode):

    x = torch.randn(batch, in_channels, *length,
                    requires_grad=True, device=device)
    conv = Conv2d(in_channels, out_channels, kernel_size, stride,
                  padding, dilation, groups, bias, padding_mode).to(device)
    fft_conv = FFTConv2d(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias, padding_mode).to(device)
    fft_conv.load_state_dict(conv.state_dict())

    y1 = conv(x)
    y2 = fft_conv(x)
    assert torch.allclose(
        y1, y2, atol=1e-5, rtol=1e-5), torch.abs(y1 - y2).max().item()
    y2.sum().backward()


@pytest.mark.parametrize('batch', [2])
@pytest.mark.parametrize('in_channels', [4, 16])
@pytest.mark.parametrize('out_channels', [2, 8])
@pytest.mark.parametrize('length', [(53, 53, 59)])
@pytest.mark.parametrize('kernel_size', [9, 16])
@pytest.mark.parametrize('stride', [1, 3])
@pytest.mark.parametrize('dilation', [1, 3])
@pytest.mark.parametrize('padding', [4])
@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('groups', [1, 2])
@pytest.mark.parametrize('padding_mode', ['zeros', 'replicate'])
def test_conv3d(batch, length,
                in_channels, out_channels,
                kernel_size, stride, padding, dilation, groups, bias, padding_mode):

    x = torch.randn(batch, in_channels, *length,
                    requires_grad=True, device=device)
    conv = Conv3d(in_channels, out_channels, kernel_size, stride,
                  padding, dilation, groups, bias, padding_mode).to(device)
    fft_conv = FFTConv3d(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias, padding_mode).to(device)
    fft_conv.load_state_dict(conv.state_dict())

    y1 = conv(x)
    y2 = fft_conv(x)
    assert torch.allclose(
        y1, y2, atol=1e-5, rtol=1e-5), torch.abs(y1 - y2).max().item()
    y2.sum().backward()
