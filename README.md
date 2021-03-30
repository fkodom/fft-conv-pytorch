# FFT Conv PyTorch

This is a fork of original [fft-conv-pytorch](https://github.com/fkodom/fft-conv-pytorch).
I made some modifications to support dilated and strided convolution, so it can be a drop-in-replacement of original PyTorch `Conv*d` modules and `conv*d` functions, with the same function parameters and behavior.

### Install

```commandline
pip install git+https://github.com/yoyololicon/fft-conv-pytorch
```

### Example Usage

```python
import torch
from torch_fftconv import fft_conv1d, FFTConv1d

# Create dummy data.  
#     Data shape: (batch, channels, length)
#     Kernel shape: (out_channels, in_channels, kernel_size)
#     Bias shape: (out channels, )
# For ordinary 1D convolution, simply set batch=1.
signal = torch.randn(3, 3, 1024 * 1024)
kernel = torch.randn(2, 3, 128)
bias = torch.randn(2)

# Functional execution.  (Easiest for generic use cases.)
out = fft_conv1d(signal, kernel, bias=bias)

# Object-oriented execution.  (Requires some extra work, since the 
# defined classes were designed for use in neural networks.)
fft_conv = FFTConv1d(3, 2, 128, bias=True)
fft_conv.weight = torch.nn.Parameter(kernel)
fft_conv.bias = torch.nn.Parameter(bias)
out = fft_conv1d(signal)
```