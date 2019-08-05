# FFT Conv PyTorch

Implementation of 1D, 2D, and 3D FFT convolutions in PyTorch.  
* Faster than direct convolution for large kernels.
* **Much slower** than direct convolution for small kernels.
* Typically, FFT convolution is faster when the kernel has >100 elements.
    * Dependent on machine and PyTorch version.


### Example Usage

```python
import torch
from fft_conv import fft_conv_nd, FFTConv1d

# Create dummy data.  
#     Data shape: (batch, channels, length)
#     Kernel shape: (out_channels, in_channels, kernel_size)
#     Bias shape: (out channels, )
# For ordinary 1D convolution, simply set batch=1.
signal = torch.randn(3, 3, 1024 * 1024)
kernel = torch.randn(2, 3, 128)
bias = torch.randn(2)

# Functional execution.  (Easiest for generic use cases.)
out = fft_conv_nd(signal, kernel, bias=bias)

# Object-oriented execution.  (Requires some extra work, since the 
# defined classes were designed for use in neural networks.)
fft_conv = FFTConv1d(3, 2, 128, bias=True)
fft_conv.weight = torch.nn.Parameter(kernel)
fft_conv.bias = torch.nn.Parameter(bias)
out = fft_conv(signal)
```