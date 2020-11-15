# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython


# %%
import torch
import torch.nn.functional as f

from fft_conv import fft_conv, FFTConv1d


# %%
# Test 1D Convolution
kernel_size = 255
padding = kernel_size // 2
kernel = torch.randn(2, 3, kernel_size)
signal = torch.randn(3, 3, 2 ** 20)
bias = torch.randn(2)

# Perform both direct and FFT convolutions
print('--- Direct Convolution ---')
get_ipython().run_line_magic('time', 'y0 = f.conv1d(signal, kernel, bias=bias, padding=padding)')
print('--- FFT Convolution ---')
get_ipython().run_line_magic('time', 'y1 = fft_conv(signal, kernel, bias=bias, padding=padding)')

# Print input/output tensor shapes
print(f'\nInput shape: {signal.shape}')
print(f'Output shape: {y0.shape}')

# Compute relative error
abs_error = torch.abs(y0 - y1)
print(f'\nAbs Error Mean: {abs_error.mean():.3E}')
print(f'Abs Error Std Dev: {abs_error.std():.3E}')


# %%
# Test 2D Convolution
kernel_size = 11
padding = kernel_size // 2
kernel = torch.randn(2, 3, kernel_size, kernel_size)
signal = torch.randn(3, 3, 1024, 1024)
bias = torch.randn(2)

# Perform both direct and FFT convolutions
print('--- Direct Convolution ---')
get_ipython().run_line_magic('time', 'y0 = f.conv2d(signal, kernel, bias=bias, padding=padding)')
print('--- FFT Convolution ---')
get_ipython().run_line_magic('time', 'y1 = fft_conv(signal, kernel, bias=bias, padding=padding)')

# Print input/output tensor shapes
print(f'\nInput shape: {signal.shape}')
print(f'Output shape: {y0.shape}')

# Compute relative error
abs_error = torch.abs(y0 - y1)
print(f'\nAbs Error Mean: {abs_error.mean():.3E}')
print(f'Abs Error Std Dev: {abs_error.std():.3E}')


# %%
# Test 3D Convolution
kernel_size = 7
padding = kernel_size // 2
kernel = torch.randn(2, 3, kernel_size, kernel_size, kernel_size)
signal = torch.randn(3, 3, 100, 300, 300)
bias = torch.randn(2)

# Perform both direct and FFT convolutions
print('--- Direct Convolution ---')
get_ipython().run_line_magic('time', 'y0 = f.conv3d(signal, kernel, bias=bias, padding=padding)')
print('--- FFT Convolution ---')
get_ipython().run_line_magic('time', 'y1 = fft_conv(signal, kernel, bias=bias, padding=padding)')

# Print input/output tensor shapes
print(f'\nInput shape: {signal.shape}')
print(f'Output shape: {y0.shape}')

abs_error = torch.abs(y0 - y1)
print(f'\nAbs Error Mean: {abs_error.mean():.3E}')
print(f'Abs Error Std Dev: {abs_error.std():.3E}')


# %%
net = torch.nn.Sequential(
    FFTConv1d(1, 3, 101, padding=50),
    FFTConv1d(3, 3, 101, padding=50),
    FFTConv1d(3, 1, 101, padding=50),
)
output = net(torch.randn(1, 1, 1024))
loss = output.sum()
loss.backward()
