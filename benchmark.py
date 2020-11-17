from typing import Callable, NamedTuple
from timeit import Timer

import torch
import torch.nn.functional as f
import numpy as np

from fft_conv import fft_conv, FFTConv1d


class Benchmark(NamedTuple):
    mean: float
    std: float

    def __repr__(self):
        return f"BenchmarkResult(mean: {self.mean:.3e}, std: {self.std:.3e})"

    def __str__(self):
        return f"({self.mean:.3e} \u00B1 {self.std:.3e}) s"


def benchmark(
    fn: Callable, *args, num_iterations: int = 10, **kwargs
) -> Benchmark:
    timer = Timer(
        "fn(*args, **kwargs)",
        globals={"fn": fn, "args": args, "kwargs": kwargs},
    )
    times = timer.repeat(number=1, repeat=num_iterations+1)
    return Benchmark(np.mean(times[1:]).item(), np.std(times[1:]).item())


print('\n--- 1D Convolution ---')
kernel_size = 1025
padding = kernel_size // 2
kernel = torch.randn(2, 3, kernel_size)
signal = torch.randn(3, 3, 4096)
bias = torch.randn(2)
print(f"Signal size: {signal.shape}")
print(f"Kernel size: {kernel.shape}")

direct_time = benchmark(f.conv1d, signal, kernel, bias=bias, padding=padding)
fourier_time = benchmark(fft_conv, signal, kernel, bias=bias, padding=padding)
print(f"Direct time: {direct_time}")
print(f"Fourier time: {fourier_time}")

y0 = f.conv1d(signal, kernel, bias=bias, padding=padding)
y1 = fft_conv(signal, kernel, bias=bias, padding=padding)
abs_error = torch.abs(y0 - y1)
print(f'Abs Error Mean: {abs_error.mean():.3E}')
print(f'Abs Error Std Dev: {abs_error.std():.3E}')


print('\n--- 2D Convolution ---')
kernel_size = 21
padding = kernel_size // 2
kernel = torch.randn(2, 3, kernel_size, kernel_size)
signal = torch.randn(3, 3, 256, 256)
bias = torch.randn(2)
print(f"Signal size: {signal.shape}")
print(f"Kernel size: {kernel.shape}")

direct_time = benchmark(f.conv1d, signal, kernel, bias=bias, padding=padding)
fourier_time = benchmark(fft_conv, signal, kernel, bias=bias, padding=padding)
print(f"Direct time: {direct_time}")
print(f"Fourier time: {fourier_time}")

y0 = f.conv1d(signal, kernel, bias=bias, padding=padding)
y1 = fft_conv(signal, kernel, bias=bias, padding=padding)
abs_error = torch.abs(y0 - y1)
print(f'Abs Error Mean: {abs_error.mean():.3E}')
print(f'Abs Error Std Dev: {abs_error.std():.3E}')


print('\n--- 3D Convolution ---')
kernel_size = 9
padding = kernel_size // 2
kernel = torch.randn(2, 3, kernel_size, kernel_size, kernel_size)
signal = torch.randn(3, 3, 64, 64, 64)
bias = torch.randn(2)
print(f"Signal size: {signal.shape}")
print(f"Kernel size: {kernel.shape}")

direct_time = benchmark(f.conv1d, signal, kernel, bias=bias, padding=padding)
fourier_time = benchmark(fft_conv, signal, kernel, bias=bias, padding=padding)
print(f"Direct time: {direct_time}")
print(f"Fourier time: {fourier_time}")

y0 = f.conv1d(signal, kernel, bias=bias, padding=padding)
y1 = fft_conv(signal, kernel, bias=bias, padding=padding)
abs_error = torch.abs(y0 - y1)
print(f'Abs Error Mean: {abs_error.mean():.3E}')
print(f'Abs Error Std Dev: {abs_error.std():.3E}')


# Test that autograd works!
net = torch.nn.Sequential(
    FFTConv1d(1, 3, 101, padding=50),
    FFTConv1d(3, 3, 101, padding=50),
    FFTConv1d(3, 1, 101, padding=50),
)
output = net(torch.randn(1, 1, 1024))
loss = output.sum()
loss.backward()
