from typing import Callable, NamedTuple
from timeit import Timer

import torch
from torch import Tensor
import torch.nn.functional as f
import numpy as np

from fft_conv import fft_conv


class Benchmark(NamedTuple):
    mean: float
    std: float

    def __repr__(self):
        return f"BenchmarkResult(mean: {self.mean:.3e}, std: {self.std:.3e})"

    def __str__(self):
        return f"({self.mean:.3e} \u00B1 {self.std:.3e}) s"


def benchmark(fn: Callable, *args, num_iterations: int = 10, **kwargs) -> Benchmark:
    timer = Timer(
        "fn(*args, **kwargs)", globals={"fn": fn, "args": args, "kwargs": kwargs},
    )
    times = timer.repeat(number=1, repeat=num_iterations + 1)
    return Benchmark(np.mean(times[1:]).item(), np.std(times[1:]).item())


def benchmark_conv(
    signal: Tensor, kernel: Tensor, bias: Tensor, padding: int = 0, stride: int = 1
):
    print(f"Signal size: {signal.shape}")
    print(f"Kernel size: {kernel.shape}")

    torch_conv = {1: f.conv1d, 2: f.conv2d, 3: f.conv3d}[signal.ndim - 2]
    direct_time = benchmark(
        torch_conv, signal, kernel, bias=bias, padding=padding, stride=stride
    )
    fourier_time = benchmark(
        fft_conv, signal, kernel, bias=bias, padding=padding, stride=stride
    )
    print(f"Direct time: {direct_time}")
    print(f"Fourier time: {fourier_time}")

    y0 = torch_conv(signal, kernel, bias=bias, padding=padding, stride=stride)
    y1 = fft_conv(signal, kernel, bias=bias, padding=padding, stride=stride)
    abs_error = torch.abs(y0 - y1)
    print(f"Output size: {y0.size()}")
    print(f"Abs Error Mean: {abs_error.mean():.3E}")
    print(f"Abs Error Std Dev: {abs_error.std():.3E}")


print("\n--- 1D Convolution ---")
benchmark_conv(
    signal=torch.randn(3, 3, 4096),
    kernel=torch.randn(2, 3, 1025),
    bias=torch.randn(2),
    padding=512,
)

print("\n--- 2D Convolution ---")
benchmark_conv(
    signal=torch.randn(3, 3, 256, 256),
    kernel=torch.randn(2, 3, 21, 21),
    bias=torch.randn(2),
    padding=10,
)

print("\n--- 3D Convolution ---")
benchmark_conv(
    signal=torch.randn(3, 3, 64, 64, 64),
    kernel=torch.randn(2, 3, 9, 9, 9),
    bias=torch.randn(2),
    padding=4,
)
