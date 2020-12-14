from typing import Callable, NamedTuple, Union, Iterable
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
    signal: Tensor,
    kernel: Tensor,
    bias: Tensor,
    padding: Union[int, Iterable[int]] = 0,
    stride: Union[int, Iterable[int]] = 1,
    groups: int = 1,
):
    print(f"Signal size: {signal.shape}")
    print(f"Kernel size: {kernel.shape}")

    torch_conv = {1: f.conv1d, 2: f.conv2d, 3: f.conv3d}[signal.ndim - 2]
    direct_time = benchmark(
        torch_conv,
        signal,
        kernel,
        bias=bias,
        padding=padding,
        stride=stride,
        groups=groups,
    )
    fourier_time = benchmark(
        fft_conv,
        signal,
        kernel,
        bias=bias,
        padding=padding,
        stride=stride,
        groups=groups,
    )
    print(f"Direct time: {direct_time}")
    print(f"Fourier time: {fourier_time}")

    y0 = torch_conv(
        signal, kernel, bias=bias, padding=padding, stride=stride, groups=groups
    )
    y1 = fft_conv(
        signal, kernel, bias=bias, padding=padding, stride=stride, groups=groups
    )
    abs_error = torch.abs(y0 - y1)
    print(f"Output size: {y0.size()}")
    print(f"Abs Error Mean: {abs_error.mean():.3E}")
    print(f"Abs Error Std Dev: {abs_error.std():.3E}")


print("\n--- 1D Convolution ---")
benchmark_conv(
    signal=torch.randn(4, 4, 4091),
    kernel=torch.randn(6, 2, 1025),
    bias=torch.randn(6),
    padding=512,
    stride=3,
    groups=2,
)

print("\n--- 2D Convolution ---")
benchmark_conv(
    signal=torch.randn(4, 4, 256, 235),
    kernel=torch.randn(6, 2, 22, 21),
    bias=torch.randn(6),
    padding=(9, 10),
    stride=(2, 3),
    groups=2,
)

print("\n--- 3D Convolution ---")
benchmark_conv(
    signal=torch.randn(4, 4, 96, 72, 61),
    kernel=torch.randn(6, 2, 12, 7, 9),
    bias=torch.randn(6),
    padding=(2, 3, 4),
    stride=(1, 2, 3),
    groups=2,
)
