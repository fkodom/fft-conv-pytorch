from functools import lru_cache, partial
from timeit import Timer
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as f
from tqdm import tqdm

from fft_conv_pytorch.fft_conv import fft_conv, to_ntuple


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


@lru_cache(maxsize=1)
def _get_conv_inputs(
    ndim: int,
    input_size: int,
    kernel_size: Union[int, Iterable[int]],
    batch_size: int = 2,
    in_channels: int = 8,
    out_channels: int = 8,
):
    dims = ndim * [input_size]
    signal = torch.randn(batch_size, in_channels, *dims)

    kernel_size = to_ntuple(kernel_size, n=signal.ndim - 2)
    weight = torch.randn(out_channels, in_channels, *kernel_size, requires_grad=True)
    bias = torch.randn(out_channels, requires_grad=True)

    return signal, weight, bias


def benchmark_conv(
    ndim: int,
    input_size: int,
    kernel_size: int,
    fft: bool = True,
    num_iterations: int = 10,
):
    conv_fn = fft_conv if fft else getattr(f, f"conv{ndim}d")
    signal, weight, bias = _get_conv_inputs(
        ndim=ndim, input_size=input_size, kernel_size=kernel_size
    )
    return benchmark(conv_fn, signal, weight, bias=bias, num_iterations=num_iterations)


def benchmark_kernel_size(
    kernel_sizes: Sequence[int],
    ndim: int,
    input_size: int,
    fft: bool = True,
    num_iterations: int = 10,
    desc: str = "",
) -> List[Benchmark]:
    fn = partial(
        benchmark_conv,
        ndim=ndim,
        input_size=input_size,
        fft=fft,
        num_iterations=num_iterations,
    )
    return [fn(kernel_size=k) for k in tqdm(kernel_sizes, desc=desc)]


def _plot_benchmarks(
    benchmarks: List[Benchmark],
    config: Dict,
    ax: plt.Axes,
    color: str,
    label: Optional[str] = None,
):
    xs = config["kernel_sizes"]
    ys = np.array([b.mean * 1000 for b in benchmarks])
    std = np.array([b.std * 1000 for b in benchmarks])
    ax.plot(xs, ys, color, label=label)
    ax.fill_between(
        xs, ys - std, ys + std, facecolor=color, alpha=0.25, label="_nolegend_"
    )

    ndim = config["ndim"]
    ax.set_title(f"{ndim}D")
    kernel_size_str = "(" + " x ".join(["n"] * ndim) + ")"
    ax.set_xlabel(f"Kernel Size {kernel_size_str}")


if __name__ == "__main__":
    import os

    configs = [
        {
            "ndim": 1,
            "input_size": 4096,
            "num_iterations": 256,
            "kernel_sizes": np.arange(64, 513, 64),
        },
        {
            "ndim": 2,
            "input_size": 512,
            "num_iterations": 16,
            "kernel_sizes": np.arange(4, 49, 6),
        },
        {
            "ndim": 3,
            "input_size": 64,
            "num_iterations": 16,
            "kernel_sizes": np.arange(2, 17, 2),
        },
    ]

    save_dir = os.path.join(os.path.dirname(__file__), os.path.pardir)
    fix, ax = plt.subplots(
        1, len(configs), figsize=(4 * len(configs), 4), squeeze=False
    )

    for i, config in enumerate(configs):
        fft = benchmark_kernel_size(fft=True, **config, desc=f"FFT {config['ndim']}D")
        _plot_benchmarks(fft, config=config, ax=ax[0, i], color="r", label="FFT")

        direct = benchmark_kernel_size(
            fft=False, **config, desc=f"Direct {config['ndim']}D"
        )
        _plot_benchmarks(direct, config=config, ax=ax[0, i], color="b", label="Direct")

    ax[0, 0].set_ylabel("Execution Time (ms)")
    plt.legend(["FFT", "Direct"])
    plt.savefig(os.path.join(save_dir, "benchmark.png"))
