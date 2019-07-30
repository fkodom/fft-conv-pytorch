import torch
from torch import nn
from torch.nn import functional as f


def complex_matmul(a, b):
	"""Given two tensors of with the last dimension being the real and imaginary part, returns a multiplied
	by the conjugate of b, the multiplication being with respect to the second dimension.
	"""
	num_dims = len(a.shape)

	def scalar_matmul(x, y):
		if num_dims == 4:
			return torch.einsum('bct,dct->bdt', x, y)
		elif num_dims == 5:
			return torch.einsum('abtu,cbtu->actu', x, y)
		elif num_dims == 6:
			return torch.einsum('abtuv,cbtuv->actuv', x, y)
		else:
			raise ValueError(f'Operation not supported for {num_dims} dimensions.  Supported dimensions: 3, 4, 5')

	return torch.stack([scalar_matmul(a[..., 0], b[..., 0]) + scalar_matmul(a[..., 1], b[..., 1]),
	                    scalar_matmul(a[..., 1], b[..., 0]) - scalar_matmul(a[..., 0], b[..., 1])], dim=-1)


def fft_conv1d(signal, kernel, bias=None, padding: int = 0, stride: int = 1):
	signal = f.pad(signal, [padding, padding])
	padded = f.pad(kernel, [0, signal.size(-1) - kernel.size(-1)])
	signal_fr = torch.rfft(signal, 1)
	weight_fr = torch.rfft(padded, 1)
	output_fr = complex_matmul(signal_fr, weight_fr)
	output = torch.irfft(output_fr, 1, signal_sizes=(signal.size(-1), ))
	output = output[..., ::stride]
	target_length = (signal.size(-1) - kernel.size(-1)) // stride + 1
	output = output[..., :target_length].contiguous()
	if bias is not None:
		output += bias
	return output


def fft_conv2d(signal, kernel, bias=None, padding: int = 0, stride: int = 1):
	signal = f.pad(signal, 2 * [padding, padding])
	padded = f.pad(kernel, [0, signal.size(-1) - kernel.size(-1),
	                        0, signal.size(-2) - kernel.size(-2)])
	signal_fr = torch.rfft(signal, 2)
	weight_fr = torch.rfft(padded, 2)
	output_fr = complex_matmul(signal_fr, weight_fr)
	output = torch.irfft(output_fr, 2, signal_sizes=(signal.size(-2), signal.size(-1)))
	output = output[..., ::stride, ::stride]
	target_length_2 = (signal.size(-2) - kernel.size(-2)) // stride + 1
	target_length_3 = (signal.size(-1) - kernel.size(-1)) // stride + 1
	output = output[..., :target_length_2, :target_length_3].contiguous()
	if bias is not None:
		output += bias
	return output


def fft_conv3d(signal, kernel, bias=None, padding: int = 0, stride: int = 1):
	signal = f.pad(signal, 3 * [padding, padding])
	padded = f.pad(kernel, [0, signal.size(-1) - kernel.size(-1),
	                        0, signal.size(-2) - kernel.size(-2),
	                        0, signal.size(-3) - kernel.size(-3)])
	signal_fr = torch.rfft(signal, 3)
	weight_fr = torch.rfft(padded, 3)
	output_fr = complex_matmul(signal_fr, weight_fr)
	output = torch.irfft(output_fr, 3, signal_sizes=(signal.size(-3), signal.size(-2), signal.size(-1)))
	output = output[..., ::stride, ::stride, ::stride]
	target_length_2 = (signal.size(-3) - kernel.size(-3)) // stride + 1
	target_length_3 = (signal.size(-2) - kernel.size(-2)) // stride + 1
	target_length_4 = (signal.size(-1) - kernel.size(-1)) // stride + 1
	output = output[..., :target_length_2, :target_length_3, :target_length_4].contiguous()
	if bias is not None:
		output += bias
	return output


class FFTConv1d(nn.Module):
	"""Convoluton based on FFT, faster for large kernels."""

	def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
		super().__init__()
		self.kernel_size = kernel_size
		self.stride = stride
		self.bias = nn.Parameter(torch.zeros(out_channels, 1)) if bias else None
		self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size))

	def forward(self, signal):
		return fft_conv1d(signal, self.weight, bias=self.bias, stride=self.stride)


class FFTConv2d(nn.Module):
	"""Convoluton based on FFT, faster for large kernels."""

	def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
		super().__init__()
		self.kernel_size = kernel_size
		self.stride = stride
		self.bias = nn.Parameter(torch.zeros(out_channels, 1)) if bias else None
		self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size))

	def forward(self, signal):
		return fft_conv2d(signal, self.weight, bias=self.bias, stride=self.stride)


class FFTConv3d(nn.Module):
	"""Convoluton based on FFT, faster for large kernels."""

	def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
		super().__init__()
		self.kernel_size = kernel_size
		self.stride = stride
		self.bias = nn.Parameter(torch.zeros(out_channels, 1)) if bias else None
		self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size))

	def forward(self, signal):
		return fft_conv3d(signal, self.weight, bias=self.bias, stride=self.stride)
