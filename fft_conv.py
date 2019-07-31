from functools import partial

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as f


def complex_matmul(a: Tensor, b: Tensor) -> Tensor:
	"""Multiplies two complex-valued tensors, where the last dimension contains the real and imaginary parts."""
	# Scalar matrix multiplication of two tensors, over only the first two dimensions.
	# Dimensions 3 and higher will have the same shape after multiplication.
	num_dims = len(a.shape)
	if num_dims == 4:
		scalar_matmul = partial(torch.einsum, 'abd, cbd -> acd')
	elif num_dims == 5:
		scalar_matmul = partial(torch.einsum, 'abde, cbde -> acde')
	elif num_dims == 6:
		scalar_matmul = partial(torch.einsum, 'abdef, cbdef -> acdef')
	else:
		raise ValueError(f'Not supported for {num_dims} dimensions.  Supported dimensions: 3, 4, 5')

	return torch.stack([scalar_matmul(a[..., 0], b[..., 0]) + scalar_matmul(a[..., 1], b[..., 1]),
	                    scalar_matmul(a[..., 1], b[..., 0]) - scalar_matmul(a[..., 0], b[..., 1])], dim=-1)


def fft_conv1d(signal: Tensor, kernel: Tensor, bias: Tensor = None, padding: int = 0, stride: int = 1) -> Tensor:
	"""Performs 1D convolution of two Tensors using a fast fourier transform, which is very fast for large kernel sizes.
	Also, optionally adds a bias Tensor after the convolution (in order ot mimic the PyTorch direct convolution).

	:param signal: Input tensor to be convolved with the kernel.  Shape: (batch, nchan, nsamples)
	:param kernel: Convolution kernel.  Shape: (channels_out, channels_in, kernel_size)
	:param bias: (Optional) bias tensor to add to the output.  Shape:  (channels_out, )
	:param padding: Number of zero samples to pad the input on the last dimension.
	:param stride: Convolution stride length
	:return: Convolved tensor
	"""
	# Pad the input signal & kernel tensors
	signal = f.pad(signal, [padding, padding])
	padded = f.pad(kernel, [0, signal.size(-1) - kernel.size(-1)])

	# Perform fourier convolution -- FFT, matrix multiply, then IFFT
	signal_fr = torch.rfft(signal, 1)
	padded_fr = torch.rfft(padded, 1)
	output_fr = complex_matmul(signal_fr, padded_fr)
	output = torch.irfft(output_fr, 1, signal_sizes=(signal.size(-1), ))

	# Keep outputs at strided intervals, then remove extra padded values
	output = output[..., ::stride]
	target_length = (signal.size(-1) - kernel.size(-1)) // stride + 1
	output = output[..., :target_length].contiguous()

	# Optionally, add a bias term before returning.
	if bias is not None:
		output += bias.view(1, -1, 1)

	return output


def fft_conv2d(signal: Tensor, kernel: Tensor, bias: Tensor = None, padding: int = 0, stride: int = 1) -> Tensor:
	"""Performs 2D convolution of two Tensors using a fast fourier transform, which is very fast for large kernel sizes.
	Also, optionally adds a bias Tensor after the convolution (in order ot mimic the PyTorch direct convolution).

	:param signal: Input tensor to be convolved with the kernel.  Shape: (batch, nchan, nrow, ncol)
	:param kernel: Convolution kernel.  Shape: (channels_out, channels_in, nrow, ncol)
	:param bias: (Optional) bias tensor to add to the output.  Shape:  (channels_out, )
	:param padding: Number of zero samples to pad the input on the last two dimensions.
	:param stride: Convolution stride length
	:return: Convolved tensor
	"""
	# Pad the input signal & kernel tensors
	signal = f.pad(signal, 2 * [padding, padding])
	padded = f.pad(kernel, [0, signal.size(-1) - kernel.size(-1),
	                        0, signal.size(-2) - kernel.size(-2)])

	# Perform fourier convolution -- FFT, matrix multiply, then IFFT
	signal_fr = torch.rfft(signal, 2)
	weight_fr = torch.rfft(padded, 2)
	output_fr = complex_matmul(signal_fr, weight_fr)
	output = torch.irfft(output_fr, 2, signal_sizes=(signal.size(-2), signal.size(-1)))

	# Keep outputs at strided intervals, then remove extra padded values
	output = output[..., ::stride, ::stride]
	target_length_2 = (signal.size(-2) - kernel.size(-2)) // stride + 1
	target_length_3 = (signal.size(-1) - kernel.size(-1)) // stride + 1
	output = output[..., :target_length_2, :target_length_3].contiguous()

	# Optionally, add a bias term before returning.
	if bias is not None:
		output += bias.view(1, -1, 1, 1)

	return output


def fft_conv3d(signal: Tensor, kernel: Tensor, bias: Tensor = None, padding: int = 0, stride: int = 1) -> Tensor:
	"""Performs 3D convolution of two Tensors using a fast fourier transform, which is very fast for large kernel sizes.
	Also, optionally adds a bias Tensor after the convolution (in order ot mimic the PyTorch direct convolution).

	:param signal: Input tensor to be convolved with the kernel.  Shape: (batch, nchan, nframe, nrow, ncol)
	:param kernel: Convolution kernel.  Shape: (channels_out, channels_in, nframe, nrow, ncol)
	:param bias: (Optional) bias tensor to add to the output.  Shape:  (channels_out, )
	:param padding: Number of zero samples to pad the input on the last three dimensions.
	:param stride: Convolution stride length
	:return: Convolved tensor
	"""
	# Pad the input signal & kernel tensors
	signal = f.pad(signal, 3 * [padding, padding])
	padded = f.pad(kernel, [0, signal.size(-1) - kernel.size(-1),
	                        0, signal.size(-2) - kernel.size(-2),
	                        0, signal.size(-3) - kernel.size(-3)])

	# Perform fourier convolution -- FFT, matrix multiply, then IFFT
	signal_fr = torch.rfft(signal, 3)
	weight_fr = torch.rfft(padded, 3)
	output_fr = complex_matmul(signal_fr, weight_fr)
	output = torch.irfft(output_fr, 3, signal_sizes=(signal.size(-3), signal.size(-2), signal.size(-1)))

	# Keep outputs at strided intervals, then remove extra padded values
	output = output[..., ::stride, ::stride, ::stride]
	target_length_2 = (signal.size(-3) - kernel.size(-3)) // stride + 1
	target_length_3 = (signal.size(-2) - kernel.size(-2)) // stride + 1
	target_length_4 = (signal.size(-1) - kernel.size(-1)) // stride + 1
	output = output[..., :target_length_2, :target_length_3, :target_length_4].contiguous()

	# Optionally, add a bias term before returning.
	if bias is not None:
		output += bias.view(1, -1, 1, 1, 1)

	return output


class FFTConv1d(nn.Module):
	"""PyTorch 1D convoluton layer based on FFT, which is faster for large kernels."""

	def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0, stride: int = 1,
	             bias: bool = True):
		"""
		:param in_channels: Number of channels in input tensors
		:param out_channels: Number of channels in output tensors
		:param kernel_size: Size of the 1D convolution kernel
		:param padding: Amount of zero-padding to add to the input tensor
		:param stride: Convolution stride length. Defaults to 1, as in standard convolution
		:param bias: If True, includes an additional bias term, which is added to the output after convolution
		"""
		super().__init__()
		self.kernel_size = kernel_size
		self.padding = padding
		self.stride = stride
		self.bias = nn.Parameter(torch.randn(out_channels, )) if bias else None
		self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))

	def forward(self, signal):
		return fft_conv1d(signal, self.weight, bias=self.bias, padding=self.padding, stride=self.stride)


class FFTConv2d(nn.Module):
	"""PyTorch 2D convoluton layer based on FFT, which is faster for large kernels."""

	def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0, stride: int = 1,
	             bias: bool = True):
		"""
		:param in_channels: Number of channels in input tensors
		:param out_channels: Number of channels in output tensors
		:param kernel_size: Size of the 2D convolution kernel.  (i.e. kernel_size=3 gives a 3x3 kernel)
		:param padding: Amount of zero-padding to add to the input tensor
		:param stride: Convolution stride length. Defaults to 1, as in standard convolution
		:param bias: If True, includes an additional bias term, which is added to the output after convolution
		"""
		super().__init__()
		self.kernel_size = kernel_size
		self.padding = padding
		self.stride = stride
		self.bias = nn.Parameter(torch.randn(out_channels, )) if bias else None
		self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))

	def forward(self, signal):
		return fft_conv2d(signal, self.weight, bias=self.bias, padding=self.padding, stride=self.stride)


class FFTConv3d(nn.Module):
	"""PyTorch 3D convoluton layer based on FFT, which is faster for large kernels."""

	def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0, stride: int = 1,
	             bias: bool = True):
		"""
		:param in_channels: Number of channels in input tensors
		:param out_channels: Number of channels in output tensors
		:param kernel_size: Size of the 2D convolution kernel.  (i.e. kernel_size=3 gives a 3x3x3 kernel)
		:param padding: Amount of zero-padding to add to the input tensor
		:param stride: Convolution stride length. Defaults to 1, as in standard convolution
		:param bias: If True, includes an additional bias term, which is added to the output after convolution
		"""
		super().__init__()
		self.kernel_size = kernel_size
		self.padding = padding
		self.stride = stride
		self.bias = nn.Parameter(torch.randn(out_channels, )) if bias else None
		self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))

	def forward(self, signal):
		return fft_conv3d(signal, self.weight, bias=self.bias, padding=self.padding, stride=self.stride)
