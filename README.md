# FFT Conv PyTorch

Implementation of 1D, 2D, and 3D FFT convolutions in PyTorch.  
* Faster than direct convolution for large kernels.
* **Much slower** than direct convolution for small kernels.
* Typically, FFT convolution is faster when the kernel has >100 elements.
    * Dependent on machine and PyTorch version.
