import torch
import torch.nn.functional as f
import numpy as np
import matplotlib.pyplot as plt

from fft_conv import fft_conv
import benchmark as bm 


device = 'cpu'


fig, ax =plt.subplots(1,3)
col1 = 'darkorange'
col2 = 'seagreen'
# number of benchmark runs
n_benchrun = 10


print('\nComputing 1d convolution>>>>>>>>>>>\n')

signal = torch.randn(1,1,10000,device = device)

# array of kern sizes to test
kern_size = np.linspace(9,700,10,dtype=np.int)

# init
direct_time = np.zeros((2,len(kern_size)))
fft_time = np.zeros((2,len(kern_size)))

for i,k in enumerate(kern_size):
	print('kernel size : ',k,' number of runs : ', n_benchrun )
	kernel = torch.randn(1,1,k,device = device)
	
	direct_bm = bm.benchmark(f.conv1d,signal,kernel,num_iterations= n_benchrun)
	fft_bm = bm.benchmark(fft_conv,signal,kernel,num_iterations= n_benchrun)

	direct_time[0,i] = direct_bm.mean
	direct_time[1,i] = direct_bm.std

	fft_time[0,i] = fft_bm.mean
	fft_time[1,i] = fft_bm.std


ax[0].plot(kern_size,direct_time[0],color=col1,label = 'torch convolution')
ax[0].plot(kern_size,fft_time[0],color=col2,label = 'fft convolution')

ax[0].fill_between(kern_size,
 				direct_time[0] - direct_time[1],
 				direct_time[0] + direct_time[1],
 				 alpha = 0.5,
 				 color=col1)
ax[0].fill_between(kern_size,
 				fft_time[0] - fft_time[1],
 				fft_time[0] + fft_time[1],
 				 alpha = 0.5,
 				 color=col2)

ax[0].set_title('1D convolution')
ax[0].set_xticks(kern_size)
ax[0].set_xlabel('kernel size ')
ax[0].set_ylabel('execution time (s)')
ax[0].legend()


print('\nComputing 2d convolution>>>>>>>>>>>\n')

signal = torch.randn(1,1,500,500,device = device)

# array of kern sizes to test
kern_size = np.linspace(3,20,10,dtype=np.int)

# init
direct_time = np.zeros((2,len(kern_size)))
fft_time = np.zeros((2,len(kern_size)))

for i,k in enumerate(kern_size):
	print('kernel size : ',k,' number of runs : ', n_benchrun )
	kernel = torch.randn(1,1,k,k,device = device)
	
	direct_bm = bm.benchmark(f.conv2d,signal,kernel,num_iterations= n_benchrun)
	fft_bm = bm.benchmark(fft_conv,signal,kernel,num_iterations= n_benchrun)

	direct_time[0,i] = direct_bm.mean
	direct_time[1,i] = direct_bm.std

	fft_time[0,i] = fft_bm.mean
	fft_time[1,i] = fft_bm.std




ax[1].plot(kern_size,direct_time[0],color=col1,label = 'torch convolution')
ax[1].plot(kern_size,fft_time[0],color=col2,label = 'fft convolution')

ax[1].fill_between(kern_size,
 				direct_time[0] - direct_time[1],
 				direct_time[0] + direct_time[1],
 				 alpha = 0.5,
 				 color=col1)
ax[1].fill_between(kern_size,
 				fft_time[0] - fft_time[1],
 				fft_time[0] + fft_time[1],
 				 alpha = 0.5,
 				 color=col2)

ax[1].set_title('2D convolution')
ax[1].set_xticks(kern_size)
ax[1].set_xlabel('kernel size (square kernel)')
ax[1].set_ylabel('execution time (s)')
ax[1].legend()


print('\nComputing 3d convolution>>>>>>>>>>>\n')

signal = torch.randn(1,1,250,250,250,device = device)

# array of kern sizes to test
kern_size = np.linspace(3,12,10,dtype=np.int)


# init
direct_time = np.zeros((2,len(kern_size)))
fft_time = np.zeros((2,len(kern_size)))

for i,k in enumerate(kern_size):
	print('kernel size : ',k,' number of runs : ', n_benchrun )
	kernel = torch.randn(1,1,k,k,k,device = device)
	
	direct_bm = bm.benchmark(f.conv3d,signal,kernel,num_iterations= n_benchrun)
	fft_bm = bm.benchmark(fft_conv,signal,kernel,num_iterations= n_benchrun)

	direct_time[0,i] = direct_bm.mean
	direct_time[1,i] = direct_bm.std

	fft_time[0,i] = fft_bm.mean
	fft_time[1,i] = fft_bm.std




ax[2].plot(kern_size,direct_time[0],color=col1,label = 'torch convolution')
ax[2].plot(kern_size,fft_time[0],color=col2,label = 'fft convolution')

ax[2].fill_between(kern_size,
 				direct_time[0] - direct_time[1],
 				direct_time[0] + direct_time[1],
 				 alpha = 0.5,
 				 color=col1)
ax[2].fill_between(kern_size,
 				fft_time[0] - fft_time[1],
 				fft_time[0] + fft_time[1],
 				 alpha = 0.5,
 				 color=col2)

ax[2].set_title('3D convolution')
ax[2].set_xticks(kern_size)
ax[2].set_xlabel('kernel size (cubic kernel)')
ax[2].set_ylabel('execution time (s)')
ax[2].legend()


plt.show()