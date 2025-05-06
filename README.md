# PCA: EXP-1  SUM ARRAY GPU
<h3>ENTER YOUR NAME: Priyadharshan S</h3>
<h3>ENTER YOUR REGISTER NO: 212223240127</h3>
<h3>EX. NO</h3>
<h3>DATE</h3>
<h1> <align=center> SUM ARRAY ON HOST AND DEVICE </h3>
PCA-GPU-based-vector-summation.-Explore-the-differences.
i) Using the program sumArraysOnGPU-timer.cu, set the block.x = 1023. Recompile and run it. Compare the result with the execution configuration of block.x = 1024. Try to explain the difference and the reason.

ii) Refer to sumArraysOnGPU-timer.cu, and let block.x = 256. Make a new kernel to let each thread handle two elements. Compare the results with other execution confi gurations.
## AIM:

To perform vector addition on host and device.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler




## PROCEDURE:

1. Initialize the device and set the device properties.
2. Allocate memory on the host for input and output arrays.
3. Initialize input arrays with random values on the host.
4. Allocate memory on the device for input and output arrays, and copy input data from host to device.
5. Launch a CUDA kernel to perform vector addition on the device.
6. Copy output data from the device to the host and verify the results against the host's sequential vector addition. Free memory on the host and the device.

## PROGRAM:

```c
#include<iostream>
#include<cuda_runtime.h>
#include"cuda_utils.cuh"

//#define N 10000

//#define N 1000 // for threads since their limit is 1024

#define N (33*1024)

// Using Blocks 

__global__ void add(int* a, int* b, int* c) {
	int tid = blockIdx.x;
	if(tid<N){
		c[tid] = a[tid] + b[tid];
	}
}

// Using Threads

//__global__ void add(int* a, int* b, int* c) {
//	int tid = threadIdx.x;
//	if (tid < N) {
//		c[tid] = a[tid] + b[tid];
//	}
//}

// Using Threads and Blocks

//__global__ void add(int* a, int* b, int* c) {
//	int tid = threadIdx.x + blockIdx.x * blockDim.x;
//	while (tid < N) {
//		c[tid] = a[tid] + b[tid];
//		tid += blockDim.x * gridDim.x;
//	}
//}


void performVecAdd() {
	int a[N], b[N], c[N];
	int* dev_a, * dev_b, * dev_c;

	HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));

	for (int i = 0;i < N;i++) {
		a[i] = -i;
		b[i] = i * i;
	}

	HANDLE_ERROR(cudaMemcpy(dev_a, &a, N * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, &b, N * sizeof(int), cudaMemcpyHostToDevice));

	//add << <N, 1 >> > (dev_a, dev_b, dev_c); // For blocks
	//add << <1, N >> > (dev_a, dev_b, dev_c); // For threads
	add << <128, 128 >> > (dev_a, dev_b, dev_c); // For blocks and threads
	HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

	for (int i = 0;i < N;i++) {
		std::cout << a[i] << "+" << b[i] << " = " << c[i] << "\n";
	}

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}

```
## OUTPUT:
![image](https://github.com/user-attachments/assets/5ab7af65-b72c-4167-8bb4-eeb855370362)

## RESULT:
Thus, Implementation of sum arrays on host and device is done in nvcc cuda using random number.
