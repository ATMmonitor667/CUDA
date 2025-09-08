#include <iostream>
#include <cuda_runtime.h>


__global__ void vectorAdd(int *A, int   *B, int  *C, int N)
{
  // The function as it states you allocate a place in the SM where the computation stems from
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
      // Each thread computes one element of the vector C, and we do this all simulataneously
        C[i] = A[i] + B[i];
    }
}

int main(void)
{
  int n = 10; // allocate 10 integers
  size_t size = n * sizeof(float); // like in C programming we use the bit size of the type to make a length;
  int h_a[10], h_b[10], h_c[10]; // Allocate the arrays in the host
  // Initialize input vectors
  for (int i = 0; i < n; i++) {
    h_a[i] = i;
    h_b[i] = i * 2;
  }
  int *d_a, *d_b, *d_c;
  cudaMalloc((void**)&d_a, size);
  cudaMalloc((void**)&d_b, size);
  cudaMalloc((void**)&d_c, size);
  // This is really important, we allocate memory for all three vectors in the device
  // Copy only the vectors which we will use as input vectors from host memory to device memory
  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
  // Notice how we did not copy the output vector to the device
  // Launch the kernel on the GPU with N threads
  vectorAdd<<<1, 10>>>(d_a, d_b, d_c, n);
  cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
  // Here this is important we did the computation for the output vector in the host device and then we do the revberse
  // Where before we copied the memory from host to device now we copy the memory from device to host
  // Print the result
  for (int i = 0; i < n; i++) {
    std::cout << h_c[i] << " ";
  }
  std::cout << std::endl;
  // Free device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  // at the end of the program we simply return 0
  return 0;


}