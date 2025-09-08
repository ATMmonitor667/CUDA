#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void multiplyByConstant(int *array, int constant, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        array[index] *= constant;
    }
}
__global__ void combineArray(int * array1, int* array2, int* array3, int size)
{
  int index = threadIdx.x + blockDim.x*blockIdx.x;
  if(index < size)
  {
    array3[index] = array1[index] * array2[index];
  }
}
// Problem 1: Implement a function in the GPU which takes an array and multiples all the element in the array
// by a specific constant value



int main(void)
{
  int n = 10;
  int size = n * sizeof(int);
  int h_a[10], h_b[10], h_c[10];
  for(int i = 0; i < 10; i ++)
  {
    h_a[i] = i;
    h_b[i] = i*2;
  }
  int *d_a, *d_b, *d_c;
  cudaMalloc((void ** )& d_a, size);
  cudaMalloc((void **)& d_b, size);
  cudaMalloc((void**)&d_c, size); //Allocate all of them but not all Host to device
  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
combineArray<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

  cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

  for(int i = 0; i < 10; i++)
  {
    cout << h_a[i] << " " << h_b[i] << " " << h_c[i] << endl;

  }
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  /*int n = 10;
  int size = n * sizeof(int);
  int h_a[10];
  for (int i = 0; i < n; i++) {
    h_a[i] = i;
  }
  int *d_a;
  cudaMalloc((void**)&d_a, size);
  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  multiplyByConstant<<<1, 10>>>(d_a, 5.0f, n);
  cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++) {
    std::cout << h_a[i] << " ";
  }
  std::cout << std::endl;
    //int count;
    //cudaGetDeviceCount(&count);
    //std::cout << "Number of CUDA devices: " << count << std::endl;*/
    return 0;
}