#include <sstream>
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

__global__ void haversine_distance_kernel(int size, const double *x1,const double *y1,
    const double *x2,const double *y2, double *dist)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) return;

  const double R = 6371;  // kilometres

  const double PI = 3.14159265358979323846;
  const double deg_to_rad = PI / 180.0;

  double x1_rad = x1[idx] * deg_to_rad;  // longitude1 in radians
  double y1_rad = y1[idx] * deg_to_rad;  // latitude1 in radians
  double x2_rad = x2[idx] * deg_to_rad;  // longitude2 in radians
  double y2_rad = y2[idx] * deg_to_rad;  // latitude2 in radians

  double dLat = y2_rad - y1_rad;
  double dLon = x2_rad - x1_rad;

  double a = pow(sin(dLat / 2), 2) +
            cos(y1_rad) * cos(y2_rad) *
            pow(sin(dLon / 2), 2);
  double c = 2 * atan2(sqrt(a), sqrt(1 - a));
  dist[idx] = R * c;
}


void run_kernel(int size, const double *x1,const double *y1, const double *x2,const double *y2, double *dist){
  dim3 dimBlock(1024);
  printf("in run_kernel dimBlock.x=%d\n",dimBlock.x);
  dim3 dimGrid(ceil((double)size / dimBlock.x));
  haversine_distance_kernel<<<dimGrid, dimBlock>>> (size,x1,y1,x2,y2,dist);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::stringstream strstr;
    strstr << "run_kernel launch failed" << std::endl;
    strstr << "dimBlock: " << dimBlock.x << ", " << dimBlock.y << std::endl;
    strstr << "dimGrid: " << dimGrid.x << ", " << dimGrid.y << std::endl;
    strstr << cudaGetErrorString(error);
    throw strstr.str();
  }
}
