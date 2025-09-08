#include <iostream>
#include <cuda_runtime.h>

void transpose_reference(const float *src, float *dst, int width, int height)
{
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            dst[x * height + y] = src[y * width + x];
}

__global__ void transpose(const float *src, float *dst, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        dst[x * height + y] = src[y * width + x];
    }
}

