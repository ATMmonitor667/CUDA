
// cuda_10_problems.cu
// Build: nvcc -O2 -std=c++17 cuda_10_problems.cu -o cuda_10_problems
// Run:   ./cuda_10_problems
//
// Implements 10 problems in increasing difficulty with both CPU and GPU versions:
// 1) Vector Add
// 2) AXPY (y = a*x + y)
// 3) ReLU (elementwise max(0,x))
// 4) Reduction Sum
// 5) Exclusive Prefix Sum (two-pass: per-block scan + CPU offsets + add)
// 6) Matrix Multiply (naive)
// 7) Matrix Multiply (tiled shared memory)
// 8) Histogram (256 bins, uint8 data)
// 9) CSR SpMV (y = A*x)
// 10) N-body (one Euler step)
//
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <cstdint>
#include <cmath>
#include <limits>
#include <string>

#define CUDA_CHECK(stmt) do { \
    cudaError_t err = (stmt); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// =========================
// 1) VECTOR ADD
// =========================
void vector_add_cpu(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) C[i] = A[i] + B[i];
}

__global__ void vector_add_kernel(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

// =========================
/* 2) AXPY: y = a*x + y */
// =========================
void axpy_cpu(float a, const float* X, float* Y, int N) {
    for (int i = 0; i < N; ++i) Y[i] = a * X[i] + Y[i];
}

__global__ void axpy_kernel(float a, const float* X, float* Y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) Y[i] = a * X[i] + Y[i];
}

// =========================
// 3) ReLU: Y = max(0, X)
// =========================
void relu_cpu(const float* X, float* Y, int N) {
    for (int i = 0; i < N; ++i) Y[i] = X[i] > 0.0f ? X[i] : 0.0f;
}

__global__ void relu_kernel(const float* X, float* Y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) Y[i] = X[i] > 0.0f ? X[i] : 0.0f;
}

// =========================
// 4) Reduction Sum
// =========================
float reduction_sum_cpu(const float* X, int N) {
    double s = 0.0; // use double for better accuracy on CPU
    for (int i = 0; i < N; ++i) s += X[i];
    return static_cast<float>(s);
}

template<int BLOCK>
__global__ void reduction_block_kernel(const float* __restrict__ d_in, float* __restrict__ d_partials, int N) {
    __shared__ float sdata[BLOCK];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;
    // grid-stride loop
    for (int i = gid; i < N; i += gridDim.x * blockDim.x) {
        sum += d_in[i];
    }
    sdata[tid] = sum;
    __syncthreads();
    // reduce in shared memory
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) d_partials[blockIdx.x] = sdata[0];
}

// =========================
// 5) Exclusive Prefix Sum (two-pass)
// =========================
// CPU reference exclusive scan
void exclusive_scan_cpu(const float* in, float* out, int N) {
    float acc = 0.0f;
    for (int i = 0; i < N; ++i) {
        out[i] = acc;
        acc += in[i];
    }
}

// Per-block Blelloch exclusive scan on up to BLOCK elements.
// Assumes blockDim.x == BLOCK and loads zeros for out-of-range.
template<int BLOCK>
__global__ void block_exclusive_scan_kernel(const float* __restrict__ in,
                                            float* __restrict__ out,
                                            float* __restrict__ block_sums,
                                            int N) {
    __shared__ float s[BLOCK];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    float x = (gid < N) ? in[gid] : 0.0f;
    s[tid] = x;
    __syncthreads();

    // upsweep
    for (int offset = 1; offset < BLOCK; offset <<= 1) {
        int idx = (tid + 1) * (offset << 1) - 1;
        if (idx < BLOCK) s[idx] += s[idx - offset];
        __syncthreads();
    }

    // save total sum for this block
    if (tid == BLOCK - 1) {
        block_sums[blockIdx.x] = s[tid];
        s[tid] = 0.0f; // convert to exclusive
    }
    __syncthreads();

    // downsweep
    for (int offset = BLOCK >> 1; offset > 0; offset >>= 1) {
        int idx = (tid + 1) * (offset << 1) - 1;
        if (idx < BLOCK) {
            float t = s[idx - offset];
            s[idx - offset] = s[idx];
            s[idx] += t;
        }
        __syncthreads();
    }

    if (gid < N) out[gid] = s[tid];
}

// Add per-block offsets to scanned output
__global__ void add_block_offsets_kernel(float* out, const float* block_offsets, int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < N) out[gid] += block_offsets[blockIdx.x];
}

// =========================
// 6) Matrix Multiply (naive)
// C[MxK] = A[MxN] * B[NxK]
// =========================
void matmul_naive_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < K; ++c) {
            float sum = 0.0f;
            for (int i = 0; i < N; ++i) {
                sum += A[r * N + i] * B[i * K + c];
            }
            C[r * K + c] = sum;
        }
    }
}

__global__ void matmul_naive_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < M && c < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += A[r * N + i] * B[i * K + c];
        }
        C[r * K + c] = sum;
    }
}

// =========================
// 7) Matrix Multiply (tiled shared memory)
// =========================
template<int TILE>
__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;
    for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < N) ? A[row * N + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < N && col < K) ? B[b_row * K + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < K) C[row * K + col] = acc;
}

// =========================
// 8) Histogram (256 bins, uint8 data)
// =========================
void histogram_cpu(const uint8_t* data, int N, unsigned int* hist) {
    std::fill(hist, hist + 256, 0u);
    for (int i = 0; i < N; ++i) hist[data[i]]++;
}

__global__ void histogram_kernel(const uint8_t* data, int N, unsigned int* hist) {
    __shared__ unsigned int hist_s[256];
    // initialize shared histogram
    for (int i = threadIdx.x; i < 256; i += blockDim.x) hist_s[i] = 0u;
    __syncthreads();

    // grid-stride over data
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
        unsigned int v = data[i];
        atomicAdd(&hist_s[v], 1u);
    }
    __syncthreads();

    // write back to global with atomics
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        atomicAdd(&hist[i], hist_s[i]);
    }
}

// =========================
// 9) CSR SpMV (y = A * x)
// =========================
void spmv_csr_cpu(const int* rowPtr, const int* colInd, const float* val,
                  const float* x, float* y, int M) {
    for (int r = 0; r < M; ++r) {
        float sum = 0.0f;
        for (int jj = rowPtr[r]; jj < rowPtr[r + 1]; ++jj) {
            sum += val[jj] * x[colInd[jj]];
        }
        y[r] = sum;
    }
}

__global__ void spmv_csr_kernel(const int* rowPtr, const int* colInd, const float* val,
                                const float* x, float* y, int M) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < M) {
        float sum = 0.0f;
        for (int jj = rowPtr[r]; jj < rowPtr[r + 1]; ++jj) {
            sum += val[jj] * x[colInd[jj]];
        }
        y[r] = sum;
    }
}

// =========================
// 10) N-body (one Euler step)
// =========================
struct float2_ {
    float x, y;
};

void nbody_step_cpu(float2_* pos, float2_* vel, const float* mass, int N, float dt, float G = 1.0f, float soft2 = 1e-6f) {
    std::vector<float2_> acc(N);
    for (int i = 0; i < N; ++i) {
        float ax = 0.0f, ay = 0.0f;
        for (int j = 0; j < N; ++j) if (i != j) {
            float dx = pos[j].x - pos[i].x;
            float dy = pos[j].y - pos[i].y;
            float r2 = dx*dx + dy*dy + soft2;
            float inv = rsqrtf(r2);            // 1/sqrt(r2)
            float inv3 = inv * inv * inv;      // 1/r^3
            float f = G * mass[j] * inv3;
            ax += f * dx;
            ay += f * dy;
        }
        acc[i].x = ax; acc[i].y = ay;
    }
    for (int i = 0; i < N; ++i) {
        vel[i].x += dt * acc[i].x;
        vel[i].y += dt * acc[i].y;
        pos[i].x += dt * vel[i].x;
        pos[i].y += dt * vel[i].y;
    }
}

__global__ void nbody_step_kernel(float2_* pos, float2_* vel, const float* mass, int N, float dt, float G, float soft2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float ax = 0.0f, ay = 0.0f;
    float pix = pos[i].x, piy = pos[i].y;
    for (int j = 0; j < N; ++j) {
        if (i == j) continue;
        float dx = pos[j].x - pix;
        float dy = pos[j].y - piy;
        float r2 = dx*dx + dy*dy + soft2;
        float inv = rsqrtf(r2);
        float inv3 = inv * inv * inv;
        float f = G * mass[j] * inv3;
        ax += f * dx;
        ay += f * dy;
    }
    vel[i].x += dt * ax;
    vel[i].y += dt * ay;
    pos[i].x += dt * vel[i].x;
    pos[i].y += dt * vel[i].y;
}

// =========================
// Helpers
// =========================
bool almost_equal(float a, float b, float eps = 1e-3f) {
    return std::fabs(a - b) <= eps * std::max(1.0f, std::max(std::fabs(a), std::fabs(b)));
}
bool arrays_close(const std::vector<float>& A, const std::vector<float>& B, float eps = 1e-3f) {
    if (A.size() != B.size()) return false;
    for (size_t i = 0; i < A.size(); ++i) if (!almost_equal(A[i], B[i], eps)) return false;
    return true;
}

// Random data generators
std::mt19937 rng(42);
float frand() {
    static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    return dist(rng);
}
uint8_t urand_byte() {
    static std::uniform_int_distribution<int> dist(0, 255);
    return static_cast<uint8_t>(dist(rng));
}

// Run a small self-test for each problem
int main() {
    CUDA_CHECK(cudaSetDevice(0));
    std::cout << "Running 10 CUDA problems (CPU vs GPU) self-tests...\n";

    // 1) Vector Add
    {
        int N = 1 << 20;
        std::vector<float> A(N), B(N), C_cpu(N), C_gpu(N);
        std::generate(A.begin(), A.end(), frand);
        std::generate(B.begin(), B.end(), frand);
        vector_add_cpu(A.data(), B.data(), C_cpu.data(), N);

        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC, N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA, A.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, B.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        int TPB = 256, blocks = ceil_div(N, TPB);
        vector_add_kernel<<<blocks, TPB>>>(dA, dB, dC, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(C_gpu.data(), dC, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
        std::cout << "1) Vector Add: " << (arrays_close(C_cpu, C_gpu) ? "OK" : "MISMATCH") << "\n";
    }

    // 2) AXPY
    {
        int N = 1 << 20; float a = 2.5f;
        std::vector<float> X(N), Y0(N), Y_cpu(N), Y_gpu(N);
        std::generate(X.begin(), X.end(), frand);
        std::generate(Y0.begin(), Y0.end(), frand);
        Y_cpu = Y0;
        axpy_cpu(a, X.data(), Y_cpu.data(), N);

        float *dX, *dY;
        CUDA_CHECK(cudaMalloc(&dX, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dY, N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dX, X.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dY, Y0.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        int TPB = 256, blocks = ceil_div(N, TPB);
        axpy_kernel<<<blocks, TPB>>>(a, dX, dY, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(Y_gpu.data(), dY, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dX)); CUDA_CHECK(cudaFree(dY));
        std::cout << "2) AXPY: " << (arrays_close(Y_cpu, Y_gpu) ? "OK" : "MISMATCH") << "\n";
    }

    // 3) ReLU
    {
        int N = 1 << 20;
        std::vector<float> X(N), Y_cpu(N), Y_gpu(N);
        std::generate(X.begin(), X.end(), frand);
        relu_cpu(X.data(), Y_cpu.data(), N);

        float *dX, *dY;
        CUDA_CHECK(cudaMalloc(&dX, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dY, N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dX, X.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        int TPB = 256, blocks = ceil_div(N, TPB);
        relu_kernel<<<blocks, TPB>>>(dX, dY, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(Y_gpu.data(), dY, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dX)); CUDA_CHECK(cudaFree(dY));
        std::cout << "3) ReLU: " << (arrays_close(Y_cpu, Y_gpu) ? "OK" : "MISMATCH") << "\n";
    }

    // 4) Reduction Sum
    {
        int N = 1 << 20;
        std::vector<float> X(N);
        std::generate(X.begin(), X.end(), frand);
        float ref = reduction_sum_cpu(X.data(), N);

        float *dX, *dPartials;
        int blocks = 256, TPB = 256;
        CUDA_CHECK(cudaMalloc(&dX, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dPartials, blocks * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dX, X.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        reduction_block_kernel<256><<<blocks, TPB>>>(dX, dPartials, N);
        CUDA_CHECK(cudaGetLastError());
        std::vector<float> partials(blocks);
        CUDA_CHECK(cudaMemcpy(partials.data(), dPartials, blocks * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dX)); CUDA_CHECK(cudaFree(dPartials));
        float gpu = std::accumulate(partials.begin(), partials.end(), 0.0f);
        std::cout << "4) Reduction Sum: " << (almost_equal(ref, gpu, 1e-2f) ? "OK" : "MISMATCH") << "\n";
    }

    // 5) Exclusive Prefix Sum (two-pass)
    {
        const int BLOCK = 1024;
        int N = (1 << 20) + 123; // not a multiple of block size, test edge
        std::vector<float> in(N), out_cpu(N), out_gpu(N);
        for (int i = 0; i < N; ++i) in[i] = 1.0f + 0.001f * (i % 7); // positive values
        exclusive_scan_cpu(in.data(), out_cpu.data(), N);

        float *dIn, *dOut, *dBlockSums;
        int numBlocks = ceil_div(N, BLOCK);
        CUDA_CHECK(cudaMalloc(&dIn, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dOut, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dBlockSums, numBlocks * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dIn, in.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        block_exclusive_scan_kernel<BLOCK><<<numBlocks, BLOCK>>>(dIn, dOut, dBlockSums, N);
        CUDA_CHECK(cudaGetLastError());
        std::vector<float> blockSums(numBlocks);
        CUDA_CHECK(cudaMemcpy(blockSums.data(), dBlockSums, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));
        // compute block offsets on CPU (exclusive scan of block sums)
        std::vector<float> blockOffsets(numBlocks, 0.0f);
        float acc = 0.0f;
        for (int i = 0; i < numBlocks; ++i) {
            blockOffsets[i] = acc;
            acc += blockSums[i];
        }
        float *dOffsets;
        CUDA_CHECK(cudaMalloc(&dOffsets, numBlocks * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dOffsets, blockOffsets.data(), numBlocks * sizeof(float), cudaMemcpyHostToDevice));
        add_block_offsets_kernel<<<numBlocks, BLOCK>>>(dOut, dOffsets, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(out_gpu.data(), dOut, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dIn)); CUDA_CHECK(cudaFree(dOut)); CUDA_CHECK(cudaFree(dBlockSums)); CUDA_CHECK(cudaFree(dOffsets));

        std::cout << "5) Exclusive Prefix Sum: " << (arrays_close(out_cpu, out_gpu, 1e-2f) ? "OK" : "MISMATCH") << "\n";
    }

    // 6) Matrix Multiply (naive)
    {
        int M = 128, N = 64, K = 96;
        std::vector<float> A(M*N), B(N*K), C_cpu(M*K), C_gpu(M*K);
        std::generate(A.begin(), A.end(), frand);
        std::generate(B.begin(), B.end(), frand);
        matmul_naive_cpu(A.data(), B.data(), C_cpu.data(), M, N, K);

        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, A.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB, B.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC, C_gpu.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice));
        dim3 TPB(16, 16);
        dim3 grid(ceil_div(K, TPB.x), ceil_div(M, TPB.y));
        matmul_naive_kernel<<<grid, TPB>>>(dA, dB, dC, M, N, K);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(C_gpu.data(), dC, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
        std::cout << "6) MatMul Naive: " << (arrays_close(C_cpu, C_gpu, 1e-2f) ? "OK" : "MISMATCH") << "\n";
    }

    // 7) Matrix Multiply (tiled)
    {
        int M = 128, N = 64, K = 96;
        std::vector<float> A(M*N), B(N*K), C_cpu(M*K), C_gpu(M*K);
        std::generate(A.begin(), A.end(), frand);
        std::generate(B.begin(), B.end(), frand);
        matmul_naive_cpu(A.data(), B.data(), C_cpu.data(), M, N, K);

        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, A.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB, B.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC, C_gpu.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice));
        const int TILE = 16;
        dim3 TPB(TILE, TILE);
        dim3 grid(ceil_div(K, TILE), ceil_div(M, TILE));
        matmul_tiled_kernel<TILE><<<grid, TPB>>>(dA, dB, dC, M, N, K);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(C_gpu.data(), dC, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
        std::cout << "7) MatMul Tiled: " << (arrays_close(C_cpu, C_gpu, 1e-2f) ? "OK" : "MISMATCH") << "\n";
    }

    // 8) Histogram (256 bins)
    {
        int N = 1 << 20;
        std::vector<uint8_t> data(N);
        for (int i = 0; i < N; ++i) data[i] = urand_byte();
        std::vector<unsigned int> H_cpu(256), H_gpu(256);
        histogram_cpu(data.data(), N, H_cpu.data());

        uint8_t *dData; unsigned int *dHist;
        CUDA_CHECK(cudaMalloc(&dData, N * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&dHist, 256 * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemcpy(dData, data.data(), N * sizeof(uint8_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dHist, 0, 256 * sizeof(unsigned int)));
        int TPB = 256, blocks = 256;
        histogram_kernel<<<blocks, TPB>>>(dData, N, dHist);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(H_gpu.data(), dHist, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dData)); CUDA_CHECK(cudaFree(dHist));
        bool ok = true;
        for (int i = 0; i < 256; ++i) if (H_cpu[i] != H_gpu[i]) { ok = false; break; }
        std::cout << "8) Histogram 256: " << (ok ? "OK" : "MISMATCH") << "\n";
    }

    // 9) CSR SpMV
    {
        // Build a tiny random sparse matrix (M x K) with ~3 nonzeros per row
        int M = 256, K = 128;
        std::vector<int> rowPtr(M+1, 0);
        std::vector<int> colInd;
        std::vector<float> val;
        std::uniform_int_distribution<int> choose_col(0, K-1);
        std::uniform_real_distribution<float> choose_val(-1.0f, 1.0f);
        for (int r = 0; r < M; ++r) {
            int nnz_row = 3;
            rowPtr[r] = static_cast<int>(val.size());
            for (int t = 0; t < nnz_row; ++t) {
                int c = choose_col(rng);
                colInd.push_back(c);
                val.push_back(choose_val(rng));
            }
        }
        rowPtr[M] = static_cast<int>(val.size());
        std::vector<float> x(K), y_cpu(M), y_gpu(M);
        std::generate(x.begin(), x.end(), frand);
        spmv_csr_cpu(rowPtr.data(), colInd.data(), val.data(), x.data(), y_cpu.data(), M);

        int *dRow, *dCol; float *dVal, *dX, *dY;
        CUDA_CHECK(cudaMalloc(&dRow, rowPtr.size()*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dCol, colInd.size()*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dVal, val.size()*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dX, x.size()*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dY, y_gpu.size()*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dRow, rowPtr.data(), rowPtr.size()*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dCol, colInd.data(), colInd.size()*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dVal, val.data(), val.size()*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dX, x.data(), x.size()*sizeof(float), cudaMemcpyHostToDevice));
        int TPB = 256, blocks = ceil_div(M, TPB);
        spmv_csr_kernel<<<blocks, TPB>>>(dRow, dCol, dVal, dX, dY, M);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(y_gpu.data(), dY, y_gpu.size()*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dRow)); CUDA_CHECK(cudaFree(dCol)); CUDA_CHECK(cudaFree(dVal)); CUDA_CHECK(cudaFree(dX)); CUDA_CHECK(cudaFree(dY));
        std::cout << "9) CSR SpMV: " << (arrays_close(y_cpu, y_gpu, 1e-2f) ? "OK" : "MISMATCH") << "\n";
    }

    // 10) N-body (one step)
    {
        int N = 512;
        std::vector<float2_> pos(N), vel(N);
        std::vector<float> mass(N);
        for (int i = 0; i < N; ++i) {
            pos[i].x = frand(); pos[i].y = frand();
            vel[i].x = frand()*0.01f; vel[i].y = frand()*0.01f;
            mass[i] = 0.5f + std::fabs(frand());
        }
        auto pos_cpu = pos, vel_cpu = vel;
        nbody_step_cpu(pos_cpu.data(), vel_cpu.data(), mass.data(), N, 1e-2f);

        float2_* dPos; float2_* dVel; float* dMass;
        CUDA_CHECK(cudaMalloc(&dPos, N * sizeof(float2_)));
        CUDA_CHECK(cudaMalloc(&dVel, N * sizeof(float2_)));
        CUDA_CHECK(cudaMalloc(&dMass, N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dPos, pos.data(), N * sizeof(float2_), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dVel, vel.data(), N * sizeof(float2_), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dMass, mass.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        int TPB = 256, blocks = ceil_div(N, TPB);
        nbody_step_kernel<<<blocks, TPB>>>(dPos, dVel, dMass, N, 1e-2f, 1.0f, 1e-6f);
        CUDA_CHECK(cudaGetLastError());
        std::vector<float2_> pos_gpu(N), vel_gpu(N);
        CUDA_CHECK(cudaMemcpy(pos_gpu.data(), dPos, N * sizeof(float2_), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(vel_gpu.data(), dVel, N * sizeof(float2_), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dPos)); CUDA_CHECK(cudaFree(dVel)); CUDA_CHECK(cudaFree(dMass));

        // Compare a few elements loosely (numerical differences expected)
        auto diff = [](float a, float b){ return std::fabs(a - b); };
        float max_diff = 0.0f;
        for (int i = 0; i < N; ++i) {
            max_diff = std::max(max_diff, diff(pos_gpu[i].x, pos_cpu[i].x));
            max_diff = std::max(max_diff, diff(pos_gpu[i].y, pos_cpu[i].y));
        }
        std::cout << "10) N-body (one step): " << (max_diff < 1e-2f ? "OK" : "CLOSE (expected small drift)") << " (max |Δ| ≈ " << max_diff << ")\n";
    }

    std::cout << "All tests done.\n";
    return 0;
}
