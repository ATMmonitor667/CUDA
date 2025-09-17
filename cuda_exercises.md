# 🚀 CUDA PROGRAMMING EXERCISES — DETAILED EDITION

Thirty CUDA challenges with progressive difficulty, each with a thorough problem statement, fixed function signature, constraints, edge cases, and I/O examples.

## 🎯 Global Rules
Unless stated otherwise:
- ❌ **No external libraries are permitted**
- ⚙️ **Use standard CUDA Runtime APIs only** (cudaMalloc, cudaMemcpy, cudaGetLastError, etc.)
- 🔒 **The specified function signature must remain unchanged**
- 📤 **Final results must be written to the named output buffer(s)**

## 📊 Legend
- **Difficulty**: 🟢 Easy | 🟠 Medium | 🔴 Hard
- **Codeforces-style rating**: 800 (easiest) → 3500 (hardest)

---

## 1️⃣ Element-wise Vector Addition (float32)
**Difficulty**: 🟢 Easy | **Codeforces Rating**: 900

### 🎯 Scenario
You're wiring up a telemetry pipeline where two sensors report N synchronized float readings. Your task is to fuse them by summing element-wise on the GPU.

### 📝 Function signature (must remain unchanged)
```cuda
void solve(const float* A, const float* B, float* C, int N);
```

### ✅ Requirements
- Launch enough threads to cover all N elements (grid-stride or 1D grid acceptable)
- Each thread computes: `C[i] = A[i] + B[i]`
- Validate N can be large (up to 1e8); use size_t where helpful, but keep the signature
- No external libs; error-check all CUDA calls

### 📥 Input format
Two arrays A and B of identical length N (1 ≤ N ≤ 100,000,000)

### 📤 Output format
One array C of length N where `C[i] = A[i] + B[i]`

### ⚠️ Edge cases
- N = 1
- N not divisible by blockDim.x
- Very large N (memory-bound; ensure no out-of-bounds)

### 💡 Examples
**Example 1:**
```
Input:  A = [1.0, 2.0, 3.0, 4.0], B = [5.0, 6.0, 7.0, 8.0]
Output: C = [6.0, 8.0, 10.0, 12.0]
```

**Example 2:**
```
Input:  A = [1.5, 1.5, 1.5], B = [2.3, 2.3, 2.3]
Output: C = [3.8, 3.8, 3.8]
```

### ⚡ Performance note
Use coalesced access: contiguous i → contiguous threads.

---

## 2️⃣ Vector Fill with Index
**Difficulty**: 🟢 Easy | **Codeforces Rating**: 850

### 🎯 Scenario
Create a sequence generator on the GPU for downstream tests.

### 📝 Signature
```cuda
void solve(float* A, int N);
```

### 🎯 Goal
Set `A[i] = static_cast<float>(i)` for 0 ≤ i < N.

### 📊 Constraints
1 ≤ N ≤ 100,000,000

### ⚠️ Edge cases
N=1; N not multiple of block size.

### 💡 Examples
```
N=5 → A=[0,1,2,3,4]
```

### 📝 Notes
Prefer grid-stride loops for portability across SM counts.

---

## 3️⃣ Vector Add Scalar
**Difficulty**: 🟢 Easy | **Codeforces Rating**: 900

### 📝 Signature
```cuda
void solve(const float* A, float* B, float s, int N);
```

### 🎯 Goal
`B[i] = A[i] + s`

### 📊 Constraints
1 ≤ N ≤ 100,000,000

### 💡 Examples
```
A=[1,2,3], s=10 → B=[11,12,13]
```

### ⚠️ Edge cases
s=0; negative/NaN s — propagate IEEE-754 semantics.

---

## 4️⃣ Element-wise Vector Multiply
**Difficulty**: 🟢 Easy | **Codeforces Rating**: 950

### 📝 Signature
```cuda
void solve(const float* A, const float* B, float* C, int N);
```

### 🎯 Goal
`C[i] = A[i] * B[i]`

### ⚠️ Edge cases
Zeros/NaNs/Infs; ensure out-of-bounds checks.

### 💡 Example
```
A=[1,2], B=[3,4] → C=[3,8]
```

---

## 5️⃣ Reduce Max (block-level + final)
**Difficulty**: 🟢 Easy | **Codeforces Rating**: 1000

### 📝 Signature
```cuda
float solve(const float* A, int N);
```

### 🎯 Goal
Return max(A).

### ✅ Requirements
Two-phase reduction: per-block shared-memory reduce → single-block final reduce.

### ⚠️ Edge cases
N not power of two; negative-only arrays.

### 💡 Example
```
[3,7,2,9,1] → 9
```

---

## 6️⃣ Count Evens (int32)
**Difficulty**: 🟢 Easy | **Codeforces Rating**: 1000

### 📝 Signature
```cuda
int solve(const int* A, int N);
```

### 🎯 Goal
Return number of even integers.

### 🔧 Implementation
Threads write local counts, reduce via atomicAdd on a global counter or per-block shared reduce + one atomic per block.

### 💡 Example
```
[1,2,3,4,5,6] → 3
```

---

## 7️⃣ In-place Array Reverse (float)
**Difficulty**: 🟠 Medium | **Codeforces Rating**: 1200

### 📝 Signature
```cuda
void solve(float* A, int N);
```

### 🎯 Goal
Reverse A in-place.

### 🔧 Details
Thread i swaps A[i] with A[N-1-i] for i < N/2. Avoid race by guarding indices.

### ⚠️ Edge cases
N odd/even; N=1.

### 💡 Example
```
[1,2,3,4,5] → [5,4,3,2,1]
```

---

## 8️⃣ Tiled Matrix Transpose (square)
**Difficulty**: 🟠 Medium | **Codeforces Rating**: 1300

### 📝 Signature
```cuda
void solve(const float* In, float* Out, int N);
```

### 🎯 Goal
`Out[j*N + i] = In[i*N + j]`

### 📊 Constraints
N up to 8192.

### ✅ Requirements
Use shared-memory tile (e.g., 32x32) with padding to avoid bank conflicts.

### 💡 Test
```
N=3 → [[1,2,3],[4,5,6],[7,8,9]] → [[1,4,7],[2,5,8],[3,6,9]]
```

---

## 9️⃣ Dot Product with Reduction
**Difficulty**: 🟠 Medium | **Codeforces Rating**: 1300

### 📝 Signature
```cuda
float solve(const float* A, const float* B, int N);
```

### 🎯 Goal
Sum(A[i]*B[i]).

### ✅ Requirements
Per-block partial sums in shared memory; final reduction on host or small kernel.

### 💡 Example
```
A=[1,2,3], B=[4,5,6] → 32
```

---

## 🔟 8-bit Histogram (256 bins)
**Difficulty**: 🟠 Medium | **Codeforces Rating**: 1400

### 📝 Signature
```cuda
void solve(const unsigned char* A, int N, unsigned int* H256);
```

### 🎯 Goal
Count occurrences of values 0..255.

### 📊 Constraints
Avoid high-contention atomics: private per-block histograms in shared mem → merge.

### 💡 Example
```
[1,2,1,3,2,1] → H[1]=3,H[2]=2,H[3]=1
```

---

## 1️⃣1️⃣ Inclusive Prefix Sum (Hillis–Steele)
**Difficulty**: 🟠 Medium | **Codeforces Rating**: 1500

### 📝 Signature
```cuda
void solve(const int* A, int* S, int N);
```

### 🎯 Goal
`S[i] = sum_{k<=i} A[k]`

### ✅ Requirements
Implement block-scan + block-sum fixup phase. Handle non-power-of-two lengths.

### 💡 Example
```
[1,2,3,4] → [1,3,6,10]
```

---

## 1️⃣2️⃣ Matrix-Vector Multiply (row-major)
**Difficulty**: 🟠 Medium | **Codeforces Rating**: 1500

### 📝 Signature
```cuda
void solve(const float* M, const float* x, float* y, int rows, int cols);
```

### 🎯 Goal
`y[r] = sum_c M[r*cols + c]*x[c]`

### 📝 Note
Organize threads for coalesced reads of M and cache x in shared mem per block.

### 💡 Example
```
M=[[1,2,3],[4,5,6]], x=[1,1,1] → y=[6,15]
```

---

## 1️⃣3️⃣ 2D Gaussian Blur (5x5)
**Difficulty**: 🟠 Medium | **Codeforces Rating**: 1600

### 📝 Signature
```cuda
void solve(const float* Img, float* Out, int H, int W);
```

### 🎯 Goal
Convolve with fixed 5x5 kernel (provided as const array in device constant memory).

### ⚠️ Edge cases
Image borders via clamp or mirror; specify your policy in comments.

### 💡 Example
5x5 impulse → blurred spot.

---

## 1️⃣4️⃣ Bitonic Sort (power-of-two)
**Difficulty**: 🔴 Hard | **Codeforces Rating**: 1700

### 📝 Signature
```cuda
void solve(float* A, int N);
```

### 🎯 Goal
Sort ascending using bitonic network stages.

### ✅ Requirements
Correct synchronization between stages; handle N power-of-two only; validate with random tests.

### 💡 Example
```
[3,7,1,9] → [1,3,7,9]
```

---

## 1️⃣5️⃣ Tiled GEMM (C = A*B)
**Difficulty**: 🔴 Hard | **Codeforces Rating**: 1800

### 📝 Signature
```cuda
void solve(const float* A, const float* B, float* C, int M, int K, int N);
```

### 🎯 Goal
Compute MxK times KxN using shared-memory tiles (e.g., 32x32), loop over K.

### ⚠️ Edge cases
Non-multiples of tile size; guard loads/stores; accumulate in registers.

### 🧪 Test
A=I, B=random → C=B.

---

## 1️⃣6️⃣ Parallel Merge Sort (chunked)
**Difficulty**: 🔴 Hard | **Codeforces Rating**: 1900

### 📝 Signature
```cuda
void solve(float* A, int N);
```

### 📋 Plan
Sort small tiles with bitonic (or thrust-like but you must implement), then iteratively merge runs with parallel merge kernels.

### ⚠️ Edge cases
Stable vs unstable; ensure global memory bounds.

---

## 1️⃣7️⃣ SpMV (CSR)
**Difficulty**: 🔴 Hard | **Codeforces Rating**: 2000

### 📝 Signature
```cuda
void solve(const int* rowPtr, const int* colInd, const float* val, const float* x, float* y, int rows);
```

### 🎯 Goal
`y[r] = sum_{i=rowPtr[r]}^{rowPtr[r+1)-1} val[i]*x[colInd[i]]`

### 📝 Notes
Balance long/short rows; use warp-level reductions for long rows.

---

## 1️⃣8️⃣ 1D FFT (Cooley–Tukey)
**Difficulty**: 🔴 Hard | **Codeforces Rating**: 2100

### 📝 Signature
```cuda
void solve(const float2* in, float2* out, int N);
```

### 🎯 Goal
In-place or out-of-place radix-2 FFT with bit-reversal and butterfly stages.

### ⚠️ Edge cases
N power-of-two; numerical stability; twiddle factors in constant memory.

---

## 1️⃣9️⃣ Graph BFS (frontier-based)
**Difficulty**: 🔴 Hard | **Codeforces Rating**: 2200

### 📝 Signature
```cuda
void solve(const int* rowPtr, const int* colInd, int* dist, int V, int src);
```

### 🎯 Goal
Level-synchronous BFS; maintain frontier and next-frontier bitmaps/queues.

### ⚠️ Edge cases
Avoid warp divergence on high-degree vertices; use atomics for distance set-once.

---

## 2️⃣0️⃣ N-Body Step (gravitation)
**Difficulty**: 🔴 Hard | **Codeforces Rating**: 2300

### 📝 Signature
```cuda
void solve(const float4* posMass, float4* vel, float dt, int N);
```

### 🎯 Goal
Compute acceleration from all pairs (O(N^2)) and update vel/pos for one time step.

### ⚡ Optimization
Tile particles into shared memory; softening term to avoid singularities.

---

## 2️⃣1️⃣ Parallel Radix Sort (32-bit keys)
**Difficulty**: 🔴 Hard | **Codeforces Rating**: 2400

### 📝 Signature
```cuda
void solve(const unsigned int* in, unsigned int* out, int N);
```

### 📋 Plan
LSD radix with 4-bit or 8-bit passes; per-block histograms → prefix scan → scatter.

### ⚠️ Edge cases
Stable scatter required between passes.

---

## 2️⃣2️⃣ Rectangular Matrix Transpose (bank-safe)
**Difficulty**: 🔴 Hard | **Codeforces Rating**: 2000

### 📝 Signature
```cuda
void solve(const float* In, float* Out, int H, int W);
```

### 🎯 Goal
Transpose HxW to WxH using padded shared tiles to prevent bank conflicts; guard edges.

---

## 2️⃣3️⃣ 2D Convex Hull (parallel)
**Difficulty**: 🔴 Hard | **Codeforces Rating**: 2500

### 📝 Signature
```cuda
void solve(const float2* points, int N, int* hullIdx, int* hullCount);
```

### 📋 Plan
Filter extreme points, parallel sort by angle (you implement), parallel hull build via monotone chain with conflict resolution.

### 📤 Output
hullIdx array of indices in CCW order; hullCount number of points.

---

## 2️⃣4️⃣ Multi-GPU GEMM (split-K or split-N)
**Difficulty**: 🔴 Hard | **Codeforces Rating**: 2600

### 📝 Signature
```cuda
void solve(const float* A, const float* B, float* C, int M, int K, int N, int deviceCount);
```

### 🎯 Goal
Partition workload across multiple GPUs; aggregate partial results; ensure reproducibility across runs.

### 📝 Notes
Use peer access if available; otherwise host-mediated copies. Error handling per device.

---

## 2️⃣5️⃣ LCS via Wavefront DP
**Difficulty**: 🔴 Hard | **Codeforces Rating**: 2400

### 📝 Signature
```cuda
int solve(const char* s, int n, const char* t, int m, int* dpOut);
```

### 🎯 Goal
Fill DP anti-diagonals in parallel; dpOut stores final DP table or last row to reconstruct length.

### ⚠️ Edge cases
Memory footprint; optionally store only two diagonals.

---

## 2️⃣6️⃣ K-Means Clustering (Lloyd's)
**Difficulty**: 🔴 Hard | **Codeforces Rating**: 2700

### 📝 Signature
```cuda
void solve(const float* X, float* centroids, int* assign, int N, int D, int K, int iters);
```

### 🎯 Goal
Alternate assignment and centroid update; reductions per cluster per dimension; handle empty clusters.

### ⚡ Performance
Use shared reductions and warp shuffles; memory coalescing for X (SoA preferred).

---

## 2️⃣7️⃣ GPU Path Tracer (spheres + materials)
**Difficulty**: 🔴 Hard | **Codeforces Rating**: 2900

### 🎯 Scenario
Build a physically-based rendering engine that traces light paths through a 3D scene containing spheres with different material properties. Each pixel launches multiple rays to sample the lighting equation and produce photorealistic images.

### 📝 Function signature (must remain unchanged)
```cuda
void solve(uchar4* framebuffer, int W, int H, Scene scene, int spp);
```

Where Scene is defined as:
```cuda
struct Scene {
    float4* spheres;     // (x,y,z,radius) for each sphere
    float4* materials;   // (r,g,b,type) where type: 0=diffuse, 1=mirror, 2=glass
    int numSpheres;
    float4 camera;       // (x,y,z,fov)
    float4 lightPos;     // Point light position and intensity
};
```

### ✅ Requirements
- Implement Monte Carlo path tracing with Russian roulette termination
- Support three material types: Lambertian diffuse, perfect mirror reflection, and refractive glass
- Use cosine-weighted hemisphere sampling for diffuse surfaces
- Implement Fresnel equations for glass materials (Schlick approximation acceptable)
- Generate random numbers using a custom XorShift implementation (no cuRAND)
- Apply gamma correction (γ=2.2) to final pixel values
- Each pixel launches 'spp' (samples per pixel) rays and averages the results
- Handle ray-sphere intersections with proper normal calculations
- Implement shadow rays to test light visibility

### 📥 Input format
- framebuffer: RGBA output buffer (W×H pixels)
- W, H: Image dimensions (typically 512×512 to 1024×1024)
- scene: Contains all geometry, materials, camera, and lighting
- spp: Samples per pixel (16-256 for quality vs speed trade-off)

### 📤 Output format
framebuffer filled with gamma-corrected RGBA values (alpha=255)

### 🔧 Implementation steps
1. Generate primary rays from camera through each pixel with anti-aliasing jitter
2. Trace rays through scene, finding closest sphere intersection
3. At intersection, evaluate material BRDF and spawn secondary rays
4. Accumulate radiance along path until Russian roulette termination
5. Average all samples for each pixel and apply gamma correction

### ⚠️ Edge cases
- Rays missing all geometry (return background color)
- Ray-sphere intersection edge cases (grazing angles, inside sphere)
- Numerical precision issues with very small/large scene scales
- Material parameters outside valid ranges

### 💡 Examples
- Single white sphere with point light → soft shadows and diffuse shading
- Mirror sphere → reflections of environment
- Glass sphere → caustics and refraction effects

### ⚡ Performance optimizations
- Use shared memory for sphere data accessed by neighboring threads
- Implement early ray termination for low-contribution paths
- Use warp-level reductions for pixel sample averaging
- Store RNG state efficiently to avoid register pressure

---

## 2️⃣8️⃣ Parallel Aho–Corasick (multiple patterns)
**Difficulty**: 🔴 Hard | **Codeforces Rating**: 3000

### 🎯 Scenario
Implement a high-performance multi-pattern string matching algorithm for intrusion detection systems. Given a large text stream and hundreds of virus signatures, find all occurrences of any pattern using the Aho-Corasick automaton parallelized across GPU threads.

### 📝 Function signature (must remain unchanged)
```cuda
void solve(const char* text, int textLen, const char* patterns, const int* patternOffsets, int numPatterns, int* matches);
```

Where:
- text: Input text to search through
- textLen: Length of input text
- patterns: Concatenated pattern strings
- patternOffsets: Starting position of each pattern in the patterns array
- numPatterns: Number of patterns to search for
- matches: Output array of size textLen, matches[i] = pattern_id if pattern ends at position i, -1 otherwise

### ✅ Requirements
- Build Aho-Corasick automaton on host (failure links and output links)
- Partition text into overlapping chunks for parallel processing
- Each thread processes a chunk while maintaining automaton state
- Handle pattern matches that span chunk boundaries
- Implement efficient state transition using lookup tables
- Support patterns of varying lengths (1 to 1000 characters)
- Use atomic operations to record matches in global array
- Optimize memory access patterns for GPU architecture

### 🔧 Implementation approach
1. **Host preprocessing:**
   - Build trie from all patterns
   - Compute failure function (KMP-style links)
   - Compute output function (which patterns end at each state)
   - Transfer automaton tables to GPU constant/global memory

2. **GPU parallel search:**
   - Divide text into chunks with overlap equal to maximum pattern length
   - Each thread block processes one chunk
   - Threads within block process consecutive characters
   - Maintain automaton state and transition according to current character
   - Record pattern matches with their positions

3. **Post-processing:**
   - Merge results from overlapping regions
   - Remove duplicate matches at chunk boundaries

### 📊 Input constraints
- 1 ≤ textLen ≤ 100,000,000
- 1 ≤ numPatterns ≤ 10,000
- Pattern lengths: 1 to 1000 characters
- Alphabet size: ASCII (256 characters)

### ⚠️ Edge cases
- Overlapping patterns (report all matches)
- Patterns that are substrings of other patterns
- Very short patterns (single character)
- Text shorter than longest pattern
- Empty patterns (handle gracefully or reject)

### 💡 Examples
```
Text: "ababcababa", Patterns: ["ab", "ba", "abc"]
Expected matches: ab at positions 0,2,5,7; ba at positions 1,8; abc at position 2
```

### ⚡ Performance considerations
- Use texture memory for automaton transition tables if beneficial
- Minimize branch divergence in state transitions
- Coalesce memory accesses when reading text chunks
- Use shared memory for frequently accessed automaton states

---

## 2️⃣9️⃣ Molecular Dynamics (Lennard–Jones + neighbor lists)
**Difficulty**: 🔴 Hard | **Codeforces Rating**: 3200

### 🎯 Scenario
Simulate the motion of atoms in a liquid or gas using classical mechanics. Implement an N-body molecular dynamics simulation with Lennard-Jones potential, spatial decomposition via cell lists, and Verlet integration. This is the foundation for materials science simulations studying phase transitions, diffusion, and thermodynamic properties.

### 📝 Function signature (must remain unchanged)
```cuda
void solve(float4* particles, float4* forces, int N, float dt, float boxSize, float cutoff, int steps);
```

Where:
- particles: Array of (x,y,z,mass) for each particle
- forces: Output array of (fx,fy,fz,potential_energy) for each particle
- N: Number of particles (typically 1000 to 1,000,000)
- dt: Time step (typically 0.001 to 0.01 in reduced units)
- boxSize: Side length of cubic simulation box with periodic boundaries
- cutoff: Interaction cutoff distance (typically 2.5σ where σ is LJ parameter)
- steps: Number of integration steps to perform

### ✅ Requirements
- Implement Lennard-Jones potential: V(r) = 4ε[(σ/r)¹² - (σ/r)⁶]
- Use spatial decomposition with cell lists for O(N) force calculation
- Apply periodic boundary conditions using minimum image convention
- Integrate equations of motion using velocity-Verlet algorithm
- Maintain neighbor lists and rebuild when particles move significantly
- Calculate system properties: kinetic energy, potential energy, temperature
- Use reduced units: σ=1, ε=1, mass=1

### 🔧 Implementation steps
1. **Cell list construction:**
   - Divide simulation box into cubic cells of size ≥ cutoff
   - Assign each particle to appropriate cell
   - Build neighbor lists including adjacent cells (27 in 3D)

2. **Force calculation:**
   - For each particle, iterate through its neighbor list
   - Calculate distance with minimum image convention
   - Apply Lennard-Jones force if distance < cutoff
   - Accumulate forces on both particles (Newton's third law)

3. **Integration (velocity-Verlet):**
   - Update positions: x(t+dt) = x(t) + v(t)dt + 0.5a(t)dt²
   - Calculate new forces
   - Update velocities: v(t+dt) = v(t) + 0.5[a(t) + a(t+dt)]dt

4. **Boundary conditions:**
   - Wrap particle positions into simulation box
   - Calculate minimum image distances for force computation

### 📥 Input format
- Initial particle positions distributed randomly in box
- Initial velocities from Maxwell-Boltzmann distribution
- Lennard-Jones parameters σ=1.0, ε=1.0 (reduced units)

### 📤 Output format
Updated particle positions and velocities after 'steps' iterations
Forces and potential energy per particle

### ⚠️ Edge cases
- Particles very close together (r → 0, force → ∞)
- Particles outside simulation box (wrapping)
- Empty cells in spatial decomposition
- Very large time steps causing instability

### ⚡ Performance optimizations
- Use shared memory for cell lists accessed by thread block
- Implement force calculation with Newton's third law to halve work
- Use texture memory for particle positions if beneficial
- Avoid atomic operations by using thread-private force arrays
- Optimize memory layout for coalesced access

### 🧪 Validation tests
- Energy conservation in microcanonical ensemble
- Radial distribution function compared to theoretical values
- Diffusion coefficient measurement
- Phase transition behavior (gas → liquid)

### 🔬 Physical parameters (reduced units)
- Temperature: kT/ε ≈ 0.5-2.0
- Density: ρσ³ ≈ 0.1-1.0
- Time step: dt√(m/ε)σ⁻¹ ≈ 0.001-0.01

---

## 3️⃣0️⃣ Custom CNN Convolution Layer (forward + backward)
**Difficulty**: 🔴 Hard | **Codeforces Rating**: 3400

### 🎯 Scenario
Implement a complete 2D convolution layer for deep neural networks from scratch, including both forward pass and backward propagation for training. This kernel forms the computational backbone of modern computer vision models like ResNet and EfficientNet, requiring careful optimization of memory hierarchy and numerical stability.

### 📝 Function signatures (must remain unchanged)
```cuda
void conv2d_forward(const float* input, const float* weights, const float* bias, float* output,
                   int batch, int in_channels, int in_height, int in_width,
                   int out_channels, int kernel_height, int kernel_width,
                   int stride_h, int stride_w, int pad_h, int pad_w);

void conv2d_backward(const float* input, const float* weights, const float* grad_output,
                    float* grad_input, float* grad_weights, float* grad_bias,
                    int batch, int in_channels, int in_height, int in_width,
                    int out_channels, int kernel_height, int kernel_width,
                    int stride_h, int stride_w, int pad_h, int pad_w);
```

### ✅ Requirements
**Forward pass:**
- Implement 2D convolution: output[n,k,h,w] = Σ(input[n,c,h',w'] * weights[k,c,kh,kw]) + bias[k]
- Support arbitrary batch sizes, channels, and spatial dimensions
- Handle padding (zero-padding) and striding correctly
- Use tiled computation with shared memory to optimize memory bandwidth
- Ensure numerical stability for large feature maps

**Backward pass:**
- Compute gradients w.r.t. input: ∂L/∂input using transposed convolution
- Compute gradients w.r.t. weights: ∂L/∂weights using input-gradient outer products
- Compute gradients w.r.t. bias: ∂L/∂bias by summing gradients across spatial dimensions
- Handle the same padding and striding as forward pass
- Maintain numerical accuracy for gradient computation

### 🔧 Implementation strategy
1. **Forward pass optimization:**
   - Use im2col transformation or direct convolution with shared memory
   - Tile input and weight data to maximize reuse
   - Implement efficient boundary handling for padding
   - Use register blocking for accumulation

2. **Backward pass implementation:**
   - Gradient w.r.t. input: convolve grad_output with flipped weights
   - Gradient w.r.t. weights: convolve input with grad_output
   - Gradient w.r.t. bias: reduce grad_output over spatial and batch dimensions
   - Use atomic operations carefully to avoid race conditions

### 🗂️ Memory layout assumptions
- All tensors in NCHW format (batch, channels, height, width)
- Weights in KCHW format (out_channels, in_channels, kernel_height, kernel_width)
- Contiguous memory layout for optimal coalescing

### 📊 Input constraints
- 1 ≤ batch ≤ 256
- 1 ≤ in_channels, out_channels ≤ 2048
- 1 ≤ in_height, in_width ≤ 1024
- 1 ≤ kernel_height, kernel_width ≤ 11 (typically 1, 3, 5, 7)
- 1 ≤ stride_h, stride_w ≤ 4
- 0 ≤ pad_h ≤ kernel_height/2, 0 ≤ pad_w ≤ kernel_width/2

### ⚠️ Edge cases
- 1×1 convolutions (pointwise)
- Large kernel sizes with small feature maps
- Stride larger than kernel size
- Asymmetric padding and kernels
- Very deep networks with many channels

### 💡 Examples
**Forward pass:**
```
Input: 1×1×4×4, Weights: 1×1×3×3, Stride: 1, Padding: 1
Expected output: 1×1×4×4 (same size due to padding)
```

**Backward pass:**
```
Given grad_output: 1×1×4×4, should produce grad_input: 1×1×4×4, grad_weights: 1×1×3×3
```

### ⚡ Performance optimizations
- Use shared memory tiles to minimize global memory traffic
- Implement register tiling for accumulation variables
- Optimize thread block dimensions for occupancy
- Use half-precision (FP16) if supported and beneficial
- Consider using Tensor Cores on newer architectures

### 🧪 Validation approach
- Compare against reference CPU implementation
- Use finite difference method to verify gradients: (f(x+ε) - f(x-ε))/(2ε)
- Test gradient consistency: forward-backward-forward should be consistent
- Verify against small examples computed by hand

### 🧮 Mathematical foundations
- Understand convolution as cross-correlation in deep learning context
- Implement proper gradient flow according to chain rule
- Handle broadcasting semantics for bias addition
- Ensure gradient shapes match parameter shapes exactly

---

## 📈 Difficulty Progression Notes

### 🟢 **Exercises 1-6: Foundation (Easy, 800-1000)**
Basic CUDA concepts and simple parallel operations
- Thread indexing and memory management
- Simple kernels with independent threads
- Basic reductions and atomic operations

### 🟠 **Exercises 7-13: Building Skills (Medium, 1100-1600)**
Intermediate algorithms with memory optimization
- Shared memory usage and bank conflict avoidance
- Matrix operations and 2D indexing
- Convolution and filtering operations

### 🔴 **Exercises 14-21: Advanced Algorithms (Hard, 1700-2400)**
Complex parallel algorithms and data structures
- Sorting networks and parallel merge operations
- Advanced linear algebra (GEMM, sparse matrices)
- Spectral methods (FFT) and graph algorithms

### 🔴 **Exercises 22-26: Expert Systems (Hard, 2000-2700)**
Sophisticated algorithms requiring deep GPU knowledge
- Multi-GPU coordination and communication
- Dynamic programming with dependency handling
- Machine learning algorithm implementation

### 🔴 **Exercises 27-30: Cutting Edge (Hard, 2900-3400)**
Production-level implementations at expert difficulty
- Real-time graphics and physics simulation
- Advanced string processing and pattern matching
- Deep learning kernel optimization
- Scientific computing with complex physics

---

## 🎯 Testing Guidance (all exercises)
- Always check `cudaGetLastError()` after kernel launches
- Validate outputs on small, crafted inputs, then scale up
- Prefer deterministic reductions where possible (pairwise summation) for test stability
- Use profiling tools (nvprof, Nsight) to validate performance improvements
- Test edge cases thoroughly, especially boundary conditions and empty inputs