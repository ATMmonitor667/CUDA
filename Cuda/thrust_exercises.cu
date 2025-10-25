/*
 * THRUST CUDA EXERCISES - Functional Programming Focus
 * Each exercise has a difficulty rating from 1-100
 * 1 = Easy, 100 = Very Complex
 */

// Required Thrust Headers
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/count.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/partition.h>
#include <thrust/unique.h>
#include <thrust/remove.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/merge.h>
#include <thrust/execution_policy.h>

// Iterator Headers
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/reverse_iterator.h>

// Utility Headers
#include <thrust/tuple.h>
#include <thrust/pair.h>
#include <thrust/extrema.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
using namespace std;
// Standard C++ Headers
#include <iostream>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <limits>

// CUDA Headers
#include <cuda_runtime.h>
__device__ int square(int x) {
    return x * x;
}
void vectorTransform()
{
  int n = 100;
  thrust::device_vector<int> d_vec(n);
  for (int i = 0; i < n; i++) {
      d_vec[i] = i + 1;
  }
  thrust::transform(d_vec.begin(), d_vec.end(), d_vec.begin(), square);
  for(int i = 0; i < n; i++) {
      std::cout << d_vec[i] << " ";
  }
}
int main(void) {
    vectorTransform(); // First function is now completed with any GPT assistance
    // the excercises here are constructed by GBT but completed by me to enhance my learning.
    return 0;
}
// ============================================================================
// EXERCISE 1: Vector Transformation (Difficulty: 15/100)
// ============================================================================
/*
 * Task: Use thrust::transform to square each element in a vector
 *
 * Given: A device_vector with values [1, 2, 3, 4, 5]
 * Output: Transform it to [1, 4, 9, 16, 25]
 *
 * Concepts:
 * - thrust::transform
 * - thrust::device_vector
 * - Functor/Lambda for squaring
 *
 * Hints:
 * - Create a functor that squares a number
 * - Use thrust::transform with the same input and output vector
 */


// ============================================================================
// EXERCISE 2: Predicate-Based Counting (Difficulty: 25/100)
// ============================================================================
/*
 * Task: Use thrust::count_if to count even numbers in a vector
 *
 * Given: A device_vector with values [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
 * Output: Count how many even numbers exist (should be 5)
 *
 * Concepts:
 * - thrust::count_if
 * - Predicate functors
 * - Unary predicates
 *
 * Hints:
 * - Create an is_even predicate functor
 * - Use modulo operator (%)
 */
void excercise2()
{
  thrust::device_vector<int> v;
  int n = 100;
  for(int i = 0; i < v.size(); i++) v[i] = i+1;
  int count = thrust::count_if(v.begin(), v.end(), []__device__ (int x){
    return x%2 == 0;
  })
  printf("This is an even number %d", count);
}

// ============================================================================
// EXERCISE 3: Reduction with Custom Operation (Difficulty: 35/100)
// ============================================================================
/*
 * Task: Use thrust::reduce to find the product of all elements
 *
 * Given: A device_vector with values [1, 2, 3, 4, 5]
 * Output: Calculate the product (1 * 2 * 3 * 4 * 5 = 120)
 *
 * Concepts:
 * - thrust::reduce
 * - Binary operations (thrust::multiplies)
 * - Custom initial value
 *
 * Hints:
 * - Use thrust::multiplies<int>() as the binary operation
 * - Set initial value to 1 (not 0!)
 */

 void excercise3(){
  int n = 5;
  thrust::device_vector<int> v(n);
  for(int i = 0; i < n; i++)v[i] = i+1;
  auto value = thrust::reduce(v.begin(), v.end(), 0, thrust::multiplies<int>())
  printf("The product of all the values is %d", value);
 }


// ============================================================================
// EXERCISE 4: Copy If with Complex Predicate (Difficulty: 40/100)
// ============================================================================
/*
 * Task: Use thrust::copy_if to filter numbers divisible by 3 OR greater than 50
 *
 * Given: A device_vector with values [3, 15, 23, 42, 51, 60, 72, 85, 90, 100]
 * Output: A new vector containing only values meeting the criteria
 *         [3, 15, 42, 51, 60, 72, 85, 90, 100]
 *
 * Concepts:
 * - thrust::copy_if
 * - Complex predicates with logical OR
 * - Separate output vector
 *
 * Hints:
 * - Create a predicate that checks (x % 3 == 0 || x > 50)
 * - Pre-allocate output vector or use back_inserter
 */

void excercise4()
{
  int n = 100;
  thrust::device_vector<int> v(n);
  thrust::sequence(v.begin(), v.end(),1);
  auto value = thrust::count_if(v.begin(), v.end(),
  []__device__(int x )
  {
    if(x%3 == 0 || x > 50)
    {
      return x;
    }
  }
)
 printf("This is the count of the numbers which are divisible by 3 and greater than 50 %d", value);
}
// ============================================================================
// EXERCISE 5: Transform with Multiple Inputs (Difficulty: 50/100)
// ============================================================================
/*
 * Task: Use thrust::transform with binary operation to compute element-wise
 *       weighted sum: result[i] = a[i] * 0.6 + b[i] * 0.4
 *
 * Given: Two device_vectors
 *        a = [10, 20, 30, 40, 50]
 *        b = [5, 15, 25, 35, 45]
 * Output: [8, 18, 28, 38, 48]
 *
 * Concepts:
 * - thrust::transform with binary operation
 * - Custom binary functor
 * - Working with multiple input iterators
 *
 * Hints:
 * - Create a binary functor that takes two arguments
 * - Use floating-point arithmetic carefully
 */


// ============================================================================
// EXERCISE 6: Scan (Prefix Sum) with Custom Operation (Difficulty: 55/100)
// ============================================================================
/*
 * Task: Use thrust::inclusive_scan to compute running maximum
 *
 * Given: A device_vector with values [3, 1, 4, 1, 5, 9, 2, 6]
 * Output: [3, 3, 4, 4, 5, 9, 9, 9] (running maximum at each position)
 *
 * Concepts:
 * - thrust::inclusive_scan
 * - thrust::maximum binary operation
 * - Prefix computation patterns
 *
 * Hints:
 * - Use thrust::maximum<int>() as the binary operation
 * - Understand how scan differs from reduce
 */


// ============================================================================
// EXERCISE 7: Partition and Transform Pipeline (Difficulty: 65/100)
// ============================================================================
/*
 * Task: Create a functional pipeline that:
 *       1. Partitions numbers into even/odd using thrust::partition
 *       2. Transforms even numbers by dividing by 2
 *       3. Transforms odd numbers by multiplying by 3 and adding 1
 *
 * Given: A device_vector with values [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
 * Output: Reordered vector with transformations applied to each partition
 *
 * Concepts:
 * - thrust::partition
 * - Multiple thrust::transform calls
 * - Iterator arithmetic to work with partitions
 * - Chaining operations
 *
 * Hints:
 * - thrust::partition returns iterator to partition point
 * - Use iterator ranges for partial transformations
 */


// ============================================================================
// EXERCISE 8: Zip Iterator with Tuple Operations (Difficulty: 75/100)
// ============================================================================
/*
 * Task: Use thrust::zip_iterator to compute Euclidean distances between
 *       2D points and origin, then sort points by distance
 *
 * Given: Two device_vectors representing x and y coordinates:
 *        x = [3, 1, 4, 1, 5]
 *        y = [4, 1, 2, 5, 0]
 * Output: Points sorted by distance from origin (0,0)
 *         Point(1,1), Point(4,2), Point(3,4), Point(5,0), Point(1,5)
 *
 * Concepts:
 * - thrust::zip_iterator
 * - thrust::make_zip_iterator
 * - Custom comparison functor for tuples
 * - thrust::sort with zip iterators
 * - thrust::get to access tuple elements
 *
 * Hints:
 * - Distance formula: sqrt(x^2 + y^2)
 * - Create comparison functor that works with thrust::tuple<T,T>
 */


// ============================================================================
// EXERCISE 9: Reduce by Key for Grouped Aggregation (Difficulty: 80/100)
// ============================================================================
/*
 * Task: Use thrust::reduce_by_key to compute sum of values for each category
 *
 * Given: Two device_vectors:
 *        keys =   [1, 1, 1, 2, 2, 3, 3, 3, 3, 4]
 *        values = [5, 2, 8, 3, 7, 1, 4, 2, 6, 9]
 * Output: Two vectors with reduced results:
 *         unique_keys = [1, 2, 3, 4]
 *         sums =        [15, 10, 13, 9]
 *
 * Concepts:
 * - thrust::reduce_by_key
 * - Working with key-value pairs
 * - Understanding adjacency requirements
 * - Output iterator management
 *
 * Hints:
 * - Keys must be sorted/adjacent for reduce_by_key
 * - Function returns pair of end iterators
 * - Pre-allocate output vectors appropriately
 */


// ============================================================================
// EXERCISE 10: Advanced Functional Pipeline with Transform_Iterator (Difficulty: 90/100)
// ============================================================================
/*
 * Task: Implement a lazy evaluation pipeline using thrust::transform_iterator
 *       that computes: sum of (sqrt(x) * log(x)) for all x > 1
 *       WITHOUT materializing intermediate vectors
 *
 * Given: A device_vector with values [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
 * Output: Single value representing the sum of transformed values where x > 1
 *
 * Concepts:
 * - thrust::transform_iterator (lazy evaluation)
 * - thrust::make_transform_iterator
 * - Composing multiple functors
 * - Iterator adapters
 * - Memory-efficient functional programming
 * - Conditional reduction with predicates
 *
 * Hints:
 * - Create a functor that computes sqrt(x) * log(x)
 * - Use transform_iterator to avoid allocating intermediate storage
 * - Combine with thrust::remove_if or use custom reduction
 * - Consider using thrust::counting_iterator with permutation_iterator
 * - This is a test of advanced iterator composition!
 */


// ============================================================================
// EXERCISE 11: Segmented Sort with Custom Comparator and Fancy Iterators (Difficulty: 110/100)
// ============================================================================
/*
 * Task: Implement a segmented sort where each segment is sorted by a different
 *       criterion, using only Thrust functional primitives (no raw CUDA kernels).
 *       Segments are defined by a segment_ids vector, and each segment should be
 *       sorted in alternating order (ascending for even segment IDs, descending for odd).
 *
 * Given: Three device_vectors:
 *        values =      [5, 2, 8, 1, 9, 3, 7, 4, 6, 0, 2, 5, 1, 8]
 *        segment_ids = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
 *
 * Output: Values sorted within each segment with alternating order:
 *         [2, 5, 8, 9, 7, 4, 3, 0, 2, 6, 8, 5, 2, 1]
 *         Segment 0 (even): [2,5,8] ascending
 *         Segment 1 (odd):  [9,7,4,3] descending
 *         Segment 2 (even): [0,2,6] ascending
 *         Segment 3 (odd):  [8,5,2,1] descending
 *
 * Concepts:
 * - thrust::stable_sort_by_key with composite keys
 * - thrust::zip_iterator for multi-field sorting
 * - thrust::transform_iterator for on-the-fly key generation
 * - Custom comparison functors with tuple types
 * - Segment boundaries detection
 * - Combining multiple iterator adapters
 * - Advanced functional composition without explicit loops
 *
 * Constraints:
 * - NO raw CUDA kernels allowed
 * - Must use only Thrust functional primitives
 * - Must handle segments of varying lengths
 * - Solution must work in-place or with minimal extra memory
 *
 * Hints:
 * - Consider creating a composite sort key: (segment_id, transformed_value)
 * - Transform values based on segment parity before sorting
 * - Use thrust::make_transform_iterator with a functor that accesses segment info
 * - May need thrust::scatter/gather with permutation_iterator
 * - Think about using thrust::upper_bound to find segment boundaries
 * - This requires deep understanding of iterator composition!
 */


// ============================================================================
// EXERCISE 12: Parallel Stream Compaction with Multi-Level Predicates (Difficulty: 125/100)
// ============================================================================
/*
 * Task: Implement a three-stage stream compaction where elements are filtered
 *       through multiple cascading predicates, maintaining a mapping of original
 *       indices to final positions, and computing statistics at each stage.
 *       All operations must be done in a single functional pipeline without
 *       materializing intermediate results until the final output.
 *
 * Given: A device_vector with 1000 random floats in range [0, 100]
 *
 * Output:
 *   1. Final compacted vector after three filters:
 *      - Stage 1: Keep if x > 25
 *      - Stage 2: Keep if x is NOT in range [45, 55]
 *      - Stage 3: Keep if floor(x) is prime
 *   2. Index mapping: original_index -> final_index (-1 if filtered out)
 *   3. Statistics tuple for each stage: (count_passed, sum, mean, variance)
 *
 * Concepts:
 * - Multi-stage thrust::copy_if with chaining
 * - thrust::transform_iterator for lazy predicate evaluation
 * - thrust::scatter and thrust::gather for index mapping
 * - thrust::reduce and thrust::transform_reduce for statistics
 * - Functor composition and predicate chaining
 * - Zero-copy transformations using iterator adapters
 * - Parallel prefix sum for index computation
 * - Custom tuple types for aggregated results
 *
 * Constraints:
 * - Must compute ALL outputs in a single pass where possible
 * - Minimize memory allocations (max 2 intermediate vectors)
 * - All predicates must be stateless functors
 * - Must handle empty results gracefully
 * - Index mapping must be computed in O(n) time
 *
 * Hints:
 * - Use thrust::inclusive_scan with custom operator for running statistics
 * - Create a mega-predicate that combines all three conditions
 * - Use thrust::counting_iterator for original indices
 * - Leverage thrust::transform_reduce for variance calculation
 * - Consider using thrust::tuple to return multiple values from functors
 * - May need thrust::make_zip_iterator for parallel array operations
 * - Prime checking in a device functor requires careful implementation
 */


// ============================================================================
// EXERCISE 13: Graph Algorithms - BFS Level Assignment via Functional Primitives (Difficulty: 140/100)
// ============================================================================
/*
 * Task: Implement Breadth-First Search (BFS) level assignment on a graph
 *       represented as an edge list, using ONLY Thrust functional operations.
 *       Assign each vertex its distance (level) from a source vertex.
 *       This is extremely challenging because BFS is inherently iterative,
 *       but you must express it functionally.
 *
 * Given: Graph as edge list (CSR-like format):
 *        num_vertices = 10
 *        row_offsets = [0, 2, 5, 7, 9, 12, 14, 16, 18, 19, 20]
 *        column_indices = [1, 3, 0, 2, 4, 1, 5, 3, 6, 2, 4, 7, 6, 8, 5, 9, 7, 9, 8, 6]
 *        source_vertex = 0
 *
 * Output: device_vector<int> levels of size num_vertices
 *         levels[i] = distance from source to vertex i (-1 if unreachable)
 *         Example: [0, 1, 2, 1, 2, 3, 3, 4, 4, 4]
 *
 * Concepts:
 * - Iterative functional programming (frontier-based BFS)
 * - thrust::gather for neighborhood expansion
 * - thrust::unique and thrust::remove_if for frontier management
 * - thrust::scatter_if for conditional level updates
 * - Fixed-point iteration using Thrust primitives
 * - CSR graph representation traversal
 * - Atomic-free parallel graph traversal
 * - Using thrust::transform_iterator with complex state
 *
 * Constraints:
 * - NO raw CUDA kernels or atomic operations
 * - Must use only Thrust library functions
 * - Must handle disconnected graphs correctly
 * - Should terminate in O(diameter) iterations
 * - Each iteration must be expressed as functional transformations
 *
 * Hints:
 * - Maintain current frontier (vertices at current level)
 * - For each iteration: expand frontier, update levels, generate new frontier
 * - Use thrust::lower_bound/upper_bound to find neighbor ranges
 * - thrust::scatter_if can conditionally update unvisited vertices
 * - Termination: continue while frontier is non-empty
 * - Consider using thrust::tuple to track (vertex, level) pairs
 * - This requires thinking of BFS as frontier expansion + reduction
 * - May need to use thrust::for_each with a complex functor
 */


// ============================================================================
// EXERCISE 14: Parallel Recursive Fibonacci with Memoization (Difficulty: 155/100)
// ============================================================================
/*
 * Task: Compute Fibonacci numbers for a vector of indices using a parallel
 *       dynamic programming approach with Thrust. Build the entire Fibonacci
 *       sequence up to the maximum requested index using only functional
 *       primitives, then gather the requested values. Must handle very large
 *       indices (up to 10^6) efficiently.
 *
 * Given: device_vector<int> indices = [100, 500, 1000, 50, 999, 42, 10000]
 *        (All indices where we want to compute Fibonacci numbers)
 *
 * Output: device_vector<uint64_t> fib_values with Fibonacci numbers
 *         corresponding to each index (handle overflow gracefully)
 *
 * Concepts:
 * - Bottom-up dynamic programming using Thrust
 * - Parallel iterative scan for sequence generation
 * - Custom binary operator for Fibonacci recurrence
 * - Using thrust::tuple to maintain (F(n-1), F(n)) state
 * - Memory-efficient computation using scanning
 * - thrust::gather for final value extraction
 * - Handling numeric overflow with appropriate types
 * - Generalized parallel recurrence relations
 *
 * Constraints:
 * - Must compute in O(max_index) time, not O(n * max_index)
 * - Cannot use recursion or iteration in device code
 * - Must express the entire solution as functional transformations
 * - Must handle overflow by using modular arithmetic (mod 10^18)
 * - Space complexity should be O(max_index), not O(n * max_index)
 *
 * Hints:
 * - Use thrust::inclusive_scan with a Fibonacci binary operator
 * - The binary operator should work on thrust::tuple<uint64_t, uint64_t>
 * - Operator should compute: (F(n-1), F(n)) -> (F(n), F(n+1))
 * - F(n+1) = F(n) + F(n-1), so new tuple is (second, first + second)
 * - Initial value for scan is thrust::make_tuple(0, 1)
 * - After scan, use thrust::gather to extract requested values
 * - Consider using thrust::transform_iterator to extract second element
 * - Modular arithmetic: use (a + b) % MOD to prevent overflow
 * - This is a test of expressing iterative DP as a scan operation!
 */


// ============================================================================
// EXERCISE 15: N-Body Simulation Force Calculation with Barnes-Hut Approximation (Difficulty: 175/100)
// ============================================================================
/*
 * Task: Implement a simplified Barnes-Hut algorithm for N-body force calculation
 *       using only Thrust primitives. Given N particles with positions and masses,
 *       compute the gravitational force on each particle from all others, using
 *       a quadtree approximation for distant particles. This is EXTREMELY difficult
 *       because tree construction and traversal are not naturally functional.
 *
 * Given: device_vectors for N particles (N = 10000):
 *        positions_x, positions_y (2D space)
 *        masses
 *        theta = 0.5 (Barnes-Hut opening angle criterion)
 *
 * Output: device_vectors:
 *         forces_x, forces_y (net gravitational force on each particle)
 *
 * Concepts:
 * - Spatial data structure construction using Thrust
 * - Morton codes (Z-order curve) for space partitioning
 * - Radix sort for Morton code ordering
 * - Parallel tree construction using reductions
 * - Hierarchical force approximation
 * - Multi-level thrust::reduce_by_key for tree building
 * - Complex functor composition for force calculations
 * - thrust::zip_iterator with 4+ components
 * - Conditional force calculation based on geometric criteria
 * - Nested parallelism using thrust::transform with internal reductions
 *
 * Algorithm Outline:
 * 1. Compute Morton codes for each particle based on position
 * 2. Sort particles by Morton code (spatial locality)
 * 3. Build octree/quadtree structure using segmented reductions
 * 4. Compute center of mass and bounds for each tree node
 * 5. For each particle, traverse tree and compute forces:
 *    - If node is far enough (s/d < theta), use approximation
 *    - Otherwise, recurse to children or compute direct force
 * 6. Sum all force contributions
 *
 * Constraints:
 * - NO raw CUDA kernels (this is the ultimate challenge!)
 * - Must use Barnes-Hut approximation (not naive O(n²))
 * - Tree must be built using only Thrust operations
 * - Force calculation must be parallelized over particles
 * - Must handle particle collisions (same position) gracefully
 *
 * Hints:
 * - Morton code: interleave bits of normalized x and y coordinates
 * - Use thrust::sort_by_key to sort particles by Morton code
 * - Tree levels can be identified by Morton code prefixes
 * - Use thrust::reduce_by_key at each tree level for center of mass
 * - Represent tree as multiple sorted arrays (one per property)
 * - For force calc, may need thrust::transform with a complex functor that
 *   internally uses thrust::reduce over tree nodes
 * - Opening criterion: s/d < theta where s=cell_size, d=distance
 * - Consider using thrust::permutation_iterator for tree traversal
 * - You may need to approximate tree traversal with multiple passes
 * - This is PhD-level difficulty - requires deep creativity!
 * - Real Barnes-Hut may require some compromise or multiple Thrust passes
 */
