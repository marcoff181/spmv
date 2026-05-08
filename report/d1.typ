
#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [A Typesetting System to Untangle the Scientific Writing Process],
  abstract: [
    The process of scientific writing is often tangled up with the intricacies of typesetting, leading to frustration and wasted time for researchers. In this paper, we introduce Typst, a new typesetting system designed specifically for scientific writing. Typst untangles the typesetting process, allowing researchers to compose papers faster. In a series of experiments we demonstrate that Typst offers several advantages, including faster document creation, simplified syntax, and increased ease-of-use.
  ],
  authors: (
    (
      name: "Filippo Marcon",
      // department: [Co-Founder],
      organization: [University of Trento],
      // location: [Berlin, Germany],
      email: "filippo.marcon@studenti.unitn.it"
    ),
  ),
  index-terms: ("Scientific writing", "Typesetting", "Document creation", "Syntax"),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

#show raw.where(block: true): set text(0.8em )


= Introduction
Sparse Matrix Vector multiplication (SpMV) is a fundamental problem in many computing fields. It is highly parallelizable and its performance is bounded by memory bandwidth.
Formally, SpMV is defined as the multiplication between $A$, a sparse matrix of size $m  times n$ and $x$, a dense vector of size $n$. The resulting vector $y$ is of size $m$.
$ y = A x $

// questions your investigation addresses
This investigation focuses on relatively basic, non-adaptive GPU SpMV kernels, both for CSR and COO. We address these questions:
- What is the penalty of the additional storage requirements of COO?
- Is this penalty balanced by the ability to split work based on non-zeros without additional preprocessing?
- How much does memory coalescing improve the performance of the kernels?
- What are the ideal launch parameters for each of the kernels? (can we use cuda tool to have them auto)
- How does matrix size, non-zeros per row, structure affect the performance of the kernels?
- How do these kernels compare to the CuSparse library performance?
- What is the memory/cache usage of our kernels?
= Methodology
// == Formats tested
The sparse formats used in the study are CSR and COO.
COO is an obvious starting point as it's the simplest storage format for sparse matrices, and it's used in the SuiteSparse dataset(see @ciao).
COO uses three arrays to store the matrix: for the $n$-th non-zero element of the matrix `rows[n]` indicates the row where it is located, `cols[n]` indicates the column, and `values[n]` stores the actual value.
CSR still uses three arrays, with the only change being that the rows array is compressed with a prefix sum.
The compression is possible only if the non-zero elements are sorted by row index.
The CSR format takes less storage than COO, however CSR-based SpMV algorithms that split tasks by row can suffer from load imbalance on sparse matrices with irregular nnz distribution along rows@req1.


// TODO: mention attempt too use restrict to improove performance
// == CPU implementation

The CPU implementation is a simple iteration over all the non-zero elements. Its only goal is to provide a reference result to measure the error of the GPU implementations.

#figure(
  caption: [Naive CPU implementation],

```cpp
for each row i in [0, m):
    for each nonzero j in [rows[i], rows[i+1]):
        y[i] ← y[i] + vals[j] * x[cols[j]]
```
) <code:naivecpu>


// == GPU implementations
Moving to the GPU kernels, we assume that `tid = blockIdx.x * blockDim.x + threadIdx.x`.
One of the simplest ways to parallelize SpMV with COO is to use one thread per non-zero.
This algorithm, commonly called _COO_flat_, assigns one thread for each non-zero element, then uses atomic adds to write the result to $y$.
This approach offers maximum parallelism and load balancing, and a certain degree of coalescing as the COO is stored in row-major order.
On the other hand the `atomicAdd` hurts the performance a lot, and with a high enough nnz we are not able to launch all threads together.

#figure(
  caption: [COO flat kernel],
```cpp
if tid < nnz:
    atomicAdd(y[rows[tid]], vals[tid] * x[cols[tid]])
```

) <code:naivecoo>

As an improvement to the shortcomings of the flat COO kernel we propose an adaptation of the COO algorithm described by Bell et.al.@bellgarland.
Computation is divided in _chunks_, which can span multiple rows, each chunk has a warp(32 threads) assigned to it.
The threads stride over the length of the chunk together, to guarantee memory coalescing.
On each step the threads first calculate the multiplication for their assigned non-zero and then they perform a _segmented reduction_ to aggregate the results.
The aggregated value is carried along through the `carry` register until a new row is found, then one thread writes the aggregated result to memory. This method drastically reduces the conflicts caused by `atomicAdd`, while guaranteeing memory coalescing and balanced distribution of non-zeros. 
The biggest difference from the algorithm presented by Bell et.al. is the usage of the more modern `shfl` instructions for the segmented reduction instead of using shared memory. 


#figure(
  caption: [COO segmented reduction kernel],
```cpp
carry_row ← -1
carry_val ← 0

for each tile of 32 nonzeros starting at n in [chunk_start, chunk_end):

    Each lane i loads nonzero (row[n+i], val[n+i] * x[col[n+i]])
    Inactive lanes (beyond chunk_end) load row = -1, val = 0

    // --- Merge with carry from previous tile 
    Lane 0 checks its row against carry_row:
        if row == carry_row:
            val ← val + carry_val        
        else:
            atomicAdd(y[carry_row], carry_val)   

    // --- segmented prefix sum (via shuffle) 
    for offset = 1, 2, 4, 8, 16:
        left_row, left_val ← values from lane (i - offset) 
        if left_row == row:
            val ← val + left_val         

    // --- Write completed rows, save carry 
    next_row ← row of lane (i + 1)      
    last_lane ← last active lane in tile

    if lane == last_lane:
        carry_row, carry_val ← row, val  
    else if row ≠ next_row:
        atomicAdd(y[row], val)           

    carry_row, carry_val ← values held by last_lane (warp broadcast)

// --- Flush final carry 
Lane 0: atomicAdd(y[carry_row], carry_val)
```
) <code:segmentedcoo>


Moving to CSR format, we start with the _CSR scalar_ kernel, which assigns one thread per matrix row.
The thread then sequentially computes the multiplication between the entire row and the corresponding value of $x$ and stores it.
This basic approach suffers from load imbalance proportionally to the distribution of non-zeros per row of the matrix.
It also has no memory coalescing, and using only one thread per row is an underutilization of GPU potential,unless the matrix is very large and with a very small amount of non-zeros per row. 


#figure(
  caption: [CSR scalar kernel],
```cpp
if tid < num_rows:
    sum ← 0
    for each nonzero j in [rows[tid], rows[tid+1]):
        sum ← sum + vals[j] * x[cols[j]]
    y[tid] ← sum
```

) <code:csrscalar>

To solve most of the issues of the scalar kernel we implement the _CSR_vector_ kernel, which assigns a warp per matrix row. Now similarly to _COO_segmented_ we have memory coalescing thanks to the threads striding together along the row. We also have a similar reduction, which is made simpler by the fact that we are not spanning rows, so we can simply use `shfl` to aggregate the 32 results and then make thread 0 write them to memory. The problem that remains unsolved is that we are still splitting by row, which makes the performance vulnerable to matrices with high variance of non-zeros per row.


#figure(
  caption: [CSR vector kernel],
```cpp
if row < num_rows:
    sum ← 0
    for each nonzero j in [rows[row] + lane, rows[row+1]) with stride 32:
        sum ← sum + vals[j] * x[cols[j]]

    for offset = 16, 8, 4, 2, 1:
        sum ← sum + value from lane (i + offset)

    if lane == 0:
        y[row] ← sum
```

) <code:csrvector>
== Validation method
First the result $y_c$ is calculated using the algorithm in @code:naivecpu, then for each execution of a kernel the result $y_k$ is compared by dividing the l2 norm of the difference by the l2 norm of the reference. For all methods the error does not reach magnitudes over $10^(-7)$.
$ "err" = (|| y_c - y_k ||_2 )/( ||y_c||_2 ) $

== Measurement methodology
Each combination of kernel and launch parameters(block and grid size) is benchmarked by first launching the kernel `WARMUP` times without logging the results, and then by running it another `NITER` times, then computing the average of each measurement across runs.
The parameters `WARMUP,NITER` are user-defined, and were set respectively to 2 and 10 during all the experiments. 

Timings are measured through Cuda Events, and only the kernel execution time is measured.
FLOP/s are measured by dividing the number of required floating point operations ($"nnz"*2$) by the arithmetic mean of the execution time across the `NITER` runs.

== Hardware/Software environment
The experiments were run on the unitn _Baldo_ cluster on a _NVIDIA L40s 48GB_ graphic card. The program was compiled with _gcc_ version 13.3.0 and _cuda_ version 12.5.0 . 

= Dataset
== Sparse Matrices <ciao>
The dataset chosen for benchmarking the various kernels is a subset of the 14 matrices selected by S.Williams et al.@williams2009spmv  and later also used in a NVIDIA technical report@bellspmv2008, hosted on the SuiteSparse Matrix Collection@suitesparse. This small selection consists of matrices derived from real-world problems in different fields.  The matrices are intentionally varied in dimension, non-zeros per row, existence of dense block structure, and degree of non-zero concentration along the diagonal. @matrix-selection provides a summary of the dataset.

#figure(
  caption: [Summary of matrix selection],
  table(
    columns: (auto, 1fr, 1fr, 1fr),
    inset: (x: 4pt, y: 2pt),
    align: (left, right, right, right, right),
    stroke: none,
    fill: (col, row) => if row == 0 { gray.lighten(60%) } else if calc.even(row) { gray.lighten(95%) },
    
    [*Matrix*], [*Rows*], [*Cols*], [*NNZ*],
    
    [Protein], [36,417], [36,417], [4,344,765], 
    [FEM/Spheres], [83,334], [83,334], [6,010,480], 
    [FEM/Cantilever], [62,451], [62,451], [4,007,383], 
    [FEM/Harbor], [46,835], [46,835], [2,374,001],
    [FEM/Ship], [140,874], [140,874], [7,813,404],
    [Economics], [206,500], [206,500], [1,273,389], 
    [Epidemiology], [525,825], [525,825], [2,100,225], 
    [FEM/Accelerator], [121,192], [121,192], [2,624,331], 
    [Circuit], [170,998], [170,998], [958,936], 
    [Webbase], [1,000,005], [1,000,005], [3,105,536], 
  )
) <matrix-selection>

== Parsing

The `.mtx` format used by SuiteSparse stores matrices in COO, with column-major order. The parsing is done through the library `fast-matrix-market`, as the `.mtx` format allows a number of different representations to save space(standard, symmetric, skew-symmetric) meaning that writing a parser from scratch could lead to errors and/or distract from the main goal.
After parsing we sort the matrix in row-major order, and then create a new vector with the rows in CSR format.
The result is that we use row-major ordering for both COO and CSR, for consistence and ease of use when writing and comparing kernels.

== Input Vector
The `Float32` Input Vector is randomly generated with a fixed seed to guarantee reproducibility across runs. The user-defined parameter `MAX_VECTOR_VALUE` defines the upper bound to the randomly generated values.

= Results
@fig:plot1 shows the average performance of each kernel across all 10 matrices, grouped by the experimented block size. The red dotted line shows the average performance of the CuSparse library over all of the matrices. We can see that starting from 128 threads per block the performance of al kernels stabilizes, and increasing past that number does not improve, and in some cases worsens the performance.
#figure(
  caption: [Aggregate Performance related to Block Size],
image("../results/plot1_blocksize_vs_gflops.png")
) <fig:plot1>

@fig:plot2, @fig:plot3, @fig:plot4 shows the best performance with a fixed block size of 128 of the 4 kernels for each matrix. In the three plots the matrices are respectively sorted by increasing density, variance of nnz per row, and total nnz. We use line plots instead of bar plots to better visualize trends across the 10 matrices.

#figure(
  caption: [Kernel best performance for each matrix, sorted by matrix density],
image("../results/plot2_density_sorted_performance.png")
) <fig:plot2>

#figure(
  caption: [Kernel best performance for each matrix, sorted by variance of nnz per row],
image("../results/plot3_variance_sorted_performance.png")
) <fig:plot3>

#figure(
  caption: [Kernel best performance for each matrix, sorted by total nnz],
image("../results/plot4_nnz_sorted_performance.png")
) <fig:plot4>

We briefly attempted to measure cache usage by choosing a fixed block size, and profiling one kernel at a time with `ncu` but found out that the unitn _Baldo_ cluster did not allow access to NVIDIA GPU performance counters. Not having access to other GPUs we had to abandon the profiling attempt.

= Discussion

= Conclusion
_COO flat_, as expected, is the worst performing algorithm on average. It performs best with the very small matrices, dropping down quickly on larger.

// TODO: add bandwitdh
