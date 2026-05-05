
#import "@preview/charged-ieee:0.1.4": ieee
#import "@preview/lilaq:0.6.0" as lq

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
for (int i = 0; i < m; ++i) {
  for (int j = rows[i]; j < rows[i + 1]; ++j) {
    y[i] += x[cols[j]] * vals[j];
  }
}
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
if (tid < nnz) {
  int row = rows[tid];
  int col = cols[tid];
  atomicAdd(&y[row], vals[tid] * x[col]);
}
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
int lane = threadIdx.x & 31; 
int warp_id = tid / 32;

int chunk_start = warp_id * chunk_size;
int chunk_end = min(chunk_start + chunk_size, nnz);

if (chunk_start >= chunk_end)
  return;

int carry_row = -1;
float carry_val = 0.0f;

for (int n = chunk_start; n < chunk_end; n += 32) {
  int idx = n + lane;
  bool active = (idx < chunk_end);

  int row = active ? rows[idx] : -1;
  float val = active ? vals[idx] * x[cols[idx]] : 0.0f;

  if (lane == 0 && n > chunk_start) {
    if (row == carry_row) {
      val += carry_val; 
    } else {
      atomicAdd(&y[carry_row], carry_val);
    }
  }

  for (int offset = 1; offset < 32; offset *= 2) {
    int left_row = __shfl_up_sync(0xffffffff, row, offset);
    float left_val = __shfl_up_sync(0xffffffff, val, offset);

    if (lane >= offset && left_row == row) {
      val += left_val;
    }
  }

  int next_row = __shfl_down_sync(0xffffffff, row, 1);

  int last_lane = min(31, chunk_end - 1 - n);

  if (lane == last_lane) {
    carry_row = row;
    carry_val = val;
  } else if (active && row != next_row) {
    atomicAdd(&y[row], val);
  }

  carry_row = __shfl_sync(0xffffffff, carry_row, last_lane);
  carry_val = __shfl_sync(0xffffffff, carry_val, last_lane);
}

if (lane == 0 && carry_row != -1) {
  atomicAdd(&y[carry_row], carry_val);
}
```

) <code:segmentedcoo>


Moving to CSR format, we start with the _CSR scalar_ kernel, which assigns one thread per matrix row.
The thread then sequentially computes the multiplication between the entire row and the corresponding value of $x$ and stores it.
This basic approach suffers from load imbalance proportionally to the distribution of non-zeros per row of the matrix.
It also has no memory coalescing, and using only one thread per row is an underutilization of GPU potential,unless the matrix is very large and with a very small amount of non-zeros per row. 


#figure(
  caption: [CSR scalar kernel],
```cpp
if (tid < num_rows) {
  float sum = 0.0f;

  for (int j = rows[tid]; j < rows[tid + 1]; ++j) {
    sum += x[cols[j]] * vals[j];
  }
  y[tid] = sum;
}```

) <code:csrscalar>

To solve most of the issues of the scalar kernel we implement the _CSR_vector_ kernel, which assigns a warp per matrix row. Now similarly to _COO_segmented_ we have memory coalescing thanks to the threads striding together along the row. We also have a similar reduction, which is made simpler by the fact that we are not spanning rows, so we can simply use `shfl` to aggregate the 32 results and then make thread 0 write them to memory. The problem that remains unsolved is that we are still splitting by row, which makes the performance vulnerable to matrices with high variance of non-zeros per row.


#figure(
  caption: [CSR vector kernel],
```cpp
int warp_id = tid / 32;
int lane = tid % 32;
int row = warp_id;

if (row < num_rows) {
  int row_start = rows[row];
  int row_end = rows[row + 1];
  float sum = 0.0;

  for (int j = row_start + lane; j < row_end; j += 32) {
    sum += x[cols[j]] * vals[j];
  }

  for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }

  if (lane == 0) {
    y[row] += sum;
  }
}
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

= Discussion

= Conclusion

// == Paper overview
// In this paper we introduce Typst, a new typesetting system designed to streamline the scientific writing process and provide researchers with a fast, efficient, and easy-to-use alternative to existing systems. Our goal is to shake up the status quo and offer researchers a better way to approach scientific writing.
//
// By leveraging advanced algorithms and a user-friendly interface, Typst offers several advantages over existing typesetting systems, including faster document creation, simplified syntax, and increased ease-of-use.
//
// To demonstrate the potential of Typst, we conducted a series of experiments comparing it to other popular typesetting systems, including LaTeX. Our findings suggest that Typst offers several benefits for scientific writing, particularly for novice users who may struggle with the complexities of LaTeX. Additionally, we demonstrate that Typst offers advanced features for experienced users, allowing for greater customization and flexibility in document creation.
//
// Overall, we believe that Typst represents a significant step forward in the field of scientific writing and typesetting, providing researchers with a valuable tool to streamline their workflow and focus on what really matters: their research. In the following sections, we will introduce Typst in more detail and provide evidence for its superiority over other typesetting systems in a variety of scenarios.
//
// = Methods <sec:methods>
// #lorem(45)
//
// $ a + b = gamma $ <eq:gamma>
//
// #lorem(80)
//
// #figure(
//   placement: none,
//   circle(radius: 15pt),
//   caption: [A circle representing the Sun.]
// ) <fig:sun>
//
// In @fig:sun you can see a common representation of the Sun, which is a star that is located at the center of the solar system.
//
// #lorem(120)
//
// #figure(
//   caption: [The Planets of the Solar System and Their Average Distance from the Sun],
//   placement: top,
//   table(
//     // Table styling is not mandated by the IEEE. Feel free to adjust these
//     // settings and potentially move them into a set rule.
//     columns: (6em, auto),
//     align: (left, right),
//     inset: (x: 8pt, y: 4pt),
//     stroke: (x, y) => if y <= 1 { (top: 0.5pt) },
//     fill: (x, y) => if y > 0 and calc.rem(y, 2) == 0  { rgb("#efefef") },
//
//     table.header[Planet][Distance (million km)],
//     [Mercury], [57.9],
//     [Venus], [108.2],
//     [Earth], [149.6],
//     [Mars], [227.9],
//     [Jupiter], [778.6],
//     [Saturn], [1,433.5],
//     [Uranus], [2,872.5],
//     [Neptune], [4,495.1],
//   )
// ) <tab:planets>
//
// In @tab:planets, you see the planets of the solar system and their average distance from the Sun.
// The distances were calculated with @eq:gamma that we presented in @sec:methods.
//
// #lorem(240)
//
// #lorem(240)
// lorem(240)
