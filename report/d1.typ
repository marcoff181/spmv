
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



= Introduction
Sparse Matrix Vector multiplication (SpMV) is a fundamental problem in many computing fields. It is highly parallelizable and its performance is bounded by memory bandwidth.
== Problem Statement
SpMV is defined as the multiplication between $A$, a sparse matrix of size $m  times n$ and $x$, a dense vector of size $n$. The resulting vector $y$ is of size $m$.
$ y = A x $
== Why SpMV matters
== Questions your investigation addresses
= Methodology
== Formats tested
The sparse formats used in the study are CSR and COO.
COO is an obvious starting point as it's the simplest storage format for sparse matrices, and it's used in the SuiteSparse dataset(see @ciao).
COO uses three arrays to store the matrix: for the $n$-th non-zero element of the matrix `rows[n]` indicates the row where it is located, `cols[n]` indicates the column, and `values[n]` stores the actual value.
CSR still uses three arrays, with the only change being that the rows array is compressed with a prefix sum.
The compression is possible only if the non-zero elements are sorted by row index.
The CSR format takes less storage than COO, however CSR-based SpMV algorithms that split tasks by row can suffer from load imbalance on sparse matrices with irregular nnz distribution along rows@req1.


// TODO: mention attempt too use restrict to improove performance
== CPU implementation

The CPU implementation is a simple iteration over all the non-zero elements. Its only goal is to provide a reference result to measure the error of the GPU implementations.

```cpp
for (int i = 0; i < m; ++i) {
  for (int j = rows[i]; j < rows[i + 1]; ++j) {
    y[i] += x[cols[j]] * vals[j];
  }
}
```

== GPU implementations
One of the simplest ways to parallelize SpMV with COO is to use one thread per non-zero.
This algorithm, commonly called _COO_flat_, assigns one thread for each non-zero element, then uses atomic adds to write the result to $y$.
This approach offers maximum parallelism and load balancing, and a certain degree of coalescing as the COO is stored in row-major order.
On the other hand the `atomicAdd` hurts the performance a lot, and with a high enough nnz we are not able to launch all threads together.

```cpp
int tid = blockIdx.x * blockDim.x + threadIdx.x;

if (tid < nnz) {
  int row = rows[tid];
  int col = cols[tid];
  atomicAdd(&y[row], vals[tid] * x[col]);
}
```



== Validation method

== Measurement methodology

== Hardware/Software environment

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
