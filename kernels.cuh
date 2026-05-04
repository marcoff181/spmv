#include <cuda_runtime.h>

__global__ void coo_flat(int nnz, int *rows, int *cols, float *vals, float *x,
                         float *y);

__global__ void coo_segmented_reduction(int nnz, int chunk_size,
                                        const int *__restrict__ rows,
                                        const int *__restrict__ cols,
                                        const float *__restrict__ vals,
                                        const float *__restrict__ x,
                                        float *__restrict__ y);

__global__ void csr_scalar(int num_rows, int *rows, int *cols, float *vals,
                           float *vec, float *res);

__global__ void csr_vector(int num_rows, int *rows, int *cols, float *vals,
                           float *vec, float *res);
