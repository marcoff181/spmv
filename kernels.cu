#include "kernels.cuh"

__global__ void coo_flat(int nnz, int *rows, int *cols, float *vals, float *x,
                         float *y) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < nnz) {
    int row = rows[tid];
    int col = cols[tid];

    atomicAdd(&y[row], vals[tid] * x[col]);
  }
}

// Adaptation of Graham 2009 COO algo
// TODO: either remove restrict or add it everywhere
// TODO: removed first_idx logic to see if it affects algo (it does not
// make much difference, mention it in report)
__global__ void coo_segmented_reduction(int nnz, int chunk_size,
                                        const int *__restrict__ rows,
                                        const int *__restrict__ cols,
                                        const float *__restrict__ vals,
                                        const float *__restrict__ x,
                                        float *__restrict__ y) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int lane = threadIdx.x & 31; // id within the warp (0-31)
  int warp_id = tid / 32;

  int chunk_start = warp_id * chunk_size;
  int chunk_end = min(chunk_start + chunk_size, nnz);

  if (chunk_start >= chunk_end)
    return;

  // Registers to hold the "carry" from the previous 32-element loop
  int carry_row = -1;
  float carry_val = 0.0f;

  // Loop through this warp's chunk, 32 elements at a time
  for (int n = chunk_start; n < chunk_end; n += 32) {
    int idx = n + lane;
    bool active = (idx < chunk_end); // Handle boundaries cleanly

    // If out of bounds, set row to -1 and val to 0 so math isn't affected
    int row = active ? rows[idx] : -1;
    float val = active ? vals[idx] * x[cols[idx]] : 0.0f;

    // the FIRST thread checks if there is a carry and either propagates it or
    // writes it down(means that he is exactly the one in the new row)
    if (lane == 0 && n > chunk_start) {
      if (row == carry_row) {
        val += carry_val; // Row continues, add the previous sum
      } else {
        // Previous row finished. Write it out safely.
        atomicAdd(&y[carry_row], carry_val);
      }
    }

    // By the end of this step, the thread situated at the highest lane index
    // for a specific row holds the accumulated sum of all multiplications for
    // that row within the current 32-element group.
    for (int offset = 1; offset < 32; offset *= 2) {
      int left_row = __shfl_up_sync(0xffffffff, row, offset);
      float left_val = __shfl_up_sync(0xffffffff, val, offset);

      if (lane >= offset && left_row == row) {
        val += left_val;
      }
    }

    // Look 1 space to the right to see if the row changes
    int next_row = __shfl_down_sync(0xffffffff, row, 1);

    // Find the last active thread in this loop (usually 31, unless at the very
    // end of chunk)
    int last_lane = min(31, chunk_end - 1 - n);

    if (lane == last_lane) {
      // This is the end of the 32-element window. Save it as the carry.
      carry_row = row;
      carry_val = val;
    } else if (active && row != next_row) {
      // The row ended inside this 32-element window. Write it out.
      atomicAdd(&y[row], val);
    }

    // get carry from last thread(only first thread will use it)
    carry_row = __shfl_sync(0xffffffff, carry_row, last_lane);
    carry_val = __shfl_sync(0xffffffff, carry_val, last_lane);
  }

  // write what's left of carry
  if (lane == 0 && carry_row != -1) {
    atomicAdd(&y[carry_row], carry_val);
  }
}

// Issue with access to cols and vals arrays, even though all items of row are
// stored next to one another, each thread accesses one at a time
__global__ void csr_scalar(int num_rows, int *rows, int *cols, float *vals,
                           float *x, float *y) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < num_rows) {
    float sum = 0.0f;

    for (int j = rows[tid]; j < rows[tid + 1]; ++j) {
      sum += x[cols[j]] * vals[j];
    }
    y[tid] = sum;
  }
}

__global__ void csr_vector(int num_rows, int *rows, int *cols, float *vals,
                           float *x, float *y) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
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

    // threads talk to each other and group the sum
    for (int offset = 16; offset > 0; offset /= 2) {
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // one thread writes result
    if (lane == 0) {
      y[row] += sum;
    }
  }
}
