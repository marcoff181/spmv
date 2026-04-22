#include "matrix_loader.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <math.h>
#include <numeric>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <vector>

#define TIMER_DEF struct timeval temp_1, temp_2

#define TIMER_START gettimeofday(&temp_1, (struct timezone *)0)

#define TIMER_STOP gettimeofday(&temp_2, (struct timezone *)0)

#define TIMER_ELAPSED                                                          \
  ((temp_2.tv_sec - temp_1.tv_sec) +                                           \
   (temp_2.tv_usec - temp_1.tv_usec) / 1000000.0)

#define STR(s) #s
#define XSTR(s) STR(s)

const std::string MATRIX_FOLDER = "matrices";
const float MAX_VECTOR_VALUE = 100.0;
// error is defined as percentage, 0.01 = 1% error
const double ERROR_THRESHOLD = 0.01;

__global__ void basic_spmv_kernel(int num_rows, int num_cols, int *rows,
                                  int *cols, float *vals, float *vec,
                                  float *res) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // int th_tot = gridDim.x * blockDim.x;

  if (tid >= num_rows) {
    return;
  }

  for (int j = rows[tid]; j < rows[tid + 1]; ++j) {
    res[tid] += vec[cols[j]] * vals[j];
  }
}

void cpu_spmv(int num_rows, int num_cols, int *rows, int *cols, float *vals,
              float *vec, float *res) {
  for (int i = 0; i < num_rows; ++i) {
    for (int j = rows[i]; j < rows[i + 1]; ++j) {
      res[i] += vec[cols[j]] * vals[j];
    }
  }
}

double l2_error(int len, float *v1, float *v2) {
  // L2 norm of reference
  double ref_sum_sq = std::inner_product(v2, v2 + len, v2, 0.0);
  double ref_norm = std::sqrt(ref_sum_sq);

  // L2 norm of difference
  double diff_sum_sq = 0.0;
  for (size_t i = 0; i < len; ++i) {
    double diff = v1[i] - v2[i];
    diff_sum_sq += diff * diff;
  }
  double diff_norm = std::sqrt(diff_sum_sq);

  return diff_norm / ref_norm;
}

namespace fs = std::filesystem;

int main() {
  // TODO: decide if to change seed for sequential gens
  srand(0);

  if (!fs::exists(MATRIX_FOLDER) || !fs::is_directory(MATRIX_FOLDER)) {
    std::cerr << "Error: Directory '" << MATRIX_FOLDER << "' does not exist.\n";
    return 1;
  }

  for (const auto &entry : fs::directory_iterator(MATRIX_FOLDER)) {
    if (entry.path().extension() != ".mtx")
      continue;

    std::string filename = entry.path().string();

    try {

      CSR_Matrix mat;
      load_mtx_file(filename, mat);

      std::cout << "Successfully read matrix from: " << filename << "\n";
      std::cout << "Dimensions: " << mat.num_rows << " x " << mat.num_cols
                << "\n";
      std::cout << "Non-zeros:  " << mat.vals.size() << "\n";

      float *vec = (float *)malloc(sizeof(float) * mat.num_cols);
      for (int i = 0; i < mat.num_cols; i++) {
        vec[i] = static_cast<float>(rand()) /
                 (static_cast<float>(RAND_MAX / MAX_VECTOR_VALUE));
      };

      float *res_cpu = (float *)malloc(sizeof(float) * mat.num_rows);
      for (int i = 0; i < mat.num_rows; i++) {
        res_cpu[i] = 0;
      };
      // TODO: maybe use pointers when reading it from the beginning
      cpu_spmv(mat.num_rows, mat.num_cols, mat.rows.data(), mat.cols.data(),
               mat.vals.data(), vec, res_cpu);

      // ====== GPU memory allocation
      float *gpu_vals, *gpu_vec, *gpu_res, gputime_event;
      int *gpu_rows, *gpu_cols;
      cudaMalloc(&gpu_rows, mat.rows.size() * sizeof(int));
      cudaMalloc(&gpu_cols, mat.cols.size() * sizeof(int));
      cudaMalloc(&gpu_vals, mat.vals.size() * sizeof(float));
      cudaMalloc(&gpu_vec, mat.num_cols * sizeof(float));
      cudaMalloc(&gpu_res, mat.num_rows * sizeof(float));

      cudaMemcpy(gpu_rows, mat.rows.data(), mat.rows.size() * sizeof(int),
                 cudaMemcpyHostToDevice);
      cudaMemcpy(gpu_cols, mat.cols.data(), mat.cols.size() * sizeof(int),
                 cudaMemcpyHostToDevice);
      cudaMemcpy(gpu_vals, mat.vals.data(), mat.vals.size() * sizeof(float),
                 cudaMemcpyHostToDevice);
      cudaMemcpy(gpu_vec, vec, mat.num_cols * sizeof(float),
                 cudaMemcpyHostToDevice);
      cudaMemset(gpu_res, 0, mat.num_rows * sizeof(float));

      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);

      cudaEventRecord(start);

      // TODO: test for tweaks to this
      int blockSize = 256;
      int numBlocks = (mat.num_rows + blockSize - 1) / blockSize;

      basic_spmv_kernel<<<numBlocks, blockSize>>>(mat.num_rows, mat.num_cols,
                                                  gpu_rows, gpu_cols, gpu_vals,
                                                  gpu_vec, gpu_res);

      cudaEventRecord(stop);
      cudaEventSynchronize(stop);

      cudaEventElapsedTime(&gputime_event, start, stop);

      cudaEventDestroy(start);
      cudaEventDestroy(stop);

      float *spmv_from_gpu_res = (float *)malloc(sizeof(float) * mat.num_rows);
      cudaMemcpy(spmv_from_gpu_res, gpu_res, mat.num_rows * sizeof(float),
                 cudaMemcpyDeviceToHost);

      double err = l2_error(mat.num_rows, res_cpu, spmv_from_gpu_res);

      if (err > ERROR_THRESHOLD) {

        printf("ERROR OVER THRESHOLD !!!");
      }
      printf("the error is  %.10e", err);

      std::cout << "-------------------------------------------\n\n";

      // ----------------- free GPU variable ---------------------

      cudaFree(gpu_rows);
      cudaFree(gpu_cols);
      cudaFree(gpu_res);
      cudaFree(gpu_vals);
      cudaFree(gpu_vec);
      // TODO: free rest of stuff I guess

    } catch (const std::exception &e) {
      std::cerr << "Error parsing '" << filename << "': " << e.what() << "\n";
    }
  }

  printf("====================================== Problem computations "
         "======================================\n");

  // ---------------------------------------------------------
  return 0;
}
