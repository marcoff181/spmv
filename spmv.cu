#include "fast_matrix_market/app/triplet.hpp"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <fast_matrix_market/fast_matrix_market.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <math.h>
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

const std::string MATRIX_FOLDER = "matrixes";
const float MAX_VECTOR_VALUE = 100.0;
const float ERROR_THRESHOLD = 0.000001;

__global__ void basic_spmv_kernel(int num_rows, int num_cols, int *rows,
                                  int *cols, float *vals, float *vec,
                                  float *res) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // int th_tot = gridDim.x * blockDim.x;

  if (tid > num_rows) {
    return;
  }

  for (int j = rows[tid]; j < rows[tid + 1]; ++j) {
    res[tid] += vec[cols[j]] * vals[j];
  }
}

void cpu_spmv(int num_rows, int num_cols, int *rows, int *cols, float *vals,
              float *vec, float *res) {
  int sum;
  for (int i = 0; i < num_rows; ++i) {
    for (int j = rows[i]; j < rows[i + 1]; ++j) {
      res[i] += vec[cols[j]] * vals[j];
    }
  }
}

bool compare_vectors(int len, float *v1, float *v2) {
  for (int i = 0; i < len; i++) {

    if (abs(v1[i] - v2[i]) > ERROR_THRESHOLD) {
      return false;
    }
  }
  return true;
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

    int64_t num_rows = 0;
    int64_t num_cols = 0;
    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<float> vals;

    std::ifstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Error: Could not open file '" << filename << "'\n";
      continue;
    }

    try {
      fast_matrix_market::read_matrix_market_triplet(file, num_rows, num_cols,
                                                     rows, cols, vals);

      std::cout << "Successfully read matrix from: " << filename << "\n";
      std::cout << "Dimensions: " << num_rows << " x " << num_cols << "\n";
      std::cout << "Non-zeros:  " << vals.size() << "\n";

      float *vec = (float *)malloc(sizeof(float) * num_cols);
      for (int i = 0; i < num_cols; i++) {
        vec[i] = static_cast<float>(rand()) /
                 (static_cast<float>(RAND_MAX / MAX_VECTOR_VALUE));
      };

      float *res_cpu = (float *)malloc(sizeof(float) * num_rows);
      for (int i = 0; i < num_rows; i++) {
        res_cpu[i] = 0;
      };
      // TODO: maybe use pointers when reading it from the beginning
      cpu_spmv(num_rows, num_rows, rows.data(), cols.data(), vals.data(), vec,
               res_cpu);

      // ====== GPU memory allocation
      float *gpu_vals, *gpu_vec, *gpu_res, gputime_event;
      int *gpu_rows, *gpu_cols;
      cudaMalloc(&gpu_rows, rows.size() * sizeof(int));
      cudaMalloc(&gpu_cols, cols.size() * sizeof(int));
      cudaMalloc(&gpu_vals, vals.size() * sizeof(float));
      cudaMalloc(&gpu_vec, num_cols * sizeof(float));
      cudaMalloc(&gpu_res, num_rows * sizeof(float));

      cudaMemcpy(gpu_rows, rows.data(), rows.size() * sizeof(int),
                 cudaMemcpyDefault);
      cudaMemcpy(gpu_cols, cols.data(), cols.size() * sizeof(int),
                 cudaMemcpyDefault);
      cudaMemcpy(gpu_vals, vals.data(), vals.size() * sizeof(float),
                 cudaMemcpyDefault);
      cudaMemcpy(gpu_vec, vec, num_cols * sizeof(float), cudaMemcpyDefault);
      cudaMemset(gpu_res, 0, num_rows * sizeof(float));

      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);

      cudaEventRecord(start);
      basic_spmv_kernel<<<1, 32>>>(num_rows, num_cols, gpu_rows, gpu_cols,
                                   gpu_vals, gpu_vec, gpu_res);

      cudaEventRecord(stop);
      cudaEventSynchronize(stop);

      cudaEventElapsedTime(&gputime_event, start, stop);

      cudaEventDestroy(start);
      cudaEventDestroy(stop);

      float *spmv_from_gpu_res = (float *)malloc(sizeof(float) * num_rows);
      cudaMemcpy(spmv_from_gpu_res, gpu_res, num_rows * sizeof(float),
                 cudaMemcpyDefault);

      bool equal = compare_vectors(num_rows, res_cpu, spmv_from_gpu_res);

      printf("the two computations are %d", equal);

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
