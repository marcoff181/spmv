#include "matrix_loader.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <filesystem>
#include <fstream>
#include <functional>
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

// #define STR(s) #s
// #define XSTR(s) STR(s)

#define WARMUP 2
#define NITER 10

const std::string MATRIX_FOLDER = "matrices";
const float MAX_VECTOR_VALUE = 100.0;
const double ERROR_THRESHOLD = 1e-5;

struct KernelTask {
  std::string name;
  dim3 grid;
  dim3 block;
  std::function<void()> launch;
};

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

double l2_error(int len, float *reference, float *other) {
  // L2 norm of reference
  double ref_sum_sq =
      std::inner_product(reference, reference + len, reference, 0.0);
  double ref_norm = std::sqrt(ref_sum_sq);

  // L2 norm of difference
  double diff_sum_sq = 0.0;
  for (size_t i = 0; i < len; ++i) {
    double diff = other[i] - reference[i];
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
      // read matrix
      CSR_Matrix mtx;
      load_mtx_file(filename, mtx);

      int num_rows = mtx.num_rows;
      int num_cols = mtx.num_cols;

      std::cout << "Successfully read matrix from: " << filename << "\n";
      std::cout << "Dimensions: " << num_rows << " x " << num_cols << "\n";
      std::cout << "Non-zeros:  " << mtx.vals.size() << "\n";

      // initialize random vector
      float *vec = (float *)malloc(sizeof(float) * num_cols);
      for (int i = 0; i < num_cols; i++) {
        vec[i] = static_cast<float>(rand()) /
                 (static_cast<float>(RAND_MAX / MAX_VECTOR_VALUE));
      };

      // compute once the reference result using cpu
      float *cpu_reference = (float *)malloc(sizeof(float) * num_rows);
      for (int i = 0; i < num_rows; i++) {
        cpu_reference[i] = 0;
      };
      cpu_spmv(num_rows, num_cols, mtx.rows.data(), mtx.cols.data(),
               mtx.vals.data(), vec, cpu_reference);

      // ====== GPU memory allocation
      float *gpu_vals, *gpu_vec, *gpu_res, gputime_event;
      int *gpu_rows, *gpu_cols;
      cudaMalloc(&gpu_rows, mtx.rows.size() * sizeof(int));
      cudaMalloc(&gpu_cols, mtx.cols.size() * sizeof(int));
      cudaMalloc(&gpu_vals, mtx.vals.size() * sizeof(float));
      cudaMalloc(&gpu_vec, num_cols * sizeof(float));
      cudaMalloc(&gpu_res, num_rows * sizeof(float));
      cudaMemcpy(gpu_rows, mtx.rows.data(), mtx.rows.size() * sizeof(int),
                 cudaMemcpyHostToDevice);
      cudaMemcpy(gpu_cols, mtx.cols.data(), mtx.cols.size() * sizeof(int),
                 cudaMemcpyHostToDevice);
      cudaMemcpy(gpu_vals, mtx.vals.data(), mtx.vals.size() * sizeof(float),
                 cudaMemcpyHostToDevice);
      cudaMemcpy(gpu_vec, vec, num_cols * sizeof(float),
                 cudaMemcpyHostToDevice);

      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);

      float *spmv_from_gpu_res = (float *)malloc(sizeof(float) * num_rows);

      // define the kernels
      std::vector<KernelTask> kernels;

      for (int blockSize : {128, 256, 512}) {
        kernels.push_back({"FastBlur", dim3(1024), dim3(blockSize),
                           [&gpu_rows, &gpu_cols, &gpu_vals, &gpu_vec, &gpu_res,
                            num_rows, num_cols, blockSize]() {
                             int numBlocks =
                                 (num_rows + blockSize - 1) / blockSize;
                             basic_spmv_kernel<<<numBlocks, blockSize>>>(
                                 num_rows, num_cols, gpu_rows, gpu_cols,
                                 gpu_vals, gpu_vec, gpu_res);
                           }});
      }

      std::ofstream csv_file("results.csv");

      // iterate through all kernel executions
      for (const KernelTask &task : kernels) {
        csv_file << task.name << ","
                 << "Grid:[" << task.grid.x << " " << task.grid.y << " "
                 << task.grid.z << "],"
                 << "Block:[" << task.block.x << " " << task.block.y << " "
                 << task.block.z << "]\n";

        cudaMemset(gpu_res, 0, num_rows * sizeof(float));

        for (int i = -WARMUP; i < NITER; i++) {
          cudaEventRecord(start);
          task.launch();
          cudaEventRecord(stop);
          cudaEventSynchronize(stop);

          ;
          if (i >= 0) {
            float iter_time = cudaEventElapsedTime(&gputime_event, start, stop);

            cudaMemcpy(spmv_from_gpu_res, gpu_res, num_rows * sizeof(float),
                       cudaMemcpyDeviceToHost);

            double err = l2_error(num_rows, cpu_reference, spmv_from_gpu_res);

            if (err > ERROR_THRESHOLD) {
              printf("ERROR OVER THRESHOLD !!!");
            }
            printf("the error is  %.10e", err);
            csv_file << "run" << (i + 1) << "," << iter_time << "," << err
                     << "\n";

            std::cout << "-------------------------------------------\n\n";
          }

          // printf("Iteration %d tooks %lfms\n", i, iter_time);
        }
        csv_file << std::flush;
      }
      csv_file << "\n";

      free(spmv_from_gpu_res);

      cudaEventDestroy(start);
      cudaEventDestroy(stop);

      // ----------------- free GPU variable ---------------------

      cudaFree(gpu_rows);
      cudaFree(gpu_cols);
      cudaFree(gpu_res);
      cudaFree(gpu_vals);
      cudaFree(gpu_vec);
      // TODO: free rest of stuff
    } catch (const std::exception &e) {
      std::cerr << "Error parsing '" << filename << "': " << e.what() << "\n";
    }
  }

  return 0;
}
