#include "kernels.cuh"
#include "matrix_loader.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#define WARMUP 2
#define NITER 10

#define CHECK_CUDA(func)                                                       \
  {                                                                            \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
      printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,                 \
             cudaGetErrorString(status));                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#define CHECK_CUSPARSE(func)                                                   \
  {                                                                            \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      printf("cuSPARSE Error at %s:%d\n", __FILE__, __LINE__);                 \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

template <typename T, typename U> inline T div_ceil(T n, U d) {
  return (n + d - 1) / d;
}

// TODO:
// - FLOPs measurement
// - more CSR kernels
// - Cache performance measurements

const std::string MATRIX_FOLDER = "matrices";
const float MAX_VECTOR_VALUE = 100.0;
// const double ERROR_THRESHOLD = 1e-5;

struct KernelTask {
  std::string name;
  dim3 grid;
  dim3 block;
  std::function<void()> launch;
};

void cpu_spmv(int m, int n, const std::vector<int> &rows,
              const std::vector<int> &cols, const std::vector<float> &vals,
              const std::vector<float> &x, std::vector<float> &y) {
  for (int i = 0; i < m; ++i) {
    for (int j = rows[i]; j < rows[i + 1]; ++j) {
      y[i] += x[cols[j]] * vals[j];
    }
  }
}

double l2_error(int len, const std::vector<float> &reference,
                const std::vector<float> &comparison) {
  // L2 norm of reference
  double ref_sum_sq = std::inner_product(reference.begin(), reference.end(),
                                         reference.begin(), 0.0);
  double ref_norm = std::sqrt(ref_sum_sq);

  // L2 norm of difference
  double diff_sum_sq = 0.0;
  for (size_t i = 0; i < reference.size(); ++i) {
    double diff =
        static_cast<double>(comparison[i]) - static_cast<double>(reference[i]);
    diff_sum_sq += diff * diff;
  }
  double diff_norm = std::sqrt(diff_sum_sq);

  return diff_norm / ref_norm;
}

namespace fs = std::filesystem;

int main() {

  double avg_error;
  float avg_time;

  srand(0);

  if (!fs::exists(MATRIX_FOLDER) || !fs::is_directory(MATRIX_FOLDER)) {
    std::cerr << "Error: Directory '" << MATRIX_FOLDER << "' does not exist.\n";
    return 1;
  }

  std::ofstream csv_file("results.csv");

  // ====== write GPU info to csv
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  float total_mem_gb = prop.totalGlobalMem / 1073741824.0f;

  int max_warps_per_sm = prop.maxThreadsPerMultiProcessor / prop.warpSize;
  int total_max_concurrent_threads =
      prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor;

  csv_file << "GPU_Name,Memory_GB,Num_SMs,Max_Threads_Per_SM,Max_Warps_Per_SM,"
              "Max_Threads_Per_Block,Total_Max_Threads"
           << std::endl;
  csv_file << "\"" << prop.name << "\"," // GPU Name (quoted in case of commas)
           << total_mem_gb << ","        // Total Memory
           << prop.multiProcessorCount << "," // SM Count
           << prop.maxThreadsPerMultiProcessor << "," << max_warps_per_sm << ","
           << prop.maxThreadsPerBlock << ","
           << total_max_concurrent_threads // Global hardware ceiling
           << std::endl;
  csv_file << "Matrix,Rows,Columns,nnz,Kernel,Grid_Size,Block_Size,Avg_Time(ms)"
              ",Avg_Err\n";

  // ====== cusparse setup
  cusparseHandle_t handle;
  cusparseCreate(&handle);

  for (const auto &entry : fs::directory_iterator(MATRIX_FOLDER)) {
    if (entry.path().extension() != ".mtx")
      continue;

    std::string filename = entry.path().filename().string();

    try {
      CSR_Matrix mtx;
      load_mtx_file(entry.path().string(), mtx);

      int m = mtx.num_rows;
      int n = mtx.num_cols;
      long nnz = mtx.vals.size();
      std::vector<int> csr_rows = mtx.csr_rows;
      std::vector<int> coo_rows = mtx.coo_rows;
      std::vector<int> cols = mtx.cols;
      std::vector<float> vals = mtx.vals;
      std::vector<float> y(m, 0.0);

      // initialize random vector
      std::vector<float> x(n);
      for (int i = 0; i < n; i++) {
        x[i] = static_cast<float>(rand()) /
               (static_cast<float>(RAND_MAX / MAX_VECTOR_VALUE));
      };

      // ====== compute reference vector with cpu
      std::vector<float> cpu_reference(m, 0.0);

      cpu_spmv(m, n, csr_rows, cols, vals, x, cpu_reference);

      // ====== GPU memory allocation
      float *gpu_vals, *gpu_x, *gpu_y;
      int *gpu_csr_rows, *gpu_coo_rows, *gpu_cols;

      cudaMalloc(&gpu_csr_rows, csr_rows.size() * sizeof(int));
      cudaMalloc(&gpu_coo_rows, coo_rows.size() * sizeof(int));
      cudaMalloc(&gpu_cols, cols.size() * sizeof(int));
      cudaMalloc(&gpu_vals, vals.size() * sizeof(float));
      cudaMalloc(&gpu_x, n * sizeof(float));
      cudaMalloc(&gpu_y, m * sizeof(float));

      cudaMemcpy(gpu_csr_rows, csr_rows.data(), csr_rows.size() * sizeof(int),
                 cudaMemcpyHostToDevice);
      cudaMemcpy(gpu_coo_rows, coo_rows.data(), coo_rows.size() * sizeof(int),
                 cudaMemcpyHostToDevice);
      cudaMemcpy(gpu_cols, cols.data(), cols.size() * sizeof(int),
                 cudaMemcpyHostToDevice);
      cudaMemcpy(gpu_vals, vals.data(), vals.size() * sizeof(float),
                 cudaMemcpyHostToDevice);
      cudaMemcpy(gpu_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice);

      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);

      // ====== CuSparse matrix setup
      float alpha = 1.0f;
      float beta = 0.0f;
      void *dBuffer = nullptr;
      size_t bufferSize = 0;
      cusparseDnVecDescr_t vecX, vecY;
      cusparseSpMatDescr_t matA;
      CHECK_CUSPARSE(cusparseCreateCsr(&matA, m, n, nnz, gpu_csr_rows, gpu_cols,
                                       gpu_vals, CUSPARSE_INDEX_32I,
                                       CUSPARSE_INDEX_32I,
                                       CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
      CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, n, gpu_x, CUDA_R_32F));
      CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, m, gpu_y, CUDA_R_32F));
      CHECK_CUSPARSE(cusparseSpMV_bufferSize(
          handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta,
          vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
      CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

      // ====== kernels "tasks" list
      std::vector<KernelTask> kernels;
      int numBlocks, cooBlocks, warpsPerBlock;

      for (int blockSize : {32, 64, 128, 256, 512, 1024}) {
        cooBlocks = div_ceil(nnz, blockSize);
        kernels.push_back({"coo flat", dim3(cooBlocks), dim3(blockSize), [=]() {
                             coo_flat<<<cooBlocks, blockSize>>>(
                                 nnz, gpu_coo_rows, gpu_cols, gpu_vals, gpu_x,
                                 gpu_y);
                           }});

        for (int chunk_size : {128, 256, 1024, 4096, 8192}) {
          int total_warps_needed = div_ceil(nnz, chunk_size);
          warpsPerBlock = blockSize / 32;
          int numBlocks = div_ceil(total_warps_needed, warpsPerBlock);

          kernels.push_back(
              {"coo seg chunksize=" + std::to_string(chunk_size),
               dim3(numBlocks), dim3(blockSize), [=]() {
                 coo_segmented_reduction<<<numBlocks, blockSize>>>(
                     nnz, chunk_size, gpu_coo_rows, gpu_cols, gpu_vals, gpu_x,
                     gpu_y);
               }});
        }

        numBlocks = div_ceil(m, blockSize);
        kernels.push_back(
            {"csr scalar", dim3(numBlocks), dim3(blockSize), [=]() {
               csr_scalar<<<numBlocks, blockSize>>>(m, gpu_csr_rows, gpu_cols,
                                                    gpu_vals, gpu_x, gpu_y);
             }});

        // 1 warp (32 threads) per row
        warpsPerBlock = blockSize / 32;
        numBlocks = div_ceil(m, warpsPerBlock);
        kernels.push_back(
            {"csr vector", dim3(numBlocks), dim3(blockSize), [=]() {
               csr_vector<<<numBlocks, blockSize>>>(m, gpu_csr_rows, gpu_cols,
                                                    gpu_vals, gpu_x, gpu_y);
             }});
      }

      kernels.push_back({"CuSparse", dim3(0), dim3(0), [=]() {
                           CHECK_CUSPARSE(cusparseSpMV(
                               handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                               matA, vecX, &beta, vecY, CUDA_R_32F,
                               CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
                         }});

      // ====== task execution
      for (const KernelTask &task : kernels) {
        avg_error = 0.0;
        avg_time = 0.0;
        for (int i = -WARMUP; i < NITER; i++) {
          cudaMemset(gpu_y, 0, m * sizeof(float));

          cudaEventRecord(start);
          task.launch();
          cudaEventRecord(stop);
          CHECK_CUDA(cudaEventSynchronize(stop));

          if (i >= 0) {
            float iter_time = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&iter_time, start, stop));
            avg_time += iter_time;

            cudaMemcpy(y.data(), gpu_y, m * sizeof(float),
                       cudaMemcpyDeviceToHost);
            avg_error += l2_error(m, cpu_reference, y);
          }
        }

        avg_error = avg_error / NITER;
        avg_time = avg_time / NITER;

        csv_file << filename << "," << m << "," << n << "," << nnz << ","
                 << task.name << "," << task.grid.x << "," << task.block.x
                 << "," << avg_time << "," << avg_error << "\n";

        csv_file << std::flush;
      }

      cudaEventDestroy(start);
      cudaEventDestroy(stop);

      cudaFree(gpu_csr_rows);
      cudaFree(gpu_coo_rows);
      cudaFree(gpu_cols);
      cudaFree(gpu_y);
      cudaFree(gpu_vals);
      cudaFree(gpu_x);

      cudaFree(dBuffer);
      cusparseDestroyDnVec(vecX);
      cusparseDestroyDnVec(vecY);
      cusparseDestroySpMat(matA);

    } catch (const std::exception &e) {
      std::cerr << "Error parsing '" << filename << "': " << e.what() << "\n";
    }
  }

  cusparseDestroy(handle);

  csv_file.close();
  return 0;
}
