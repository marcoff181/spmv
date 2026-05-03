#include "matrix_loader.h"
#include <fast_matrix_market/fast_matrix_market.hpp>
#include <fstream>
#include <iostream>
#include <numeric>

bool load_mtx_file(const std::string &filename, CSR_Matrix &out_matrix) {
  std::ifstream f(filename);
  if (!f.is_open()) {
    std::cerr << "Failed to open " << filename << std::endl;
    return false;
  }

  fast_matrix_market::read_matrix_market_triplet(
      f, out_matrix.num_rows, out_matrix.num_cols, out_matrix.coo_rows,
      out_matrix.cols, out_matrix.vals);

  // Sort by rows instead of columns
  std::vector<int> permutation(out_matrix.vals.size());
  std::iota(permutation.begin(), permutation.end(), 0);

  std::sort(permutation.begin(), permutation.end(), [&](int a, int b) {
    if (out_matrix.coo_rows[a] != out_matrix.coo_rows[b])
      return out_matrix.coo_rows[a] < out_matrix.coo_rows[b];
    return out_matrix.cols[a] < out_matrix.cols[b];
  });

  std::vector<int> sorted_coo_rows(out_matrix.coo_rows.size());
  std::vector<int> sorted_cols(out_matrix.cols.size());
  std::vector<float> sorted_vals(out_matrix.vals.size());

  for (size_t i = 0; i < permutation.size(); i++) {
    sorted_coo_rows[i] = out_matrix.coo_rows[permutation[i]];
    sorted_cols[i] = out_matrix.cols[permutation[i]];
    sorted_vals[i] = out_matrix.vals[permutation[i]];
  }

  out_matrix.coo_rows = std::move(sorted_coo_rows);
  out_matrix.cols = std::move(sorted_cols);
  out_matrix.vals = std::move(sorted_vals);

  // CSR rows conversion
  std::vector<int> csr_rows(out_matrix.num_rows + 1, 0);
  for (size_t i = 0; i < out_matrix.coo_rows.size(); i++) {
    csr_rows[out_matrix.coo_rows[i] + 1]++;
  }
  for (int i = 0; i < out_matrix.num_rows; i++) {
    csr_rows[i + 1] += csr_rows[i];
  }
  out_matrix.csr_rows = std::move(csr_rows);

  return true;
}
