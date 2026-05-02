#include "matrix_loader.h"
#include <fast_matrix_market/fast_matrix_market.hpp>
#include <fstream>
#include <iostream>

bool load_mtx_file(const std::string &filename, CSR_Matrix &out_matrix) {
  std::ifstream f(filename);
  if (!f.is_open()) {
    std::cerr << "Failed to open " << filename << std::endl;
    return false;
  }

  fast_matrix_market::read_matrix_market_triplet(
      f, out_matrix.num_rows, out_matrix.num_cols, out_matrix.coo_rows,
      out_matrix.cols, out_matrix.vals);

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
