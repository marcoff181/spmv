#include <string>
#include <vector>

struct CSR_Matrix {
  int num_rows;
  int num_cols;
  std::vector<int> coo_rows;
  std::vector<int> csr_rows;
  std::vector<int> cols;
  std::vector<float> vals;
};

bool load_mtx_file(const std::string &filename, CSR_Matrix &out_matrix);
