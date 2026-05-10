import os
import glob
import csv
import math
from scipy.io import mmread
import numpy as np


INPUT_GLOB = "../matrices/*.mtx"
OUTPUT_CSV = "matrix_stats.csv"


def analyze_matrix(filepath):
    matrix = mmread(filepath).tocsr()

    rows, cols = matrix.shape
    nnz = matrix.nnz
    density = nnz / (rows * cols) if (rows * cols) > 0 else 0.0

    nnz_per_row = np.diff(matrix.indptr).astype(np.float64)

    avg_nnz = float(np.mean(nnz_per_row))
    var_nnz = float(np.var(nnz_per_row))          # population variance
    std_nnz = math.sqrt(var_nnz)
    cv_nnz = (std_nnz / avg_nnz) if avg_nnz > 0 else 0.0

    return {
        "rows":        rows,
        "cols":        cols,
        "nnz":         nnz,
        "density":     density,
        "avg_nnz_row": avg_nnz,
        "var_nnz_row": var_nnz,
        "cv_nnz_row":  cv_nnz,
    }


def main():
    mtx_files = sorted(glob.glob(INPUT_GLOB))

    if not mtx_files:
        print(f"No .mtx files found matching '{INPUT_GLOB}'")
        return

    fieldnames = [
        "Matrix",
        "Rows",
        "Columns",
        "NNZ",
        "Density",
        "Avg NNZ per Row",
        "Variance NNZ per Row",
        "CV NNZ per Row",
    ]

    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for filepath in mtx_files:
            name = os.path.basename(filepath)   # e.g. cant.mtx
            print(f"Processing {name} ...", end=" ", flush=True)

            try:
                stats = analyze_matrix(filepath)
                writer.writerow({
                    "Matrix":               name,
                    "Rows":                 stats["rows"],
                    "Columns":              stats["cols"],
                    "NNZ":                  stats["nnz"],
                    "Density":              f"{stats['density']:.6e}",
                    "Avg NNZ per Row":      f"{stats['avg_nnz_row']:.4f}",
                    "Variance NNZ per Row": f"{stats['var_nnz_row']:.4f}",
                    "CV NNZ per Row":       f"{stats['cv_nnz_row']:.4f}",
                })
                print("done")
            except Exception as e:
                print(f"ERROR: {e}")

    print(f"\nResults written to '{OUTPUT_CSV}'")


if __name__ == "__main__":
    main()
