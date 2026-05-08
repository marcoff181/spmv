import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io
import scipy.sparse

# Load the CSV file
csv_path = '../results/results.csv'
df = pd.read_csv(csv_path)

# Ensure numeric types and clean data
df['GFLOP/s'] = pd.to_numeric(df['GFLOP/s'])
df['Block_Size'] = pd.to_numeric(df['Block_Size'])
df['nnz'] = pd.to_numeric(df['nnz'])
df['Rows'] = pd.to_numeric(df['Rows'])
df['Columns'] = pd.to_numeric(df['Columns'])

# 1. Calculate Density for each matrix row
df['Density'] = df['nnz'] / (df['Rows'] * df['Columns'])

# --- Calculate Variance of non-zeros per row ---
def get_row_nnz_variance(matrix_name):
    mat_path = os.path.join('../matrices/', matrix_name)
    if not os.path.exists(mat_path):
        print(f"Warning: {mat_path} not found. Variance set to 0.")
        return 0.0
    try:
        # Load the matrix using scipy
        mat = scipy.io.mmread(mat_path)
        
        # Convert to CSR format to easily extract the number of non-zeros per row
        if scipy.sparse.issparse(mat):
            nnz_per_row = mat.tocsr().getnnz(axis=1)
        else:
            nnz_per_row = np.count_nonzero(mat, axis=1)
            
        return np.var(nnz_per_row)
    except Exception as e:
        print(f"Error calculating variance for {matrix_name}: {e}")
        return 0.0

# Map variance to each matrix and add as a new column
unique_matrices = df['Matrix'].unique()
variance_map = {m: get_row_nnz_variance(m) for m in unique_matrices}
df['Row_NNZ_Variance'] = df['Matrix'].map(variance_map)
# ----------------------------------------------------

# 2. Create helpers for cleaned names and formatting
def format_label_density(row):
    name = row['Matrix'].replace('.mtx', '')
    return f"{name}\n({row['Density']:.2e})"

def format_label_nnz(row):
    name = row['Matrix'].replace('.mtx', '')
    return f"{name}\n(nnz: {row['nnz']:.1e})"

def format_label_variance(row):
    name = row['Matrix'].replace('.mtx', '')
    return f"{name}\n(Var: {row['Row_NNZ_Variance']:.1e})"

# Create a mapping of Matrix Name -> Clean Label and Metrics
matrix_info = df[['Matrix', 'Density', 'nnz', 'Row_NNZ_Variance']].drop_duplicates()
matrix_info['Label_Density'] = matrix_info.apply(format_label_density, axis=1)
matrix_info['Label_NNZ'] = matrix_info.apply(format_label_nnz, axis=1)
matrix_info['Label_Variance'] = matrix_info.apply(format_label_variance, axis=1)

# Sort matrix info for the respective plots
matrix_info_sorted_density = matrix_info.sort_values('Density')
matrix_info_sorted_nnz = matrix_info.sort_values('nnz')
matrix_info_sorted_variance = matrix_info.sort_values('Row_NNZ_Variance')

# =====================================================================
# Plot 1: Average GFLOP/s vs Block_Size
# =====================================================================
target_kernels = ['coo flat', 'csr scalar', 'csr vector', 'coo seg']
df_custom = df[df['Kernel'].isin(target_kernels)]
df_cusparse = df[df['Kernel'].str.lower().str.contains('cusparse')]

# Calculate Aggregate Average for custom kernels
agg_p1 = df_custom.groupby(['Kernel', 'Block_Size'])['GFLOP/s'].mean().reset_index()

# Calculate Global Average for CuSparse
avg_cusparse = df_cusparse['GFLOP/s'].mean()

plt.figure(figsize=(10, 6))

# Plot custom kernels
for kernel in target_kernels:
    group = agg_p1[agg_p1['Kernel'] == kernel].sort_values('Block_Size')
    if not group.empty:
        plt.plot(group['Block_Size'], group['GFLOP/s'], marker='o', linewidth=2, label=kernel)

# Add CuSparse Reference Line
if not np.isnan(avg_cusparse):
    plt.axhline(y=avg_cusparse, color='red', linestyle='--', linewidth=2, 
                label=f'CuSparse Avg')

# Formatting
ticks = sorted(agg_p1['Block_Size'].unique())
plt.xticks(ticks, rotation=-45)
plt.xlabel('Block Size')
plt.ylabel('Average GFLOP/s')
plt.title('Aggregate Performance vs Block Size')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('plot1_blocksize_vs_gflops.png', dpi=300)
print("Saved: plot1_blocksize_vs_gflops.png")


# =====================================================================
# Data Preparation for Plots 2, 3, 4 (Fixed Block Size = 128)
# =====================================================================
# Filter dataset for Block_Size == 128. Also include CuSparse since it might not be tied to a block size.
df_bs128 = df[(df['Block_Size'] == 128) | (df['Kernel'].str.lower().str.contains('cusparse'))]

# Get max performance per matrix/kernel for this subset (using max handles potential duplicate runs gracefully)
perf_full = df_bs128.groupby(['Matrix', 'Kernel'])['GFLOP/s'].max().unstack()

# Add CuSparse to target_kernels for the remaining plots
target_kernels.append("CuSparse")

# =====================================================================
# Plot 2: GFLOP/s per Matrix (Block Size 128, Ordered by Density)
# =====================================================================
sorted_matrices_density = matrix_info_sorted_density['Matrix'].tolist()
perf_p2 = perf_full.reindex(sorted_matrices_density)

label_map_density = dict(zip(matrix_info['Matrix'], matrix_info['Label_Density']))
perf_p2.index = [label_map_density[m] for m in perf_p2.index]

plt.figure(figsize=(12, 7))

for kernel in target_kernels:
    if kernel in perf_p2.columns:
        plt.plot(perf_p2.index, perf_p2[kernel], 
                 marker='o', linewidth=2, label=kernel)

plt.xlabel('Matrix (Sorted by Density: NNZ / (R*C))')
plt.ylabel('GFLOP/s (Block Size 128)')
plt.title('Kernel Performance for each matrix (Block Size 128, Ordered by Matrix Density)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('plot2_density_sorted_performance.png', dpi=300)
print("Saved: plot2_density_sorted_performance.png")


# =====================================================================
# Plot 3: GFLOP/s per Matrix (Block Size 128, Ordered by Row NNZ Variance)
# =====================================================================
sorted_matrices_variance = matrix_info_sorted_variance['Matrix'].tolist()
perf_p3 = perf_full.reindex(sorted_matrices_variance)

label_map_variance = dict(zip(matrix_info['Matrix'], matrix_info['Label_Variance']))
perf_p3.index = [label_map_variance[m] for m in perf_p3.index]

plt.figure(figsize=(12, 7))

for kernel in target_kernels:
    if kernel in perf_p3.columns:
        plt.plot(perf_p3.index, perf_p3[kernel], 
                 marker='o', linewidth=2, label=kernel)

plt.xlabel('Matrix (Sorted by Variance of Non-Zeros per Row)')
plt.ylabel('GFLOP/s (Block Size 128)')
plt.title('Kernel Performance for each matrix (Block Size 128, Ordered by Row NNZ Variance)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('plot3_variance_sorted_performance.png', dpi=300)
print("Saved: plot3_variance_sorted_performance.png")


# =====================================================================
# Plot 4: GFLOP/s per Matrix (Block Size 128, Ordered by Total NNZ)
# =====================================================================
sorted_matrices_nnz = matrix_info_sorted_nnz['Matrix'].tolist()
perf_p4 = perf_full.reindex(sorted_matrices_nnz)

label_map_nnz = dict(zip(matrix_info['Matrix'], matrix_info['Label_NNZ']))
perf_p4.index = [label_map_nnz[m] for m in perf_p4.index]

plt.figure(figsize=(12, 7))

for kernel in target_kernels:
    if kernel in perf_p4.columns:
        plt.plot(perf_p4.index, perf_p4[kernel], 
                 marker='o', linewidth=2, label=kernel)

plt.xlabel('Matrix (Sorted by Total Non-Zeros)')
plt.ylabel('GFLOP/s (Block Size 128)')
plt.title('Kernel Performance for each matrix (Block Size 128, Ordered by NNZ)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('plot4_nnz_sorted_performance.png', dpi=300)
print("Saved: plot4_nnz_sorted_performance.png")
