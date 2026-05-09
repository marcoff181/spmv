import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io
import scipy.sparse

# Load the CSV file
csv_path = '../results/results.csv'

# Skip the first 2 rows so that the 3rd row becomes the header
df = pd.read_csv(csv_path, skiprows=2)

# Ensure numeric types and clean data
df['GFLOP/s'] = pd.to_numeric(df['GFLOP/s'])
df['Bandwidth(GB/s)'] = pd.to_numeric(df['Bandwidth(GB/s)'])
df['Block_Size'] = pd.to_numeric(df['Block_Size'])
df['Chunk_Size'] = pd.to_numeric(df['Chunk_Size'])
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
# Plot 1: Average GFLOP/s vs Block_Size (Original)
# =====================================================================
target_kernels =['coo flat', 'csr scalar', 'csr vector', 'coo seg']
df_custom = df[df['Kernel'].isin(target_kernels)]
df_cusparse = df[df['Kernel'].str.lower().str.contains('cusparse')]

# Calculate Aggregate Average for custom kernels
agg_p1_gflops = df_custom.groupby(['Kernel', 'Block_Size'])['GFLOP/s'].mean().reset_index()

# Calculate Global Average for CuSparse
avg_cusparse_gflops = df_cusparse['GFLOP/s'].mean()

plt.figure(figsize=(10, 6))

# Plot custom kernels
for kernel in target_kernels:
    group = agg_p1_gflops[agg_p1_gflops['Kernel'] == kernel].sort_values('Block_Size')
    if not group.empty:
        plt.plot(group['Block_Size'], group['GFLOP/s'], marker='o', linewidth=2, label=kernel)

# Add CuSparse Reference Line
if not np.isnan(avg_cusparse_gflops):
    plt.axhline(y=avg_cusparse_gflops, color='red', linestyle='--', linewidth=2, 
                label=f'CuSparse Avg')

# Formatting
ticks = sorted(agg_p1_gflops['Block_Size'].unique())
plt.xticks(ticks, rotation=-45)
plt.xlabel('Block Size')
plt.ylabel('Average GFLOP/s')
plt.title('Aggregate Performance vs Block Size (GFLOP/s)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('plot1_blocksize_vs_gflops.png', dpi=300)
print("Saved: plot1_blocksize_vs_gflops.png")


# =====================================================================
# Plot 1b: Average Bandwidth vs Block_Size
# =====================================================================
# Calculate Aggregate Average for Bandwidth
agg_p1_bw = df_custom.groupby(['Kernel', 'Block_Size'])['Bandwidth(GB/s)'].mean().reset_index()

# Calculate Global Average for CuSparse Bandwidth
avg_cusparse_bw = df_cusparse['Bandwidth(GB/s)'].mean()

plt.figure(figsize=(10, 6))

for kernel in target_kernels:
    group = agg_p1_bw[agg_p1_bw['Kernel'] == kernel].sort_values('Block_Size')
    if not group.empty:
        plt.plot(group['Block_Size'], group['Bandwidth(GB/s)'], marker='o', linewidth=2, label=kernel)

if not np.isnan(avg_cusparse_bw):
    plt.axhline(y=avg_cusparse_bw, color='red', linestyle='--', linewidth=2, 
                label=f'CuSparse Avg')

ticks_bw = sorted(agg_p1_bw['Block_Size'].unique())
plt.xticks(ticks_bw, rotation=-45)
plt.xlabel('Block Size')
plt.ylabel('Average Bandwidth (GB/s)')
plt.title('Aggregate Performance vs Block Size (Bandwidth)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('plot1b_blocksize_vs_bandwidth.png', dpi=300)
print("Saved: plot1b_blocksize_vs_bandwidth.png")


# =====================================================================
# Plot 1c: Matplotlib Heatmap for 'coo seg' Kernel (No inner text)
# =====================================================================
df_coo_seg = df[df['Kernel'] == 'coo seg']

if not df_coo_seg.empty:
    # Prepare data for heatmap
    heatmap_data = df_coo_seg.groupby(['Chunk_Size', 'Block_Size'])['Bandwidth(GB/s)'].mean().unstack()
    
    # Extract values and labels
    data = heatmap_data.values
    chunk_sizes = heatmap_data.index.tolist()
    block_sizes = heatmap_data.columns.tolist()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Generate the heatmap
    im = ax.imshow(data, cmap='viridis', aspect='auto')
    
    # Set ticks and labels, making the X-axis vertical reading
    ax.set_xticks(np.arange(len(block_sizes)))
    ax.set_yticks(np.arange(len(chunk_sizes)))
    ax.set_xticklabels(block_sizes, rotation=90)
    ax.set_yticklabels(chunk_sizes)
    
    # Invert Y-axis so smaller chunk sizes are at the bottom (Standard for coordinate systems)
    ax.invert_yaxis()
    
    # Create horizontal colorbar at the bottom
    cbar = ax.figure.colorbar(im, ax=ax, orientation='horizontal')
    cbar.set_label('Bandwidth (GB/s)')
    
    ax.set_xlabel('Block Size')
    ax.set_ylabel('Chunk Size')
    ax.set_title('Bandwidth Heatmap for coo seg Kernel')
    fig.tight_layout()
    plt.savefig('plot1c_coo_seg_heatmap.png', dpi=300)
    print("Saved: plot1c_coo_seg_heatmap.png")
    
else:
    print("No data found for 'coo seg' kernel, skipping heatmap.")


# =====================================================================
# Data Preparation for Plots 2, 3, 4 (Best Performance across all sizes)
# =====================================================================
# Get max performance per matrix/kernel across ALL block sizes based on Bandwidth
perf_full_bw = df.groupby(['Matrix', 'Kernel'])['Bandwidth(GB/s)'].max().unstack()

# Safely add potential CuSparse kernels to target list to ensure they plot
target_kernels.extend([k for k in df['Kernel'].unique() if 'cusparse' in k.lower() and k not in target_kernels])

# =====================================================================
# Plot 2: Best Bandwidth per Matrix (Ordered by Density)
# =====================================================================
sorted_matrices_density = matrix_info_sorted_density['Matrix'].tolist()
perf_p2_bw = perf_full_bw.reindex(sorted_matrices_density)

label_map_density = dict(zip(matrix_info['Matrix'], matrix_info['Label_Density']))
perf_p2_bw.index =[label_map_density[m] for m in perf_p2_bw.index]

plt.figure(figsize=(12, 7))

for kernel in target_kernels:
    if kernel in perf_p2_bw.columns:
        plt.plot(perf_p2_bw.index, perf_p2_bw[kernel], 
                 marker='o', linewidth=2, label=kernel)

plt.xlabel('Matrix (Sorted by Density: NNZ / (R*C))')
plt.ylabel('Best Bandwidth (GB/s)')
plt.title('Best Kernel Performance for each matrix (Ordered by Matrix Density)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('plot2_density_sorted_performance_bw.png', dpi=300)
print("Saved: plot2_density_sorted_performance_bw.png")


# =====================================================================
# Plot 3: Best Bandwidth per Matrix (Ordered by Row NNZ Variance)
# =====================================================================
sorted_matrices_variance = matrix_info_sorted_variance['Matrix'].tolist()
perf_p3_bw = perf_full_bw.reindex(sorted_matrices_variance)

label_map_variance = dict(zip(matrix_info['Matrix'], matrix_info['Label_Variance']))
perf_p3_bw.index = [label_map_variance[m] for m in perf_p3_bw.index]

plt.figure(figsize=(12, 7))

for kernel in target_kernels:
    if kernel in perf_p3_bw.columns:
        plt.plot(perf_p3_bw.index, perf_p3_bw[kernel], 
                 marker='o', linewidth=2, label=kernel)

plt.xlabel('Matrix (Sorted by Variance of Non-Zeros per Row)')
plt.ylabel('Best Bandwidth (GB/s)')
plt.title('Best Kernel Performance for each matrix (Ordered by Row NNZ Variance)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('plot3_variance_sorted_performance_bw.png', dpi=300)
print("Saved: plot3_variance_sorted_performance_bw.png")


# =====================================================================
# Plot 4: Best Bandwidth per Matrix (Ordered by Total NNZ)
# =====================================================================
sorted_matrices_nnz = matrix_info_sorted_nnz['Matrix'].tolist()
perf_p4_bw = perf_full_bw.reindex(sorted_matrices_nnz)

label_map_nnz = dict(zip(matrix_info['Matrix'], matrix_info['Label_NNZ']))
perf_p4_bw.index =[label_map_nnz[m] for m in perf_p4_bw.index]

plt.figure(figsize=(12, 7))

for kernel in target_kernels:
    if kernel in perf_p4_bw.columns:
        plt.plot(perf_p4_bw.index, perf_p4_bw[kernel], 
                 marker='o', linewidth=2, label=kernel)

plt.xlabel('Matrix (Sorted by Total Non-Zeros)')
plt.ylabel('Best Bandwidth (GB/s)')
plt.title('Best Kernel Performance for each matrix (Ordered by NNZ)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('plot4_nnz_sorted_performance_bw.png', dpi=300)
print("Saved: plot4_nnz_sorted_performance_bw.png")
