import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter,LogLocator

df = pd.read_csv('../results/results.csv', skiprows=2)
stats = pd.read_csv('matrix_stats.csv')

stats = stats.rename(columns={
    'NNZ':                  'nnz',
    'Density':              'Density',
    'Variance NNZ per Row': 'Row_NNZ_Variance',
})

for col in ['GFLOP/s', 'Bandwidth(GB/s)', 'Avg_Time(ms)', 'Block_Size', 'Chunk_Size']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.merge(stats[['Matrix', 'Density', 'Row_NNZ_Variance']], on='Matrix', how='left')

target_kernels = ['coo flat', 'csr scalar', 'csr vector', 'coo seg']
df_custom   = df[df['Kernel'].isin(target_kernels)]
df_cusparse = df[df['Kernel'].str.lower().str.contains('cusparse')]

all_kernels = target_kernels + [k for k in df['Kernel'].unique()
                                 if 'cusparse' in k.lower() and k not in target_kernels]

matrix_info = df[['Matrix', 'nnz', 'Density', 'Row_NNZ_Variance']].drop_duplicates()

def label(row, metric, fmt):
    name = row['Matrix'].replace('.mtx', '')
    return f"{name}\n({metric}: {row[metric]:{fmt}})"

matrix_info = matrix_info.copy()
matrix_info['Label_Density']  = matrix_info.apply(lambda r: label(r, 'Density', '.2e'), axis=1)
matrix_info['Label_NNZ']      = matrix_info.apply(lambda r: label(r, 'nnz', '.1e'), axis=1)
matrix_info['Label_Variance'] = matrix_info.apply(lambda r: label(r, 'Row_NNZ_Variance', '.1e'), axis=1)


agg_gflops = df_custom.groupby(['Kernel', 'Block_Size'])['GFLOP/s'].mean().reset_index()
avg_cusparse_gflops = df_cusparse['GFLOP/s'].mean()

plt.figure(figsize=(10, 6))
for kernel in target_kernels:
    g = agg_gflops[agg_gflops['Kernel'] == kernel].sort_values('Block_Size')
    if not g.empty:
        plt.plot(g['Block_Size'], g['GFLOP/s'], marker='o', linewidth=2, label=kernel)
if not np.isnan(avg_cusparse_gflops):
    plt.axhline(y=avg_cusparse_gflops, color='red', linestyle='--', linewidth=2, label='CuSparse Avg')
plt.xticks(sorted(agg_gflops['Block_Size'].unique()), rotation=-45)
plt.xlabel('Block Size')
plt.ylabel('Average GFLOP/s')
plt.title('Aggregate Performance vs Block Size (GFLOP/s)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('plot1_blocksize_vs_gflops.png', dpi=300)
print("Saved: plot1_blocksize_vs_gflops.png")


agg_bw = df_custom.groupby(['Kernel', 'Block_Size'])['Bandwidth(GB/s)'].mean().reset_index()
avg_cusparse_bw = df_cusparse['Bandwidth(GB/s)'].mean()

plt.figure(figsize=(10, 6))
for kernel in target_kernels:
    g = agg_bw[agg_bw['Kernel'] == kernel].sort_values('Block_Size')
    if not g.empty:
        plt.plot(g['Block_Size'], g['Bandwidth(GB/s)'], marker='o', linewidth=2, label=kernel)
if not np.isnan(avg_cusparse_bw):
    plt.axhline(y=avg_cusparse_bw, color='red', linestyle='--', linewidth=2, label='CuSparse Avg')
plt.xticks(sorted(agg_bw['Block_Size'].unique()), rotation=-45)
plt.xlabel('Block Size')
plt.ylabel('Average Bandwidth (GB/s)')
plt.title('Aggregate Performance vs Block Size (Bandwidth)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('plot1b_blocksize_vs_bandwidth.png', dpi=300)
print("Saved: plot1b_blocksize_vs_bandwidth.png")


df_coo_seg = df[df['Kernel'] == 'coo seg']
if not df_coo_seg.empty:
    heatmap_data = df_coo_seg.groupby(['Chunk_Size', 'Block_Size'])['Bandwidth(GB/s)'].mean().unstack()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(heatmap_data.values, cmap='viridis', aspect='auto')
    ax.set_xticks(np.arange(len(heatmap_data.columns)))
    ax.set_yticks(np.arange(len(heatmap_data.index)))
    ax.set_xticklabels(heatmap_data.columns.tolist(), rotation=90)
    ax.set_yticklabels(heatmap_data.index.tolist())
    ax.invert_yaxis()
    cbar = ax.figure.colorbar(im, ax=ax, orientation='horizontal')
    cbar.set_label('Bandwidth (GB/s)')
    ax.set_xlabel('Block Size')
    ax.set_ylabel('Chunk Size')
    ax.set_title('Bandwidth Heatmap for coo seg Kernel')
    fig.tight_layout()
    plt.savefig('plot1c_coo_seg_heatmap.png', dpi=300)
    print("Saved: plot1c_coo_seg_heatmap.png")


perf_rt = df.groupby(['Matrix', 'Kernel'])['Avg_Time(ms)'].min().unstack()

for sort_col, label_col, xlabel, filename in [
    ('Density',          'Label_Density',  'Matrix (Sorted by Density)',           'plot2_density_sorted_runtime.png'),
    ('Row_NNZ_Variance', 'Label_Variance', 'Matrix (Sorted by Row NNZ Variance)',  'plot3_variance_sorted_runtime.png'),
    ('nnz',              'Label_NNZ',      'Matrix (Sorted by Total NNZ)',          'plot4_nnz_sorted_runtime.png'),
]:
    sorted_info   = matrix_info.sort_values(sort_col)
    label_map     = dict(zip(sorted_info['Matrix'], sorted_info[label_col]))
    perf_sorted   = perf_rt.reindex(sorted_info['Matrix'].tolist())
    perf_sorted.index = [label_map[m] for m in perf_sorted.index]

    plt.figure(figsize=(12, 7))
    for kernel in all_kernels:
        if kernel in perf_sorted.columns:
            plt.plot(perf_sorted.index, perf_sorted[kernel], marker='o', linewidth=2, label=kernel)
    plt.xlabel(xlabel)
    plt.ylabel('Best Avg Time (ms)')
    plt.yscale('log')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))

    plt.title(f'Best Kernel Runtime per Matrix ({xlabel.split("by")[1].strip()})')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved: {filename}")
