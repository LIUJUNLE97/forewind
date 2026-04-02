import xarray as xr
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 确保在 SSH 环境下正常运行
import matplotlib.pyplot as plt
import glob
import os
import warnings
from matplotlib.lines import Line2D

# 忽略警告
warnings.filterwarnings("ignore")

# ================= 配置区域 =================
NC_DIR = '/mimer/NOBACKUP/groups/kthmech/junle/guandi_wind_project/data_2/processed'
CLUSTER_CSV = '/mimer/NOBACKUP/groups/kthmech/junle/guandi_wind_project/data_2/som/som_full_results.csv'
OUTPUT_DIR = '/mimer/NOBACKUP/groups/kthmech/junle/guandi_wind_project/som_post_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_LINES = 50   # 采样线条数
# ===========================================

def load_data(nc_dir, cluster_csv):
    print("Step 1: Loading Data...")
    df_clusters = pd.read_csv(cluster_csv)
    if 'cluster_id' in df_clusters.columns:
        df_clusters.rename(columns={'cluster_id': 'cluster'}, inplace=True)
    
    # 编号从1开始
    print("  - Adjusting Cluster IDs to start from 1...")
    df_clusters['cluster'] = df_clusters['cluster'] + 1
    
    nc_files = sorted(glob.glob(os.path.join(nc_dir, "processed_*.nc")))
    ds = xr.open_mfdataset(nc_files, chunks={'time': 256}, parallel=False)
    
    if 'expver' in ds.dims:
        ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))

    return ds, df_clusters

def plot_temporal_consistency_optimized(ds, df_clusters, target_col='wind_speed', n_samples=SAMPLE_LINES):
    """仅保留时序一致性图"""
    print("Step 2: Plotting Temporal Consistency...")
    
    if target_col == 'wind_speed' and 'wind_speed' not in ds:
        ds['wind_speed'] = np.sqrt(ds.u10**2 + ds.v10**2)
    
    # 计算月平均值以平滑曲线
    ds_monthly = ds[target_col].resample(time='1M').mean().load()
    unique_clusters = sorted(df_clusters['cluster'].unique())
    n_clusters = len(unique_clusters)
    cols = 3
    rows = int(np.ceil(n_clusters / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4.2), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, cluster_id in enumerate(unique_clusters):
        ax = axes[idx]
        grids_in_cluster = df_clusters[df_clusters['cluster'] == cluster_id]
        
        if grids_in_cluster.empty: continue
            
        cluster_lats = xr.DataArray(grids_in_cluster['latitude'].values, dims='points')
        cluster_lons = xr.DataArray(grids_in_cluster['longitude'].values, dims='points')
        
        try:
            # 提取该聚类所有格点的数据
            data_all_points = ds_monthly.sel(
                latitude=cluster_lats, longitude=cluster_lons, method='nearest'
            )
            
            # 维度对齐处理
            if data_all_points.shape[0] != data_all_points.time.size:
                 data_all_points = data_all_points.T

            # A. 绘制个体采样线 (彩色背景，展示多样性)
            current_samples = min(n_samples, data_all_points.shape[1]) 
            sample_indices = np.random.choice(
                data_all_points.shape[1], size=current_samples, replace=False
            )
            subset = data_all_points.isel(points=sample_indices)

            colors = plt.cm.jet(np.linspace(0, 1, current_samples))
            ax.set_prop_cycle(color=colors)
            ax.plot(subset.time, subset, linewidth=0.6, alpha=0.3, zorder=1)
            
            # B. 绘制区域平均线 (黑色前景，展示主趋势)
            spatial_mean = data_all_points.mean(dim='points')
            ax.plot(spatial_mean.time, spatial_mean, color='black', linewidth=2.5, label='Spatial Mean', zorder=10)
            
            # C. 统计信息
            mean_val = float(spatial_mean.mean())
            std_val = float(spatial_mean.std())
            stats_text = f"Mean={mean_val:.2f}\nStd={std_val:.2f}"
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=16,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='lightgray'),
                    zorder=11)

            ax.set_title(f'Cluster {int(cluster_id)}', fontsize=20, fontweight='bold')
            ax.grid(True, linestyle=':', alpha=0.4)
            
            # D. 仅在第一个子图中设置图例，字体调大
            if idx == 0:
                custom_lines = [
                    Line2D([0], [0], color='black', lw=2.5),
                    Line2D([0], [0], color='tab:blue', lw=1, alpha=0.6)
                ]
                ax.legend([custom_lines[0], custom_lines[1]], 
                          ['Spatial Mean', 'Individuals'], 
                          loc='upper left', 
                          fontsize=16, frameon=True)

        except Exception as e:
            print(f"Error in Cluster {cluster_id}: {e}")

    # 移除多余的空子图
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    save_path = os.path.join(OUTPUT_DIR, 'cluster_temporal_consistency_only.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved temporal consistency plot: {save_path}")

def main():
    ds, df_clusters = load_data(NC_DIR, CLUSTER_CSV)
    # 移除了对 plot_grid_quivers_styled 的调用
    plot_temporal_consistency_optimized(ds, df_clusters)
    print("Task completed.")

if __name__ == "__main__":
    main()