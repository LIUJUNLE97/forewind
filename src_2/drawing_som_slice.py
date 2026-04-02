import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import xarray as xr
import os
import glob

# --- 路径配置 ---
SOM_MAP_PATH = '/mimer/NOBACKUP/groups/kthmech/junle/guandi_wind_project/data_2/som/som_map.npy'
ORIGINAL_NC_DIR = '/mimer/NOBACKUP/groups/kthmech/junle/guandi_wind_project/data_2/processed'
OUTPUT_FIG_PATH = '/mimer/NOBACKUP/groups/kthmech/junle/guandi_wind_project/data_2/som/locations_cluster_distribution.png'

LOCATIONS = {
    "shanghai": (31.23, 121.47), "chengdu": (30.67, 104.06),
    "hohhot": (40.84, 111.75), "qingdao": (36.06, 120.38),
    "guangzhou": (23.13, 113.26), "urumqi": (43.82, 87.62),
    "lhasa": (29.65, 91.11), "golmud": (36.40, 94.90),
    "kunming": (25.04, 102.71), "wuhan": (30.59, 114.30),
    "fuzhou": (26.08, 119.30), "haikou": (20.03, 110.32),
}

PATCH_SIZE = 64

def visualize_location_clusters():
    # 1. 加载数据与元数据
    som_map = np.load(SOM_MAP_PATH)
    full_h, full_w = som_map.shape
    
    sample_nc = sorted(glob.glob(os.path.join(ORIGINAL_NC_DIR, '*.nc')))[0]
    with xr.open_dataset(sample_nc) as ds:
        lats = ds.latitude.values
        lons = ds.longitude.values

    # 2. 配置配色方案 (Morandi Colors)
    morandi_colors = [
        '#8FB1CC', '#E6B9C0', '#A8C8A6', '#E8D595', 
        '#C6B4CE', '#D4C4B7', '#D8A49B', '#7B8D8E', '#9DAD7F'
    ]
    n_classes = 9
    cmap = mcolors.ListedColormap(morandi_colors[:n_classes])
    norm = mcolors.BoundaryNorm(np.arange(n_classes + 1) - 0.5, n_classes)

    # 3. 开始绘图
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()

    for i, (name, (lat, lon)) in enumerate(LOCATIONS.items()):
        # 遵循 Dataset 的切片逻辑
        lat_idx = np.argmin(np.abs(lats - lat))
        lon_idx = np.argmin(np.abs(lons - lon))
        
        y_s = np.clip(lat_idx - PATCH_SIZE // 2, 0, full_h - PATCH_SIZE)
        x_s = np.clip(lon_idx - PATCH_SIZE // 2, 0, full_w - PATCH_SIZE)
        
        # 提取 SOM 切片
        som_patch = som_map[y_s:y_s+PATCH_SIZE, x_s:x_s+PATCH_SIZE]
        
        # 绘图
        ax = axes[i]
        im = ax.imshow(som_patch, cmap=cmap, norm=norm, interpolation='nearest')
        ax.set_title(f"{name.capitalize()}\n({lat}N, {lon}E)", fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # 在中心标出一个点，代表实际城市位置
        ax.scatter(PATCH_SIZE//2, PATCH_SIZE//2, c='red', marker='+', s=100, label='Center')

    # 添加整体色条
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=range(n_classes))
    cbar.set_label('SOM Cluster ID', fontsize=12, fontweight='bold')

    plt.suptitle(f"Topographic Regime Distribution (Size: {PATCH_SIZE}x{PATCH_SIZE})", 
                 fontsize=22, fontweight='bold', y=0.95)
    
    plt.savefig(OUTPUT_FIG_PATH, bbox_inches='tight', dpi=300)
    print(f"可视化结果已保存至: {OUTPUT_FIG_PATH}")

if __name__ == "__main__":
    visualize_location_clusters()