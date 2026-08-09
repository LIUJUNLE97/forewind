import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import xarray as xr
import os
import glob
from pathlib import Path

# --- 路径配置 ---
PROJECT_ROOT = Path(os.environ.get('WIND_PROJECT_ROOT', Path(__file__).resolve().parents[1]))
SOM_MAP_PATH = os.environ.get('WIND_SOM_MAP', str(PROJECT_ROOT / 'data' / 'metadata' / 'som_map.npy'))
ORIGINAL_NC_DIR = os.environ.get('WIND_PROCESSED_DIR', str(PROJECT_ROOT / 'processed'))
OUTPUT_FIG_PATH = os.environ.get('WIND_SOM_SLICE_FIG', str(PROJECT_ROOT / 'fig' / 'som_location_clusters.png'))

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
    fig, axes = plt.subplots(3, 4, figsize=(22, 16))
    axes = axes.flatten()

    # 与脚本二保持严格一致的数据分辨率设定（例如 ERA5 的 0.25 度）
    dx = 0.25
    dy = 0.25

    for i, (name, (lat, lon)) in enumerate(LOCATIONS.items()):
        # 遵循 Dataset 的切片逻辑
        lat_idx = np.argmin(np.abs(lats - lat))
        lon_idx = np.argmin(np.abs(lons - lon))

        y_s = np.clip(lat_idx - PATCH_SIZE // 2, 0, full_h - PATCH_SIZE)
        x_s = np.clip(lon_idx - PATCH_SIZE // 2, 0, full_w - PATCH_SIZE)

        # 提取 SOM 切片
        som_patch = som_map[y_s:y_s+PATCH_SIZE, x_s:x_s+PATCH_SIZE]

        # --- 🔒 核心修改一：引入与脚本二完全相同的物理坐标系映射 ---
        lon_min = lon - (PATCH_SIZE / 2) * dx
        lon_max = lon + (PATCH_SIZE / 2) * dx
        lat_min = lat - (PATCH_SIZE / 2) * dy
        lat_max = lat + (PATCH_SIZE / 2) * dy
        map_extent = [lon_min, lon_max, lat_min, lat_max]

        ax = axes[i]

        # --- 🔒 核心修改二：对齐渲染视界，防止几何形变与上下颠倒 ---
        # 显式指定 origin='lower'、aspect='equal' 以及物理范围 extent
        im = ax.imshow(som_patch, cmap=cmap, norm=norm, interpolation='nearest',
                       origin='lower', aspect='equal', extent=map_extent)

        ax.set_title(f"{name.capitalize()}\n({lat}°N, {lon}°E)", fontsize=14, fontweight='bold')

        # --- 🔒 核心修改三：规范坐标轴标签，以便直观检验对齐状态 ---
        ax.axis('on')  # 恢复坐标显示以供自检
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.set_xlabel('Longitude (°E)', fontsize=12)
        if i % 4 == 0:
            ax.set_ylabel('Latitude (°N)', fontsize=12)

        # --- 🔒 核心修改四：将测站锚点由像素死代码改为真实物理坐标 ---
        ax.scatter(lon, lat, c='red', marker='+', s=120, linewidths=2, label='Center')

    # 添加整体色条
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=range(n_classes))
    cbar.set_label('SOM Cluster ID', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=11)

    plt.suptitle(f"Topographic Regime Distribution (Geographical Alignment, Size: {PATCH_SIZE}x{PATCH_SIZE})",
                 fontsize=22, fontweight='bold', y=0.96)

    # 调整子图布局呼吸感
    plt.subplots_adjust(wspace=0.25, hspace=0.28)

    plt.savefig(OUTPUT_FIG_PATH, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"安全对齐的可视化结果已保存至: {OUTPUT_FIG_PATH}")

if __name__ == "__main__":
    visualize_location_clusters()
