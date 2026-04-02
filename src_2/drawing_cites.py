import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import glob
import xarray as xr
from matplotlib.colors import ListedColormap, BoundaryNorm

# ================= 配置区域 =================
BASE_DIR = '/mimer/NOBACKUP/groups/kthmech/junle/guandi_wind_project/data_2'
SOM_PATH = os.path.join(BASE_DIR, 'som/som_map.npy') # 请确认路径
NC_DIR = os.path.join(BASE_DIR, 'processed') # 用于读取经纬度

# 采样配置
PATCH_SIZE = 64  # 训练时的切片大小

# 城市列表 (保持一致)
CANDIDATE_CITIES = {
    "Shanghai": [31.23, 121.47],
    "Guangzhou": [23.13, 113.26],
    "Wuhan":     [30.59, 114.30],
    "Hohhot":    [40.84, 111.75],
    "Chengdu":   [30.67, 104.06],
    "Urumqi":    [43.82, 87.62],
    "Lhasa":     [29.65, 91.11],
    "Golmud":    [36.40, 94.90],
    "Kunming":   [25.04, 102.71]
}

# 莫兰迪色系 (C1-C9)
COLORS = [
    '#8FB1CC', '#E6B9C0', '#A8C8A6', '#E8D595', '#C6B4CE', 
    '#D4C4B7', '#D8A49B', '#7B8D8E', '#9DAD7F'
]

# ================= 绘图逻辑 =================

def load_grid_info(nc_dir):
    """读取网格经纬度信息"""
    nc_files = sorted(glob.glob(os.path.join(nc_dir, '*.nc')))
    if not nc_files:
        raise FileNotFoundError("No NC files found for grid info.")
        
    with xr.open_dataset(nc_files[0]) as ds:
        lats = ds.latitude.values if 'latitude' in ds else ds.lat.values
        lons = ds.longitude.values if 'longitude' in ds else ds.lon.values
    return lats, lons

def plot_som_with_sampling():
    # 1. 加载数据
    print("Loading SOM map and Grid...")
    try:
        som_map = np.load(SOM_PATH)
        lats, lons = load_grid_info(NC_DIR)
    except Exception as e:
        print(f"Error loading data: {e}")
        print("请检查 SOM_PATH 和 NC_DIR 是否正确。")
        return

    # 2. 准备画布
    fig = plt.figure(figsize=(12, 8), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # 设置地图范围 (中国区域)
    ax.set_extent([73, 136, 17, 55], crs=ccrs.PlateCarree())
    
    # 绘制地理底图
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='gray')
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, color='gray')
    ax.add_feature(cfeature.LAND, facecolor='#FCFCFC')

    # 3. 绘制 SOM 背景
    # 创建离散色盘
    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(np.arange(10)-0.5, 9)
    
    print("Plotting SOM background...")
    mesh = ax.pcolormesh(lons, lats, som_map, cmap=cmap, norm=norm, 
                         alpha=0.6, transform=ccrs.PlateCarree(), shading='nearest')

    # 4. 绘制采样点和框
    print("Plotting sampling boxes...")
    
    # 获取网格分辨率 (用于估算框的大小，或直接用索引)
    # 这里我们用精确的索引反查法来画框
    
    for city_name, (target_lat, target_lon) in CANDIDATE_CITIES.items():
        # 找到最近邻索引 (Center)
        lat_idx = np.argmin(np.abs(lats - target_lat))
        lon_idx = np.argmin(np.abs(lons - target_lon))
        
        # 计算采样框的边界索引 (Dataset 逻辑: y = center - patch/2)
        # 注意：要处理边界溢出
        y_start = max(0, lat_idx - PATCH_SIZE // 2)
        x_start = max(0, lon_idx - PATCH_SIZE // 2)
        y_end = min(len(lats) - 1, y_start + PATCH_SIZE)
        x_end = min(len(lons) - 1, x_start + PATCH_SIZE)
        
        # 根据索引反查实际经纬度边界
        # 假设 lats 是从南到北或北到南，这里取 min/max 确保框正确
        box_lats = lats[y_start:y_end]
        box_lons = lons[x_start:x_end]
        
        min_lat, max_lat = np.min(box_lats), np.max(box_lats)
        min_lon, max_lon = np.min(box_lons), np.max(box_lons)
        
        # 绘制矩形框
        # xy 是左下角
        width = max_lon - min_lon
        height = max_lat - min_lat
        
        rect = mpatches.Rectangle((min_lon, min_lat), width, height, 
                                  linewidth=1.2, edgecolor='black', facecolor='none', 
                                  linestyle='--', transform=ccrs.PlateCarree(), zorder=5)
        ax.add_patch(rect)
        
        # 绘制中心点
        ax.scatter(target_lon, target_lat, s=30, c='black', marker='o', 
                   transform=ccrs.PlateCarree(), zorder=6)
        
        # 添加标签
        ax.text(target_lon + 1.5, target_lat + 0.5, city_name, 
                transform=ccrs.PlateCarree(), fontsize=9, fontweight='bold', 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1), zorder=6)

    # 5. 添加 Colorbar
    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7, aspect=30)
    cbar.set_label('Wind Regimes (SOM Clusters)')
    cbar.set_ticks(np.arange(9))
    cbar.set_ticklabels([f'C{i+1}' for i in range(9)])

    # 6. 保存
    save_path = 'som_sampling_map.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"✅ SOM Sampling Map saved to {save_path}")

if __name__ == "__main__":
    plot_som_with_sampling()