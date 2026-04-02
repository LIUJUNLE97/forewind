import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import os

# 引入 cnmaps 获取标准国界数据
from cnmaps import get_adm_maps

# =======================================================
# 1. 配置与路径
# =======================================================
SOM_RESULT_PATH = '/mimer/NOBACKUP/groups/kthmech/junle/guandi_wind_project/data_2/som/som_map.npy'
LAT_LON_PATH = '/mimer/NOBACKUP/groups/kthmech/junle/guandi_wind_project/data_2/lat_lon_grid.npz'
OUTPUT_DIR = '/mimer/NOBACKUP/groups/kthmech/junle/guandi_wind_project/som_vis'
EXTENT = [70, 140, 15, 55] 

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =======================================================
# 2. 加载数据
# =======================================================
print("Loading result data...")
coords = np.load(LAT_LON_PATH)
lons_raw, lats_raw = coords['lons'], coords['lats']
som_clusters = np.load(SOM_RESULT_PATH)

if lons_raw.ndim == 1:
    lons_grid, lats_grid = np.meshgrid(lons_raw, lats_raw)
else:
    lons_grid, lats_grid = lons_raw, lats_raw

n_classes = len(np.unique(som_clusters))

# =======================================================
# 3. 核心绘图逻辑
# =======================================================
def plot_som_china_final_layout(lons, lats, clusters, n_classes, filename):
    print(f"Generating map: {filename}...")
    import matplotlib
    matplotlib.use('Agg') # 适用于集群环境
    
    # A. 莫兰迪配色方案
    morandi_colors = [
        '#8FB1CC', '#E6B9C0', '#A8C8A6', '#E8D595', 
        '#C6B4CE', '#D4C4B7', '#D8A49B', '#7B8D8E', '#9DAD7F'
    ]
    if len(morandi_colors) < n_classes:
        cmap_base = plt.get_cmap('Set3', n_classes)
    else:
        cmap_base = mcolors.ListedColormap(morandi_colors[:n_classes])

    norm = mcolors.BoundaryNorm(np.arange(n_classes + 1) - 0.5, n_classes)
    proj = ccrs.PlateCarree()

    # B. 创建画布
    fig = plt.figure(figsize=(12, 9), dpi=300)

    # 获取中国边界几何对象 (避开 draw_maps 的逻辑错误)
    china_map = get_adm_maps(level='国', record='first')
    china_geometry = china_map['geometry']

    # --- C. 主图绘制 ---
    # 定义主图位置 [left, bottom, width, height]
    main_rect = [0.1, 0.12, 0.8, 0.85]
    ax = fig.add_axes(main_rect, projection=proj)
    ax.set_extent(EXTENT, crs=proj)

    # 绘制 SOM 填色
    cf = ax.pcolormesh(lons, lats, clusters, 
                       cmap=cmap_base, 
                       norm=norm,              
                       alpha=0.9, 
                       transform=proj, 
                       shading='nearest', 
                       zorder=1)

    # 基础地理特征
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5, edgecolor='#666666', zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle='--', linewidth=0.4, edgecolor='gray', zorder=2)

    # 绘制中国标准边界（解决此前的 UnboundLocalError）
    ax.add_geometries([china_geometry], proj, facecolor='none', edgecolor='#222222', linewidth=1.1, zorder=5)

    # 网格线 (取消顶部和右侧标签)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.2, linestyle='--')
    gl.top_labels = gl.right_labels = False 

    # --- D. 南海缩略图 (右下角对齐) ---
    # 计算右下角位置：
    # 主图右边缘是 0.1 + 0.8 = 0.9，底部是 0.12
    # 缩略图宽度 0.15，则起始左坐标为 0.9 - 0.15 = 0.75
    sub_rect = [0.75, 0.12, 0.15, 0.22] 
    ax_sub = fig.add_axes(sub_rect, projection=proj)
    ax_sub.set_extent([106, 122, 2, 25], crs=proj) 
    
    # 同步绘制填色与国界
    ax_sub.pcolormesh(lons, lats, clusters, cmap=cmap_base, norm=norm, 
                      alpha=0.9, transform=proj, shading='nearest', zorder=1)
    ax_sub.add_geometries([china_geometry], proj, facecolor='none', edgecolor='#222222', linewidth=0.8, zorder=3)
    ax_sub.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.3, edgecolor='#666666', zorder=2)

    # --- E. Colorbar (底部居中) ---
    # 为了避免和右下角缩略图重叠，将 Colorbar 稍微向左移动或收窄
    cbar_ax = fig.add_axes([0.2, 0.05, 0.4, 0.02])
    cbar = plt.colorbar(cf, cax=cbar_ax, orientation='horizontal', ticks=np.arange(n_classes))
    cbar.set_label('SOM Cluster Label', fontsize=10, labelpad=8)
    cbar.set_ticklabels([f'Cluster {i+1}' for i in range(n_classes)])
    cbar.outline.set_linewidth(0.5)

    # --- F. 保存 ---
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Map saved to: {save_path}")

# =======================================================
# 4. 执行
# =======================================================
if __name__ == "__main__":
    plot_som_china_final_layout(
        lons_grid, 
        lats_grid, 
        som_clusters, 
        n_classes, 
        "som_china_bottom_right.png"
    )