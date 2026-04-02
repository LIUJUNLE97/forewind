import matplotlib
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import numpy as np
import os

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'stix' # 数学公式字体也匹配
plt.rcParams['font.size'] = 12

SOM_RESULT_PATH = '/mimer/NOBACKUP/groups/kthmech/junle/guandi_wind_project/data_2/som/som_map.npy'
LAT_LON_PATH = '/mimer/NOBACKUP/groups/kthmech/junle/guandi_wind_project/data_2/lat_lon_grid.npz'
OUTPUT_DIR = '/mimer/NOBACKUP/groups/kthmech/junle/guandi_wind_project/som_vis'
EXTENT = [70, 140, 15, 55]

# 6 个典型城市的坐标
CITY_COORDS = {
    "Shanghai":  [121.47, 31.23],
    "Chengdu":   [104.06, 30.67],
    "Lhasa":     [91.11, 29.65], 
    "Hohhot":    [111.75, 40.84],
    "Urumqi":    [87.62, 43.82], 
    "Guangzhou": [113.26, 23.13] 
}

# 完整候选城市列表 (Cluster ID: [List of Cities])
# 每个城市包含 [Lon, Lat]
CANDIDATE_CITIES = {
    "C1": {
        "Shanghai": [121.47, 31.23],
        "Qingdao":  [120.38, 36.06],
        "Fuzhou":   [119.30, 26.08]
    },
    "C2": {
        "Guangzhou": [113.26, 23.13],
        "Nanning":   [108.37, 22.82],
        "Haikou":    [110.32, 20.03]
    },
    "C3": {
        "Wuhan":    [114.30, 30.59],
        "Changsha": [112.93, 28.23],
        "Nanchang": [115.85, 28.68]
    },
    "C4": {
        "Hohhot":   [111.75, 40.84],
        "Harbin":   [126.53, 45.80],
        "Yinchuan": [106.23, 38.47]
    },
    "C5": {
        "Chengdu":   [104.06, 30.67],
        "Chongqing": [106.55, 29.56]
    },
    "C6": {
        "Urumqi":  [87.62, 43.82],
        "Hami":    [93.52, 42.82],
        "Kashgar": [75.99, 39.47]
    },
    "C7": {
        "Lhasa":    [91.11, 29.65],
        "Shigatse": [88.88, 29.27],
        "Nyingchi": [94.36, 29.64]
    },
    "C8": {
        "Golmud":   [94.90, 36.40],
        "Delingha": [97.37, 37.12]
    },
    "C9": {
        "Kunming": [102.71, 25.04],
        "Guiyang": [106.63, 26.65]
    }
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================
# 2. 加载数据
# ===========================
print("Loading result data...")
coords = np.load(LAT_LON_PATH)
lons_raw = coords['lons']
lats_raw = coords['lats']
som_clusters = np.load(SOM_RESULT_PATH)

if lons_raw.ndim == 1:
    lons_grid, lats_grid = np.meshgrid(lons_raw, lats_raw)
else:
    lons_grid, lats_grid = lons_raw, lats_raw

unique_labels = np.unique(som_clusters)
n_classes = len(unique_labels)

# ===========================
# 3. 辅助函数：绘制城市标记 (SCI 风格)
# ===========================
def add_city_markers_sci(ax, cities, transform):
    """
    JCLI 风格站点标注：
    - Marker: 黑底白圈 (高对比度，不干扰背景色)
    - Text: Times New Roman, 无粗暴描边
    """
    print(">>> Adding scientific station markers...")

    for name, (lon, lat) in cities.items():
        # --- A. 绘制标记点 (站点风格) ---
        # 使用 scatter 绘制，zorder 设置高一点确保在最上层
        # s=50 控制大小, c='k' (黑色填充), edgecolors='w' (白色边缘), linewidth=1.2 (边缘宽度)
        ax.scatter(lon, lat, s=60, c='black', edgecolors='white', linewidth=1.5, 
                   transform=transform, zorder=100, marker='o')

        # --- B. 精细调整文字位置 ---
        x_off, y_off = 1.5, 0.0
        ha, va = 'left', 'center'

        # 微调逻辑：避开拥挤区域
        if name == "Shanghai":
            x_off, y_off = -1.8, 0.0
            ha = 'right'
        elif name == "Guangzhou":
            x_off, y_off = 1.5, 0.0
        elif name == "Hohhot":
            x_off, y_off = 0.0, 1.2
            ha = 'center'; va = 'bottom'
        elif name == "Lhasa":
            x_off, y_off = 0.0, -1.5
            ha = 'center'; va = 'top'
        elif name == "Urumqi":
            x_off, y_off = 1.5, 0.0
        elif name == "Chengdu":
            x_off, y_off = 1.5, 0.0

        # --- C. 绘制文字 (纯净学术风) ---
        # 不使用粗描边，而是利用高对比度的黑色字体。
        # 如果背景色太深导致看不清，可以开启极其微弱的 halo (alpha=0.3)
        t = ax.text(lon + x_off, lat + y_off, name, 
                    transform=transform, zorder=101,
                    fontsize=11, 
                    color='black',
                    fontweight='normal', # 学术论文通常不用 bold，除非强调
                    ha=ha, va=va)
        
        # 可选：极淡的白色光晕，仅用于防止文字融入深色背景，模仿论文印刷效果
        # 如果不需要光晕，注释掉下面两行
        import matplotlib.patheffects as pe
        t.set_path_effects([pe.withStroke(linewidth=1.5, foreground='white', alpha=0.7)])

# ===========================
# 4. 核心绘图逻辑
# ===========================
def plot_som_morandi_china_highlight(lons, lats, clusters, n_classes, title, filename):
    print(f"Plotting {filename}...")
    
    # 画布设置：长宽比适中，DPI 300 满足出版要求
    fig = plt.figure(figsize=(10, 8), dpi=300) 
    proj = ccrs.PlateCarree()
    ax = plt.axes(projection=proj)
    ax.set_extent(EXTENT, crs=proj)

    # -------------------------------------------------------
    # A. 基础地理特征 (线条更细，颜色更深，符合论文印刷)
    # -------------------------------------------------------
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), 
                   linewidth=0.8, edgecolor='black', zorder=5)
    
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), 
                   linestyle=':', linewidth=0.5, edgecolor='gray', zorder=5)

    # -------------------------------------------------------
    # B. 莫兰迪配色 SOM 填色
    # -------------------------------------------------------
    morandi_colors = [
        '#8FB1CC', '#E6B9C0', '#A8C8A6', '#E8D595', '#C6B4CE', 
        '#D4C4B7', '#D8A49B', '#7B8D8E', '#9DAD7F'
    ]
    
    if len(morandi_colors) < n_classes:
        cmap_base = plt.get_cmap('Set3', n_classes)
    else:
        cmap_base = mcolors.ListedColormap(morandi_colors[:n_classes])

    boundaries = np.arange(n_classes + 1) - 0.5
    norm = mcolors.BoundaryNorm(boundaries, n_classes)

    cf = ax.pcolormesh(lons, lats, clusters, 
                       cmap=cmap_base, norm=norm, alpha=0.9, 
                       transform=proj, shading='nearest', zorder=2) 

    # -------------------------------------------------------
    # C. 中国国境线 (精致加粗)
    # -------------------------------------------------------
    print("Highlighting China borders...")
    try:
        shpfilename = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries')
        reader = shpreader.Reader(shpfilename)
        countries = reader.records()
        
        china_geoms = []
        for country in countries:
            name = country.attributes['ADMIN']
            if name in ['China', 'Taiwan']: 
                china_geoms.append(country.geometry)
        
        # 使用深灰色而不是纯黑，显得更有质感
        ax.add_geometries(china_geoms, ccrs.PlateCarree(), 
                          facecolor='none', edgecolor='#333333',
                          linewidth=1.2, linestyle='-', zorder=6)
    except Exception as e:
        print(f"Warning: Could not load shapefiles. {e}")

    # -------------------------------------------------------
    # D. 添加 JCLI 风格的城市标记
    # -------------------------------------------------------
    add_city_markers_sci(ax, CITY_COORDS, transform=proj)

    # -------------------------------------------------------
    # E. 装饰 (刻度向内，字体 Times New Roman)
    # -------------------------------------------------------
    # 使用 Matplotlib 标准 ticks 而不是 Gridlines draw_labels，控制力更强
    # 这里为了简便仍用 gridlines，但调整样式
    gl = ax.gridlines(crs=proj, draw_labels=True,
                      linewidth=0.5, color='gray', alpha=0.3, linestyle=':')
    gl.top_labels = False  
    gl.right_labels = False
    gl.xlabel_style = {'size': 11, 'color': 'black', 'family': 'serif'}
    gl.ylabel_style = {'size': 11, 'color': 'black', 'family': 'serif'}

    # Colorbar调整：更细，放在底部
    cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', 
                        pad=0.08, aspect=30, shrink=0.8,
                        ticks=np.arange(n_classes)) 
    
    cbar.set_label('SOM Cluster Regime', fontsize=12, fontname='serif')
    cbar.set_ticklabels([f'C{i+1}' for i in range(n_classes)]) # 简写 C1, C2... 更整洁
    cbar.ax.tick_params(labelsize=10)

    plt.title(title, fontsize=14, fontname='serif', weight='bold', pad=15)

    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Saved to: {save_path}")

# ===========================
# 5. 执行
# ===========================
if __name__ == "__main__":
    plot_som_morandi_china_highlight(
        lons_grid, 
        lats_grid, 
        som_clusters, 
        n_classes=n_classes, 
        title="Spatial Distribution of Wind Regimes and Representative Sites",
        filename="som_clusters_sci_style.png"
    )