import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
try:
    from cnmaps import get_adm_maps
    _HAS_CNMAPS = True
except Exception:
    _HAS_CNMAPS = False
import numpy as np
import matplotlib.patheffects as path_effects
import os

# =======================================================
# 1. 配置与路径
# =======================================================
PROJECT_ROOT = os.environ.get('WIND_PROJECT_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOM_RESULT_PATH = os.environ.get('WIND_SOM_MAP', os.path.join(PROJECT_ROOT, 'data', 'metadata', 'som_map.npy'))
LAT_LON_PATH = os.environ.get('WIND_LAT_LON_GRID', os.path.join(PROJECT_ROOT, 'data', 'metadata', 'lat_lon_grid.npz'))
OUTPUT_DIR = os.environ.get('WIND_FIG_DIR', os.path.join(PROJECT_ROOT, 'fig'))
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
    # 优先使用 cnmaps（若安装），否则回退到 Natural Earth shapefiles
    has_cnmaps = _HAS_CNMAPS
    china_geometry = []
    china_province_geoms = []
    taiwan_geoms = []
    china_admin_taiwan_geoms = []
    if has_cnmaps:
        try:
            from shapely.geometry import shape
            china_map = get_adm_maps(level='国', record='first')
            if isinstance(china_map, dict) and 'geometry' in china_map:
                g = china_map['geometry']
                china_geometry = [shape(g.__geo_interface__) if hasattr(g, '__geo_interface__') else g]
            else:
                china_geometry = []
                for r in china_map:
                    g = r.get('geometry') if isinstance(r, dict) else getattr(r, 'geometry', r)
                    if g is None:
                        continue
                    china_geometry.append(shape(g.__geo_interface__) if hasattr(g, '__geo_interface__') else g)

            provs = get_adm_maps(level='省')
            for rec in provs:
                if isinstance(rec, dict):
                    geom = rec.get('geometry')
                    name = rec.get('name', '')
                else:
                    geom = getattr(rec, 'geometry', rec)
                    name = getattr(rec, 'name', '') or getattr(rec, 'NAME', '') or getattr(rec, 'ADM_NAME', '')
                if geom is None:
                    continue
                geom_shp = shape(geom.__geo_interface__) if hasattr(geom, '__geo_interface__') else geom
                if '台' in str(name) or 'Taiwan' in str(name) or '臺' in str(name):
                    taiwan_geoms.append(geom_shp)
                else:
                    china_province_geoms.append(geom_shp)

            # 从国家级几何中提取与台湾区域相交的几何（以防省级数据缺失）
            from shapely.geometry import box
            taiwan_bbox = box(119, 20, 123, 26)
            china_admin_taiwan_geoms = [g for g in china_geometry if getattr(g, 'bounds', None) and (g.intersects(taiwan_bbox) if hasattr(g, 'intersects') else False)]
        except Exception:
            has_cnmaps = False

    if not has_cnmaps:
        shpfilename = shpreader.natural_earth(resolution='50m', category='cultural', name='admin_0_countries')
        reader = shpreader.Reader(shpfilename)
        china_geometry = [rec.geometry for rec in reader.records() if rec.attributes.get('ADMIN') == 'China']
        from shapely.geometry import box
        taiwan_bbox = box(119, 20, 123, 26)
        china_admin_taiwan_geoms = [g for g in china_geometry if getattr(g, 'bounds', None) and (g.intersects(taiwan_bbox) if hasattr(g, 'intersects') else False)]

        shpfilename_province = shpreader.natural_earth(resolution='50m', category='cultural', name='admin_1_states_provinces')
        reader_province = shpreader.Reader(shpfilename_province)
        for rec in reader_province.records():
            admin = rec.attributes.get('admin')
            sover = rec.attributes.get('sovereignt')
            name = rec.attributes.get('name')
            if name and ('台' in name or 'Tai' in name or '臺' in name):
                taiwan_geoms.append(rec.geometry)
            elif admin == 'China' or sover == 'China':
                china_province_geoms.append(rec.geometry)

    # --- C. 主图绘制 ---
    # 定义主图位置 [left, bottom, width, height]
    # 缩窄主图以为右侧垂直 colorbar 留出空间
    main_rect = [0.1, 0.12, 0.78, 0.85]
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

    # --- 城市标注（12个） ---
    cities = {
        'Shanghai': (121.47, 31.23),
        'Qingdao': (120.38, 36.06),
        'Fuzhou': (119.30, 26.08),
        'Guangzhou': (113.26, 23.13),
        'Haikou': (110.32, 20.03),
        'Chengdu': (104.06, 30.67),
        'Wuhan': (114.30, 30.59),
        'Lhasa': (91.11, 29.65),
        'Golmud': (94.90, 36.40),
        'Kunming': (102.71, 25.04),
        'Hohhot': (111.75, 40.84),
        'Urumqi': (87.62, 43.82),
    }
    # 每个城市的标签偏移 (dx, dy) 以度为单位，覆盖常见重叠情形
    label_offsets = {
        'Shanghai': (0.8, 0.6),
        'Qingdao': (0.6, -0.4),
        'Fuzhou': (0.6, -0.3),
        'Guangzhou': (0.6, -0.4),
        'Haikou': (0.6, -0.6),
        'Chengdu': (0.5, 0.3),
        'Wuhan': (0.6, -0.3),
        'Lhasa': (0.6, -0.4),
        'Golmud': (0.6, 0.2),
        'Kunming': (0.5, -0.3),
        'Hohhot': (0.6, 0.3),
        'Urumqi': (0.6, 0.3),
    }
    for name, (lon, lat) in cities.items():
        ax.plot(lon, lat, marker='o', markersize=5, color='black', markeredgecolor='white', markeredgewidth=0.6, transform=proj, zorder=11)
        dx, dy = label_offsets.get(name, (0.6, 0.3))
        txt = ax.text(lon + dx, lat + dy, name, fontsize=12, fontweight='normal', transform=proj, zorder=12,
                  bbox=dict(facecolor='white', alpha=0.7, pad=0.2, edgecolor='none'))

    # 基础地理特征
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5, edgecolor='#666666', zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle='--', linewidth=0.4, edgecolor='gray', zorder=2)

    # 绘制中国标准边界（解决此前的 UnboundLocalError）
    ax.add_geometries(china_geometry, proj, facecolor='none', edgecolor='#222222', linewidth=1.1, zorder=5)
    ax.add_geometries(china_province_geoms, proj, facecolor='none', edgecolor='#555555', linewidth=0.6, zorder=6)
    # 台湾使用与国界相同的线宽和颜色，以便突出显示
    # 若 admin_1 中没有台湾单独记录，则用国家几何中与台湾相交的部分覆盖绘制
    if taiwan_geoms:
        ax.add_geometries(taiwan_geoms, proj, facecolor='none', edgecolor='#222222', linewidth=1.1, zorder=7)
    elif china_admin_taiwan_geoms:
        ax.add_geometries(china_admin_taiwan_geoms, proj, facecolor='none', edgecolor='#222222', linewidth=1.1, zorder=7)

    # 网格线 (取消顶部和右侧标签)，并增大经纬度字体
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.2, linestyle='--')
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = {'size': 14}
    gl.ylabel_style = {'size': 14}

    # --- D. 南海缩略图 (右下角对齐) ---
    # 计算右下角位置：
    # 主图右边缘是 0.1 + 0.8 = 0.9，底部是 0.12
    # 缩略图宽度 0.15，则起始左坐标为 0.9 - 0.15 = 0.75
    # 提高右下角缩略图以避免遮挡刻度，同时略微减小高度
    sub_rect = [0.76, 0.28, 0.12, 0.18]
    ax_sub = fig.add_axes(sub_rect, projection=proj)
    ax_sub.set_extent([106, 122, 2, 25], crs=proj) 
    
    # 同步绘制填色与国界
    ax_sub.pcolormesh(lons, lats, clusters, cmap=cmap_base, norm=norm, 
                      alpha=0.9, transform=proj, shading='nearest', zorder=1)
    ax_sub.add_geometries(china_geometry, proj, facecolor='none', edgecolor='#222222', linewidth=0.8, zorder=3)
    ax_sub.add_geometries(china_province_geoms, proj, facecolor='none', edgecolor='#555555', linewidth=0.4, zorder=4)
    if taiwan_geoms:
        ax_sub.add_geometries(taiwan_geoms, proj, facecolor='none', edgecolor='#222222', linewidth=0.8, zorder=5)
    elif china_admin_taiwan_geoms:
        ax_sub.add_geometries(china_admin_taiwan_geoms, proj, facecolor='none', edgecolor='#222222', linewidth=0.8, zorder=5)
    ax_sub.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.3, edgecolor='#666666', zorder=2)

    # --- E. Colorbar (右侧竖排) ---
    # 将 colorbar 精确对齐到主图的 axes bbox（避免微小偏差），并去掉色块间分隔线
    bbox = ax.get_position()
    cbar_ax = fig.add_axes([bbox.x1 + 0.01, bbox.y0, 0.02, bbox.height])
    # 将 ticks 放在每个色块的中间（整数索引），并隐藏刻度短线
    cbar = plt.colorbar(cf, cax=cbar_ax, orientation='vertical', ticks=np.arange(n_classes))
    cbar.set_label('SOM Cluster Label', fontsize=12, labelpad=8)
    cbar.set_ticklabels([f'Cluster {i+1}' for i in range(n_classes)])
    cbar.ax.yaxis.set_tick_params(labelsize=12, length=0)
    # 移除色块之间的分隔线（将边缘色设为 face）
    try:
        cbar.solids.set_edgecolor('face')
        cbar.solids.set_linewidth(0.0)
    except Exception:
        pass
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
