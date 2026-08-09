import xarray as xr
import numpy as np
import pyvista as pv
import os

# ================= 配置区 =================
PROJECT_ROOT = os.environ.get('WIND_PROJECT_ROOT', os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.environ.get('WIND_PROCESSED_DIR', os.path.join(PROJECT_ROOT, 'processed'))
INPUT_FILE = os.environ.get('WIND_REFERENCE_NC', os.path.join(PROCESSED_DIR, 'processed_2021.nc'))
OUTPUT_VTS = 'terrain_model_final.vts'

# 检查文件
if not os.path.exists(INPUT_FILE):
    import glob
    files = glob.glob(os.path.join(PROCESSED_DIR, '*.nc'))
    INPUT_FILE = files[0] if files else exit("错误：未找到 .nc 文件")

print(f"开始处理文件: {os.path.basename(INPUT_FILE)}")

# ================= 1. 数据提取与计算 =================
ds = xr.open_dataset(INPUT_FILE)

# 转换位势为海拔高度 (m)
# $$Elevation = \frac{\Phi}{g}$$
g = 9.80665
if 'time' in ds['geopotential'].dims:
    elevation = ds['geopotential'].isel(time=0).values / g
else:
    elevation = ds['geopotential'].values / g

# 计算粗糙度 (3x3 标准差)
# 暂时转回 xarray 处理滚动窗口
da_ele = xr.DataArray(elevation, dims=['lat', 'lon'])
roughness = da_ele.rolling(lat=3, lon=3, center=True).std().fillna(0).values

# 获取坐标
lats = ds['latitude'].values
lons = ds['longitude'].values
ny, nx = elevation.shape

# ================= 2. 构建三维几何 =================
print(f"数据形状: {ny}x{nx}, 正在构建 3D 网格...")

# 使用 indexing='ij' 确保 xx[i,j] 对应 lons[j], yy[i,j] 对应 lats[i]
# 此时生成的网格矩阵形状与 elevation 矩阵完全一致
yy, xx = np.meshgrid(lats, lons, indexing='ij')

# 解决比例问题：
# 地理单位是“度”，高度单位是“米”。1度约为111,000米。
# 为使地形起伏可见，我们需要应用垂直夸张系数。
# 目标：让 6000m 的高度在视觉上约为经度跨度 (70度) 的 10%
visual_scale = (lons.max() - lons.min()) * 0.1 / (np.nanmax(elevation) - np.nanmin(elevation))
zz = elevation * visual_scale

# ================= 3. 创建 PyVista 对象 =================
# 将坐标堆叠为点集 (N, 3)
# 注意：PyVista 需要 [nx, ny, 1] 顺序的 dimensions，数据需配合转置
grid = pv.StructuredGrid()
# 将数据拍平并组合成 [X, Y, Z]
points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
grid.points = points
grid.dimensions = [nx, ny, 1]

# 绑定属性数据（用于颜色映射，不缩放）
grid.point_data["Elevation_m"] = elevation.ravel()
grid.point_data["Roughness"] = roughness.ravel()

# ================= 4. 输出与导出 =================
grid.save(OUTPUT_VTS)

print("-" * 30)
print(f"✅ 处理成功！")
print(f"垂直缩放系数: {visual_scale:.6f} (已将高度压缩至地理跨度比例)")
print(f"输出文件: {os.getcwd()}/{OUTPUT_VTS}")
print(f"提示：在 ParaView 中打开后，请使用 'Elevation_m' 进行染色。")
print("-" * 30)

# 可选：如果在支持 GUI 的环境，取消下行注释可直接看图
# grid.plot(scalars="Elevation_m", cmap="terrain", show_grid=True)
