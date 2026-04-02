import xarray as xr
import numpy as np

# 1. 指向你的原始 NetCDF 文件路径 (ERA5 原始数据)
# 请替换为你 Alvis 上的真实路径，比如 data/weather_data_2023.nc
NC_FILE_PATH = '/mimer/NOBACKUP/groups/kthmech/junle/guandi_wind_project/data_2/processed/processed_1999_01.nc'

try:
    # 使用 xarray 打开
    ds = xr.open_dataset(NC_FILE_PATH)
    
    # 2. 提取经纬度 (通常变量名叫 'longitude' 和 'latitude'，或者 'lon' 和 'lat')
    # 打印一下看看变量名对不对
    print("Variables in NC file:", ds.data_vars)
    
    lons = ds.longitude.values
    lats = ds.latitude.values
    
    # 3. 保存为 npz 文件供后续画图使用
    save_path = '/mimer/NOBACKUP/groups/kthmech/junle/guandi_wind_project/data_2/lat_lon_grid.npz'
    np.savez(save_path, lons=lons, lats=lats)
    
    print(f"✅ 成功提取坐标！")
    print(f"经度范围: {lons.min()} 到 {lons.max()}，形状: {lons.shape}")
    print(f"纬度范围: {lats.min()} 到 {lats.max()}，形状: {lats.shape}")
    print(f"文件已保存至: {save_path}")

except Exception as e:
    print(f"❌ 提取失败: {e}")