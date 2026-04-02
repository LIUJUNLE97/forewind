import xarray as xr
import numpy as np
import argparse
import os
import pandas as pd

def inspect_era5_nc(file_path):
    """
    详细检查 ERA5 NetCDF (.nc) 文件的元数据、尺度和变量信息。
    """
    if not os.path.exists(file_path):
        print(f"❌ 错误: 文件不存在 -> {file_path}")
        return

    print(f"\n{'='*60}")
    print(f"📂 ERA5 数据体检报告: {os.path.basename(file_path)}")
    print(f"{'='*60}")

    try:
        # 使用 xarray 打开数据集
        ds = xr.open_dataset(file_path)
    except Exception as e:
        print(f"❌ 无法打开文件: {e}")
        return

    # ----------------------------------------------------
    # 0. 关键：自动探测时间维度名
    # ----------------------------------------------------
    time_dim = None
    if 'time' in ds.coords:
        time_dim = 'time'
    elif 'valid_time' in ds.coords:
        time_dim = 'valid_time'
    
    # ----------------------------------------------------
    # 1. 维度与尺度 (Dimensions & Scale)
    # ----------------------------------------------------
    print(f"\n[1] 📏 维度与尺度 (Dimensions & Scale)")
    print(f"-"*40)
    
    # --- 空间分辨率 (Spatial Resolution) ---
    if 'latitude' in ds.coords and 'longitude' in ds.coords:
        lat = ds.coords['latitude'].values
        lon = ds.coords['longitude'].values
        
        lat_res = abs(lat[1] - lat[0]) if len(lat) > 1 else 0
        lon_res = abs(lon[1] - lon[0]) if len(lon) > 1 else 0
        
        print(f"   * 纬度范围 (Lat): {lat.min():.2f} ~ {lat.max():.2f} (Grid Size: {len(lat)})")
        print(f"   * 经度范围 (Lon): {lon.min():.2f} ~ {lon.max():.2f} (Grid Size: {len(lon)})")
        print(f"   * 空间分辨率: {lat_res:.2f}° x {lon_res:.2f}°")
        
        if lat.max() >= 53 and lat.min() <= 18 and lon.min() <= 73 and lon.max() >= 135:
            print("     (✅ 覆盖中国全境)")
        else:
            print("     (⚠️ 仅部分区域)")
            
    # --- 时间尺度 (Temporal Scale) ---
    if time_dim:
        time_vals = ds.coords[time_dim].values
        # 尝试转换为 pandas datetime 以便阅读
        try:
            time_pd = pd.to_datetime(time_vals)
            print(f"\n   * 时间维度名: '{time_dim}'")
            print(f"   * 时间范围: {time_pd.min()} ~ {time_pd.max()}")
            print(f"   * 时间步长: {len(time_vals)} 个时间点")
            
            if len(time_vals) > 1:
                dt = time_pd[1] - time_pd[0]
                print(f"   * 时间分辨率: {dt}")
        except:
            print(f"   * 时间维度名: '{time_dim}' (格式转换失败，显示原始值)")
            print(f"   * Range: {time_vals[0]} ~ {time_vals[-1]}")
    else:
        print("\n   ⚠️ 未找到标准时间维度 (time 或 valid_time)")

    # ----------------------------------------------------
    # 2. 原始变量详情 (Original Variables)
    # ----------------------------------------------------
    print(f"\n[2] 📦 原始变量详情 (Variables)")
    print(f"-"*40)
    print(f"{'变量名 (Key)':<10} | {'形状 (Shape)':<18} | {'单位 (Units)':<10} | {'全名 (Long Name)'}")
    print(f"{'-'*10} | {'-'*18} | {'-'*10} | {'-'*30}")

    for var_name in ds.data_vars:
        var = ds[var_name]
        dims = str(var.shape)
        units = var.attrs.get('units', 'N/A')
        long_name = var.attrs.get('long_name', 'N/A')
        # 截断过长的名字
        if len(long_name) > 28: long_name = long_name[:25] + "..."
        
        print(f"{var_name:<10} | {dims:<18} | {units:<10} | {long_name}")

    # ----------------------------------------------------
    # 3. 数据统计预览 (Statistics)
    # ----------------------------------------------------
    first_var_name = list(ds.data_vars)[0]
    print(f"\n[3] 📊 数据统计概览 (首个变量: {first_var_name})")
    print(f"-"*40)
    
    first_var = ds[first_var_name]
    
    try:
        # 动态使用检测到的时间维度进行切片
        if time_dim and time_dim in first_var.dims:
            # 使用字典解包进行参数传递: isel(valid_time=0) 或 isel(time=0)
            sample_data = first_var.isel({time_dim: 0}).values
        else:
            # 如果没有时间维，或变量本身不随时间变化，直接取值
            sample_data = first_var.values

        print(f"   * Min  : {np.nanmin(sample_data):.4f}")
        print(f"   * Max  : {np.nanmax(sample_data):.4f}")
        print(f"   * Mean : {np.nanmean(sample_data):.4f}")
        
        nan_count = np.isnan(sample_data).sum()
        if nan_count > 0:
            print(f"   * ⚠️ NaNs : {nan_count} (数据中包含缺失值，请注意清洗)")
        else:
            print(f"   * NaNs : 0 (数据完整)")
            
    except Exception as e:
        print(f"   ❌ 统计计算失败: {e}")

    ds.close()
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ERA5 .nc Data Inspector V2")
    parser.add_argument('--file', type=str, help='Path to the .nc file')
    args = parser.parse_args()

    target_file = args.file
    
    # 自动搜索当前目录
    if not target_file or not os.path.exists(target_file):
        files = [f for f in os.listdir('.') if f.endswith('.nc')]
        if files:
            target_file = files[0]
            if not args.file: 
                print(f"Auto-detected file: {target_file}")
        else:
            print("Please provide a file path using --file")
            exit()

    inspect_era5_nc(target_file)