import sys
import os
import glob
import numpy as np
import xarray as xr
import json
import argparse
from tqdm import tqdm

# 路径设置
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
sys.path.append(project_root)

from src_2.utils.scalers import StandardScaler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nc_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--stats_file', type=str, default='normalization_stats.json')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    
    # 1. 加载统计量
    stats_path = os.path.join(args.nc_dir, args.stats_file)
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    scalers = {}
    vars_in = ['u10', 'v10', 't2m', 'sp', 'geopotential']
    for var in vars_in:
        scalers[var] = StandardScaler(mean=stats[var]['mean'], std=stats[var]['std'])

    all_files = sorted(glob.glob(os.path.join(args.nc_dir, '*.nc')))
    print(f"Found {len(all_files)} files.")

    # =========================================================
    # Pass 1: 快速扫描，计算总时间步数
    # =========================================================
    print(">>> Pass 1: Counting total time steps...")
    total_time_steps = 0
    
    # 读取第一个文件获取 H, W
    with xr.open_dataset(all_files[0]) as ds_first:
        H = ds_first.sizes['latitude']
        W = ds_first.sizes['longitude']
    
    print(f"Detected Shape: [Time=?, 5, {H}, {W}]")

    for f in tqdm(all_files, desc="Counting"):
        with xr.open_dataset(f) as ds:
            total_time_steps += ds.sizes['time']
            
    print(f"Total Time Steps: {total_time_steps}")
    
    # =========================================================
    # Pass 2: 初始化磁盘上的巨型数组 (Memmap)
    # =========================================================
    npy_path = os.path.join(args.save_dir, 'weather_data_all.npy')
    time_path = os.path.join(args.save_dir, 'timestamps.npy')
    
    print(f">>> Initializing memmap file at {npy_path}...")
    fp = np.lib.format.open_memmap(
        npy_path, 
        mode='w+', 
        dtype='float32', 
        shape=(total_time_steps, 5, H, W)
    )
    
    all_timestamps = []
    
    # =========================================================
    # Pass 3: 逐个文件填充
    # =========================================================
    print(">>> Pass 2: Writing data to disk incrementally...")
    
    current_idx = 0
    
    for f_path in tqdm(all_files, desc="Writing"):
        try:
            ds = xr.open_dataset(f_path, engine='netcdf4')
            
            # ========================================================
            # [FIXED] 更加稳健的 expver 处理逻辑
            # ========================================================
            if 'expver' in ds.coords:
                # 情况 1: expver 是一个维度 (Dimension)
                if 'expver' in ds.dims:
                    if ds.sizes['expver'] > 1:
                        # 混合数据: 合并
                        try:
                            ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))
                        except:
                            # 备用方案: 只要第一个
                            ds = ds.isel(expver=0)
                    else:
                        # 维度为1: 挤压
                        ds = ds.squeeze('expver')
                # 情况 2: expver 只是一个标量坐标 (不是维度)
                else:
                    # 直接丢弃，不要碰它，否则报错 "no index found"
                    ds = ds.drop_vars('expver')
            # ========================================================

            # 获取当前文件的时间长度
            t_len = ds.sizes['time']
            
            # 记录时间戳
            all_timestamps.append(ds.time.values)
            
            # 处理变量
            file_vars_data = []
            for var in vars_in:
                raw_data = ds[var].values.astype(np.float32)
                
                if np.isnan(raw_data).any():
                    raw_data = np.nan_to_num(raw_data, nan=0.0)
                
                norm_data = scalers[var].transform(raw_data)
                file_vars_data.append(norm_data)
            
            # 堆叠
            chunk_data = np.stack(file_vars_data, axis=1)
            
            # 写入硬盘
            fp[current_idx : current_idx + t_len] = chunk_data
            
            current_idx += t_len
            ds.close()
            
        except Exception as e:
            # 打印完整错误，不再跳过，方便调试
            print(f"\nCRITICAL ERROR processing {f_path}: {e}")
            import traceback
            traceback.print_exc()
            # 这里如果不 raise，后面的 timestamps 会对不上，建议直接退出
            raise e

    print(">>> Saving timestamps...")
    if len(all_timestamps) > 0:
        final_times = np.concatenate(all_timestamps, axis=0)
        np.save(time_path, final_times)
        print("✅ Conversion Complete! Data saved directly to disk.")
    else:
        print("❌ Error: No timestamps collected. Conversion failed.")

    # 释放 memmap
    del fp 

if __name__ == "__main__":
    main()