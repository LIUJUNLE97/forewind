import xarray as xr
import numpy as np
import os
import glob
import json
import re
from tqdm import tqdm

# ================= 配置路径 =================
PROCESSED_DIR = '/mimer/NOBACKUP/groups/kthmech/junle/guandi_wind_project/data_2/processed'
OUTPUT_STATS_PATH = os.path.join(PROCESSED_DIR, 'normalization_stats.json')

def generate_normalization_stats():
    # 1. 筛选训练集文件 (1998-2018)
    all_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, 'processed_*.nc')))
    train_files = []
    for f in all_files:
        year_match = re.search(r'(\d{4})', os.path.basename(f))
        if year_match:
            year = int(year_match.group(1))
            if 1998 <= year <= 2018:
                train_files.append(f)
    
    print(f">>> 发现 {len(train_files)} 个训练集文件 (1998-2018)")

    # 2. 改进：显式切块 (Explicit Chunking)
    # 不再使用 chunks={}，而是手动设置较小的块，确保单次加载到内存的数据量可控
    # 建议按时间步切，比如每 24 个步长(1天)一个块
    ds = xr.open_mfdataset(
        train_files, 
        chunks={'time': 24, 'latitude': 360, 'longitude': 360}, 
        combine='by_coords',
        data_vars='minimal',
        coords='minimal',
        compat='override'
    )
    
    vars_to_stats = ['u10', 'v10', 't2m', 'sp', 'geopotential']
    stats_result = {}

    # 3. 改进：串行计算每个变量 (Sequential Computation)
    # 虽然多变量并行更快，但串行能确保内存峰值只来自于一个变量，极大降低崩溃风险
    print(">>> 开始逐个变量计算统计量 (稳健模式)...")
    
    for var in vars_to_stats:
        if var not in ds:
            continue
            
        print(f"    正在处理: {var} ...")
        
        # 使用单个变量的 compute，避免同时缓存多个变量的中间结果
        # .mean() 计算相对轻量
        var_mean = float(ds[var].mean().compute())
        # .std() 是最耗内存的操作，因为它需要基于 mean 再次遍历并计算平方差
        var_std = float(ds[var].std().compute())
        
        stats_result[var] = {
            'mean': var_mean,
            'std': var_std
        }
        print(f"    [完成] {var}: mean={var_mean:.4f}, std={var_std:.4f}")

    # 4. 保存结果
    with open(OUTPUT_STATS_PATH, 'w') as f:
        json.dump(stats_result, f, indent=4)
    
    print(f"\n✅ 统计量已成功保存至: {OUTPUT_STATS_PATH}")
    ds.close()

if __name__ == "__main__":
    # 注意：在 Alvis 上如果内存持续不足，建议不要显式开启 LocalCluster，
    # 而是让 Dask 使用默认的单进程多线程模式（由 xarray 自动管理）
    generate_normalization_stats()