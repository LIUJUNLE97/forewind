import xarray as xr
import pandas as pd
import numpy as np
import glob
import os

# ---------------- 配置 ----------------
# 你的 SOM 结果路径
SOM_CSV = '/mimer/NOBACKUP/groups/kthmech/junle/guandi_wind_project/data_2/som/som_full_results.csv'
# 你的 NC 数据路径
NC_DIR = '/mimer/NOBACKUP/groups/kthmech/junle/guandi_wind_project/data_2/processed'
# 上海坐标
TARGET_LAT = 31.23
TARGET_LON = 121.47
# -------------------------------------

def check_shanghai_status():
    print(f"1. 正在查询上海 ({TARGET_LAT}, {TARGET_LON}) 的 Cluster ID...")
    
    # 读取 CSV
    df = pd.read_csv(SOM_CSV)
    
    # 找最近的点
    dist = (df['latitude'] - TARGET_LAT)**2 + (df['longitude'] - TARGET_LON)**2
    shanghai_row = df.loc[dist.idxmin()]
    
    print(f"   >>> 匹配格点坐标: {shanghai_row['latitude']:.2f}, {shanghai_row['longitude']:.2f}")
    print(f"   >>> 归属类别: Cluster {int(shanghai_row['cluster_id'])}")
    print(f"   >>> 地形粗糙度 (Granularity): {shanghai_row['granularity']:.4f}")
    
    print("-" * 30)
    print("2. 正在从 NC 文件提取历史数据以计算 MSE (这可能需要几十秒)...")
    
    files = sorted(glob.glob(os.path.join(NC_DIR, "*.nc")))
    all_vals = []
    
    # 快速提取逻辑 (只针对单点)
    for f in files:
        try:
            with xr.open_dataset(f) as ds:
                if 'u10' not in ds or 'v10' not in ds: continue
                
                # 提取单点分量 (先提取再算风速，比全场算风速快得多)
                u = ds['u10'].sel(latitude=TARGET_LAT, longitude=TARGET_LON, method='nearest').values
                v = ds['v10'].sel(latitude=TARGET_LAT, longitude=TARGET_LON, method='nearest').values
                
                ws = np.sqrt(u**2 + v**2)
                all_vals.append(ws)
        except:
            continue
            
    if not all_vals:
        print("错误：未能提取到数据。")
        return

    # 拼接全序列
    full_series = np.concatenate(all_vals)
    print(f"   >>> 成功提取时间序列长度: {len(full_series)} 小时")
    
    # 计算 3小时 惯性误差 (Persistence MSE)
    # 逻辑：预测未来3小时的风速 = 当前风速
    horizon = 3
    # 真实值(t+3) - 预测值(t)
    diff = full_series[horizon:] - full_series[:-horizon]
    mse = np.mean(diff**2)
    
    print("-" * 30)
    print(f"   >>> 上海单点 3小时惯性 MSE: {mse:.4f} (m/s)^2")
    print("-" * 30)
    
    if mse > 2.0:
        print("结论：上海属于【极难预测】区域 (MSE > 2.0)。挑战巨大！")
    elif mse > 1.0:
        print("结论：上海属于【中等难度】区域。")
    else:
        print("结论：上海属于【相对平稳】区域。如果是这样，我们的模型需要关注 '极端事件' 捕捉能力。")

if __name__ == "__main__":
    check_shanghai_status()