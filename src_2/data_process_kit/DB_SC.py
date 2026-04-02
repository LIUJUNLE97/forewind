import xarray as xr
import numpy as np
import pandas as pd
import glob
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score, silhouette_score
from minisom import MiniSom
import lmoments3 as lm
from lmoments3 import distr
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ================= 配置路径 =================
INPUT_DIR = '/mimer/NOBACKUP/groups/kthmech/junle/guandi_wind_project/data_2/processed'
OUTPUT_DIR = '/mimer/NOBACKUP/groups/kthmech/junle/guandi_wind_project/data_2/som_evaluation'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 对应文章 Fig 3，测试从 2 到 12 个聚类 [cite: 248]
CLUSTER_CONFIGS = [
    (2, 1), (3, 1), (2, 2), (5, 1), (3, 2), 
    (7, 1), (4, 2), (3, 3), (5, 2), (11, 1), (4, 3)
]
# ===========================================

def load_and_preprocess_data(file_pattern):
    """
    直接从 NetCDF 读取原始数据 (u10, v10, geopotential) 并构建特征矩阵
    """
    print("Loading NetCDF files...")
    files = sorted(glob.glob(file_pattern))
    if not files:
        raise FileNotFoundError("No NetCDF files found.")
        
    ds = xr.open_mfdataset(files, chunks={'time': 256}, parallel=False)

    # === 修复 1: 处理 ERA5 的 expver 维度 (如果存在) ===
    # ERA5 有时混合了历史数据(1)和实时数据(5)，导致多出一个维度
    if 'expver' in ds.dims:
        print("Combining ERA5 expver dimensions...")
        # 将两个 expver 合并（互补缺失值）
        ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))

    print("Calculating derived variables (Wind Speed & Height)...")
    
    # === 修复 2: 实时计算风速 (WS = sqrt(u^2 + v^2)) ===
    # 注意：这里并没有 'ProcessedData_normal'，我们需要自己算
    wind_speed = np.sqrt(ds['u10']**2 + ds['v10']**2)
    
    # === 修复 3: 实时计算海拔 (Height = Geopotential / 9.80665) ===
    # 原始 ERA5 提供的是位势 (m^2/s^2)，需要除以重力加速度得到米
    if 'z_height' not in ds:
        if 'geopotential' in ds:
            z_height = ds['geopotential'] / 9.80665
        else:
            # 如果没有位势，尝试用 slope 或其他代替，或者报错
            # 这里假设一定有 geopotential (根据你的报错信息)
            z_height = ds['geopotential'] / 9.80665
    else:
        z_height = ds['z_height']

    # --- 特征提取 ---
    print("Extracting statistical features...")
    
    # 1. 平均风速
    mean_ws = wind_speed.mean(dim='time').compute().values.flatten()
    
    # 2. 风速标准差
    std_ws = wind_speed.std(dim='time').compute().values.flatten()
    
    # 3. 日较差 (Diurnal Range)
    t2m = ds['t2m'] - 273.15 # 转换为摄氏度
    mean_diurnal = (t2m.resample(time='1D').max() - t2m.resample(time='1D').min()).mean(dim='time').compute().values.flatten()
    
    # 4. 海拔 (取第一个时间步，地形通常不变)
    elevation = z_height.isel(time=0).compute().values.flatten()
    
    # 5. 地形粗糙度 (Granularity)
    # 基于计算出的 z_height 计算
    granularity = z_height.isel(time=0).rolling(latitude=3, longitude=3, center=True).std().fillna(0).values.flatten()
    
    # 构建 DataFrame
    lats = ds.latitude.values
    lons = ds.longitude.values
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    df = pd.DataFrame({
        'latitude': lat_grid.flatten(),
        'longitude': lon_grid.flatten(),
        'wind_mean': mean_ws,
        'wind_std': std_ws,
        'diurnal_range': mean_diurnal,
        'elevation': elevation,
        'granularity': granularity
    })
    
    # 去除空值 (比如海洋区域可能有NaN，或者边缘计算导致的NaN)
    df = df.dropna()
    
    ds.close()
    
    # --- 标准化 ---
    feature_cols = ['wind_mean', 'wind_std', 'diurnal_range', 'elevation', 'granularity']
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df[feature_cols].values)
    
    print(f"Data loaded successfully. Samples: {len(df)}")
    return df, data_scaled, feature_cols

def calculate_h_statistic(region_values, n_sim=500):
    """
    计算 H-statistic
    region_values: 某个聚类内所有网格点的物理变量值 (如 wind_mean)
    """
    # 类型强制转换修复 lmoments3 bug
    data = np.array(region_values, dtype=np.float64).tolist()
    
    # 数据量检查
    if len(data) < 5:
        return np.nan, "Insufficient Data"
    if len(set(data)) < 4:
        return np.nan, "Degenerate Data" # 避免完全相同的数据导致拟合失败

    try:
        # 1. 计算观测数据的 L-CV [cite: 205]
        lmom_obs = lm.lmom_ratios(data, nmom=4)
        v_obs = lmom_obs[1] 

        # 2. 拟合 Kappa 分布
        kap_params = distr.kap.lmom_fit(lmom_obs)

        # 3. 蒙特卡洛模拟 [cite: 207]
        sim_lcvs = []
        n_sites = len(data)
        for _ in range(n_sim):
            sim_data = distr.kap.rvs(*kap_params.values(), size=n_sites)
            sim_lmom = lm.lmom_ratios(sim_data, nmom=2)
            sim_lcvs.append(sim_lmom[1])

        # 4. 计算 H 统计量 [cite: 211]
        mu_v = np.mean(sim_lcvs)
        sigma_v = np.std(sim_lcvs)
        
        if sigma_v == 0: return np.nan, "Zero Variance"
        
        h_stat = (v_obs - mu_v) / sigma_v

        if h_stat < 1: status = "Homogeneous"
        elif 1 <= h_stat < 2: status = "Possibly homogeneous"
        else: status = "Definitely heterogeneous"
            
        return h_stat, status

    except Exception as e:
        return np.nan, "Fit Error"

def main():
    # 1. 获取原始数据并标准化
    print("Step 1: Preparing Data...")
    nc_pattern = os.path.join(INPUT_DIR, "processed_*.nc")
    df, data_scaled, features = load_and_preprocess_data(nc_pattern)
    print(f"Data ready. Shape: {data_scaled.shape}")

    # 2. 循环寻找最佳 K 值 (复现 Fig 3)
    print("\nStep 2: Iterating Cluster Configs...")
    eval_results = []
    
    for rows, cols in CLUSTER_CONFIGS:
        n_clusters = rows * cols
        
        # 训练 SOM
        som = MiniSom(rows, cols, len(features), sigma=1.0, learning_rate=0.5, random_seed=42)
        som.train_batch(data_scaled, 20000, verbose=False)
        
        # 获取标签
        winners = np.array([som.winner(x) for x in data_scaled])
        labels = winners[:, 0] * cols + winners[:, 1]
        
        # 计算指标 (使用标准化后的数据!) [cite: 197]
        dbi = davies_bouldin_score(data_scaled, labels)
        sc = silhouette_score(data_scaled, labels, sample_size=5000)
        
        print(f"Config {rows}x{cols} (K={n_clusters}): DBI={dbi:.4f}, SC={sc:.4f}")
        eval_results.append({'n_clusters': n_clusters, 'rows': rows, 'cols': cols, 'dbi': dbi, 'sc': sc})

    # 3. 确定最佳 K 值 (基于最小 DBI)
    eval_df = pd.DataFrame(eval_results)
    best_config = eval_df.loc[eval_df['dbi'].idxmin()]
    best_rows, best_cols = int(best_config['rows']), int(best_config['cols'])
    print(f"\nOptimal Configuration found: {best_rows}x{best_cols} (K={int(best_config['n_clusters'])})")

    # 4. 训练最终的最佳模型
    print("Step 3: Training Final Model & Calculating H-statistics...")
    som_final = MiniSom(best_rows, best_cols, len(features), sigma=1.0, learning_rate=0.5, random_seed=42)
    som_final.train_batch(data_scaled, 50000, verbose=True)
    
    winners = np.array([som_final.winner(x) for x in data_scaled])
    df['cluster_id'] = winners[:, 0] * best_cols + winners[:, 1]
    
    # 5. 计算异质性 H-statistic
    # 这里使用的是原始物理量 (df['wind_mean'])，而不是标准化数据
    h_stats = []
    for cid in sorted(df['cluster_id'].unique()):
        # 提取该聚类下的原始风速数据
        cluster_wind_data = df[df['cluster_id'] == cid]['wind_mean'].values
        
        h_val, status = calculate_h_statistic(cluster_wind_data)
        
        h_stats.append({
            'Cluster': cid, 
            'Size': len(cluster_wind_data),
            'H-stat': h_val,
            'Status': status,
            'Mean_Wind': np.mean(cluster_wind_data)
        })
    
    # 6. 保存所有结果
    df.to_csv(os.path.join(OUTPUT_DIR, "final_som_clusters.csv"), index=False)
    pd.DataFrame(h_stats).to_csv(os.path.join(OUTPUT_DIR, "heterogeneity_test.csv"), index=False)
    
    # 绘图 (Evaluation Metrics)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(eval_df['n_clusters'], eval_df['dbi'], 'r-o', label='DBI')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('DBI (Lower is better)', color='r')
    ax1.invert_yaxis()
    
    ax2 = ax1.twinx()
    ax2.plot(eval_df['n_clusters'], eval_df['sc'], 'b-s', label='SC')
    ax2.set_ylabel('SC (Higher is better)', color='b')
    
    plt.title('SOM Cluster Evaluation')
    plt.savefig(os.path.join(OUTPUT_DIR, "evaluation_metrics.png"))
    print("Done. Results saved.")

if __name__ == "__main__":
    main()