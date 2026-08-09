import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import gc

# ---------------- 配置区 ----------------
PROJECT_ROOT = os.environ.get('WIND_PROJECT_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SOM_RESULT_PATH = os.environ.get('WIND_SOM_CSV', os.path.join(PROJECT_ROOT, 'data', 'metadata', 'som_full_results.csv'))
NC_DATA_DIR = os.environ.get('WIND_PROCESSED_DIR', os.path.join(PROJECT_ROOT, 'processed'))
OUTPUT_DIR = os.environ.get('WIND_SOM_OUTPUT_DIR', os.path.join(PROJECT_ROOT, 'fig', 'som_analysis'))
# ----------------------------------------

def extract_points_simple(nc_files, points_df):
    all_time_series = []
    target_lats = xr.DataArray(points_df['latitude'].values, dims="points")
    target_lons = xr.DataArray(points_df['longitude'].values, dims="points")
    
    print(f"开始逐个处理 {len(nc_files)} 个文件 (纯净模式)...")
    
    for i, fpath in enumerate(nc_files):
        try:
            with xr.open_dataset(fpath) as ds:
                if 'u10' not in ds or 'v10' not in ds:
                    continue
                
                # 计算风速
                u10 = ds['u10']
                v10 = ds['v10']
                wind_speed = np.sqrt(u10**2 + v10**2)
                
                # 提取点
                subset = wind_speed.sel(
                    latitude=target_lats, 
                    longitude=target_lons, 
                    method='nearest'
                ).load()
                
                subset.name = 'wind_speed'
                all_time_series.append(subset)
            
            if (i + 1) % 10 == 0:
                print(f"已处理 {i + 1}/{len(nc_files)} 个文件...")
                gc.collect() 
                
        except Exception as e:
            print(f"!!! 处理文件 {os.path.basename(fpath)} 失败: {e}")
            
    print("拼接时间序列...")
    combined_data = xr.concat(all_time_series, dim='time')
    combined_data = combined_data.sortby('time')
    return combined_data

def main_analysis():
    # 1. 读取 SOM 结果
    if not os.path.exists(SOM_RESULT_PATH):
        print(f"错误: 找不到文件 {SOM_RESULT_PATH}")
        return
    df_som = pd.read_csv(SOM_RESULT_PATH)
    
    # === 关键修改 1: 将 Cluster ID 转换为 1-9 用于显示 ===
    # 我们创建一个新列 'Display_ID'，原有的 'cluster_id' (0-8) 仅用于逻辑处理
    df_som['Display_ID'] = df_som['cluster_id'] + 1
    
    # 抽样
    sampled_points = df_som.groupby('cluster_id').apply(lambda x: x.sample(n=min(10, len(x)))).reset_index(drop=True)
    # 确保 sampled_points 也有 Display_ID
    sampled_points['Display_ID'] = sampled_points['cluster_id'] + 1
    
    print(f"已选定 {len(sampled_points)} 个测试点。")

    # 2. 获取文件并提取 (这部分逻辑不变)
    files = sorted(glob.glob(os.path.join(NC_DATA_DIR, "*.nc")))
    if not files: return

    try:
        da_matrix = extract_points_simple(files, sampled_points)
    except Exception as e:
        print(f"提取流程终止: {e}")
        return

    # 3. 计算 Persistence MSE
    data_np = da_matrix.values 
    horizon = 3 
    
    diff = data_np[horizon:] - data_np[:-horizon]
    mse_per_point = np.nanmean(diff**2, axis=0) 
    sampled_points['prediction_mse'] = mse_per_point

    # 4. 绘图 (完全重写以匹配地图风格)
    print("正在绘图...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # === 关键修改 2: 定义与地图完全一致的莫兰迪色板 ===
    # 顺序必须严格对应 Cluster 1 到 Cluster 9
    morandi_colors = [
        '#8FB1CC', # Cluster 1: 雾霾蓝
        '#E6B9C0', # Cluster 2: 灰粉
        '#A8C8A6', # Cluster 3: 豆沙绿
        '#E8D595', # Cluster 4: 姜黄
        '#C6B4CE', # Cluster 5: 香芋紫
        '#D4C4B7', # Cluster 6: 燕麦灰
        '#D8A49B', # Cluster 7: 砖红
        '#7B8D8E', # Cluster 8: 灰蓝
        '#9DAD7F', # Cluster 9: 橄榄
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    
    # --- 图 A: 预测误差 (Barplot) ---
    # x轴使用 Display_ID (1-9)
    # palette 使用 morandi_colors
    sns.barplot(x='Display_ID', y='prediction_mse', data=sampled_points, 
                ax=axes[0], 
                palette=morandi_colors, # <--- 颜色对应
                errorbar='sd', capsize=.1, err_kws={'linewidth': 1})
    
    axes[0].set_title('Analysis 1: Predictability Difficulty by Cluster', fontsize=14)
    axes[0].set_ylabel('Persistence MSE (Wind Speed)', fontsize=12)
    axes[0].set_xlabel('SOM Cluster Label', fontsize=12)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    # --- 图 B: 物理一致性/颗粒度 (Boxplot) ---
    # x轴使用 Display_ID (1-9)
    sns.boxplot(x='Display_ID', y='granularity', data=df_som, 
                ax=axes[1], 
                palette=morandi_colors, # <--- 颜色对应
                showfliers=False) # 隐藏过多的离群点使图表更整洁
    
    axes[1].set_title('Analysis 2: Terrain Complexity (Granularity)', fontsize=14)
    axes[1].set_ylabel('Granularity / Roughness Index', fontsize=12)
    axes[1].set_xlabel('SOM Cluster Label', fontsize=12)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    
    save_path = os.path.join(OUTPUT_DIR, 'som_physical_analysis_morandi.png')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ 分析完成！图片已保存至: {save_path}")

if __name__ == "__main__":
    main_analysis()
