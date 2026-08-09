import xarray as xr
import numpy as np
import os
import glob
import pandas as pd
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.environ.get('WIND_PROJECT_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
INPUT_DIR = os.environ.get('WIND_PROCESSED_DIR', os.path.join(PROJECT_ROOT, 'processed'))
OUTPUT_DIR = os.environ.get('WIND_SOM_OUTPUT_DIR', os.path.join(PROJECT_ROOT, 'fig', 'som'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

SOM_ROWS, SOM_COLS = 3, 3 # 保持 3x3=9 个原始聚类

 
def extract_features_fast(files):
    

    ds = xr.open_mfdataset(files, chunks={'time': 256}, parallel=False)
    ds_sampled = ds.isel(time=slice(None, None, 6))
    
    mean_ws = ds_sampled['ProcessedData_normal'].mean(dim='time').compute().values.flatten()
  
    std_ws = ds_sampled['ProcessedData_normal'].std(dim='time').compute().values.flatten()
    
    t2m = ds_sampled['t2m'] - 273.15
    # 先算极值均值，再释放
    mean_diurnal = (t2m.resample(time='1D').max() - t2m.resample(time='1D').min()).mean(dim='time').compute().values.flatten()
    del t2m # 显式释放
    
   
    elevation_ds = ds['z_height'].isel(time=0).compute()
    elevation = elevation_ds.values.flatten()
    # 颗粒度计算使用 NumPy 层面操作
    granularity = elevation_ds.rolling(latitude=3, longitude=3, center=True).std().fillna(0).values.flatten()
    
    # 3. 构造结果 DataFrame
    # 获取经纬度坐标
    lats = ds_sampled.latitude.values
    lons = ds_sampled.longitude.values
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
    
    # 清理并关闭数据集
    ds.close()
    return df.dropna()

def main():
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "processed_*.nc")))
    df = extract_features_fast(files)
    
    # SOM 训练 (空间采样加速)
   
    feats = ['wind_mean', 'wind_std', 'diurnal_range', 'elevation', 'granularity']
    data_scaled = StandardScaler().fit_transform(df[feats].values)
    train_idx = np.random.choice(data_scaled.shape[0], int(data_scaled.shape[0]*0.2), replace=False)
    
    som = MiniSom(SOM_ROWS, SOM_COLS, len(feats), sigma=1.2, learning_rate=0.5)
    som.train_batch(data_scaled[train_idx], 25000, verbose=True)
    
    winners = np.array([som.winner(x) for x in data_scaled])
    df['cluster_id'] = winners[:, 0] * SOM_COLS + winners[:, 1]
    
    # 保存结果
    df.to_csv(os.path.join(OUTPUT_DIR, "som_full_results.csv"), index=False)
    
    # 绘图
    plt.figure(figsize=(12, 8))
    plt.scatter(df.longitude, df.latitude, c=df.cluster_id, cmap='tab20b', s=2, alpha=1, marker='s')
    plt.title("Original SOM Clustering (3x3)")
    plt.savefig(os.path.join(OUTPUT_DIR, "som_map_original.png"), dpi=300)

    
if __name__ == "__main__":
    main()
