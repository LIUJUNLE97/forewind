import xarray as xr
import numpy as np
import os
import glob
import warnings
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ================= 配置路径 =================
RAW_DIR = '/mimer/NOBACKUP/groups/kthmech/junle/guandi_wind_project/data/raw'
PROCESSED_DIR = '/mimer/NOBACKUP/groups/kthmech/junle/guandi_wind_project/data_2/processed'

if not os.path.exists(PROCESSED_DIR):
    os.makedirs(PROCESSED_DIR)

def handle_era5_anomalies(ds):
    """处理 ERA5 常见的维度异常"""
    # 1. 处理 expver (针对最近月份数据)
    if 'expver' in ds.dims:
        try:
            ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))
        except:
            ds = ds.mean(dim='expver', skipna=True)

    # 2. 统一时间维度名称
    if 'valid_time' in ds.dims:
        ds = ds.rename({'valid_time': 'time'})
        
    # 3. 剔除冗余维度
    if 'number' in ds.dims:
        ds = ds.squeeze('number', drop=True)
    if 'number' in ds.coords:
        ds = ds.drop_vars('number')
        
    return ds

def process_single_file(f_path):
    try:
        fname = os.path.basename(f_path)
        save_name = fname.replace("era5_china_", "processed_")
        save_path = os.path.join(PROCESSED_DIR, save_name)

        if os.path.exists(save_path):
            return f"[跳过] {save_name}"

        with xr.open_dataset(f_path) as ds:
            ds.load() 
            ds = handle_era5_anomalies(ds)
            
            # --- 改进 1: 物理量转换 (Geopotential) ---
            # 借鉴 ai-models 逻辑：模型通常需要位势 (z)
            # 如果原始变量是 'z' (位势)，保持不变；如果是 'gh' (高度)，则乘以 G
            G = 9.80665
            if 'z' in ds:
                ds = ds.rename({'z': 'geopotential'})
            elif 'geopotential' not in ds:
                # 如果只有高度层，执行转换：z = gh * G
                if 'z_height' in ds:
                    ds['geopotential'] = ds['z_height'] * G
            
            # --- 改进 2: 安全填充 NaN ---
            # 气温不能填 0.0 (绝对零度)，应使用插值
            for var in ds.data_vars:
                if var == 't2m':
                    # 使用线性插值填充气温空值
                    ds[var] = ds[var].interpolate_na(dim='time', method='linear').fillna(ds[var].mean())
                elif ds[var].dtype.kind in 'fc':
                    # 风速和压力填充 0 是相对安全的
                    ds[var] = ds[var].fillna(0.0)

            # --- 改进 3: 提取地形特征 (Slope) ---
            # 虽然不进入主分支，但保留 slope 作为辅助评估指标是有价值的
            if 'geopotential' in ds:
                # 位势高度在时间维度上是静态的，取第一个步长计算
                z_static = ds['geopotential'].isel(time=0).values
                dy, dx = np.gradient(z_static)
                ds['slope'] = (('latitude', 'longitude'), np.sqrt(dx**2 + dy**2))

            # --- 改进 4: 变量精简与类型统一 ---
            # 仅保留核心输入变量
            keep_vars = [
                'u10', 'v10',        # 核心预测目标
                't2m', 'sp',         # 物理耦合变量
                'geopotential',      # 核心地形变量 (m^2/s^2)
                'slope'              # 辅助地形指标
            ]
            
            final_vars = [v for v in keep_vars if v in ds]
            ds_clean = ds[final_vars].astype(np.float32)

            # 保存为压缩的 NetCDF
            comp = dict(zlib=True, complevel=1)
            encoding = {var: comp for var in ds_clean.data_vars}
            ds_clean.to_netcdf(save_path, encoding=encoding)
            
            return f"[成功] {fname}"

    except Exception as e:
        return f"[错误] {fname}: {str(e)}"

# ================= 主控制程序 =================
def main():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.nc")))
    print(f">>> 开始并行预处理 {len(files)} 个文件 (纯净原数据模式)...")
    
    max_workers = 10 
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_single_file, files), total=len(files)))

    # 错误统计
    error_count = sum(1 for res in results if "[错误]" in res)
    if error_count > 0:
        for res in results:
            if "[错误]" in res: print(res)
            
    print(f"\n>>> 处理完成。成功: {len(files) - error_count}, 失败: {error_count}")

if __name__ == "__main__":
    main()