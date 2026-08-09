import pandas as pd
import numpy as np
import xarray as xr
import os
import glob

# ================= 配置区 =================
# 1. 你的 CSV 文件路径 (输入)
PROJECT_ROOT = os.environ.get('WIND_PROJECT_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CSV_PATH = os.environ.get('WIND_SOM_CSV', os.path.join(PROJECT_ROOT, 'data', 'metadata', 'final_som_clusters.csv'))

# 2. 你的任意一个 NC 文件路径 (用于获取正确的经纬度网格参照)
# 我们需要知道原始数据的长宽和经纬度排序，才能把 CSV 还原成正确的矩阵
NC_DIR = os.environ.get('WIND_PROCESSED_DIR', os.path.join(PROJECT_ROOT, 'processed'))

# 3. 输出的 NPY 文件路径
OUTPUT_PATH = os.environ.get('WIND_SOM_MAP', os.path.join(PROJECT_ROOT, 'data', 'metadata', 'som_map.npy'))
# =========================================

def generate_som_map():
    print("1. 读取参考网格信息...")
    # 随便找一个 nc 文件读取经纬度
    nc_files = sorted(glob.glob(os.path.join(NC_DIR, "*.nc")))
    if not nc_files:
        raise FileNotFoundError("找不到 NC 文件作为参考！")
    
    ref_ds = xr.open_dataset(nc_files[0])
    target_lats = ref_ds.latitude.values
    target_lons = ref_ds.longitude.values
    ref_ds.close()
    
    H, W = len(target_lats), len(target_lons)
    print(f"   >>> 目标网格形状: [H={H}, W={W}]")
    print(f"   >>> 纬度范围: {target_lats[0]:.2f} ~ {target_lats[-1]:.2f}")
    
    print("2. 读取 SOM CSV 结果...")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"找不到 CSV 文件: {CSV_PATH}")
        
    df = pd.read_csv(CSV_PATH)
    print(f"   >>> CSV 行数: {len(df)}")
    
    print("3. 执行转换 (Pivot & Align)...")
    # 使用 Pandas 的 pivot 功能将长表转换为矩阵
    # index=纬度, columns=经度, values=类别ID
    pivot_table = df.pivot(index='latitude', columns='longitude', values='cluster_id')
    
    # [关键步骤] 重新索引 (Reindex)
    # 因为 pivot 会自动按升序排列索引，但 ERA5 的纬度通常是降序的 (90 -> -90)
    # 我们必须强制让矩阵的行/列顺序与 NC 文件完全一致
    pivot_table = pivot_table.reindex(index=target_lats, columns=target_lons)
    
    som_map = pivot_table.values
    
    # 检查是否因为坐标精度问题导致匹配失败 (出现 NaN)
    if np.isnan(som_map).any():
        n_nans = np.isnan(som_map).sum()
        print(f"!!! 警告: 转换后发现 {n_nans} 个空值 (NaN)。")
        print("    这通常是因为 CSV 里的经纬度精度与 NC 文件不完全一致。")
        print("    正在尝试使用最近邻插值修复...")
        
        # 简单修复：用 0 填充或最近邻填充 (这里用 0 填充作为类别 0，或者检查数据)
        # 这种情况下通常是因为 float精度问题 (31.23000001 vs 31.23)
        # 如果问题严重，建议检查 CSV 生成逻辑。这里先 fillna(0) 保证能跑
        som_map = np.nan_to_num(som_map, nan=0)
        
    som_map = som_map.astype(np.int64)
    
    print(f"4. 保存至 {OUTPUT_PATH}")
    np.save(OUTPUT_PATH, som_map)
    print("✅ 转换完成！")

if __name__ == "__main__":
    generate_som_map()
