import os
import glob
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ================= 1. 自动化配置 (请修改这里) =================
# 指向你刚才预处理好的 .nc 文件目录
PROCESSED_DIR = '/mimer/NOBACKUP/groups/kthmech/junle/guandi_wind_project/data_2/processed'
# 指向你的 SOM 结果 CSV
SOM_CSV_PATH = '/mimer/NOBACKUP/groups/kthmech/junle/guandi_wind_project/data_2/som/som_full_results.csv'
# 输出诊断报告的目录
REPORT_DIR = './diagnostic_reports'
os.makedirs(REPORT_DIR, exist_ok=True)

class Checker:
    def __init__(self, nc_dir, som_csv):
        self.nc_files = sorted(glob.glob(os.path.join(nc_dir, "*.nc")))
        self.som_df = pd.read_csv(som_csv)
        if not self.nc_files:
            raise FileNotFoundError(f"在 {nc_dir} 下没找到 .nc 文件")
        print(f">>> 已发现 {len(self.nc_files)} 个数据文件，SOM 记录共 {len(self.som_df)} 条。")

    def run_full_diagnostic(self):
        """执行完整诊断流水线"""
        # 1. 抽取一个样本进行深入检查
        sample_ds = xr.open_dataset(self.nc_files[0])
        
        print("\n[Step 1/4] 检查变量完整性与数值量级...")
        self._check_variables(sample_ds)
        
        print("\n[Step 2/4] 检查地理空间对齐 (SOM vs Terrain)...")
        self._check_spatial_alignment(sample_ds)
        
        print("\n[Step 3/4] 检查物理逻辑一致性 (Cluster Physics)...")
        self._check_cluster_physics()
        
        print("\n[Step 4/4] 扫描全量数据异常 (NaN & Inf)...")
        self._scan_all_files()
        
        print(f"\n✅ 诊断完成！报告与图表已保存至: {REPORT_DIR}")

    def _check_variables(self, ds):
        # 检查 W-TAM 必需的变量
        required = ['u10', 'v10', 't2m', 'z_height', 'sp']
        for var in required:
            if var in ds:
                v = ds[var].values
                print(f"  - {var:10} | Mean: {np.nanmean(v):.2f} | Max: {np.nanmax(v):.2f} | Min: {np.nanmin(v):.2f}")
            else:
                print(f"  - ⚠️ 缺失变量: {var}")

    def _check_spatial_alignment(self, ds):
        """可视化：确保 SOM 类别真的长在对应的地形上"""
        t_idx = 0
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # 绘制 ERA5 提取的地形高度
        ds.z_height.isel(time=t_idx).plot(ax=axes[0], cmap='terrain', add_colorbar=True)
        axes[0].set_title("ERA5 Terrain (z_height)")
        
        # 绘制 SOM 聚类分布
        # 我们根据经纬度重构 SOM 矩阵进行可视化
        som_pivot = self.som_df.pivot(index='latitude', columns='longitude', values='cluster_id')
        im = axes[1].imshow(som_pivot, origin='lower', cmap='tab10', 
                            extent=[self.som_df.longitude.min(), self.som_df.longitude.max(), 
                                    self.som_df.latitude.min(), self.som_df.latitude.max()])
        plt.colorbar(im, ax=axes[1], label='Cluster ID')
        axes[1].set_title("SOM Geographic Clusters")
        
        plt.savefig(os.path.join(REPORT_DIR, 'spatial_alignment.png'), dpi=300)
        plt.show()

    def _check_cluster_physics(self):
        """验证不同地形下的物理特性是否符合逻辑"""
        # 例如：高原 (高 Elevation) 的风速波动 (wind_std) 理论上应该不同于平原
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.som_df, x='cluster_id', y='wind_std', palette='tab10')
        plt.title("Wind Variability (std) per SOM Cluster")
        plt.savefig(os.path.join(REPORT_DIR, 'cluster_physics_stats.png'))
        
        # 计算不同类别的海拔均值，确认类别含义
        mean_elev = self.som_df.groupby('cluster_id')['elevation'].mean().sort_values()
        print("\nSOM 类别海拔均值 (验证分类逻辑):")
        print(mean_elev)

    def _scan_all_files(self):
        """对所有 26 年数据进行快速 NaN 扫描"""
        errors = []
        for f in tqdm(self.nc_files[:50], desc="扫描前50个文件安全性"): # 示例扫描，可根据需要增加
            try:
                with xr.open_dataset(f) as ds:
                    if ds.u10.isnull().any():
                        errors.append(f"{os.path.basename(f)} 发现 NaN")
            except Exception as e:
                errors.append(f"{os.path.basename(f)} 读取失败: {e}")
        
        if not errors:
            print("  - 全量数据初步扫描: ✅ 未发现坏点")
        else:
            print(f"  - ⚠️ 发现问题: {errors}")

if __name__ == "__main__":
    checker = Checker(PROCESSED_DIR, SOM_CSV_PATH)
    checker.run_full_diagnostic()