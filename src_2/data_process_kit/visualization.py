import pandas as pd
import matplotlib.pyplot as plt
import os

# ================= 1. 配置路径 =================
# 输入文件：SOM 全量结果与选定的 12 个站点坐标
PROJECT_ROOT = os.environ.get('WIND_PROJECT_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SOM_RESULTS_CSV = os.environ.get('WIND_SOM_CSV', os.path.join(PROJECT_ROOT, 'data', 'metadata', 'som_full_results.csv'))
SITES_COORDS_CSV = os.environ.get('WIND_SITES_CSV', os.path.join(PROJECT_ROOT, 'data', 'metadata', 'final_12_sites_coords.csv'))
# 输出目录
OUTPUT_DIR = os.environ.get('WIND_FIG_DIR', os.path.join(PROJECT_ROOT, 'fig', 'som_plots'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_maps():
    # 检查文件是否存在
    if not os.path.exists(SOM_RESULTS_CSV) or not os.path.exists(SITES_COORDS_CSV):
        print("❌ 错误：未找到 CSV 结果文件，请检查路径。")
        return

    print(">>> 正在读取聚类数据...")
    df_full = pd.read_csv(SOM_RESULTS_CSV)
    df_sites = pd.read_csv(SITES_COORDS_CSV)

    # 设置绘图风格：使用学术常用的 tab20b 颜色映射
    cmap = 'tab20b'
    
    # ---------------------------------------------------------
    # 图 1：原始聚类底图 (展示 9 个类的空间分布)
    # ---------------------------------------------------------
    print(">>> 正在绘制原始聚类底图...")
    plt.figure(figsize=(14, 9))
    
    # 使用较小的方形标记模拟网格感，不设置透明度以展示原始分布
    sc1 = plt.scatter(df_full.longitude, df_full.latitude, 
                      c=df_full.cluster_id, cmap=cmap, 
                      s=3, marker='s', edgecolors='none', alpha=1.0)
    
    plt.colorbar(sc1, label='SOM Cluster ID (0-8)', fraction=0.02, pad=0.04)
    plt.title("Original 3x3 SOM Clustering Map (25-Year Statistics)", fontsize=14, pad=15)
    plt.xlabel("Longitude (°E)")
    plt.ylabel("Latitude (°N)")
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.savefig(os.path.join(OUTPUT_DIR, "map_01_original_clusters.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # 图 2：12 站点标注图 (展示 6 组 x 2 采样的定标位置)
    # ---------------------------------------------------------
    print(">>> 正在绘制 12 站点标注图...")
    plt.figure(figsize=(14, 9))
    
    # 背景：淡化处理，作为辅助参考 (Alpha = 0.35)
    plt.scatter(df_full.longitude, df_full.latitude, 
                c=df_full.cluster_id, cmap=cmap, 
                s=3, marker='s', edgecolors='none', alpha=0.35)
    
    # 前景：循环标注 12 个站点
    for idx, site in df_sites.iterrows():
        # 使用红色圆点标注，增加白色描边提升辨识度
        plt.scatter(site['Lon'], site['Lat'], 
                    color='red', s=45, marker='o', 
                    edgecolors='white', linewidths=0.8, zorder=10)
        
        # 文本标注：错位显示并添加半透明背景框
        plt.annotate(site['Tag'], 
                     xy=(site['Lon'], site['Lat']), 
                     xytext=(6, 4), 
                     textcoords='offset points',
                     fontsize=9, fontweight='bold',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1),
                     zorder=11)

    plt.title("6 Groups x 2 Sites Selection Map (Medoid & Hardest)\n"
              "C: Coastal, E: Extreme Mt, H: High Plateau, W: Wind Mean, S: Wind Std, I: Inland Plain", 
              fontsize=14, pad=15)
    plt.xlabel("Longitude (°E)")
    plt.ylabel("Latitude (°N)")
    plt.grid(True, linestyle='--', alpha=0.2)
    
    plt.savefig(os.path.join(OUTPUT_DIR, "map_02_annotated_12sites.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ 绘图完成！图片已保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    plot_maps()
