import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ================= 1. 配置区 =================
# 输入：上一步 SOM 生成的全量数据
PROJECT_ROOT = os.environ.get('WIND_PROJECT_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
INPUT_CSV = os.environ.get('WIND_SOM_CSV', os.path.join(PROJECT_ROOT, 'data', 'metadata', 'som_full_results.csv'))
OUTPUT_DIR = os.environ.get('WIND_SITES_OUTPUT_DIR', os.path.join(PROJECT_ROOT, 'fig', 'sites'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

START_YEAR, END_YEAR = 1999, 2024

# ================= 2. 选点逻辑：S-NAC 优势域 =================
# 专注于方法最擅长的三种气候情景：地形强迫、热力强迫、高能基准

def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

def select_snac_specialized_sites(df):
    print(">>> 正在执行 S-NAC 优势域选点 (Topo/Therm/Base)...")
    
    # 归一化特征
    g_norm = normalize(df['granularity'])
    w_mean_norm = normalize(df['wind_mean'])
    w_std_norm = normalize(df['wind_std'])
    d_norm = normalize(df['diurnal_range'])
    
    # --- A. 地形强迫区 (Topographic Forcing) ---
    # 逻辑：崎岖度高 + 风速大 = 地形对流场的非线性修正最强
    df['score_topo'] = 0.7 * g_norm + 0.3 * w_mean_norm
    # 选取 score 最高的 4 个点
    topo_sites = pick_diverse_sites(df.sort_values('score_topo', ascending=False), 4, 'Topo-Force')

    # --- B. 热力强迫区 (Thermal Forcing) ---
    # 逻辑：日较差大 + 波动大 = 热力驱动的高频非稳态流场 (如海陆风)
    # 排除高海拔以专注于海陆/低地热力
    df['score_therm'] = 0.6 * d_norm + 0.4 * w_std_norm
    therm_pool = df[df['elevation'] < 3000]
    therm_sites = pick_diverse_sites(therm_pool.sort_values('score_therm', ascending=False), 4, 'Therm-Force')

    # --- C. 高能基准区 (High-Energy Baseline) ---
    # 逻辑：极度平坦 + 风速极大 = 纯净的流体力学背景，验证 Backbone 鲁棒性
    df['score_base'] = 0.6 * w_mean_norm - 0.3 * g_norm - 0.1 * d_norm
    base_pool = df[df['granularity'] < df['granularity'].quantile(0.4)] # 取平坦的前40%
    base_sites = pick_diverse_sites(base_pool.sort_values('score_base', ascending=False), 4, 'Baseline')

    # 合并结果
    final_sites = pd.concat([topo_sites, therm_sites, base_sites])
    return final_sites

def pick_diverse_sites(candidates, n, group_name):
    """辅助：确保选点尽量分散在不同 Cluster，避免扎堆"""
    selected = []
    seen_clusters = set()
    # 简单的去重策略
    for idx, row in candidates.iterrows():
        if len(selected) >= n: break
        if row['cluster_id'] not in seen_clusters or len(seen_clusters) >= 3:
            selected.append(row)
            seen_clusters.add(row['cluster_id'])
            
    # 格式化输出
    res = []
    for i, row in pd.DataFrame(selected).iterrows():
        res.append({
            'Group': group_name,
            'Lat': row['latitude'], 'Lon': row['longitude'],
            'Tag': f"{group_name[0]}-{i+1}" # T-1, H-1, B-1
        })
    return pd.DataFrame(res)

# ================= 3. 绘图逻辑：您指定的高饱和度风格 =================

def plot_maps(df_full, df_sites):
    print(">>> 正在生成分布图...")
    
    # 设置统一的 Figure 大小和 DPI
    plt.figure(figsize=(14, 10))
    
    # --- 图 1：原始分布图 (High Saturation) ---
    # 完全采用您提供的参数：s=15, marker='s', alpha=1.0
    # 这会产生类似 "pcolormesh" 的连续色块效果
    sc = plt.scatter(df_full['longitude'], df_full['latitude'], 
                     c=df_full['cluster_id'], cmap='tab20b', 
                     s=15, marker='s', alpha=1.0, edgecolors='none')
    
    plt.colorbar(sc, label='Climate Cluster ID', fraction=0.025, pad=0.04)
    plt.title(f'S-NAC Climate Zoning ({START_YEAR}-{END_YEAR})\nFeatures: Wind Dynamics, Terrain, Thermal Stability', fontsize=14)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    save_path1 = os.path.join(OUTPUT_DIR, "map_01_zones_saturated.png")
    plt.savefig(save_path1, dpi=300, bbox_inches='tight')
    print(f"✅ 原始分布图已保存: {save_path1}")
    plt.close()

    # --- 图 2：站点标注图 (Controlled Saturation) ---
    plt.figure(figsize=(14, 10))
    
    # 关键修改：alpha=0.5 -> 控制饱和度，让背景变淡，使前景站点突出
    # 依然保持 s=15, marker='s' 以维持方块拼图的质感
    plt.scatter(df_full['longitude'], df_full['latitude'], 
                c=df_full['cluster_id'], cmap='tab20b', 
                s=15, marker='s', alpha=0.5, edgecolors='none') # <--- 降低饱和度
    
    # 绘制 12 个站点
    # 使用不同形状/颜色区分三大战场
    styles = {
        'Topo-Force':  {'c': '#d62728', 'm': '^', 'l': 'Topographic (NAC)'}, # 红/三角
        'Therm-Force': {'c': '#ff7f0e', 'm': 's', 'l': 'Thermal (NAC)'},     # 橙/方块
        'Baseline':    {'c': '#1f77b4', 'm': 'o', 'l': 'Baseline (Backbone)'} # 蓝/圆
    }
    
    # 循环绘制站点
    for g in ['Topo-Force', 'Therm-Force', 'Baseline']:
        subset = df_sites[df_sites['Group'] == g]
        style = styles[g]
        plt.scatter(subset['Lon'], subset['Lat'], 
                    c=style['c'], marker=style['m'], s=200, 
                    edgecolors='white', linewidth=1.5, zorder=10, label=style['l'])
        
        # 添加标签文本
        for _, row in subset.iterrows():
            plt.text(row['Lon'], row['Lat'], row['Tag'], fontsize=9, fontweight='bold',
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor=style['c'], boxstyle='round,pad=0.2'))

    plt.legend(loc='lower right', frameon=True, fontsize=12)
    plt.title(f'S-NAC Experimental Sites: 12 Strategic Locations\nTopographic vs. Thermal vs. Baseline', fontsize=14)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    save_path2 = os.path.join(OUTPUT_DIR, "map_02_sites_annotated.png")
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    print(f"✅ 站点标注图已保存: {save_path2}")
    plt.close()

# ================= 主程序 =================

if __name__ == "__main__":
    if not os.path.exists(INPUT_CSV):
        print(f"❌ 错误：找不到文件 {INPUT_CSV}")
    else:
        # 1. 读取数据
        df_full = pd.read_csv(INPUT_CSV)
        
        # 2. 执行优势域选点
        df_sites = select_snac_specialized_sites(df_full)
        
        # 3. 保存坐标
        csv_save_path = os.path.join(OUTPUT_DIR, "final_12_sites_specialized.csv")
        df_sites.to_csv(csv_save_path, index=False)
        print(f"✅ 12 个优势域站点已选定: {csv_save_path}")
        print(df_sites[['Group', 'Tag', 'Lat', 'Lon']])
        
        # 4. 绘图 (应用您的样式)
        plot_maps(df_full, df_sites)
