import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import xarray as xr
from tqdm import tqdm
from collections import defaultdict

# ================= 配置区域 =================
BASE_DIR = '/mimer/NOBACKUP/groups/kthmech/junle/guandi_wind_project/data_2'
SOM_PATH = os.path.join(BASE_DIR, 'som/som_map.npy')
NC_DIR = os.path.join(BASE_DIR, 'processed')

# 采样策略配置
SAMPLING_STRATEGY = 'stratified'  # 'dense' | 'stratified' | 'full'
SAMPLES_PER_CLUSTER = 200    # 每个聚类的目标采样点数
VISUALIZE_SAMPLING = True    # 是否可视化采样分布

# 🔥 绘图优化配置（关键：避免 KDE 太慢）
MAX_SAMPLES_FOR_PLOT = 50000  # 绘图时每个聚类最多使用的样本数

# 聚类配色
CLUSTER_COLORS = {
    "C1": "#8FB1CC", "C2": "#E6B9C0", "C3": "#A8C8A6", 
    "C4": "#E8D595", "C5": "#C6B4CE", "C6": "#D4C4B7", 
    "C7": "#D8A49B", "C8": "#7B8D8E", "C9": "#9DAD7F"
}

# 🔥 聚类区域描述（替代具体城市名）
CLUSTER_REGIONS = {
    "C1": "East Coast",      
    "C2": "South China",     
    "C3": "Central China",   
    "C4": "North China",     
    "C5": "Southwest",       
    "C6": "Northwest",       
    "C7": "Tibetan Plateau", 
    "C8": "Qinghai Region",         
    "C9": "Yunnan-Guizhou"   
}

# 代表性城市（仅用于采样可视化标注）
REPRESENTATIVE_CITIES = {
    "C1": "Shanghai", "C2": "Guangzhou", "C3": "Wuhan",
    "C4": "Hohhot", "C5": "Chengdu", "C6": "Urumqi",
    "C7": "Lhasa", "C8": "Golmud", "C9": "Kunming"
}

# 字体设置
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Liberation Serif', 'serif']
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# ================= 核心采样逻辑 =================

def load_som_map(som_path):
    """加载 SOM 聚类结果"""
    som_map = np.load(som_path)
    print(f"SOM map shape: {som_map.shape}")
    print(f"Unique clusters: {np.unique(som_map)}")
    
    # 统计每个聚类的网格点数
    for cluster_id in range(9):
        n_points = np.sum(som_map == cluster_id)
        print(f"  C{cluster_id+1}: {n_points} grid points")
    
    return som_map

def generate_sampling_indices_stratified(som_map, samples_per_cluster=200):
    """
    分层随机采样：确保空间分布均匀
    
    策略：
    1. 将每个聚类的空间范围划分为 k×k 网格
    2. 每个网格随机采样 1 个点
    3. 确保采样点在空间上分布均匀
    """
    cluster_indices = defaultdict(list)
    np.random.seed(42)  # 可复现
    
    print(f"\n=== 使用分层随机采样策略 ===")
    
    for cluster_id in range(9):
        coords = np.argwhere(som_map == cluster_id)
        n_available = len(coords)
        
        if n_available == 0:
            print(f"  C{cluster_id+1}: ❌ No points found!")
            continue
        
        # 如果点数太少，全部采样
        if n_available <= samples_per_cluster:
            cluster_indices[cluster_id] = [(int(y), int(x)) for y, x in coords]
            print(f"  C{cluster_id+1}: {n_available}/{n_available} points (全采样)")
            continue
        
        # 分层采样
        # 计算网格大小 (使 k×k ≈ samples_per_cluster)
        k = int(np.sqrt(samples_per_cluster)) + 1
        
        lat_min, lat_max = coords[:, 0].min(), coords[:, 0].max()
        lon_min, lon_max = coords[:, 1].min(), coords[:, 1].max()
        
        # 创建网格边界
        lat_bins = np.linspace(lat_min, lat_max + 1, k + 1)
        lon_bins = np.linspace(lon_min, lon_max + 1, k + 1)
        
        sampled_coords = []
        
        # 遍历每个网格单元
        for i in range(k):
            for j in range(k):
                # 找到该网格内的所有点
                mask = ((coords[:, 0] >= lat_bins[i]) & 
                        (coords[:, 0] < lat_bins[i+1]) &
                        (coords[:, 1] >= lon_bins[j]) & 
                        (coords[:, 1] < lon_bins[j+1]))
                
                candidates = coords[mask]
                
                # 如果该网格有点，随机选 1 个
                if len(candidates) > 0:
                    idx = np.random.randint(len(candidates))
                    sampled_coords.append(candidates[idx])
        
        cluster_indices[cluster_id] = [(int(y), int(x)) for y, x in sampled_coords]
        sampling_rate = 100 * len(sampled_coords) / n_available
        print(f"  C{cluster_id+1}: {len(sampled_coords)}/{n_available} points "
              f"(分层采样 {k}×{k}网格, {sampling_rate:.1f}%)")
    
    return cluster_indices

def generate_sampling_indices_dense(som_map, samples_per_cluster=200):
    """
    简单随机采样（原方法）
    """
    cluster_indices = defaultdict(list)
    np.random.seed(42)
    
    print(f"\n=== 使用简单随机采样策略 ===")
    
    for cluster_id in range(9):
        coords = np.argwhere(som_map == cluster_id)
        n_available = len(coords)
        
        if n_available == 0:
            print(f"  C{cluster_id+1}: ❌ No points found!")
            continue
        
        n_samples = min(samples_per_cluster, n_available)
        sampled_idx = np.random.choice(n_available, size=n_samples, replace=False)
        sampled_coords = coords[sampled_idx]
        
        cluster_indices[cluster_id] = [(int(y), int(x)) for y, x in sampled_coords]
        sampling_rate = 100 * n_samples / n_available
        print(f"  C{cluster_id+1}: {n_samples}/{n_available} points "
              f"(随机采样, {sampling_rate:.1f}%)")
    
    return cluster_indices

def generate_sampling_indices_full(som_map):
    """
    全量统计（不采样）
    """
    cluster_indices = defaultdict(list)
    
    print(f"\n=== 使用全量统计（无采样） ===")
    
    for cluster_id in range(9):
        coords = np.argwhere(som_map == cluster_id)
        cluster_indices[cluster_id] = [(int(y), int(x)) for y, x in coords]
        print(f"  C{cluster_id+1}: {len(coords)} points (全量)")
    
    return cluster_indices

def generate_sampling_indices(som_map, strategy='stratified', samples_per_cluster=200):
    """采样入口函数"""
    if strategy == 'stratified':
        return generate_sampling_indices_stratified(som_map, samples_per_cluster)
    elif strategy == 'dense':
        return generate_sampling_indices_dense(som_map, samples_per_cluster)
    elif strategy == 'full':
        return generate_sampling_indices_full(som_map)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def visualize_sampling(som_map, cluster_indices, save_dir='sampling_vis'):
    """
    可视化采样点分布
    为每个聚类生成一张图，显示采样点在空间上的分布
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n=== 生成采样可视化 ===")
    
    # 创建总览图
    fig_overview, axes_overview = plt.subplots(3, 3, figsize=(15, 15), dpi=150)
    axes_overview = axes_overview.flatten()
    
    for cluster_id in range(9):
        # 创建该聚类的掩码
        mask = (som_map == cluster_id).astype(float)
        mask[mask == 0] = np.nan
        
        # 单独保存每个聚类
        fig_single, ax_single = plt.subplots(figsize=(8, 8), dpi=150)
        
        # 背景：该聚类的所有点
        ax_single.imshow(mask, cmap='Blues', alpha=0.4, interpolation='nearest')
        
        # 采样点
        if cluster_id in cluster_indices:
            sampled = cluster_indices[cluster_id]
            if len(sampled) > 0:
                sampled_y = [coord[0] for coord in sampled]
                sampled_x = [coord[1] for coord in sampled]
                
                ax_single.scatter(sampled_x, sampled_y, c='red', s=15, 
                                alpha=0.6, edgecolors='darkred', linewidths=0.5,
                                label=f'Sampled: {len(sampled)} points')
        
        total_points = np.sum(som_map == cluster_id)
        sampling_rate = 100 * len(sampled) / total_points if total_points > 0 else 0
        
        ax_single.set_title(f'C{cluster_id+1}: {REPRESENTATIVE_CITIES[f"C{cluster_id+1}"]}\n'
                          f'Sampled: {len(sampled)}/{total_points} ({sampling_rate:.1f}%)',
                          fontsize=16, fontweight='bold')
        ax_single.legend(loc='upper right')
        ax_single.set_xlabel('Longitude Index')
        ax_single.set_ylabel('Latitude Index')
        ax_single.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'sampling_C{cluster_id+1}.png'), 
                   bbox_inches='tight', dpi=150)
        plt.close(fig_single)
        
        # 总览图
        ax = axes_overview[cluster_id]
        ax.imshow(mask, cmap='Blues', alpha=0.4, interpolation='nearest')
        if cluster_id in cluster_indices and len(sampled) > 0:
            ax.scatter(sampled_x, sampled_y, c='red', s=5, alpha=0.6)
        ax.set_title(f'C{cluster_id+1}: {len(sampled)}/{total_points}', fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sampling_overview.png'), 
               bbox_inches='tight', dpi=150)
    plt.close(fig_overview)
    
    print(f"  ✅ 采样可视化保存至 {save_dir}/")

def extract_cluster_data(nc_dir, cluster_indices):
    """
    从 NC 文件提取各聚类的风速数据
    
    Returns:
    --------
    cluster_data : dict
        {cluster_id: np.ndarray of wind speeds}
    """
    nc_files = sorted(glob.glob(os.path.join(nc_dir, '*.nc')))
    if not nc_files:
        raise FileNotFoundError(f"No .nc files found in {nc_dir}")
    
    print(f"\n=== 从 {len(nc_files)} 个 NC 文件提取数据 ===")
    
    # 初始化数据缓冲区
    cluster_data_buffer = {cid: [] for cid in cluster_indices.keys()}
    
    for nc_file in tqdm(nc_files, desc="Processing NC files"):
        try:
            with xr.open_dataset(nc_file) as ds:
                # 确定变量名
                u_var = 'u10' if 'u10' in ds else 'u'
                v_var = 'v10' if 'v10' in ds else 'v'
                
                # 读取 u, v 数据
                u_data = ds[u_var].values  # shape: (time, lat, lon) 或 (lat, lon)
                v_data = ds[v_var].values
                
                # 处理时间维度
                if u_data.ndim == 3:
                    # 有时间维度，逐时次处理
                    for t in range(u_data.shape[0]):
                        u_t = u_data[t]
                        v_t = v_data[t]
                        
                        for cluster_id, indices in cluster_indices.items():
                            for lat_idx, lon_idx in indices:
                                try:
                                    u_val = u_t[lat_idx, lon_idx]
                                    v_val = v_t[lat_idx, lon_idx]
                                    ws = np.sqrt(u_val**2 + v_val**2)
                                    cluster_data_buffer[cluster_id].append(ws)
                                except IndexError:
                                    continue
                else:
                    # 无时间维度
                    for cluster_id, indices in cluster_indices.items():
                        for lat_idx, lon_idx in indices:
                            try:
                                u_val = u_data[lat_idx, lon_idx]
                                v_val = v_data[lat_idx, lon_idx]
                                ws = np.sqrt(u_val**2 + v_val**2)
                                cluster_data_buffer[cluster_id].append(ws)
                            except IndexError:
                                continue
        
        except Exception as e:
            print(f"\n  ⚠️  Warning: Failed to process {os.path.basename(nc_file)}: {e}")
            continue
    
    # 转换为 numpy 数组并输出统计
    cluster_data = {}
    print("\n=== 数据提取统计 ===")
    for cluster_id in sorted(cluster_data_buffer.keys()):
        data_list = cluster_data_buffer[cluster_id]
        cluster_data[cluster_id] = np.array(data_list) if data_list else np.array([])
        
        if len(cluster_data[cluster_id]) > 0:
            mean_ws = np.mean(cluster_data[cluster_id])
            std_ws = np.std(cluster_data[cluster_id])
            print(f"  C{cluster_id+1}: {len(cluster_data[cluster_id]):,} samples "
                  f"(μ={mean_ws:.2f} m/s, σ={std_ws:.2f} m/s)")
        else:
            print(f"  C{cluster_id+1}: ❌ No data")
    
    return cluster_data

# ================= 绘图函数 =================

def subsample_for_plotting(data, max_samples=50000):
    """
    对大数据集降采样用于绘图
    
    Parameters:
    -----------
    data : np.ndarray
        原始数据
    max_samples : int
        最大保留样本数
    
    Returns:
    --------
    subsampled_data : np.ndarray
        降采样后的数据
    """
    if len(data) <= max_samples:
        return data
    else:
        np.random.seed(42)
        indices = np.random.choice(len(data), max_samples, replace=False)
        return data[indices]

def plot_distributions(cluster_data, max_samples_for_plot=50000):
    """绘制风速分布图（优化版）"""
    
    # 数据量诊断
    print("\n=== 数据量诊断 ===")
    for cluster_id in range(9):
        ws_data = cluster_data.get(cluster_id, np.array([]))
        if len(ws_data) > 0:
            print(f"C{cluster_id+1}: {len(ws_data):,} samples ({len(ws_data)/1e6:.2f}M)")
    
    # 1. 单独分布图 (3x3 布局)
    print("\n=== 绘制单独分布图 ===")
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12), dpi=300)
    axes = axes.flatten()
    
    for cluster_id in range(9):
        print(f"  正在绘制 C{cluster_id+1}...", end='', flush=True)
        
        ax = axes[cluster_id]
        ws_data = cluster_data.get(cluster_id, np.array([]))
        color = CLUSTER_COLORS[f"C{cluster_id+1}"]
        region_name = CLUSTER_REGIONS[f"C{cluster_id+1}"]  # 🔥 使用区域名
        
        if len(ws_data) > 100:
            # 统计值用全部数据
            mean_val = np.mean(ws_data)
            std_val = np.std(ws_data)
            median_val = np.median(ws_data)
            
            # 🔥 降采样用于绘图
            ws_data_plot = subsample_for_plotting(ws_data, max_samples_for_plot)
            if len(ws_data) > max_samples_for_plot:
                print(f" (降采样 {len(ws_data):,} -> {len(ws_data_plot):,})", end='')
            
            try:
                sns.kdeplot(ws_data_plot, ax=ax, fill=True, color=color,
                           alpha=0.6, linewidth=1.5, cut=0, 
                           bw_adjust=0.5, warn_singular=False)
                print(" ✓")
            except Exception as e:
                print(f" 使用直方图")
                ax.hist(ws_data_plot, bins=100, color=color, alpha=0.6, 
                       density=True, edgecolor='none')
            
            ax.set_xlim(0, 18)
            ax.axvline(mean_val, color='darkred', linestyle='--', 
                      linewidth=1.2, alpha=0.7)
        else:
            mean_val, std_val, median_val = 0, 0, 0
            ax.text(0.5, 0.5, "No Data", transform=ax.transAxes,
                   ha='center', va='center', fontsize=12, color='#999')
            print(" (无数据)")
        
        ax.set_ylim(bottom=0)
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # 只在最后一行显示 x 轴标签
        if cluster_id >= 6:
            ax.set_xlabel("Wind Speed (m/s)", fontsize=14, fontweight='bold')
        else:
            ax.set_xlabel("")
        
        # 🔥 标题：使用区域名
        idx_char = chr(97 + cluster_id)
        ax.set_title(f"({idx_char}) C{cluster_id+1}: {region_name}",
                    fontweight='bold', fontsize=16, loc='left', pad=10)
        
        # 统计信息（使用完整数据的统计值）
        if len(ws_data) > 100:
            n_samples = len(ws_data)
            if n_samples >= 1e6:
                n_str = f"n={n_samples/1e6:.1f}M"
            elif n_samples >= 1e3:
                n_str = f"n={n_samples/1e3:.0f}k"
            else:
                n_str = f"n={n_samples}"
            
            stats_text = f"{n_str}\nμ={mean_val:.2f}\nσ={std_val:.2f}\nmed={median_val:.2f}"
            ax.text(0.97, 0.85, stats_text,
                   transform=ax.transAxes, fontsize=14, family='monospace',
                   ha='right', va='top',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                            edgecolor='gray', alpha=0.9, linewidth=0.5))
    
    plt.tight_layout()
    print("  保存图片...")
    plt.savefig('wind_distributions_cluster_based.png', bbox_inches='tight', dpi=300)
    print("  ✅ 单独分布图保存为: wind_distributions_cluster_based.png")
    plt.close()
    
    # 2. 叠加分布图
    print("\n=== 绘制叠加分布图 ===")
    fig_overlay, ax_overlay = plt.subplots(figsize=(12, 7), dpi=300)
    
    for cluster_id in range(9):
        print(f"  正在叠加 C{cluster_id+1}...", end='', flush=True)
        
        ws_data = cluster_data.get(cluster_id, np.array([]))
        color = CLUSTER_COLORS[f"C{cluster_id+1}"]
        region_name = CLUSTER_REGIONS[f"C{cluster_id+1}"]  # 🔥 使用区域名
        
        if len(ws_data) > 100:
            mean_val = np.mean(ws_data)
            ws_data_plot = subsample_for_plotting(ws_data, max_samples_for_plot)
            
            try:
                sns.kdeplot(ws_data_plot, ax=ax_overlay, color=color,
                           linewidth=2.5, alpha=0.8,
                           label=f"C{cluster_id+1}: {region_name} (μ={mean_val:.2f})",  # 🔥 修改图例
                           cut=0, bw_adjust=0.5, warn_singular=False)
                print(" ✓")
            except Exception as e:
                print(f" 直方图")
                ax_overlay.hist(ws_data_plot, bins=100, color=color, alpha=0.3,
                               density=True, 
                               label=f"C{cluster_id+1}: {region_name} (μ={mean_val:.2f})")
    
    ax_overlay.set_xlim(0, 18)
    ax_overlay.set_ylim(bottom=0)
    ax_overlay.set_xlabel("Wind Speed (m/s)", fontsize=18, fontweight='bold')
    ax_overlay.set_ylabel("Probability Density", fontsize=18, fontweight='bold')
    ax_overlay.set_title("Wind Speed Distributions by SOM Clusters (Cluster-wide Statistics)",
                        fontsize=19, fontweight='bold', pad=20)
    
    ax_overlay.spines['top'].set_visible(False)
    ax_overlay.spines['right'].set_visible(False)
    ax_overlay.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 图例
    ax_overlay.legend(loc='upper right', frameon=True, framealpha=0.95, 
                     fontsize=18, ncol=1, edgecolor='gray', fancybox=True)
    
    plt.tight_layout()
    print("  保存图片...")
    plt.savefig('wind_distributions_cluster_overlay.png', bbox_inches='tight', dpi=300)
    print("  ✅ 叠加分布图保存为: wind_distributions_cluster_overlay.png")
    plt.close()

# ================= 主函数 =================

def main():
    print("=" * 60)
    print("   Wind Speed Distribution Analysis by SOM Clusters")
    print("=" * 60)
    
    # 1. 加载 SOM 聚类结果
    print("\n【步骤 1】加载 SOM 聚类结果")
    som_map = load_som_map(SOM_PATH)
    
    # 2. 生成采样索引
    print("\n【步骤 2】生成采样索引")
    cluster_indices = generate_sampling_indices(
        som_map, 
        strategy=SAMPLING_STRATEGY,
        samples_per_cluster=SAMPLES_PER_CLUSTER
    )
    
    # 3. 可视化采样分布（可选）
    if VISUALIZE_SAMPLING:
        print("\n【步骤 3】可视化采样分布")
        visualize_sampling(som_map, cluster_indices)
    
    # 4. 提取风速数据
    print("\n【步骤 4】提取风速数据")
    cluster_data = extract_cluster_data(NC_DIR, cluster_indices)
    
    # 5. 绘制分布图
    print("\n【步骤 5】绘制风速分布图")
    plot_distributions(cluster_data, max_samples_for_plot=MAX_SAMPLES_FOR_PLOT)
    
    

if __name__ == "__main__":
    main()