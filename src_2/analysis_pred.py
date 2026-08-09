#!/usr/bin/env python3
"""
风场预测可视化脚本 - 优化布局与形变修复版 (批量运行版)
使用 ParaView Fast 配色方案
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from pathlib import Path
from tqdm import tqdm
import json
import os

try:
    from data_provider.data_factory import data_provider
    from model.model_factory import get_model
except ImportError:
    print("Warning: Could not import project modules. Ensure data_provider and model are in path.")

LOCATIONS = {
    "shanghai": (31.23, 121.47),
    "chengdu": (30.67, 104.06),
    "hohhot": (40.84, 111.75),
    "qingdao": (36.06, 120.38),
    "guangzhou": (23.13, 113.26),
    "urumqi": (43.82, 87.62),
    "lhasa": (29.65, 91.11),
    "golmud": (36.40, 94.90),
    "kunming": (25.04, 102.71),
    "wuhan": (30.59, 114.30),
    "fuzhou": (26.08, 119.30),
    "haikou": (20.03, 110.32),
}

MODELS = [
    "predrnn", 
    "trajgru", 
    "convlstm", 
    "convgru", 
    "kan", 
    "cnn", 
    "fno", 
    "unet"
]

# ============================================
# 配色方案实现 (未修改)
# ============================================

def get_fast_colormap():
    """创建 ParaView Fast 配色方案"""
    rgb_points = [
        0,      0.0564, 0.0564, 0.4700,
        0.1716, 0.2430, 0.4604, 0.8100,
        0.2985, 0.3568, 0.7450, 0.9544,
        0.4321, 0.6882, 0.9300, 0.9179,
        0.5000, 0.8995, 0.9446, 0.7687,
        0.5882, 0.9571, 0.8338, 0.5089,
        0.7061, 0.9275, 0.6214, 0.3154,
        0.8476, 0.8000, 0.3520, 0.1600,
        1.0000, 0.5900, 0.0767, 0.1195
    ]
    positions = [rgb_points[i] for i in range(0, len(rgb_points), 4)]
    colors = [[rgb_points[i+1], rgb_points[i+2], rgb_points[i+3]] for i in range(0, len(rgb_points), 4)]
    return mcolors.LinearSegmentedColormap.from_list('Fast', list(zip(positions, colors)), N=256)

def get_fast_diverging_colormap():
    """创建用于误差图的发散配色"""
    colors = [
        (0.0564, 0.0564, 0.4700), (0.3568, 0.7450, 0.9544), (1.0000, 1.0000, 1.0000),
        (0.9275, 0.6214, 0.3154), (0.5900, 0.0767, 0.1195)
    ]
    return mcolors.LinearSegmentedColormap.from_list('Fast_Diverging', colors, N=256)

# ============================================
# 数据处理与推理 (未修改)
# ============================================

def denormalize_wind(data, stats_file):
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    u_mean, u_std = stats['u10']['mean'], stats['u10']['std']
    v_mean, v_std = stats['v10']['mean'], stats['v10']['std']
    data_denorm = data.copy()
    data_denorm[..., 0, :, :] = data[..., 0, :, :] * u_std + u_mean
    data_denorm[..., 1, :, :] = data[..., 1, :, :] * v_std + v_mean
    return data_denorm

def compute_wind_speed(u, v):
    return np.sqrt(u**2 + v**2)

def load_model_and_predict(args, device):
    _, test_loader = data_provider(args, 'test')
    model = get_model(args, device)
    
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
        
    state_dict = torch.load(args.checkpoint_path, map_location=device)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    if "_metadata" in state_dict:
        del state_dict["_metadata"]
        
    model.load_state_dict(state_dict)
    model.eval()

    needs_pred_len = args.model.lower() in ['predrnn', 'trajgru', 'convlstm', 'convgru']
    all_preds, all_trues = [], []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Inference", leave=False)):
            if len(all_preds) * args.batch_size >= args.num_samples: break
            x_vars = batch['x_vars'].to(device, non_blocking=True)
            t_feat = batch['t_feat'].to(device, non_blocking=True)
            y = batch['y'].to(device, non_blocking=True)
            
            B, T, _, H, W = x_vars.shape
            x = torch.cat([x_vars, t_feat.view(B, T, 4, 1, 1).expand(-1, -1, -1, H, W)], dim=2)
            pred = model(x, pred_len=args.pred_len) if needs_pred_len else model(x)
            
            all_preds.append(pred.cpu().numpy())
            all_trues.append(y.cpu().numpy())
            
    preds = denormalize_wind(np.concatenate(all_preds, axis=0)[:args.num_samples], args.stats_file)
    trues = denormalize_wind(np.concatenate(all_trues, axis=0)[:args.num_samples], args.stats_file)
    return preds, trues


def plot_wind_field_comparison(pred, true, sample_idx, t_idx, save_path, city, model_name, center_lat, center_lon):
    u_t, v_t = true[t_idx, 0], true[t_idx, 1]
    u_p, v_p = pred[t_idx, 0], pred[t_idx, 1]
    speed_t, speed_p = compute_wind_speed(u_t, v_t), compute_wind_speed(u_p, v_p)
    error = speed_p - speed_t

    # --- 坐标系映射计算 ---
    H, W = speed_t.shape
    # 请根据您的实际数据分辨率修改 dx 和 dy。例如 ERA5 通常为 0.25 度
    dx = 0.25  
    dy = 0.25  
    
    # 计算以城市为中心的经纬度边界: [左, 右, 下, 上]
    lon_min = center_lon - (W / 2) * dx
    lon_max = center_lon + (W / 2) * dx
    lat_min = center_lat - (H / 2) * dy
    lat_max = center_lat + (H / 2) * dy
    map_extent = [lon_min, lon_max, lat_min, lat_max]
    # -----------------------

    fig = plt.figure(figsize=(24, 8), constrained_layout=True)
    gs = GridSpec(1, 5, figure=fig, width_ratios=[1, 1, 1, 0.04, 0.04], wspace=0.05)
    
    fast_cmap = get_fast_colormap()
    div_cmap = get_fast_diverging_colormap()
    
    vmin, vmax = min(speed_t.min(), speed_p.min()), max(speed_t.max(), speed_p.max())
    elim = max(abs(error.min()), abs(error.max())) * 1.1

    axes_data = [
        (speed_t, 'Ground Truth', fast_cmap, vmin, vmax),
        (speed_p, f'Prediction ({model_name.upper()})', fast_cmap, vmin, vmax),
        (error, 'Error (Pred - Truth)', div_cmap, -elim, elim)
    ]

    axes = []
    for i, (data, title, cmap, _min, _max) in enumerate(axes_data):
        ax = fig.add_subplot(gs[i])
        # 传入 extent 参数，将像素映射到经纬度
        im = ax.imshow(data, cmap=cmap, vmin=_min, vmax=_max, origin='lower', aspect='equal', extent=map_extent)
        ax.set_title(title, fontsize=22, fontweight='bold', pad=15)
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(1.2)
        
        # 修改为更标准的经纬度标签
        ax.set_xlabel('Longitude (°E)', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        if i == 0:
            ax.set_ylabel('Latitude (°N)', fontsize=20)
        axes.append((ax, im))

    # Colorbar 1: Speed
    cbar_ax1 = fig.add_subplot(gs[3])
    cb1 = fig.colorbar(axes[1][1], cax=cbar_ax1)
    cb1.set_label('Speed (m/s)', fontsize=20)
    cb1.ax.tick_params(labelsize=18)
    
    # Colorbar 2: Error
    cbar_ax2 = fig.add_subplot(gs[4])
    cb2 = fig.colorbar(axes[2][1], cax=cbar_ax2)
    cb2.set_label('Error (m/s)', fontsize=20)
    cb2.ax.tick_params(labelsize=18)

    rmse = np.sqrt(np.mean(error**2))
    axes[2][0].text(0.05, 0.95, f'RMSE: {rmse:.3f}', transform=axes[2][0].transAxes, 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'), verticalalignment='top', fontsize=14)

    fig.suptitle(f'{city.upper()} | Time Step: {t_idx+1}', fontsize=22, fontweight='bold')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

def plot_wind_speed_evolution(pred, true, save_path, city, model_name):
    T = pred.shape[0]
    sp_p = [compute_wind_speed(pred[t,0], pred[t,1]).mean() for t in range(T)]
    sp_t = [compute_wind_speed(true[t,0], true[t,1]).mean() for t in range(T)]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), constrained_layout=True)
    
    # 折线图
    ax1.plot(range(T), sp_t, 'o-', color='#0E0E78', label='Truth', lw=2, markersize=8)
    ax1.plot(range(T), sp_p, 's-', color='#96131E', label='Pred', lw=2, markersize=8)
    ax1.set_title(f'{city.upper()} - Wind Speed Trend', fontsize=20, fontweight='bold', pad=20)
    ax1.set_ylabel('Wind Speed (m/s)', fontsize=18)
    # 调大刻度字体和图例字体
    ax1.tick_params(axis='both', labelsize=24)
    ax1.legend(fontsize=24)
    ax1.grid(alpha=0.3)
    for spine in ax1.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.2)

    # 误差箱线图
    err_dist = [(compute_wind_speed(pred[t,0], pred[t,1]) - compute_wind_speed(true[t,0], true[t,1])).flatten() for t in range(T)]
    ax2.boxplot(err_dist, positions=range(T), patch_artist=True, boxprops=dict(facecolor='#CC5A29', alpha=0.6))
    ax2.axhline(0, color='red', linestyle='--', alpha=0.5, lw=2)
    ax2.set_title('Spatial Error Distribution per Step', fontsize=20, fontweight='bold', pad=20)
    ax2.set_ylabel('Error (m/s)', fontsize=24)
    ax2.set_xlabel('Forecast Hour', fontsize=24)
    # 调大刻度字体
    ax2.tick_params(axis='both', labelsize=20)
    for spine in ax2.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.2)
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

# ============================================
# 主逻辑 (修改为双重循环批量运行)
# ============================================

def main():
    parser = argparse.ArgumentParser(description='Wind Field Visualization - Batch Mode')
    project_root = Path(os.environ.get("WIND_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
    # 移除了命令行输入的 --city 和 --model，由代码内部的循环接管
    
    parser.add_argument('--base_dir', type=str, default=str(project_root / '7-24'))
    parser.add_argument('--nc_dir', type=str, default=str(project_root / 'npy_data'))
    parser.add_argument('--som_map_path', type=str, default=str(project_root / 'som_map.npy'))
    parser.add_argument('--output_dir', type=str, default='./wind_visualizations_7_24')
    
    parser.add_argument('--seq_len', type=int, default=168)
    parser.add_argument('--pred_len', type=int, default=24)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_samples', type=int, default=3)
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Starting batch processing for {len(LOCATIONS)} cities and {len(MODELS)} models.")

    # 双重循环：遍历所有城市和所有模型
    for city_name, (lat, lon) in LOCATIONS.items():
        for model_name in MODELS:
            print(f"\n{'-'*60}")
            print(f">>> Processing [City: {city_name.upper()}] with [Model: {model_name}]")
            print(f"{'-'*60}")

            # 动态补全 data_provider 每次循环所需的参数
            args.city = city_name
            args.model = model_name
            args.target_lat = lat
            args.target_lon = lon
            args.fix_location = True
            args.features = 'M'
            args.embed = 'timeF'
            args.freq = 'h'
            args.model_name = model_name
            args.test_location_config = {'type': 'fixed', 'lat': lat, 'lon': lon}
            
            args.stats_file = os.path.join(args.nc_dir, 'normalization_stats.json')
            args.checkpoint_path = os.path.join(args.base_dir, f'analysis_results_{city_name}', model_name, 'checkpoint.pth')
            
            output_dir = Path(args.output_dir) / city_name / model_name
            output_dir.mkdir(parents=True, exist_ok=True)

            # 检查 checkpoint 文件是否存在，避免中断整个批量任务
            if not os.path.exists(args.checkpoint_path):
                print(f"    [Skip] Checkpoint not found: {args.checkpoint_path}")
                continue

            try:
                preds, trues = load_model_and_predict(args, device)

                for i in range(len(preds)):
                    T_max = preds[i].shape[0]
                    
                    # 边界保护：确保请求的时间步没有超出模型输出长度
                    for t in [0, 6, 12, 18, 23]:
                        if t >= T_max:
                            continue
                        save_path = output_dir / f"sample_{i}_t{t:02d}_comp.png"
                        plot_wind_field_comparison(preds[i], trues[i], i, t, save_path, city_name, model_name, lat, lon)
                    
                    evol_save_path = output_dir / f"sample_{i}_evol.png"
                    plot_wind_speed_evolution(preds[i], trues[i], evol_save_path, city_name, model_name)

                print(f"    [Success] Results saved in: {output_dir.absolute()}")

            except Exception as e:
                print(f"    [Error] Failed processing {city_name} with {model_name}. Error details: {str(e)}")
                continue

    print("\n>>> All batch tasks completed.")

if __name__ == "__main__":
    main()
