import argparse
import torch
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

# 设置无头模式，防止服务器报错
matplotlib.use('Agg')

# 引入项目模块
from data_provider.data_factory import data_provider
from model.model_factory import get_model
from evaluation.eval import Evaluator
from evaluation.vis import BenchmarkVisualizer

def get_args():
    parser = argparse.ArgumentParser(description='Recover Test - Exact Logic Copy')
    project_root = Path(os.environ.get("WIND_PROJECT_ROOT", Path(__file__).resolve().parents[1]))

    # 核心参数
    parser.add_argument('--model_name', type=str, required=True, help='cnn, kan, mlp, trajgru')
    parser.add_argument('--nc_dir', type=str, default=str(project_root / 'npy_data'))
    parser.add_argument('--stats_file', type=str, default='normalization_stats.json')
    parser.add_argument('--output_dir', type=str, default=str(project_root / 'analysis_results_lhasa'))
    parser.add_argument('--som_map_path', type=str, default=str(project_root / 'som_map.npy'))

    # 模型参数
    parser.add_argument('--seq_len', type=int, default=168)
    parser.add_argument('--pred_len', type=int, default=24)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--freq', type=str, default='h')

    # 地理参数["guangzhou"]="23.13 113.26"   ["shanghai"]="31.23 121.47"  ["haikou"]="20.03 110.32"
    # ["kunming"]="25.04 102.71" ["urumqi"]="43.82 87.62" ["lhasa"]="29.65 91.11"
    parser.add_argument('--target_lat', type=float, default=29.65)
    parser.add_argument('--target_lon', type=float, default=91.11)
    parser.add_argument('--fix_location', action='store_true', default=True)

    # 关键参数：评估窗口大小
    parser.add_argument('--eval_window_size', type=int, default=1, help='单点评估窗口大小')

    args = parser.parse_args()

    # 构造 Location Config
    args.test_location_config = {
        'type': 'fixed' if args.fix_location else 'random',
        'lat': args.target_lat,
        'lon': args.target_lon
    }

    if not os.path.isabs(args.stats_file):
        args.stats_file = os.path.join(args.nc_dir, args.stats_file)

    return args


def plot_error_dist(errors, save_prefix):
    """绘制误差分布直方图"""
    plt.figure(figsize=(8, 5))
    errors = errors[np.isfinite(errors)]

    plt.hist(errors, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--', label='Zero Error')

    mu, std = np.mean(errors), np.std(errors)
    plt.title(f"Single Point Speed Error Distribution\nMean: {mu:.2f}, Std: {std:.2f}")
    plt.xlabel("Error (m/s)")
    plt.ylabel("Frequency")
    plt.legend()

    save_path = f"{save_prefix}_error_hist.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 Error histogram saved to {save_path}")

def test(model, loader, evaluator, save_prefix, args):
    device = next(model.parameters()).device
    model.eval()

    all_patch_metrics = []
    all_preds_list, all_trues_list = [], []

    print(">>> 启动协调评估流程 (Recover Mode)...")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            # 1. 变量名对齐: 将 'x' 修改为 'x_vars'
            x = batch['x_vars'].to(device)  # [B, T, 5, H, W]
            y = batch['y'].to(device)       # [B, T, 2, H, W]

            # 2. 特征拼接逻辑: 将时间特征 [B, T, 4] 广播并拼接到空间特征上
            t = batch['t_feat'].to(device)  # [B, T, 4]
            B, T, C_t = t.shape
            H, W = x.shape[-2:]

            # 将 t 变形并广播为 [B, T, 4, H, W]
            t_expanded = t.view(B, T, C_t, 1, 1).expand(-1, -1, -1, H, W)

            # 在通道维度拼接，形成 [B, T, 9, H, W]
            x_combined = torch.cat([x, t_expanded], dim=2)

            # 3. 模型调用: 传入拼接后的特征和预测长度参数
            if args.model_name.lower() == 'trajgru':
                pred = model(x_combined, pred_len=args.pred_len)
            else:
                pred = model(x_combined)

            # 评估计算
            all_patch_metrics.append(evaluator.evaluate_batch(pred, y))
            point_data = evaluator.evaluate_single_point(pred, y, window_size=args.eval_window_size)
            all_preds_list.append(point_data['preds_np'])
            all_trues_list.append(point_data['trues_np'])

    # --- 逻辑汇总 ---
    patch_df = pd.DataFrame([
        {f"Patch_{var}_{metric}": val
         for var, metrics in m.items()
         for metric, val in metrics.items()}
        for m in all_patch_metrics
    ])
    final_results = patch_df.mean().to_dict()

    total_p = np.concatenate(all_preds_list, axis=0)
    total_t = np.concatenate(all_trues_list, axis=0)

    point_stats = Evaluator.summarize_stats(total_p.flatten(), total_t.flatten())
    final_results.update({f"SinglePoint_{k}": v for k, v in point_stats.items()})
    final_results.update({'Lat': args.target_lat, 'Lon': args.target_lon})

    np.save(f"{save_prefix}_point_pred.npy", total_p)
    np.save(f"{save_prefix}_point_true.npy", total_t)
    print(f"✅ NPY data saved to {save_prefix}_*.npy")

    plot_error_dist(total_p.flatten() - total_t.flatten(), save_prefix)

    return final_results, total_p, total_t

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_dir = os.path.join(args.output_dir, args.model_name)
    checkpoint_path = os.path.join(model_dir, 'checkpoint.pth')
    save_prefix = os.path.join(model_dir, 'final_eval')

    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    print(f"=== Recovering Test for: {args.model_name.upper()} ===")
    print(f"Output Path: {model_dir}")

    _, test_loader = data_provider(args, 'test')
    model = get_model(args, device)

    print(f">>> Loading weights from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    if "_metadata" in state_dict: del state_dict["_metadata"]
    model.load_state_dict(state_dict)

    evaluator = Evaluator(stats_file=args.stats_file, device=device)

    final_metrics, preds, trues = test(model, test_loader, evaluator, save_prefix, args)

    print("\n=== Final Metrics ===")
    print(pd.Series(final_metrics))

    csv_path = f"{save_prefix}_metrics.csv"
    pd.DataFrame([final_metrics]).to_csv(csv_path, index=False)
    print(f"✅ Metrics saved to: {os.path.abspath(csv_path)}")

    print(">>> Generating Time-Series Plot...")
    visualizer = BenchmarkVisualizer(output_dir=model_dir)

    sample_p = preds[0]
    sample_t = trues[0]
    if sample_p.ndim > 1:
        sample_p = sample_p.mean(axis=tuple(range(1, sample_p.ndim)))
        sample_t = sample_t.mean(axis=tuple(range(1, sample_t.ndim)))

    visualizer.plot_time_series(
        sample_p,
        sample_t,
        title_suffix=f"{args.model_name.upper()} Recovered ({args.target_lat}, {args.target_lon})"
    )
    print(f"✅ All Done! Results in {model_dir}")

import multiprocessing

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
