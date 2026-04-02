import argparse
import torch
import numpy as np
import pandas as pd
import os
import sys
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
    
    # 核心参数
    parser.add_argument('--model_name', type=str, required=True, help='cnn, kan, mlp')
    parser.add_argument('--nc_dir', type=str, default='/mimer/NOBACKUP/groups/kthmech/junle/guandi_wind_project/data_2/npy_data')
    parser.add_argument('--stats_file', type=str, default='normalization_stats.json')
    parser.add_argument('--output_dir', type=str, default='/mimer/NOBACKUP/groups/kthmech/junle/guandi_wind_project/analysis_results_lhasa')
    parser.add_argument('--som_map_path', type=str, default='/mimer/NOBACKUP/groups/kthmech/junle/guandi_wind_project/data_2/som/som_map.npy')
    # 模型参数
    parser.add_argument('--seq_len', type=int, default=24)
    parser.add_argument('--pred_len', type=int, default=24)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--freq', type=str, default='h')
#"Shanghai":  [121.47, 31.23],36.40 94.90, 29.65 91.11
#"Chengdu":   [104.06, 30.67], [111.75, 40.84]  [120.38, 36.06]
    # 地理参数
    parser.add_argument('--target_lat', type=float, default=91.11)
    parser.add_argument('--target_lon', type=float, default=29.65)
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
    # 移除 NaN
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
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            
            pred = model(x)
            
            # 1. 调用 Evaluator 计算全图指标 (Patch Level)
            # 返回格式: {'U': {'RMSE':...}, 'Speed': {'RMSE':...}}
            all_patch_metrics.append(evaluator.evaluate_batch(pred, y))
            
            # 2. 调用 Evaluator 提取单点数据 (已反归一化)
            point_data = evaluator.evaluate_single_point(pred, y, window_size=args.eval_window_size)
            all_preds_list.append(point_data['preds_np'])
            all_trues_list.append(point_data['trues_np'])

    # --- 逻辑汇总 ---
    
    # A. 全图指标平均 (展平字典结构)
    # 将嵌套字典 {'U': {'RMSE': 1}} 展平为 Patch_U_RMSE: 1
    patch_df = pd.DataFrame([
        {f"Patch_{var}_{metric}": val 
         for var, metrics in m.items() 
         for metric, val in metrics.items()} 
        for m in all_patch_metrics
    ])
    final_results = patch_df.mean().to_dict()
    
    # B. 拼接中心点数据
    total_p = np.concatenate(all_preds_list, axis=0)
    total_t = np.concatenate(all_trues_list, axis=0)
    
    # C. 调用 Evaluator 静态方法计算全量统计指标 (解决相关系数 R 计算的严谨性)
    point_stats = Evaluator.summarize_stats(total_p.flatten(), total_t.flatten())
    final_results.update({f"SinglePoint_{k}": v for k, v in point_stats.items()})
    final_results.update({'Lat': args.target_lat, 'Lon': args.target_lon})
    
    # D. 保存数据与误差分布图
    np.save(f"{save_prefix}_point_pred.npy", total_p)
    np.save(f"{save_prefix}_point_true.npy", total_t)
    print(f"✅ NPY data saved to {save_prefix}_*.npy")
    
    plot_error_dist(total_p.flatten() - total_t.flatten(), save_prefix)
    
    # 返回给 main 函数用于保存 CSV
    return final_results, total_p, total_t

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 路径准备
    model_dir = os.path.join(args.output_dir, args.model_name)
    checkpoint_path = os.path.join(model_dir, 'checkpoint.pth')
    
    # 为了区别，我们可以叫 result_recover，也可以直接覆盖叫 result
    # 这里使用 result_recover 以免误删，你也可以改成 'result'
    save_prefix = os.path.join(model_dir, 'final_eval')
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    print(f"=== Recovering Test for: {args.model_name.upper()} ===")
    print(f"Output Path: {model_dir}")

    # 1. 加载组件
    _, test_loader = data_provider(args, 'test')
    model = get_model(args, device)
    
    # 2. 加载权重
    print(f">>> Loading weights from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    if "_metadata" in state_dict: del state_dict["_metadata"]
    model.load_state_dict(state_dict)
    
    # 3. 初始化评估器
    evaluator = Evaluator(stats_file=args.stats_file, device=device)
    
    # 4. 运行测试 (完全一致的逻辑)
    # 注意：这里的 test 返回三个值，适配上面的 test 函数定义
    final_metrics, preds, trues = test(model, test_loader, evaluator, save_prefix, args)
    
    # 5. 保存结果 CSV
    print("\n=== Final Metrics ===")
    print(pd.Series(final_metrics))
    
    csv_path = f"{save_prefix}_metrics.csv"
    pd.DataFrame([final_metrics]).to_csv(csv_path, index=False)
    print(f"✅ Metrics saved to: {os.path.abspath(csv_path)}")
    
    # 6. 生成时序对比图 (BenchmarkVisualizer)
    # 这部分通常在 run_models 的 main 后面，这里也补上
    print(">>> Generating Time-Series Plot...")
    visualizer = BenchmarkVisualizer(output_dir=model_dir)
    
    # 维度检查与降维 (Robustness)
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

if __name__ == "__main__":
    main()