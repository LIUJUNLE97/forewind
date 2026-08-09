import torch
import os
import numpy as np
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg') # 关键：防止服务器报错
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

from model.model_factory import get_model
from data_provider.data_factory import data_provider
from evaluation.eval import Evaluator
from evaluation.vis import BenchmarkVisualizer
from run_models import test 

def get_args():
    parser = argparse.ArgumentParser()
    project_root = Path(os.environ.get('WIND_PROJECT_ROOT', Path(__file__).resolve().parents[1]))
    default_data_dir = Path(os.environ.get('WIND_NPY_DATA_DIR', project_root / 'npy_data'))
    # -------------------------------------------------------
    # 1. 路径参数
    # -------------------------------------------------------
    parser.add_argument('--output_dir', type=str, default=str(project_root / 'analysis_results_3'), help='Path to load pre-trained model')
    parser.add_argument('--save_dir', type=str, default=str(project_root / 'mlp_singlepoint'), help='Path to save new analysis results')
    parser.add_argument('--nc_dir', type=str, default=str(default_data_dir))
    parser.add_argument('--som_map_path', type=str, default=str(project_root / 'data' / 'metadata' / 'som_map.npy'))
    parser.add_argument('--stats_file', type=str, default='normalization_stats.json')

    # -------------------------------------------------------
    # 2. 模型与数据配置
    # -------------------------------------------------------
    parser.add_argument('--model_name', type=str, default='mlp')
    parser.add_argument('--seq_len', type=int, default=24)
    parser.add_argument('--pred_len', type=int, default=24)
    parser.add_argument('--patch_size', type=int, default=64) # 保持 64
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--freq', type=str, default='h')

    # -------------------------------------------------------
    # 3. 地理定位参数
    # -------------------------------------------------------
    parser.add_argument('--target_lat', type=float, default=31.23)
    parser.add_argument('--target_lon', type=float, default=121.47)
    parser.add_argument('--fix_location', action='store_true', default=True)

    # -------------------------------------------------------
    # 4. 切片窗口大小
    # -------------------------------------------------------
    parser.add_argument('--eval_window_size', type=int, default=1, 
                        help='Window size for center evaluation (1 or 3)')

    args = parser.parse_args()

    args.test_location_config = {
        'type': 'fixed' if args.fix_location else 'random',
        'lat': args.target_lat,
        'lon': args.target_lon
    }
    return args

def extract_center_tensor(pred, true, window_size=1):
    # 获取最后两个维度 (H, W)
    h, w = pred.shape[-2], pred.shape[-1]
    center_h, center_w = h // 2, w // 2
    
    start_offset = window_size // 2
    end_offset = window_size - start_offset
    
    # 计算索引
    start_h = center_h - start_offset
    end_h = center_h + end_offset
    start_w = center_w - start_offset
    end_w = center_w + end_offset
    
    # 切片
    pred_center = pred[..., start_h:end_h, start_w:end_w]
    true_center = true[..., start_h:end_h, start_w:end_w]
    
    return pred_center, true_center

def recover():
    args = get_args()

    # -------------------------------------------------------
    # Logic A: 路径分离逻辑
    # -------------------------------------------------------
    MODEL_DIR = os.path.join(args.output_dir, args.model_name)
    
    if args.save_dir:
        RESULT_DIR = args.save_dir
    else:
        RESULT_DIR = MODEL_DIR 
        
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR, exist_ok=True)
        print(f">>> Created result directory: {RESULT_DIR}")

    print(f">>> Model Path : {MODEL_DIR}")
    print(f">>> Result Path: {RESULT_DIR}")
    
    REAL_STATS_PATH = args.stats_file
    if not os.path.isabs(REAL_STATS_PATH):
        REAL_STATS_PATH = os.path.join(args.nc_dir, REAL_STATS_PATH)

    # -------------------------------------------------------
    # Logic B: 运行全图推断
    # -------------------------------------------------------
    print(">>> Loading Test Data...")
    _, test_loader = data_provider(args, 'test')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(args, device)
    
    checkpoint_path = os.path.join(MODEL_DIR, 'checkpoint.pth')
    print(f">>> Loading Checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if list(checkpoint.keys())[0].startswith('module.'):
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    if "_metadata" in checkpoint: del checkpoint["_metadata"]
    model.load_state_dict(checkpoint)
    model.eval()
    
    evaluator = Evaluator(stats_file=REAL_STATS_PATH, device=device)
    
    SAVE_PREFIX = os.path.join(RESULT_DIR, 'result')
    
    # 假设 run_models 已经处理了反归一化
    test(model, test_loader, evaluator, SAVE_PREFIX)
    print(">>> Full Evaluation Finished.")

    # -------------------------------------------------------
    # Logic C: 读取数据并自适应处理 (合并修复版)
    # -------------------------------------------------------
    print(f"\n>>> Extracting Data for Detailed Analysis...")
    
    # 1. 读取数据
    preds_full = np.load(f"{SAVE_PREFIX}_pred.npy")
    trues_full = np.load(f"{SAVE_PREFIX}_true.npy")

    print(f"    [Debug] Loaded Original Shape: {preds_full.shape}")

    # =======================================================
    # 分支判断：处理不同的数据形状
    # =======================================================
    
    # 情况 A: 数据已经是 (N, T) 的 2D 形状 -> 直接使用，跳过切片
    # (这是为了解决你之前遇到的 Cannot Reshape 错误)
    if preds_full.ndim == 2:
        print(">>> [Info] Data is already 2D (Samples, Time). Using as is (skipping spatial crop).")
        preds_center = preds_full
        trues_center = trues_full
        
    # 情况 B: 数据是 (N, T, H*W) 或类似 Flatten 过的 3D 形状 -> 尝试还原后切片
    elif preds_full.ndim == 3:
        # 检查能不能还原成 64x64
        expected_elements = preds_full.shape[0] * preds_full.shape[1] * 64 * 64
        if preds_full.size == expected_elements:
            print(">>> [Info] Data is flattened 3D. Reshaping to 4D and cropping center...")
            preds_full = preds_full.reshape(-1, args.pred_len, 64, 64)
            trues_full = trues_full.reshape(-1, args.pred_len, 64, 64)
            # 执行切片
            preds_center, trues_center = extract_center_tensor(
                preds_full, trues_full, window_size=args.eval_window_size
            )
        else:
            print(">>> [Warning] Data is 3D but cannot reshape to 64x64 (Size mismatch). Using as is.")
            preds_center = preds_full
            trues_center = trues_full

    # 情况 C: 数据是标准的 (N, T, H, W) 4D 形状 -> 执行切片
    elif preds_full.ndim == 4:
        print(f">>> [Info] Data is 4D {preds_full.shape}. Cropping center {args.eval_window_size}x{args.eval_window_size}...")
        preds_center, trues_center = extract_center_tensor(
            preds_full, trues_full, window_size=args.eval_window_size
        )
        
    else:
        print(f"❌ Error: Unknown data shape {preds_full.shape}")
        return

    print(f"    [Debug] Final Analysis Shape: {preds_center.shape}")

    # 2. 计算指标
    pred_flat = preds_center.flatten()
    true_flat = trues_center.flatten()
    
    mse = mean_squared_error(true_flat, pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_flat, pred_flat)
    
    print("\n✅ Metrics (Real Values):")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE:  {mae:.4f}")
    
    # -------------------------------------------------------
    # Logic D: 保存结果 (CSV + 可视化)
    # -------------------------------------------------------
    # 1. 保存 CSV
    metrics_df = pd.DataFrame([{
        'Model': args.model_name,
        'Lat': args.target_lat,
        'Lon': args.target_lon,
        'Window': f"{args.eval_window_size}x{args.eval_window_size}",
        'RMSE': rmse,
        'MAE': mae
    }])
    csv_path = os.path.join(RESULT_DIR, f"metrics_lat{args.target_lat:.2f}.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"✅ Metrics saved to {csv_path}")

    # 2. 画图 (鲁棒版本)
    print("\n>>> Generating Visualization...")
    sample_idx = 0
    
    try:
        p_data = preds_center[sample_idx] 
        t_data = trues_center[sample_idx]
        
        # 维度降维处理：如果还有空间维 (H, W)，求平均变回时间序列
        # 如果本来就是 1D (T,)，ndim=1，不会触发这个 if
        if p_data.ndim > 1:
            ts_pred = np.mean(p_data, axis=(-2, -1))
            ts_true = np.mean(t_data, axis=(-2, -1))
        else:
            ts_pred = p_data
            ts_true = t_data

        print(f"    Plotting Shape: {ts_pred.shape}")
        
    except Exception as e:
        print(f"❌ Error extracting series for plot: {e}")
        return 

    plt.figure(figsize=(12, 6))
    plt.plot(ts_true, color='black', label='Ground Truth (Center Point)', linewidth=2)
    plt.plot(ts_pred, color='red', label=f'Prediction ({args.model_name})', linestyle='--', linewidth=2, alpha=0.9)
    
    plt.title(f"Detailed Analysis: Center Point ({args.target_lat}, {args.target_lon})\nWindow: {args.eval_window_size}x{args.eval_window_size}", fontsize=14)
    plt.ylabel("Wind Speed (m/s)", fontsize=12)
    plt.xlabel("Time Steps (Hours)", fontsize=12)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plot_path = os.path.join(RESULT_DIR, f"plot_lat{args.target_lat:.2f}_lon{args.target_lon:.2f}.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"✅ Visualization saved to: {plot_path}")
    
if __name__ == "__main__":
    recover()
