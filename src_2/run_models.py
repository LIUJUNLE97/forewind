import argparse
import torch
import numpy as np
import pandas as pd
import os
import time
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from data_provider.data_factory import data_provider
from model.model_factory import get_model
from evaluation.eval import Evaluator
from evaluation.vis import BenchmarkVisualizer

# --- 1. 参数配置 ---
PROJECT_ROOT = Path(os.environ.get("WIND_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
parser = argparse.ArgumentParser(description='Weather Prediction Training & Evaluation')
parser.add_argument('--no_amp', action='store_false', dest='use_amp',
                    help='禁用混合精度训练')
parser.add_argument('--use_amp', action='store_true', default=True,
                    help='使用混合精度训练 (推荐开启)')
parser.add_argument('--nc_dir', type=str, default=str(PROJECT_ROOT / 'npy_data'))
parser.add_argument('--stats_file', type=str, default='normalization_stats.json')
parser.add_argument('--som_map_path', type=str, default=str(PROJECT_ROOT / 'som_map.npy'))
parser.add_argument('--original_nc_dir', type=str, default=str(PROJECT_ROOT / 'processed'))
parser.add_argument('--output_dir', type=str, default='./analysis_results')

parser.add_argument('--model_name', type=str, default='mlp')
parser.add_argument('--seq_len', type=int, default=24)
parser.add_argument('--pred_len', type=int, default=24)
parser.add_argument('--patch_size', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=3)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--num_workers', type=int, default=8)
#"Shanghai":  [121.47, 31.23],"Chengdu":   [104.06, 30.67], "Hohhot":    [111.75, 40.84],"Qingdao":  [120.38, 36.06]
#"Guangzhou": [23.13, 113.26],"Urumqi":    [43.82, 87.62],
parser.add_argument('--target_lat', type=float, default=43.82)   #1
parser.add_argument('--target_lon', type=float, default=87.62)   #2
parser.add_argument('--fix_location', action='store_true', help='固定采样')
parser.add_argument('--no_fix_location', action='store_false', dest='fix_location')
parser.set_defaults(fix_location=True)
parser.add_argument('--eval_window_size', type=int, default=1)

parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--freq', type=str, default='h')
parser.add_argument('--features', type=str, default='M')

args = parser.parse_args()

args.test_location_config = {
    'type': 'fixed' if args.fix_location else 'random',
    'lat': args.target_lat,
    'lon': args.target_lon
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def reconstruct_input_on_gpu(batch, device):
    """
    在 GPU 上将物理变量和时间特征拼接，复原为 [B, T, C+4, H, W]
    """
    # 1. 搬运数据 (non_blocking=True 配合 pin_memory 提速)
    x_vars = batch['x_vars'].to(device, non_blocking=True) # [B, T, C, H, W]
    t_feat = batch['t_feat'].to(device, non_blocking=True) # [B, T, 4]
    y = batch['y'].to(device, non_blocking=True)           # [B, T_out, 2, H, W]

    # 2. 广播时间特征
    B, T, _, H, W = x_vars.shape
    # view + expand 几乎不耗时，也不占额外显存
    t_feat_expanded = t_feat.view(B, T, 4, 1, 1).expand(-1, -1, -1, H, W)

    # 3. 拼接 (这一步产生最终输入)
    x_final = torch.cat([x_vars, t_feat_expanded], dim=2)

    return x_final, y

def train(model, loader, optimizer, criterion, pred_len, scaler=None, use_amp=False):
    model.train()
    total_loss = 0
    loop = tqdm(loader, desc="Train", leave=False)
    for batch in loop:
        x, y = reconstruct_input_on_gpu(batch, device)

        optimizer.zero_grad()
        if use_amp:
            with autocast():
                pred = model(x, pred_len=pred_len)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            # 普通训练流程
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)

def validate(model, loader, criterion, pred_len, scaler=None, use_amp=False):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            x, y = reconstruct_input_on_gpu(batch, device)
            if use_amp:
                with autocast():
                    pred = model(x, pred_len=pred_len)
                    loss = criterion(pred, y)
            else:
                pred = model(x)
                loss = criterion(pred, y)

            total_loss += loss.item()

    return total_loss / len(loader)

# --- 3. 协调评估核心 ---

def test(model, loader, evaluator, pred_len, save_prefix, args):
    model.eval()
    all_patch_metrics = []
    all_preds_list, all_trues_list = [], []

    print(">>> 启动协调评估流程...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            x, y = reconstruct_input_on_gpu(batch, device)
            pred = model(x, pred_len=pred_len)

            # 1. 调用 Evaluator 计算全图指标 (Patch Level)
            all_patch_metrics.append(evaluator.evaluate_batch(pred, y))

            # 2. 调用 Evaluator 提取单点数据 (已反归一化)
            point_data = evaluator.evaluate_single_point(pred, y, window_size=args.eval_window_size)
            all_preds_list.append(point_data['preds_np'])
            all_trues_list.append(point_data['trues_np'])

    # --- 逻辑汇总 ---

    # 全图指标平均
    patch_df = pd.DataFrame([{f"Patch_{v}_{k}": val for v, d in m.items() for k, val in d.items()} for m in all_patch_metrics])
    final_results = patch_df.mean().to_dict()

    # 拼接中心点数据
    total_p = np.concatenate(all_preds_list, axis=0)
    total_t = np.concatenate(all_trues_list, axis=0)

    # 3. 调用 Evaluator 静态方法计算全量统计指标 (解决相关系数 R 计算的严谨性)
    point_stats = Evaluator.summarize_stats(total_p.flatten(), total_t.flatten())
    final_results.update({f"SinglePoint_{k}": v for k, v in point_stats.items()})
    final_results.update({'Lat': args.target_lat, 'Lon': args.target_lon})

    # 保存数据与误差分布图
    np.save(f"{save_prefix}_point_pred.npy", total_p)
    np.save(f"{save_prefix}_point_true.npy", total_t)
    plot_error_dist(total_p.flatten() - total_t.flatten(), save_prefix)

    return final_results

def plot_error_dist(errors, save_prefix):
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--')
    plt.title("Single Point Speed Error Distribution")
    plt.xlabel("Error (m/s)")
    plt.ylabel("Frequency")
    plt.savefig(f"{save_prefix}_error_hist.png", dpi=300)
    plt.close()

# --- 4. 主程序 ---

def main():

    model_output_path = os.path.join(args.output_dir, args.model_name)
    os.makedirs(model_output_path, exist_ok=True)
    best_model_file = os.path.join(model_output_path, 'checkpoint.pth')
    latest_ckpt_path = os.path.join(model_output_path, 'checkpoint_latest.pth')

    train_set, train_loader = data_provider(args, 'train')
    val_set, val_loader = data_provider(args, 'val')
    test_set, test_loader = data_provider(args, 'test')

    # initialize the model
    model = get_model(args, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
    model_name_lower = args.model_name.lower()

    if model_name_lower == 'fno':
        use_amp_actual = False
        # fno不支持amp运算。
    else:
        use_amp_actual = args.use_amp

    scaler = GradScaler() if use_amp_actual else None

    start_epoch = 0
    best_val_loss = float('inf')

    if os.path.exists(latest_ckpt_path):
        print(f">>> [Auto-Resume] Found checkpoint: {latest_ckpt_path}")
        try:
            checkpoint = torch.load(latest_ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scaler is not None and checkpoint.get('scaler_state_dict'):
                scaler.load_state_dict(checkpoint['scaler_state_dict'])

            start_epoch = checkpoint['epoch'] + 1
            print(f">>> Resuming from Epoch {start_epoch}")

        except Exception as e:
            print(f">>> [Error] Resume failed: {e}")

    else:
        print(f">>> [Start] No checkpoint found. Starting fresh from Epoch 0.")

    # train loop

    patience_counter = 0
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss = train(model, train_loader, optimizer, criterion,
                          scaler=scaler,pred_len=args.pred_len, use_amp=use_amp_actual)

        val_loss = validate(model, val_loader, criterion, pred_len=args.pred_len, use_amp=use_amp_actual)
        dt = time.time() - t0

        print(f"Epoch {epoch+1:03d} | Loss: T {train_loss:.4f} V {val_loss:.4f} | {dt:.1f}s")


        torch.save({
                    'epoch': epoch,                          # 当前跑到第几轮了
                    'model_state_dict': model.state_dict(),  # 模型权重
                    'optimizer_state_dict': optimizer.state_dict(), # 优化器状态 (动量等)
                    'scaler_state_dict': scaler.state_dict() if scaler else None, # 混合精度状态
                    'best_val_loss': best_val_loss,          # 别忘了这个，否则 best 记录会丢失
        }, latest_ckpt_path)

        print(f" Latest state saved. ")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_file)
            print("   >>> Best Model Saved")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("   >>> Early Stopping Triggered")
                break


    print("\n>>> TEST...")
    if os.path.exists(best_model_file):
        state_dict = torch.load(best_model_file)
        if "_metadata" in state_dict:
            del state_dict["_metadata"]

        # 3. 加载处理过的权重
        model.load_state_dict(state_dict)
        print(f"Successfully loaded best model from {best_model_file}")
    else:
        print("Warning: No checkpoint found!")

    evaluator = Evaluator(stats_file=os.path.join(args.nc_dir, args.stats_file), device=device)

    save_path_prefix = os.path.join(model_output_path, 'final_eval')
    avg_metrics = test(model, test_loader, evaluator,args.pred_len, save_path_prefix, args)

    # 存 CSV 并打印
    pd.DataFrame([avg_metrics]).to_csv(f"{save_path_prefix}_metrics.csv", index=False)
    print("\n=== Final Summary Metrics ===\n", pd.Series(avg_metrics))

    # 绘制可视化拟合图
    # 取第一个样本的中心点序列 [T]
    preds_plot = np.load(f"{save_path_prefix}_point_pred.npy")
    trues_plot = np.load(f"{save_path_prefix}_point_true.npy")

    visualizer = BenchmarkVisualizer(output_dir=model_output_path)
    loc_info = f"at ({args.target_lat}, {args.target_lon})" if args.fix_location else "Random Sampling"

    visualizer.plot_time_series(
        preds_plot[0],
        trues_plot[0],
        title_suffix=f"{args.model_name.upper()} {loc_info}"
    )

    if os.path.exists(latest_ckpt_path):
        os.remove(latest_ckpt_path)
        print(f">>> [Cleanup] Training finished. Removed temporary checkpoint: {latest_ckpt_path}")

if __name__ == "__main__":
    main()
