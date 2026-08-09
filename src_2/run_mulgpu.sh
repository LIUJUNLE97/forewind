import argparse
import torch
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm

# 分布式训练必备
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from data_provider.data_factory import data_provider
from model.model_factory import get_model
from evaluation.eval import Evaluator
from evaluation.vis import BenchmarkVisualizer

# --- 1. 参数配置 ---
parser = argparse.ArgumentParser(description='Formal Training Script - DDP Version')
project_root = os.environ.get('WIND_PROJECT_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
default_data_dir = os.environ.get('WIND_NPY_DATA_DIR', os.path.join(project_root, 'npy_data'))

# 路径参数
default_out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'analysis_results')
parser.add_argument('--nc_dir', type=str, default=default_data_dir)
parser.add_argument('--som_map_path', type=str, default=os.path.join(project_root, 'data', 'metadata', 'som_map.npy'))
parser.add_argument('--stats_file', type=str, default='normalization_stats.json')
parser.add_argument('--output_dir', type=str, default=default_out_dir)

# 模型与训练参数
parser.add_argument('--model_name', type=str, default='mlp')
parser.add_argument('--seq_len', type=int, default=24)
parser.add_argument('--pred_len', type=int, default=24)
parser.add_argument('--patch_size', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=3) # 单卡 Batch Size
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--target_lat', type=float, default=31.23)
parser.add_argument('--target_lon', type=float, default=121.47)
parser.add_argument('--use_ddp', type=bool, default=False, help='是否使用分布式训练')

args = parser.parse_args()

def train(model, loader, optimizer, criterion, device, rank):
    model.train()
    total_loss = 0
    # 仅在 Rank 0 显示进度条
    loop = tqdm(loader, desc="Train", leave=False, disable=(rank != 0))
    for batch in loop:
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if rank == 0:
            loop.set_postfix(loss=loss.item())
    
    # 汇总各卡的 Loss (可选，这里简便起见返回单卡平均值)
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item()
    return total_loss / len(loader)

def test(model, loader, evaluator, save_npy_path, device, rank):
    # 测试通常在 Rank 0 上单卡运行，保证数据顺序和保存逻辑一致
    model.eval()
    metrics_log = []
    all_preds_u, all_trues_u = [], []
    
    if rank == 0:
        print(">>> Testing & Extracting Results (Rank 0)...")
        with torch.no_grad():
            for batch in tqdm(loader, desc="Test"):
                x = batch['x'].to(device)
                y = batch['y'].to(device)
                # DDP 模型在测试时需要调用 model.module
                pred = model.module(x) if hasattr(model, 'module') else model(x)
                
                batch_metrics = evaluator.evaluate_batch(pred, y)
                flat_metrics = {f"{var}_{k}": v for var in ['U', 'V'] for k, v in batch_metrics[var].items()}
                metrics_log.append(flat_metrics)
                
                pred_real_u = evaluator.denormalize(pred[:, :, 0:1], 'u10').cpu().numpy()
                true_real_u = evaluator.denormalize(y[:, :, 0:1], 'u10').cpu().numpy()
                
                cy, cx = args.patch_size // 2, args.patch_size // 2
                all_preds_u.append(pred_real_u[:, :, 0, cy, cx])
                all_trues_u.append(true_real_u[:, :, 0, cy, cx])

        total_pred = np.concatenate(all_preds_u, axis=0)
        total_true = np.concatenate(all_trues_u, axis=0)
        np.save(f"{save_npy_path}_pred.npy", total_pred)
        np.save(f"{save_npy_path}_true.npy", total_true)
        return pd.DataFrame(metrics_log).mean().to_dict()
    return None

def main():
    # --- DDP 环境初始化 ---
    if args.use_ddp:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        rank = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if rank == 0:
        print(f"🚀 Training: {args.model_name.upper()} (DDP: {args.use_ddp})")
        args.output_dir = os.path.join(args.output_dir, args.model_name)
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 路径定义
    best_model_path = os.path.join(args.output_dir if rank==0 else "/tmp", 'checkpoint.pth')
    save_prefix = os.path.join(args.output_dir if rank==0 else "/tmp", 'result')

    # --- 数据加载 (使用 DistributedSampler) ---
    train_set, train_loader = data_provider(args, 'train')
    val_set, val_loader = data_provider(args, 'val')
    test_set, test_loader = data_provider(args, 'test')

    # --- 模型初始化与 DDP 包装 ---
    model = get_model(args, device).to(device)
    if args.use_ddp:
        model = DDP(model, device_ids=[rank], output_device=rank)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
    
    # --- 训练循环 ---
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # 必须设置 sampler 的 epoch 以保证 shuffle 正常
        if args.use_ddp and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        t0 = time.time()
        train_loss = train(model, train_loader, optimizer, criterion, device, rank)
        val_loss = validate(model, val_loader, criterion, device)
        
        # 等待所有卡完成验证
        if args.use_ddp:
            val_loss_tensor = torch.tensor(val_loss).to(device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            val_loss = val_loss_tensor.item() / world_size

        if rank == 0:
            dt = time.time() - t0
            print(f"Epoch {epoch+1}/{args.epochs} | Loss: T {train_loss:.4f} V {val_loss:.4f} | {dt:.1f}s")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # DDP 保存模型需要保存 model.module.state_dict()
                state_dict = model.module.state_dict() if args.use_ddp else model.state_dict()
                torch.save(state_dict, best_model_path)
                print(f"   >>> Best Model Saved")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print("   >>> Early Stopping Triggered!")
                    break
        
        # 广播停止信号（可选，这里简单处理）
        if args.use_ddp:
            dist.barrier()

    # --- 测试与可视化 (仅在 Rank 0 运行) ---
    if rank == 0:
        print("\n>>> Testing Best Model...")
        # 重新加载模型（去掉 DDP 包装或重新加载到单卡）
        model_to_test = get_model(args, device).to(device)
        model_to_test.load_state_dict(torch.load(best_model_path))
        
        evaluator = Evaluator(stats_file=os.path.join(args.nc_dir, args.stats_file), device=device)
        avg_metrics = test(model_to_test, test_loader, evaluator, save_prefix, device, rank)
        
        print("\n=== Summary ===")
        print(pd.Series(avg_metrics))
        pd.DataFrame([avg_metrics]).to_csv(f"{save_prefix}_metrics.csv", index=False)
        
        print(">>> Generating visualization...")
        preds = np.load(f"{save_prefix}_pred.npy")
        trues = np.load(f"{save_prefix}_true.npy")
        visualizer = BenchmarkVisualizer(output_dir=args.output_dir)
        visualizer.plot_time_series(preds[0], trues[0], title_suffix=f"{args.model_name.upper()} at ({args.target_lat}, {args.target_lon})")

    if args.use_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
