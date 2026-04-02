import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import pandas as pd
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy_dir', type=str, required=True, help='存放 .npy 文件的目录')
    args = parser.parse_args()

    data_path = os.path.join(args.npy_dir, 'weather_data_all.npy')
    time_path = os.path.join(args.npy_dir, 'timestamps.npy')

    print(f"Checking data in: {args.npy_dir}")

    # =========================================================
    # 1. 基础信息检查
    # =========================================================
    if not os.path.exists(data_path) or not os.path.exists(time_path):
        print("❌ Error: Files not found!")
        return

    print(">>> Loading metadata...")
    # 使用 mmap 模式，只读不进内存，速度快
    data = np.load(data_path, mmap_mode='r')
    timestamps = np.load(time_path, allow_pickle=True)

    print(f"✅ Data Shape: {data.shape} (Time, Var, H, W)")
    print(f"✅ Time Shape: {timestamps.shape}")

    if data.shape[0] != timestamps.shape[0]:
        print(f"❌ Error: Length mismatch! Data={data.shape[0]}, Time={timestamps.shape[0]}")
    else:
        print("✅ Length check passed.")

    # =========================================================
    # 2. 数值健康度检查 (抽样)
    # =========================================================
    print("\n>>> Checking value distribution (Sampling 100 random frames)...")
    
    # 随机抽 100 个时间点
    indices = np.random.choice(data.shape[0], 100, replace=False)
    sample = data[indices] # 这一步会真正读取数据到内存

    print(f"   Sample Min: {np.min(sample):.4f}")
    print(f"   Sample Max: {np.max(sample):.4f}")
    print(f"   Sample Mean: {np.mean(sample):.4f}")
    
    if np.isnan(sample).any():
        print("❌ CRITICAL: NaN values found in sample!")
    else:
        print("✅ No NaN values found in sample.")

    if np.isinf(sample).any():
        print("❌ CRITICAL: Infinite values found in sample!")
    else:
        print("✅ No Infinite values found in sample.")

    # =========================================================
    # 3. 时间连续性检查
    # =========================================================
    print("\n>>> Checking time continuity...")
    ts = pd.to_datetime(timestamps)
    
    # 计算时间差
    deltas = np.diff(ts)
    
    # 假设数据是逐小时的 (1 hour)
    expected_delta = pd.Timedelta(hours=1)
    
    # 允许少量的缺失，但要打印出来
    gaps = np.where(deltas > expected_delta)[0]
    
    if len(gaps) == 0:
        print(f"✅ Time series is perfectly continuous from {ts[0]} to {ts[-1]}.")
    else:
        print(f"⚠️ Warning: Found {len(gaps)} gaps in time series!")
        print(f"   First gap at index {gaps[0]}: {ts[gaps[0]]} -> {ts[gaps[0]+1]}")
        print(f"   Max gap: {np.max(deltas)}")

    # =========================================================
    # 4. 可视化抽查 (保存图片)
    # =========================================================
    print("\n>>> Generating inspection plot...")
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    vars_name = ['u10', 'v10', 't2m', 'sp', 'z']
    
    # 取中间那一帧
    idx = data.shape[0] // 2
    frame = data[idx] # [5, H, W]
    time_str = str(ts[idx])

    for i in range(5):
        im = axes[i].imshow(frame[i], cmap='viridis', origin='lower') # origin='lower' 对应 lat/lon 习惯
        axes[i].set_title(f"{vars_name[i]}")
        plt.colorbar(im, ax=axes[i])

    plt.suptitle(f"Data Inspection: Index={idx}, Time={time_str}")
    
    save_path = os.path.join(args.npy_dir, 'inspect_sample.png')
    plt.savefig(save_path)
    print(f"✅ Plot saved to: {save_path}")
    print("Please download and view this image to ensure data looks like 'weather maps' and not noise.")

if __name__ == "__main__":
    main()