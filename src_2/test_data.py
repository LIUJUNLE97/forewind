import argparse
import torch
from data_provider.data_factory import data_provider

# 1. 模拟 args 参数 (假装是从命令行输入的)
parser = argparse.ArgumentParser(description='Data Loading Test')

# 路径设置 (请修改为你服务器的真实路径!)
parser.add_argument('--nc_dir', type=str, default='/mimer/NOBACKUP/groups/kthmech/junle/guandi_wind_project/data_2/processed')
parser.add_argument('--som_map_path', type=str, default='/mimer/NOBACKUP/groups/kthmech/junle/guandi_wind_project/data_2/som/som_map.npy')
parser.add_argument('--stats_file', type=str, default='normalization_stats.json')

# 模型参数
parser.add_argument('--seq_len', type=int, default=24, help='输入过去24小时')
parser.add_argument('--pred_len', type=int, default=24, help='预测未来24小时')
parser.add_argument('--patch_size', type=int, default=64, help='切片大小 64x64')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_workers', type=int, default=0) # 调试时设为0避免多进程报错

# 特殊配置：上海定点测试
# 这是一个 Python 技巧，给 args 动态添加属性
args = parser.parse_args()
args.fix_location = True
args.test_location_config = {
    'type': 'fixed',
    'lat': 31.23,
    'lon': 121.47
}

def test_pipeline():
    print("===== 开始测试：Unified Data Loading Pipeline =====")
    
    # 1. 测试训练集 (随机切片)
    print("\n>>> Testing TRAIN Loader (Random Crop)...")
    train_set, train_loader = data_provider(args, flag='train')
    
    for i, batch in enumerate(train_loader):
        print(f"Batch {i} shapes:")
        print(f"  Input X: {batch['x'].shape} (Should be [B, T_in, C+4, H, W])")
        print(f"  Target Y: {batch['y'].shape} (Should be [B, T_out, 2, H, W])")
        print(f"  SOM Mask: {batch['som_mask'].shape}")
        
        # 检查是否有 NaN
        if torch.isnan(batch['x']).any():
            print("!!! 警告：输入数据中发现 NaN !!!")
        else:
            print("  数据完整性检查: PASS (无 NaN)")
        
        break # 只测一个 batch

    # 2. 测试测试集 (上海定点)
    print("\n>>> Testing TEST Loader (Shanghai Fixed Crop)...")
    test_set, test_loader = data_provider(args, flag='test')
    
    batch = next(iter(test_loader))
    coords = batch['coords']
    print(f"切片坐标 (Top-Left): y={coords[0,0]}, x={coords[0,1]}")
    print("请核对上面的坐标是否每次运行都一致 (应该是固定的)！")
    
    # 3. 测试 Scaler 的还原功能
    print("\n>>> Testing Scaler Inverse Transform...")
    scaler = test_set.scalers['u10']
    # 拿出一个归一化的值
    sample_val = batch['x'][0, 0, 0, 0, 0] # [B, T, C, H, W] -> u10
    # 还原
    restored_val = scaler.inverse_transform(sample_val)
    print(f"Normalized Value: {sample_val:.4f}")
    print(f"Restored Value (m/s): {restored_val:.4f}")

    print("\n===== 测试完成！如果没有报错，说明数据层构建成功 =====")

if __name__ == "__main__":
    test_pipeline()