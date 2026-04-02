import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import glob
import xarray as xr
import sys

class UnifiedWeatherDataset(Dataset):
    def __init__(self, 
                 nc_dir,            
                 som_map_path, 
                 original_nc_dir,  
                 mode='train', 
                 patch_size=64, 
                 location_config=None,  
                 stats_file=None, 
                 time_steps_in=24,      
                 time_steps_out=24):        
        
        self.mode = mode
        self.patch_size = patch_size
        self.input_len = time_steps_in
        self.pred_len = time_steps_out
        self.total_seq_len = time_steps_in + time_steps_out
        self.location_config = location_config

        # 1. Load SOM Map
        self.som_map = np.load(som_map_path).astype(np.int64)
        self.full_h, self.full_w = self.som_map.shape

        data_path = os.path.join(nc_dir, 'weather_data_all.npy')
        time_path = os.path.join(nc_dir, 'timestamps.npy')
        
        # 使用 mmap_mode='r'，但注意后续一定要 copy 出来
        print(f"[{mode.upper()}] Loading data map from {data_path}...")
        self.data = np.load(data_path, mmap_mode='r') 
        
        # 2. 预计算时间特征 (CPU 只计算一次)
        # 结果维度: [Total_Time, 4]
        raw_timestamps = np.load(time_path, allow_pickle=True)
        ts_pd = pd.to_datetime(raw_timestamps)
        
        day_rad = (ts_pd.dayofyear / 365.0) * 2 * np.pi
        hour_rad = (ts_pd.hour / 24.0) * 2 * np.pi
        
        # 提前转为 float32，节省后续转换时间
        self.time_encoded = np.stack([
            np.cos(day_rad), np.sin(day_rad), 
            np.cos(hour_rad), np.sin(hour_rad)
        ], axis=1).astype(np.float32)

        # 3. 划分索引
        years = ts_pd.year.values
        del ts_pd 
        
        if mode == 'train':
            self.indices = np.where((years >= 1998) & (years < 2019))[0]
        elif mode == 'val':
            self.indices = np.where((years >= 2019) & (years < 2021))[0]
        elif mode == 'test':
            self.indices = np.where((years >= 2021) & (years < 2025))[0]
        else:
            self.indices = np.where(years >= 2024)[0]
            
        total_time = self.data.shape[0]
        valid_mask = self.indices < (total_time - self.total_seq_len)
        self.indices = self.indices[valid_mask]

        # 4. 定点坐标映射
        if self.location_config and self.location_config.get('type') == 'fixed':
            # ... (保持原逻辑不变) ...
            sample_nc = sorted(glob.glob(os.path.join(original_nc_dir, '*.nc')))[0]
            with xr.open_dataset(sample_nc) as ds_meta:
                lats = ds_meta.latitude.values
                lons = ds_meta.longitude.values
            lat_idx = np.argmin(np.abs(lats - self.location_config['lat']))
            lon_idx = np.argmin(np.abs(lons - self.location_config['lon']))
            y = np.clip(lat_idx - self.patch_size // 2, 0, self.full_h - self.patch_size)
            x = np.clip(lon_idx - self.patch_size // 2, 0, self.full_w - self.patch_size)
            self.fixed_y, self.fixed_x = int(y), int(x)
        else:
            self.fixed_y, self.fixed_x = None, None
            if mode != 'train':
                np.random.seed(2024) 
                self.eval_coords = [self._gen_random_coords() for _ in range(len(self))]

    def _gen_random_coords(self):
        y = np.random.randint(0, self.full_h - self.patch_size)
        x = np.random.randint(0, self.full_w - self.patch_size)
        return y, x

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        
        if self.fixed_y is not None:
            y_s, x_s = self.fixed_y, self.fixed_x
        elif self.mode == 'train':
            y_s, x_s = self._gen_random_coords()
        else:
            y_s, x_s = self.eval_coords[idx]

        y_e, x_e = y_s + self.patch_size, x_s + self.patch_size
        
        # --- 核心修改区 ---
        
        # 1. 提取 mmap 数据并立即拷贝到内存
        # 这里的 np.array(..., copy=True) 是必须的，否则返回的是 mmap view，
        # 传递给 DataLoader worker 时可能会引起文件锁或性能下降。
        vars_slice_mmap = self.data[real_idx : real_idx + self.total_seq_len, :, y_s:y_e, x_s:x_e]
        vars_slice = np.array(vars_slice_mmap, dtype=np.float32) # 显式强转 float32

        # 2. 仅提取 1D 时间特征 [T_in, 4]
        # 绝对不要在这里做 broadcast_to，这是 CPU 瓶颈的根源
        t_feat = self.time_encoded[real_idx : real_idx + self.input_len]
        
        x_vars = vars_slice[:self.input_len]        # [T_in, C, H, W]
        y_final = vars_slice[self.input_len:, 0:2]  # [T_out, 2, H, W]
        som_mask = self.som_map[y_s:y_e, x_s:x_e]   # [H, W]

        # 返回纯 Numpy 字典，不转 Tensor，把转换压力留给 collate_fn 或 GPU
        return {
            'x_vars': x_vars,
            't_feat': t_feat,
            'y': y_final,
            'som_mask': som_mask,
            # coords 这种小数据直接用 array
            'coords': np.array([y_s, x_s, y_e, x_e], dtype=np.int32)
        }


def fast_collate_fn(batch):
    """
    负责将 batch 中的 numpy array 堆叠成 Tensor。
    不进行广播，只进行 stack。
    """
    # 预分配内存通常比 list comprehension + stack 更快，但在变长序列下不可用。
    # 这里我们使用 stack，因为它兼容性好。
    
    # x_vars: List of [T, C, H, W] -> Tensor [B, T, C, H, W]
    x_vars = torch.from_numpy(np.stack([b['x_vars'] for b in batch]))
    
    # t_feat: List of [T, 4] -> Tensor [B, T, 4]
    # 数据量极小，这里 stack 毫无压力
    t_feat = torch.from_numpy(np.stack([b['t_feat'] for b in batch]))
    
    y = torch.from_numpy(np.stack([b['y'] for b in batch]))
    som_mask = torch.from_numpy(np.stack([b['som_mask'] for b in batch])).long()
    coords = torch.from_numpy(np.stack([b['coords'] for b in batch]))

    return {
        'x_vars': x_vars,
        't_feat': t_feat,
        'y': y,
        'som_mask': som_mask,
        'coords': coords
    }