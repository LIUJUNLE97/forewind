import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
import pandas as pd
import json
import os
import glob

class Dataset(Dataset):
    def __init__(self, nc_dir, som_map_path, mode='train', patch_size=64, stats_file='normalization_stats.json'):
        self.nc_dir = nc_dir
        self.patch_size = patch_size
        self.mode = mode
        
       
        self.som_map = np.load(som_map_path).astype(np.int64)
        self.full_h, self.full_w = self.som_map.shape

        # 2. 统计量加载
        stats_path = os.path.join(nc_dir, stats_file)
        with open(stats_path, 'r') as f:
            self.stats = json.load(f)

        # 3. 高效延迟加载
        all_files = sorted(glob.glob(os.path.join(nc_dir, '*.nc')))
        self.file_list = self._split_by_year(all_files, mode)
        
        # 使用 open_mfdataset，注意 chunks 设置
        self.ds = xr.open_mfdataset(self.file_list, chunks={'time': 128}, combine='by_coords')
        self.T = len(self.ds.time)

        # 4. 验证/测试集确定性采样
        if mode != 'train':
            np.random.seed(42)
            self.eval_coords = [self._gen_random_coords() for _ in range(self.T - 1)]

        print(f"[{mode.upper()}] 初始化完成。时间步总数: {self.T}")

    def _split_by_year(self, files, mode):
        import re
        selected = []
        for f in files:
            match = re.search(r'(\d{4})', os.path.basename(f))
            if match:
                year = int(match.group(1))
                if mode == 'train' and (1998 <= year <= 2018): selected.append(f)
                elif mode == 'val' and (2019 <= year <= 2020): selected.append(f)
                elif mode == 'test' and (2021 <= year <= 2024): selected.append(f)
        return selected

    def _gen_random_coords(self):
        y = np.random.randint(0, self.full_h - self.patch_size)
        x = np.random.randint(0, self.full_w - self.patch_size)
        return y, x

    def _normalize(self, data, var_name):
        mean = self.stats[var_name]['mean']
        std = self.stats[var_name]['std']
        return (data - mean) / (std + 1e-6)

    def __len__(self):
        return self.T - 1

    def __getitem__(self, idx):
        if self.mode == 'train':
            y_s, x_s = self._gen_random_coords()
        else:
            y_s, x_s = self.eval_coords[idx]
        
        y_e, x_e = y_s + self.patch_size, x_s + self.patch_size

        # 改进：变量名与预处理脚本严格对齐
        vars_in = ['u10', 'v10', 't2m', 'sp', 'geopotential']
        
        # 一次性切片读取
        sub_ds = self.ds.isel(
            time=slice(idx, idx + 2), 
            latitude=slice(y_s, y_e), 
            longitude=slice(x_s, x_e)
        ).compute()

        # 物理量读取 
        data_x = sub_ds[vars_in].isel(time=0).to_array().values.astype(np.float32)
        data_y = sub_ds[['u10', 'v10']].isel(time=1).to_array().values.astype(np.float32)

        # 改进：同步 SOM Mask
        som_mask = self.som_map[y_s:y_e, x_s:x_e]

        # 归一化 (直接使用 geopotential 作为 key)
        for i, var in enumerate(vars_in):
            data_x[i] = self._normalize(data_x[i], var)

        # 时间特征嵌入
        curr_time = pd.Timestamp(sub_ds.time.values[0])
        day_rad = (curr_time.dayofyear / 365.0) * 2 * np.pi
        hour_rad = (curr_time.hour / 24.0) * 2 * np.pi
        time_map = np.tile(
            np.array([np.cos(day_rad), np.sin(day_rad), np.cos(hour_rad), np.sin(hour_rad)], dtype=np.float32)[:, None, None],
            (1, self.patch_size, self.patch_size)
        )

        return {
            'x': torch.from_numpy(np.concatenate([data_x, time_map], axis=0)),
            'som_mask': torch.from_numpy(som_mask),
            'y': torch.from_numpy(data_y)
        }