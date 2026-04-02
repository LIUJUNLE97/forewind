import torch
import json
import numpy as np
import os

class Evaluator:
    def __init__(self, stats_file, lat_range=(54, 18), patch_size=64, device='cuda'):
        """
        1. Metric Calculation: RMSE, RMSE_OOD (Extreme), ACC (Correlation)
        2. Physical Conversion: Z-Score -> m/s, UV -> Speed
        3. Visualization Support: Single point extraction
        """
        self.device = device
        self.patch_size = patch_size
        
        # 1. 加载统计量文件
        if isinstance(stats_file, str) and os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                self.stats = json.load(f)
        else:
            raise FileNotFoundError(f"Stats file not found: {stats_file}")

        # 2. 预加载均值和方差 (健壮性处理：兼容 'u10'/'v10' 和 '0'/'1' 键名)
        try:
            self.u_mean = self.stats['u10']['mean']
            self.u_std = self.stats['u10']['std']
            self.v_mean = self.stats['v10']['mean']
            self.v_std = self.stats['v10']['std']
        except KeyError:
            print("⚠️ Warning: JSON keys 'u10'/'v10' not found. Trying fallback keys '0'/'1'...")
            self.u_mean = self.stats.get('0', {}).get('mean', 0)
            self.u_std = self.stats.get('0', {}).get('std', 1)
            self.v_mean = self.stats.get('1', {}).get('mean', 0)
            self.v_std = self.stats.get('1', {}).get('std', 1)

        # 3. 计算纬度权重 (用于 Patch 级别的加权 RMSE)
        # 模拟地球曲率：高纬度网格面积小，权重低；低纬度权重高
        lats = np.linspace(lat_range[0], lat_range[1], patch_size)
        weights = np.cos(np.deg2rad(lats))
        self.weights = torch.from_numpy(weights / weights.mean()).float().to(device).view(1, 1, -1, 1)

    def denormalize(self, data, var_name='u10'):
        """物理单位还原: Z-Score -> m/s"""
        std = self.u_std if var_name == 'u10' else self.v_std
        mean = self.u_mean if var_name == 'u10' else self.v_mean
        return data * std + mean

    def calculate_speed(self, u, v):
        return torch.sqrt(u**2 + v**2)

    @torch.no_grad()
    def evaluate_batch(self, pred, target):
        """
        计算单批次全图 (Patch-level) 的核心指标
        返回: 
           - RMSE: 均方根误差 (基础性能)
           - RMSE_OOD: 极端天气误差 (鲁棒性, Threshold > 12m/s)
           - ACC: 异常相关系数 (模式匹配度)
        """
        # 1. 反归一化 (还原到真实物理空间 m/s)
        p_u = self.denormalize(pred[:, :, 0:1], 'u10')
        t_u = self.denormalize(target[:, :, 0:1], 'u10')
        p_v = self.denormalize(pred[:, :, 1:2], 'v10')
        t_v = self.denormalize(target[:, :, 1:2], 'v10')

        # 2. 计算合速度
        p_speed = self.calculate_speed(p_u, p_v)
        t_speed = self.calculate_speed(t_u, t_v)

        batch_results = {}
        
        # 准备气候态均值 (用于计算 ACC 的距平 Anomaly)
        # 这里使用训练集的全局均值作为简化的气候态
        means = {
            'U': self.u_mean,
            'V': self.v_mean,
            'Speed': np.sqrt(self.u_mean**2 + self.v_mean**2) 
        }

        # 定义需要评估的变量
        metrics_targets = {
            'U': (p_u, t_u), 
            'V': (p_v, t_v), 
            'Speed': (p_speed, t_speed)
        }

        for name, (p, t) in metrics_targets.items():
            # --- Metric 1: RMSE (Weighted) ---
            rmse = torch.sqrt(torch.mean(self.weights * (p - t)**2))
            
            # --- Metric 2: RMSE_OOD (Extreme Weather > 12m/s) ---
            # 仅统计真实风速大于 12m/s 的格点
            # 对于 U/V 分量，依然使用合速度 > 12 作为筛选条件
            if name == 'Speed':
                mask_ood = (t > 12.0).float()
            else:
                mask_ood = (t_speed > 12.0).float()

            if mask_ood.sum() > 0:
                weighted_sq_diff = self.weights * (p - t)**2 * mask_ood
                rmse_ood = torch.sqrt(torch.sum(weighted_sq_diff) / (torch.sum(mask_ood * self.weights) + 1e-8))
            else:
                rmse_ood = torch.tensor(0.0, device=self.device)

            # --- Metric 3: ACC (Anomaly Correlation Coefficient) ---
            # ACC = cov(p', t') / (std(p') * std(t'))
            clim_mean = means[name]
            p_ano = p - clim_mean
            t_ano = t - clim_mean
            
            numerator = torch.sum(self.weights * p_ano * t_ano)
            denominator = torch.sqrt(torch.sum(self.weights * p_ano**2) * torch.sum(self.weights * t_ano**2))
            acc = numerator / (denominator + 1e-8)

            # 存入结果
            batch_results[name] = {
                'RMSE': rmse.item(),
                'RMSE_OOD': rmse_ood.item(),
                'ACC': acc.item()
            }
            
        return batch_results

    @torch.no_grad()
    def evaluate_single_point(self, pred, target, window_size=1):
        """
        提取中心区域数据，用于时序绘图
        返回 NumPy 格式的 [Batch, Time] 数组
        """
        # 1. 反归一化
        p_u = self.denormalize(pred[:, :, 0:1], 'u10')
        p_v = self.denormalize(pred[:, :, 1:2], 'v10')
        t_u = self.denormalize(target[:, :, 0:1], 'u10')
        t_v = self.denormalize(target[:, :, 1:2], 'v10')

        # 2. 计算合速度
        p_speed = self.calculate_speed(p_u, p_v)
        t_speed = self.calculate_speed(t_u, t_v)

        # 3. 提取中心窗口
        H, W = self.patch_size, self.patch_size
        cy, cx = H // 2, W // 2
        offset = window_size // 2
        
        y_s = max(0, cy - offset)
        y_e = min(H, cy + offset + 1)
        x_s = max(0, cx - offset)
        x_e = min(W, cx + offset + 1)
        
        # 提取并求均值 (降维: B, T, 1, h, w -> B, T)
        p_point = p_speed[:, :, 0, y_s:y_e, x_s:x_e].mean(dim=(-2, -1))
        t_point = t_speed[:, :, 0, y_s:y_e, x_s:x_e].mean(dim=(-2, -1))

        return {
            'preds_np': p_point.cpu().numpy(), 
            'trues_np': t_point.cpu().numpy()
        }

    @staticmethod
    def summarize_stats(p_flat, t_flat):
        """
        静态方法：计算全量长序列的统计指标
        用于在测试结束后，对拼接好的所有时间步数据计算 Correlation 和 Bias
        """
        # 移除无效值
        mask = np.isfinite(p_flat) & np.isfinite(t_flat)
        p, t = p_flat[mask], t_flat[mask]
        
        if len(p) < 2: 
            return {'RMSE': 0.0, 'MAE': 0.0, 'MBE': 0.0, 'R': 0.0}
        
        errors = p - t
        rmse = np.sqrt(np.mean(errors**2))
        mae = np.mean(np.abs(errors))
        mbe = np.mean(errors) # Mean Bias Error (系统性偏差)
        
        # 计算 Pearson 相关系数 R
        if np.std(p) > 1e-6 and np.std(t) > 1e-6:
            r_val = np.corrcoef(p, t)[0, 1]
        else:
            r_val = 0.0
            
        return {
            'RMSE': float(rmse), 
            'MAE': float(mae), 
            'MBE': float(mbe), 
            'R': float(r_val)
        }