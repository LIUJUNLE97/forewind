import torch
import numpy as np

class StandardScaler:
    """
    可逆的标准化器，支持 PyTorch Tensor 和 NumPy
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def transform(self, data):
        # 支持 data 是 tensor 或 numpy
        if torch.is_tensor(data):
            device = data.device
            # 确保 mean/std 也在同一个 device
            mean = torch.tensor(self.mean, device=device) if not torch.is_tensor(self.mean) else self.mean.to(device)
            std = torch.tensor(self.std, device=device) if not torch.is_tensor(self.std) else self.std.to(device)
            return (data - mean) / (std + 1e-6)
        else:
            return (data - self.mean) / (self.std + 1e-6)
    
    def inverse_transform(self, data):
        if torch.is_tensor(data):
            device = data.device
            mean = torch.tensor(self.mean, device=device) if not torch.is_tensor(self.mean) else self.mean.to(device)
            std = torch.tensor(self.std, device=device) if not torch.is_tensor(self.std) else self.std.to(device)
            return (data * std) + mean
        else:
            return (data * self.std) + self.mean