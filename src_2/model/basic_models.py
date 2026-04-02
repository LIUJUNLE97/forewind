import torch
import torch.nn as nn
from neuralop.models import FNO
import torch.nn.functional as F
import math


class BaselineCNN(nn.Module):
    def __init__(self, in_channels, seq_len, pred_len, hidden_dim=256, output_ch=2):
        super().__init__()
        self.pred_len = pred_len      
        # 输入通道总数 = 变量数 * 时间步 (例如 2 * 24 = 48)
        total_in_ch = in_channels * seq_len
        # 输出通道总数 = 预测变量 * 预测步 (例如 2 * 24 = 48)
        total_out_ch = output_ch * pred_len

        self.conv = nn.Sequential(
            # layer0 
            nn.Conv2d(total_in_ch, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),

            # layer1
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            
            # layer2
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            
            # layer3
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            
            # layer4 adjust channels
            nn.Conv2d(hidden_dim // 2, total_out_ch, kernel_size=1) 
            )

    def forward(self, x, pred_len=None):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        
        # 1. 扁平化时间维度: [B, T*C, H, W]
        x_flat = x.reshape(B, T * C, H, W)
        
        # 2. 卷积计算
        out = self.conv(x_flat) # -> [B, pred_len*2, H, W]
        
        # 3. 还原形状: [B, pred_len, 2, H, W]
        out = out.reshape(B, self.pred_len, 2, H, W)
        return out
  

class BaselineFNO(nn.Module):
    def __init__(self, input_dim, seq_len, pred_len, modes=12, width=32):
        super().__init__()
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        in_channels = seq_len * input_dim
        out_channels = pred_len * 2
        
        self.fno = FNO(
            n_modes=(modes, modes),   # 2D 模式下传两个 mode
            hidden_channels=width,
            in_channels=in_channels,
            out_channels=out_channels
            # 可能不再需要显式 projection_channels，默认会自动处理
        )

    def forward(self, x, pred_len=None):
        # x: [B, T_in, C, H, W]
        B, T, C, H, W = x.shape
        
        # FNO 需要输入: [B, Channels, H, W]
        # 我们的维度变换逻辑不变
        x = x.permute(0, 3, 4, 1, 2).reshape(B, H, W, -1) 
        x = x.permute(0, 3, 1, 2) # [B, T*C, H, W]
        
        out = self.fno(x) # [B, T_out*2, H, W]
        
        # 还原形状
        out = out.reshape(B, self.pred_len, 2, H, W)
        return out


class BSpline(nn.Module):
    """
    B-Spline 基函数实现
    """
    def __init__(self, grid_size=5, spline_order=3, grid_range=[-1, 1]):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # 创建均匀网格
        # 需要额外的边界节点来支持 B-spline
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.linspace(
            grid_range[0] - spline_order * h,
            grid_range[1] + spline_order * h,
            grid_size + 2 * spline_order + 1
        )
        self.register_buffer('grid', grid)
    
    def forward(self, x):
        """
        计算 B-spline 基函数值
        x: [*, in_features]
        返回: [*, in_features, grid_size + spline_order]
        """
        # 确保 x 在合理范围内
        x = x.unsqueeze(-1)  # [*, in_features, 1]
        
        # 计算 B-spline 基函数
        # 使用 Cox-de Boor 递推公式
        bases = self._compute_bspline_basis(x)
        
        return bases
    
    def _compute_bspline_basis(self, x):
        grid = self.grid
        k = self.spline_order
        
        # 初始化：0阶基函数（指示函数）
        # bases[i] = 1 if grid[i] <= x < grid[i+1] else 0
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()
        
        # 递推计算高阶基函数
        for order in range(1, k + 1):
            # 左侧递推项
            left_num = x - grid[:-order-1]
            left_den = grid[order:-1] - grid[:-order-1]
            left_den = torch.where(left_den == 0, torch.ones_like(left_den), left_den)
            left_term = (left_num / left_den) * bases[..., :-1]
            
            # 右侧递推项
            right_num = grid[order+1:] - x
            right_den = grid[order+1:] - grid[1:-order]
            right_den = torch.where(right_den == 0, torch.ones_like(right_den), right_den)
            right_term = (right_num / right_den) * bases[..., 1:]
            
            bases = left_term + right_term
        
        return bases


class KANLinear(nn.Module):
    """
    Φ(x) = w_b * b(x) + w_s * spline(x)
    """
    def __init__(
        self, 
        in_features, 
        out_features, 
        grid_size=5, 
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1]
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Base activation function (SiLU 作为默认)
        self.base_activation = base_activation()
        
        # Base weights: 基础线性变换
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Spline weights: 样条系数
        # 每个输入特征有 (grid_size + spline_order) 个基函数
        num_bases = grid_size + spline_order
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, num_bases))
        
        # B-spline 计算模块
        self.spline = BSpline(grid_size, spline_order, grid_range)
        
        # 缩放因子
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        
        # 初始化权重
        self.reset_parameters(scale_noise)
    
    def reset_parameters(self, scale_noise=0.1):
        # Base weight 使用 Kaiming 初始化
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        self.base_weight.data *= self.scale_base
        
        # Spline weight 使用小的随机值初始化
        nn.init.normal_(self.spline_weight, mean=0.0, std=scale_noise * self.scale_spline)
    
    def forward(self, x, pred_len=None):
        """
        x: [batch_size, ..., in_features]
        返回: [batch_size, ..., out_features]
        """
        # 保存原始形状
        original_shape = x.shape
        batch_dims = original_shape[:-1]
        
        # Flatten to 2D for computation
        x_flat = x.reshape(-1, self.in_features)
        
        # 1. Base transformation: Linear(activation(x))
        base_output = F.linear(self.base_activation(x_flat), self.base_weight)
        
        # 2. Spline transformation
        # 计算 B-spline 基函数值
        # x_flat: [batch, in_features]
        bases = self.spline(x_flat)  # [batch, in_features, num_bases]
        
        # 应用样条权重
        # spline_weight: [out_features, in_features, num_bases]
        # bases: [batch, in_features, num_bases]
        # 需要计算: sum over (in_features, num_bases)
        
        # 方法1: 使用 einsum（清晰但可能较慢）
        spline_output = torch.einsum('oik,bik->bo', self.spline_weight, bases)
        
        # 或者方法2: 使用矩阵乘法（更快）
        # spline_output = (bases.unsqueeze(1) * self.spline_weight.unsqueeze(0)).sum(dim=[2, 3])
        
        # 3. 组合输出
        output = base_output + spline_output
        
        # 恢复原始形状
        output = output.reshape(*batch_dims, self.out_features)
        
        return output


class KANLayer(nn.Module):
    """
    KAN Layer with multiple KANLinear units
    可选择性地添加 LayerNorm 和 Dropout
    """
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        use_layernorm=False,
        dropout=0.0,
        **kwargs
    ):
        super().__init__()
        self.kan_linear = KANLinear(in_features, out_features, grid_size, spline_order, **kwargs)
        
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.layernorm = nn.LayerNorm(out_features)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, x):
        x = self.kan_linear(x)
        
        if self.use_layernorm:
            x = self.layernorm(x)
        
        if self.dropout is not None:
            x = self.dropout(x)
        
        return x


class BaselineKAN(nn.Module):
    """
    完整的 Pixel-wise KAN Baseline
    使用多层 KAN 进行时序预测
    """
    def __init__(
        self, 
        input_dim, 
        seq_len, 
        pred_len, 
        hidden_dim=64,
        num_layers=2,
        grid_size=5,
        spline_order=3,
        use_layernorm=True,
        dropout=0.1
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_layers = num_layers
        
        # 输入展平维度
        flat_in = seq_len * input_dim
        flat_out = pred_len * 2
        
        # 构建多层 KAN
        layers = []
        
        # 第一层: flat_in -> hidden_dim
        layers.append(KANLayer(
            flat_in, 
            hidden_dim, 
            grid_size=grid_size,
            spline_order=spline_order,
            use_layernorm=use_layernorm,
            dropout=dropout
        ))
        
        # 中间层: hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            layers.append(KANLayer(
                hidden_dim,
                hidden_dim,
                grid_size=grid_size,
                spline_order=spline_order,
                use_layernorm=use_layernorm,
                dropout=dropout
            ))
        
        # 最后一层: hidden_dim -> flat_out
        if num_layers > 1:
            layers.append(KANLayer(
                hidden_dim,
                flat_out,
                grid_size=grid_size,
                spline_order=spline_order,
                use_layernorm=False,  # 输出层不用 LayerNorm
                dropout=0.0
            ))
        else:
            # 如果只有一层，直接从输入到输出
            layers[0] = KANLayer(
                flat_in,
                flat_out,
                grid_size=grid_size,
                spline_order=spline_order,
                use_layernorm=False,
                dropout=0.0
            )
        
        self.kan_layers = nn.Sequential(*layers)

    def forward(self, x, pred_len=None):
        """
        x: [B, T, C, H, W]
        返回: [B, T_out, 2, H, W]
        """
        B, T, C, H, W = x.shape
        
        # 像素级展平: [B, H, W, T, C] -> [B*H*W, T*C]
        x = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, -1)
        
        # 通过 KAN 层
        x = self.kan_layers(x)  # [B*H*W, T_out*2]
        
        # 还原形状: [B*H*W, T_out*2] -> [B, T_out, 2, H, W]
        x = x.reshape(B, H, W, self.pred_len, 2).permute(0, 3, 4, 1, 2)
        
        return x



class ConvLSTMCell(nn.Module):
    """标准 ConvLSTM Cell"""
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size,
            padding=padding,
            bias=bias
        )
    
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size, spatial_size):
        h, w = spatial_size
        device = self.conv.weight.device
        return (
            torch.zeros(batch_size, self.hidden_dim, h, w, device=device),
            torch.zeros(batch_size, self.hidden_dim, h, w, device=device)
        )


class BaselineConvLSTM(nn.Module):
    """
    ConvLSTM Baseline
    适用场景: 时空序列预测, 标准LSTM时间建模
    """
    def __init__(self, input_dim, hidden_dims, num_layers, output_dim, kernel_size=3):
        super().__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims if isinstance(hidden_dims, list) else [hidden_dims] * num_layers
        self.output_dim = output_dim
        
        cell_list = []
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else self.hidden_dims[i-1]
            cell_list.append(ConvLSTMCell(cur_input_dim, self.hidden_dims[i], kernel_size))
        self.cell_list = nn.ModuleList(cell_list)
        
        self.output_conv = nn.Conv2d(self.hidden_dims[-1], output_dim, kernel_size=1)
    
    def forward(self, x, pred_len):
        
        b, seq_len, _, height, width = x.size()  # ← 改名为 height, width
        
        # 初始化隐状态
        hidden_states = [self.cell_list[i].init_hidden(b, (height, width)) for i in range(self.num_layers)]
        
        # Encoder: 处理输入序列
        for time_step in range(seq_len):
            x_t = x[:, time_step]
            for layer_idx in range(self.num_layers):
                h_t, c_t = self.cell_list[layer_idx](x_t, hidden_states[layer_idx])  # ← 改名为 h_t, c_t
                hidden_states[layer_idx] = (h_t, c_t)
                x_t = h_t
        
        # Decoder: 预测未来
        outputs = []
        last_output = self.output_conv(hidden_states[-1][0])
        
        for t in range(pred_len):
            # 【修改2】使用 input_dim 进行判断和填充
            if self.output_dim < self.input_dim:
                padding = torch.zeros(b, self.input_dim - self.output_dim, height, width, device=x.device)
                decoder_input = torch.cat([last_output, padding], dim=1)
            else:
                decoder_input = last_output[:, :self.input_dim]
            
            x_t = decoder_input
            for layer_idx in range(self.num_layers):
                h_t, c_t = self.cell_list[layer_idx](x_t, hidden_states[layer_idx])
                hidden_states[layer_idx] = (h_t, c_t)
                x_t = h_t
            
            last_output = self.output_conv(hidden_states[-1][0])
            outputs.append(last_output)
        
        return torch.stack(outputs, dim=1)

class ConvGRUCell(nn.Module):
    """标准 ConvGRU Cell"""
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        
        self.conv_gates = nn.Conv2d(
            input_dim + hidden_dim,
            2 * hidden_dim,
            kernel_size,
            padding=padding,
            bias=bias
        )
        
        self.conv_can = nn.Conv2d(
            input_dim + hidden_dim,
            hidden_dim,
            kernel_size,
            padding=padding,
            bias=bias
        )
    
    def forward(self, x, h_prev):
        combined = torch.cat([x, h_prev], dim=1)
        
        gates = self.conv_gates(combined)
        r, z = torch.split(gates, self.hidden_dim, dim=1)
        r = torch.sigmoid(r)
        z = torch.sigmoid(z)
        
        combined_r = torch.cat([x, r * h_prev], dim=1)
        h_tilde = torch.tanh(self.conv_can(combined_r))
        
        h_next = (1 - z) * h_prev + z * h_tilde
        
        return h_next
    
    def init_hidden(self, batch_size, spatial_size):
        h, w = spatial_size
        return torch.zeros(batch_size, self.hidden_dim, h, w, device=self.conv_gates.weight.device)


class BaselineConvGRU(nn.Module):
    """
    ConvGRU Baseline
    """
    def __init__(self, input_dim, hidden_dims, num_layers, output_dim, kernel_size=3):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dims = hidden_dims if isinstance(hidden_dims, list) else [hidden_dims] * num_layers
        self.output_dim = output_dim
        self.input_dim = input_dim
        
        cell_list = []
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else self.hidden_dims[i-1]
            cell_list.append(ConvGRUCell(cur_input_dim, self.hidden_dims[i], kernel_size))
        self.cell_list = nn.ModuleList(cell_list)
        
        self.output_conv = nn.Conv2d(self.hidden_dims[-1], output_dim, kernel_size=1)
    
    def forward(self, x, pred_len):
        
        b, seq_len, _, height, width = x.size()  # ← 改名
        
        hidden_states = [self.cell_list[i].init_hidden(b, (height, width)) for i in range(self.num_layers)]
        
        # Encoder
        for time_step in range(seq_len):
            x_t = x[:, time_step]
            for layer_idx in range(self.num_layers):
                h_t = self.cell_list[layer_idx](x_t, hidden_states[layer_idx])  # ← 改名
                hidden_states[layer_idx] = h_t
                x_t = h_t
        
        # Decoder
        outputs = []
        last_output = self.output_conv(hidden_states[-1])
        
        for t in range(pred_len):
            # 【使用 input_dim 进行判断和填充
            if self.output_dim < self.input_dim:
                padding = torch.zeros(b, self.input_dim - self.output_dim, height, width, device=x.device)
                decoder_input = torch.cat([last_output, padding], dim=1)
            else:
                decoder_input = last_output[:, :self.input_dim]
            
            x_t = decoder_input
            for layer_idx in range(self.num_layers):
                h_t = self.cell_list[layer_idx](x_t, hidden_states[layer_idx])
                hidden_states[layer_idx] = h_t
                x_t = h_t
            
            last_output = self.output_conv(hidden_states[-1])
            outputs.append(last_output)
        
        return torch.stack(outputs, dim=1)



def warp(input, flow):
    """
    Warp function for TrajGRU
    根据光流场对特征图进行变形
    
    Args:
        input: [B, C, H, W] - 输入特征图
        flow: [B, 2, H, W] - 光流场 (dx, dy)
    Returns:
        output: [B, C, H, W] - 变形后的特征图
    """
    B, C, H, W = input.size()
    device = input.device
    
    # 创建基础网格
    xx = torch.arange(0, W, device=device).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, device=device).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    
    # 加上光流
    vgrid = grid + flow
    
    # 归一化到 [-1, 1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(input, vgrid, align_corners=True)
    
    return output


class TrajGRUCell(nn.Module):
    """
    Trajectory GRU Cell (优化版)
    """
    def __init__(self, input_dim, hidden_dim, kernel_size=5, L=9):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.L = L  # 光流场数量
        
        # 添加网格缓存
        self.register_buffer('base_grid', None)
        self._cached_size = None
        
        padding = kernel_size // 2
        
        # 输入到隐状态 (标准GRU的3个门)
        self.i2h = nn.Conv2d(
            input_dim,
            hidden_dim * 3,
            kernel_size,
            padding=padding
        )
        
        # 流场生成网络
        self.i2f = nn.Conv2d(input_dim, 32, 5, padding=2)
        self.h2f = nn.Conv2d(hidden_dim, 32, 5, padding=2)
        self.flows_conv = nn.Conv2d(32, L * 2, 5, padding=2)
        
        # Warped特征融合 (1x1卷积学习权重)
        self.ret = nn.Conv2d(hidden_dim * L, hidden_dim * 3, 1)
    
    # 添加这个方法
    def _get_grid(self, B, H, W, device):
        """获取归一化网格（带缓存）"""
        if self.base_grid is None or self._cached_size != (H, W):
            # 创建归一化网格 [-1, 1]
            yy = torch.linspace(-1, 1, H, device=device, dtype=torch.float32)
            xx = torch.linspace(-1, 1, W, device=device, dtype=torch.float32)
            yy, xx = torch.meshgrid(yy, xx, indexing='ij')
            
            # [1, H, W, 2] - grid_sample 需要的格式
            grid = torch.stack([xx, yy], dim=-1).unsqueeze(0)
            
            # 注册为 buffer（会随模型移动到 GPU/CPU）
            self.register_buffer('base_grid', grid, persistent=False)
            self._cached_size = (H, W)
        
        # expand 是零拷贝操作，非常快
        return self.base_grid.expand(B, -1, -1, -1)
    
    def _flow_generator(self, x, h):
        """生成 L 个光流场"""
        if x is not None:
            i2f = self.i2f(x)
            h2f = self.h2f(h)
            f_conv = torch.relu(i2f + h2f)
        else:
            f_conv = torch.relu(self.h2f(h))
        
        # 生成 flows: [B, L*2, H, W]
        flows = self.flows_conv(f_conv)
        # 分割成 L 个 (dx, dy)
        flows = torch.split(flows, 2, dim=1)
        
        return flows
    
    def forward(self, x, h_prev):
        """
        Args:
            x: [B, input_dim, H, W] or None
            h_prev: [B, hidden_dim, H, W]
        Returns:
            h_next: [B, hidden_dim, H, W]
        """
        B, _, H, W = h_prev.shape
        
        # 1. 生成光流场
        flows = self._flow_generator(x, h_prev)
        
        # 2. 获取缓存的归一化网格
        base_grid = self._get_grid(B, H, W, h_prev.device)
        
        # 3. 对每个光流场做 warp
        wrapped_data = []
        for flow in flows:
            # 归一化光流 [B, 2, H, W] → [B, H, W, 2]
            flow_norm = torch.stack([
                2.0 * flow[:, 0] / max(W - 1, 1),
                2.0 * flow[:, 1] / max(H - 1, 1)
            ], dim=1).permute(0, 2, 3, 1)
            
            # 应用负光流（在归一化空间）
            vgrid = base_grid - flow_norm
            
            # 用优化的 grid_sample
            wrapped = F.grid_sample(
                h_prev, 
                vgrid,
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            )
            wrapped_data.append(wrapped)
        
        # 4. 拼接所有 warped features
        wrapped_data = torch.cat(wrapped_data, dim=1)  # [B, hidden*L, H, W]
        
        # 5. 1x1 卷积融合
        h2h = self.ret(wrapped_data)
        h2h_r, h2h_z, h2h_n = torch.split(h2h, self.hidden_dim, dim=1)
        
        # 6. GRU 更新
        if x is not None:
            i2h = self.i2h(x)
            i2h_r, i2h_z, i2h_n = torch.split(i2h, self.hidden_dim, dim=1)
            
            r = torch.sigmoid(i2h_r + h2h_r)
            z = torch.sigmoid(i2h_z + h2h_z)
            n = torch.tanh(i2h_n + r * h2h_n)
        else:
            r = torch.sigmoid(h2h_r)
            z = torch.sigmoid(h2h_z)
            n = torch.tanh(r * h2h_n)
        
        h_next = (1 - z) * h_prev + z * n
        
        return h_next
    
    def init_hidden(self, batch_size, spatial_size):
        h, w = spatial_size
        return torch.zeros(batch_size, self.hidden_dim, h, w, device=self.i2h.weight.device)


class BaselineTrajGRU(nn.Module):
    """
    TrajGRU Baseline
    """
    def __init__(self, input_dim, hidden_dims, num_layers, output_dim, L=9):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dims = hidden_dims if isinstance(hidden_dims, list) else [hidden_dims] * num_layers
        self.output_dim = output_dim
        self.input_dim = input_dim
        
        cell_list = []
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else self.hidden_dims[i-1]
            cell_list.append(TrajGRUCell(cur_input_dim, self.hidden_dims[i], 5, L))
        self.cell_list = nn.ModuleList(cell_list)
        
        self.output_conv = nn.Conv2d(self.hidden_dims[-1], output_dim, kernel_size=1)
    
    def forward(self, x, pred_len):
          
        b, seq_len, _, height, width = x.size()
        
        hidden_states = [self.cell_list[i].init_hidden(b, (height, width)) for i in range(self.num_layers)]
        
        # Encoder
        for time_step in range(seq_len):
            x_t = x[:, time_step]
            for layer_idx in range(self.num_layers):
                h_t = self.cell_list[layer_idx](x_t, hidden_states[layer_idx])
                hidden_states[layer_idx] = h_t
                x_t = h_t
        
        # Decoder
        outputs = []
        last_output = self.output_conv(hidden_states[-1])
        
        for t in range(pred_len):
            if self.output_dim < self.input_dim:
                padding = torch.zeros(b, self.input_dim - self.output_dim, height, width, device=x.device)
                decoder_input = torch.cat([last_output, padding], dim=1)
            else:
                decoder_input = last_output[:, :self.input_dim]
            
            x_t = decoder_input
            for layer_idx in range(self.num_layers):
                h_t = self.cell_list[layer_idx](x_t, hidden_states[layer_idx])
                hidden_states[layer_idx] = h_t
                x_t = h_t
            
            last_output = self.output_conv(hidden_states[-1])
            outputs.append(last_output)
        
        return torch.stack(outputs, dim=1)

class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, memory_dim, kernel_size=5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        
        padding = kernel_size // 2
        
        # 时间记忆流 (Temporal Memory)
        self.conv_t = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size,
            padding=padding
        )
        
        # 空间记忆流 (Spatial Memory)
        self.conv_s = nn.Conv2d(
            hidden_dim + memory_dim,
            4 * memory_dim,
            kernel_size,
            padding=padding
        )
        
        # 时空融合门
        self.conv_fusion = nn.Conv2d(
            hidden_dim + memory_dim,
            hidden_dim,
            kernel_size=1
        )
    
    def forward(self, x, h_prev, c_prev, m_prev):
        """
        Args:
            x: [B, input_dim, H, W] - 输入
            h_prev: [B, hidden_dim, H, W] - 上一时刻隐状态
            c_prev: [B, hidden_dim, H, W] - 上一时刻细胞状态
            m_prev: [B, memory_dim, H, W] - 上一层的空间记忆
        Returns:
            h_next, c_next, m_next
        """
        # === 时间路径 (Temporal Path) ===
        combined_t = torch.cat([x, h_prev], dim=1)
        gates_t = self.conv_t(combined_t)
        i_t, f_t, g_t, o_t = torch.split(gates_t, self.hidden_dim, dim=1)
        
        i_t = torch.sigmoid(i_t)
        f_t = torch.sigmoid(f_t)
        g_t = torch.tanh(g_t)
        
        # 时间细胞状态更新
        c_new = f_t * c_prev + i_t * g_t
        
        # === 空间路径 (Spatial Path) ===
        combined_s = torch.cat([h_prev, m_prev], dim=1)
        gates_s = self.conv_s(combined_s)
        i_s, f_s, g_s, o_s = torch.split(gates_s, self.memory_dim, dim=1)
        
        i_s = torch.sigmoid(i_s)
        f_s = torch.sigmoid(f_s)
        g_s = torch.tanh(g_s)
        
        # 空间记忆更新
        m_new = f_s * m_prev + i_s * g_s
        
        # === 时空融合 ===
        o_t = torch.sigmoid(o_t)
        combined_output = torch.cat([c_new, m_new], dim=1)
        h_new = o_t * torch.tanh(self.conv_fusion(combined_output))
        
        return h_new, c_new, m_new
    
    def init_hidden(self, batch_size, spatial_size):
        h, w = spatial_size
        device = self.conv_t.weight.device
        return (
            torch.zeros(batch_size, self.hidden_dim, h, w, device=device),
            torch.zeros(batch_size, self.hidden_dim, h, w, device=device),
            torch.zeros(batch_size, self.memory_dim, h, w, device=device)
        )

class BaselinePredRNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_layers, output_dim):
        super().__init__()
        self.input_dim = input_dim  # <--- 1. 记得加上这个 (上一轮的修复)
        self.num_layers = num_layers
        
        # === 核心修复开始 ===
        # PredRNN 强制要求所有层维度一致，因为 Memory 要在层间传递
        if isinstance(hidden_dims, list):
            self.hidden_dim = hidden_dims[0] # 强制取第一个维度作为所有层的维度
        else:
            self.hidden_dim = hidden_dims
        # === 核心修复结束 ===
            
        self.output_dim = output_dim
        
        cell_list = []
        for i in range(num_layers):
            # 第一层输入是 input_dim，后续层输入是 hidden_dim
            cur_input_dim = input_dim if i == 0 else self.hidden_dim
            
            cell_list.append(
                SpatioTemporalLSTMCell(
                    cur_input_dim,
                    self.hidden_dim, # 所有层使用相同的 hidden_dim
                    self.hidden_dim  # 所有层使用相同的 memory_dim
                )
            )
        self.cell_list = nn.ModuleList(cell_list)
        
        # 输出层
        self.output_conv = nn.Conv2d(self.hidden_dim, output_dim, kernel_size=1)

    def forward(self, x, pred_len):
           
        b, seq_len, _, height, width = x.size()
        
        # 初始化状态
        states = [self.cell_list[i].init_hidden(b, (height, width)) for i in range(self.num_layers)]
        
        # Encoder
        for time_step in range(seq_len):
            x_t = x[:, time_step]
            
            # Layer 0 (最底层)
            h_t, c_t, m_t = self.cell_list[0](x_t, *states[0])
            states[0] = (h_t, c_t, m_t)
            x_t = h_t
            
            # Subsequent layers (上层)
            for layer_idx in range(1, self.num_layers):
                h_prev, c_prev, _ = states[layer_idx]
                m_prev = states[layer_idx-1][2] # <--- 这里从下一层获取 Memory
                
                # 如果层之间维度不一样，这里的 m_prev 维度就会和 layer_idx 层期待的不一致
                h_t, c_t, m_t = self.cell_list[layer_idx](x_t, h_prev, c_prev, m_prev)
                states[layer_idx] = (h_t, c_t, m_t)
                x_t = h_t
        
        # Decoder
        outputs = []
        last_output = self.output_conv(states[-1][0])
        
        for t in range(pred_len):
            # <--- 2. 这里也记得应用上一轮的 input_dim 修复
            if self.output_dim < self.input_dim:
                padding = torch.zeros(b, self.input_dim - self.output_dim, height, width, device=x.device)
                decoder_input = torch.cat([last_output, padding], dim=1)
            else:
                decoder_input = last_output[:, :self.input_dim]
            
            x_t = decoder_input
            
            # Layer 0
            h_t, c_t, m_t = self.cell_list[0](x_t, *states[0])
            states[0] = (h_t, c_t, m_t)
            x_t = h_t
            
            # Subsequent layers
            for layer_idx in range(1, self.num_layers):
                h_prev, c_prev, _ = states[layer_idx]
                m_prev = states[layer_idx-1][2]
                h_t, c_t, m_t = self.cell_list[layer_idx](x_t, h_prev, c_prev, m_prev)
                states[layer_idx] = (h_t, c_t, m_t)
                x_t = h_t
            
            last_output = self.output_conv(states[-1][0])
            outputs.append(last_output)
        
        return torch.stack(outputs, dim=1)


class DoubleConv(nn.Module):
    """UNet 基础模块: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """下采样模块: MaxPool -> DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """上采样模块: ConvTranspose -> Concat -> DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        """
        Args:
            x1: 上一层的输出 (要上采样的)
            x2: 跳跃连接的特征 (encoder的)
        """
        x1 = self.up(x1)
        
        # 处理尺寸不匹配
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # 拼接
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class BaselineUNet(nn.Module):
    """
    U-Net Baseline
    适用场景: 超分辨率任务, 多尺度特征提取
    
    特点:
    - Encoder-Decoder 架构
    - 跳跃连接保留空间细节
    - 适合 8x8 -> 64x64 的超分辨率
    """
    def __init__(self, input_dim, seq_len, pred_len, features=[64, 128, 256, 512]):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 输入: 时间维度展平到通道
        in_channels = seq_len * input_dim
        out_channels = pred_len * 2
        
        # Encoder (下采样路径)
        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        
        # Bottleneck (瓶颈层)
        self.bottleneck = DoubleConv(features[3], features[3])
        
        # Decoder (上采样路径)
        self.up1 = Up(features[3], features[2])
        self.up2 = Up(features[2], features[1])
        self.up3 = Up(features[1], features[0])
        
        # 输出层
        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x, pred_len=None):
        """
        Args:
            x: [B, T, C, H, W]
            future_steps: 忽略此参数 (保持接口统一)
        Returns:
            output: [B, pred_len, 2, H, W]
        """
        B, T, C, H, W = x.shape
        
        # 时间维度展平到通道: [B, T*C, H, W]
        x = x.reshape(B, T * C, H, W)
        
        # Encoder
        x1 = self.inc(x)      # [B, 64, H, W]
        x2 = self.down1(x1)   # [B, 128, H/2, W/2]
        x3 = self.down2(x2)   # [B, 256, H/4, W/4]
        x4 = self.down3(x3)   # [B, 512, H/8, W/8]
        
        # Bottleneck
        x5 = self.bottleneck(x4)  # [B, 512, H/8, W/8]
        
        # Decoder (带跳跃连接)
        x = self.up1(x5, x3)  # [B, 256, H/4, W/4]
        x = self.up2(x, x2)   # [B, 128, H/2, W/2]
        x = self.up3(x, x1)   # [B, 64, H, W]
        
        # 输出
        out = self.outc(x)    # [B, pred_len*2, H, W]
        
        # 还原形状
        out = out.reshape(B, self.pred_len, 2, H, W)
        
        return out
