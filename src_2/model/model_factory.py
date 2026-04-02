
import torch
from .basic_models import (
    BaselineCNN,
    BaselineFNO,
    BaselineKAN,
    BaselineConvLSTM,
    BaselineConvGRU,
    BaselineTrajGRU,
    BaselinePredRNN,
    BaselineUNet
)


def get_model(args, device):
    model_name = args.model_name.lower()
    
    # 输入维度 = 5个物理变量 + 4个时间编码 = 9
    input_dim = 9
    
    if model_name == 'cnn':
        return BaselineCNN(
            in_channels=input_dim, 
            seq_len=args.seq_len, 
            pred_len=args.pred_len,
            hidden_dim=256
        ).to(device)
    
    elif model_name == 'fno':
        return BaselineFNO(
            input_dim=input_dim, 
            seq_len=args.seq_len, 
            pred_len=args.pred_len
        ).to(device)
    
    elif model_name == 'kan':
        return BaselineKAN(
            input_dim=input_dim, 
            seq_len=args.seq_len, 
            pred_len=args.pred_len
        ).to(device)
    
    elif model_name == 'convlstm':
        return BaselineConvLSTM(
            input_dim=input_dim,
            hidden_dims=[64, 64],
            num_layers=2,
            output_dim=2
        ).to(device)
    
    elif model_name == 'convgru':
        return BaselineConvGRU(
            input_dim=input_dim,
            hidden_dims=[64,64],
            num_layers=2,
            output_dim=2
        ).to(device)
    
    elif model_name == 'trajgru':
        return BaselineTrajGRU(
            input_dim=input_dim,
            hidden_dims=[64, 64],
            num_layers=2,
            output_dim=2,
            L=9
        ).to(device)
    
    elif model_name == 'predrnn':
        return BaselinePredRNN(
            input_dim=input_dim,
            hidden_dims=[64, 64],
            num_layers=2,
            output_dim=2
        ).to(device)

    elif model_name == 'unet':  
        return BaselineUNet(
            input_dim=input_dim,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            features=[64, 128, 256, 512]
        ).to(device)
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")