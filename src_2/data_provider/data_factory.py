import os

from .data_loader import UnifiedWeatherDataset, fast_collate_fn
from torch.utils.data import DataLoader

def data_provider(args, flag):
    
    Data = UnifiedWeatherDataset
    timeenc = 0 if args.embed != 'timeF' else 1
   
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size 
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
    else:
        # train / val
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    print(f"[{flag.upper()}] Loading UnifiedWeatherDataset...")
    
    current_loc_config = args.test_location_config if args.fix_location else None

    print(f"[{flag.upper()}] Loading UnifiedWeatherDataset (Fixed Location: {args.fix_location})...")
    original_nc_dir = getattr(
        args,
        'original_nc_dir',
        os.path.join(os.path.dirname(os.path.abspath(args.nc_dir)), 'processed'),
    )
    dataset = Data(
        nc_dir=args.nc_dir,  
        som_map_path=args.som_map_path,
        original_nc_dir=original_nc_dir,
        mode=flag,
        patch_size=args.patch_size,       
        time_steps_in=args.seq_len,
        time_steps_out=args.pred_len,
        location_config=current_loc_config
    )
    
    print(f"[{flag}] Dataset Length: {len(dataset)}")
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        collate_fn=fast_collate_fn,  # 1. 使用自定义的高速 collate
        pin_memory=True,             # 2. 必须开启，加速 CPU->GPU
        persistent_workers=True,     # 3. 必须开启，避免重复启动进程
        prefetch_factor=2
    )
    return dataset, data_loader
