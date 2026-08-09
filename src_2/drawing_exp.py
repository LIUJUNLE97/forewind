import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# 服务器无显示器运行
matplotlib.use('Agg') 

def parse_args():
    parser = argparse.ArgumentParser(description="Batch Visualization for Multiple Regions")
    project_root = Path(os.environ.get('WIND_PROJECT_ROOT', Path(__file__).resolve().parents[1]))
    parser.add_argument('--base_dir', type=str, 
                        default=str(project_root / '24-24'),
                        help='Base directory containing regional result folders')
    parser.add_argument('--regions', nargs='+', default=['golmud', 'shanghai','haikou','lhasa','kunming','qingdao','fuzhou','hohhot','guangzhou','urumqi','chengdu', 'wuhan'], 
                        help='List of regions to process')
    parser.add_argument('--models', nargs='+', 
                        default=['cnn', 'fno', 'kan', 'convgru', 'trajgru', 'unet', 'predrnn', 'convlstm'], 
                        help='List of models to compare')
    parser.add_argument('--cols', type=int, default=4, help='Number of columns in the puzzle grid')
    return parser.parse_args()

def load_model_data(region_path, models):
    """加载特定地区所有模型的数据"""
    model_preds = {}
    ground_truth = None
    
    for model_name in models:
        model_sub_dir = os.path.join(region_path, model_name)
        pred_names = ["result_recover_point_pred.npy", "final_eval_point_pred.npy"]
        true_names = ["result_recover_point_true.npy", "final_eval_point_true.npy"]
        
        found = False
        for p_n, t_n in zip(pred_names, true_names):
            p_path = os.path.join(model_sub_dir, p_n)
            t_path = os.path.join(model_sub_dir, t_n)
            
            if os.path.exists(p_path):
                preds = np.load(p_path)
                trues = np.load(t_path)
                model_preds[model_name] = preds[0].flatten()
                if ground_truth is None:
                    ground_truth = trues[0].flatten()
                found = True
                break
        
        if not found:
            print(f"  ⚠️ Warning: {model_name} in {region_path} results not found.")
            
    return ground_truth, model_preds

def main():
    # --- 1. 参数与全局字体设置 ---
    args = parse_args()
    
    # 全局字体配置 (必须在 subplots 之前)
    plt.rcParams.update({
        'font.size': 18,            # 基础字号
        'axes.titlesize': 24,       # 子图标题 (城市名)
        'axes.labelsize': 20,       # 轴标签 (Wind Speed等)
        'xtick.labelsize': 16,      # 刻度值
        'ytick.labelsize': 16,
        'legend.fontsize': 12,      # 图例
        'font.weight': 'bold',      # 字体加粗
        'axes.labelweight': 'bold'
    })

    n_regions = len(args.regions)
    n_rows = (n_regions + args.cols - 1) // args.cols
    
    # 调大画布尺寸 (figsize)，给大字体留空间
    fig, axes = plt.subplots(n_rows, args.cols, figsize=(10 * args.cols, 6 * n_rows), constrained_layout=True)
    
    if n_regions == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()

    print(f"=== 正在处理地区: {args.regions} ===")

    # --- 2. 循环绘制每个子图 ---
    for i, region in enumerate(args.regions):
        region_folder = f"analysis_results_{region}"
        region_path = os.path.join(args.base_dir, region_folder)
        ax = axes_flat[i]
        
        print(f"✅ 正在处理: {region}...")
        true_data, pred_dict = load_model_data(region_path, args.models)
        
        if true_data is not None and len(pred_dict) > 0:
            ax.plot(true_data, label='Ground Truth', color='black', linewidth=2.5, linestyle='--')
            for model_name, p_data in pred_dict.items():
                ax.plot(p_data, label=model_name.upper(), alpha=0.7, linewidth=1.5)
            
            ax.set_title(f"{region.capitalize()}")
            ax.set_xlabel("Time (h)")
            ax.set_ylabel("Wind Speed")
            # Keep the outer frame of every regional prediction panel crisp
            # and consistent with the other prediction visualizations.
            for spine in ax.spines.values():
                spine.set_color('black')
                spine.set_linewidth(1.2)
            
            # --- 优化后的图例设置 ---
            if i == 0: 
                ax.legend(loc='upper left', 
                          frameon=True, 
                          framealpha=0.4,    # 半透明处理
                          shadow=False,      # 去掉阴影，减少笨重感
                          fontsize=11,       # 适当缩小图例字号
                          ncol=2,            # 分两列排布，减少高度占用
                          handletextpad=0.4) # 缩小图标与文字的间距
        else:
            ax.text(0.5, 0.5, "Data Missing", ha='center', va='center')

    # --- 3. 隐藏多余的空白子图 ---
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    # --- 4. 保存图片并结束 main ---
    output_path = os.path.join(args.base_dir, "combined_regions_comparison.png")
    plt.savefig(output_path, dpi=300)
    print(f"\n🎉 处理完成！图片保存在: {output_path}")

# --- 脚本入口 ---
if __name__ == "__main__":
    main()
