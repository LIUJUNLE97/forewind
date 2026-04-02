import numpy as np
import os
import argparse
import matplotlib
# 关键：服务器无显示器运行
matplotlib.use('Agg') 
# 确保导入路径正确
from evaluation.vis import BenchmarkVisualizer

parser = argparse.ArgumentParser()
# 修正默认路径，指向你的 analysis_results_shanghai
parser.add_argument('--output_dir', type=str, 
                    default='/mimer/NOBACKUP/groups/kthmech/junle/guandi_wind_project/analysis_results_golmud')
# 根据你实际跑通的模型修改默认列表
parser.add_argument('--models', nargs='+', default=['cnn', 'fno', 'kan', 'convgru', 'trajgru','unet','predrnn', 'convlstm'], help='List of models to compare')
args = parser.parse_args()

def main():
    print(f"=== Comparing Models: {args.models} ===")
    visualizer = BenchmarkVisualizer(output_dir=args.output_dir)
    
    model_preds = {}
    ground_truth_sample = None  # 专门用来存降维后的真值样本
    
    for model_name in args.models:
        # 进入模型子文件夹查找
        model_sub_dir = os.path.join(args.output_dir, model_name)
        # 注意：这里文件名要和你 recover_test.py 生成的一致
        # 你的脚本里生成的是 result_recover_point_pred.npy，请根据实际情况修改
        # 如果你之前用的是 final_eval_point_pred.npy，就保持原样
        # 建议这里写成和 recover_test.py 一致：
        pred_path = os.path.join(model_sub_dir, "result_recover_point_pred.npy")
        true_path = os.path.join(model_sub_dir, "result_recover_point_true.npy")
        
        # 兼容旧文件名 (以防万一)
        if not os.path.exists(pred_path):
            pred_path = os.path.join(model_sub_dir, "final_eval_point_pred.npy")
            true_path = os.path.join(model_sub_dir, "final_eval_point_true.npy")

        if not os.path.exists(pred_path):
            print(f"⚠️ Warning: {model_name} 的结果未找到: {pred_path}")
            continue
            
        print(f"✅ Loading {model_name}...")
        preds = np.load(pred_path) # 可能形状: [N, 24, 1, 1]
        trues = np.load(true_path) # 可能形状: [N, 24, 1, 1]
        
        # --- 关键修复：取出第0个样本并降维 ---
        # 1. 取出第一个样本 [24, 1, 1]
        p_sample = preds[0]
        t_sample = trues[0]
        
        # 2. 强行展平为 1D 数组 [24]
        # 无论它是 (24, 1, 1) 还是 (24, 1) 都会变成 (24,)
        p_sample = p_sample.flatten()
        t_sample = t_sample.flatten()
        
        # 3. 存入字典
        model_preds[model_name] = p_sample
        
        # 4. 存储真值 (只需要存一次)
        if ground_truth_sample is None:
            ground_truth_sample = t_sample
        
    if ground_truth_sample is not None and len(model_preds) > 0:
        # 画多模型对比图
        print(">>> Drawing comparison plot...")
        # 此时 ground_truth_sample 和 model_preds 里的 value 都是 (24,) 形状，不会报错了
        visualizer.plot_multi_model_series(ground_truth_sample, model_preds, title_suffix="Comparison on Golmud")
        print(f"🎉 Done! 对比图已保存至: {args.output_dir}")
    else:
        print("❌ 错误：未找到任何有效的模型结果，请检查路径。")

if __name__ == "__main__":
    main()