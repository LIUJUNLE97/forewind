import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

class BenchmarkVisualizer:
    def __init__(self, output_dir="./analysis_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid", font_scale=1.1)
        plt.rcParams['axes.unicode_minus'] = False 

    def plot_time_series(self, pred_series, true_series, title_suffix="", sample_idx=0):
        """单模型拟合图"""
        plt.figure(figsize=(12, 5))
        time_steps = np.arange(len(true_series))
        plt.plot(time_steps, true_series, 'k.-', label='Ground Truth', linewidth=1.5, alpha=0.7)
        plt.plot(time_steps, pred_series, 'r.-', label='Prediction', linewidth=1.5, alpha=0.8)
        plt.fill_between(time_steps, true_series, pred_series, color='red', alpha=0.1, label='Error Gap')
        plt.title(f"Forecast: {title_suffix}", fontsize=14, fontweight='bold')
        plt.xlabel("Forecast Lead Time (Hours)")
        plt.ylabel("Wind Speed (m/s)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        fname = f"seq_fitting_{title_suffix.replace(' ', '_')}_sample{sample_idx}.png"
        plt.savefig(os.path.join(self.output_dir, fname), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_multi_model_series(self, true_series, model_preds_dict, title_suffix="Comparison"):
        """
        [新增] 多模型对比图
        Args:
            true_series: (T,) 真实序列
            model_preds_dict: {'LSTM': (T,), 'CNN': (T,), 'Mamba': (T,)...}
        """
        plt.figure(figsize=(14, 6))
        time_steps = np.arange(len(true_series))
        
        # 画真值 (黑色粗线)
        plt.plot(time_steps, true_series, 'k-', label='Ground Truth', linewidth=2.0, alpha=0.8, zorder=10)
        
        # 画各个模型的预测
        markers = ['o', '^', 's', 'D', 'x']
        colors = sns.color_palette("tab10", len(model_preds_dict))
        
        for i, (name, pred) in enumerate(model_preds_dict.items()):
            marker = markers[i % len(markers)]
            color = colors[i]
            plt.plot(time_steps, pred, label=name, color=color, 
                     linewidth=1.5, marker=marker, markersize=4, alpha=0.8, markevery=4)

        plt.title(f"Multi-Model Comparison: {title_suffix}", fontsize=16, fontweight='bold')
        plt.xlabel("Forecast Lead Time (Hours)")
        plt.ylabel("Wind Speed (m/s)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        fname = f"multi_model_comparison_{title_suffix.replace(' ', '_')}.png"
        save_path = os.path.join(self.output_dir, fname)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   >>> [Plot] 多模型对比图已保存: {save_path}")

    def plot_conditional_bias(self, y_true, model_preds_dict, bins=20, var_name="Wind Speed (m/s)"):
        """条件偏差图"""
        bin_edges = np.linspace(y_true.min(), y_true.max(), bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.figure(figsize=(8, 8))
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2, label='Ideal 1:1')
        for model_name, y_pred in model_preds_dict.items():
            means = []
            for j in range(bins):
                mask = (y_true >= bin_edges[j]) & (y_true < bin_edges[j+1])
                if mask.sum() > 10:
                    means.append(y_pred[mask].mean())
                else:
                    means.append(np.nan)
            plt.plot(bin_centers, means, 'o-', label=model_name, lw=2)
        plt.xlabel(f"True {var_name}")
        plt.ylabel(f"Predicted {var_name}")
        plt.title("Conditional Bias Analysis")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "conditional_bias.png"), bbox_inches='tight')
        plt.close()