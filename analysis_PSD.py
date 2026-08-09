import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import warnings

# 忽略绘图和计算时的冗余警告，保持终端输出整洁
warnings.filterwarnings("ignore")

def plot_psd_comparison(city, model, base_dir="7-24", output_base_dir="psd_7_24"):
    """
    读取指定城市与模型的真实/预测数据，绘制功率谱密度 (PSD) 对比图，
    并按 psd/city/model.png 的层级结构保存为 PNG 文件。
    """
    target_dir = os.path.join(base_dir, f"analysis_results_{city}", model)
    pred_path = os.path.join(target_dir, "final_eval_point_pred.npy")
    true_path = os.path.join(target_dir, "final_eval_point_true.npy")

    if not (os.path.exists(pred_path) and os.path.exists(true_path)):
        print(f"[-] 缺失数据文件，跳过: {city} - {model}")
        return False

    # 1. 载入时空预测数据
    try:
        pred = np.load(pred_path)
        true = np.load(true_path)
    except Exception as e:
        print(f"[!] 读取 {city} - {model} 时发生错误: {e}")
        return False

    # 2. 选取预报步长进行分析 (第 1 小时 vs 第 24 小时)
    lead_times_to_plot = [0, 23]
    labels = ['1h Lead Time', '24h Lead Time']

    # 设定无衬线字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle(f'Power Spectral Density (PSD) Analysis - {city.capitalize()} ({model.upper()})', fontsize=14)

    fs = 1.0

    for idx, (ax, lead_idx) in enumerate(zip(axes, lead_times_to_plot)):
        ts_pred = pred[:, lead_idx]
        ts_true = true[:, lead_idx]

        # 3. 使用 Welch 方法计算 PSD
        f_true, pxx_true = welch(ts_true, fs=fs, nperseg=2048)
        f_pred, pxx_pred = welch(ts_pred, fs=fs, nperseg=2048)

        # 4. 绘制对数曲线 (Log-Log)
        ax.loglog(f_true, pxx_true, label='Ground Truth', color='black', linewidth=1.5, alpha=0.8)
        ax.loglog(f_pred, pxx_pred, label='Model Prediction', color='red', linewidth=1.5, alpha=0.8)

        # 标注日变化周期 (1/24 hour ≈ 0.0416)
        ax.axvline(x=1/24, color='blue', linestyle='--', alpha=0.5, label='Diurnal Cycle (24h)')

        ax.set_title(f'{labels[idx]}')
        ax.set_xlabel('Frequency (cycles/hour)')
        if idx == 0:
            ax.set_ylabel('Power Spectral Density')
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend()
        ax.set_xlim([1e-3, 0.5])

    plt.tight_layout()

    # 5. 构建输出目录层级
    output_dir = os.path.join(output_base_dir, city)
    os.makedirs(output_dir, exist_ok=True)

    # 6. 保存图像并释放内存
    output_filename = os.path.join(output_dir, f"{model}.png")
    plt.savefig(output_filename, format='png', dpi=300, bbox_inches='tight')

    # 【核心安全机制】：显式关闭图表，防止批量画图撑爆内存
    plt.close(fig)

    return True

if __name__ == "__main__":
    # 实验全量城市列表
    cities = [
        "shanghai", "chengdu", "hohhot", "qingdao",
        "guangzhou", "urumqi", "lhasa", "golmud",
        "kunming", "wuhan", "fuzhou", "haikou"
    ]

    # 实验全量模型列表
    models = [
        "cnn", "fno", "kan", "convlstm",
        "convgru", "trajgru", "predrnn", "unet"
    ]

    total_tasks = len(cities) * len(models)
    print(f"开始批量生成 PSD 分析图，预期任务: {total_tasks} 组...")

    success_count = 0
    for city in cities:
        for model in models:
            success = plot_psd_comparison(city=city, model=model)
            if success:
                success_count += 1
                if success_count % 10 == 0:
                    print(f"  [+] 已完成 {success_count} 组制图任务...")

    print(f"\n执行完毕。成功生成并导出 {success_count} 张 PNG 图像。")
    print("文件已按空间结构存储在当前目录的 ./psd/ 文件夹中。")
