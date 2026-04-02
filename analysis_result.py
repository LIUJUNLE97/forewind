#!/usr/bin/env python3
"""
注：这个代码是用于对所有结果做综合评估的，例如24-24的数据。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. 高质量学术配色方案 (提取自预设的期刊风格)
# ============================================

# 离散模型颜色 (用于区分 8 个模型)
MODEL_COLORS = [
    '#1867ad',  # CNN: 经典蓝
    '#f87c01',  # FNO: 明橙色
    '#df1200',  # KAN: 深红色
    '#fdd905',  # ConvLSTM: 亮黄色
    '#4f9bcb',  # ConvGRU: 天蓝色
    '#85623b',  # TrajGRU: 泥棕色
    '#ee7e65',  # PredRNN: 珊瑚红
    '#08316b'   # UNet: 深藏青
]

# 连续渐变色 A (用于 RMSE, MAE 等越小越好的指标)
# 浅黄 -> 金黄 -> 橙红 -> 正红 -> 深棕 (表现从好到坏)
error_colors = ['#fefe51', '#fdbb00', '#f58457', '#df1200', '#392514']
cmap_error = LinearSegmentedColormap.from_list('error_cmap', error_colors)

# 发散渐变色 B (用于 R, ACC 等越高越好的指标)
# 正红(差) -> 珊瑚红 -> 淡黄(中) -> 天蓝 -> 经典蓝(好)
corr_colors = ['#df1200', '#ee7e65', '#fefe98', '#4f9bcb', '#1867ad']
cmap_corr = LinearSegmentedColormap.from_list('corr_cmap', corr_colors)

# 排名专属渐变色 (1为最好用蓝色，数字越大越差用红色)
rank_colors = ['#1867ad', '#4f9bcb', '#fefe98', '#ee7e65', '#df1200']
cmap_rank = LinearSegmentedColormap.from_list('rank_cmap', rank_colors)


# ============================================
# 2. 全局绘图参数配置 (全局放大字体)
# ============================================

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']

# 将基础字号整体放大以适应报告或论文展示需求
plt.rcParams['font.size'] = 22
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['figure.titlesize'] = 28
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['grid.linewidth'] = 1.2
plt.rcParams['lines.linewidth'] = 2.5


# ============================================
# 3. 数据分析基础配置
# ============================================

CITIES = [
    'shanghai', 'chengdu', 'hohhot', 'qingdao', 
    'guangzhou', 'urumqi', 'lhasa', 'golmud', 
    'kunming', 'wuhan', 'fuzhou', 'haikou'
]

MODELS = [
    'cnn', 'fno', 'kan', 'convlstm',
    'convgru', 'trajgru', 'predrnn', 'unet'
]

BASE_DIR = Path('./24-24')
OUTPUT_DIR = Path('aggregated_results')
OUTPUT_DIR.mkdir(exist_ok=True)

# 关键指标分组
KEY_METRICS_GROUPS = {
    'Patch U Component': ['Patch_U_RMSE', 'Patch_U_RMSE_OOD', 'Patch_U_ACC'],
    'Patch V Component': ['Patch_V_RMSE', 'Patch_V_RMSE_OOD', 'Patch_V_ACC'],
    'Patch Speed': ['Patch_Speed_RMSE', 'Patch_Speed_RMSE_OOD', 'Patch_Speed_ACC'],
    'Single Point': ['SinglePoint_RMSE', 'SinglePoint_MAE', 'SinglePoint_MBE', 'SinglePoint_R']
}

MAIN_METRICS = [
    'SinglePoint_RMSE', 'SinglePoint_MAE', 
    'SinglePoint_R', 'Patch_Speed_RMSE'
]


def collect_results():
    """收集所有结果"""
    results = []
    missing_cities = []
    print("\n开始收集数据...")
    print("=" * 70)
    for city in CITIES:
        city_has_data = False
        for model in MODELS:
            csv_path = BASE_DIR / f'analysis_results_{city}' / model / 'final_eval_metrics.csv'
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df['City'] = city.capitalize()
                df['Model'] = model.upper()
                results.append(df)
                print(f"✓ {city:12s} - {model:10s}")
                city_has_data = True
            else:
                print(f"✗ {city:12s} - {model:10s} - 文件不存在")
        if not city_has_data:
            missing_cities.append(city)
    print("=" * 70)
    if not results:
        raise ValueError("没有找到任何数据文件！请检查 BASE_DIR 配置是否正确。")
    combined = pd.concat(results, ignore_index=True)
    print(f"\n✓ 总计: {len(combined)} 条记录")
    if missing_cities:
        print(f"\n⚠ 缺失数据的城市: {', '.join(missing_cities)}")
    return combined


def create_summary_tables(df):
    """创建汇总统计表"""
    metric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                   if col not in ['Lat', 'Lon']]
    model_summary = df.groupby('Model')[metric_cols].agg(['mean', 'std', 'min', 'max']).round(4)
    city_summary = df.groupby('City')[metric_cols].agg(['mean', 'std', 'min', 'max']).round(4)
    model_city_detail = df.groupby(['Model', 'City'])[metric_cols].mean().round(4)
    return model_summary, city_summary, model_city_detail, metric_cols


def create_ranking_tables(df, metric_cols):
    """创建排名表"""
    rankings = {}
    for metric in MAIN_METRICS:
        model_avg = df.groupby('Model')[metric].mean()
        if 'R' in metric or 'ACC' in metric:
            model_avg = model_avg.sort_values(ascending=False)
        else:
            model_avg = model_avg.sort_values(ascending=True)
        rankings[metric] = model_avg
    return pd.DataFrame(rankings).round(4)


# ============================================
# 4. 可视化函数 (局部放大字体 & 应用自定义配色)
# ============================================

def plot_main_metrics_comparison(df, output_dir):
    """绘制主要指标对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Model Performance Comparison', fontweight='bold', y=0.99)
    
    for idx, metric in enumerate(MAIN_METRICS):
        ax = axes[idx // 2, idx % 2]
        plot_data = df.groupby('Model')[metric].agg(['mean', 'std']).reset_index()
        
        # 根据指标特性排序并映射颜色
        if 'R' in metric or 'ACC' in metric:
            plot_data = plot_data.sort_values('mean', ascending=False)
            colors = [cmap_corr(1.0 - i / max(1, len(plot_data)-1)) for i in range(len(plot_data))]
        else:
            plot_data = plot_data.sort_values('mean', ascending=True)
            colors = [cmap_error(i / max(1, len(plot_data)-1)) for i in range(len(plot_data))]
            
        x = np.arange(len(plot_data))
        
        bars = ax.bar(x, plot_data['mean'], yerr=plot_data['std'], 
                      capsize=8, alpha=0.9, edgecolor='black', linewidth=1.5,
                      error_kw={'linewidth': 2.5, 'ecolor': 'black', 'alpha': 0.7})
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)
            bar.set_edgecolor('black')
            
        ax.set_xticks(x)
        ax.set_xticklabels(plot_data['Model'], rotation=45, ha='right', fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' '), fontweight='bold')
        ax.set_title(f'{metric.replace("_", " ")}', fontweight='bold', pad=18)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        for i, (mean_val, std_val) in enumerate(zip(plot_data['mean'], plot_data['std'])):
            ax.text(i, mean_val + std_val + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02, 
                    f'{mean_val:.3f}', ha='center', va='bottom', fontsize=18, fontweight='bold')
            
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison_main.png', dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_city_comparison(df, output_dir):
    """绘制城市对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    fig.suptitle('City Performance Comparison', fontweight='bold', y=0.99)
    
    for idx, metric in enumerate(MAIN_METRICS):
        ax = axes[idx // 2, idx % 2]
        plot_data = df.groupby('City')[metric].agg(['mean', 'std']).reset_index()
        
        # 同上，根据指标特性进行排序与色彩映射
        if 'R' in metric or 'ACC' in metric:
            plot_data = plot_data.sort_values('mean', ascending=False)
            colors = [cmap_corr(1.0 - i / max(1, len(plot_data)-1)) for i in range(len(plot_data))]
        else:
            plot_data = plot_data.sort_values('mean', ascending=True)
            colors = [cmap_error(i / max(1, len(plot_data)-1)) for i in range(len(plot_data))]
            
        x = np.arange(len(plot_data))
        
        bars = ax.bar(x, plot_data['mean'], yerr=plot_data['std'], 
                      capsize=8, alpha=0.9, edgecolor='black', linewidth=1.5,
                      error_kw={'linewidth': 2.5, 'ecolor': 'black', 'alpha': 0.7})
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)
            bar.set_edgecolor('black')
            
        ax.set_xticks(x)
        ax.set_xticklabels(plot_data['City'], rotation=45, ha='right', fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' '), fontweight='bold')
        ax.set_title(f'{metric.replace("_", " ")}', fontweight='bold', pad=18)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        for i, (mean_val, std_val) in enumerate(zip(plot_data['mean'], plot_data['std'])):
            ax.text(i, mean_val + std_val + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02, 
                    f'{mean_val:.3f}', ha='center', va='bottom', fontsize=16, fontweight='bold', rotation=45)
            
    plt.tight_layout()
    plt.savefig(output_dir / 'city_comparison_main.png', dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_comprehensive_heatmap(df, output_dir):
    """绘制综合热力图"""
    # 针对热力图单元格被挤压的情况，大幅增加了画布的大小 (从 24x20 扩大至 32x24)
    fig, axes = plt.subplots(2, 2, figsize=(32, 24))
    fig.suptitle('Performance Heatmap: Model × City', fontweight='bold', y=0.99)
    
    for idx, metric in enumerate(MAIN_METRICS):
        ax = axes[idx // 2, idx % 2]
        pivot_data = df.pivot_table(values=metric, index='Model', columns='City', aggfunc='mean')
        
        if 'R' in metric or 'ACC' in metric:
            cmap = cmap_corr
        else:
            cmap = cmap_error
            
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap=cmap, 
                   ax=ax, cbar_kws={'label': metric.replace('_', ' '), 'shrink': 0.8},
                   linewidths=2, linecolor='white',
                   annot_kws={'fontsize': 18, 'fontweight': 'bold'},
                   vmin=pivot_data.min().min(), vmax=pivot_data.max().max())
        
        ax.set_title(f'{metric.replace("_", " ")}', fontweight='bold', pad=22)
        ax.set_xlabel('City', fontweight='bold', labelpad=12)
        ax.set_ylabel('Model', fontweight='bold', labelpad=12)
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontweight='bold')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontweight='bold')
        
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=18)
        cbar.set_label(metric.replace('_', ' '), fontsize=22, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_heatmap.png', dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_metric_groups(df, output_dir):
    """按指标组绘图"""
    for group_name, metrics in KEY_METRICS_GROUPS.items():
        available_metrics = [m for m in metrics if m in df.columns]
        if not available_metrics:
            continue
            
        n_metrics = len(available_metrics)
        ncols = min(n_metrics, 3)
        nrows = (n_metrics + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(9*ncols, 7.5*nrows))
        if nrows == 1 and ncols == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten()
            
        fig.suptitle(f'{group_name}', fontweight='bold', y=0.99)
        
        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]
            plot_data = df.groupby('Model')[metric].mean().sort_values(
                ascending=('R' not in metric and 'ACC' not in metric)
            )
            
            # 使用预设的 8 种颜色，确保无论顺序如何，同一模型颜色保持一致
            model_order = plot_data.index.tolist()
            model_to_color = dict(zip(['CNN', 'FNO', 'KAN', 'CONVLSTM', 'CONVGRU', 'TRAJGRU', 'PREDRNN', 'UNET'], 
                                     MODEL_COLORS))
            colors = [model_to_color.get(model, '#7f7f7f') for model in model_order]
            
            bars = plot_data.plot(kind='bar', ax=ax, color=colors, 
                                 edgecolor='black', alpha=0.9, linewidth=1.8)
            
            ax.set_title(metric.replace('_', ' '), fontweight='bold', pad=18)
            ax.set_ylabel('Value', fontweight='bold')
            ax.set_xlabel('Model', fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontweight='bold')
            
            for i, (model, val) in enumerate(plot_data.items()):
                ax.text(i, val + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01, 
                        f'{val:.3f}', ha='center', va='bottom', 
                        fontsize=16, fontweight='bold')
                        
        for idx in range(len(available_metrics), len(axes)):
            axes[idx].set_visible(False)
            
        plt.tight_layout()
        filename = f'{group_name.lower().replace(" ", "_")}_comparison.png'
        plt.savefig(output_dir / filename, dpi=600, bbox_inches='tight', facecolor='white')
        plt.close()


def plot_normalized_comparison(df, output_dir):
    """绘制归一化综合对比图"""
    fig, ax = plt.subplots(figsize=(18, 12))
    models = df['Model'].unique()
    normalized_data = {}
    
    for metric in MAIN_METRICS:
        values = df.groupby('Model')[metric].mean()
        if 'R' in metric or 'ACC' in metric:
            norm_values = (values - values.min()) / (values.max() - values.min())
        else:
            norm_values = 1 - (values - values.min()) / (values.max() - values.min())
        normalized_data[metric] = norm_values
        
    norm_df = pd.DataFrame(normalized_data)
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(MAIN_METRICS):
        offset = width * (i - len(MAIN_METRICS)/2 + 0.5)
        # 从模型配色组中顺延提取颜色，区分不同指标
        metric_color = MODEL_COLORS[i % len(MODEL_COLORS)]
        
        ax.bar(x + offset, norm_df[metric], width, 
               label=metric.replace('_', ' '), 
               alpha=0.9, edgecolor='black', linewidth=1.5,
               color=metric_color)
               
    ax.set_xlabel('Model', fontweight='bold', labelpad=15)
    ax.set_ylabel('Normalized Performance Score', fontweight='bold', labelpad=15)
    ax.set_title('Normalized Performance Comparison Across Metrics\n(Higher is Better for All Metrics)', 
                 fontweight='bold', pad=22)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight='bold', rotation=45, ha='right')
    ax.legend(loc='upper left', framealpha=0.95, edgecolor='black', fancybox=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim([0, 1.15])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'normalized_comparison.png', dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_ranking_summary(df, output_dir):
    """绘制排名汇总图"""
    fig, ax = plt.subplots(figsize=(16, 12))
    models = df['Model'].unique()
    ranking_data = []
    
    for metric in MAIN_METRICS:
        values = df.groupby('Model')[metric].mean()
        if 'R' in metric or 'ACC' in metric:
            ranks = values.rank(ascending=False)
        else:
            ranks = values.rank(ascending=True)
        for model in models:
            ranking_data.append({
                'Model': model,
                'Metric': metric.replace('_', ' '),
                'Rank': ranks[model]
            })
            
    rank_df = pd.DataFrame(ranking_data)
    pivot_ranks = rank_df.pivot(index='Model', columns='Metric', values='Rank')
    
    # 使用排名专属的渐变色 (蓝->红)
    sns.heatmap(pivot_ranks, annot=True, fmt='.0f', cmap=cmap_rank, 
               ax=ax, cbar_kws={'label': 'Rank (Lower is Better)', 'shrink': 0.8},
               linewidths=2, linecolor='white',
               annot_kws={'fontsize': 22, 'fontweight': 'bold'},
               vmin=1, vmax=len(models))
               
    ax.set_title(f'Model Ranking Across Metrics\n(1 = Best, {len(models)} = Worst)', 
                fontweight='bold', pad=22)
    ax.set_xlabel('Metric', fontweight='bold', labelpad=15)
    ax.set_ylabel('Model', fontweight='bold', labelpad=15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontweight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontweight='bold')
    
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label('Rank (Lower is Better)', fontsize=22, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ranking_summary.png', dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()


def save_excel_compatible(df, model_summary, city_summary, model_city_detail, ranking_df, output_path):
    """保存Excel文件（兼容xlsxwriter）"""
    try:
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='原始数据', index=False)
            model_summary.to_excel(writer, sheet_name='按模型汇总')
            city_summary.to_excel(writer, sheet_name='按城市汇总')
            model_city_detail.to_excel(writer, sheet_name='模型城市详细')
            ranking_df.to_excel(writer, sheet_name='模型排名')
            for metric in MAIN_METRICS:
                pivot = df.pivot_table(values=metric, index='Model', columns='City', aggfunc='mean')
                sheet_name = metric[:25]
                pivot.to_excel(writer, sheet_name=sheet_name)
        return True
    except Exception as e:
        print(f"⚠ Excel保存失败 ({e})，将只保存CSV文件")
        return False


def main():
    print("\n" + "=" * 70)
    print("   风场预测模型性能汇总分析")
    print("=" * 70)
    
    df = collect_results()
    model_summary, city_summary, model_city_detail, metric_cols = create_summary_tables(df)
    ranking_df = create_ranking_tables(df, metric_cols)
    
    print("\n步骤: 生成可视化图表 (应用学术期刊配色并放大字体)")
    plot_main_metrics_comparison(df, OUTPUT_DIR)
    plot_city_comparison(df, OUTPUT_DIR)
    plot_comprehensive_heatmap(df, OUTPUT_DIR)
    plot_metric_groups(df, OUTPUT_DIR)
    plot_normalized_comparison(df, OUTPUT_DIR)
    plot_ranking_summary(df, OUTPUT_DIR)
    
    print("\n" + "=" * 70)
    print("   分析完成！图表及结果均已更新并保存至:")
    print(f"   {OUTPUT_DIR.absolute()}")
    print("=" * 70 + "\n")

if __name__ == '__main__':
    main()