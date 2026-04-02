import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
import warnings

# 忽略不必要的警告，保持终端输出整洁
warnings.filterwarnings("ignore")

# 统一设置学术级图表字体与样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42  # 保证导出 PDF 的字体可编辑

def plot_model_sensitivity_heatmap(df, output_prefix="fig_sensitivity_heatmap"):
    """
    复刻文献范式：绘制不同模型对各物理/时序因子的相对贡献度热力图。
    对每种模型单独训练 BRT，提取 Elevation, Roughness, Lead_Time 的相对重要性。
    """
    print("正在计算各模型的特征相对贡献度矩阵...")
    models = df['Model'].unique()
    features = ['Roughness', 'Elevation', 'Lead_Time']
    
    importance_matrix = []
    
    for model in models:
        df_sub = df[df['Model'] == model]
        X = df_sub[features]
        y = df_sub['RMSE']
        
        # 针对单一模型拟合
        brt = HistGradientBoostingRegressor(random_state=42, max_iter=100)
        brt.fit(X, y)
        
        # 计算特征重要性
        result = permutation_importance(brt, X, y, n_repeats=5, random_state=42)
        importances = result.importances_mean
        
        # 归一化为百分比 (相对贡献率)
        total_importance = importances.sum()
        if total_importance > 0:
            rel_importances = (importances / total_importance) * 100
        else:
            rel_importances = [0, 0, 0]
            
        importance_matrix.append(rel_importances)
        
    # 构建热力图数据表
    df_hm = pd.DataFrame(importance_matrix, columns=features, index=models)
    
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_hm, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={'label': 'Relative Contribution (%)'}, ax=ax)
    
    ax.set_title("Relative Contribution of Driving Factors to Model RMSE", pad=15, fontsize=12, fontweight='bold')
    ax.set_ylabel("Model Architecture")
    ax.set_xlabel("Driving Factors")
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_prefix}.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [+] 敏感度热力图已生成: {output_prefix}.png")


def plot_lmm_forest(df, output_prefix="fig_lmm_forest"):
    """
    复刻文献范式：绘制 LMM 交互项的森林图，展示置信区间与统计显著性。
    客观展示相对于 CNN，其他模型家族在地形惩罚下的误差衰退斜率。
    """
    print("正在计算 LMM 统计效应区间并绘制森林图...")
    # 设定 CNN 为基准
    formula = "RMSE ~ Elevation + Roughness + C(Model_Family, Treatment('CNN')) + Lead_Time + Roughness:C(Model_Family, Treatment('CNN'))"
    lmm_model = smf.mixedlm(formula, df, groups=df["City"])
    res = lmm_model.fit()
    
    # 提取系数与置信区间
    params = res.params
    conf = res.conf_int()
    pvalues = res.pvalues
    
    # 我们主要关注交互项和主效应
    target_terms = [
        "Roughness", 
        "Elevation",
        "Lead_Time",
        "Roughness:C(Model_Family, Treatment('CNN'))[T.Operator]",
        "Roughness:C(Model_Family, Treatment('CNN'))[T.RNN]"
    ]
    
    # 重命名以利于图表展示
    display_names = {
        "Roughness": "Roughness Main Effect",
        "Elevation": "Elevation Main Effect",
        "Lead_Time": "Lead Time Main Effect",
        "Roughness:C(Model_Family, Treatment('CNN'))[T.Operator]": "Roughness × Operator (vs CNN)",
        "Roughness:C(Model_Family, Treatment('CNN'))[T.RNN]": "Roughness × RNN (vs CNN)"
    }
    
    plot_data = []
    for term in target_terms:
        coef = params[term]
        ci_lower = conf.loc[term][0]
        ci_upper = conf.loc[term][1]
        pval = pvalues[term]
        plot_data.append({
            'Term': display_names[term],
            'Coef': coef,
            'CI_Lower': ci_lower,
            'CI_Upper': ci_upper,
            'Significant': 'Yes' if pval < 0.05 else 'No'
        })
        
    df_forest = pd.DataFrame(plot_data)
    
    # 绘制森林图
    fig, ax = plt.subplots(figsize=(9, 4))
    
    # 绘制 0 参照线
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    
    y_pos = np.arange(len(df_forest))[::-1]  # 反转以确保按列表顺序从上到下排列
    
    for i, row in enumerate(df_forest.itertuples()):
        color = 'darkred' if row.Significant == 'Yes' else 'slategray'
        
        # 绘制误差棒
        ax.errorbar(row.Coef, y_pos[i], 
                    xerr=[[row.Coef - row.CI_Lower], [row.CI_Upper - row.Coef]], 
                    fmt='o', color=color, capsize=5, markersize=8, linewidth=2)
        
        # 添加 P 值标注
        pval_text = f"p < 0.05" if row.Significant == 'Yes' else f"n.s."
        ax.text(row.CI_Upper + (ax.get_xlim()[1]*0.05), y_pos[i], pval_text, va='center', color=color)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_forest['Term'])
    ax.set_xlabel("Effect Size (Coefficient & 95% CI)")
    ax.set_title("Forest Plot of Fixed Effects and Interactions (LMM)", pad=15, fontsize=12, fontweight='bold')
    
    # 去除多余的边框线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_prefix}.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [+] 统计效应森林图已生成: {output_prefix}.png")


if __name__ == "__main__":
    csv_path = 'attribution_panel_data.csv'
    
    try:
        df = pd.read_csv(csv_path)
        print("数据加载成功，开始执行可视化流水线...")
        
        plot_model_sensitivity_heatmap(df)
        plot_lmm_forest(df)
        
        print("\n所有学术图表生成完毕。")
    except FileNotFoundError:
        print(f"错误: 找不到数据文件 {csv_path}。请确保数据准备脚本已正确运行。")