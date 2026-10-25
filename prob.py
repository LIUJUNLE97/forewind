#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 persistence skill score，directly to point_pred/true.npy

  final_eval_point_true.npy / _pred.npy : shape [N, 24], single point velocity (m/s)
  
def persistence :
 Use the first ground-truth frame in each sample's forecast window, y[:, 0],
 as the initialization anchor and hold it constant across all 24 forecast steps.

output
  a1_skill_scores.csv  (for each city-station/model: rmse_model / rmse_persist / ss_persist, per-leadtime)
  config SS<0  +  SS 
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get('WIND_PROJECT_ROOT', Path(__file__).resolve().parent))
BASE = Path(os.environ.get('WIND_RESULTS_DIR', PROJECT_ROOT / '24-24'))
OUT_CSV = Path('a1_skill_scores.csv')

CITIES = ['shanghai', 'chengdu', 'hohhot', 'qingdao', 'guangzhou',
          'urumqi', 'lhasa', 'golmud', 'kunming', 'wuhan', 'fuzhou', 'haikou']
MODELS = ['cnn', 'fno', 'kan', 'convlstm', 'convgru', 'trajgru', 'predrnn', 'unet']
MODEL_DISPLAY = {'cnn': 'CNN', 'fno': 'FNO', 'kan': 'KAN', 'convlstm': 'ConvLSTM',
                 'convgru': 'ConvGRU', 'trajgru': 'TrajGRU', 'predrnn': 'PredRNN',
                 'unet': 'U-Net'}
ORDERED_CITIES = ['Shanghai', 'Qingdao', 'Fuzhou', 'Guangzhou', 'Haikou',
                  'Wuhan', 'Chengdu', 'Hohhot', 'Urumqi', 'Golmud', 'Kunming', 'Lhasa']


def persistence_pred(true):
    """true: [N, T]. 返回 persistence 预测 [N, T]: 每步 = 该样本首帧真值。"""
    anchor = true[:, [0]]                  # [N, 1]
    return np.repeat(anchor, true.shape[1], axis=1)


def process(city, model):
    d = BASE / f'analysis_results_{city}' / model
    ft = d / 'final_eval_point_true.npy'
    fp = d / 'final_eval_point_pred.npy'
    if not (ft.exists() and fp.exists()):
        return None
    true = np.load(ft).astype(np.float64)   # [N, 24]
    pred = np.load(fp).astype(np.float64)
    persist = persistence_pred(true)

    rows = []
    # 'all': 对所有 (sample, step) 平方误差求总均值 (与 CSV 的 RMSE 同口径)
    se_m = (pred - true) ** 2
    se_p = (persist - true) ** 2
    mse_m_all, mse_p_all = se_m.mean(), se_p.mean()
    rows.append(dict(City=city.capitalize(), Model=MODEL_DISPLAY[model], Step='all',
                     rmse_model=np.sqrt(mse_m_all),
                     rmse_persist=np.sqrt(mse_p_all),
                     ss_persist=1 - mse_m_all / mse_p_all))
    # per-leadtime
    for h in range(true.shape[1]):
        mse_m, mse_p = se_m[:, h].mean(), se_p[:, h].mean()
        rows.append(dict(City=city.capitalize(), Model=MODEL_DISPLAY[model],
                         Step=h + 1,
                         rmse_model=np.sqrt(mse_m),
                         rmse_persist=np.sqrt(mse_p),
                         ss_persist=1 - mse_m / mse_p))
    return rows


def main():
    all_rows = []
    print('计算 persistence skill score...')
    for city in CITIES:
        for model in MODELS:
            r = process(city, model)
            if r:
                all_rows.extend(r)
        print(f'  ✓ {city}')
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)
    print(f'\n写入 {OUT_CSV} ({len(df)} 行)')

    agg = df[df.Step == 'all'].copy()

    # 健全性检查: 重算的 model RMSE 应与 CSV 的 SinglePoint_RMSE 吻合
    print('\n=== 健全性检查 (重算 RMSE, 应与 final_eval_metrics 吻合) ===')
    for city in ['haikou', 'chengdu']:
        sub = agg[(agg.City == city.capitalize()) & (agg.Model == 'CNN')]
        if not sub.empty:
            print(f'  {city} CNN: 重算 rmse_model = '
                  f'{sub.rmse_model.values[0]:.3f}')

    print('\n=== SS_persist < 0 的配置 (打不过 persistence) ===')
    bad = agg[agg.ss_persist < 0][['City', 'Model', 'ss_persist']]
    print(bad.to_string(index=False) if not bad.empty else '  (无)')

    print('\n=== 各站平均 SS_persist (Chengdu 检验) ===')
    print(agg.groupby('City')['ss_persist'].mean()
          .reindex(ORDERED_CITIES).round(3).to_string())

    print('\n=== 各站 × 模型 SS_persist 透视表 ===')
    piv = agg.pivot_table(values='ss_persist', index='Model',
                          columns='City', aggfunc='mean')
    piv = piv.reindex(columns=[c for c in ORDERED_CITIES if c in piv.columns])
    print(piv.round(3).to_string())


if __name__ == '__main__':
    main()
