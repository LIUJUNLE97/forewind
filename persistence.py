#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  1. from final_eval_point_{true,pred}.npy compute model / persistence de MSE
  2. compute SS_persist = 1 - MSE_model / MSE_persist 
  3. a1_skill_scores.csv
  4. heatmap

persistence definition):
  y_hat_{t+h} = y_t  
    correct_anchor[i] = true[i-1, 0]  
  
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from pathlib import Path
from matplotlib.colors import TwoSlopeNorm

warnings.filterwarnings('ignore')

# config
PROJECT_ROOT = Path(os.environ.get('WIND_PROJECT_ROOT', Path(__file__).resolve().parent))
BASE = Path(os.environ.get('WIND_RESULTS_DIR', PROJECT_ROOT / '24-24'))
OUTPUT_DIR = Path(os.environ.get('WIND_FIG_DIR', PROJECT_ROOT / 'fig')) / 'fig_skill'
OUT_CSV = Path('a1_skill_scores.csv')

CLIP_COLORBAR = 1.0   

CITIES = ['shanghai', 'chengdu', 'hohhot', 'qingdao', 'guangzhou',
          'urumqi', 'lhasa', 'golmud', 'kunming', 'wuhan', 'fuzhou', 'haikou']
MODELS = ['cnn', 'fno', 'kan', 'convlstm', 'convgru', 'trajgru', 'predrnn', 'unet']
MODEL_DISPLAY = {'cnn': 'CNN', 'fno': 'FNO', 'kan': 'KAN', 'convlstm': 'ConvLSTM',
                 'convgru': 'ConvGRU', 'trajgru': 'TrajGRU', 'predrnn': 'PredRNN',
                 'unet': 'U-Net'}
COASTAL = ['Shanghai', 'Qingdao', 'Fuzhou', 'Guangzhou', 'Haikou']
INLAND  = ['Wuhan', 'Chengdu']
PLATEAU = ['Hohhot', 'Urumqi', 'Golmud', 'Kunming', 'Lhasa']
ORDERED_CITIES = COASTAL + INLAND + PLATEAU
ORDERED_MODELS = ['CNN', 'U-Net', 'ConvLSTM', 'ConvGRU', 'TrajGRU', 'PredRNN', 'FNO', 'KAN']

FIG_WIDTH_IN = 7.09

FONT_SCALE = 1.35
PT_TITLE = 7.0 * FONT_SCALE
PT_LABEL = 6.5 * FONT_SCALE
PT_TICK  = 6.0 * FONT_SCALE
PT_ANNOT = 5.5 * FONT_SCALE
PT_CBAR  = 6.0 * FONT_SCALE
LW_SPINE = 0.5
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Liberation Sans', 'FreeSans', 'DejaVu Sans', 'Arial'],
    'font.size': PT_TICK, 'axes.labelsize': PT_LABEL, 'axes.titlesize': PT_TITLE,
    'xtick.labelsize': PT_TICK, 'ytick.labelsize': PT_TICK, 'axes.linewidth': LW_SPINE,
    'xtick.major.width': LW_SPINE, 'ytick.major.width': LW_SPINE,
    'xtick.major.size': 2.0, 'ytick.major.size': 2.0, 'axes.unicode_minus': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


def process(city, model):
    
    d = BASE / f'analysis_results_{city}' / model
    ft, fp = d / 'final_eval_point_true.npy', d / 'final_eval_point_pred.npy'
    if not (ft.exists() and fp.exists()):
        return None
    true_raw = np.load(ft).astype(np.float64)   # [N, 24]
    pred_raw = np.load(fp).astype(np.float64)

  
    anchor = true_raw[:-1, 0]       # [N-1]  = y_t for samples 1..N-1
    true   = true_raw[1:]           # [N-1, 24] throw sample 0
    pred   = pred_raw[1:]           # [N-1, 24]
    persist = np.repeat(anchor[:, None], true.shape[1], axis=1)  # [N-1, 24]

    se_m = (pred - true) ** 2
    se_p = (persist - true) ** 2

    def ss(mse_m, mse_p):
        return 1 - mse_m / mse_p if mse_p > 1e-12 else np.nan

    rows = []
    mse_m_all, mse_p_all = se_m.mean(), se_p.mean()
    rows.append(dict(City=city.capitalize(), Model=MODEL_DISPLAY[model], Step='all',
                     rmse_model=np.sqrt(mse_m_all), rmse_persist=np.sqrt(mse_p_all),
                     ss_persist=ss(mse_m_all, mse_p_all)))
    for h in range(true.shape[1]):
        mse_m, mse_p = se_m[:, h].mean(), se_p[:, h].mean()
        rows.append(dict(City=city.capitalize(), Model=MODEL_DISPLAY[model], Step=h + 1,
                         rmse_model=np.sqrt(mse_m), rmse_persist=np.sqrt(mse_p),
                         ss_persist=ss(mse_m, mse_p)))
    return rows


def compute():
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
    print(f'写入 {OUT_CSV} ({len(df)} 行)')

    agg = df[df.Step == 'all'].copy()
    agg['ss_persist'] = agg['ss_persist'].replace([np.inf, -np.inf], np.nan)

    print('\n  check check check check check check')
    for city in ['haikou', 'chengdu']:
        sub = agg[(agg.City == city.capitalize()) & (agg.Model == 'CNN')]
        if not sub.empty:
            print(f'  {city} CNN: rmse_model = {sub.rmse_model.values[0]:.4f}'
                  f'  (CSV ref: haikou≈0.579, chengdu≈1.092)')

  
    bad = agg[agg.ss_persist < 0][['City', 'Model', 'ss_persist']]
    print(bad.to_string(index=False) if not bad.empty else '  (无)')

    print('\n=== SS_persist ===')
    print(agg.groupby('City')['ss_persist'].mean()
          .reindex(ORDERED_CITIES).round(3).to_string())
    return agg


def plot(agg, out_stem='skill_persistence'):
    piv = agg.pivot_table(values='ss_persist', index='Model',
                          columns='City', aggfunc='mean')
    am = [m for m in ORDERED_MODELS if m in piv.index]
    ac = [c for c in ORDERED_CITIES if c in piv.columns]
    pivot = piv.loc[am, ac]
    vals = pivot.values
    n_rows, n_cols = pivot.shape

    print('\nSS_persist')
    print(pivot.round(3).to_string())

    row_h, top_pad, bot_pad = 0.34, 0.80, 0.82
    left_pad, right_pad, cbar_gap, cbar_w = 1.05, 0.68, 0.10, 0.14
    heat_w = FIG_WIDTH_IN - left_pad - right_pad
    heat_h = n_rows * row_h
    fig_h = heat_h + top_pad + bot_pad
    fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_h))
    ax_l, ax_b = left_pad / FIG_WIDTH_IN, bot_pad / fig_h
    ax_w, ax_h = heat_w / FIG_WIDTH_IN, heat_h / fig_h
    ax = fig.add_axes([ax_l, ax_b, ax_w, ax_h])
    ax_cb = fig.add_axes([ax_l + ax_w + cbar_gap / FIG_WIDTH_IN, ax_b,
                          cbar_w / FIG_WIDTH_IN, ax_h])

    vmax = CLIP_COLORBAR if CLIP_COLORBAR else np.nanmax(np.abs(vals))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(vals, aspect='auto', cmap='RdBu', norm=norm,
                   interpolation='nearest')
    for sp in ax.spines.values():
        sp.set_linewidth(LW_SPINE)

    for r in range(n_rows):
        for c in range(n_cols):
            v = vals[r, c]
            if np.isnan(v):
                txt, fg = 'NA', '#999999'
            else:
                txt = f'{v:.2f}'
                # 截断后超范围的格子用纯白字
                t = norm(np.clip(v, -vmax, vmax))
                fg = 'white' if abs(t - 0.5) > 0.34 else '#1a1a1a'
            ax.text(c, r, txt, ha='center', va='center',
                    fontsize=PT_ANNOT, color=fg, fontweight='bold')

    ax.set_xticks(np.arange(n_cols)); ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right', fontsize=PT_TICK)
    ax.set_yticklabels(pivot.index, rotation=0, fontsize=PT_TICK)
    ax.tick_params(length=2, width=LW_SPINE, pad=2)

    cn = sum(1 for c in COASTAL if c in pivot.columns)
    inn = sum(1 for c in INLAND if c in pivot.columns)
    for xd in [cn, cn + inn]:
        if 0 < xd < n_cols:
            ax.axvline(xd - 0.5, color='#333333', lw=0.6, ls='--', zorder=5)
    for gname, x0, x1 in [('Coastal', 0, cn), ('Inland', cn, cn + inn),
                          ('Plateau', cn + inn, n_cols)]:
        if x1 > x0 and n_cols > 1:
            ax.text((x0 + x1 - 1) / 2 / (n_cols - 1), 1.015, gname,
                    transform=ax.transAxes, ha='center', va='bottom',
                    fontsize=PT_TICK - 0.5, color='#555555', style='italic')

    ax.set_ylabel('Model', fontsize=PT_LABEL, labelpad=4)
    ax.set_xlabel('City', fontsize=PT_LABEL, labelpad=4)
  
    #   "Color scale is clipped at +/-1; FNO at Golmud has SS = -2.35."
    ax.set_title('Skill score vs persistence', fontsize=PT_TITLE, pad=14, loc='left')

    cb = fig.colorbar(im, cax=ax_cb, extend='min' if CLIP_COLORBAR else 'neither')
    cb.outline.set_linewidth(LW_SPINE)
    cb.ax.tick_params(labelsize=PT_TICK - 0.5, width=LW_SPINE, length=2, pad=2)
    cb.set_label('Skill score', fontsize=PT_CBAR, labelpad=3)
    cb.locator = ticker.MaxNLocator(nbins=5); cb.update_ticks()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suf in ('.png', '.pdf'):
        fig.savefig(OUTPUT_DIR / f'{out_stem}{suf}', dpi=300,
                    bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'\n  ✓ {out_stem}: PNG + PDF -> {OUTPUT_DIR}')


# ════════════════════════════════════════════════════════════════
def main():
    agg = compute()
    plot(agg)


if __name__ == '__main__':
    main()
