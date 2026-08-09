#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A11: Two-way fixed-effect variance decomposition (Station × Model)

This script decomposes the variance of station-model aggregate metrics into:
  1. Station / lower-boundary setting
  2. Model architecture
  3. Station × Model interaction

Important statistical interpretation:
  - This is a descriptive fixed-effect variance decomposition, not an
    inferential ANOVA with p-values.
  - Each station-model pair contributes one aggregate score.
  - Therefore the Station × Model term absorbs the remaining structured
    variation, and no residual significance test is reported.

Inputs:
  - final_eval_metrics.csv for each station-model pair
      Required / optional columns:
        SinglePoint_RMSE
        SinglePoint_R
        Patch_Speed_ACC
  - a1_skill_scores.csv
      Required columns:
        City, Model, Step, ss_persist

Outputs:
  - Console: publication-ready summary + LaTeX table
  - a11_anova.csv: tidy variance-decomposition table
  - a11_input_metrics.csv: merged station-model metrics used as input
  - anova_variance_decomposition.png / .pdf
  - anova_latex_table.txt
"""

import warnings
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(os.environ.get("WIND_PROJECT_ROOT", Path(__file__).resolve().parent))
BASE = Path(os.environ.get("WIND_RESULTS_DIR", PROJECT_ROOT / "24-24"))
SS_CSV = Path("a1_skill_scores.csv")
OUTPUT_DIR = Path(os.environ.get("WIND_FIG_DIR", PROJECT_ROOT / "fig")) / "fig_anova"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CITIES = [
    "shanghai", "chengdu", "hohhot", "qingdao", "guangzhou",
    "urumqi", "lhasa", "golmud", "kunming", "wuhan", "fuzhou", "haikou"
]

MODELS_KEY = [
    "cnn", "fno", "kan", "convlstm", "convgru", "trajgru", "predrnn", "unet"
]

MODEL_DISPLAY = {
    "cnn": "CNN",
    "fno": "FNO",
    "kan": "KAN",
    "convlstm": "ConvLSTM",
    "convgru": "ConvGRU",
    "trajgru": "TrajGRU",
    "predrnn": "PredRNN",
    "unet": "U-Net",
}

COASTAL = ["Shanghai", "Qingdao", "Fuzhou", "Guangzhou", "Haikou"]
INLAND = ["Wuhan", "Chengdu"]
PLATEAU = ["Hohhot", "Urumqi", "Golmud", "Kunming", "Lhasa"]

ORDERED_CITIES = COASTAL + INLAND + PLATEAU
ORDERED_MODELS = ["CNN", "U-Net", "ConvLSTM", "ConvGRU", "TrajGRU", "PredRNN", "FNO", "KAN"]

# Internal metric name -> source column and publication display name.
# RMSE is kept as single-point RMSE because the earlier ANOVA result used it.
# ACC is patch-level speed ACC from final_eval_metrics.csv.
METRIC_CONFIG = {
    "RMSE": {
        "column": "SinglePoint_RMSE",
        "display": "RMSE",
        "latex": r"RMSE",
        "description": "single-point wind-speed RMSE",
    },
    "R": {
        "column": "SinglePoint_R",
        "display": "R",
        "latex": r"$R$",
        "description": "single-point Pearson correlation",
    },
    "ACC": {
        "column": "Patch_Speed_ACC",
        "display": "ACC",
        "latex": r"ACC",
        "description": "patch-level speed anomaly correlation coefficient",
    },
    "SS_pers": {
        "column": "SS_pers",
        "display": "SS_pers",
        "latex": r"$SS_{\mathrm{pers}}$",
        "description": "persistence skill score",
    },
}

# Nature-style typography.
FONT_SCALE = 1.35
PT_TITLE = 7.0 * FONT_SCALE
PT_LABEL = 6.5 * FONT_SCALE
PT_TICK = 6.0 * FONT_SCALE
LW_SPINE = 0.5

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "FreeSans", "DejaVu Sans", "Arial"],
    "font.size": PT_TICK,
    "axes.labelsize": PT_LABEL,
    "axes.titlesize": PT_TITLE,
    "xtick.labelsize": PT_TICK,
    "ytick.labelsize": PT_TICK,
    "axes.linewidth": LW_SPINE,
    "xtick.major.width": LW_SPINE,
    "ytick.major.width": LW_SPINE,
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


# ════════════════════════════════════════════════════════════════
# 1. Data collection
# ════════════════════════════════════════════════════════════════
def collect_metrics() -> pd.DataFrame:
    """
    Collect final station-model aggregate metrics.

    Returns
    -------
    pd.DataFrame
        One row per station-model pair. Expected complete size: 12 × 8 = 96.
    """
    records = []

    for city in CITIES:
        for model in MODELS_KEY:
            p = BASE / f"analysis_results_{city}" / model / "final_eval_metrics.csv"
            if not p.exists():
                print(f"  ⚠ Missing metrics file: {p}")
                continue

            df = pd.read_csv(p)
            rec = {
                "City": city.capitalize(),
                "Model": MODEL_DISPLAY[model],
                "City_key": city,
                "Model_key": model,
            }

            for metric_name, cfg in METRIC_CONFIG.items():
                col = cfg["column"]
                if metric_name == "SS_pers":
                    continue
                rec[metric_name] = float(df[col].iloc[0]) if col in df.columns else np.nan

            records.append(rec)

    df_metrics = pd.DataFrame(records)

    # Merge persistence skill score.
    if SS_CSV.exists():
        ss = pd.read_csv(SS_CSV)
        ss_agg = (
            ss[ss["Step"].astype(str) == "all"][["City", "Model", "ss_persist"]]
            .copy()
            .rename(columns={"ss_persist": "SS_pers"})
        )
        df_metrics = df_metrics.merge(ss_agg, on=["City", "Model"], how="left")
    else:
        print(f"  ⚠ {SS_CSV} not found. SS_pers will be NaN.")
        df_metrics["SS_pers"] = np.nan

    # Persist merged input table for reproducibility.
    df_metrics.to_csv(OUTPUT_DIR / "a11_input_metrics.csv", index=False)

    return df_metrics


# ════════════════════════════════════════════════════════════════
# 2. Two-way fixed-effect variance decomposition
# ════════════════════════════════════════════════════════════════
def two_way_variance_decomposition(matrix: np.ndarray) -> dict:
    """
    Two-way fixed-effect variance decomposition for a balanced station × model matrix.

    Model:
        y_ij = grand_mean + station_i + model_j + interaction_ij

    Because each station-model pair contributes one aggregate value, this is a
    saturated descriptive decomposition:
        SS_total = SS_station + SS_model + SS_interaction

    No residual term and no p-value are reported.

    Parameters
    ----------
    matrix : np.ndarray
        Shape [n_stations, n_models].

    Returns
    -------
    dict
        Sum of squares, degrees of freedom, and ordinary eta-squared values.
    """
    matrix = np.asarray(matrix, dtype=np.float64)

    if np.isnan(matrix).any():
        n_missing = int(np.isnan(matrix).sum())
        raise ValueError(
            f"Balanced ANOVA matrix contains {n_missing} NaN values. "
            "Complete the missing station-model metrics or explicitly remove "
            "the corresponding station/model before decomposition."
        )

    a, b = matrix.shape  # a = stations, b = models

    grand_mean = matrix.mean()
    row_means = matrix.mean(axis=1)
    col_means = matrix.mean(axis=0)

    ss_total = np.sum((matrix - grand_mean) ** 2)
    ss_station = b * np.sum((row_means - grand_mean) ** 2)
    ss_model = a * np.sum((col_means - grand_mean) ** 2)
    ss_inter = ss_total - ss_station - ss_model

    # Numerical roundoff can produce tiny negative values around -1e-15.
    if ss_inter < 0 and abs(ss_inter) < 1e-10:
        ss_inter = 0.0

    df_station = a - 1
    df_model = b - 1
    df_inter = df_station * df_model

    eta2_station = ss_station / ss_total if ss_total > 0 else np.nan
    eta2_model = ss_model / ss_total if ss_total > 0 else np.nan
    eta2_inter = ss_inter / ss_total if ss_total > 0 else np.nan

    return {
        "eta2_station": eta2_station,
        "eta2_model": eta2_model,
        "eta2_inter": eta2_inter,
        "ss_station": ss_station,
        "ss_model": ss_model,
        "ss_inter": ss_inter,
        "ss_total": ss_total,
        "df_station": df_station,
        "df_model": df_model,
        "df_inter": df_inter,
        "n_station": a,
        "n_model": b,
    }


def build_matrix(df: pd.DataFrame, metric: str) -> tuple[np.ndarray, list[str], list[str]]:
    """Build a balanced [stations, models] matrix for one metric."""
    piv = df.pivot_table(values=metric, index="City", columns="Model", aggfunc="mean")
    ac = [c for c in ORDERED_CITIES if c in piv.index]
    am = [m for m in ORDERED_MODELS if m in piv.columns]

    matrix = piv.loc[ac, am].values
    return matrix, ac, am


# ════════════════════════════════════════════════════════════════
# 3. Reporting
# ════════════════════════════════════════════════════════════════
SOURCE_ROWS = [
    ("Station / lower-boundary setting", "eta2_station", "df_station", "ss_station"),
    ("Model architecture", "eta2_model", "df_model", "ss_model"),
    (r"Station $\times$ Model", "eta2_inter", "df_inter", "ss_inter"),
]


def build_latex_table(results: dict) -> str:
    """Build publication-ready LaTeX table."""
    metric_keys = list(results.keys())
    metric_cols = " & ".join(METRIC_CONFIG[m]["latex"] for m in metric_keys)

    lines = [
        r"\begin{table}[htb]",
        r"\centering",
        r"\caption{",
        r"\textbf{Two-way fixed-effect variance decomposition of model performance.}",
        r"Values denote ordinary $\eta^2$, defined as the proportion of total variance",
        r"in the 96 station--model aggregate scores explained by each factor.",
        r"Because each station--model pair contributes one aggregate value, the",
        r"decomposition is descriptive and no residual significance test is reported.",
        r"}",
        r"\label{tab:anova}",
        r"\begin{tabular}{l" + "c" * (len(metric_keys) + 1) + r"}",
        r"\toprule",
        rf"\textbf{{Source}} & \textbf{{df}} & {metric_cols} \\",
        r"\midrule",
    ]

    for source_tex, eta_key, df_key, _ in SOURCE_ROWS:
        df_val = list(results.values())[0][df_key]
        vals = " & ".join(f"{results[m][eta_key] * 100:.1f}\\%" for m in metric_keys)
        lines.append(f"{source_tex} & {df_val} & {vals} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def print_results(results: dict):
    """Print summary table and LaTeX version."""
    print("\n" + "=" * 78)
    print("  Two-way fixed-effect variance decomposition")
    print("  Ordinary η²: proportion of total variance explained")
    print("=" * 78)

    metric_keys = list(results.keys())
    header = f'{"Source":<38} {"df":>4}  '
    header += "  ".join(f'{METRIC_CONFIG[m]["display"]:>10}' for m in metric_keys)
    print(header)
    print("-" * len(header))

    for source, eta_key, df_key, _ in SOURCE_ROWS:
        df_val = list(results.values())[0][df_key]
        line = f"{source:<38} {df_val:>4}  "
        line += "  ".join(f"{results[m][eta_key]:>10.1%}" for m in metric_keys)
        print(line)

    print("\nInterpretation note:")
    print("  This is a saturated descriptive decomposition over aggregate station-model scores.")
    print("  No residual df or p-value is reported. The interaction term captures station-specific")
    print("  architecture dependence, together with any aggregate noise not separated by replication.")

    latex = build_latex_table(results)
    (OUTPUT_DIR / "anova_latex_table.txt").write_text(latex, encoding="utf-8")

    print("\n--- LaTeX-ready ---")
    print(latex)


def plot_stacked_bar(results: dict, out_stem: str = "anova_variance_decomposition"):
    """Stacked horizontal bar plot for ordinary eta-squared components."""
    metrics = list(results.keys())
    metric_labels = [METRIC_CONFIG[m]["display"] for m in metrics]
    n = len(metrics)

    eta_station = np.array([results[m]["eta2_station"] for m in metrics])
    eta_model = np.array([results[m]["eta2_model"] for m in metrics])
    eta_inter = np.array([results[m]["eta2_inter"] for m in metrics])

    fig_h = 0.65 + n * 0.70 + 0.58
    fig, ax = plt.subplots(figsize=(7.09, fig_h))

    y = np.arange(n)
    bar_h = 0.60

    c_station = "#1B3A5C"
    c_model = "#C4943B"
    c_inter = "#7A8B8B"

    ax.barh(
        y, eta_station, bar_h,
        label="Station / lower-boundary setting",
        color=c_station, edgecolor="#333333", linewidth=0.3,
    )
    ax.barh(
        y, eta_model, bar_h, left=eta_station,
        label="Model architecture",
        color=c_model, edgecolor="#333333", linewidth=0.3,
    )
    left3 = eta_station + eta_model
    ax.barh(
        y, eta_inter, bar_h, left=left3,
        label=r"Station $\times$ Model",
        color=c_inter, edgecolor="#333333", linewidth=0.3,
    )

    # Percentage labels inside bars.
    segments = [
        (eta_station, np.zeros(n), c_station),
        (eta_model, eta_station, c_model),
        (eta_inter, left3, c_inter),
    ]

    for widths, lefts, color in segments:
        rgb = mpl.colors.to_rgb(color)
        lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
        fg_default = "white" if lum < 0.45 else "#1a1a1a"

        for i in range(n):
            w = widths[i]
            if w < 0.07:
                continue
            cx = lefts[i] + w / 2
            ax.text(
                cx, i, f"{w:.1%}",
                ha="center", va="center",
                fontsize=PT_LABEL, color=fg_default, fontweight="bold",
            )

    ax.set_yticks(y)
    ax.set_yticklabels(metric_labels, fontsize=PT_TITLE, style="italic")

    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax.tick_params(axis="x", length=3, width=0.6, pad=3, labelsize=PT_TICK)
    ax.tick_params(axis="y", length=0)

    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.8)
        sp.set_color("#1a1a1a")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, -0.16),
        ncol=3, fontsize=PT_LABEL,
        frameon=False,
        handlelength=1.2, handleheight=0.8,
        handletextpad=0.4, columnspacing=1.4,
    )

    ax.set_xlabel("Proportion of total variance explained", fontsize=PT_LABEL, labelpad=4)

    for suf in (".png", ".pdf"):
        fig.savefig(
            OUTPUT_DIR / f"{out_stem}{suf}",
            dpi=300, bbox_inches="tight", facecolor="white",
        )
    plt.close()

    print(f"  ✓ {out_stem}: PNG + PDF -> {OUTPUT_DIR}")


def write_results_csv(results: dict):
    """Write tidy decomposition table."""
    rows = []
    design_note = (
        "Two-way fixed-effect descriptive decomposition. "
        "Each station-model pair contributes one aggregate score; "
        "no residual significance test is reported."
    )

    for metric, res in results.items():
        for source, eta_key, df_key, ss_key in SOURCE_ROWS:
            rows.append({
                "Metric": metric,
                "Metric_description": METRIC_CONFIG[metric]["description"],
                "Source": source.replace(r"$\times$", "×"),
                "df": int(res[df_key]),
                "SS": float(res[ss_key]),
                "SS_total": float(res["ss_total"]),
                "eta2": float(res[eta_key]),
                "eta2_percent": float(res[eta_key] * 100.0),
                "Design_note": design_note,
            })

    out = pd.DataFrame(rows)
    out.to_csv("a11_anova.csv", index=False)
    out.to_csv(OUTPUT_DIR / "a11_anova.csv", index=False)


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════
def main():
    print("=" * 64)
    print("  A11 — Two-Way Fixed-Effect Variance Decomposition")
    print("=" * 64)

    df = collect_metrics()
    print(f"  ✓ Collected {len(df)} station-model records")
    print(f"  ✓ Merged input metrics written to {OUTPUT_DIR / 'a11_input_metrics.csv'}")

    expected = len(CITIES) * len(MODELS_KEY)
    if len(df) != expected:
        print(f"  ⚠ Expected {expected} records, got {len(df)}. Check missing final_eval_metrics.csv files.")

    results = {}
    for metric, cfg in METRIC_CONFIG.items():
        if metric not in df.columns:
            print(f"  ⚠ Metric {metric} missing from input table, skipped.")
            continue
        if df[metric].isna().all():
            print(f"  ⚠ Metric {metric} is all NaN, skipped.")
            continue

        # If only a few NaNs occur, fail explicitly. Silent nanmean would distort eta².
        n_nan = int(df[metric].isna().sum())
        if n_nan > 0:
            print(f"  ⚠ Metric {metric} contains {n_nan} NaN values and will be skipped.")
            continue

        matrix, cities, models = build_matrix(df, metric)
        res = two_way_variance_decomposition(matrix)
        results[metric] = res

        print(
            f"\n  [{metric}] "
            f"η²_station={res['eta2_station']:.1%}  "
            f"η²_model={res['eta2_model']:.1%}  "
            f"η²_interaction={res['eta2_inter']:.1%}"
        )

    if not results:
        print("  No valid decomposition results.")
        return

    print_results(results)
    plot_stacked_bar(results)
    write_results_csv(results)

    print("\n  ✓ Wrote a11_anova.csv")
    print(f"  ✓ Output directory: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
