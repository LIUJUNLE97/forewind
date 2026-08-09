#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(os.environ.get("WIND_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
ROOT = PROJECT_ROOT / "fig/a14_grid_skill"
OUT = ROOT / "journal_figures_final"
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# Global journal style
# ============================================================
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans", "Arial"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "figure.dpi": 150,
    "savefig.dpi": 500,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.unicode_minus": False,
})

# ============================================================
# Color map: blue-white-orange, no red-green
# Blue   = below persistence / weak skill
# White  = persistence parity
# Orange = positive skill
# ============================================================
SKILL_CMAP = LinearSegmentedColormap.from_list(
    "skill_blue_white_orange",
    ["#2B6CB0", "#F7F7F7", "#C77700"],
    N=256,
)

SKILL_NORM = TwoSlopeNorm(vmin=-0.20, vcenter=0.0, vmax=0.85)

# Linear map scale for Figure B: equal numeric increments occupy equal lengths.
# White is placed at the normalized location of zero instead of forcing zero
# to the middle of the colorbar.
MAP_VMIN, MAP_VMAX = -0.20, 0.85
MAP_ZERO = (0.0 - MAP_VMIN) / (MAP_VMAX - MAP_VMIN)
MAP_CMAP = LinearSegmentedColormap.from_list(
    "skill_blue_zero_white_orange",
    [(0.0, "#2B6CB0"), (MAP_ZERO, "#F7F7F7"), (1.0, "#C77700")],
    N=256,
)
MAP_NORM = Normalize(vmin=MAP_VMIN, vmax=MAP_VMAX)

# ============================================================
# Spatial map orientation
# ============================================================
# Raw station-centered maps appear reversed in both latitude and longitude.
# A 180-degree rotation corrects both axes simultaneously.
# Do NOT add fliplr afterwards, otherwise east-west direction becomes wrong again.
ROTATE_SPATIAL_MAP_180 = True

# ============================================================
# Orders and labels
# ============================================================
CITY_ORDER = [
    "Shanghai", "Qingdao", "Fuzhou", "Guangzhou", "Haikou",
    "Wuhan", "Chengdu",
    "Hohhot", "Urumqi", "Golmud", "Kunming", "Lhasa",
]

MODEL_ORDER = [
    "cnn", "unet",
    "convlstm", "convgru", "trajgru", "predrnn",
    "fno", "kan",
]

MODEL_LABEL = {
    "cnn": "CNN",
    "unet": "U-Net",
    "convlstm": "ConvLSTM",
    "convgru": "ConvGRU",
    "trajgru": "TrajGRU",
    "predrnn": "PredRNN",
    "fno": "FNO",
    "kan": "KAN",
}

MODEL_FAMILY = {
    "cnn": "Local conv.",
    "unet": "Local conv.",
    "convlstm": "Recurrent",
    "convgru": "Recurrent",
    "trajgru": "Recurrent",
    "predrnn": "Recurrent",
    "fno": "Spectral",
    "kan": "KAN",
}

FAMILY_MARKER = {
    "Local conv.": "o",
    "Recurrent": "s",
    "Spectral": "^",
    "KAN": "X",
}

FAMILY_COLOR = {
    "Local conv.": "#4C78A8",
    "Recurrent": "#F28E2B",
    "Spectral": "#59A14F",
    "KAN": "#E15759",
}

# Main A14 mechanism-map selection
MAP_CITIES = ["Shanghai", "Chengdu", "Golmud", "Kunming"]
MAP_MODELS = ["unet", "trajgru", "fno", "kan"]


# ============================================================
# Helpers
# ============================================================
def savefig(fig, name: str):
    fig.savefig(OUT / f"{name}.png")
    # Text/lines remain vector; dpi only controls embedded raster map panels.
    fig.savefig(OUT / f"{name}.pdf", dpi=200)
    plt.close(fig)
    print(f"[OK] saved {OUT / f'{name}.png'}")
    print(f"[OK] saved {OUT / f'{name}.pdf'}")


def city_to_dir(city: str) -> str:
    return str(city).lower()


def load_skill_map(city: str, model: str) -> np.ndarray:
    f = ROOT / f"analysis_results_{city_to_dir(city)}" / model / "ss_persist_map.npy"
    if not f.exists():
        raise FileNotFoundError(f"Missing skill map: {f}")

    arr = np.load(f)

    # Correct latitude orientation only.
    # Raw patch rows follow north-to-south latitude ordering,
    # while imshow(origin="lower") expects south-to-north.
    # Longitude direction is already correct, so do NOT rotate 180°.
    arr = np.flipud(arr)

    return arr


def merge_case_summaries(root: Path) -> pd.DataFrame:
    """
    Merge all per-case A14 summaries.

    a14_all_grid_skill_summary.csv is overwritten by each script call,
    so this function scans all case folders instead.
    """
    files = sorted(root.glob("analysis_results_*/*/a14_grid_skill_summary.csv"))

    if len(files) == 0:
        raise FileNotFoundError(f"No a14_grid_skill_summary.csv files found under {root}")

    dfs = []
    for f in files:
        try:
            d = pd.read_csv(f)
            d["summary_file"] = str(f)
            dfs.append(d)
        except Exception as e:
            print(f"[WARN] skip {f}: {e}")

    df = pd.concat(dfs, ignore_index=True)

    df["City"] = df["City"].astype(str).str.strip()
    df["Model"] = df["Model"].astype(str).str.lower().str.strip()

    df = df[df["Model"].isin(MODEL_ORDER)].copy()

    # Deduplicate in case a case was rerun.
    df = df.drop_duplicates(subset=["City", "Model"], keep="last")

    df["City"] = pd.Categorical(df["City"], categories=CITY_ORDER, ordered=True)
    df["Model"] = pd.Categorical(df["Model"], categories=MODEL_ORDER, ordered=True)
    df = df.sort_values(["City", "Model"]).reset_index(drop=True)

    out_csv = root / "a14_all_grid_skill_summary_merged.csv"
    df.to_csv(out_csv, index=False)

    print(f"[OK] merged {len(df)} case summaries -> {out_csv}")

    print("\nCompleteness by model:")
    print(df.groupby("Model", observed=False)["City"].nunique())

    print("\nCompleteness by city:")
    print(df.groupby("City", observed=False)["Model"].nunique())

    return df


def write_tables(df: pd.DataFrame):
    pivot = df.pivot(index="City", columns="Model", values="Mean_SS_grid")
    pivot = pivot.loc[CITY_ORDER, MODEL_ORDER]

    table = pivot.copy()
    table.columns = [f"{MODEL_LABEL[str(c)]} mean SS" for c in table.columns]
    table["City mean"] = pivot.mean(axis=1)
    table["Best model"] = pivot.idxmax(axis=1).map(MODEL_LABEL)
    table["Worst model"] = pivot.idxmin(axis=1).map(MODEL_LABEL)

    mean_row = pd.DataFrame(
        [list(pivot.mean(axis=0).values) + [pivot.values.mean(), "-", "-"]],
        columns=table.columns,
        index=["Mean"],
    )

    table_final = pd.concat([table, mean_row]).round(4)
    out1 = OUT / "journal_table_a14_grid_skill_all_models.csv"
    table_final.to_csv(out1)
    print(f"[OK] saved {out1}")

    long_cols = [
        "City", "Model", "Lat", "Lon",
        "N_times_T_count",
        "Patch_RMSE_from_map_weighted",
        "Old_Patch_Speed_RMSE_csv",
        "Center_RMSE_from_map",
        "Old_SinglePoint_RMSE_csv",
        "Mean_SS_grid",
        "Median_SS_grid",
        "Center_SS_grid",
        "Frac_grid_SS_negative",
        "Min_SS_grid",
        "Max_SS_grid",
    ]
    long_cols = [c for c in long_cols if c in df.columns]

    out2 = OUT / "journal_table_a14_long_diagnostics.csv"
    df[long_cols].to_csv(out2, index=False)
    print(f"[OK] saved {out2}")


# ============================================================
# Figure A: all-model heatmap
# ============================================================
def plot_mean_ss_heatmap(df: pd.DataFrame):
    pivot = df.pivot(index="City", columns="Model", values="Mean_SS_grid")
    pivot = pivot.loc[CITY_ORDER, MODEL_ORDER]
    data = pivot.values.astype(float)

    fig, ax = plt.subplots(figsize=(7.3, 4.4))

    im = ax.imshow(
        data,
        cmap=SKILL_CMAP,
        norm=SKILL_NORM,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(len(MODEL_ORDER)))
    ax.set_xticklabels([MODEL_LABEL[m] for m in MODEL_ORDER], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(CITY_ORDER)))
    ax.set_yticklabels(CITY_ORDER)

    ax.set_xlabel("Model architecture")
    ax.set_ylabel("Station-centered patch")
    ax.set_title(r"Patch-mean grid-level persistence skill")

    # Regime separators
    ax.axhline(4.5, color="black", lw=0.6)
    ax.axhline(6.5, color="black", lw=0.6)

    # Regime labels
    ax.text(len(MODEL_ORDER) + 0.10, 2.0, "Coastal", va="center", fontsize=7)
    ax.text(len(MODEL_ORDER) + 0.10, 5.5, "Inland", va="center", fontsize=7)
    ax.text(len(MODEL_ORDER) + 0.10, 9.0, "Plateau", va="center", fontsize=7)

    # Annotate values
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if not np.isfinite(val):
                continue
            text_color = "white" if (val < -0.05 or val > 0.70) else "black"
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                fontsize=5.8,
                color=text_color,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label(r"$SS_{\mathrm{pers}}$")

    savefig(fig, "fig_a14a_mean_ss_heatmap_all_models")


# ============================================================
# Figure B: selected spatial skill maps, horizontal-axis flip
# ============================================================
def plot_selected_skill_maps():
    nrows = len(MAP_CITIES)
    ncols = len(MAP_MODELS)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(11.0, 9.7),
        gridspec_kw={"wspace": 0.018, "hspace": 0.025},
    )

    for i, city in enumerate(MAP_CITIES):
        for j, model in enumerate(MAP_MODELS):
            ax = axes[i, j]
            arr = load_skill_map(city, model)

            im = ax.imshow(
                arr,
                origin="lower",
                cmap=MAP_CMAP,
                norm=MAP_NORM,
                interpolation="nearest",
            )

            h, w = arr.shape
            ax.plot(w // 2, h // 2, marker="+", color="black", ms=11, mew=2.0)

            ax.set_xticks([])
            ax.set_yticks([])

            if i == 0:
                ax.set_title(MODEL_LABEL[model], fontsize=20,
                             fontweight="normal", pad=7)

            if j == 0:
                ax.set_ylabel(city, rotation=90, labelpad=13,
                              fontsize=19, fontweight="normal")

            mean_ss = float(np.nanmean(arr))
            frac_neg = float(np.nanmean(arr < 0.0))

            ax.text(
                0.03, 0.04,
                f"mean={mean_ss:.2f}\nneg={100 * frac_neg:.1f}%",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=12.5,
                linespacing=1.05,
                bbox=dict(
                    facecolor="white",
                    edgecolor="0.78",
                    linewidth=0.6,
                    alpha=0.86,
                    pad=2.2,
                ),
            )

    fig.subplots_adjust(left=0.075, right=0.885, bottom=0.04, top=0.955)

    # Match the colorbar exactly to the top and bottom of the 4x4 panel grid.
    grid_bottom = axes[-1, 0].get_position().y0
    grid_top = axes[0, 0].get_position().y1
    cax = fig.add_axes([0.905, grid_bottom, 0.024, grid_top - grid_bottom])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_ticks(np.arange(-0.2, 0.81, 0.2))
    cbar.set_label(r"Grid-level $SS_{\mathrm{pers}}$", fontsize=19, labelpad=13)
    cbar.ax.tick_params(labelsize=18, width=1.2, length=6)
    cbar.outline.set_linewidth(1.0)

    savefig(fig, "fig_a14b_selected_skill_maps")


# ============================================================
# Figure C: center-point vs patch-average skill
# ============================================================
def plot_center_vs_patch(df: pd.DataFrame):
    d = df.copy()
    d["Model_str"] = d["Model"].astype(str)
    d["City_str"] = d["City"].astype(str)
    d["Family"] = d["Model_str"].map(MODEL_FAMILY)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    family_order = ["Local conv.", "Recurrent", "Spectral", "KAN"]

    for fam in family_order:
        sub = d[d["Family"] == fam]
        if len(sub) == 0:
            continue

        ax.scatter(
            sub["Mean_SS_grid"],
            sub["Center_SS_grid"],
            s=45,
            marker=FAMILY_MARKER[fam],
            color=FAMILY_COLOR[fam],
            label=fam,
            alpha=0.78,
            edgecolor="black",
            linewidth=0.35,
            zorder=3,
        )

    min_v = min(d["Mean_SS_grid"].min(), d["Center_SS_grid"].min()) - 0.04
    max_v = max(d["Mean_SS_grid"].max(), d["Center_SS_grid"].max()) + 0.04

    ax.plot(
        [min_v, max_v],
        [min_v, max_v],
        linestyle="--",
        color="0.55",
        lw=1.1,
        zorder=1,
    )

    corr = d[["Mean_SS_grid", "Center_SS_grid"]].corr().iloc[0, 1]
    ax.text(
        0.04,
        0.96,
        f"Pearson r = {corr:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
    )

    # Manually selected labels to avoid heavy overlap.
    annotation_cases = [
        ("Kunming", "kan", "Kunming\nKAN", (12, 14), "left"),
        ("Kunming", "cnn", "Kunming\nCNN", (10, 10), "left"),
        ("Chengdu", "fno", "Chengdu\nFNO", (-58, -16), "right"),
        ("Chengdu", "trajgru", "Chengdu\nTrajGRU", (12, -28), "left"),
        ("Lhasa", "predrnn", "Lhasa\nPredRNN", (12, 12), "left"),
    ]

    for city, model, label, offset, ha in annotation_cases:
        r = d[(d["City_str"] == city) & (d["Model_str"] == model)]
        if len(r) == 0:
            continue

        r = r.iloc[0]
        ax.annotate(
            label,
            xy=(r["Mean_SS_grid"], r["Center_SS_grid"]),
            xytext=offset,
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=7.0,
            arrowprops=dict(
                arrowstyle="-",
                lw=0.55,
                color="0.35",
                alpha=0.70,
                shrinkA=2,
                shrinkB=2,
            ),
            zorder=5,
        )

    ax.set_xlim(min_v, max_v)
    ax.set_ylim(min_v, max_v)

    ax.set_xlabel(r"Patch-mean grid-level $SS_{\mathrm{pers}}$")
    ax.set_ylabel(r"Center-point $SS_{\mathrm{pers}}$")
    ax.set_title("Center-point versus patch-averaged skill")

    ax.grid(alpha=0.20, linewidth=0.6)

    ax.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.50),
        borderaxespad=0.0,
        handletextpad=0.5,
    )

    savefig(fig, "fig_a14c_center_vs_patch_all_models_clean")


# ============================================================
# Figure D: geographic all-model mean skill
# ============================================================
def plot_geographic_all_model_mean(df: pd.DataFrame):
    city_df = (
        df.groupby("City", observed=False)
        .agg(
            Lat=("Lat", "first"),
            Lon=("Lon", "first"),
            Mean_SS=("Mean_SS_grid", "mean"),
            Median_SS=("Median_SS_grid", "mean"),
            NegFrac=("Frac_grid_SS_negative", "mean"),
            RMSE=("Patch_RMSE_from_map_weighted", "mean"),
        )
        .reset_index()
    )

    city_df["City"] = pd.Categorical(city_df["City"], categories=CITY_ORDER, ordered=True)
    city_df = city_df.sort_values("City")

    fig, ax = plt.subplots(figsize=(5.9, 4.6))

    rmse = city_df["RMSE"].values
    size = 55 + 220 * (rmse - np.nanmin(rmse)) / (
        np.nanmax(rmse) - np.nanmin(rmse) + 1e-12
    )

    sc = ax.scatter(
        city_df["Lon"],
        city_df["Lat"],
        c=city_df["Mean_SS"],
        s=size,
        cmap=SKILL_CMAP,
        norm=SKILL_NORM,
        edgecolor="black",
        linewidth=0.45,
        alpha=0.92,
        zorder=3,
    )

    label_offsets = {
        "Shanghai": (-1.0, 0.45, "right"),
        "Qingdao": (0.45, 0.35, "left"),
        "Fuzhou": (0.45, 0.25, "left"),
        "Guangzhou": (0.45, 0.25, "left"),
        "Haikou": (0.45, 0.25, "left"),
        "Wuhan": (0.45, 0.20, "left"),
        "Chengdu": (0.45, 0.25, "left"),
        "Hohhot": (0.45, 0.25, "left"),
        "Urumqi": (0.45, 0.30, "left"),
        "Golmud": (0.45, 0.25, "left"),
        "Kunming": (0.45, 0.25, "left"),
        "Lhasa": (0.45, 0.25, "left"),
    }

    for _, r in city_df.iterrows():
        dx, dy, ha = label_offsets.get(str(r["City"]), (0.45, 0.25, "left"))
        ax.text(
            r["Lon"] + dx,
            r["Lat"] + dy,
            str(r["City"]),
            fontsize=6.8,
            ha=ha,
            va="center",
            zorder=4,
        )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(r"Geographic distribution of patch-mean skill")

    ax.set_xlim(85.5, 124.5)
    ax.set_ylim(19.0, 45.5)

    ax.set_aspect(1.0 / np.cos(np.deg2rad(34.0)), adjustable="box")
    ax.grid(alpha=0.20, linewidth=0.6)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.025)
    cbar.set_label(r"All-model mean $SS_{\mathrm{pers}}$")

    savefig(fig, "fig_a14d_geographic_all_model_mean_skill")


# ============================================================
# Figure E: selected-model city comparison
# ============================================================
def plot_selected_model_city_bar(df: pd.DataFrame):
    selected_models = ["unet", "trajgru", "fno", "kan"]
    d = df[df["Model"].astype(str).isin(selected_models)].copy()

    pivot = d.pivot(index="City", columns="Model", values="Mean_SS_grid")
    pivot = pivot.loc[CITY_ORDER, selected_models]

    fig, ax = plt.subplots(figsize=(7.4, 3.7))

    x = np.arange(len(CITY_ORDER))
    width = 0.18

    for k, model in enumerate(selected_models):
        ax.bar(
            x + (k - 1.5) * width,
            pivot[model].values,
            width,
            label=MODEL_LABEL[model],
        )

    ax.axhline(0.0, color="black", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(CITY_ORDER, rotation=35, ha="right")
    ax.set_ylabel(r"Patch-mean $SS_{\mathrm{pers}}$")
    ax.set_xlabel("Station-centered patch")
    ax.set_title("Selected model comparison across stations")
    ax.grid(axis="y", alpha=0.20, linewidth=0.6)

    ax.legend(
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
    )

    savefig(fig, "fig_a14e_selected_model_city_bar")


# ============================================================
# Main
# ============================================================
def main():
    df = merge_case_summaries(ROOT)

    expected = len(CITY_ORDER) * len(MODEL_ORDER)
    if len(df) != expected:
        print(f"[WARN] Expected {expected} rows, got {len(df)}. Figures will use available cases.")

    write_tables(df)

    plot_mean_ss_heatmap(df)
    plot_selected_skill_maps()
    plot_center_vs_patch(df)
    plot_geographic_all_model_mean(df)
    plot_selected_model_city_bar(df)

    print(f"\nSaved all final A14 journal figures to: {OUT}")


if __name__ == "__main__":
    main()
