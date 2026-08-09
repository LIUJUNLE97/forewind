"""Separate temporal-consistency and spatial-variability plots for SOM regions.

Each panel represents one SOM cluster.  Thin lines are monthly wind-speed
series at sampled grid cells; the solid line and shaded band are the cluster
spatial mean and +/- one spatial standard deviation, respectively.
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

_ARIAL_FONT = os.environ.get("WIND_PROJECT_ARIAL_FONT", "")
if _ARIAL_FONT and os.path.isfile(_ARIAL_FONT):
    font_manager.fontManager.addfont(_ARIAL_FONT)
plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Arial"
plt.rcParams["mathtext.it"] = "Arial:italic"
plt.rcParams["mathtext.bf"] = "Arial:bold"


PROJECT_DIR = os.environ.get(
    "WIND_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
DEFAULT_NC_DIR = os.path.join(PROJECT_DIR, "processed")
DEFAULT_CLUSTER_CSV = None
DEFAULT_SOM_MAP = os.path.join(PROJECT_DIR, "som_map.npy")
DEFAULT_LAT_LON = os.path.join(PROJECT_DIR, "lat_lon_grid.npz")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_DIR, "som_post_analysis", "individual_clusters")

CLUSTER_COLORS = [
    "#6F8FAF", "#C89A78", "#7FA17B", "#B98282", "#9A87A8",
    "#72A3A0", "#B9A56A", "#987F70", "#929292",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nc-dir", default=DEFAULT_NC_DIR)
    parser.add_argument("--cluster-csv", default=DEFAULT_CLUSTER_CSV)
    parser.add_argument("--som-map", default=DEFAULT_SOM_MAP)
    parser.add_argument("--lat-lon", default=DEFAULT_LAT_LON)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Recompute cluster plot data instead of using cached NPZ files")
    parser.add_argument("--cluster", type=int, choices=range(1, 10),
                        help="Only draw this cluster (1--9); default draws all")
    parser.add_argument("--samples", type=int, default=30,
                        help="Number of grid-cell series drawn per cluster")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--figure-size", nargs=2, type=float, default=[7.2, 4.6],
                        metavar=("WIDTH", "HEIGHT"),
                        help="Figure size in inches for a two-column LaTeX row")
    parser.add_argument("--title-size", type=float, default=22)
    parser.add_argument("--axis-label-size", type=float, default=17)
    parser.add_argument("--tick-label-size", type=float, default=13,
                        help="Y-axis tick-label size")
    parser.add_argument("--x-tick-label-size", type=float, default=15,
                        help="X-axis year-label size")
    parser.add_argument("--annotation-size", type=float, default=9)
    parser.add_argument("--legend-size", type=float, default=9)
    parser.add_argument("--output-format", choices=("pdf", "png", "both"), default="pdf",
                        help="Output format for individual cluster plots")
    return parser.parse_args()


def load_cluster_table(cluster_csv, som_map: str, lat_lon: str):
    """Load a cluster CSV, or reconstruct it from the local SOM grid."""
    if cluster_csv:
        clusters = pd.read_csv(cluster_csv)
        required = {"latitude", "longitude", "cluster_id"}
        missing = required.difference(clusters.columns)
        if missing:
            raise ValueError(f"Cluster CSV lacks columns: {sorted(missing)}")
    else:
        labels = np.load(som_map)
        coords = np.load(lat_lon)
        lons, lats = coords["lons"], coords["lats"]
        if lons.ndim == 1 and lats.ndim == 1:
            lon_grid, lat_grid = np.meshgrid(lons, lats)
        else:
            lon_grid, lat_grid = lons, lats
        if labels.shape != lat_grid.shape or labels.shape != lon_grid.shape:
            raise ValueError(
                f"Shape mismatch: SOM {labels.shape}, latitude {lat_grid.shape}, "
                f"longitude {lon_grid.shape}"
            )
        clusters = pd.DataFrame({
            "latitude": lat_grid.ravel(),
            "longitude": lon_grid.ravel(),
            "cluster_id": labels.ravel(),
        })

    labels = clusters["cluster_id"].astype(int)
    if labels.min() == 1 and labels.max() == 9:
        labels = labels - 1
    return clusters.assign(cluster_id=labels)


def load_inputs(nc_dir: str, cluster_csv, som_map: str, lat_lon: str):
    files = sorted(glob.glob(os.path.join(nc_dir, "processed_*.nc")))
    if not files:
        files = sorted(glob.glob(os.path.join(nc_dir, "*.nc")))
    if not files:
        raise FileNotFoundError(f"No NetCDF files found in {nc_dir}")

    clusters = load_cluster_table(cluster_csv, som_map, lat_lon)

    ds = xr.open_mfdataset(files, combine="by_coords", parallel=False)
    if "expver" in ds.dims:
        parts = [ds.sel(expver=value) for value in ds.expver.values]
        ds = parts[0]
        for part in parts[1:]:
            ds = ds.combine_first(part)
    if "wind_speed" not in ds:
        if not {"u10", "v10"}.issubset(ds.data_vars):
            raise KeyError("NetCDF must contain wind_speed or both u10 and v10")
        ds["wind_speed"] = np.hypot(ds["u10"], ds["v10"])
    return ds, clusters


def compute_monthly_wind(ds):
    """Compute the monthly wind-speed field once for all clusters."""
    return ds["wind_speed"].resample(time="MS").mean(skipna=True).compute()


def select_cluster_monthly(monthly_wind, rows):
    """Return wind speed with dimensions (time, points) for one cluster."""
    lats = xr.DataArray(rows["latitude"].to_numpy(), dims="points")
    lons = xr.DataArray(rows["longitude"].to_numpy(), dims="points")
    data = monthly_wind.sel(latitude=lats, longitude=lons, method="nearest")
    extra_dims = [d for d in data.dims if d not in {"time", "points"}]
    if extra_dims:
        data = data.mean(extra_dims, skipna=True)
    data = data.transpose("time", "points")
    return data


def plot_cluster(monthly_wind, clusters, cluster_id: int, output_dir: str,
                 samples: int, seed: int, figure_size=(7.2, 4.6),
                 title_size=22, axis_label_size=17, tick_label_size=13,
                 x_tick_label_size=15, annotation_size=9, legend_size=9,
                 rebuild_cache=False, output_format="pdf"):
    output_dir = os.path.abspath(output_dir)
    cache_dir = os.path.join(output_dir, "cache")
    cache_file = os.path.join(cache_dir, f"cluster_{cluster_id + 1}_plot_data.npz")

    if os.path.exists(cache_file) and not rebuild_cache:
        cached = np.load(cache_file)
        plot_time = pd.to_datetime(cached["time"])
        sample_values = cached["sample_values"]
        mean_values = cached["spatial_mean"]
        std_values = cached["spatial_std"]
        temporal_cv = float(cached["temporal_cv"])
        spatial_cv = float(cached["spatial_cv"])
        valid_point_count = int(cached["valid_point_count"])
        print(f"Loaded cache: {cache_file}")
    else:
        if monthly_wind is None or clusters is None:
            raise RuntimeError(f"Missing source data needed to build {cache_file}")
        rng = np.random.default_rng(seed)
        rows = clusters.loc[clusters.cluster_id == cluster_id]
        if rows.empty:
            raise ValueError(f"Cluster {cluster_id + 1} has no grid cells")

        monthly = select_cluster_monthly(monthly_wind, rows)
        valid_points = np.flatnonzero(monthly.notnull().any("time").values)
        if valid_points.size == 0:
            raise ValueError(f"Cluster {cluster_id + 1} has no valid data")

        chosen = rng.choice(valid_points, min(samples, valid_points.size), replace=False)
        sample_data = monthly.isel(points=chosen)
        spatial_mean = monthly.mean("points", skipna=True)
        spatial_std = monthly.std("points", skipna=True)
        plot_time = pd.to_datetime(monthly.time.values)
        sample_values = sample_data.values
        mean_values = spatial_mean.values
        std_values = spatial_std.values
        temporal_cv = float(spatial_mean.std("time") / spatial_mean.mean("time") * 100)
        spatial_cv = float((spatial_std / spatial_mean).mean("time") * 100)
        valid_point_count = int(valid_points.size)
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(
            cache_file,
            time=monthly.time.values,
            sample_values=sample_values,
            spatial_mean=mean_values,
            spatial_std=std_values,
            temporal_cv=temporal_cv,
            spatial_cv=spatial_cv,
            valid_point_count=valid_point_count,
        )
        print(f"Saved cache: {cache_file}")

    # Use a cluster-specific y range instead of the former fixed 0--16 m/s.
    # Include both the sampled series and the displayed mean +/- 1 SD band.
    band_lower = mean_values - std_values
    band_upper = mean_values + std_values
    visible_values = np.concatenate([
        sample_values[np.isfinite(sample_values)].ravel(),
        band_lower[np.isfinite(band_lower)].ravel(),
        band_upper[np.isfinite(band_upper)].ravel(),
    ])
    data_min = float(visible_values.min())
    data_max = float(visible_values.max())
    padding = max(0.08 * (data_max - data_min), 0.25)
    y_min = max(0.0, data_min - padding)
    y_max = data_max + padding

    color = CLUSTER_COLORS[cluster_id]
    fig, ax = plt.subplots(figsize=figure_size)
    ax.plot(plot_time, sample_values, color=color, lw=0.7,
            alpha=0.18, zorder=1, rasterized=True)
    ax.fill_between(plot_time, band_lower, band_upper,
                    color=color, alpha=0.28, linewidth=0, zorder=2)
    ax.plot(plot_time, mean_values, color=color, lw=3.2, zorder=3)

    ax.text(0.97, 0.95,
            f"Temporal CV: {temporal_cv:.1f}%\nSpatial CV: {spatial_cv:.1f}%\n"
            f"Grid cells: {valid_point_count:,}",
            transform=ax.transAxes, ha="right", va="top", fontsize=annotation_size,
            bbox={"facecolor": "white", "edgecolor": "0.78", "alpha": 0.92,
                  "boxstyle": "round,pad=0.4"})
    ax.set_title(f"Cluster {cluster_id + 1}",
                 fontsize=title_size, fontweight="bold", pad=12)
    ax.set_ylabel("10-m wind speed (m s$^{-1}$)", fontsize=axis_label_size, labelpad=8)
    ax.tick_params(axis="y", which="major", labelsize=tick_label_size,
                   width=1.3, length=6)
    ax.tick_params(axis="x", which="major", labelsize=x_tick_label_size,
                   labelrotation=40,
                   width=1.3, length=6)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.grid(True, linestyle=":", color="0.78", linewidth=0.8)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylim(y_min, y_max)

    handles = [
        Line2D([0], [0], color="0.25", lw=0.6, alpha=0.5,
               label="Sampled grid-cell monthly series"),
        Line2D([0], [0], color="0.25", lw=2.0, label="Cluster spatial mean"),
        Patch(facecolor="0.5", alpha=0.25, label="Spatial mean +/- 1 SD"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=legend_size, frameon=True,
              framealpha=0.92)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.join(output_dir, f"cluster_{cluster_id + 1}_temporal_spatial")
    outputs = []
    if output_format in {"pdf", "both"}:
        output = f"{stem}.pdf"
        fig.savefig(output, dpi=300, bbox_inches="tight")
        outputs.append(output)
    if output_format in {"png", "both"}:
        output = f"{stem}.png"
        fig.savefig(output, dpi=300, bbox_inches="tight")
        outputs.append(output)
    plt.close(fig)
    for output in outputs:
        print(f"Saved: {output}")


def main():
    args = parse_args()
    cluster_ids = [args.cluster - 1] if args.cluster else list(range(9))
    cache_dir = os.path.join(os.path.abspath(args.output_dir), "cache")
    cache_files = [os.path.join(cache_dir, f"cluster_{i + 1}_plot_data.npz")
                   for i in cluster_ids]
    use_cache_only = not args.rebuild_cache and all(os.path.exists(p) for p in cache_files)
    ds = None
    try:
        if use_cache_only:
            monthly_wind, clusters = None, None
            print("All requested cluster caches found; skipping NetCDF computation.")
        else:
            ds, clusters = load_inputs(
                args.nc_dir, args.cluster_csv, args.som_map, args.lat_lon
            )
            print("Computing monthly wind-speed field once for missing/rebuilt caches...")
            monthly_wind = compute_monthly_wind(ds)
        for cluster_id in cluster_ids:
            plot_cluster(monthly_wind, clusters, cluster_id, args.output_dir,
                         args.samples, args.seed + cluster_id,
                         figure_size=tuple(args.figure_size),
                         title_size=args.title_size,
                         axis_label_size=args.axis_label_size,
                         tick_label_size=args.tick_label_size,
                         x_tick_label_size=args.x_tick_label_size,
                         annotation_size=args.annotation_size,
                         legend_size=args.legend_size,
                         rebuild_cache=args.rebuild_cache,
                         output_format=args.output_format)
    finally:
        if ds is not None:
            ds.close()


if __name__ == "__main__":
    main()
