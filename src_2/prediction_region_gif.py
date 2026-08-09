#!/usr/bin/env python3
"""Cache 24-hour regional forecasts and render model-comparison animations."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation, font_manager
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch

from model.model_factory import get_model


PROJECT_DIR = Path(os.environ.get(
    "WIND_PROJECT_ROOT", Path(__file__).resolve().parents[1]
))
LOCATIONS = {
    "shanghai": (31.23, 121.47),
    "chengdu": (30.67, 104.06),
    "golmud": (36.40, 94.90),
    "wuhan": (30.59, 114.30),
}
DISPLAY_NAMES = {
    "truth": "Ground Truth",
    "persistence": "Persistence",
    "unet": "U-Net",
    "trajgru": "TrajGRU",
    "fno": "FNO",
    "kan": "KAN",
}
PANEL_ORDER = ["truth", "persistence", "unet", "trajgru", "fno", "kan"]


# The SOM maps use cnmaps administrative geometries in WGS84/PlateCarree.
# The x86 rendering environment does not include cartopy/cnmaps, so the
# renderer has a small, dependency-free fallback that reads the same cnmaps
# GeoJSON files shipped in this project and converts their GCJ-02 coordinates
# using cnmaps' own conversion formula.
_BOUNDARY_LINES = None


def _gcj02_to_wgs84(lon, lat):
    """Match cnmaps.geo.gcj02_to_wgs84 for one coordinate pair."""
    pi = math.pi
    sma = 6378245.0
    eer = 0.00669342162296594323
    x = lon - 105.0
    y = lat - 35.0
    dlat = (
        -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y
        + 0.2 * math.sqrt(abs(x))
    )
    dlat += (20.0 * math.sin(6.0 * x * pi) + 20.0 * math.sin(2.0 * x * pi)) * 2.0 / 3.0
    dlat += (20.0 * math.sin(y * pi) + 40.0 * math.sin(y / 3.0 * pi)) * 2.0 / 3.0
    dlat += (160.0 * math.sin(y / 12.0 * pi) + 320.0 * math.sin(y * pi / 30.0)) * 2.0 / 3.0
    dlon = (
        300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y
        + 0.1 * math.sqrt(abs(x))
    )
    dlon += (20.0 * math.sin(6.0 * x * pi) + 20.0 * math.sin(2.0 * x * pi)) * 2.0 / 3.0
    dlon += (20.0 * math.sin(x * pi) + 40.0 * math.sin(x / 3.0 * pi)) * 2.0 / 3.0
    dlon += (150.0 * math.sin(x / 12.0 * pi) + 300.0 * math.sin(x * pi / 30.0)) * 2.0 / 3.0
    radlat = lat / 180.0 * pi
    magic = 1.0 - eer * math.sin(radlat) ** 2
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((sma * (1.0 - eer)) / (magic * sqrtmagic) * pi)
    dlon = (dlon * 180.0) / (sma / sqrtmagic * math.cos(radlat) * pi)
    return lon * 2.0 - (lon + dlon), lat * 2.0 - (lat + dlat)


def _geometry_lines(geometry, convert=False):
    """Return polygon rings/line strings as Nx2 arrays for LineCollection."""
    if geometry is None:
        return []
    if hasattr(geometry, "__geo_interface__"):
        geometry = geometry.__geo_interface__
    if isinstance(geometry, dict) and "geometry" in geometry:
        geometry = geometry["geometry"]
    if not isinstance(geometry, dict):
        return []
    kind = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if kind == "Polygon":
        rings = coords
    elif kind == "MultiPolygon":
        rings = [ring for polygon in coords for ring in polygon]
    elif kind == "LineString":
        rings = [coords]
    elif kind == "MultiLineString":
        rings = coords
    else:
        rings = []
    lines = []
    for ring in rings:
        if len(ring) < 2:
            continue
        if convert:
            ring = [_gcj02_to_wgs84(float(lon), float(lat)) for lon, lat, *rest in ring]
        lines.append(np.asarray(ring, dtype=np.float64)[:, :2])
    return lines


def _load_admin_boundaries():
    """Load SOM-style country/province/Taiwan boundary lines once per process."""
    global _BOUNDARY_LINES
    if _BOUNDARY_LINES is not None:
        return _BOUNDARY_LINES

    # Prefer the exact cnmaps API used by drawing_som.py when it is available.
    try:
        from cnmaps import get_adm_maps

        country = get_adm_maps(level="国", record="first")
        provinces = get_adm_maps(level="省")
        country_lines = _geometry_lines(country.get("geometry") if isinstance(country, dict) else country)
        province_lines, taiwan_lines = [], []
        for record in provinces:
            if isinstance(record, dict):
                geometry = record.get("geometry")
                name = record.get("省/直辖市", record.get("name", ""))
            else:
                geometry = getattr(record, "geometry", record)
                name = getattr(record, "省/直辖市", "") or getattr(record, "name", "")
            target = taiwan_lines if any(token in str(name) for token in ("台", "臺", "Taiwan")) else province_lines
            target.extend(_geometry_lines(geometry))
    except Exception as exc:
        # Read the same cnmaps files directly. This keeps rendering usable in
        # the lightweight torch/matplotlib runtime that lacks cartopy/cnmaps.
        data_dir = PROJECT_DIR / "venv/lib/python3.11/site-packages/cnmaps/data/geojson.min/administrative/amap/land"
        country_path = data_dir / "100000.geojson"
        if not country_path.exists():
            print(f"[boundaries] cnmaps data unavailable ({exc}); skipping boundaries.")
            _BOUNDARY_LINES = ([], [], [])
            return _BOUNDARY_LINES

        def read_geometry(path):
            with path.open(encoding="utf-8") as handle:
                return json.load(handle)

        country_lines = _geometry_lines(read_geometry(country_path), convert=True)
        province_lines, taiwan_lines = [], []
        for path in sorted(data_dir.glob("*0000.geojson")):
            if path.name == "100000.geojson":
                continue
            lines = _geometry_lines(read_geometry(path), convert=True)
            (taiwan_lines if path.stem == "710000" else province_lines).extend(lines)

    _BOUNDARY_LINES = (country_lines, province_lines, taiwan_lines)
    print(
        "[boundaries] Loaded SOM-style administrative lines: "
        f"country={len(country_lines)}, province={len(province_lines)}, "
        f"taiwan={len(taiwan_lines)}"
    )
    return _BOUNDARY_LINES


def _add_admin_boundaries(ax, extent):
    """Overlay national/provincial boundaries with the SOM line styles."""
    country_lines, province_lines, taiwan_lines = _load_admin_boundaries()
    xmin, xmax, ymin, ymax = extent

    def visible(lines):
        result = []
        # Keep only line rings whose bounding boxes touch this regional panel.
        # Matplotlib clips them to the axes, while this avoids carrying all of
        # China's distant islands into every panel.
        for line in lines:
            if line[:, 0].max() >= xmin and line[:, 0].min() <= xmax \
                    and line[:, 1].max() >= ymin and line[:, 1].min() <= ymax:
                result.append(line)
        return result

    for lines, color, width, zorder in (
        (country_lines, "#ffffff", 2.0, 5),
        (province_lines, "#ffffff", 1.2, 6),
        (taiwan_lines, "#ffffff", 2.0, 7),
    ):
        selected = visible(lines)
        if selected:
            ax.add_collection(LineCollection(
                selected, colors=color, linewidths=width,
                zorder=zorder, clip_on=True,
            ))


def wind_colormap():
    """Match the ParaView Fast map used by analysis_pred.py."""
    rgb_points = [
        0, 0.0564, 0.0564, 0.4700,
        0.1716, 0.2430, 0.4604, 0.8100,
        0.2985, 0.3568, 0.7450, 0.9544,
        0.4321, 0.6882, 0.9300, 0.9179,
        0.5000, 0.8995, 0.9446, 0.7687,
        0.5882, 0.9571, 0.8338, 0.5089,
        0.7061, 0.9275, 0.6214, 0.3154,
        0.8476, 0.8000, 0.3520, 0.1600,
        1.0000, 0.5900, 0.0767, 0.1195,
    ]
    positions = [rgb_points[i] for i in range(0, len(rgb_points), 4)]
    colors = [
        [rgb_points[i + 1], rgb_points[i + 2], rgb_points[i + 3]]
        for i in range(0, len(rgb_points), 4)
    ]
    return LinearSegmentedColormap.from_list("Fast", list(zip(positions, colors)), N=256)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("infer", "render", "all"), default="all")
    parser.add_argument("--cities", nargs="+", default=list(LOCATIONS))
    parser.add_argument("--models", nargs="+", default=["unet", "trajgru", "fno", "kan"])
    parser.add_argument("--base-dir", type=Path, default=PROJECT_DIR / "24-24")
    parser.add_argument("--npy-dir", type=Path, default=PROJECT_DIR / "npy_data")
    parser.add_argument("--som-map", type=Path, default=PROJECT_DIR / "som_map.npy")
    parser.add_argument("--original-nc-dir", type=Path, default=PROJECT_DIR / "processed")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_DIR / "prediction_region_animations_24-24")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--pred-len", type=int, default=24)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--force-inference", action="store_true")
    return parser.parse_args()


def configure_fonts():
    arial = Path(os.environ.get("WIND_PROJECT_ARIAL_FONT", ""))
    if arial.is_file():
        font_manager.fontManager.addfont(str(arial))
    plt.rcParams.update({
        "font.family": "Arial",
        "axes.unicode_minus": False,
    })


def read_stats(path: Path):
    with path.open() as handle:
        return json.load(handle)


def denormalize_uv(values, stats):
    result = np.asarray(values, dtype=np.float32).copy()
    result[..., 0, :, :] = result[..., 0, :, :] * stats["u10"]["std"] + stats["u10"]["mean"]
    result[..., 1, :, :] = result[..., 1, :, :] * stats["v10"]["std"] + stats["v10"]["mean"]
    return result


def speed(uv):
    return np.hypot(uv[..., 0, :, :], uv[..., 1, :, :])


def build_model_args(args, city, model):
    # The factory only needs these architecture parameters.
    return argparse.Namespace(
        model_name=model,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        city=city,
    )


def encode_time_features(timestamps):
    year_start = timestamps.astype("datetime64[Y]")
    day_of_year = (
        timestamps.astype("datetime64[D]") - year_start.astype("datetime64[D]")
    ).astype(np.int64) + 1
    hour = (
        timestamps.astype("datetime64[h]") - timestamps.astype("datetime64[D]")
    ).astype(np.int64)
    day_angle = day_of_year / 365.0 * 2 * np.pi
    hour_angle = hour / 24.0 * 2 * np.pi
    return np.stack(
        [np.cos(day_angle), np.sin(day_angle), np.cos(hour_angle), np.sin(hour_angle)],
        axis=1,
    ).astype(np.float32)


def load_sample(args, city):
    lat, lon = LOCATIONS[city]
    weather = np.load(args.npy_dir / "weather_data_all.npy", mmap_mode="r")
    timestamps = np.load(args.npy_dir / "timestamps.npy", allow_pickle=True)
    years = timestamps.astype("datetime64[Y]").astype(np.int64) + 1970
    indices = np.flatnonzero((years >= 2021) & (years < 2025))
    total_steps = args.seq_len + args.pred_len
    indices = indices[indices < weather.shape[0] - total_steps]
    real_index = int(indices[args.sample_index])

    grid = np.load(PROJECT_DIR / "lat_lon_grid.npz")
    lats, lons = grid["lats"], grid["lons"]
    lat_index = int(np.argmin(np.abs(lats - lat)))
    lon_index = int(np.argmin(np.abs(lons - lon)))
    y_start = int(np.clip(lat_index - args.patch_size // 2, 0, len(lats) - args.patch_size))
    x_start = int(np.clip(lon_index - args.patch_size // 2, 0, len(lons) - args.patch_size))
    y_end, x_end = y_start + args.patch_size, x_start + args.patch_size

    values = np.array(
        weather[real_index:real_index + total_steps, :, y_start:y_end, x_start:x_end],
        dtype=np.float32,
    )
    input_times = timestamps[real_index:real_index + args.seq_len]
    return {
        "x_vars": values[:args.seq_len],
        "t_feat": encode_time_features(input_times),
        "y": values[args.seq_len:, :2],
        "coords": np.array([y_start, x_start, y_end, x_end], dtype=np.int32),
        "forecast_start": timestamps[real_index + args.seq_len],
    }


def cache_city(args, city, device):
    cache_dir = args.output_dir / "cache" / city
    cache_dir.mkdir(parents=True, exist_ok=True)
    reference_path = cache_dir / f"sample_{args.sample_index}_reference.npz"
    model_paths = {
        model: cache_dir / f"sample_{args.sample_index}_{model}.npz"
        for model in args.models
    }
    if (not args.force_inference and reference_path.exists()
            and all(path.exists() for path in model_paths.values())):
        print(f"[{city}] All inference caches exist; skipping.")
        return

    print(f"[{city}] Loading sample {args.sample_index}...")
    sample = load_sample(args, city)
    stats = read_stats(args.npy_dir / "normalization_stats.json")
    x_vars_np = sample["x_vars"]
    true_uv = denormalize_uv(sample["y"], stats)
    persistence_uv = denormalize_uv(x_vars_np[-1:, :2], stats)
    persistence_uv = np.repeat(persistence_uv, args.pred_len, axis=0)
    np.savez_compressed(
        reference_path,
        truth=true_uv,
        persistence=persistence_uv,
        coords=sample["coords"],
        forecast_start=sample["forecast_start"],
        sample_index=args.sample_index,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
    )
    print(f"[{city}] Saved reference cache: {reference_path}")

    x_vars = torch.from_numpy(x_vars_np[None]).to(device)
    t_feat = torch.from_numpy(sample["t_feat"][None]).to(device)
    batch, steps, _, height, width = x_vars.shape
    time_grid = t_feat.view(batch, steps, 4, 1, 1).expand(-1, -1, -1, height, width)
    model_input = torch.cat([x_vars, time_grid], dim=2)

    for model_name in args.models:
        output_path = model_paths[model_name]
        if output_path.exists() and not args.force_inference:
            print(f"[{city}/{model_name}] Cache exists; skipping.")
            continue
        checkpoint = args.base_dir / f"analysis_results_{city}" / model_name / "checkpoint.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        print(f"[{city}/{model_name}] Loading checkpoint and inferring...")
        model = get_model(build_model_args(args, city, model_name), device)
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if state and next(iter(state)).startswith("module."):
            state = {key.removeprefix("module."): value for key, value in state.items()}
        state.pop("_metadata", None)
        model.load_state_dict(state)
        model.eval()
        with torch.inference_mode():
            if model_name in {"trajgru", "convlstm", "convgru", "predrnn"}:
                prediction = model(model_input, pred_len=args.pred_len)
            else:
                prediction = model(model_input)
        pred_uv = denormalize_uv(prediction[0].detach().cpu().numpy(), stats)
        np.savez_compressed(output_path, prediction=pred_uv)
        print(f"[{city}/{model_name}] Saved: {output_path}")
        del model, prediction, pred_uv
        torch.cuda.empty_cache()


def spatial_metrics(pred, truth):
    error = pred - truth
    rmse = float(np.sqrt(np.mean(error ** 2)))
    p = pred.ravel()
    t = truth.ravel()
    if np.std(p) == 0 or np.std(t) == 0:
        corr = np.nan
    else:
        corr = float(np.corrcoef(p, t)[0, 1])
    amplitude = float(np.std(p) / np.std(t)) if np.std(t) else np.nan
    return rmse, corr, amplitude


def render_city(args, city):
    cache_dir = args.output_dir / "cache" / city
    reference_path = cache_dir / f"sample_{args.sample_index}_reference.npz"
    if not reference_path.exists():
        raise FileNotFoundError(reference_path)
    reference = np.load(reference_path)
    fields_uv = {
        "truth": reference["truth"],
        "persistence": reference["persistence"],
    }
    for model in args.models:
        path = cache_dir / f"sample_{args.sample_index}_{model}.npz"
        fields_uv[model] = np.load(path)["prediction"]
    # Match a14_vis.py grid-skill-map orientation exactly: raw patch rows are
    # north-to-south, so flip latitude only before imshow(origin="lower").
    fields = {
        name: np.flip(speed(values), axis=-2)
        for name, values in fields_uv.items()
    }

    all_values = np.concatenate([value.ravel() for value in fields.values()])
    vmin, vmax = np.nanpercentile(all_values, [1, 99])
    lat, lon = LOCATIONS[city]
    half = args.patch_size * 0.25 / 2
    extent = [lon - half, lon + half, lat - half, lat + half]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), constrained_layout=True)
    images = []
    metric_texts = []
    cmap = wind_colormap()
    for ax, name in zip(axes.flat, PANEL_ORDER):
        image = ax.imshow(fields[name][0], origin="lower", extent=extent,
                          cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        _add_admin_boundaries(ax, extent)
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.2)
        ax.set_title(DISPLAY_NAMES[name], fontsize=20, fontweight="bold")
        ax.tick_params(labelsize=18)
        images.append(image)
        if name == "truth":
            metric_texts.append(None)
        else:
            text = ax.text(
                0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=13,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78},
            )
            metric_texts.append(text)
    for ax in axes[-1]:
        ax.set_xlabel("Longitude (°E)", fontsize=18)
    for ax in axes[:, 0]:
        ax.set_ylabel("Latitude (°N)", fontsize=18)
    colorbar = fig.colorbar(images[0], ax=axes, location="right", shrink=0.9, pad=0.02)
    colorbar.set_label("10-m wind speed (m s$^{-1}$)", fontsize=18)
    colorbar.ax.tick_params(labelsize=20)
    title = fig.suptitle("", fontsize=20, fontweight="bold")

    def update(frame):
        truth = fields["truth"][frame]
        artists = []
        for index, name in enumerate(PANEL_ORDER):
            images[index].set_data(fields[name][frame])
            artists.append(images[index])
            if name != "truth":
                rmse, corr, amp = spatial_metrics(fields[name][frame], truth)
                metric_texts[index].set_text(
                    f"RMSE {rmse:.2f}\nr {corr:.2f} | Amp {amp:.2f}"
                )
                artists.append(metric_texts[index])
        title.set_text(f"{city.title()} — Forecast lead time: {frame + 1:02d} h")
        artists.append(title)
        return artists

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = args.output_dir / f"{city}_sample_{args.sample_index}_model_comparison.gif"
    anim = animation.FuncAnimation(fig, update, frames=args.pred_len, interval=1000 / args.fps,
                                   blit=False, repeat=True)
    anim.save(gif_path, writer=animation.PillowWriter(fps=args.fps), dpi=args.dpi)
    plt.close(fig)
    print(f"[{city}] Saved animation: {gif_path}")


def main():
    args = parse_args()
    configure_fonts()
    unknown = set(args.cities).difference(LOCATIONS)
    if unknown:
        raise ValueError(f"Unknown cities: {sorted(unknown)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if args.mode in {"infer", "all"}:
        for city in args.cities:
            cache_city(args, city, device)
    if args.mode in {"render", "all"}:
        for city in args.cities:
            render_city(args, city)


if __name__ == "__main__":
    main()
