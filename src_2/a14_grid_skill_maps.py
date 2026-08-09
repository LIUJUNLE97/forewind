#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A14: Grid-level persistence skill maps for station-centered 64x64 patches.

Purpose
-------
For each city-model pair:
  1. Load trained checkpoint.
  2. Re-run test forward pass only.
  3. Compute grid-cell-level wind-speed MSE for model and persistence.
  4. Save 64x64 maps:
       - mse_model_map.npy
       - mse_pers_map.npy
       - rmse_model_map.npy
       - rmse_pers_map.npy
       - ss_persist_map.npy
  5. Save a summary CSV for sanity checks.

Persistence definition
----------------------
For every sample n, grid cell (i,j), and forecast lead h:

    yhat_pers[n,h,i,j] = y_init[n,i,j]

where y_init is the last observed wind speed from the input window.
This uses batch['x_vars'][:, -1, 0:2] directly, so it does not rely on
stride-1 reconstruction or test-set ordering.

Important
---------
All skill maps are computed in physical wind-speed units (m/s), after
denormalizing u10 and v10 using normalization_stats.json.
"""

import argparse
import json
import os
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data_provider.data_factory import data_provider
from model.model_factory import get_model


# ---------------------------------------------------------------------
# Station / model configuration
# ---------------------------------------------------------------------
CITY_COORDS = {
    "shanghai":  (31.23, 121.47),
    "chengdu":   (30.67, 104.06),
    "hohhot":    (40.84, 111.75),
    "qingdao":   (36.06, 120.38),
    "guangzhou": (23.13, 113.26),
    "urumqi":    (43.82, 87.62),
    "lhasa":     (29.65, 91.11),
    "golmud":    (36.40, 94.90),
    "kunming":   (25.04, 102.71),
    "wuhan":     (30.59, 114.30),
    "fuzhou":    (26.08, 119.30),
    "haikou":    (20.03, 110.32),
}

MODELS = [
    "cnn", "fno", "kan", "convlstm",
    "convgru", "trajgru", "predrnn", "unet"
]

MODEL_BATCH_SIZE = {
    "cnn": 512,
    "fno": 512,
    "kan": 32,
    "convlstm": 32,
    "convgru": 32,
    "trajgru": 16,
    "predrnn": 16,
    "unet": 512,
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_stats(stats_file: Path):
    with open(stats_file, "r") as f:
        stats = json.load(f)

    try:
        u_mean = stats["u10"]["mean"]
        u_std = stats["u10"]["std"]
        v_mean = stats["v10"]["mean"]
        v_std = stats["v10"]["std"]
    except KeyError:
        print("Warning: keys 'u10'/'v10' not found. Trying fallback keys '0'/'1'.")
        u_mean = stats.get("0", {}).get("mean", 0.0)
        u_std = stats.get("0", {}).get("std", 1.0)
        v_mean = stats.get("1", {}).get("mean", 0.0)
        v_std = stats.get("1", {}).get("std", 1.0)

    return {
        "u_mean": float(u_mean),
        "u_std": float(u_std),
        "v_mean": float(v_mean),
        "v_std": float(v_std),
    }


def denorm_uv(uv_norm: torch.Tensor, stats: dict) -> torch.Tensor:
    """
    Denormalize u/v components.

    Supported shapes:
      [B, T, 2, H, W]
      [B, 2, H, W]

    Returns tensor with the same shape.
    """
    uv = uv_norm.clone()

    if uv.dim() == 5:
        uv[:, :, 0] = uv[:, :, 0] * stats["u_std"] + stats["u_mean"]
        uv[:, :, 1] = uv[:, :, 1] * stats["v_std"] + stats["v_mean"]
    elif uv.dim() == 4:
        uv[:, 0] = uv[:, 0] * stats["u_std"] + stats["u_mean"]
        uv[:, 1] = uv[:, 1] * stats["v_std"] + stats["v_mean"]
    else:
        raise ValueError(f"Unsupported uv tensor shape: {tuple(uv.shape)}")

    return uv


def speed_from_uv(uv: torch.Tensor) -> torch.Tensor:
    """
    Convert u/v to wind speed.

    Input:
      [B, T, 2, H, W] -> output [B, T, H, W]
      [B, 2, H, W]    -> output [B, H, W]
    """
    if uv.dim() == 5:
        return torch.sqrt(uv[:, :, 0] ** 2 + uv[:, :, 1] ** 2)
    if uv.dim() == 4:
        return torch.sqrt(uv[:, 0] ** 2 + uv[:, 1] ** 2)
    raise ValueError(f"Unsupported uv tensor shape: {tuple(uv.shape)}")


def reconstruct_input_on_gpu(batch, device):
    """
    Same logic as run_models.py:
    x_vars [B,T,C,H,W] + t_feat [B,T,4] -> x_final [B,T,C+4,H,W].
    Also return raw x_vars because persistence anchor must use physical variables only.
    """
    x_vars = batch["x_vars"].to(device, non_blocking=True)
    t_feat = batch["t_feat"].to(device, non_blocking=True)
    y = batch["y"].to(device, non_blocking=True)

    B, T, _, H, W = x_vars.shape
    t_feat_expanded = t_feat.view(B, T, 4, 1, 1).expand(-1, -1, -1, H, W)
    x_final = torch.cat([x_vars, t_feat_expanded], dim=2)

    return x_final, x_vars, y


def forward_model(model, x, pred_len):
    """
    Robust forward wrapper.
    Your official test path uses model(x, pred_len=pred_len), but this fallback
    protects models whose forward does not expose pred_len.
    """
    try:
        pred = model(x, pred_len=pred_len)
    except TypeError:
        pred = model(x)

    if isinstance(pred, (tuple, list)):
        pred = pred[0]

    return pred


def load_checkpoint(model, checkpoint_path: Path, device):
    obj = torch.load(checkpoint_path, map_location=device)

    if isinstance(obj, dict) and "model_state_dict" in obj:
        state_dict = obj["model_state_dict"]
    else:
        state_dict = obj

    if isinstance(state_dict, dict) and "_metadata" in state_dict:
        del state_dict["_metadata"]

    model.load_state_dict(state_dict, strict=True)
    return model


def latitude_weights(patch_size: int, lat_range=(54, 18)):
    """
    Same weighting convention as Evaluator:
    weights are normalized cosine-latitude weights.
    Shape: [H, 1]
    """
    lats = np.linspace(lat_range[0], lat_range[1], patch_size)
    weights = np.cos(np.deg2rad(lats))
    weights = weights / weights.mean()
    return weights.reshape(patch_size, 1)


# ---------------------------------------------------------------------
# Core A14 computation
# ---------------------------------------------------------------------
@torch.no_grad()
def compute_grid_skill_for_one(args, city: str, model_name: str, device):
    lat, lon = CITY_COORDS[city]

    run_args = deepcopy(args)
    run_args.model_name = model_name
    run_args.target_lat = lat
    run_args.target_lon = lon
    run_args.fix_location = True
    run_args.test_location_config = {
        "type": "fixed",
        "lat": lat,
        "lon": lon,
    }

    if args.batch_size <= 0:
        run_args.batch_size = MODEL_BATCH_SIZE.get(model_name, 32)

    model_dir = Path(args.base_dir) / f"analysis_results_{city}" / model_name
    checkpoint_path = model_dir / args.checkpoint_name

    if not checkpoint_path.exists():
        print(f"[SKIP] Missing checkpoint: {checkpoint_path}")
        return None

    out_dir = Path(args.output_root) / f"analysis_results_{city}" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"A14 grid skill | city={city} | model={model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output:     {out_dir}")
    print("=" * 80)

    # Data
    _, test_loader = data_provider(run_args, "test")

    # Model
    model = get_model(run_args, device)
    model = load_checkpoint(model, checkpoint_path, device)
    model.to(device)
    model.eval()

    stats = load_stats(Path(run_args.nc_dir) / run_args.stats_file)

    sum_se_model = None
    sum_se_pers = None
    count = 0

    for batch in tqdm(test_loader, desc=f"{city}-{model_name}", leave=False):
        x, x_vars, y_norm = reconstruct_input_on_gpu(batch, device)

        pred_norm = forward_model(model, x, pred_len=run_args.pred_len)

        # Expected: pred/y [B,T,2,H,W]
        if pred_norm.shape[:3] != y_norm.shape[:3]:
            raise RuntimeError(
                f"pred/y shape mismatch: pred={tuple(pred_norm.shape)}, y={tuple(y_norm.shape)}"
            )

        # Denormalize u/v into m/s
        pred_uv = denorm_uv(pred_norm[:, :, 0:2], stats)
        true_uv = denorm_uv(y_norm[:, :, 0:2], stats)

        # Persistence anchor: last input frame u/v, not time features.
        # x_vars is before time-feature concatenation.
        init_uv = denorm_uv(x_vars[:, -1, 0:2], stats)

        speed_pred = speed_from_uv(pred_uv)       # [B,T,H,W]
        speed_true = speed_from_uv(true_uv)       # [B,T,H,W]
        speed_init = speed_from_uv(init_uv)       # [B,H,W]

        # Squared errors accumulated over B and T, leaving grid [H,W]
        se_model = ((speed_pred - speed_true) ** 2).sum(dim=(0, 1))
        se_pers = ((speed_init[:, None, :, :] - speed_true) ** 2).sum(dim=(0, 1))

        se_model_np = se_model.detach().cpu().numpy().astype(np.float64)
        se_pers_np = se_pers.detach().cpu().numpy().astype(np.float64)

        if sum_se_model is None:
            H, W = se_model_np.shape
            sum_se_model = np.zeros((H, W), dtype=np.float64)
            sum_se_pers = np.zeros((H, W), dtype=np.float64)

        sum_se_model += se_model_np
        sum_se_pers += se_pers_np
        count += speed_true.shape[0] * speed_true.shape[1]

    if count == 0:
        print(f"[SKIP] Empty test set for {city}-{model_name}.")
        return None

    mse_model = sum_se_model / count
    mse_pers = sum_se_pers / count

    rmse_model = np.sqrt(mse_model)
    rmse_pers = np.sqrt(mse_pers)

    ratio = np.full_like(mse_model, np.nan, dtype=np.float64)
    np.divide(mse_model, mse_pers, out=ratio, where=mse_pers > 1e-12)
    ss_map = 1.0 - ratio

    # Save maps
    np.save(out_dir / "mse_model_map.npy", mse_model)
    np.save(out_dir / "mse_pers_map.npy", mse_pers)
    np.save(out_dir / "rmse_model_map.npy", rmse_model)
    np.save(out_dir / "rmse_pers_map.npy", rmse_pers)
    np.save(out_dir / "ss_persist_map.npy", ss_map)
    np.save(out_dir / "count.npy", np.array(count, dtype=np.int64))

    # Sanity checks
    H, W = mse_model.shape
    weights = latitude_weights(H)
    patch_rmse_unweighted = float(np.sqrt(np.nanmean(mse_model)))
    patch_rmse_weighted = float(np.sqrt(np.nanmean(weights * mse_model)))

    center_y, center_x = H // 2, W // 2
    center_rmse = float(rmse_model[center_y, center_x])
    center_ss = float(ss_map[center_y, center_x])

    mean_ss = float(np.nanmean(ss_map))
    median_ss = float(np.nanmedian(ss_map))
    frac_negative = float(np.nanmean(ss_map < 0))

    # Compare with previous CSV if available
    old_csv = model_dir / "final_eval_metrics.csv"
    old_patch_rmse = np.nan
    old_single_rmse = np.nan
    if old_csv.exists():
        try:
            old = pd.read_csv(old_csv)
            if "Patch_Speed_RMSE" in old.columns:
                old_patch_rmse = float(old["Patch_Speed_RMSE"].iloc[0])
            if "SinglePoint_RMSE" in old.columns:
                old_single_rmse = float(old["SinglePoint_RMSE"].iloc[0])
        except Exception as e:
            import traceback
            print(f"[FAILED] city={city}, model={model_name}: {repr(e)}")
            traceback.print_exc()

    summary = {
        "City": city.capitalize(),
        "Model": model_name,
        "Lat": lat,
        "Lon": lon,
        "N_times_T_count": int(count),
        "Patch_RMSE_from_map_unweighted": patch_rmse_unweighted,
        "Patch_RMSE_from_map_weighted": patch_rmse_weighted,
        "Old_Patch_Speed_RMSE_csv": old_patch_rmse,
        "Center_RMSE_from_map": center_rmse,
        "Old_SinglePoint_RMSE_csv": old_single_rmse,
        "Mean_SS_grid": mean_ss,
        "Median_SS_grid": median_ss,
        "Center_SS_grid": center_ss,
        "Frac_grid_SS_negative": frac_negative,
        "Min_SS_grid": float(np.nanmin(ss_map)),
        "Max_SS_grid": float(np.nanmax(ss_map)),
    }

    pd.DataFrame([summary]).to_csv(out_dir / "a14_grid_skill_summary.csv", index=False)

    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return summary


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="A14 grid-level persistence skill maps")
    project_root = Path(os.environ.get("WIND_PROJECT_ROOT", Path(__file__).resolve().parents[1]))

    parser.add_argument(
        "--base_dir",
        type=str,
        default=str(project_root / "24-24"),
        help="Base directory containing analysis_results_{city}/{model}/checkpoint.pth",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(project_root / "fig/a14_grid_skill"),
        help="Output root for A14 maps",
    )

    parser.add_argument(
        "--nc_dir",
        type=str,
        default=str(project_root / "npy_data"),
    )
    parser.add_argument(
        "--stats_file",
        type=str,
        default="normalization_stats.json",
    )
    parser.add_argument(
        "--som_map_path",
        type=str,
        default=str(project_root / "som_map.npy"),
    )

    parser.add_argument("--checkpoint_name", type=str, default="checkpoint.pth")

    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--pred_len", type=int, default=24)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=-1,
        help="If <=0, use model-specific default batch size.",
    )
    parser.add_argument("--num_workers", type=int, default=4)

    # These are required by data_provider / model constructors.
    parser.add_argument("--model_name", type=str, default="cnn")
    parser.add_argument("--target_lat", type=float, default=31.23)
    parser.add_argument("--target_lon", type=float, default=121.47)
    parser.add_argument("--fix_location", action="store_true", default=True)
    parser.add_argument("--eval_window_size", type=int, default=1)

    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--features", type=str, default="M")

    # Model constructor may expect these even during inference.
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--use_amp", action="store_true", default=False)

    parser.add_argument(
        "--cities",
        nargs="+",
        default=list(CITY_COORDS.keys()),
        help="Cities to process. Example: --cities shanghai chengdu",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        help="Models to process. Example: --models cnn fno unet",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_summaries = []

    for city in args.cities:
        city = city.lower()
        if city not in CITY_COORDS:
            print(f"[SKIP] Unknown city: {city}")
            continue

        for model_name in args.models:
            model_name = model_name.lower()
            if model_name not in MODELS:
                print(f"[SKIP] Unknown model: {model_name}")
                continue

            try:
                summary = compute_grid_skill_for_one(args, city, model_name, device)
                if summary is not None:
                    all_summaries.append(summary)
            except Exception as e:
                print(f"[FAILED] city={city}, model={model_name}: {repr(e)}")

    if all_summaries:
        out_root = Path(args.output_root)
        out_root.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(all_summaries)
        df.to_csv(out_root / "a14_all_grid_skill_summary.csv", index=False)

        print("\n" + "=" * 80)
        print("A14 all-summary saved:")
        print(out_root / "a14_all_grid_skill_summary.csv")
        print("=" * 80)
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
