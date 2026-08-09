# Lightweight release metadata

These files are small enough to distribute with the source code:

- `som_map.npy`: final 3×3 SOM cluster labels;
- `lat_lon_grid.npz`: latitude/longitude grid used by the maps;
- `normalization_stats.json`: statistics used to normalize the model variables.

The raw/processed ERA5 fields and model checkpoints are intentionally not copied here. Their locations, sizes, and publication requirements are documented in [../../docs/DATA_MANIFEST.md](../../docs/DATA_MANIFEST.md).
