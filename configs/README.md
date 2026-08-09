# Release configuration conventions

The historical scripts contain cluster-specific defaults such as `/mimer/...`, `/nobackup/...`, and personal font locations. For a public release, use paths relative to the checkout or pass them explicitly on the command line.

Recommended local layout:

```text
<checkout>/
  data/processed/       processed ERA5 NetCDF files
  data/npy_data/        normalized NumPy arrays and statistics
  data/som/             SOM map and coordinate metadata
  checkpoints/24-24/    model checkpoints (download separately)
  outputs/              generated figures, tables, and animations
```

Do not put credentials, cluster account names, absolute home paths, or Slurm logs in a public configuration file.
