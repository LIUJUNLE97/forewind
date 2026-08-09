# Public submission scope

The GitHub repository is intended to contain the reusable source code and the
small metadata needed to understand or rerun the workflows. The following
items remain local or are published separately:

## Include in Git

- Python source files, shell entry points, and lightweight configuration
  examples;
- documentation, dependency specifications, and license/citation templates;
- `data/metadata/` files that are small enough to distribute and do not contain
  the underlying weather archive.

## Keep out of Git

- raw or processed ERA5 files (`*.nc`, `processed/`);
- full NumPy weather arrays and other derived archives (`npy_data/`, `*.npy`);
- model checkpoints and optimizer state (`*.pth`, `*.pt`, `*.ckpt`);
- experiment directories and generated figures (`24-24/`, `7-24/`, `fig/`,
  `results/`, `*.png`, `*.pdf`, `*.gif`, `*.avi`);
- virtual environments, runtime libraries, caches, and scheduler logs.

Large datasets and checkpoints should be deposited in a citable data archive
(for example Zenodo or Hugging Face) and referenced by an immutable revision or
DOI in the final release README. The repository itself must remain cloneable
without Git LFS or access to a compute cluster.

Before the public push, run:

```bash
git status --short
git ls-files -z | xargs -0 -n1 du -h 2>/dev/null | sort -h | tail
git diff --check
```

These checks are intended to catch accidental inclusion of weights, raw data,
generated products, or whitespace errors in the release commit.
