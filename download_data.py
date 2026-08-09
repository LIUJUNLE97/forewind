"""Download the processed ERA5 dataset used by the project."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="Guandi123/ERA5_processed")
    parser.add_argument(
        "--local-dir", type=Path, default=Path("data"),
        help="Local dataset root; the repository's processed/ subdirectory is preserved",
    )
    parser.add_argument("--allow-pattern", default="processed/*.nc")
    return parser.parse_args()


def main():
    args = parse_args()
    args.local_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Downloading {args.allow_pattern} from {args.repo_id} "
        f"to {args.local_dir}..."
    )
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        allow_patterns=args.allow_pattern,
        local_dir=str(args.local_dir),
    )
    print(f"Download complete: {args.local_dir.resolve()}")


if __name__ == "__main__":
    main()
