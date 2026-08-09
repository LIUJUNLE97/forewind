"""Upload a selected data directory to Hugging Face without hard-coded credentials.

Set ``HF_TOKEN`` in the environment (or use an existing Hugging Face login)
before running this helper. Do not commit tokens to the repository.
"""

import argparse
import os

from huggingface_hub import HfApi, login


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", default="data", help="Directory to upload")
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("HF_REPO_ID", "Guandi123/ERA5_processed"),
        help="Hugging Face dataset repository ID",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

    print("Connecting to repository: " + args.repo_id)
    api = HfApi()
    api.upload_folder(folder_path=args.folder, repo_id=args.repo_id, repo_type="dataset")
    print("Upload finished successfully!")


if __name__ == "__main__":
    main()
