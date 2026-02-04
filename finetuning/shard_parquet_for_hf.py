import argparse
import inspect
import os
import shutil
import sys

from datasets import load_dataset
from huggingface_hub import upload_folder


DEFAULT_MAX_SHARD_GB = 4.5


def _ensure_empty_dir(path, overwrite):
    if os.path.exists(path):
        if overwrite:
            shutil.rmtree(path)
        elif os.listdir(path):
            raise RuntimeError(f"output_dir is not empty: {path}")
    os.makedirs(path, exist_ok=True)


def _to_parquet(ds, output_dir, split, max_shard_bytes, overwrite):
    os.makedirs(output_dir, exist_ok=True)
    target = os.path.join(output_dir, f"{split}.parquet")
    kwargs = {}
    sig = inspect.signature(ds.to_parquet)
    if "max_shard_size" in sig.parameters:
        kwargs["max_shard_size"] = max_shard_bytes
    if "overwrite" in sig.parameters:
        kwargs["overwrite"] = overwrite
    ds.to_parquet(target, **kwargs)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Shard a large parquet file into valid parquet shards for HF Dataset Viewer."
    )
    parser.add_argument(
        "--input-parquet",
        required=True,
        help="Path to the source parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        default="parquet_shards",
        help="Where to write parquet shards.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split name used for shard file prefix (train/test/validation).",
    )
    parser.add_argument(
        "--max-shard-size-gb",
        type=float,
        default=DEFAULT_MAX_SHARD_GB,
        help="Shard size in GB (binary GiB).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output-dir if it exists.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="If set, upload shards to this repo after writing.",
    )
    parser.add_argument(
        "--path-in-repo",
        default="data",
        help="Repo subfolder to upload shards into.",
    )
    parser.add_argument(
        "--repo-type",
        default="dataset",
        help="dataset or model.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Max parallel workers for upload_folder.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload parquet shards",
        help="Commit message for the Hub.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token (optional; otherwise uses HF_TOKEN env).",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete local shards after upload.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    input_path = args.input_parquet
    if not os.path.isfile(input_path):
        raise RuntimeError(f"input parquet not found: {input_path}")

    _ensure_empty_dir(args.output_dir, args.overwrite)
    max_shard_bytes = int(args.max_shard_size_gb * (1024 ** 3))

    print("loading parquet...")
    ds = load_dataset("parquet", data_files=input_path, split="train")
    print("writing shards...")
    _to_parquet(ds, args.output_dir, args.split, max_shard_bytes, args.overwrite)

    if args.repo_id:
        print("uploading shards...")
        kwargs = {
            "folder_path": str(args.output_dir),
            "path_in_repo": args.path_in_repo,
            "repo_id": args.repo_id,
            "repo_type": args.repo_type,
            "commit_message": args.commit_message,
            "max_workers": args.max_workers,
        }
        if args.token:
            kwargs["token"] = args.token
        upload_folder(**kwargs)
        print("upload complete")
        if args.cleanup:
            shutil.rmtree(args.output_dir)
    else:
        print("done (no upload requested)")


if __name__ == "__main__":
    main()
