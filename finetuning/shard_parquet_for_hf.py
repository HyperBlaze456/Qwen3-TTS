import argparse
import os
import shutil
import sys

from huggingface_hub import upload_folder
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_MAX_SHARD_GB = 4.5
DEFAULT_BATCH_ROWS = 65536


def _ensure_empty_dir(path, overwrite):
    if os.path.exists(path):
        if overwrite:
            shutil.rmtree(path)
        elif os.listdir(path):
            raise RuntimeError(f"output_dir is not empty: {path}")
    os.makedirs(path, exist_ok=True)


def _iter_tables(parquet_file, batch_rows):
    schema = parquet_file.schema_arrow
    try:
        batch_iter = parquet_file.iter_batches(batch_size=batch_rows, use_threads=False)
        first_batch = next(batch_iter)
    except StopIteration:
        return
    except pa.ArrowNotImplementedError as exc:
        print(
            f"iter_batches failed ({exc}); falling back to row-group reader",
            file=sys.stderr,
        )
    else:
        yield pa.Table.from_batches([first_batch], schema=schema)
        for batch in batch_iter:
            yield pa.Table.from_batches([batch], schema=schema)
        return

    for rg in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(rg)
        for offset in range(0, table.num_rows, batch_rows):
            yield table.slice(offset, batch_rows)


def _stream_shard_parquet(
    input_path,
    output_dir,
    split,
    max_shard_bytes,
    batch_rows,
):
    os.makedirs(output_dir, exist_ok=True)
    parquet_file = pq.ParquetFile(input_path)
    schema = parquet_file.schema_arrow

    shard_paths = []
    shard_index = 0
    writer = None
    current_path = None
    wrote_any = False

    def _open_writer():
        nonlocal shard_index, writer, current_path
        current_path = os.path.join(output_dir, f"{split}-{shard_index:05d}.parquet")
        writer = pq.ParquetWriter(current_path, schema)
        shard_paths.append(current_path)
        shard_index += 1

    for table in _iter_tables(parquet_file, batch_rows):
        if writer is None:
            _open_writer()
        writer.write_table(table)
        wrote_any = True
        if os.path.getsize(current_path) >= max_shard_bytes:
            writer.close()
            writer = None
            current_path = None

    if writer is not None:
        writer.close()
    if not wrote_any:
        empty = pa.Table.from_arrays([], schema=schema)
        _open_writer()
        writer.write_table(empty)
        writer.close()

    total = len(shard_paths)
    renamed = []
    for idx, path in enumerate(shard_paths):
        new_name = f"{split}-{idx:05d}-of-{total:05d}.parquet"
        new_path = os.path.join(output_dir, new_name)
        os.replace(path, new_path)
        renamed.append(new_path)
    return renamed


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
        "--batch-rows",
        type=int,
        default=DEFAULT_BATCH_ROWS,
        help="Rows per batch when streaming (lower if you still OOM).",
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

    print("streaming parquet and writing shards...")
    shard_paths = _stream_shard_parquet(
        input_path=input_path,
        output_dir=args.output_dir,
        split=args.split,
        max_shard_bytes=max_shard_bytes,
        batch_rows=args.batch_rows,
    )
    print(f"wrote {len(shard_paths)} shards")

    if args.repo_id:
        print("uploading shards...")
        kwargs = {
            "folder_path": str(args.output_dir),
            "path_in_repo": args.path_in_repo,
            "repo_id": args.repo_id,
            "repo_type": args.repo_type,
            "commit_message": args.commit_message,
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
