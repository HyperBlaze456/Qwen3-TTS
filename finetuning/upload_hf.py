import argparse
import json
import os
import shutil
import sys

from huggingface_hub import upload_folder


README_NAME = "README_CHUNKS.txt"
DEFAULT_CHUNK_GB = 4.5


def _iter_files(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [
            name for name in dirnames if name not in {".git", ".hg", ".svn", "__pycache__"}
        ]
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            if os.path.islink(path):
                print(f"skip symlink: {path}", file=sys.stderr)
                continue
            yield path


def _ensure_empty_dir(path, overwrite):
    if os.path.exists(path):
        if overwrite:
            shutil.rmtree(path)
        elif os.listdir(path):
            raise RuntimeError(f"staging_dir is not empty: {path}")
    os.makedirs(path, exist_ok=True)


def _copy_or_link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst)
        return "linked"
    except OSError:
        shutil.copy2(src, dst)
        return "copied"


def _chunk_file(src_path, dest_base, chunk_bytes, overwrite_chunks):
    manifest_path = f"{dest_base}.parts.json"
    if os.path.exists(manifest_path) and not overwrite_chunks:
        print(f"skip chunking (manifest exists): {manifest_path}")
        return

    os.makedirs(os.path.dirname(dest_base), exist_ok=True)
    part_names = []
    idx = 0
    with open(src_path, "rb") as src:
        while True:
            chunk = src.read(chunk_bytes)
            if not chunk:
                break
            part_path = f"{dest_base}.part{idx:05d}"
            with open(part_path, "wb") as out:
                out.write(chunk)
            part_names.append(os.path.basename(part_path))
            idx += 1

    manifest = {
        "original_file": os.path.basename(src_path),
        "original_bytes": os.path.getsize(src_path),
        "chunk_bytes": chunk_bytes,
        "parts": part_names,
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def _write_readme(target_dir):
    readme_path = os.path.join(target_dir, README_NAME)
    content = (
        "Chunked files were created by finetuning/upload_hf.py.\n"
        "Each original file has a .parts.json manifest next to the parts.\n"
        "Rebuild example:\n"
        "  python - <<'PY'\n"
        "  import json, os, sys\n"
        "  manifest = sys.argv[1]\n"
        "  with open(manifest, 'r', encoding='utf-8') as f:\n"
        "      meta = json.load(f)\n"
        "  out_path = os.path.join(os.path.dirname(manifest), meta['original_file'])\n"
        "  with open(out_path, 'wb') as out:\n"
        "      for part_name in meta['parts']:\n"
        "          part_path = os.path.join(os.path.dirname(manifest), part_name)\n"
        "          with open(part_path, 'rb') as pf:\n"
        "              out.write(pf.read())\n"
        "  print(out_path)\n"
        "  PY /path/to/file.parts.json\n"
    )
    with open(readme_path, "w", encoding="utf-8") as handle:
        handle.write(content)


def _normalize_relpath(root_dir, file_path):
    rel = os.path.relpath(file_path, root_dir)
    return rel.replace(os.sep, "/")


def _validate_dirs(output_dir, staging_dir, in_place):
    if not os.path.isdir(output_dir):
        raise RuntimeError(f"output_dir not found: {output_dir}")
    if in_place:
        return
    output_real = os.path.realpath(output_dir)
    staging_real = os.path.realpath(staging_dir)
    if os.path.commonpath([output_real, staging_real]) == output_real:
        raise RuntimeError("staging_dir cannot be inside output_dir")


def _prepare_staging(
    output_dir,
    staging_dir,
    chunk_bytes,
    overwrite_staging,
    overwrite_chunks,
    remove_original,
    write_readme,
):
    _ensure_empty_dir(staging_dir, overwrite_staging)
    for src_path in _iter_files(output_dir):
        rel = os.path.relpath(src_path, output_dir)
        dest_base = os.path.join(staging_dir, rel)
        size = os.path.getsize(src_path)
        if size > chunk_bytes:
            print(f"chunking {rel} ({size} bytes)")
            _chunk_file(src_path, dest_base, chunk_bytes, overwrite_chunks)
            if remove_original:
                os.remove(src_path)
        else:
            _copy_or_link(src_path, dest_base)
    if write_readme:
        _write_readme(staging_dir)


def _prepare_in_place(
    output_dir,
    chunk_bytes,
    overwrite_chunks,
    remove_original,
    write_readme,
):
    ignore_patterns = []
    for src_path in _iter_files(output_dir):
        size = os.path.getsize(src_path)
        if size > chunk_bytes:
            rel = _normalize_relpath(output_dir, src_path)
            print(f"chunking {rel} ({size} bytes)")
            _chunk_file(src_path, src_path, chunk_bytes, overwrite_chunks)
            if remove_original:
                os.remove(src_path)
            else:
                ignore_patterns.append(rel)
    if write_readme:
        _write_readme(output_dir)
    return ignore_patterns


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Chunk large files and upload a folder to the Hugging Face Hub."
    )
    parser.add_argument("--output-dir", required=True, help="Local folder to upload.")
    parser.add_argument("--repo-id", required=True, help="Target repo, e.g. org/name.")
    parser.add_argument("--path-in-repo", default="data", help="Repo subfolder.")
    parser.add_argument("--repo-type", default="dataset", help="dataset or model.")
    parser.add_argument(
        "--chunk-size-gb",
        type=float,
        default=DEFAULT_CHUNK_GB,
        help="Chunk size in GB (binary GiB).",
    )
    parser.add_argument(
        "--staging-dir",
        default=None,
        help="Where to build chunked upload tree (default: <output-dir>_hf_chunks).",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Chunk files in-place to avoid a staging copy.",
    )
    parser.add_argument(
        "--remove-original",
        action="store_true",
        help="Delete original large files after chunking.",
    )
    parser.add_argument(
        "--overwrite-staging",
        action="store_true",
        help="Delete and recreate staging-dir if it exists.",
    )
    parser.add_argument(
        "--overwrite-chunks",
        action="store_true",
        help="Re-chunk files even if .parts.json exists.",
    )
    parser.add_argument(
        "--no-readme",
        action="store_true",
        help="Do not write README_CHUNKS.txt into the upload root.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Max parallel workers for upload_folder.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload chunked dataset",
        help="Commit message for the Hub.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token (optional; otherwise uses HF_TOKEN env).",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    chunk_bytes = int(args.chunk_size_gb * (1024 ** 3))
    staging_dir = args.staging_dir or f"{args.output_dir.rstrip(os.sep)}_hf_chunks"
    _validate_dirs(args.output_dir, staging_dir, args.in_place)

    ignore_patterns = ["**/.git/**", "**/__pycache__/**"]
    if args.in_place:
        ignore_patterns.extend(
            _prepare_in_place(
                output_dir=args.output_dir,
                chunk_bytes=chunk_bytes,
                overwrite_chunks=args.overwrite_chunks,
                remove_original=args.remove_original,
                write_readme=not args.no_readme,
            )
        )
        upload_root = args.output_dir
    else:
        _prepare_staging(
            output_dir=args.output_dir,
            staging_dir=staging_dir,
            chunk_bytes=chunk_bytes,
            overwrite_staging=args.overwrite_staging,
            overwrite_chunks=args.overwrite_chunks,
            remove_original=args.remove_original,
            write_readme=not args.no_readme,
        )
        upload_root = staging_dir

    kwargs = {
        "folder_path": str(upload_root),
        "path_in_repo": args.path_in_repo,
        "repo_id": args.repo_id,
        "repo_type": args.repo_type,
        "commit_message": args.commit_message,
        "max_workers": args.max_workers,
    }
    if args.token:
        kwargs["token"] = args.token
    if ignore_patterns:
        kwargs["ignore_patterns"] = ignore_patterns

    upload_folder(**kwargs)
    print("upload complete")


if __name__ == "__main__":
    main()
