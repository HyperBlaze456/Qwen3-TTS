"""Add audio_codes column to an existing parquet produced by prepare_hf_sft_data.py.

Usage:
    python finetuning/add_audio_codes.py \
        --input_parquet data/existing.parquet \
        --output_parquet data/with_codes.parquet \
        --batch_size 4
"""

import argparse
import gc
import os
import re
import sys
import tempfile
import time

import torch

from prepare_hf_sft_data import (
    BATCH_INFER_NUM,
    PART_RE,
    _clear_cuda_cache,
    _disable_model_cache,
    _ensure_dir,
    _get_parquet_row_count,
    _init_log_file,
    _is_cuda_oom,
    _load_json,
    _log_progress,
    _release_exception,
    _scan_parquet_dir,
    _write_json_atomic,
    _write_temp_audio,
)


def _scan_input_dir(input_dir):
    """Find all .parquet files in a directory, sorted by name."""
    if not input_dir or not os.path.isdir(input_dir):
        return []
    files = []
    for name in sorted(os.listdir(input_dir)):
        if name.endswith(".parquet"):
            files.append(os.path.join(input_dir, name))
    return files


def _iter_input_rows(input_paths, skip_rows=0):
    """Yield (global_idx, row_dict) reading one row-group at a time for bounded memory."""
    import pyarrow.parquet as pq

    global_idx = 0
    for path in input_paths:
        pf = pq.ParquetFile(path)
        for rg_idx in range(pf.num_row_groups):
            table = pf.read_row_group(rg_idx)
            rows = table.to_pylist()
            del table
            for row in rows:
                if global_idx < skip_rows:
                    global_idx += 1
                    continue
                yield global_idx, row
                global_idx += 1


def _encode_batch_with_fallback(tokenizer, batch_rows, batch_paths, batch_idx):
    """Encode a batch of audio paths, splitting on OOM."""

    def _encode_chunk(chunk_rows, chunk_paths, depth):
        enc = None
        try:
            with torch.no_grad():
                enc = tokenizer.encode(chunk_paths)
                codes = []
                for i in range(len(enc.audio_codes)):
                    codes.append(enc.audio_codes[i].contiguous().cpu().tolist())
                    enc.audio_codes[i] = None
                del enc
                enc = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                return list(zip(chunk_rows, codes))
        except Exception as exc:
            if enc is not None:
                try:
                    for i in range(len(enc.audio_codes)):
                        enc.audio_codes[i] = None
                except Exception:
                    pass
                del enc
                enc = None
            gc.collect()
            _clear_cuda_cache()

            is_oom = _is_cuda_oom(exc)
            if is_oom:
                _release_exception(exc)
                exc = None
                gc.collect()
                _clear_cuda_cache()

            if not is_oom:
                raise
            _log_progress(
                f"[warn] CUDA OOM in batch {batch_idx} (size={len(chunk_rows)}, depth={depth}). "
                "Splitting batch to reduce max sequence length."
            )
            if len(chunk_rows) == 1:
                _log_progress(f"[warn] Skipping sample due to CUDA OOM: {chunk_paths[0]}")
                return []
            mid = len(chunk_rows) // 2
            left = _encode_chunk(chunk_rows[:mid], chunk_paths[:mid], depth + 1)
            gc.collect()
            _clear_cuda_cache()
            right = _encode_chunk(chunk_rows[mid:], chunk_paths[mid:], depth + 1)
            return left + right

    return _encode_chunk(batch_rows, batch_paths, 0)


def _resolve_output_mode(output_path, output_mode):
    if output_mode != "auto":
        return output_mode
    if output_path.endswith(os.sep) or os.path.isdir(output_path):
        return "dataset"
    return "file"


def main():
    parser = argparse.ArgumentParser(
        description="Add audio_codes column to an existing parquet."
    )
    parser.add_argument(
        "--input_parquet",
        type=str,
        required=True,
        help="Single .parquet file or directory of part-NNNNN.parquet shards.",
    )
    parser.add_argument(
        "--output_parquet",
        type=str,
        required=True,
        help="Output path (file or directory, must differ from input).",
    )
    parser.add_argument(
        "--output_mode",
        type=str,
        choices=("auto", "file", "dataset"),
        default="auto",
        help="file: single parquet, dataset: sharded directory, auto: infer from path.",
    )
    parser.add_argument(
        "--tokenizer_model_path",
        type=str,
        default="Qwen/Qwen3-TTS-Tokenizer-12Hz",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=BATCH_INFER_NUM)
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Start fresh, ignore progress.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Append log to file.",
    )
    args = parser.parse_args()

    if args.log_file:
        _init_log_file(args.log_file)

    # --- Resolve input paths ---
    input_path = os.path.abspath(args.input_parquet)
    output_path = os.path.abspath(args.output_parquet)

    if os.path.isdir(input_path):
        input_paths = _scan_input_dir(input_path)
        if not input_paths:
            _log_progress(f"No .parquet files found in {input_path}")
            sys.exit(1)
    elif os.path.isfile(input_path):
        input_paths = [input_path]
    else:
        _log_progress(f"Input not found: {input_path}")
        sys.exit(1)

    if input_path == output_path:
        _log_progress("Input and output paths must differ.")
        sys.exit(1)

    total_input_rows = sum(_get_parquet_row_count(p) for p in input_paths)
    output_mode = _resolve_output_mode(output_path, args.output_mode)

    _log_progress("=" * 50)
    _log_progress("add_audio_codes")
    _log_progress(f"  Input: {args.input_parquet} ({len(input_paths)} file(s), {total_input_rows} rows)")
    _log_progress(f"  Output: {args.output_parquet} (mode={output_mode})")
    _log_progress(f"  Tokenizer: {args.tokenizer_model_path}")
    _log_progress(f"  Device: {args.device}")
    _log_progress(f"  Batch size: {args.batch_size}")
    _log_progress("=" * 50)

    # --- Load tokenizer ---
    _log_progress(f"Loading tokenizer from {args.tokenizer_model_path}...")
    try:
        from qwen_tts import Qwen3TTSTokenizer
    except Exception as exc:
        raise SystemExit(
            "Missing dependency: qwen_tts. Install qwen-tts first."
        ) from exc

    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        args.tokenizer_model_path,
        device_map=args.device,
    )
    _disable_model_cache(tokenizer.model)
    _log_progress("Tokenizer loaded (encoder cache disabled).")

    # --- Resume state ---
    if output_mode == "file":
        state_path = f"{output_path}.resume.json"
        output_dir = os.path.dirname(output_path) or "."
    else:
        output_dir = output_path
        state_path = os.path.join(output_dir, "_resume_state.json")

    rows_completed = 0
    output_rows_written = 0
    next_output_part = 0

    if not args.no_resume:
        resume_state = _load_json(state_path)
        if resume_state:
            saved_input = resume_state.get("input_parquet", "")
            if os.path.abspath(saved_input) != input_path:
                _log_progress(
                    f"[warn] Resume state input ({saved_input}) != current input ({input_path}). "
                    "Ignoring state."
                )
            else:
                rows_completed = int(resume_state.get("rows_completed", 0))
                output_rows_written = int(resume_state.get("output_rows_written", 0))
                next_output_part = int(resume_state.get("next_output_part", 0))
                _log_progress(
                    f"Resuming: {rows_completed} rows completed, "
                    f"{output_rows_written} output rows, next part={next_output_part}"
                )

    # --- PyArrow imports ---
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:
        raise SystemExit("Missing dependency: pyarrow.") from exc

    # --- Processing ---
    tmp_dir = tempfile.TemporaryDirectory()
    writer = None
    tmp_output_file = None
    existing_schema = None
    batch_count = 0
    skipped_oom = 0
    skipped_no_audio = 0
    completed_ok = False

    batch_rows = []
    batch_paths = []
    temp_paths = []

    def _persist_state():
        _write_json_atomic(state_path, {
            "input_parquet": args.input_parquet,
            "rows_completed": rows_completed,
            "output_rows_written": output_rows_written,
            "next_output_part": next_output_part,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

    def _build_table(rows_list):
        nonlocal existing_schema
        if existing_schema is not None:
            return pa.Table.from_pylist(rows_list, schema=existing_schema)
        table = pa.Table.from_pylist(rows_list)
        existing_schema = table.schema
        return table

    def _init_file_writer(schema):
        nonlocal writer, tmp_output_file
        if writer is not None:
            return
        _ensure_dir(os.path.dirname(output_path) or ".")
        fd, tmp_path = tempfile.mkstemp(
            suffix=".parquet", dir=os.path.dirname(output_path) or "."
        )
        os.close(fd)
        tmp_output_file = tmp_path
        writer = pq.ParquetWriter(tmp_output_file, schema)

    def _flush_batch(is_final=False):
        nonlocal batch_count, skipped_oom, output_rows_written, next_output_part
        if not batch_rows:
            return
        batch_count += 1
        _log_progress(
            f"{'Processing final batch' if is_final else 'Encoding batch'} "
            f"{batch_count} ({len(batch_rows)} samples)..."
        )
        encoded_pairs = _encode_batch_with_fallback(
            tokenizer, batch_rows, batch_paths, batch_count
        )
        skipped_oom += len(batch_rows) - len(encoded_pairs)
        out_rows = []
        for row, codes in encoded_pairs:
            out_row = dict(row)
            out_row["audio_codes"] = codes
            out_rows.append(out_row)
        del encoded_pairs

        batch_rows.clear()
        batch_paths.clear()

        if out_rows:
            table = _build_table(out_rows)
            del out_rows
            if output_mode == "file":
                _init_file_writer(table.schema)
                writer.write_table(table)
            else:
                _ensure_dir(output_dir)
                part_path = os.path.join(
                    output_dir, f"part-{next_output_part:05d}.parquet"
                )
                pq.write_table(table, part_path)
                next_output_part += 1
            output_rows_written += len(table)
            del table
        else:
            del out_rows

        _log_progress(f"Batch {batch_count} done. Output rows: {output_rows_written}")

        gc.collect()
        for p in temp_paths:
            try:
                os.remove(p)
            except Exception:
                pass
        temp_paths.clear()

        if output_mode == "dataset":
            _persist_state()

    try:
        _log_progress("Starting processing...")
        if rows_completed > 0:
            _log_progress(f"Skipping first {rows_completed} rows (resume)...")

        for global_idx, row in _iter_input_rows(input_paths, skip_rows=rows_completed):
            # Extract audio bytes
            audio_value = row.get("audio")
            if not isinstance(audio_value, dict):
                skipped_no_audio += 1
                rows_completed = global_idx + 1
                continue
            audio_bytes = audio_value.get("bytes")
            if isinstance(audio_bytes, memoryview):
                audio_bytes = audio_bytes.tobytes()
            if audio_bytes is None:
                _log_progress(f"[warn] Row {global_idx}: missing audio bytes, skipping.")
                skipped_no_audio += 1
                rows_completed = global_idx + 1
                continue

            # Write to temp file for tokenizer
            path_hint = audio_value.get("path")
            encode_path = _write_temp_audio(audio_bytes, path_hint, tmp_dir.name)
            temp_paths.append(encode_path)

            batch_rows.append(row)
            batch_paths.append(encode_path)

            if len(batch_rows) >= args.batch_size:
                _flush_batch()
                rows_completed = global_idx + 1

        if batch_rows:
            _flush_batch(is_final=True)
            rows_completed = global_idx + 1

        completed_ok = True
    finally:
        try:
            if writer is not None:
                writer.close()
                if tmp_output_file:
                    if completed_ok:
                        os.replace(tmp_output_file, output_path)
                    else:
                        try:
                            os.remove(tmp_output_file)
                        except Exception:
                            pass
        finally:
            tmp_dir.cleanup()

    _persist_state()

    if skipped_oom:
        _log_progress(f"[warn] Skipped {skipped_oom} samples due to CUDA OOM.")
    if skipped_no_audio:
        _log_progress(f"[warn] Skipped {skipped_no_audio} rows with missing audio.")
    _log_progress(
        f"Done! Processed {rows_completed} input rows, "
        f"wrote {output_rows_written} output rows to {args.output_parquet}"
    )


if __name__ == "__main__":
    main()
