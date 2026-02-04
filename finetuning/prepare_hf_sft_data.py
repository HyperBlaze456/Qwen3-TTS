import argparse
import gc
import io
import json
import os
import re
import sys
import tempfile
import time
import traceback
import urllib.parse
import urllib.request
from typing import Dict, Iterable, Optional, Set, Tuple

import torch

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from datasets import Audio, load_dataset
except Exception as exc:
    raise SystemExit(
        "Missing dependency: datasets. Please install it first, `pip install datasets`."
    ) from exc


_LOG_FILE = None


def _init_log_file(path: str) -> None:
    global _LOG_FILE
    if path:
        _LOG_FILE = open(path, "a", encoding="utf-8")
        _install_excepthook()


def _log_progress(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, file=sys.stderr, flush=True)
    if _LOG_FILE:
        print(line, file=_LOG_FILE, flush=True)


def _log_exception(exc_type, exc_value, exc_tb) -> None:
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
    tb_str = "".join(tb_lines)
    _log_progress(f"EXCEPTION:\n{tb_str}")
    sys.__excepthook__(exc_type, exc_value, exc_tb)


def _install_excepthook() -> None:
    sys.excepthook = _log_exception

TAG_RE = re.compile(r"<[^>]*>")
BRACE_RE = re.compile(r"\{[^}]*\}")
PART_RE = re.compile(r"^part-(\d+)\.parquet$")
BATCH_INFER_NUM = 32


def _is_cuda_oom(exc: BaseException) -> bool:
    oom_type = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
    if isinstance(exc, oom_type):
        return True
    msg = str(exc).lower()
    if "out of memory" in msg and any(token in msg for token in ("cuda", "cublas", "cudnn", "hip", "mps")):
        return True
    if "cublas_status_alloc_failed" in msg or "cudnn_status_alloc_failed" in msg:
        return True
    return False


def _release_exception(exc: BaseException) -> None:
    """Clear traceback references to help free GPU memory after OOM."""
    try:
        tb = exc.__traceback__
        if tb is not None:
            traceback.clear_frames(tb)
        exc.__traceback__ = None
        exc.__context__ = None
        exc.__cause__ = None
    except Exception:
        pass


def _clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    gc.collect()


def _disable_model_cache(model) -> None:
    def _disable(module) -> None:
        cfg = getattr(module, "config", None)
        if cfg is not None and hasattr(cfg, "use_cache"):
            cfg.use_cache = False

    _disable(model)
    for submodule in model.modules():
        _disable(submodule)


def _get_audio_num_samples(path: str, target_sr: Optional[int]) -> Optional[int]:
    if not path or target_sr is None:
        return None
    try:
        import soundfile as sf
    except Exception:
        return None
    try:
        info = sf.info(path)
    except Exception:
        return None
    if info.frames <= 0 or info.samplerate <= 0:
        return None
    return int(info.frames * (float(target_sr) / float(info.samplerate)))


def _get_parquet_row_count(path: str) -> int:
    """Get number of rows in existing parquet file, or 0 if not exists."""
    if not path or not os.path.exists(path):
        return 0
    try:
        import pyarrow.parquet as pq
        metadata = pq.read_metadata(path)
        return metadata.num_rows
    except Exception as exc:
        _log_progress(f"[warn] Failed to read parquet metadata: {exc}")
        return 0


def _load_json(path: str) -> Optional[Dict]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        _log_progress(f"[warn] Failed to read resume state: {exc}")
        return None


def _write_json_atomic(path: str, payload: Dict) -> None:
    if not path:
        return
    tmp_path = f"{path}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp_path, path)
    except Exception as exc:
        _log_progress(f"[warn] Failed to write resume state: {exc}")


def _resolve_output_mode(output_path: str, output_mode: str) -> str:
    if output_mode != "auto":
        return output_mode
    if output_path.endswith(os.sep) or os.path.isdir(output_path):
        return "dataset"
    return "file"


def _scan_parquet_dir(output_dir: str) -> Tuple[Tuple[str, ...], int]:
    if not output_dir or not os.path.isdir(output_dir):
        return tuple(), 0
    parquet_files = []
    max_part = -1
    for name in os.listdir(output_dir):
        match = PART_RE.match(name)
        if not match:
            continue
        parquet_files.append(name)
        max_part = max(max_part, int(match.group(1)))
    parquet_files.sort()
    next_part = max_part + 1 if max_part >= 0 else len(parquet_files)
    return tuple(os.path.join(output_dir, name) for name in parquet_files), next_part


def _normalize_lang(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {"ja", "jpn", "jp", "japanese", "ja-jp", "ja_jp"}:
        return "japanese"
    if s in {"ko", "kor", "kr", "korean", "ko-kr", "ko_kr"}:
        return "korean"
    return s


def _extract_lang(row: Dict) -> Optional[str]:
    for key in ("language", "lang", "locale", "lang_id", "voice_language"):
        if key in row:
            return _normalize_lang(row.get(key))
    return None


def _clean_dialog_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text = str(text)
    text = TAG_RE.sub("", text)
    text = BRACE_RE.sub("", text)
    text = " ".join(text.split())
    return text.strip()


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _audio_array_to_wav_bytes(audio_array, sampling_rate: int) -> bytes:
    try:
        import soundfile as sf
    except Exception as exc:
        raise SystemExit(
            "Need soundfile to serialize audio arrays. Install it with `pip install soundfile`."
        ) from exc
    buf = io.BytesIO()
    sf.write(buf, audio_array, sampling_rate, format="WAV")
    return buf.getvalue()


def _read_bytes(path: str) -> Optional[bytes]:
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return None


def _audio_obj_to_bytes_and_path(
    audio_obj, dataset_short: str, idx: int
) -> Tuple[Optional[bytes], Optional[str]]:
    if audio_obj is None:
        return None, None
    if isinstance(audio_obj, str):
        return _read_bytes(audio_obj), audio_obj
    if isinstance(audio_obj, dict):
        path = audio_obj.get("path") or audio_obj.get("filename")
        audio_bytes = audio_obj.get("bytes")
        if isinstance(audio_bytes, memoryview):
            audio_bytes = audio_bytes.tobytes()
        if audio_bytes is not None:
            return audio_bytes, path
        audio_array = audio_obj.get("array")
        sampling_rate = audio_obj.get("sampling_rate")
        if audio_array is not None and sampling_rate is not None:
            wav_bytes = _audio_array_to_wav_bytes(audio_array, int(sampling_rate))
            return wav_bytes, path or f"{dataset_short}_{idx}.wav"
        if path:
            return _read_bytes(path), path
        return None, None

    encoded = getattr(audio_obj, "_hf_encoded", None)
    if isinstance(encoded, dict):
        path = encoded.get("path")
        audio_bytes = encoded.get("bytes")
        if isinstance(audio_bytes, memoryview):
            audio_bytes = audio_bytes.tobytes()
        if audio_bytes is not None:
            return audio_bytes, path
        if path:
            return _read_bytes(path), path

    meta = getattr(audio_obj, "metadata", None)
    path = getattr(meta, "path", None) if meta is not None else None
    if path:
        audio_bytes = _read_bytes(path)
        if audio_bytes is not None:
            return audio_bytes, path

    try:
        audio_array = audio_obj["array"]
        sampling_rate = audio_obj["sampling_rate"]
        wav_bytes = _audio_array_to_wav_bytes(audio_array, int(sampling_rate))
        return wav_bytes, f"{dataset_short}_{idx}.wav"
    except Exception:
        return None, None


def _make_match_key(text: Optional[str], speaker: Optional[str], language: Optional[str]) -> Optional[str]:
    if not text:
        return None
    t = _clean_dialog_text(text)
    if not t:
        return None
    s = str(speaker).strip() if speaker is not None else ""
    l = _normalize_lang(language) or ""
    return f"{t}||{s}||{l}"


def _read_parquet_tail_rows(
    parquet_paths: Tuple[str, ...], num_rows: int, columns: Tuple[str, ...]
) -> Tuple[Dict, ...]:
    if not parquet_paths or num_rows <= 0:
        return tuple()
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:
        raise SystemExit(
            "Missing dependency: pyarrow. Please install it first, `pip install pyarrow`."
        ) from exc

    try:
        schema = pq.ParquetFile(parquet_paths[0]).schema
        available = set(schema.names)
        columns = tuple([c for c in columns if c in available])
    except Exception:
        columns = tuple(columns)
    if not columns:
        return tuple()

    remaining = num_rows
    collected = []
    for path in reversed(parquet_paths):
        if remaining <= 0:
            break
        try:
            parquet_file = pq.ParquetFile(path)
        except Exception:
            continue
        for row_group in reversed(range(parquet_file.num_row_groups)):
            table = parquet_file.read_row_group(row_group, columns=list(columns))
            if table.num_rows >= remaining:
                table = table.slice(table.num_rows - remaining, remaining)
                collected.append(table)
                remaining = 0
                break
            collected.append(table)
            remaining -= table.num_rows
        if remaining <= 0:
            break

    if not collected:
        return tuple()
    tail_table = pa.concat_tables(list(reversed(collected)))
    return tuple(tail_table.to_pylist())


def _infer_resume_cursor_from_tail(
    output_mode: str,
    output_path: str,
    output_dir: str,
    tail_rows: int,
    sequence_len: int,
    datasets: Tuple[Tuple[str, str], ...],
    arknights_name: str,
    arknights_split: str,
    cache_dir: Optional[str],
    allowed_langs: Set[str],
    log_fn,
) -> Optional[Dict[str, object]]:
    if tail_rows <= 0:
        return None
    if output_mode == "file":
        parquet_paths = (output_path,)
    else:
        parquet_paths, _ = _scan_parquet_dir(output_dir)
    if not parquet_paths:
        return None

    columns = ("source", "text", "transcript", "speaker", "language")
    rows = _read_parquet_tail_rows(parquet_paths, tail_rows, columns)
    if not rows:
        return None

    last_source = None
    for row in reversed(rows):
        source = row.get("source")
        text_val = row.get("text") or row.get("transcript")
        if source and text_val:
            last_source = source
            break
    if not last_source:
        log_fn("[warn] Unable to infer resume dataset: missing source/text in parquet tail.")
        return None

    anchor_keys = []
    for row in rows:
        if row.get("source") != last_source:
            continue
        text_val = row.get("text") or row.get("transcript")
        key = _make_match_key(text_val, row.get("speaker"), row.get("language"))
        if key:
            anchor_keys.append(key)
    if not anchor_keys:
        log_fn("[warn] No valid anchor keys found in parquet tail.")
        return None

    sequence_len = max(1, min(sequence_len, len(anchor_keys)))
    anchor_seq = anchor_keys[-sequence_len:]

    split = None
    dataset_kind = None
    for name, ds_split in datasets:
        if name == last_source:
            split = ds_split
            dataset_kind = "simon"
            break
    if last_source == arknights_name:
        split = arknights_split
        dataset_kind = "arknights"

    if split is None:
        log_fn(f"[warn] Resume dataset not in configured list: {last_source}")
        return None

    log_fn(
        f"Searching dataset for resume anchors: {last_source} split={split} "
        f"(tail_rows={tail_rows}, sequence_len={sequence_len})"
    )

    ds = load_dataset(last_source, split=split, streaming=False, cache_dir=cache_dir)
    if "audio" in ds.features:
        try:
            ds = ds.cast_column("audio", Audio(decode=False))
        except Exception:
            pass

    def _make_key_from_row(row: Dict) -> Optional[str]:
        if dataset_kind == "arknights":
            text_val = row.get("voice_text")
            if not text_val:
                return None
            text_val = " ".join(str(text_val).split()).strip()
            if not text_val:
                return None
            return _make_match_key(text_val, row.get("char_id"), "korean")

        text_val = row.get("transcription")
        if not text_val or not str(text_val).strip():
            return None
        lang = _extract_lang(row)
        if allowed_langs and (lang is None or lang not in allowed_langs):
            return None
        return _make_match_key(text_val, row.get("speaker"), lang)

    match_end_index = None
    seq_pos = 0
    first_anchor = anchor_seq[0]
    for idx, row in enumerate(ds):
        key = _make_key_from_row(row)
        if key is None:
            continue
        if key == anchor_seq[seq_pos]:
            seq_pos += 1
            if seq_pos == len(anchor_seq):
                match_end_index = idx
                seq_pos = 0
        else:
            seq_pos = 1 if key == first_anchor else 0

    if match_end_index is None:
        log_fn("[warn] Failed to locate resume anchors in dataset.")
        return None

    return {"name": last_source, "split": split, "raw_index": int(match_end_index + 1)}


def _merge_parquet_shards(part_files: Tuple[str, ...], output_path: str) -> int:
    if not part_files:
        _log_progress("[warn] No parquet shards found to merge.")
        return 0
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise SystemExit(
            "Missing dependency: pyarrow. Please install it first, `pip install pyarrow`."
        ) from exc

    part_files = tuple(sorted(part_files))
    _ensure_dir(os.path.dirname(output_path) or ".")
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".parquet", dir=os.path.dirname(output_path) or ".")
    os.close(tmp_fd)
    total_rows = 0
    writer = None
    try:
        schema = pq.read_schema(part_files[0])
        writer = pq.ParquetWriter(tmp_path, schema)
        for path in part_files:
            parquet_file = pq.ParquetFile(path)
            for i in range(parquet_file.num_row_groups):
                table = parquet_file.read_row_group(i)
                writer.write_table(table)
                total_rows += table.num_rows
        writer.close()
        writer = None
        os.replace(tmp_path, output_path)
    finally:
        if writer is not None:
            writer.close()
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    return total_rows


def _write_temp_audio(audio_bytes: bytes, path_hint: Optional[str], tmp_dir: str) -> str:
    ext = os.path.splitext(path_hint or "")[1]
    if not ext:
        ext = ".wav"
    fd, tmp_path = tempfile.mkstemp(suffix=ext, dir=tmp_dir)
    with os.fdopen(fd, "wb") as f:
        f.write(audio_bytes)
    return tmp_path


def _build_output_row(item: Dict, audio_codes, schema: str) -> Dict:
    if schema == "minimal":
        out = {
            "audio": item["audio"],
            "transcript": item["text"],
            "speaker": item.get("speaker"),
            "codes": audio_codes,
        }
        return out
    out = dict(item)
    if audio_codes is not None:
        out["audio_codes"] = audio_codes
    return out


def _download_url(url: str, out_path: str) -> Optional[str]:
    try:
        if os.path.exists(out_path):
            return out_path
        _ensure_dir(os.path.dirname(out_path))
        with urllib.request.urlopen(url) as response, open(out_path, "wb") as f:
            f.write(response.read())
        return out_path
    except Exception as exc:
        print(f"[warn] failed to download: {url} ({exc})", file=sys.stderr)
        return None


def _iter_simon_dataset(
    dataset_name: str,
    split: str,
    allowed_langs: Set[str],
    cache_dir: Optional[str],
    max_rows: Optional[int],
    start_index: int = 0,
    streaming: bool = True,
) -> Iterable[Tuple[int, Dict]]:
    _log_progress(
        f"Loading dataset: {dataset_name} (split={split}, streaming={streaming}, start_index={start_index})"
    )
    ds = None
    base_index = 0
    manual_skip = False

    if streaming:
        ds = load_dataset(dataset_name, split=split, streaming=True, cache_dir=cache_dir)
        if start_index > 0:
            try:
                ds = ds.skip(start_index)
                base_index = start_index
            except Exception:
                manual_skip = True
    else:
        if start_index > 0:
            try:
                ds = load_dataset(
                    dataset_name,
                    split=f"{split}[{start_index}:]",
                    streaming=False,
                    cache_dir=cache_dir,
                )
                base_index = start_index
            except Exception:
                ds = load_dataset(dataset_name, split=split, streaming=False, cache_dir=cache_dir)
                try:
                    ds = ds.select(range(start_index, len(ds)))
                    base_index = start_index
                except Exception:
                    manual_skip = True
        else:
            ds = load_dataset(dataset_name, split=split, streaming=False, cache_dir=cache_dir)

    if "audio" in ds.features:
        try:
            ds = ds.cast_column("audio", Audio(decode=False))
        except Exception:
            pass

    dataset_short = dataset_name.split("/")[-1]
    _log_progress(f"Starting iteration over {dataset_short}...")
    yielded = 0

    if manual_skip:
        _log_progress(f"[{dataset_short}] Falling back to manual skip for start_index={start_index}")

    enum_start = base_index if base_index > 0 and not manual_skip else 0
    for idx, row in enumerate(ds, start=enum_start):
        if manual_skip and idx < start_index:
            continue
        if max_rows is not None and yielded >= max_rows:
            _log_progress(f"[{dataset_short}] Reached max_rows limit ({max_rows})")
            break

        text = row.get("transcription")
        if not text or not str(text).strip():
            continue
        lang = _extract_lang(row)
        if allowed_langs and (lang is None or lang not in allowed_langs):
            continue
        text = _clean_dialog_text(text)
        if not text:
            continue

        audio_bytes, audio_path = _audio_obj_to_bytes_and_path(row.get("audio"), dataset_short, idx)
        if audio_bytes is None and (audio_path is None or not os.path.isfile(audio_path)):
            continue
        audio_value = {"bytes": audio_bytes, "path": audio_path}
        speaker = row.get("speaker")
        if speaker is not None:
            speaker = str(speaker)
        item = {
            "audio": audio_value,
            "text": text,
            "source": dataset_name,
            "speaker": speaker,
        }
        if lang:
            item["language"] = lang
        yielded += 1
        if yielded % 500 == 0:
            _log_progress(f"[{dataset_short}] Yielded {yielded} samples (iter idx={idx})")
        yield idx, item
    _log_progress(f"[{dataset_short}] Finished. Total yielded: {yielded}")


def _iter_arknights_dataset(
    dataset_name: str,
    split: str,
    cache_dir: Optional[str],
    audio_dir: Optional[str],
    download_audio: bool,
    max_rows: Optional[int],
    start_index: int = 0,
) -> Iterable[Tuple[int, Dict]]:
    _log_progress(f"Loading dataset: {dataset_name} (split={split}, start_index={start_index})")
    base_index = 0
    manual_skip = False
    ds = None
    if start_index > 0:
        try:
            ds = load_dataset(
                dataset_name, split=f"{split}[{start_index}:]", cache_dir=cache_dir
            )
            base_index = start_index
        except Exception:
            ds = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
            try:
                ds = ds.select(range(start_index, len(ds)))
                base_index = start_index
            except Exception:
                manual_skip = True
    else:
        ds = load_dataset(dataset_name, split=split, cache_dir=cache_dir)

    total_rows = len(ds)
    _log_progress(f"[arknights] Total rows in dataset: {total_rows}")
    yielded = 0
    skipped = 0
    if manual_skip:
        _log_progress(f"[arknights] Falling back to manual skip for start_index={start_index}")
    enum_start = base_index if base_index > 0 and not manual_skip else 0
    for idx, row in enumerate(ds, start=enum_start):
        if manual_skip and idx < start_index:
            continue
        if max_rows is not None and yielded >= max_rows:
            _log_progress(f"[arknights] Reached max_rows limit ({max_rows})")
            break
        text = row.get("voice_text")
        if not text:
            skipped += 1
            continue
        text = " ".join(str(text).split()).strip()
        if not text:
            skipped += 1
            continue
        file_url = row.get("file_url")
        if not file_url:
            skipped += 1
            continue
        audio_path = file_url
        if download_audio and audio_dir:
            parsed = urllib.parse.urlparse(file_url)
            basename = os.path.basename(parsed.path) or f"arknights_{idx}.wav"
            out_path = os.path.join(audio_dir, "arknights", f"{idx}_{basename}")
            downloaded = _download_url(file_url, out_path)
            if not downloaded:
                skipped += 1
                continue
            audio_path = downloaded
        audio_bytes = _read_bytes(audio_path) if os.path.isfile(audio_path) else None
        if audio_bytes is None:
            skipped += 1
            continue
        audio_value = {"bytes": audio_bytes, "path": audio_path}
        speaker = row.get("char_id")
        if speaker is not None:
            speaker = str(speaker)
        item = {
            "audio": audio_value,
            "text": text,
            "source": dataset_name,
            "language": "korean",
            "speaker": speaker,
        }
        yielded += 1
        if yielded % 500 == 0:
            _log_progress(f"[arknights] Yielded {yielded} / processed {idx+1}/{total_rows} (skipped {skipped})")
        yield idx, item
    _log_progress(f"[arknights] Finished. Total yielded: {yielded}, skipped: {skipped}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_parquet", type=str, required=True)
    parser.add_argument(
        "--output_mode",
        type=str,
        choices=("auto", "file", "dataset"),
        default="auto",
        help=(
            "Output mode: file writes a single parquet (resume rewrites), "
            "dataset writes sharded parquet files in a directory for true resume, "
            "auto uses dataset if output_parquet is a directory."
        ),
    )
    parser.add_argument(
        "--merge_output_parquet",
        type=str,
        default=None,
        help=(
            "If set and output_mode=dataset, merge all part-*.parquet shards into a single "
            "parquet at this path after processing completes."
        ),
    )
    parser.add_argument("--schema", type=str, choices=("train", "minimal"), default="train")
    parser.add_argument("--languages", type=str, default="Japanese,Korean")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--audio_dir", type=str, default="data/hf_audio")
    parser.add_argument("--max_per_dataset", type=int, default=None)
    parser.add_argument(
        "--no_streaming",
        action="store_true",
        help="Disable streaming for simon datasets (genshin/starrail) to enable fast indexed resume.",
    )
    parser.add_argument("--with_audio_codes", action="store_true")
    parser.add_argument("--tokenizer_model_path", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=BATCH_INFER_NUM)
    parser.add_argument(
        "--max_audio_seconds",
        type=float,
        default=None,
        help="Skip samples longer than this duration (seconds). Default: no limit.",
    )
    parser.add_argument(
        "--max_length_ratio",
        type=float,
        default=4.0,
        help="Flush current batch if next sample is this multiple longer than current max length.",
    )
    parser.add_argument("--no_arknights_download", action="store_true")
    parser.add_argument("--genshin_split", type=str, default="train")
    parser.add_argument("--starrail_split", type=str, default="train")
    parser.add_argument("--arknights_split", type=str, default="train")
    parser.add_argument(
        "--resume_search_tail_rows",
        type=int,
        default=200,
        help="If resume cursor is missing, scan this many tail rows from existing parquet to infer resume point.",
    )
    parser.add_argument(
        "--resume_search_sequence",
        type=int,
        default=5,
        help="Number of tail rows (same dataset) to use as an ordered sequence when searching resume point.",
    )
    parser.add_argument("--log_file", type=str, default=None, help="Path to log file (appends)")
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Ignore existing parquet file and start fresh.",
    )
    args = parser.parse_args()

    if args.log_file:
        _init_log_file(args.log_file)

    _log_progress("=" * 50)
    _log_progress("Starting HF SFT data preparation")
    _log_progress(f"  Output: {args.output_parquet}")
    _log_progress(f"  Output mode: {args.output_mode}")
    if args.merge_output_parquet:
        _log_progress(f"  Merge output parquet: {args.merge_output_parquet}")
    _log_progress(f"  Languages: {args.languages}")
    _log_progress(f"  Max per dataset: {args.max_per_dataset}")
    _log_progress(f"  With audio codes: {args.with_audio_codes}")
    _log_progress(f"  Batch size: {args.batch_size}")
    _log_progress(f"  Simon streaming: {not args.no_streaming}")
    _log_progress(f"  Max audio seconds: {args.max_audio_seconds}")
    _log_progress(f"  Max length ratio: {args.max_length_ratio}")
    _log_progress(f"  Resume search tail rows: {args.resume_search_tail_rows}")
    _log_progress(f"  Resume search sequence: {args.resume_search_sequence}")
    _log_progress("=" * 50)

    allowed_langs = {
        _normalize_lang(x) for x in args.languages.split(",") if x.strip()
    }
    allowed_langs.discard(None)

    datasets = [
        ("simon3000/genshin-voice", args.genshin_split),
        ("simon3000/starrail-voice", args.starrail_split),
    ]
    datasets_tuple = tuple(datasets)

    def item_iter_with_index(resume_cursor=None, resume_from_index=0, use_streaming: bool = True):
        """Yields (global_index, dataset_name, dataset_split, raw_index, item)."""
        global_index = resume_from_index
        _log_progress(f"Processing {len(datasets)} simon datasets + arknights")

        cursor_key = None
        cursor_raw_index = 0
        cursor_reached = True
        if resume_cursor:
            cursor_key = (resume_cursor.get("name"), resume_cursor.get("split"))
            cursor_raw_index = max(0, int(resume_cursor.get("raw_index", 0)))
            cursor_reached = False
            _log_progress(
                f"Resume cursor: dataset={cursor_key[0]} split={cursor_key[1]} raw_index={cursor_raw_index}"
            )

        valid_keys = [(name, split) for name, split in datasets] + [
            ("deepghs/arknights_voices_kr", args.arknights_split)
        ]
        if cursor_key and cursor_key not in valid_keys:
            _log_progress(f"[warn] Resume cursor dataset not found: {cursor_key}. Ignoring cursor.")
            cursor_key = None
            cursor_reached = True

        for name, split in datasets:
            dataset_key = (name, split)
            start_index = 0
            if cursor_key and not cursor_reached:
                if dataset_key == cursor_key:
                    start_index = cursor_raw_index
                    cursor_reached = True
                    _log_progress(
                        f"--- Resuming dataset: {name} (split={split}) from raw index {start_index} ---"
                    )
                else:
                    _log_progress(f"--- Skipping dataset: {name} (resume cursor ahead) ---")
                    continue

            if cursor_key is None or cursor_reached:
                _log_progress(f"--- Starting dataset: {name} ---")
            for raw_index, item in _iter_simon_dataset(
                name,
                split,
                allowed_langs,
                args.cache_dir,
                args.max_per_dataset,
                start_index=start_index,
                streaming=use_streaming,
            ):
                yield global_index, name, split, raw_index, item
                global_index += 1
            _log_progress(f"--- Completed dataset: {name} ---")

        arknights_name = "deepghs/arknights_voices_kr"
        arknights_split = args.arknights_split
        dataset_key = (arknights_name, arknights_split)
        start_index = 0
        if cursor_key and not cursor_reached:
            if dataset_key == cursor_key:
                start_index = cursor_raw_index
                cursor_reached = True
                _log_progress(
                    f"--- Resuming dataset: {arknights_name} (split={arknights_split}) "
                    f"from raw index {start_index} ---"
                )
            else:
                _log_progress(f"--- Skipping dataset: {arknights_name} (resume cursor ahead) ---")
                return

        _log_progress("--- Starting dataset: arknights ---")
        for raw_index, item in _iter_arknights_dataset(
            arknights_name,
            arknights_split,
            args.cache_dir,
            args.audio_dir,
            not args.no_arknights_download,
            args.max_per_dataset,
            start_index=start_index,
        ):
            yield global_index, arknights_name, arknights_split, raw_index, item
            global_index += 1
        _log_progress("--- Completed dataset: arknights ---")

    tokenizer = None
    if args.with_audio_codes:
        _log_progress(f"Loading tokenizer from {args.tokenizer_model_path}...")
        try:
            from qwen_tts import Qwen3TTSTokenizer
        except Exception as exc:
            raise SystemExit(
                "Missing dependency: qwen_tts. Install qwen-tts or run without --with_audio_codes."
            ) from exc
        tokenizer = Qwen3TTSTokenizer.from_pretrained(
            args.tokenizer_model_path,
            device_map=args.device,
        )
        # Disable KV caching in encoder transformer - not needed for encoding
        # and causes massive memory accumulation via DynamicCache
        _disable_model_cache(tokenizer.model)
        _log_progress("Tokenizer loaded successfully (encoder cache disabled).")
    else:
        _log_progress("Skipping tokenizer loading (--with_audio_codes not set)")

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:
        raise SystemExit(
            "Missing dependency: pyarrow. Please install it first, `pip install pyarrow`."
        ) from exc

    def _encode_batch_with_fallback(batch_items, batch_paths, batch_idx):
        def _encode_chunk(chunk_items, chunk_paths, depth):
            enc = None
            try:
                with torch.no_grad():
                    enc = tokenizer.encode(chunk_paths)
                    codes = []
                    # .contiguous()로 view→독립 텐서 변환 후 CPU로 복사
                    # 원본 참조를 끊어 GPU 메모리 즉시 해제 가능하게 함
                    for i in range(len(enc.audio_codes)):
                        codes.append(enc.audio_codes[i].contiguous().cpu().tolist())
                        enc.audio_codes[i] = None
                del enc
                enc = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                return list(zip(chunk_items, codes))
            except Exception as exc:
                # OOM 여부와 관계없이 enc가 있으면 정리
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
                    # Clear exception traceback to avoid holding GPU tensors alive during retry.
                    _release_exception(exc)
                    exc = None
                    gc.collect()
                    _clear_cuda_cache()

                if not is_oom:
                    raise
                _log_progress(
                    f"[warn] CUDA OOM in batch {batch_idx} (size={len(chunk_items)}, depth={depth}). "
                    "Splitting batch to reduce max sequence length."
                )
                if len(chunk_items) == 1:
                    _log_progress(f"[warn] Skipping sample due to CUDA OOM: {chunk_paths[0]}")
                    return []
                mid = len(chunk_items) // 2
                left = _encode_chunk(chunk_items[:mid], chunk_paths[:mid], depth + 1)
                # left 처리 후 right 처리 전에 GPU 플러시
                gc.collect()
                _clear_cuda_cache()
                right = _encode_chunk(chunk_items[mid:], chunk_paths[mid:], depth + 1)
                return left + right

        return _encode_chunk(batch_items, batch_paths, 0)

    def _process_and_write_streaming():
        tmp_dir = tempfile.TemporaryDirectory()
        writer = None
        tmp_output_path = None

        output_mode = _resolve_output_mode(args.output_parquet, args.output_mode)
        output_path = args.output_parquet
        output_dir = args.output_parquet
        merge_output_path = args.merge_output_parquet
        state_path = (
            f"{output_path}.resume.json"
            if output_mode == "file"
            else os.path.join(output_dir, "_resume_state.json")
        )
        _log_progress(f"Resolved output mode: {output_mode}")

        resume_state = None
        if not args.no_resume:
            resume_state = _load_json(state_path)

        resume_cursor = None
        if resume_state:
            cursor_state = resume_state.get("cursor") or resume_state.get("dataset_cursor")
            if isinstance(cursor_state, dict):
                if cursor_state.get("name") and cursor_state.get("split") is not None:
                    resume_cursor = {
                        "name": cursor_state.get("name"),
                        "split": cursor_state.get("split"),
                        "raw_index": int(cursor_state.get("raw_index", 0)),
                    }

        existing_schema = None
        existing_rows = 0
        resume_from_index = 0
        next_part = 0

        def _read_schema(path: str):
            try:
                return pq.read_schema(path)
            except Exception as exc:
                _log_progress(f"[warn] Failed to read parquet schema: {exc}")
                return None

        def _copy_existing_parquet(src_path: str, writer_obj) -> None:
            parquet_file = pq.ParquetFile(src_path)
            for i in range(parquet_file.num_row_groups):
                writer_obj.write_table(parquet_file.read_row_group(i))

        def _persist_state(
            next_index: int,
            rows_written: int,
            next_part_idx: int,
            cursor: Optional[Dict[str, object]] = None,
        ) -> None:
            payload = {
                "mode": output_mode,
                "next_index": int(next_index),
                "rows_written": int(rows_written),
                "next_part": int(next_part_idx),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            if cursor:
                payload["cursor"] = cursor
            _ensure_dir(os.path.dirname(state_path) or ".")
            _write_json_atomic(state_path, payload)

        if args.no_resume:
            resume_state = None
            resume_cursor = None

        if output_mode == "dataset":
            _ensure_dir(output_dir)
            part_files, next_part_scan = _scan_parquet_dir(output_dir)
            if args.no_resume and part_files:
                _log_progress("[warn] --no_resume set: removing existing parquet shards in output directory.")
                for path in part_files:
                    try:
                        os.remove(path)
                    except Exception:
                        pass
                part_files = tuple()
                next_part_scan = 0

            if part_files:
                existing_rows = sum(_get_parquet_row_count(p) for p in part_files)
                existing_schema = _read_schema(part_files[0])
                next_part = next_part_scan
                _log_progress(
                    f"Found existing parquet dataset with {len(part_files)} files, {existing_rows} rows."
                )

            if resume_state and existing_rows > 0:
                state_rows = int(resume_state.get("rows_written", existing_rows))
                state_next = int(resume_state.get("next_index", existing_rows))
                if state_rows != existing_rows:
                    _log_progress(
                        f"[warn] Resume state rows ({state_rows}) != existing rows ({existing_rows}); "
                        "using filesystem rows."
                    )
                    resume_cursor = None
                resume_from_index = max(state_next, existing_rows)
                if resume_cursor:
                    _log_progress(
                        f"Using resume cursor for fast pickup: {resume_cursor.get('name')} "
                        f"(split={resume_cursor.get('split')}) raw_index={resume_cursor.get('raw_index')}"
                    )
            elif resume_state and existing_rows == 0:
                _log_progress("[warn] Resume state found but no parquet files exist; ignoring state.")
                resume_cursor = None
                resume_from_index = existing_rows
            else:
                resume_from_index = existing_rows

            total_yielded = existing_rows
        else:
            if args.no_resume:
                if os.path.exists(output_path):
                    _log_progress("[warn] --no_resume set: existing parquet will be overwritten.")
                existing_rows = 0
            else:
                if os.path.exists(output_path):
                    existing_rows = _get_parquet_row_count(output_path)
                    existing_schema = _read_schema(output_path)
                    if existing_rows > 0:
                        _log_progress(f"Found existing parquet with {existing_rows} rows. Will resume...")

            if resume_state and existing_rows > 0:
                state_rows = int(resume_state.get("rows_written", existing_rows))
                state_next = int(resume_state.get("next_index", existing_rows))
                if state_rows != existing_rows:
                    _log_progress(
                        f"[warn] Resume state rows ({state_rows}) != file rows ({existing_rows}); "
                        "ignoring state."
                    )
                    resume_cursor = None
                    resume_from_index = existing_rows
                else:
                    resume_from_index = max(state_next, existing_rows)
                if resume_cursor:
                    _log_progress(
                        f"Using resume cursor for fast pickup: {resume_cursor.get('name')} "
                        f"(split={resume_cursor.get('split')}) raw_index={resume_cursor.get('raw_index')}"
                    )
            else:
                resume_from_index = existing_rows

            if resume_from_index > 0:
                _log_progress(
                    "[warn] Single-file resume rewrites the parquet; use --output_mode dataset for true resume."
                )

            total_yielded = existing_rows

        force_non_streaming = False
        if resume_cursor is None and resume_from_index > 0 and args.resume_search_tail_rows > 0:
            inferred_cursor = _infer_resume_cursor_from_tail(
                output_mode=output_mode,
                output_path=output_path,
                output_dir=output_dir,
                tail_rows=args.resume_search_tail_rows,
                sequence_len=args.resume_search_sequence,
                datasets=datasets_tuple,
                arknights_name="deepghs/arknights_voices_kr",
                arknights_split=args.arknights_split,
                cache_dir=args.cache_dir,
                allowed_langs=allowed_langs,
                log_fn=_log_progress,
            )
            if inferred_cursor:
                resume_cursor = inferred_cursor
                force_non_streaming = True
                _log_progress(
                    f"Inferred resume cursor: {resume_cursor.get('name')} "
                    f"(split={resume_cursor.get('split')}) raw_index={resume_cursor.get('raw_index')}"
                )
                _log_progress("Fast resume enabled: forcing non-streaming dataset load.")
            else:
                _log_progress("[warn] Resume cursor inference failed; falling back to slow resume.")

        use_streaming = not args.no_streaming and not force_non_streaming

        valid_cursor_keys = set(datasets_tuple)
        valid_cursor_keys.add(("deepghs/arknights_voices_kr", args.arknights_split))
        if resume_cursor:
            key = (resume_cursor.get("name"), resume_cursor.get("split"))
            if key not in valid_cursor_keys:
                _log_progress(f"[warn] Resume cursor dataset not found: {key}. Ignoring cursor.")
                resume_cursor = None

        batch_count = 0
        skipped_oom = 0
        skipped_for_resume = 0
        pending_next_index = resume_from_index
        pending_cursor = resume_cursor.copy() if resume_cursor else None
        last_seen_index = -1
        last_seen_dataset_key = None
        last_seen_raw_index = None
        use_cursor_resume = resume_cursor is not None

        completed_ok = False
        try:
            batch_items = []
            batch_paths = []
            batch_indices = []
            batch_dataset_key = None
            batch_last_raw_index = None
            temp_paths = []

            if resume_from_index > 0 and not use_cursor_resume:
                _log_progress(f"Skipping first {resume_from_index} items...")

            _log_progress("Starting streaming parquet generation...")

            def _build_table(rows):
                nonlocal existing_schema
                if existing_schema is not None:
                    try:
                        return pa.Table.from_pylist(rows, schema=existing_schema)
                    except Exception as exc:
                        raise SystemExit(
                            f"Schema mismatch with existing parquet. "
                            f"Use --no_resume or consistent settings. Details: {exc}"
                        ) from exc
                table = pa.Table.from_pylist(rows)
                existing_schema = table.schema
                return table

            def _init_file_writer(table_schema):
                nonlocal writer, tmp_output_path, existing_schema
                if writer is not None:
                    return
                _ensure_dir(os.path.dirname(output_path) or ".")
                fd, tmp_path = tempfile.mkstemp(suffix=".parquet", dir=os.path.dirname(output_path) or ".")
                os.close(fd)
                tmp_output_path = tmp_path
                writer = pq.ParquetWriter(tmp_output_path, table_schema)
                if existing_rows > 0 and os.path.exists(output_path):
                    _copy_existing_parquet(output_path, writer)

            def _flush_batch(is_final: bool = False):
                nonlocal batch_count, skipped_oom, total_yielded, pending_next_index, next_part
                nonlocal pending_cursor, batch_dataset_key, batch_last_raw_index
                if not batch_items:
                    return
                batch_count += 1
                rows = []
                if args.with_audio_codes:
                    _log_progress(
                        f"{'Processing final batch' if is_final else 'Encoding batch'} "
                        f"{batch_count} ({len(batch_items)} samples)..."
                    )
                    encoded_pairs = _encode_batch_with_fallback(batch_items, batch_paths, batch_count)
                    skipped_oom += len(batch_items) - len(encoded_pairs)
                    for row, code in encoded_pairs:
                        rows.append(_build_output_row(row, code, args.schema))
                    del encoded_pairs
                else:
                    if is_final:
                        _log_progress(f"Processing final batch {batch_count} ({len(batch_items)} samples)...")
                    for row in batch_items:
                        rows.append(_build_output_row(row, None, args.schema))

                batch_last_index = batch_indices[-1] if batch_indices else None
                batch_items.clear()
                batch_paths.clear()
                batch_indices.clear()

                if rows:
                    table = _build_table(rows)
                    del rows
                    if output_mode == "file":
                        _init_file_writer(table.schema)
                        writer.write_table(table)
                    else:
                        _ensure_dir(output_dir)
                        part_path = os.path.join(output_dir, f"part-{next_part:05d}.parquet")
                        pq.write_table(table, part_path)
                        next_part += 1
                    total_yielded += len(table)
                    del table
                else:
                    del rows

                if batch_last_index is not None:
                    pending_next_index = max(pending_next_index, batch_last_index + 1)
                if batch_dataset_key is not None and batch_last_raw_index is not None:
                    pending_cursor = {
                        "name": batch_dataset_key[0],
                        "split": batch_dataset_key[1],
                        "raw_index": int(batch_last_raw_index + 1),
                    }

                _log_progress(f"Batch {batch_count} done. Total rows: {total_yielded}")

                gc.collect()
                for p in temp_paths:
                    try:
                        os.remove(p)
                    except Exception:
                        pass
                temp_paths.clear()

                if output_mode == "dataset":
                    _persist_state(pending_next_index, total_yielded, next_part, cursor=pending_cursor)

            for global_index, dataset_name, dataset_split, raw_index, item in item_iter_with_index(
                resume_cursor=resume_cursor if use_cursor_resume else None,
                resume_from_index=resume_from_index,
                use_streaming=use_streaming,
            ):
                last_seen_index = global_index
                last_seen_dataset_key = (dataset_name, dataset_split)
                last_seen_raw_index = raw_index
                if not use_cursor_resume and global_index < resume_from_index:
                    skipped_for_resume += 1
                    if skipped_for_resume % 1000 == 0:
                        _log_progress(f"  Skipped {skipped_for_resume} items...")
                    continue

                if not use_cursor_resume and skipped_for_resume > 0 and global_index == resume_from_index:
                    _log_progress(f"Resuming from index {global_index} (skipped {skipped_for_resume} items)")

                dataset_key = (dataset_name, dataset_split)
                if batch_dataset_key is None:
                    batch_dataset_key = dataset_key
                elif dataset_key != batch_dataset_key:
                    _flush_batch()
                    batch_dataset_key = dataset_key
                    batch_last_raw_index = None

                audio_value = item.get("audio")
                if not isinstance(audio_value, dict):
                    continue
                audio_bytes = audio_value.get("bytes")
                if isinstance(audio_bytes, memoryview):
                    audio_bytes = audio_bytes.tobytes()
                if audio_bytes is None:
                    continue
                path_hint = audio_value.get("path")
                if path_hint and os.path.isfile(path_hint):
                    encode_path = path_hint
                else:
                    encode_path = _write_temp_audio(audio_bytes, path_hint, tmp_dir.name)
                    temp_paths.append(encode_path)
                audio_value["bytes"] = audio_bytes
                item["audio"] = audio_value
                batch_items.append(item)
                batch_paths.append(encode_path)
                batch_indices.append(global_index)
                batch_last_raw_index = raw_index

                if len(batch_items) >= args.batch_size:
                    _flush_batch()

            if batch_items:
                _flush_batch(is_final=True)

            if last_seen_index >= 0:
                pending_next_index = max(pending_next_index, last_seen_index + 1)
            if last_seen_dataset_key is not None and last_seen_raw_index is not None:
                pending_cursor = {
                    "name": last_seen_dataset_key[0],
                    "split": last_seen_dataset_key[1],
                    "raw_index": int(last_seen_raw_index + 1),
                }

            if output_mode == "dataset":
                if batch_count == 0 and total_yielded == existing_rows:
                    _log_progress("No new data to add. Existing parquet unchanged.")
            else:
                if writer is None and existing_rows > 0:
                    _log_progress("No new data to add. Existing parquet unchanged.")

            if skipped_oom:
                _log_progress(f"[warn] Skipped {skipped_oom} samples due to CUDA OOM.")
            _log_progress(f"Parquet generation complete. Total batches: {batch_count}, Total rows: {total_yielded}")
            completed_ok = True
        finally:
            try:
                if writer is not None:
                    writer.close()
                    if tmp_output_path:
                        if completed_ok:
                            os.replace(tmp_output_path, output_path)
                        else:
                            try:
                                os.remove(tmp_output_path)
                            except Exception:
                                pass
            finally:
                tmp_dir.cleanup()

        if output_mode == "file" and completed_ok:
            _persist_state(pending_next_index, total_yielded, 0, cursor=pending_cursor)
        elif output_mode == "dataset" and total_yielded == existing_rows and not batch_count:
            _persist_state(pending_next_index, total_yielded, next_part, cursor=pending_cursor)

        if output_mode == "dataset" and completed_ok and merge_output_path:
            part_files, _ = _scan_parquet_dir(output_dir)
            if part_files:
                _log_progress(f"Merging {len(part_files)} parquet shards into {merge_output_path}...")
                merged_rows = _merge_parquet_shards(part_files, merge_output_path)
                _log_progress(f"Merged parquet rows: {merged_rows}")
            else:
                _log_progress("[warn] No parquet shards found for merge; skipping merge.")

        return total_yielded

    _log_progress("Starting streaming write to parquet...")
    total_samples = _process_and_write_streaming()
    _log_progress(f"Done! Wrote {total_samples} samples to {args.output_parquet}")


if __name__ == "__main__":
    main()
