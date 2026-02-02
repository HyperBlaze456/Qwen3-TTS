import argparse
import io
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
BATCH_INFER_NUM = 32


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
) -> Iterable[Dict]:
    _log_progress(f"Loading dataset: {dataset_name} (split={split}, streaming=True)")
    ds = load_dataset(dataset_name, split=split, streaming=True, cache_dir=cache_dir)
    if "audio" in ds.features:
        try:
            ds = ds.cast_column("audio", Audio(decode=False))
        except Exception:
            pass

    def _row_filter(row: Dict) -> bool:
        text = row.get("transcription")
        if not text or not str(text).strip():
            return False
        if not allowed_langs:
            return True
        lang = _extract_lang(row)
        return lang is not None and lang in allowed_langs

    ds = ds.filter(_row_filter)
    dataset_short = dataset_name.split("/")[-1]
    _log_progress(f"Starting iteration over {dataset_short}...")
    yielded = 0
    for idx, row in enumerate(ds):
        if max_rows is not None and idx >= max_rows:
            _log_progress(f"[{dataset_short}] Reached max_rows limit ({max_rows})")
            break
        text = _clean_dialog_text(row.get("transcription"))
        if not text:
            continue
        lang = _extract_lang(row)
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
        yield item
    _log_progress(f"[{dataset_short}] Finished. Total yielded: {yielded}")


def _iter_arknights_dataset(
    dataset_name: str,
    split: str,
    cache_dir: Optional[str],
    audio_dir: Optional[str],
    download_audio: bool,
    max_rows: Optional[int],
) -> Iterable[Dict]:
    _log_progress(f"Loading dataset: {dataset_name} (split={split})")
    ds = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    total_rows = len(ds)
    _log_progress(f"[arknights] Total rows in dataset: {total_rows}")
    yielded = 0
    skipped = 0
    for idx, row in enumerate(ds):
        if max_rows is not None and idx >= max_rows:
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
        yield item
    _log_progress(f"[arknights] Finished. Total yielded: {yielded}, skipped: {skipped}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_parquet", type=str, required=True)
    parser.add_argument("--schema", type=str, choices=("train", "minimal"), default="train")
    parser.add_argument("--languages", type=str, default="Japanese,Korean")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--audio_dir", type=str, default="data/hf_audio")
    parser.add_argument("--max_per_dataset", type=int, default=None)
    parser.add_argument("--with_audio_codes", action="store_true")
    parser.add_argument("--tokenizer_model_path", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=BATCH_INFER_NUM)
    parser.add_argument("--no_arknights_download", action="store_true")
    parser.add_argument("--genshin_split", type=str, default="train")
    parser.add_argument("--starrail_split", type=str, default="train")
    parser.add_argument("--arknights_split", type=str, default="train")
    parser.add_argument("--log_file", type=str, default=None, help="Path to log file (appends)")
    args = parser.parse_args()

    if args.log_file:
        _init_log_file(args.log_file)

    _log_progress("=" * 50)
    _log_progress("Starting HF SFT data preparation")
    _log_progress(f"  Output: {args.output_parquet}")
    _log_progress(f"  Languages: {args.languages}")
    _log_progress(f"  Max per dataset: {args.max_per_dataset}")
    _log_progress(f"  With audio codes: {args.with_audio_codes}")
    _log_progress(f"  Batch size: {args.batch_size}")
    _log_progress("=" * 50)

    allowed_langs = {
        _normalize_lang(x) for x in args.languages.split(",") if x.strip()
    }
    allowed_langs.discard(None)

    datasets = [
        ("simon3000/genshin-voice", args.genshin_split),
        ("simon3000/starrail-voice", args.starrail_split),
    ]

    def item_iter():
        _log_progress(f"Processing {len(datasets)} simon datasets + arknights")
        for name, split in datasets:
            _log_progress(f"--- Starting dataset: {name} ---")
            for item in _iter_simon_dataset(
                name,
                split,
                allowed_langs,
                args.cache_dir,
                args.max_per_dataset,
            ):
                yield item
            _log_progress(f"--- Completed dataset: {name} ---")

        _log_progress("--- Starting dataset: arknights ---")
        for item in _iter_arknights_dataset(
            "deepghs/arknights_voices_kr",
            args.arknights_split,
            args.cache_dir,
            args.audio_dir,
            not args.no_arknights_download,
            args.max_per_dataset,
        ):
            yield item
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
        tokenizer.model.encoder.config.use_cache = False
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

    def _process_and_write_streaming():
        tmp_dir = tempfile.TemporaryDirectory()
        writer = None
        total_yielded = 0
        batch_count = 0

        try:
            batch_items = []
            batch_paths = []
            temp_paths = []
            _log_progress("Starting streaming parquet generation...")

            for item in item_iter():
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

                if len(batch_items) >= args.batch_size:
                    batch_count += 1
                    rows = []
                    if args.with_audio_codes:
                        _log_progress(f"Encoding batch {batch_count} ({len(batch_items)} samples)...")
                        with torch.no_grad():
                            enc = tokenizer.encode(batch_paths)
                            codes_list = []
                            for code in enc.audio_codes:
                                codes_list.append(code.cpu().tolist())
                            del enc.audio_codes
                        del enc
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        for code, row in zip(codes_list, batch_items):
                            rows.append(_build_output_row(row, code, args.schema))
                    else:
                        for row in batch_items:
                            rows.append(_build_output_row(row, None, args.schema))

                    # Write batch to parquet
                    table = pa.Table.from_pylist(rows)
                    if writer is None:
                        _ensure_dir(os.path.dirname(args.output_parquet) or ".")
                        writer = pq.ParquetWriter(args.output_parquet, table.schema)
                    writer.write_table(table)
                    total_yielded += len(rows)
                    _log_progress(f"Batch {batch_count} done. Total rows written: {total_yielded}")

                    # Clear memory
                    del rows, table
                    batch_items.clear()
                    batch_paths.clear()
                    for p in temp_paths:
                        try:
                            os.remove(p)
                        except Exception:
                            pass
                    temp_paths.clear()

            # Process remaining items
            if batch_items:
                batch_count += 1
                rows = []
                _log_progress(f"Processing final batch {batch_count} ({len(batch_items)} samples)...")
                if args.with_audio_codes:
                    enc = tokenizer.encode(batch_paths)
                    codes_list = [code.cpu().tolist() for code in enc.audio_codes]
                    del enc
                    torch.cuda.empty_cache()
                    for code, row in zip(codes_list, batch_items):
                        rows.append(_build_output_row(row, code, args.schema))
                else:
                    for row in batch_items:
                        rows.append(_build_output_row(row, None, args.schema))

                table = pa.Table.from_pylist(rows)
                if writer is None:
                    _ensure_dir(os.path.dirname(args.output_parquet) or ".")
                    writer = pq.ParquetWriter(args.output_parquet, table.schema)
                writer.write_table(table)
                total_yielded += len(rows)

                for p in temp_paths:
                    try:
                        os.remove(p)
                    except Exception:
                        pass

            _log_progress(f"Parquet generation complete. Total batches: {batch_count}, Total rows: {total_yielded}")
        finally:
            if writer is not None:
                writer.close()
            tmp_dir.cleanup()

        return total_yielded

    _log_progress("Starting streaming write to parquet...")
    total_samples = _process_and_write_streaming()
    _log_progress(f"Done! Wrote {total_samples} samples to {args.output_parquet}")


if __name__ == "__main__":
    main()
