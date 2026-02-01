import argparse
import io
import json
import os
import re
import sys
import tempfile
import urllib.parse
import urllib.request
from typing import Dict, Iterable, Optional, Set, Tuple

try:
    from datasets import Audio, load_dataset
except Exception as exc:
    raise SystemExit(
        "Missing dependency: datasets. Please install it first, `pip install datasets`."
    ) from exc

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


def _write_audio_bytes(audio_bytes: bytes, out_path: str) -> str:
    _ensure_dir(os.path.dirname(out_path))
    with open(out_path, "wb") as f:
        f.write(audio_bytes)
    return out_path


def _write_audio_array(audio_array, sampling_rate: int, out_path: str) -> str:
    try:
        import soundfile as sf
    except Exception as exc:
        raise SystemExit(
            "Need soundfile to write audio arrays. Install it with `pip install soundfile`."
        ) from exc
    _ensure_dir(os.path.dirname(out_path))
    sf.write(out_path, audio_array, sampling_rate)
    return out_path


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
            "codes": audio_codes,
        }
        return out
    out = dict(item)
    if audio_codes is not None:
        out["audio_codes"] = audio_codes
    return out


def _resolve_audio_path(
    audio_obj, dataset_short: str, idx: int, audio_dir: Optional[str]
) -> Optional[str]:
    if isinstance(audio_obj, str):
        return audio_obj
    if isinstance(audio_obj, dict):
        path = audio_obj.get("path") or audio_obj.get("filename")
        if path:
            return path
        audio_bytes = audio_obj.get("bytes")
        if audio_bytes is not None and audio_dir:
            out_path = os.path.join(audio_dir, dataset_short, f"{idx}.wav")
            return _write_audio_bytes(audio_bytes, out_path)
        audio_array = audio_obj.get("array")
        sampling_rate = audio_obj.get("sampling_rate")
        if audio_array is not None and sampling_rate is not None and audio_dir:
            out_path = os.path.join(audio_dir, dataset_short, f"{idx}.wav")
            return _write_audio_array(audio_array, sampling_rate, out_path)
    return None


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
    audio_dir: Optional[str],
    max_rows: Optional[int],
    return_audio_obj: bool,
) -> Iterable[Dict]:
    ds = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    if "audio" in ds.features:
        try:
            ds = ds.cast_column("audio", Audio(decode=False))
        except Exception:
            pass
    for idx, row in enumerate(ds):
        if max_rows is not None and idx >= max_rows:
            break
        text = _clean_dialog_text(row.get("transcription"))
        if not text:
            continue
        lang = _extract_lang(row)
        if allowed_langs and (lang is None or lang not in allowed_langs):
            continue
        dataset_short = dataset_name.split("/")[-1]
        if return_audio_obj:
            audio_bytes, audio_path = _audio_obj_to_bytes_and_path(row.get("audio"), dataset_short, idx)
            if audio_bytes is None and (audio_path is None or not os.path.isfile(audio_path)):
                continue
            audio_value = {"bytes": audio_bytes, "path": audio_path}
        else:
            audio_path = _resolve_audio_path(row.get("audio"), dataset_short, idx, audio_dir)
            if not audio_path:
                continue
            audio_value = audio_path
        item = {
            "audio": audio_value,
            "text": text,
            "source": dataset_name,
        }
        if lang:
            item["language"] = lang
        yield item


def _iter_arknights_dataset(
    dataset_name: str,
    split: str,
    cache_dir: Optional[str],
    audio_dir: Optional[str],
    download_audio: bool,
    max_rows: Optional[int],
    return_audio_obj: bool,
) -> Iterable[Dict]:
    ds = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    for idx, row in enumerate(ds):
        if max_rows is not None and idx >= max_rows:
            break
        text = row.get("voice_text")
        if not text:
            continue
        text = " ".join(str(text).split()).strip()
        if not text:
            continue
        file_url = row.get("file_url")
        if not file_url:
            continue
        audio_path = file_url
        if download_audio and audio_dir:
            parsed = urllib.parse.urlparse(file_url)
            basename = os.path.basename(parsed.path) or f"arknights_{idx}.wav"
            out_path = os.path.join(audio_dir, "arknights", f"{idx}_{basename}")
            downloaded = _download_url(file_url, out_path)
            if not downloaded:
                continue
            audio_path = downloaded
        if return_audio_obj:
            audio_bytes = _read_bytes(audio_path) if os.path.isfile(audio_path) else None
            if audio_bytes is None:
                continue
            audio_value = {"bytes": audio_bytes, "path": audio_path}
        else:
            audio_value = audio_path
        item = {
            "audio": audio_value,
            "text": text,
            "source": dataset_name,
            "language": "korean",
        }
        yield item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_jsonl", type=str, default=None)
    parser.add_argument("--output_parquet", type=str, default=None)
    parser.add_argument("--schema", type=str, choices=("train", "minimal"), default="train")
    parser.add_argument("--ref_audio", type=str, default=None)
    parser.add_argument("--languages", type=str, default="Japanese,Korean")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--audio_dir", type=str, default="data/hf_audio")
    parser.add_argument("--max_per_dataset", type=int, default=None)
    parser.add_argument("--with_audio_codes", action="store_true")
    parser.add_argument("--tokenizer_model_path", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=BATCH_INFER_NUM)
    parser.add_argument("--no_arknights_download", action="store_true")
    parser.add_argument("--genshin_split", type=str, default="train")
    parser.add_argument("--starrail_split", type=str, default="train")
    parser.add_argument("--arknights_split", type=str, default="train")
    args = parser.parse_args()

    if not args.output_jsonl and not args.output_parquet:
        parser.error("At least one of --output_jsonl or --output_parquet must be set.")

    allowed_langs = {
        _normalize_lang(x) for x in args.languages.split(",") if x.strip()
    }
    allowed_langs.discard(None)

    datasets = [
        ("simon3000/genshin-voice", args.genshin_split),
        ("simon3000/starrail-voice", args.starrail_split),
    ]

    def item_iter(return_audio_obj: bool):
        for name, split in datasets:
            for item in _iter_simon_dataset(
                name,
                split,
                allowed_langs,
                args.cache_dir,
                args.audio_dir,
                args.max_per_dataset,
                return_audio_obj,
            ):
                item["ref_audio"] = args.ref_audio or item["audio"]
                yield item

        for item in _iter_arknights_dataset(
            "deepghs/arknights_voices_kr",
            args.arknights_split,
            args.cache_dir,
            args.audio_dir,
            not args.no_arknights_download,
            args.max_per_dataset,
            return_audio_obj,
        ):
            item["ref_audio"] = args.ref_audio or item["audio"]
            yield item

    tokenizer = None
    if args.with_audio_codes:
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

    if args.output_jsonl:
        total = 0
        _ensure_dir(os.path.dirname(args.output_jsonl) or ".")
        with open(args.output_jsonl, "w", encoding="utf-8") as f:
            if not args.with_audio_codes:
                for item in item_iter(return_audio_obj=False):
                    row = _build_output_row(item, None, args.schema)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    total += 1
            else:
                batch_items = []
                batch_audios = []
                for item in item_iter(return_audio_obj=False):
                    batch_items.append(item)
                    batch_audios.append(item["audio"])
                    if len(batch_items) >= args.batch_size:
                        enc = tokenizer.encode(batch_audios)
                        for code, row in zip(enc.audio_codes, batch_items):
                            out_row = _build_output_row(row, code.cpu().tolist(), args.schema)
                            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                            total += 1
                        batch_items.clear()
                        batch_audios.clear()

                if batch_items:
                    enc = tokenizer.encode(batch_audios)
                    for code, row in zip(enc.audio_codes, batch_items):
                        out_row = _build_output_row(row, code.cpu().tolist(), args.schema)
                        f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                        total += 1

        print(f"Wrote {total} samples to {args.output_jsonl}")

    if args.output_parquet:
        try:
            from datasets import Dataset
        except Exception as exc:
            raise SystemExit(
                "Missing dependency: datasets. Please install it first, `pip install datasets`."
            ) from exc

        def _parquet_rows():
            tmp_dir = tempfile.TemporaryDirectory()
            try:
                batch_items = []
                batch_paths = []
                temp_paths = []
                for item in item_iter(return_audio_obj=True):
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
                        if args.with_audio_codes:
                            enc = tokenizer.encode(batch_paths)
                            for code, row in zip(enc.audio_codes, batch_items):
                                yield _build_output_row(row, code.cpu().tolist(), args.schema)
                        else:
                            for row in batch_items:
                                yield _build_output_row(row, None, args.schema)
                        batch_items.clear()
                        batch_paths.clear()
                        for p in temp_paths:
                            try:
                                os.remove(p)
                            except Exception:
                                pass
                        temp_paths.clear()

                if batch_items:
                    if args.with_audio_codes:
                        enc = tokenizer.encode(batch_paths)
                        for code, row in zip(enc.audio_codes, batch_items):
                            yield _build_output_row(row, code.cpu().tolist(), args.schema)
                    else:
                        for row in batch_items:
                            yield _build_output_row(row, None, args.schema)
                    for p in temp_paths:
                        try:
                            os.remove(p)
                        except Exception:
                            pass
            finally:
                tmp_dir.cleanup()

        _ensure_dir(os.path.dirname(args.output_parquet) or ".")
        ds = Dataset.from_generator(_parquet_rows, cache_dir=args.cache_dir)
        ds = ds.cast_column("audio", Audio(decode=False))
        ds.to_parquet(args.output_parquet)
        print(f"Wrote {len(ds)} samples to {args.output_parquet}")


if __name__ == "__main__":
    main()
