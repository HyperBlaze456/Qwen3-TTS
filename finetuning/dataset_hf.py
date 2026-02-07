from typing import Any, List, Optional, Tuple, Union
import io

import librosa
import numpy as np
import soundfile as sf
import torch
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
from torch.utils.data import Dataset

AudioLike = Union[
    str,
    np.ndarray,
    Tuple[np.ndarray, int],
    bytes,
    dict,
]

MaybeList = Union[Any, List[Any]]


class HFTTSDataset(Dataset):
    """
    Dataset for TTS fine-tuning using HuggingFace datasets.

    Expected dataset columns:
        - audio: dict with 'bytes' and/or 'path', or Audio feature
        - text: transcription text
        - audio_codes: pre-computed audio codes (list of 16 lists)
        - speaker: (optional) speaker ID
        - language: (optional) language tag
    """

    def __init__(
        self,
        hf_dataset,
        processor,
        config: Qwen3TTSConfig,
        lag_num: int = -1,
        target_sr: int = 24000,
        audio_column: str = "audio",
        text_column: str = "text",
        codes_column: str = "audio_codes",
    ):
        self.dataset = hf_dataset
        self.processor = processor
        self.lag_num = lag_num
        self.config = config
        self.target_sr = target_sr
        self.audio_column = audio_column
        self.text_column = text_column
        self.codes_column = codes_column

    def __len__(self):
        return len(self.dataset)

    def _load_audio_from_bytes(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        """Load audio from bytes buffer."""
        buf = io.BytesIO(audio_bytes)
        audio, sr = sf.read(buf, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        return audio.astype(np.float32), int(sr)

    def _load_audio_from_path(self, path: str) -> Tuple[np.ndarray, int]:
        """Load audio from file path."""
        audio, sr = librosa.load(path, sr=None, mono=True)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        return audio.astype(np.float32), int(sr)

    def _normalize_audio_input(self, audio_obj) -> Tuple[np.ndarray, int]:
        """
        Normalize audio input from HuggingFace dataset format.

        Handles:
          - dict with 'bytes' key (from prepare_hf_sft_data.py)
          - dict with 'array' and 'sampling_rate' (decoded Audio feature)
          - str path
          - bytes directly
        """
        if isinstance(audio_obj, dict):
            # Check for decoded Audio feature format
            if "array" in audio_obj and "sampling_rate" in audio_obj:
                audio = np.array(audio_obj["array"], dtype=np.float32)
                sr = int(audio_obj["sampling_rate"])
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=-1)
                return audio, sr

            # Check for bytes format (from prepare_hf_sft_data.py)
            audio_bytes = audio_obj.get("bytes")
            if audio_bytes is not None:
                if isinstance(audio_bytes, memoryview):
                    audio_bytes = audio_bytes.tobytes()
                return self._load_audio_from_bytes(audio_bytes)

            # Fall back to path
            path = audio_obj.get("path")
            if path:
                return self._load_audio_from_path(path)

            raise ValueError(f"Cannot extract audio from dict: {audio_obj.keys()}")

        if isinstance(audio_obj, bytes):
            return self._load_audio_from_bytes(audio_obj)

        if isinstance(audio_obj, str):
            return self._load_audio_from_path(audio_obj)

        if isinstance(audio_obj, np.ndarray):
            raise ValueError("For numpy waveform input, pass a tuple (audio, sr).")

        if isinstance(audio_obj, tuple) and len(audio_obj) == 2:
            return (audio_obj[0].astype(np.float32), int(audio_obj[1]))

        raise TypeError(f"Unsupported audio input type: {type(audio_obj)}")

    def _resample_if_needed(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        """Resample audio to target sample rate if needed."""
        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr
        return audio, sr

    def _build_assistant_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _tokenize_texts(self, text: str) -> torch.Tensor:
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        input_id = inputs["input_ids"]
        input_id = input_id.unsqueeze(0) if input_id.dim() == 1 else input_id
        return input_id

    @torch.inference_mode()
    def extract_mels(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        """Extract mel spectrogram for speaker embedding."""
        audio, sr = self._resample_if_needed(audio, sr)
        assert sr == 24000, f"Expected 24kHz audio after resampling, got {sr}"
        mels = mel_spectrogram(
            torch.from_numpy(audio).unsqueeze(0),
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        ).transpose(1, 2)
        return mels

    def _parse_audio_codes(self, codes_obj) -> torch.Tensor:
        """
        Parse audio codes from dataset.

        Handles:
          - List of 16 lists (from prepare_hf_sft_data.py with codes transposed)
          - 2D array/list with shape (T, 16)
        """
        if isinstance(codes_obj, torch.Tensor):
            codes = codes_obj
        else:
            codes = torch.tensor(codes_obj, dtype=torch.long)

        # Ensure shape is (T, 16) - time steps x codebooks
        if codes.dim() == 1:
            # Single codebook - expand
            codes = codes.unsqueeze(-1)

        if codes.shape[-1] != 16 and codes.shape[0] == 16:
            # Shape is (16, T) - transpose to (T, 16)
            codes = codes.T

        assert codes.shape[-1] == 16, f"Expected 16 codebooks, got shape {codes.shape}"
        return codes

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Get audio for reference mel extraction
        audio_obj = item[self.audio_column]
        audio, sr = self._normalize_audio_input(audio_obj)

        # Get text
        text = item[self.text_column]
        text = self._build_assistant_text(text)
        text_ids = self._tokenize_texts(text)

        # Get pre-computed audio codes
        audio_codes = self._parse_audio_codes(item[self.codes_column])

        # Extract mel for speaker embedding (using same audio as target)
        ref_mel = self.extract_mels(audio=audio, sr=sr)

        return {
            "text_ids": text_ids[:, :-5],  # (1, T)
            "audio_codes": audio_codes,     # (T, 16)
            "ref_mel": ref_mel,
        }

    def collate_fn(self, batch):
        assert self.lag_num == -1

        item_length = [b["text_ids"].shape[1] + b["audio_codes"].shape[0] for b in batch]
        max_length = max(item_length) + 8
        b, t = len(batch), max_length

        input_ids = torch.zeros((b, t, 2), dtype=torch.long)
        codec_ids = torch.zeros((b, t, 16), dtype=torch.long)
        text_embedding_mask = torch.zeros((b, t), dtype=torch.bool)
        codec_embedding_mask = torch.zeros((b, t), dtype=torch.bool)
        codec_mask = torch.zeros((b, t), dtype=torch.bool)
        attention_mask = torch.zeros((b, t), dtype=torch.long)
        codec_0_labels = torch.full((b, t), -100, dtype=torch.long)

        for i, data in enumerate(batch):
            text_ids = data["text_ids"]
            audio_codec_0 = data["audio_codes"][:, 0]
            audio_codecs = data["audio_codes"]

            text_ids_len = text_ids.shape[1]
            codec_ids_len = audio_codec_0.shape[0]

            # text channel
            input_ids[i, :3, 0] = text_ids[0, :3]
            input_ids[i, 3:7, 0] = self.config.tts_pad_token_id
            input_ids[i, 7, 0] = self.config.tts_bos_token_id
            input_ids[i, 8 : 8 + text_ids_len - 3, 0] = text_ids[0, 3:]
            input_ids[i, 8 + text_ids_len - 3, 0] = self.config.tts_eos_token_id
            input_ids[i, 8 + text_ids_len - 2 : 8 + text_ids_len + codec_ids_len, 0] = (
                self.config.tts_pad_token_id
            )
            text_embedding_mask[i, : 8 + text_ids_len + codec_ids_len] = True

            # codec channel
            input_ids[i, 3:8, 1] = torch.tensor(
                [
                    self.config.talker_config.codec_nothink_id,
                    self.config.talker_config.codec_think_bos_id,
                    self.config.talker_config.codec_think_eos_id,
                    0,  # for speaker embedding
                    self.config.talker_config.codec_pad_id,
                ]
            )
            input_ids[i, 8 : 8 + text_ids_len - 3, 1] = self.config.talker_config.codec_pad_id
            input_ids[i, 8 + text_ids_len - 3, 1] = self.config.talker_config.codec_pad_id
            input_ids[i, 8 + text_ids_len - 2, 1] = self.config.talker_config.codec_bos_id
            input_ids[i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len, 1] = (
                audio_codec_0
            )
            input_ids[i, 8 + text_ids_len - 1 + codec_ids_len, 1] = (
                self.config.talker_config.codec_eos_token_id
            )

            codec_0_labels[i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len] = (
                audio_codec_0
            )
            codec_0_labels[i, 8 + text_ids_len - 1 + codec_ids_len] = (
                self.config.talker_config.codec_eos_token_id
            )

            codec_ids[i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len, :] = (
                audio_codecs
            )

            codec_embedding_mask[i, 3 : 8 + text_ids_len + codec_ids_len] = True
            codec_embedding_mask[i, 6] = False  # for speaker embedding

            codec_mask[i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len] = True
            attention_mask[i, : 8 + text_ids_len + codec_ids_len] = True

        ref_mels = [data["ref_mel"] for data in batch]
        ref_mels = torch.cat(ref_mels, dim=0)

        return {
            "input_ids": input_ids,
            "ref_mels": ref_mels,
            "attention_mask": attention_mask,
            "text_embedding_mask": text_embedding_mask.unsqueeze(-1),
            "codec_embedding_mask": codec_embedding_mask.unsqueeze(-1),
            "codec_0_labels": codec_0_labels,
            "codec_ids": codec_ids,
            "codec_mask": codec_mask,
        }
