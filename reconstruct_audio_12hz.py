# coding=utf-8
# Qwen3-TTS 12Hz Audio Tokenizer - Encode/Decode Reconstruction Script
"""
간단한 오디오 재구성 스크립트
업로드된 오디오 파일을 12Hz tokenizer로 encode한 후 decode하여 재구성합니다.

Usage:
    python reconstruct_audio_12hz.py input.wav
    python reconstruct_audio_12hz.py input.wav -o output.wav
    python reconstruct_audio_12hz.py input.wav --device cuda:0
"""
import argparse
from pathlib import Path

import soundfile as sf
import torch

from qwen_tts import Qwen3TTSTokenizer


def reconstruct_audio(
    input_path: str,
    output_path: str | None = None,
    device: str = "cuda:0",
) -> None:
    """
    오디오 파일을 12Hz tokenizer로 encode/decode하여 재구성합니다.

    Args:
        input_path: 입력 오디오 파일 경로
        output_path: 출력 파일 경로 (None이면 자동 생성)
        device: 사용할 디바이스 (cuda:0, cpu 등)
    """
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")

    if output_path is None:
        output_path = str(input_file.stem) + "_reconstructed.wav"

    print(f"[1/4] 12Hz Tokenizer 로딩 중... (device: {device})")
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        device_map=device,
    )

    print(f"[2/4] 오디오 인코딩 중: {input_path}")
    encoded = tokenizer.encode(str(input_path))

    audio_codes = encoded.audio_codes[0]
    print(f"      - Audio codes shape: {audio_codes.shape}")
    print(f"      - Codes dtype: {audio_codes.dtype}")

    print("[3/4] 오디오 디코딩 중...")
    wavs, sample_rate = tokenizer.decode(encoded)

    print(f"[4/4] 결과 저장 중: {output_path}")
    sf.write(output_path, wavs[0], sample_rate)

    print(f"\n완료!")
    print(f"  - 입력: {input_path}")
    print(f"  - 출력: {output_path}")
    print(f"  - 샘플레이트: {sample_rate} Hz")
    print(f"  - 출력 길이: {len(wavs[0]) / sample_rate:.2f}초")


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS 12Hz Tokenizer를 사용하여 오디오를 encode/decode합니다."
    )
    parser.add_argument(
        "input",
        type=str,
        help="입력 오디오 파일 경로 (wav, mp3 등)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="출력 파일 경로 (기본값: <input>_reconstructed.wav)",
    )
    parser.add_argument(
        "-d", "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="사용할 디바이스 (기본값: cuda:0 또는 cpu)",
    )

    args = parser.parse_args()
    reconstruct_audio(args.input, args.output, args.device)


if __name__ == "__main__":
    main()
