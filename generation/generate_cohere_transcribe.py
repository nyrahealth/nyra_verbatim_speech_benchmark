#!/usr/bin/env python3
"""Generate raw predictions with CohereLabs/cohere-transcribe-03-2026."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import librosa
import soundfile as sf
import torch
from transformers import AutoProcessor, CohereAsrForConditionalGeneration

from common import add_common_args, generate_raw_predictions, languages_from_arg


MODEL_ID = "cohere"
HF_MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"


def load_audio(audio_path: Path, target_sr: int = 16000):
    audio, sample_rate = sf.read(audio_path, dtype="float32")
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    if sample_rate != target_sr:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)
    return audio


def build_transcriber():
    processor = AutoProcessor.from_pretrained(HF_MODEL_ID)
    model = CohereAsrForConditionalGeneration.from_pretrained(
        HF_MODEL_ID,
        device_map="auto",
    )

    def transcribe(audio_path: Path, language: str) -> Optional[str]:
        audio = load_audio(audio_path)
        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            language=language,
        )
        inputs.to(model.device, dtype=model.dtype)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512)
        text = processor.decode(outputs, skip_special_tokens=True)
        if isinstance(text, list):
            return text[0].strip() if text else ""
        return str(text).strip()

    return transcribe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()

    transcribe = build_transcriber()
    for language in languages_from_arg(args.language):
        path = generate_raw_predictions(
            model_id=MODEL_ID,
            language=language,
            splits=args.splits,
            max_samples=args.max_samples,
            output_path=Path(args.output_path) if args.output_path else None,
            resume=not args.no_resume,
            transcribe=transcribe,
        )
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
