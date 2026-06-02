#!/usr/bin/env python3
"""Generate raw predictions with openai/whisper-large-v3."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
from transformers import pipeline

from common import add_common_args, generate_raw_predictions, languages_from_arg


MODEL_ID = "whisper_large_v3"
HF_MODEL_ID = "openai/whisper-large-v3"


def build_transcriber(device: str):
    device_id = 0 if device == "cuda" and torch.cuda.is_available() else -1
    asr = pipeline(
        "automatic-speech-recognition",
        model=HF_MODEL_ID,
        device=device_id,
    )

    def transcribe(audio_path: Path, language: str) -> Optional[str]:
        result = asr(
            str(audio_path),
            generate_kwargs={"language": language, "task": "transcribe"},
        )
        return str(result.get("text", "")).strip()

    return transcribe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    transcribe = build_transcriber(args.device)
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
