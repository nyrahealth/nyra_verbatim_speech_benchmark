#!/usr/bin/env python3
"""Generate raw predictions with CrisperWhisper 1.0."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
from transformers import pipeline

from common import add_common_args, generate_raw_predictions, languages_from_arg


MODEL_ID = "crisperwhisper_1"
DEFAULT_HF_MODEL_ID = "nyrahealth/CrisperWhisper"


def build_transcriber(model_name_or_path: str, device: str):
    device_id = 0 if device == "cuda" and torch.cuda.is_available() else -1
    asr = pipeline(
        "automatic-speech-recognition",
        model=model_name_or_path,
        device=device_id,
    )

    def transcribe(audio_path: Path, language: str) -> Optional[str]:
        del language
        result = asr(str(audio_path))
        return str(result.get("text", "")).strip()

    return transcribe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--model", default=DEFAULT_HF_MODEL_ID)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    transcribe = build_transcriber(args.model, args.device)
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
