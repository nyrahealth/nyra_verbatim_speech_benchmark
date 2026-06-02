#!/usr/bin/env python3
"""Generate raw predictions with Fish Audio ASR."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import requests

from common import add_common_args, generate_raw_predictions, languages_from_arg


MODEL_ID = "fish_audio"
API_URL = "https://api.fish.audio/v1/asr"


def build_transcriber(api_key: str):
    def transcribe(audio_path: Path, language: str) -> Optional[str]:
        with audio_path.open("rb") as audio_file:
            response = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                files={"audio": (audio_path.name, audio_file, "audio/wav")},
                data={"language": language, "ignore_timestamps": "true"},
                timeout=120,
            )
        response.raise_for_status()
        data = response.json()
        return data.get("text") or data.get("transcript") or ""

    return transcribe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--api_key", default=os.environ.get("FISH_AUDIO_API_KEY", ""))
    args = parser.parse_args()
    if not args.api_key:
        parser.error("--api_key or FISH_AUDIO_API_KEY is required")

    transcribe = build_transcriber(args.api_key)
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
