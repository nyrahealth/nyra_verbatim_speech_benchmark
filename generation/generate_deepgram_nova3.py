#!/usr/bin/env python3
"""Generate raw predictions with Deepgram Nova-3."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import requests

from common import add_common_args, generate_raw_predictions, languages_from_arg


MODEL_ID = "deepgram"
API_URL = "https://api.deepgram.com/v1/listen"


def build_transcriber(api_key: str):
    def transcribe(audio_path: Path, language: str) -> Optional[str]:
        headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": "audio/wav",
        }
        params = {
            "model": "nova-3",
            "language": language,
            "smart_format": "true",
            "filler_words": "true",
        }
        with audio_path.open("rb") as audio_file:
            response = requests.post(
                API_URL,
                headers=headers,
                params=params,
                data=audio_file.read(),
                timeout=120,
            )
        response.raise_for_status()
        data = response.json()
        return data["results"]["channels"][0]["alternatives"][0].get("transcript", "")

    return transcribe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--api_key", default=os.environ.get("DEEPGRAM_API_KEY", ""))
    args = parser.parse_args()
    if not args.api_key:
        parser.error("--api_key or DEEPGRAM_API_KEY is required")

    transcribe = build_transcriber(args.api_key)
    for language in languages_from_arg(args.language):
        output_path = Path(args.output_path) if args.output_path else None
        path = generate_raw_predictions(
            model_id=MODEL_ID,
            language=language,
            splits=args.splits,
            max_samples=args.max_samples,
            output_path=output_path,
            resume=not args.no_resume,
            transcribe=transcribe,
        )
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
