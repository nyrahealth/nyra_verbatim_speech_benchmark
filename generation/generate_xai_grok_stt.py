#!/usr/bin/env python3
"""Generate raw predictions with the xAI Grok speech-to-text API."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import requests

from common import add_common_args, generate_raw_predictions, languages_from_arg


MODEL_ID = "xai_grok"
API_URL = "https://api.x.ai/v1/stt"


def build_transcriber(api_key: str):
    def transcribe(audio_path: Path, language: str) -> Optional[str]:
        with audio_path.open("rb") as audio_file:
            response = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                data={
                    "filler_words": "true",
                    "format": "false",
                    "language": language,
                },
                files={"file": (audio_path.name, audio_file, "audio/wav")},
                timeout=120,
            )
        response.raise_for_status()
        return response.json().get("text", "")

    return transcribe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--api_key", default=os.environ.get("XAI_API_KEY", ""))
    args = parser.parse_args()
    if not args.api_key:
        parser.error("--api_key or XAI_API_KEY is required")

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
