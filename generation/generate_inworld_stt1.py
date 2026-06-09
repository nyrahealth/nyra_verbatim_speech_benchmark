#!/usr/bin/env python3
"""Generate raw predictions with Inworld STT (inworld/inworld-stt-1)."""

from __future__ import annotations

import argparse
import base64
import os
from pathlib import Path
from typing import Optional

import requests

from common import add_common_args, generate_raw_predictions, languages_from_arg
from language_codes import inworld_code


MODEL_ID = "inworld"
API_URL = "https://api.inworld.ai/stt/v1/transcribe"


def build_transcriber(api_key: str):
    def transcribe(audio_path: Path, language: str) -> Optional[str]:
        with audio_path.open("rb") as audio_file:
            audio_b64 = base64.b64encode(audio_file.read()).decode("ascii")
        language_code = inworld_code(language)
        payload = {
            "transcribeConfig": {
                "modelId": "inworld/inworld-stt-1",
                "audioEncoding": "LINEAR16",
                "sampleRateHertz": 16000,
                "enableLanguageDetection": True,
                "languageCode": language_code,
            },
            "audioData": {"content": audio_b64},
        }
        response = requests.post(
            API_URL,
            headers={
                "Authorization": f"Basic {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        transcription = data.get("transcription") or {}
        if isinstance(transcription, dict):
            text = transcription.get("transcript")
            if text:
                return text
        return data.get("text") or data.get("transcript") or ""

    return transcribe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--api_key", default=os.environ.get("INWORLD_API_KEY", ""))
    args = parser.parse_args()
    if not args.api_key:
        parser.error("--api_key or INWORLD_API_KEY is required")

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
