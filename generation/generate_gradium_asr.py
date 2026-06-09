#!/usr/bin/env python3
"""Generate raw predictions with Gradium ASR."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
from pathlib import Path
from typing import Optional

import requests

from common import add_common_args, generate_raw_predictions, languages_from_arg


MODEL_ID = "gradium"
API_URL = "https://api.gradium.ai/api/post/speech/asr"


def build_transcriber(api_key: str):
    def transcribe(audio_path: Path, language: str) -> Optional[str]:
        config = urllib.parse.quote(json.dumps({"language": language}))
        url = f"{API_URL}?json_config={config}"
        headers = {"x-api-key": api_key, "Content-Type": "audio/wav"}

        for attempt in range(3):
            try:
                with audio_path.open("rb") as audio_file:
                    response = requests.post(
                        url,
                        headers=headers,
                        data=audio_file.read(),
                        timeout=120,
                    )
                if response.status_code == 200:
                    words = []
                    for line in response.text.splitlines():
                        if not line.strip():
                            continue
                        obj = json.loads(line)
                        if obj.get("type") == "text" and obj.get("text"):
                            words.append(obj["text"])
                    return " ".join(words)
                if response.status_code == 429:
                    time.sleep(2 ** (attempt + 1))
                    continue
                if response.status_code in (401, 403):
                    return None
                if attempt < 2:
                    time.sleep(2 ** attempt)
            except requests.exceptions.RequestException:
                if attempt < 2:
                    time.sleep(2)
        return None

    return transcribe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--api_key", default=os.environ.get("GRADIUM_API_KEY", ""))
    args = parser.parse_args()
    if not args.api_key:
        parser.error("--api_key or GRADIUM_API_KEY is required")

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
