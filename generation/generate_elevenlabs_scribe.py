#!/usr/bin/env python3
"""Generate raw predictions with ElevenLabs Scribe v2."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

from common import add_common_args, generate_raw_predictions, languages_from_arg
from language_codes import elevenlabs_code


MODEL_ID = "elevenlabs"


def build_transcriber(api_key: str):
    from elevenlabs.client import ElevenLabs

    client = ElevenLabs(api_key=api_key)

    def transcribe(audio_path: Path, language: str) -> Optional[str]:
        language_code = elevenlabs_code(language)
        with audio_path.open("rb") as audio_file:
            result = client.speech_to_text.convert(
                file=audio_file,
                model_id="scribe_v2",
                language_code=language_code,
                tag_audio_events=True,
            )
        if hasattr(result, "text"):
            return result.text or ""
        if isinstance(result, dict):
            return result.get("text") or result.get("transcript") or ""
        return str(result or "").strip()

    return transcribe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--api_key", default=os.environ.get("ELEVENLABS_API_KEY", ""))
    args = parser.parse_args()
    if not args.api_key:
        parser.error("--api_key or ELEVENLABS_API_KEY is required")

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
