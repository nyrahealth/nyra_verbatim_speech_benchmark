#!/usr/bin/env python3
"""Generate raw predictions with AssemblyAI Universal-3 Pro."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import assemblyai as aai

from common import add_common_args, generate_raw_predictions, languages_from_arg
from language_codes import assemblyai_code, assemblyai_speech_models


PROMPTS = {
    1: "Transcribe the audio verbatim, preserving disfluencies.",
    2: """Transcribe verbatim:
- Fillers: yes (um, uh, like, you know)
- Repetitions: yes (I I, the the)
- False starts: yes (I was I went)
- Stutters: yes (th- that, b- but)
- Vocal sounds: yes ([laughter], [cough], [breath], [throatclearing], [lipsmack], [sigh], [sniff])""",
}


def build_transcriber(api_key: str, prompt: int):
    aai.settings.api_key = api_key

    def transcribe(audio_path: Path, language: str) -> Optional[str]:
        language_code = assemblyai_code(language)
        config = aai.TranscriptionConfig(
            language_code=language_code,
            speech_models=assemblyai_speech_models(language),
            prompt=PROMPTS[prompt],
            punctuate=True,
        )
        transcriber = aai.Transcriber(config=config)
        for attempt in range(3):
            transcript = transcriber.transcribe(str(audio_path))
            if transcript.status != aai.TranscriptStatus.error:
                return transcript.text or ""
            if attempt < 2:
                time.sleep(2)
        return None

    return transcribe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--api_key", default=os.environ.get("ASSEMBLYAI_API_KEY", ""))
    parser.add_argument("--prompt", type=int, choices=[1, 2], default=2)
    args = parser.parse_args()
    if not args.api_key:
        parser.error("--api_key or ASSEMBLYAI_API_KEY is required")

    model_id = f"assemblyai_p{args.prompt}"
    transcribe = build_transcriber(args.api_key, args.prompt)
    for language in languages_from_arg(args.language):
        path = generate_raw_predictions(
            model_id=model_id,
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
