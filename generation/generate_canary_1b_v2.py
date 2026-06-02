#!/usr/bin/env python3
"""Generate raw predictions with NVIDIA Canary 1B v2."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from nemo.collections.asr.models import EncDecMultiTaskModel

from common import add_common_args, generate_raw_predictions, languages_from_arg


MODEL_ID = "canary"
NEMO_MODEL_ID = "nvidia/canary-1b-v2"


def build_transcriber(device: str):
    model = EncDecMultiTaskModel.from_pretrained(
        NEMO_MODEL_ID,
        map_location=device,
    )

    def transcribe(audio_path: Path, language: str) -> Optional[str]:
        result = model.transcribe(
            [str(audio_path)],
            source_lang=language,
            target_lang=language,
            return_hypotheses=False,
        )
        if isinstance(result, list):
            result = result[0] if result else ""
        if hasattr(result, "text"):
            return result.text
        return str(result).strip()

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
