#!/usr/bin/env python3
"""Generate raw predictions with Microsoft MAI-Transcribe-1.5."""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Optional

import requests

from common import (
    add_common_args,
    generate_raw_predictions,
    languages_from_arg,
)
from language_codes import mai_locale


MODEL_ID = "mai_transcribe_15"
DEFAULT_ENDPOINT = "https://eastus.api.cognitive.microsoft.com"
API_VERSION = "2025-10-15"


def build_definition(locale: str) -> dict:
    return {
        "locales": [locale],
        "enhancedMode": {
            "enabled": True,
            "model": "mai-transcribe-1.5",
            "transcribeStyle": "verbatim",
        },
    }


def extract_text(response_json: dict) -> str:
    combined = response_json.get("combinedPhrases") or []
    if combined and combined[0].get("text"):
        return combined[0]["text"]
    phrases = response_json.get("phrases") or []
    return " ".join(
        phrase.get("text", "")
        for phrase in phrases
        if phrase.get("text")
    ).strip()


def build_transcriber(api_key: str, endpoint: str):
    url = (
        f"{endpoint.rstrip('/')}/speechtotext/transcriptions:transcribe"
        f"?api-version={API_VERSION}"
    )

    def transcribe_with_definition(
        audio_path: Path,
        definition: dict,
    ) -> Optional[str]:
        headers = {"Ocp-Apim-Subscription-Key": api_key}
        for attempt in range(3):
            with audio_path.open("rb") as audio_file:
                response = requests.post(
                    url,
                    headers=headers,
                    files={
                        "audio": (audio_path.name, audio_file, "audio/wav"),
                        "definition": (
                            None,
                            json.dumps(definition),
                            "application/json",
                        ),
                    },
                    timeout=300,
                )

            if response.ok:
                return extract_text(response.json())

            text = (response.text or "").lower()
            if (
                response.status_code == 400
                and "enhanced mode with model" in text
            ):
                fallback = copy.deepcopy(definition)
                fallback.get("enhancedMode", {}).pop("model", None)
                return transcribe_with_definition(audio_path, fallback)
            if (
                response.status_code == 400
                and "enhanced mode is currently not supported" in text
            ):
                fallback = copy.deepcopy(definition)
                fallback.pop("enhancedMode", None)
                return transcribe_with_definition(audio_path, fallback)

            retryable_status = {429, 500, 502, 503, 504}
            if response.status_code in retryable_status and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            response.raise_for_status()

        return None

    def transcribe(audio_path: Path, language: str) -> Optional[str]:
        return transcribe_with_definition(
            audio_path,
            build_definition(mai_locale(language)),
        )

    return transcribe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument(
        "--api_key",
        default=os.environ.get("MAI_TRANSCRIBE_15_KEY", ""),
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("MAI_TRANSCRIBE_15_ENDPOINT", DEFAULT_ENDPOINT),
    )
    args = parser.parse_args()
    if not args.api_key:
        parser.error("--api_key or MAI_TRANSCRIBE_15_KEY is required")

    transcribe = build_transcriber(args.api_key, args.endpoint)
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
