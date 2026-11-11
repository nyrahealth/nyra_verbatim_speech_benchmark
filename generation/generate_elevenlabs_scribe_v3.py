#!/usr/bin/env python3
"""Generate raw predictions with ElevenLabs Scribe v3 (POST /v1/speech-to-text).

Same request settings as the Scribe v2 run (`generate_elevenlabs_scribe.py`):
language hint + ``tag_audio_events=True``; only ``model_id`` differs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from common import (
    add_common_args,
    iter_dataset_rows,
    languages_from_arg,
    load_existing,
    raw_record,
    resolve_generation_config,
    sample_id,
    wav_path,
    write_json,
)
from language_codes import elevenlabs_code


MODEL_ID = "elevenlabs_scribe_v3"
ELEVENLABS_MODEL = "scribe_v3"
# Data-residency keys need their regional host, e.g. https://api.eu.residency.elevenlabs.io
API_BASE = os.environ.get("ELEVENLABS_API_BASE", "https://api.elevenlabs.io").rstrip("/")
API_URL = f"{API_BASE}/v1/speech-to-text"
MAX_ATTEMPTS = 6


def build_transcriber(api_key: str):
    def transcribe(audio_path: Path, language: str) -> str:
        last_error: Optional[str] = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                with audio_path.open("rb") as audio_file:
                    response = requests.post(
                        API_URL,
                        headers={"xi-api-key": api_key},
                        data={
                            "model_id": ELEVENLABS_MODEL,
                            "language_code": elevenlabs_code(language),
                            "tag_audio_events": "true",
                        },
                        files={"file": (audio_path.name, audio_file, "audio/wav")},
                        timeout=300,
                    )
                if response.status_code == 200:
                    return (response.json().get("text") or "").strip()
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                if response.status_code in (400, 401, 403, 404, 422):
                    break
            except requests.RequestException as error:
                last_error = str(error)
            time.sleep(min(2 ** attempt * 2, 60))
        raise RuntimeError(f"ElevenLabs transcription failed: {last_error}")

    return transcribe


def generate_language(
    *,
    language: str,
    splits,
    max_samples: int,
    output_path: Optional[Path],
    resume: bool,
    workers: int,
    transcribe,
) -> Path:
    out_path, effective_splits, dataset_name = resolve_generation_config(
        model_id=MODEL_ID,
        language=language,
        splits=splits,
        output_path=output_path,
    )
    rows = load_existing(out_path) if resume else []
    rows = [row for row in rows if row.get("raw_verbatim_prediction")]
    done = {str(row.get("id")) for row in rows}

    def worker(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        sid = sample_id(sample)
        try:
            with wav_path(sample) as audio_path:
                prediction = transcribe(audio_path, language)
        except Exception as error:  # noqa: BLE001 - keep going, resume refetches
            print(f"[{language}] FAILED {sid}: {error}", flush=True)
            return None
        return raw_record(
            sample,
            dataset_name=dataset_name,
            language=language,
            model_id=MODEL_ID,
            prediction=prediction,
        )

    pending = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:

        def drain(block_until_below: int) -> None:
            while len(pending) > block_until_below:
                record = pending.pop(0).result()
                if record is not None:
                    rows.append(record)
                    if len(rows) % 25 == 0:
                        write_json(out_path, rows)
                        print(f"[{language}] {len(rows)} predictions cached", flush=True)

        for sample in iter_dataset_rows(
            language, splits=effective_splits, max_samples=max_samples
        ):
            if sample_id(sample) in done:
                continue
            pending.append(pool.submit(worker, sample))
            drain(workers * 2)
        drain(0)

    write_json(out_path, rows)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--api_key", default=os.environ.get("ELEVENLABS_API_KEY", ""))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not args.api_key:
        parser.error("--api_key or ELEVENLABS_API_KEY is required")

    transcribe = build_transcriber(args.api_key)
    for language in languages_from_arg(args.language):
        path = generate_language(
            language=language,
            splits=args.splits,
            max_samples=args.max_samples,
            output_path=Path(args.output_path) if args.output_path else None,
            resume=not args.no_resume,
            workers=args.workers,
            transcribe=transcribe,
        )
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
