#!/usr/bin/env python3
"""Generate raw predictions with CrisperWhisper 2.0."""

from __future__ import annotations

import argparse
import contextlib
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from common import (
    DATASETS,
    add_common_args,
    audio_name,
    manifest_language_config,
    manifest_raw_path,
    raw_record,
    sample_id,
    write_json,
)
from datasets import Audio, load_dataset


MODEL_ID = "crisperwhisper_2"
DEFAULT_MODEL_PATH = (
    "/home/ubuntu/data/models/CrisperWhisper_2_0/"
    "verbatimize-longform-english-no-timing-large-v2"
)


def _ensure_extra_path(path: str) -> None:
    if path and path not in sys.path:
        sys.path.insert(0, path)


def build_model(args: argparse.Namespace):
    _ensure_extra_path(args.ctranslate2_python_path)
    _ensure_extra_path(args.crisperwhisper_python_path)

    from crisperwhisper import CrisperWhisperModel

    return CrisperWhisperModel(
        args.model,
        backend="ct2",
        compute_type=args.compute_type,
        device=args.device,
        cache_dir=args.cache_dir or None,
    )


def load_existing(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    import json

    with path.open(encoding="utf-8") as f:
        return json.load(f)


def iter_dataset_rows(
    language: str,
    splits: Optional[Iterable[str]],
    max_samples: int,
) -> Iterator[Dict[str, Any]]:
    emitted = 0
    for split in splits or []:
        dataset = load_dataset(DATASETS[language], split=split)
        dataset = dataset.cast_column("audio", Audio(decode=False))
        for row in dataset:
            row = dict(row)
            row["split"] = row.get("split") or split
            yield row
            emitted += 1
            if max_samples and emitted >= max_samples:
                return


@contextlib.contextmanager
def encoded_audio_path(row: Dict[str, Any]) -> Iterator[Path]:
    audio = row["audio"]
    if isinstance(audio, dict) and audio.get("path") and Path(audio["path"]).exists():
        yield Path(audio["path"])
        return
    if isinstance(audio, str) and Path(audio).exists():
        yield Path(audio)
        return
    if not isinstance(audio, dict) or audio.get("bytes") is None:
        raise ValueError("Expected Hugging Face audio bytes or an existing path")

    suffix = Path(str(audio.get("path") or "audio.wav")).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(audio["bytes"])
        tmp.flush()
        yield Path(tmp.name)


def generate_raw_predictions(
    *,
    model,
    language: str,
    splits: Optional[Iterable[str]],
    max_samples: int,
    output_path: Optional[Path],
    resume: bool,
) -> Path:
    manifest_config = manifest_language_config(MODEL_ID, language)
    effective_splits = list(splits) if splits else manifest_config.get("splits")
    out_path = output_path or manifest_raw_path(MODEL_ID, language)
    rows = load_existing(out_path) if resume else []
    done = {str(row.get("id")) for row in rows if row.get("raw_verbatim_prediction")}

    for index, sample in enumerate(
        iter_dataset_rows(language, splits=effective_splits, max_samples=max_samples),
        start=1,
    ):
        sid = sample_id(sample)
        if sid in done:
            continue
        print(f"[{language}] {index}: {sid}", flush=True)
        with encoded_audio_path(sample) as audio_path:
            verbatim = model.transcribe(
                str(audio_path),
                language=language,
                mode="verbatim",
            ).text.strip()
            intended = model.transcribe(
                str(audio_path),
                language=language,
                mode="intended",
            ).text.strip()

        record = raw_record(
            sample,
            dataset_name=DATASETS[language],
            language=language,
            model_id=MODEL_ID,
            prediction=verbatim,
            intended_prediction=intended,
        )
        record["audio"] = audio_name(sample)
        rows.append(record)
        write_json(out_path, rows)

    write_json(out_path, rows)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.set_defaults(language="en")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute_type", default="float16")
    parser.add_argument("--cache_dir", default="/ephemeral/crisperwhisper_cache")
    parser.add_argument(
        "--ctranslate2_python_path",
        default="/home/ubuntu/code/CTranslate2/python",
        help="Path to the local CTranslate2 Python package.",
    )
    parser.add_argument(
        "--crisperwhisper_python_path",
        default="/home/ubuntu/code/research",
        help="Path containing the local crisperwhisper package.",
    )
    args = parser.parse_args()

    model = build_model(args)
    languages = ["en", "de"] if args.language == "all" else [args.language]
    for language in languages:
        path = generate_raw_predictions(
            model=model,
            language=language,
            splits=args.splits,
            max_samples=args.max_samples,
            output_path=Path(args.output_path) if args.output_path else None,
            resume=not args.no_resume,
        )
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
