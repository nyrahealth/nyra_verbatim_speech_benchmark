#!/usr/bin/env python3
"""Generate raw predictions with CrisperWhisper 2.0."""

from __future__ import annotations

import argparse
import contextlib
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from common import (
    add_common_args,
    audio_name,
    iter_dataset_rows,
    languages_from_arg,
    raw_record,
    resolve_audio_path,
    resolve_generation_config,
    sample_id,
    write_json,
)


DEFAULT_MODEL_ID = "crisperwhisper_2"
DEFAULT_MODEL_PATH = (
    "/home/ubuntu/data/models/CrisperWhisper_2_0/"
    "verbatimize-longform-english-no-timing-large-v2"
)


def _ensure_extra_path(path: str) -> None:
    if path and path not in sys.path:
        sys.path.insert(0, path)


def build_model(args: argparse.Namespace):
    if args.ctranslate2_python_path:
        _ensure_extra_path(args.ctranslate2_python_path)
    _ensure_extra_path(args.crisperwhisper_python_path)

    from crisperwhisper import CrisperWhisperModel

    return CrisperWhisperModel(
        args.model,
        backend=args.backend,
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


@contextlib.contextmanager
def encoded_audio_path(row: Dict[str, Any]) -> Iterator[Path]:
    resolved = resolve_audio_path(row)
    if resolved is not None:
        yield resolved
        return

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
    model_id: str,
    language: str,
    splits: Optional[list[str]],
    max_samples: int,
    output_path: Optional[Path],
    resume: bool,
) -> Path:
    out_path, effective_splits, dataset_name = resolve_generation_config(
        model_id=model_id,
        language=language,
        splits=splits,
        output_path=output_path,
    )
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
            dataset_name=dataset_name,
            language=language,
            model_id=model_id,
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
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
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
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "ct2", "transformers"],
        help="CrisperWhisper v2 inference backend. Use transformers if CT2 PTX/CUDA build mismatches.",
    )
    args = parser.parse_args()

    model = build_model(args)
    languages = languages_from_arg(args.language)
    for language in languages:
        path = generate_raw_predictions(
            model=model,
            model_id=args.model_id,
            language=language,
            splits=args.splits,
            max_samples=args.max_samples,
            output_path=Path(args.output_path) if args.output_path else None,
            resume=not args.no_resume,
        )
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
