#!/usr/bin/env python3
"""Generate raw predictions with CrisperWhisper 2.0.

Uses the released ``crisperwhisper`` package (``pip install crisperwhisper[ct2]``
for the fast CTranslate2 backend, or ``crisperwhisper[transformers]`` for the
pure-torch backend). ``--model`` accepts an official size shorthand (``large``,
``turbo``, ...), a Hugging Face model id, or a local checkpoint directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

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


DEFAULT_MODEL_ID = "crisperwhisper_2"
DEFAULT_MODEL = "large"  # -> nyralabs/CrisperWhisper2.0_large


def build_model(args: argparse.Namespace):
    from crisperwhisper import CrisperWhisperModel

    return CrisperWhisperModel(
        args.model,
        backend=args.backend,
        compute_type=args.compute_type,
        device=args.device,
        cache_dir=args.cache_dir or None,
    )


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
    # Drop cached rows with empty predictions so resume can refetch them.
    rows = [row for row in rows if row.get("raw_verbatim_prediction")]
    done = {str(row.get("id")) for row in rows}

    for index, sample in enumerate(
        iter_dataset_rows(language, splits=effective_splits, max_samples=max_samples),
        start=1,
    ):
        sid = sample_id(sample)
        if sid in done:
            continue
        print(f"[{language}] {index}: {sid}", flush=True)
        with wav_path(sample) as audio_path:
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

        rows.append(
            raw_record(
                sample,
                dataset_name=dataset_name,
                language=language,
                model_id=model_id,
                prediction=verbatim,
                intended_prediction=intended,
            )
        )
        write_json(out_path, rows)

    write_json(out_path, rows)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.set_defaults(language="en")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Size shorthand, Hugging Face id, or local checkpoint path.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute_type", default="float16")
    parser.add_argument(
        "--cache_dir",
        default="",
        help="Optional cache directory for HF->CT2 model conversion.",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "ct2", "transformers"],
        help="Inference backend. Use transformers if no compatible GPU/CUDA setup.",
    )
    args = parser.parse_args()

    model = build_model(args)
    for language in languages_from_arg(args.language):
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
