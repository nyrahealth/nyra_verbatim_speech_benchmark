# Adding a Model

This benchmark is designed so new ASR systems can be added without changing the
evaluator.

## Overview

Adding a model has two parts:

1. Register the model in `benchmark/models.json`.
2. Provide raw predictions in `predictions/raw/<model_id>/<language>.json`.

If you want others to regenerate the raw predictions, also add a compact script
under `generation/` that loads the public Hugging Face datasets and writes the
same raw prediction schema.

## 1. Choose a Model ID

Use a stable, lowercase identifier:

```text
my_model
```

Use the human-readable display name for the exact model version that was
evaluated, for example:

```text
My ASR System (provider/model-version)
```

## 2. Register the Model

Add an entry to `benchmark/models.json`:

```json
{
  "id": "my_model",
  "display_name": "My ASR System (provider/model-version)",
  "adapter": "default",
  "single_output": true,
  "languages": {
    "en": {
      "raw_file": "predictions/raw/my_model/en.json",
      "dataset": "nyrahealth/disfluency_speech_english",
      "splits": ["train", "validation", "test"]
    },
    "de": {
      "raw_file": "predictions/raw/my_model/de.json",
      "dataset": "nyrahealth/disfluency_speech_german",
      "splits": ["test"]
    }
  }
}
```

Most models should use `"adapter": "default"`. Add a source adapter only for
provider-specific output conventions that would be unsafe to normalize globally.

## 3. Create Raw Predictions

Write one JSON list per model/language:

```text
predictions/raw/<model_id>/<language>.json
```

Each record must use the fixed raw schema:

```json
{
  "id": "DISFLUENCY_TEST_000000",
  "audio": "audio/DISFLUENCY_TEST_000000.wav",
  "dataset": "nyrahealth/disfluency_speech_english",
  "split": "test",
  "language": "en",
  "model": "my_model",
  "raw_verbatim_prediction": "Well, uh, I think th— that works.",
  "raw_intended_prediction": null,
  "gold_verbatim": "Well, [UH], I think th* that works.",
  "gold_intended": "Well, I think that works."
}
```

If a model only has one transcript output, set `raw_intended_prediction` to
`null` or the same string as `raw_verbatim_prediction`. If a model has true
intended output, put it in `raw_intended_prediction`.

## 4. Add a Generation Example

Generation examples are optional for scoring, but recommended for reproducible
benchmark submissions.

Create:

```text
generation/generate_my_model.py
```

Use `generation/common.py` so the script automatically:

- loads the benchmark Hugging Face datasets
- follows the splits configured in `benchmark/models.json`
- writes to the configured raw prediction path
- emits the canonical raw schema

A minimal script only needs to define the model call:

```python
#!/usr/bin/env python3
"""Generate raw predictions with My ASR System."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from common import add_common_args, generate_raw_predictions, languages_from_arg


MODEL_ID = "my_model"


def transcribe(audio_path: Path, language: str) -> Optional[str]:
    # Call the ASR system here and return the raw transcript string.
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()

    for language in languages_from_arg(args.language):
        generate_raw_predictions(
            model_id=MODEL_ID,
            language=language,
            splits=args.splits,
            max_samples=args.max_samples,
            output_path=Path(args.output_path) if args.output_path else None,
            resume=not args.no_resume,
            transcribe=transcribe,
        )


if __name__ == "__main__":
    main()
```

Document the script in `generation/README.md`, including the exact model name or
model identifier used for the benchmark.

## 5. Convert and Evaluate

```bash
python scripts/convert_outputs.py --skip_import --model my_model
python scripts/validate_predictions.py
python scripts/evaluate_all_cached.py --model my_model
python scripts/print_leaderboard.py
```

For a small generation smoke test:

```bash
python generation/generate_my_model.py --language en --max_samples 2 --no_resume
python scripts/convert_outputs.py --skip_import --model my_model --language en
python scripts/validate_predictions.py
```

## When To Add Normalization

Add global normalization only for safe, cross-model conventions such as
`[laughs] -> [laughter]` or Unicode dash cutoffs. Use a model adapter only for
provider-specific quirks that would be unsafe globally.

See `docs/normalization.md` for details.
