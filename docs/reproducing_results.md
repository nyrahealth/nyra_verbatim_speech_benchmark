# Reproducing Results

You can reproduce cached benchmark results without running model inference.

## Setup

```bash
pip install -r requirements.txt
```

## Validate Inputs

```bash
python scripts/validate_predictions.py
```

This checks:

- model manifest consistency
- raw prediction schema
- normalized prediction schema
- duplicate IDs
- idempotence of normalized predictions

## Regenerate Normalized Predictions

```bash
python scripts/convert_outputs.py --skip_import
```

Use `--skip_import` when the canonical raw files already exist in
`predictions/raw/`.

## Evaluate Cached Predictions

```bash
python scripts/evaluate_all_cached.py --output_dir results/cached
```

## Print Tables

```bash
python scripts/print_leaderboard.py
```

To save the table:

```bash
python scripts/print_leaderboard.py --output results/cached/leaderboard.md
```

## Dataset-Backed Evaluation

Cached prediction files include gold transcripts so results can be verified
without downloading datasets. If you prefer to load gold transcripts directly
from Hugging Face:

```bash
python scripts/download_datasets.py
python evaluate.py \
  --predictions predictions/normalized/elevenlabs/en.jsonl \
  --datasets nyrahealth/disfluency_speech_english \
  --splits train validation test \
  --output_dir results/elevenlabs_english_hf
```

Hugging Face uses its normal cache directory unless you configure it otherwise.
