# Adding a Model

This benchmark is designed so new ASR systems can be added without changing the
evaluator.

## 1. Create Raw Predictions

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

## 2. Register the Model

Add an entry to `benchmark/models.json`:

```json
{
  "id": "my_model",
  "display_name": "My Model",
  "adapter": "default",
  "single_output": true,
  "languages": {
    "en": {
      "raw_file": "predictions/raw/my_model/en.json",
      "dataset": "nyrahealth/disfluency_speech_english",
      "splits": ["train", "validation", "test"]
    }
  }
}
```

Most models should use `"adapter": "default"`.

## 3. Convert and Evaluate

```bash
python scripts/convert_outputs.py --skip_import --model my_model
python scripts/validate_predictions.py
python scripts/evaluate_all_cached.py --model my_model
python scripts/print_leaderboard.py
```

## When To Add Normalization

Add global normalization only for safe, cross-model conventions such as
`[laughs] -> [laughter]` or Unicode dash cutoffs. Use a model adapter only for
provider-specific quirks that would be unsafe globally.

See `docs/normalization.md` for details.
