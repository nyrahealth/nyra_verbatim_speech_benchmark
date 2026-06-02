# Raw Prediction Provenance

The benchmark can be verified from the cached files in `predictions/` without
running model inference. This directory documents how the raw JSON files were
created for readers who want to understand or independently reproduce the
reported comparisons.

Raw predictions are stored in the canonical benchmark schema under:

```text
predictions/raw/<model_id>/<language>.json
```

Each file is a JSON list:

```json
[
  {
    "id": "DISFLUENCY_TEST_000000",
    "audio": "audio/example.wav",
    "dataset": "nyrahealth/disfluency_speech_english",
    "split": "test",
    "language": "en",
    "model": "example_model",
    "raw_verbatim_prediction": "raw model transcript",
    "raw_intended_prediction": null,
    "gold_verbatim": "gold verbatim transcript",
    "gold_intended": "gold intended transcript"
  }
]
```

Models that only produce one transcript set `raw_intended_prediction` to
`null`. Models that produce separate verbatim and intended outputs can provide
both fields.

The normalized evaluator inputs are regenerated with:

```bash
python scripts/convert_outputs.py --skip_import
```
