# Raw Prediction Generation

The benchmark can be verified from the cached files in `predictions/` without
running model inference. This directory documents how the raw JSON files were
created for users who want to regenerate them independently.

Do not hardcode API keys in generation scripts. Use environment variables such
as:

- `ASSEMBLYAI_API_KEY`
- `CARTESIA_API_KEY`
- `DEEPGRAM_API_KEY`
- `ELEVENLABS_API_KEY`
- `FISH_AUDIO_API_KEY`
- `GRADIUM_API_KEY`
- `INWORLD_API_KEY`
- `XAI_API_KEY`

Expected raw JSON schema:

```json
[
  {
    "audio": "audio/example.wav",
    "prediction": "raw model transcript",
    "verbatim_ground_truth": "gold verbatim transcript",
    "intended_ground_truth": "gold intended transcript"
  }
]
```

Models that can produce separate outputs may use:

```json
{
  "verbatim_prediction": "raw verbatim prediction",
  "intended_prediction": "raw intended prediction"
}
```

The converter accepts both schemas and produces canonical JSONL files with
`pred_verbatim` and `pred_intended`.

The original experiment scripts used to create the current cached files live in
the research workspace. When upstreaming public reproducer scripts, sanitize
them first so credentials are read only from environment variables.
