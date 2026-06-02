# Raw Prediction Provenance

The benchmark can be verified from the cached files in `predictions/` without
running model inference. This directory documents how the raw JSON files were
created for readers who want to understand or independently reproduce the
reported comparisons.

The scripts here are cleaned examples derived from the research-generation code.
They load the public Hugging Face datasets, run one ASR system, and write raw
predictions directly into the benchmark schema.

## Included Examples

| Script | Benchmark entry |
|---|---|
| `generate_assemblyai_universal3.py` | AssemblyAI Universal-3 Pro (`universal-3-pro`) |
| `generate_canary_1b_v2.py` | NVIDIA Canary 1B v2 (`nvidia/canary-1b-v2`) |
| `generate_cartesia_ink_whisper.py` | Cartesia Ink-Whisper (`ink-whisper`) |
| `generate_cohere_transcribe.py` | Cohere Transcribe (`CohereLabs/cohere-transcribe-03-2026`) |
| `generate_crisperwhisper_1.py` | CrisperWhisper 1.0 (`nyrahealth/CrisperWhisper`) |
| `generate_deepgram_nova3.py` | Deepgram Nova-3 (`nova-3`) |
| `generate_fish_audio_asr.py` | Fish Audio ASR |
| `generate_gradium_asr.py` | Gradium ASR |
| `generate_inworld_stt1.py` | Inworld STT (`inworld/inworld-stt-1`) |
| `generate_whisper_large_v3.py` | Whisper Large v3 (`openai/whisper-large-v3`) |
| `generate_xai_grok_stt.py` | xAI Grok Speech-to-Text |

## Usage

Install the optional generation dependencies for the scripts you want to run:

```bash
pip install -r generation/requirements.txt
```

Run a script from the repository root. For example:

```bash
python generation/generate_deepgram_nova3.py --language en
python generation/generate_canary_1b_v2.py --language de
python generation/generate_assemblyai_universal3.py --prompt 2 --language all
```

By default, scripts write to the raw prediction path configured in
`benchmark/models.json`, for example:

```text
predictions/raw/deepgram/en.json
```

The default language splits also come from `benchmark/models.json`, so script
defaults match the benchmarked cached runs.

Use `--max_samples` for a small smoke test.

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

Then evaluate with:

```bash
python scripts/reproduce_results.py
```
