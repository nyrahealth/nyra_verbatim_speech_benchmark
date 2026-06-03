# Nyra Verbatim Speech Benchmark

**A benchmark for evaluating verbatim automatic speech recognition: not only the words, but also fillers, repetitions, cutoffs, and vocal sound events.**

Most speech benchmarks focus on clean transcript accuracy. This repository
focuses on the messy but important details of real speech: `uh`, `um`, repeated
words, interrupted words like `th-`, and vocal sound events like `[laughter]` or
`[cough]`.

The benchmark compares two kinds of transcripts:

- **Verbatim transcript**: what the speaker actually said, including disfluencies.
- **Intended transcript**: a cleaned version of what the speaker meant to say.

From that pair, the evaluator automatically derives disfluency labels and
reports detailed ASR quality metrics. The goal is to make verbatim ASR
evaluation reproducible, comparable, and easy to extend across systems.

Designed for:

- 🗣️ conversational and spontaneous speech recognition
- 🩺 clinical speech and communication research
- 🎙️ natural text-to-speech and voice data pipelines
- 🔬 speech science, linguistics, and disfluency analysis
- 🧪 ASR model comparison beyond ordinary WER

## Why This Exists ✨

WER is useful, but it is not a complete definition of speech recognition
quality. It means different things on different datasets and often depends on
underspecified references: should a filler count, should laughter be written
down, should a false start be preserved, should the transcript represent what
was said or what was meant?

Many models have low WER on read speech, or optimize implicitly for intended
transcription, while dropping the exact phenomena that matter for clinical,
conversational, natural text-to-speech, and speech-science use cases.

This benchmark therefore reports both transcript-level quality and typed
disfluency scores:

- **Transcript quality**: `vWER`, `iWER`, `vCER`, `iCER`
- **Fillers**: `[UH]`, `[UM]`, including exact and per-label scores
- **Vocal sound events**: `[laughter]`, `[cough]`, `[breath]`, and related events
- **Cutoffs**: interrupted words such as `th-`, with several matching criteria
- **Repetitions**: repeated words and repetition-count accuracy
- **Diagnostics**: disfluency omission and hallucination rates

## What You Can Do

- **Reproduce the included model comparisons** from cached predictions.
- **Evaluate another ASR system** by adding its raw transcripts.
- **Compare verbatim and intended transcription behavior** when both are available.
- **Analyze errors by disfluency type**, rather than relying on WER alone.

## Current Scope 🌍

- Languages: **English** and **German**
- Datasets: [`nyrahealth/disfluency_speech_english`](https://huggingface.co/datasets/nyrahealth/disfluency_speech_english) and [`nyrahealth/disfluency_speech_german`](https://huggingface.co/datasets/nyrahealth/disfluency_speech_german)
- Cached model families include AssemblyAI, Canary, Cartesia, Cohere, CrisperWhisper, Deepgram, ElevenLabs, Fish Audio, Gradium, Inworld, MAI-Transcribe, Whisper Large v3, and xAI Grok.

The evaluator is model-agnostic: any ASR system can be scored if its outputs are
converted to the benchmark prediction format.

## Current Results 🏆

Tables are sorted by **Disfluency F1**, the headline metric for detecting
disfluency-related events. Here, **Disfluency F1** is the benchmark's `event_f1`
reported under a clearer name, and **Cutoff F1** uses the lenient positional
cutoff score.

The `n` column shows how many cached predictions were scored. Gradium English is
reported on the subset of entries with existing predictions.

### English

| Rank | Model | n | Disfluency F1 ↑ | Filler F1 ↑ | Vocal Sound F1 ↑ | Cutoff F1 ↑ | Rep F1 ↑ | vWER ↓ |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | ![ElevenLabs](https://img.shields.io/badge/-ElevenLabs-000000?style=flat-square&logo=elevenlabs&logoColor=white) **ElevenLabs Scribe v2** | 4957 | **92.9%** | 95.5% | 83.4% | 80.1% | 87.9% | 3.2% |
| 2 | ![Inworld](https://img.shields.io/badge/-Inworld-FF6B00?style=flat-square) Inworld STT (inworld/inworld-stt-1) | 4957 | 83.3% | 95.2% | 10.7% | 84.8% | 86.8% | 4.0% |
| 3 | ![Microsoft](https://img.shields.io/badge/-Microsoft-5E5E5E?style=flat-square&logo=microsoft&logoColor=white) MAI-Transcribe-1.5 (Microsoft, mai-transcribe-1.5) | 4957 | 83.0% | 95.8% | 0.0% | 83.3% | 86.8% | 3.6% |
| 4 | ![AssemblyAI](https://img.shields.io/badge/-AssemblyAI-111111?style=flat-square&logo=assemblyai&logoColor=white) AssemblyAI Universal-3 Pro (universal-3-pro, Prompt 2) | 4957 | 78.2% | 90.8% | 2.8% | 27.9% | 82.7% | 4.5% |
| 5 | ![AssemblyAI](https://img.shields.io/badge/-AssemblyAI-111111?style=flat-square&logo=assemblyai&logoColor=white) AssemblyAI Universal-3 Pro (universal-3-pro, Prompt 1) | 249 | 75.2% | 92.5% | 0.0% | 28.6% | 79.7% | 3.8% |
| 6 | ![Nyra Health](https://img.shields.io/badge/-Nyra%20Health-0F766E?style=flat-square) CrisperWhisper 1.0 (nyrahealth/CrisperWhisper) | 4957 | 73.2% | 81.8% | 0.0% | 81.1% | 67.9% | 6.2% |
| 7 | ![xAI](https://img.shields.io/badge/-xAI-111111?style=flat-square) xAI Grok Speech-to-Text | 4957 | 72.2% | 84.4% | 0.0% | 60.3% | 83.4% | 4.7% |
| 8 | ![Deepgram](https://img.shields.io/badge/-Deepgram-13EF93?style=flat-square&logo=deepgram&logoColor=black) Deepgram Nova-3 (nova-3) | 4957 | 37.0% | 45.7% | 0.0% | 72.2% | 83.2% | 6.3% |
| 9 | ![Fish Audio](https://img.shields.io/badge/-Fish%20Audio-2563EB?style=flat-square) Fish Audio ASR | 4957 | 24.3% | 30.5% | 0.0% | 60.1% | 82.1% | 5.9% |
| 10 | ![NVIDIA](https://img.shields.io/badge/-NVIDIA-76B900?style=flat-square&logo=nvidia&logoColor=white) NVIDIA Canary 1B v2 (nvidia/canary-1b-v2) | 4957 | 21.0% | 26.7% | 0.0% | 48.9% | 30.5% | 9.0% |
| 11 | ![Cohere](https://img.shields.io/badge/-Cohere-39594D?style=flat-square&logo=cohere&logoColor=white) Cohere Transcribe (CohereLabs/cohere-transcribe-03-2026) | 4957 | 12.4% | 15.8% | 0.0% | 45.0% | 18.9% | 9.9% |
| 12 | ![Gradium](https://img.shields.io/badge/-Gradium-7C3AED?style=flat-square) Gradium ASR | 616 | 7.6% | 10.3% | 0.0% | 45.4% | 62.7% | 9.6% |
| 13 | ![OpenAI](https://img.shields.io/badge/-OpenAI-412991?style=flat-square&logo=openai&logoColor=white) Whisper Large v3 (openai/whisper-large-v3) | 4957 | 7.3% | 9.4% | 0.0% | 31.4% | 6.4% | 10.2% |
| 14 | ![Cartesia](https://img.shields.io/badge/-Cartesia-5B5BD6?style=flat-square) Cartesia Ink-Whisper (ink-whisper) | 4957 | 4.7% | 6.0% | 0.0% | 32.5% | 3.7% | 10.4% |

### German

| Rank | Model | n | Disfluency F1 ↑ | Filler F1 ↑ | Vocal Sound F1 ↑ | Cutoff F1 ↑ | Rep F1 ↑ | vWER ↓ |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | ![ElevenLabs](https://img.shields.io/badge/-ElevenLabs-000000?style=flat-square&logo=elevenlabs&logoColor=white) **ElevenLabs Scribe v2** | 202 | **94.3%** | 98.5% | 80.4% | 77.7% | 88.1% | 4.2% |
| 2 | ![Microsoft](https://img.shields.io/badge/-Microsoft-5E5E5E?style=flat-square&logo=microsoft&logoColor=white) MAI-Transcribe-1.5 (Microsoft, mai-transcribe-1.5) | 202 | 82.5% | 97.5% | 0.0% | 89.5% | 88.9% | 6.6% |
| 3 | ![Nyra Health](https://img.shields.io/badge/-Nyra%20Health-0F766E?style=flat-square) CrisperWhisper 1.0 (nyrahealth/CrisperWhisper) | 202 | 57.7% | 70.0% | 0.0% | 61.9% | 60.5% | 15.2% |
| 4 | ![Inworld](https://img.shields.io/badge/-Inworld-FF6B00?style=flat-square) Inworld STT (inworld/inworld-stt-1) | 202 | 53.5% | 63.2% | 17.2% | 18.0% | 67.7% | 15.9% |
| 5 | ![xAI](https://img.shields.io/badge/-xAI-111111?style=flat-square) xAI Grok Speech-to-Text | 202 | 39.7% | 49.8% | 0.9% | 14.2% | 54.1% | 18.2% |
| 6 | ![AssemblyAI](https://img.shields.io/badge/-AssemblyAI-111111?style=flat-square&logo=assemblyai&logoColor=white) AssemblyAI Universal-3 Pro (universal-3-pro, Prompt 1) | 202 | 7.2% | 9.4% | 0.0% | 34.4% | 70.9% | 18.8% |
| 7 | ![Cohere](https://img.shields.io/badge/-Cohere-39594D?style=flat-square&logo=cohere&logoColor=white) Cohere Transcribe (CohereLabs/cohere-transcribe-03-2026) | 202 | 5.3% | 7.2% | 0.0% | 1.5% | 10.3% | 28.2% |
| 8 | ![AssemblyAI](https://img.shields.io/badge/-AssemblyAI-111111?style=flat-square&logo=assemblyai&logoColor=white) AssemblyAI Universal-3 Pro (universal-3-pro, Prompt 2) | 202 | 3.9% | 5.0% | 0.9% | 23.9% | 79.9% | 18.8% |
| 9 | ![NVIDIA](https://img.shields.io/badge/-NVIDIA-76B900?style=flat-square&logo=nvidia&logoColor=white) NVIDIA Canary 1B v2 (nvidia/canary-1b-v2) | 202 | 1.9% | 2.6% | 0.0% | 1.5% | 2.7% | 24.4% |
| 10 | ![Deepgram](https://img.shields.io/badge/-Deepgram-13EF93?style=flat-square&logo=deepgram&logoColor=black) Deepgram Nova-3 (nova-3) | 202 | 1.6% | 2.2% | 0.0% | 14.1% | 79.8% | 19.2% |
| 11 | ![Cartesia](https://img.shields.io/badge/-Cartesia-5B5BD6?style=flat-square) Cartesia Ink-Whisper (ink-whisper) | 202 | 1.4% | 1.9% | 0.0% | 1.0% | 4.1% | 24.1% |
| 12 | ![Fish Audio](https://img.shields.io/badge/-Fish%20Audio-2563EB?style=flat-square) Fish Audio ASR | 202 | 1.4% | 1.9% | 0.0% | 6.4% | 63.5% | 20.3% |
| 13 | ![OpenAI](https://img.shields.io/badge/-OpenAI-412991?style=flat-square&logo=openai&logoColor=white) Whisper Large v3 (openai/whisper-large-v3) | 202 | 1.4% | 1.9% | 0.0% | 0.5% | 4.0% | 24.2% |
| 14 | ![Gradium](https://img.shields.io/badge/-Gradium-7C3AED?style=flat-square) Gradium ASR | 202 | 1.2% | 1.6% | 0.0% | 6.4% | 13.7% | 23.5% |

## Table of Contents

- [Current Results](#current-results-)
- [Quick Start](#quick-start-)
- [Benchmark Workflow](#benchmark-workflow-)
- [Reproducing Results](#reproducing-results-)
- [Adding a New System](#adding-a-new-system-)
- [Verbatim Transcript Conventions](#verbatim-transcript-conventions-)
- [Intended Transcript](#intended-transcript)
- [How Gold Labels Are Derived](#how-gold-labels-are-derived-)
- [How Evaluation Works](#how-evaluation-works-)
- [Metrics](#metrics-)
- [Supported Datasets](#supported-datasets-)
- [Output Files](#output-files-)
- [Repository Structure](#repository-structure-)
- [License](#license)

---

## Quick Start

Reproduce the included benchmark tables:

```bash
pip install -r requirements.txt
python scripts/reproduce_results.py
```

Main outputs:

- `results/cached/summary.json`: machine-readable metrics for all cached runs.
- `results/cached/leaderboard.md`: markdown tables for English and German.
- `results/cached/<model>/<language>/report.json`: full per-model metric report.
- `results/cached/<model>/<language>/summary.md`: human-readable per-model summary.

## Benchmark Workflow

The benchmark uses a simple four-step workflow:

```text
raw model predictions
  -> canonical prediction format
  -> normalized benchmark format
  -> metrics and leaderboard
```

The raw predictions remain available for inspection. The normalized files are
the exact inputs used by the evaluator.

## Reproducing Results

The repository includes cached predictions for the systems listed in
`benchmark/models.json`. The quick-start command above runs the full cached
evaluation workflow. The same steps can be run separately:

```bash
python scripts/validate_predictions.py
python scripts/convert_outputs.py --skip_import
python scripts/evaluate_all_cached.py --output_dir results/cached
python scripts/print_leaderboard.py --output results/cached/leaderboard.md
```

Raw predictions live under:

```text
predictions/raw/<model>/<language>.json
```

Normalized evaluator-ready predictions are generated under:

```text
predictions/normalized/<model>/<language>.jsonl
```

Each prediction row contains the sample metadata, raw model output, normalized
prediction, and gold verbatim/intended transcripts:

```json
{
  "id": "DISFLUENCY_TEST_000000",
  "audio": "audio/DISFLUENCY_TEST_000000.wav",
  "dataset": "nyrahealth/disfluency_speech_english",
  "split": "test",
  "language": "en",
  "model": "elevenlabs",
  "raw_verbatim_prediction": "raw model output",
  "raw_intended_prediction": null,
  "pred_verbatim": "canonical verbatim prediction",
  "pred_intended": "intended prediction",
  "gold_verbatim": "gold verbatim transcript",
  "gold_intended": "gold intended transcript"
}
```

For single-output ASR systems, `pred_intended` is set equal to
`pred_verbatim`. For systems that produce separate verbatim and intended
transcripts, both fields can be populated separately.

To evaluate one cached file directly:

```bash
python evaluate.py \
    --predictions predictions/normalized/elevenlabs/en.jsonl \
    --use_prediction_gold \
    --output_dir results/elevenlabs_english_cached
```

Dataset-backed evaluation is also supported:

```bash
python evaluate.py \
    --predictions predictions/normalized/elevenlabs/en.jsonl \
    --datasets nyrahealth/disfluency_speech_english \
    --splits train validation test \
    --output_dir results/elevenlabs_english_hf
```

## Adding a New System

Adding a system requires three pieces:

1. Raw predictions:

```text
predictions/raw/<model_id>/<language>.json
```

2. A model entry in `benchmark/models.json`.

3. Convert, validate, and evaluate:

```bash
python scripts/convert_outputs.py --skip_import --model <model_id>
python scripts/validate_predictions.py
python scripts/evaluate_all_cached.py --model <model_id>
python scripts/print_leaderboard.py
```

For reproducible submissions, add a compact script under `generation/` that
loads the public Hugging Face datasets and writes the same raw prediction
schema. The existing generation scripts can be used as templates.

See [`docs/adding_a_model.md`](docs/adding_a_model.md) for the full schema and
checklist. See [`docs/normalization.md`](docs/normalization.md) for how output
formatting conventions are handled.

## Verbatim Transcript Conventions 📝

The verbatim transcript uses a small set of conventions:

- **Cutoffs** use `*` in the gold transcript: `th*`, `w*`
- **Fillers** are bracketed, e.g. `[UH]`, `[UM]`
- **Vocal sound events** are bracketed, e.g. `[laughter]`, `[cough]`, `[breath]`
- **Numbers and dates** are written out as spoken
- Otherwise, the transcript follows exactly what the speaker says


Example:

```text
I mean we we [UH] should go on th* Thursday [laughter]
```

The benchmark can also evaluate models that use a slightly different output
convention, for example `-` instead of `*` for cutoffs, as long as their output
can be mapped to the benchmark format.

Cutoffs are inherently a little ambiguous: different systems may spell the same
interrupted word with different cutoff markers. Cached model predictions are
therefore normalized to the benchmark's internal `-` cutoff marker before metric
calculation.

Because even normalized cutoff spelling is not always semantically clear, the
benchmark reports several cutoff metrics: marker-based cutoff F1, stricter
stem-matching cutoff F1, and lenient positional cutoff scores.

The current English and German reference transcripts contain these vocal sound
event tags:

- `[breath]`
- `[cough]`
- `[laughter]`
- `[lipsmack]`
- `[noise]`
- `[sigh]`
- `[sniff]`
- `[throatclearing]`
- `[yawn]`

These tags are supported by the benchmark. Systems that use different labels
for vocal sound events should map them to this canonical set before scoring. The
global normalizer already maps common variants such as `[laughs]` to
`[laughter]` and `[clears throat]` to `[throatclearing]`.

---

## Intended Transcript

The intended transcript is a cleaned version of the same utterance.

We generate intended transcripts by stripping fillers and vocal sound event
tags, collapsing repetitions and false starts to their repairs, removing
fragments, and reformatting numbers and dates for readability.

Example:

```text
verbatim: I mean we we [UH] should go on th* Thursday [laughter]
intended: we should go on Thursday
```

---

## How Gold Labels Are Derived 🔬

The benchmark does **not** require manually labeled disfluency tags. Instead, it derives them from the alignment between the gold verbatim and gold intended transcripts.

Start with:

```text
verbatim: I mean we we [UH] should go on th* Thursday [laughter]
intended: we should go on Thursday
```

### 1. Normalize Cutoffs

First, cutoffs are normalized internally so everything uses one convention:

```text
th* -> th-
w— -> w-
```

### 2. Align Gold Verbatim To Gold Intended

Then the benchmark aligns the gold verbatim transcript to the gold intended transcript:

```text
verbatim:  I   mean  we   we   [UH]  should  go  on  th-  Thursday  [laughter]
intended:  -    -    -    we    -    should  go  on   -   Thursday      -
```

What aligns is treated as intended content. What does not align is disfluent.

### 3. Derive Token Labels

From that alignment, labels are derived like this:

- bracketed fillers such as `[UH]`, `[UM]` -> `FILLER`
- bracketed vocal sound event tags such as `[laughter]` -> `SOUND`
- cutoff fragments such as `th-` -> `CUTOFF`
- aligned content words -> `FLUENT`
- repeated unmatched words immediately before a fluent keeper -> `REP`
- remaining unmatched disfluent material -> `OTHER`

For the running example, the final gold labels are:

```text
token: I     mean  we   we      [UH]   should  go   on   th-    Thursday  [laughter]
tag:   OTHER OTHER REP  FLUENT  FILLER FLUENT  FLUENT FLUENT CUTOFF FLUENT    SOUND
```

So the typed disfluencies show up explicitly in the alignment and labels. In this example, `I mean` is a false start and falls into `OTHER`, while the repeated `we` becomes `REP`.

---

## How Evaluation Works ⚖️

After gold labels are derived, the model prediction is aligned against the gold verbatim transcript.

For the **intended metrics**, the model should produce a **intended prediction**. That intended prediction is compared against the gold `intended_transcript`.

This is especially interesting for verbatim speech: it tests whether the model removes or ignores the **right** disfluencies while still preserving the intended content.

### 1. Align Gold Verbatim To Predicted Verbatim

This second alignment is **tag-aware**:

- matching a `FLUENT` token is preferred over matching a disfluent token
- deleting a disfluent token is preferred over deleting a `FLUENT` token

That matters in ambiguous cases like repetitions. For example:

```text
gold tags: REP  FLUENT FILLER FLUENT CUTOFF FLUENT FLUENT FLUENT SOUND FLUENT
gold:       I    I     [UH]  think  th-    the   answer   is   [laughter] yes
pred:       -    I     [UH]  think   -     the   answer   is   [laughter] yes
```

Here the predicted `I` is aligned to the **FLUENT** gold `I`, while the `REP` token is deleted. This is why the second alignment is tag-aware.

From this alignment the benchmark computes:

- event metrics for fillers, vocal sound events, cutoffs, and repetitions
- exact-label metrics for filler and vocal sound event labels
- diagnostics such as omission rates, hallucination rates, and tag confusion

### 2. Compute Transcript Metrics Separately

The transcript accuracy metrics are computed separately from the normalized transcripts themselves, not from this alignment:

- the **verbatim prediction** is compared against the **verbatim reference** for `vWER` and `vCER`
- the **intended prediction** is compared against the **intended reference** for `iWER` and `iCER`

---

## Metrics 📊

### Transcript Metrics

- `vWER`: word error rate between the normalized verbatim prediction and normalized verbatim reference.
- `vCER`: character error rate between the normalized verbatim prediction and normalized verbatim reference.
- `iWER`: word error rate between the normalized intended prediction and normalized intended reference.
- `iCER`: character error rate between the normalized intended prediction and normalized intended reference.
- `iSR`: intended substitution rate, i.e. the fraction of intended reference words replaced by the wrong word; this corresponds roughly to ordinary transcription errors.
- `iDR`: intended deletion rate, i.e. the fraction of intended reference words that are missing; this is a proxy for over-removal of actual content.
- `iIR`: intended insertion rate, i.e. the fraction of extra words added relative to the intended reference; this is a proxy for under-removal, where disfluent material is kept when it should have been removed.
- `fluent_deletion_rate`: the fraction of gold FLUENT tokens that are not covered by the prediction.

### Event Metrics

- `filler_f1`: F1 for detecting filler tokens such as `[UH]` and `[UM]`.
- `sound_f1`: F1 for detecting vocal sound event tokens such as `[laughter]` and `[cough]`.
- `event_f1`: combined F1 for fillers and vocal sound events together.
- `cutoff_f1`: F1 for detecting cutoff fragments such as `th-`; here the prediction must itself be marked as a cutoff.
- `cutoff_strict_f1`: cutoff F1 where the cutoff stem itself must also match; here the prediction does not have to carry the cutoff marker if the aligned token has the correct stem.
- `cutoff_lenient_f1`: cutoff F1 where a gold cutoff is counted as covered if any predicted token aligns to that cutoff position; hallucinated predicted cutoff markers are still penalized.
- `cutoff_lenient_recall`: recall for whether a gold cutoff is at least acknowledged by some aligned prediction token; here the prediction does not need a cutoff marker.
- `rep_f1`: F1 for detecting repetition tokens.
- `exact_filler_f1`: filler F1 where the exact filler label must match, e.g. `[UH]` vs `[UM]`.
- `exact_sound_f1`: vocal sound event F1 where the exact label must match, e.g. `[laughter]` vs `[cough]`.
- `filler_by_label`: per-label filler scores, e.g. separate metrics for `uh` and `um`.
- `sound_by_label`: per-label vocal sound event scores, e.g. separate metrics for `laughter`, `cough`, `breath`, and `lipsmack`.

### Diagnostics

- `disfluency_omission_rate`: the fraction of gold disfluencies that the model drops completely.
- `disfluency_hallucination_rate`: the fraction of disfluencies the model adds that are not supported by the gold transcript.
- `tag_confusion`: confusion matrix showing which gold disfluency types are predicted as which output types.
- `tag_prf`: precision, recall, and F1 for each tag type individually.
- `rep_count_mae`: mean absolute error in the predicted number of repetition groups per utterance.
- `rep_count_exact`: fraction of utterances where the repetition-group count is exactly correct.
- `rep_count_within1`: fraction of utterances where the repetition-group count is off by at most one.

---

## Supported Datasets 📚

Any HuggingFace audio dataset with:

- `id`
- `audio`
- `verbatim_transcript`
- `intended_transcript`

Local `load_from_disk` datasets are also supported.

Current datasets:

- [`nyrahealth/disfluency_speech_english`](https://huggingface.co/datasets/nyrahealth/disfluency_speech_english)
- [`nyrahealth/disfluency_speech_german`](https://huggingface.co/datasets/nyrahealth/disfluency_speech_german)

---

## Output Files 📦

`report.json` contains the full metric output.

`predictions.jsonl` contains one prediction per sample, for example:

```json
{"id":"sample_001","pred_verbatim":"I [UH] think the answer is yes","pred_intended":"I think the answer is yes"}
```

---

## Repository Structure 🗂️

```text
nyra_verbatim_speech_benchmark/
├── README.md
├── requirements.txt
├── evaluate.py
├── normalize.py
├── align.py
├── tag.py
├── benchmark/
│   ├── models.json
│   └── schemas/
├── scripts/
│   ├── convert_outputs.py
│   ├── download_datasets.py
│   ├── evaluate_all_cached.py
│   ├── print_leaderboard.py
│   ├── reproduce_results.py
│   └── validate_predictions.py
├── predictions/
│   ├── raw/
│   └── normalized/
├── generation/
├── docs/
└── examples/
```

---

## License

MIT
