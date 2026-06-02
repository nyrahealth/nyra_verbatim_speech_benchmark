# Nyra Verbatim Speech Benchmark

**A benchmark for evaluating verbatim automatic speech recognition: not only the words, but also fillers, repetitions, cutoffs, and non-speech sounds.**

Most speech benchmarks focus on clean transcript accuracy. This repository focuses on the messy but important details of real speech: `uh`, `um`, repeated words, interrupted words like `th-`, and events like `[laughter]` or `[cough]`.

The benchmark compares two kinds of transcripts:

- **Verbatim transcript**: what the speaker actually said, including disfluencies.
- **Intended transcript**: a cleaned version of what the speaker meant to say.

From that pair, the evaluator automatically derives disfluency labels and reports detailed ASR quality metrics. The goal is to make verbatim ASR evaluation reproducible, comparable, and easy to extend across systems.

## Why This Exists ✨

Verbatim ASR is not just ordinary word error rate. A model can have low WER while still dropping the exact phenomena that matter for clinical, conversational, and speech-science use cases.

This benchmark therefore reports both transcript-level quality and typed
disfluency scores:

- **Transcript quality**: `vWER`, `iWER`, `vCER`, `iCER`
- **Fillers**: `[UH]`, `[UM]`, including exact and per-label scores
- **Sounds**: `[laughter]`, `[cough]`, `[breath]`, and related events
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
- Cached model families include AssemblyAI, Canary, Cartesia, Cohere, CrisperWhisper, Deepgram, ElevenLabs, Fish Audio, Gradium, Inworld, Whisper Large v3, and xAI Grok.

The evaluator is model-agnostic: any ASR system can be scored if its outputs are
converted to the benchmark prediction format.

## Table of Contents

- [Quick Start](#quick-start-)
- [Benchmark Workflow](#benchmark-workflow-)
- [Reproducing Results](#reproducing-results-)
- [Adding a New System](#adding-a-new-system-)
- [Optional: Run CrisperWhisper Inference](#optional-run-crisperwhisper-inference-)
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

See [`docs/adding_a_model.md`](docs/adding_a_model.md) for the full schema and
checklist. See [`docs/normalization.md`](docs/normalization.md) for how output
formatting conventions are handled.

## Optional: Run CrisperWhisper Inference 🎙️

The commands below use the built-in transcription path, which is intended for
**CrisperWhisper 2.0 models**.

```bash
python run.py \
    --model /path/to/CrisperWhisper_2.0_model \
    --datasets nyrahealth/disfluency_speech_english \
    --splits test \
    --output_dir results/
```

Or run transcription and evaluation separately:

```bash
python transcribe.py \
    --model /path/to/CrisperWhisper_2.0_model \
    --datasets nyrahealth/disfluency_speech_english \
    --splits test \
    --modes verbatim intended \
    --output predictions.jsonl

python evaluate.py \
    --predictions predictions.jsonl \
    --datasets nyrahealth/disfluency_speech_english \
    --splits test \
    --output_dir results/
```

To be usable by `evaluate.py`, each line in `predictions.jsonl` should contain
at least:

```json
{
  "id": "sample_001",
  "pred_verbatim": "I [UH] think th- the answer is yes",
  "pred_intended": "I think the answer is yes"
}
```

Notes:

- `id` must match the dataset sample id.
- `pred_verbatim` is compared against the gold `verbatim_transcript`.
- `pred_intended` is compared against the gold `intended_transcript`.
- If a model cannot produce intended transcripts, intended metrics are less meaningful.

---

## Verbatim Transcript Conventions 📝

The verbatim transcript uses a small set of conventions:

- **Cutoffs** use `*` in the gold transcript: `th*`, `w*`
- **Fillers** are bracketed, e.g. `[UH]`, `[UM]`
- **Sound events** are bracketed, e.g. `[laughter]`, `[cough]`, `[breath]`
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
interrupted word as `w*`, `w-`, `w—`, `w–`, or even use another Unicode
dash/minus-like symbol. Cached model predictions are therefore normalized to the
benchmark's internal `-` cutoff marker before metric calculation. The normalizer
handles common Unicode dash variants (`‐`, `‑`, `‒`, `–`, `—`, `―`, `−`, `﹘`,
`﹣`, `－`) when they are attached to a word fragment.

Because even normalized cutoff spelling is not always semantically clear, the
benchmark reports several cutoff metrics: marker-based cutoff F1, stricter
stem-matching cutoff F1, and lenient positional cutoff scores.

---

## Intended Transcript

The intended transcript is a cleaned version of the same utterance.

We generate intended transcripts by stripping fillers and sound tags, collapsing repetitions and false starts to their repairs, removing fragments, and reformatting numbers and dates for readability.

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
- bracketed sound tags such as `[laughter]` -> `SOUND`
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

- event metrics for fillers, sounds, cutoffs, and repetitions
- exact-label metrics for filler and sound labels
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
- `sound_f1`: F1 for detecting sound-event tokens such as `[laughter]` and `[cough]`.
- `event_f1`: combined F1 for fillers and sound events together.
- `cutoff_f1`: F1 for detecting cutoff fragments such as `th-`; here the prediction must itself be marked as a cutoff.
- `cutoff_strict_f1`: cutoff F1 where the cutoff stem itself must also match; here the prediction does not have to carry the cutoff marker if the aligned token has the correct stem.
- `cutoff_lenient_f1`: cutoff F1 where a gold cutoff is counted as covered if any predicted token aligns to that cutoff position; hallucinated predicted cutoff markers are still penalized.
- `cutoff_lenient_recall`: recall for whether a gold cutoff is at least acknowledged by some aligned prediction token; here the prediction does not need a cutoff marker.
- `rep_f1`: F1 for detecting repetition tokens.
- `exact_filler_f1`: filler F1 where the exact filler label must match, e.g. `[UH]` vs `[UM]`.
- `exact_sound_f1`: sound F1 where the exact sound label must match, e.g. `[laughter]` vs `[cough]`.
- `filler_by_label`: per-label filler scores, e.g. separate metrics for `uh` and `um`.
- `sound_by_label`: per-label sound scores, e.g. separate metrics for `laughter`, `cough`, `breath`, and `lipsmack`.

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
├── transcribe.py
├── evaluate.py
├── normalize.py
├── align.py
├── tag.py
├── run.py
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
