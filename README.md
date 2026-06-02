# Nyra Verbatim Speech Benchmark


Evaluate **verbatim transcription** in detail and distinguish between different failure modes, especially:

- sounds
- fillers
- cutoffs
- repetitions

It can be used to evaluate how accurately ASR models transcribe different kinds of disfluencies.

Any model that produces transcripts following the [verbatim transcript conventions](#verbatim-transcript-conventions), or whose outputs can be mapped to those conventions, can be evaluated with this benchmark.

The **evaluation** part of the benchmark is model-agnostic. The current `transcribe.py` implementation is specifically written for **CrisperWhisper 2.0 only**. For other models, transcription has to be implemented separately.

The benchmark uses paired gold transcripts:

- `verbatim_transcript`: what the speaker actually said
- `intended_transcript`: a cleaned version of what the speaker meant to say

From that pair, the benchmark derives gold disfluency labels automatically and evaluates model predictions against them.

## Index

- [Quick Start](#quick-start-)
- [Cached Prediction Evaluation](#cached-prediction-evaluation)
- [Adding a New Model](#adding-a-new-model)
- [Normalization Philosophy](#normalization-philosophy)
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

## Quick Start 🚀

The commands below use the current built-in transcription path, which is intended for **CrisperWhisper 2.0 models**.

```bash
pip install -r requirements.txt

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

Outputs:

- `results/report.json`
- `results/summary.md`

If you want to evaluate a different model, the easiest path is:

1. write raw predictions in the fixed raw prediction schema
2. add the model to `benchmark/models.json`
3. run conversion and evaluation scripts

To be usable by `evaluate.py`, each line in `predictions.jsonl` should contain at least:

```json
{
  "id": "sample_001",
  "pred_verbatim": "I [UH] think th- the answer is yes",
  "pred_intended": "I think the answer is yes"
}
```

Notes:

- `id` must match the dataset sample id
- `pred_verbatim` is compared against the gold `verbatim_transcript` and should follow the verbatim transcription conventions described below
- `pred_intended` is compared against the gold `intended_transcript`
- if a model cannot produce intended transcripts yet, you would need to add that transcription path before the intended metrics are meaningful

---

## Cached Prediction Evaluation

The repository can also evaluate cached model outputs without running model
inference. This is the recommended path for reproducing benchmark tables across
many external ASR systems.

Raw model outputs live under:

```text
predictions/raw/<model>/<language>.json
```

Canonical evaluator-ready predictions are generated under:

```text
predictions/normalized/<model>/<language>.jsonl
```

Each normalized JSONL row contains:

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

Model metadata lives in:

```text
benchmark/models.json
```

That manifest tells the converter where raw files live, which adapter to use,
which languages are available, and which splits should be considered covered.

To regenerate normalized files from raw cached JSONs:

```bash
python scripts/convert_outputs.py --skip_import
```

To evaluate every cached model output:

```bash
python scripts/evaluate_all_cached.py \
    --output_dir results/cached
```

To evaluate one cached file directly:

```bash
python evaluate.py \
    --predictions predictions/normalized/elevenlabs/en.jsonl \
    --use_prediction_gold \
    --output_dir results/elevenlabs_english_cached
```

If you prefer to evaluate against datasets downloaded from Hugging Face rather
than the gold transcripts embedded in the cached JSONL files, use the existing
dataset-backed mode:

```bash
python evaluate.py \
    --predictions predictions/normalized/elevenlabs/en.jsonl \
    --datasets nyrahealth/disfluency_speech_english \
    --splits train validation test \
    --output_dir results/elevenlabs_english_hf
```

Hugging Face will use its normal cache directory. You can pre-download datasets
with:

```bash
python scripts/download_datasets.py
```

---

## Adding a New Model

Adding a model should not require changing the evaluator.

1. Create raw predictions:

```text
predictions/raw/<model_id>/<language>.json
```

2. Add a model entry to `benchmark/models.json`.

3. Convert, validate, and evaluate:

```bash
python scripts/convert_outputs.py --skip_import --model <model_id>
python scripts/validate_predictions.py
python scripts/evaluate_all_cached.py --model <model_id>
python scripts/print_leaderboard.py
```

See [`docs/adding_a_model.md`](docs/adding_a_model.md) for the full schema and
checklist.

---

## Normalization Philosophy

The benchmark separates raw model outputs from normalized benchmark inputs.
Contributors submit auditable raw predictions, and the benchmark applies a
deterministic normalizer before scoring.

```text
raw model transcript
  -> fixed raw prediction schema
  -> global normalizer + optional source adapter
  -> normalized prediction JSONL
  -> evaluate.py
  -> report.json / leaderboard
```

Rules belong in the global normalizer only when they are conservative and
idempotent across models. Provider-specific or risky rules belong in a source
adapter. For example, `[laughs] -> [laughter]` is global, but AssemblyAI's
observed `ah -> [laughter]` behavior is adapter-only.

See [`docs/normalization.md`](docs/normalization.md) for details.

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
