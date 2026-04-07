# Nyra Verbatim Speech Benchmark

`verbatim-speech-benchmark` is the **Nyra Verbatim Speech Benchmark**.

It is used to evaluate **verbatim transcription** in detail and distinguish between different failure modes in verbatim transcription, especially:

- sounds
- fillers
- cutoffs
- repetitions

It evaluates verbatim ASR models: models that transcribe not only the intended words, but also the disfluencies and sound events that the speaker actually produced.

Any model that produces transcripts in these verbatim conventions, or can be mapped to them, can be evaluated with this benchmark.

The **evaluation** part of the benchmark is model-agnostic. The current `transcribe.py` implementation is specifically written for **CrisperWhisper 2.0 only**. For other models, transcription has to be implemented separately.

The benchmark uses paired gold transcripts:

- `verbatim_transcript`: what the speaker actually said
- `intended_transcript`: a cleaned version of what the speaker meant to say

From that pair, the benchmark derives gold disfluency labels automatically and evaluates model predictions against them.

---

## Quick Start

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

1. implement a small transcription adapter that writes `predictions.jsonl`
2. keep using `evaluate.py` unchanged

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

## Verbatim Transcript Conventions

The verbatim transcript uses a small set of conventions:

- **Cutoffs** use `*`: `th*`, `w*`
- **Fillers** are bracketed, e.g. `[UH]`, `[UM]`
- **Sound events** are bracketed, e.g. `[laughter]`, `[cough]`, `[breath]`
- **Numbers and dates** are written out as spoken
- Otherwise, the transcript follows exactly what the speaker says

Example:

```text
I I [UH] think th* the answer is [laughter] yes
```

The benchmark can also evaluate models that use a slightly different output convention, for example `-` instead of `*` for cutoffs, as long as their output can be mapped to the benchmark format.

---

## Intended Transcript

The intended transcript is a cleaned version of the same utterance.

We generate intended transcripts by stripping fillers and sound tags, collapsing repetitions and false starts to their repairs, removing fragments, and reformatting numbers and dates for readability.

Example:

```text
verbatim: I I [UH] think th* the answer is [laughter] yes
intended: I think the answer is yes
```

---

## How Gold Labels Are Derived

The benchmark does **not** require manually labeled disfluency tags. Instead, it derives them from the alignment between the gold verbatim and gold intended transcripts.

Start with:

```text
verbatim: I I [UH] think th* the answer is [laughter] yes
intended: I think the answer is yes
```

First, cutoffs are normalized internally so everything uses one convention:

```text
th* -> th-
```

Then the benchmark aligns the gold verbatim transcript to the gold intended transcript:

```text
verbatim:  I   I   [UH]  think  th-  the  answer  is  [laughter]  yes
intended:  -   I    -    think   -   the  answer   is      -       yes
```

What aligns is treated as intended content. What does not align is disfluent.

From that alignment, labels are derived like this:

- bracketed fillers such as `[UH]`, `[UM]` -> `FILLER`
- bracketed sound tags such as `[laughter]` -> `SOUND`
- cutoff fragments such as `th-` -> `CUTOFF`
- aligned content words -> `FLUENT`
- repeated unmatched words immediately before a fluent keeper -> `REP`
- remaining unmatched disfluent material -> `OTHER`

For the running example, the final gold labels are:

```text
token: I    I      [UH]   think   th-    the    answer   is    [laughter]  yes
tag:   REP  FLUENT FILLER FLUENT  CUTOFF FLUENT FLUENT  FLUENT SOUND       FLUENT
```

So the typed disfluencies show up explicitly in the alignment and labels. The rest, such as false starts that are not one of the typed categories, remains disfluent and falls into `OTHER`.

---

## How Evaluation Works

After gold labels are derived, the model prediction is aligned against the gold verbatim transcript.

For the **intended metrics**, the model should produce a **intended prediction**. That intended prediction is compared against the gold `intended_transcript`.

This is especially interesting for verbatim speech: it tests whether the model removes or ignores the **right** disfluencies while still preserving the intended content.

This second alignment is **tag-aware**:

- matching a `FLUENT` token is preferred over matching a disfluent token
- deleting a disfluent token is preferred over deleting a `FLUENT` token

That matters in ambiguous cases like repetitions:

```text
gold tags: REP  FLUENT FILLER FLUENT CUTOFF FLUENT FLUENT FLUENT SOUND FLUENT
gold:       I    I     [UH]  think  th-    the   answer   is   [laughter] yes
pred:       -    I     [UH]  think   -     the   answer   is   [laughter] yes
```

Here the predicted `I` is aligned to the **FLUENT** gold `I`, while the `REP` token is deleted.

From this alignment the benchmark computes:

- event metrics for fillers, sounds, cutoffs, and repetitions
- exact-label metrics for filler and sound labels
- diagnostics such as omission rates, hallucination rates, and tag confusion

The transcript accuracy metrics are computed separately from the normalized transcripts themselves, not from this alignment:

- the **verbatim prediction** is compared against the **verbatim reference** for `vWER` and `vCER`
- the **intended prediction** is compared against the **intended reference** for `iWER` and `iCER`

---

## Metrics

### Transcript Metrics

- `vWER`, `vCER`: error rate against gold verbatim
- `iWER`, `iCER`: error rate between the model's **predicted intended transcript** and the gold intended transcript
- `iSR`, `iDR`, `iIR`: intended substitution, deletion, and insertion rates
- `fluent_deletion_rate`: how often gold fluent words are missing

### Event Metrics

- `filler_f1`
- `sound_f1`
- `event_f1`
- `cutoff_f1`
- `cutoff_strict_f1`
- `cutoff_lenient_recall`
- `rep_f1`
- `exact_filler_f1`
- `exact_sound_f1`

### Diagnostics

- `disfluency_omission_rate`
- `disfluency_hallucination_rate`
- `tag_confusion`
- `tag_prf`
- repetition count metrics

---

## Supported Datasets

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

## Output Files

`report.json` contains the full metric output.

`predictions.jsonl` contains one prediction per sample, for example:

```json
{"id":"sample_001","pred_verbatim":"I [UH] think the answer is yes","pred_intended":"I think the answer is yes"}
```

---

## Repository Structure

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
└── examples/
```

---

## License

MIT