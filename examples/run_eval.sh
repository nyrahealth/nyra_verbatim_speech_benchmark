#!/usr/bin/env bash
# Example: evaluate CrisperWhisper on the English disfluency benchmark.
#
# Step 1 — Transcribe
# Step 2 — Evaluate
# Or use run.py for both in one command.

set -euo pipefail

MODEL="/path/to/CrisperWhisper"
DATASETS="nyrahealth/disfluency_speech_english nyrahealth/disfluency_speech_german"
SPLITS="test"
OUTPUT_DIR="results/crisperwhisper"

# ── Option A: Two-step workflow ─────────────────────────────────────────────

# Step 1: Transcribe
python transcribe.py \
    --model "$MODEL" \
    --datasets $DATASETS \
    --splits $SPLITS \
    --modes verbatim intended \
    --tag_count 5 \
    --batch_size 16 \
    --num_beams 1 \
    --output "$OUTPUT_DIR/predictions.jsonl"

# Step 2: Evaluate
python evaluate.py \
    --predictions "$OUTPUT_DIR/predictions.jsonl" \
    --datasets $DATASETS \
    --splits $SPLITS \
    --cutoff_source "*" \
    --cutoff_target "-" \
    --output_dir "$OUTPUT_DIR"

# ── Option B: One-shot ──────────────────────────────────────────────────────
# python run.py \
#     --model "$MODEL" \
#     --datasets $DATASETS \
#     --splits $SPLITS \
#     --output_dir "$OUTPUT_DIR"
