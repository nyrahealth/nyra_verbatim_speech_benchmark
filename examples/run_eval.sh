#!/usr/bin/env bash
# Example: reproduce cached benchmark results and evaluate one cached model.

set -euo pipefail

# Reproduce all cached results.
python scripts/reproduce_results.py

# Evaluate a single cached model/language pair directly.
python evaluate.py \
    --predictions predictions/normalized/elevenlabs/en.jsonl \
    --use_prediction_gold \
    --output_dir results/elevenlabs_english_cached
