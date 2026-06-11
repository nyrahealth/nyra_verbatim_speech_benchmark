#!/bin/bash
# Generate raw predictions for the pro "with timing" CrisperWhisper checkpoints
# across ALL benchmark languages (en, de + synthetic multilingual), running
# (model, language) jobs in parallel with a concurrency cap.
#
# Usage:
#   MAX_JOBS=9 ./scripts/run_pro_with_timing_generation.sh pro-turbo-with-timing pro-small-with-timing pro-medium-with-timing
#   ./scripts/run_pro_with_timing_generation.sh pro-large-v2-with-timing

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK="$(cd "$SCRIPT_DIR/.." && pwd)"
RESEARCH="/home/ubuntu/code/research"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/finetune_whisper/bin/python}"
LOG_DIR="$BENCHMARK/logs/pro_with_timing"
MAX_JOBS="${MAX_JOBS:-9}"

ALL_LANGS=(en de es nl fr it pl pt ru sv)

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$RESEARCH:${PYTHONPATH:-}"
export CRISPERWHISPER_BACKEND="${CRISPERWHISPER_BACKEND:-transformers}"

mkdir -p "$LOG_DIR"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_DIR/runner.log"; }

model_path() {
    case "$1" in
        pro-turbo-with-timing)    echo "/home/ubuntu/data2/pro_models/turbo_verbatimize_longform_pro_with_timing" ;;
        pro-small-with-timing)    echo "/home/ubuntu/data2/pro_models/small_verbatimize_longform_pro_with_timing" ;;
        pro-medium-with-timing)   echo "/home/ubuntu/data2/pro_models/medium_verbatimize_longform_pro_with_timing" ;;
        pro-large-v2-with-timing) echo "/home/ubuntu/data2/pro_models/large-v2_verbatimize_longform_pro_with_timing" ;;
        *) echo "" ;;
    esac
}

splits_for() {
    # English benchmarks on train+validation+test (matches existing en raw);
    # every other language uses the test split only.
    if [ "$1" = "en" ]; then echo "train validation test"; else echo "test"; fi
}

gen_one() {
    local model="$1" lang="$2" model_path="$3"
    local out="$BENCHMARK/predictions/raw/$model/$lang.json"
    local logf="$LOG_DIR/${model}_${lang}.log"
    mkdir -p "$(dirname "$out")"
    log "START $model/$lang"
    # shellcheck disable=SC2046
    "$PYTHON" "$BENCHMARK/generation/generate_crisperwhisper_2.py" \
        --model_id "$model" \
        --model "$model_path" \
        --language "$lang" \
        --splits $(splits_for "$lang") \
        --output_path "$out" \
        --device cuda \
        --compute_type float16 \
        --backend "${CRISPERWHISPER_BACKEND}" \
        --cache_dir /home/ubuntu/data2/crisperwhisper_cache \
        --crisperwhisper_python_path "$RESEARCH" \
        --ctranslate2_python_path "" \
        >> "$logf" 2>&1 \
        && log "DONE $model/$lang" \
        || log "FAIL $model/$lang"
}

TARGETS=("$@")
[ "${#TARGETS[@]}" -eq 0 ] && { echo "No models given" >&2; exit 1; }

log "=== PRO WITH-TIMING GENERATION (MAX_JOBS=$MAX_JOBS) ==="
log "Targets: ${TARGETS[*]}"

for model in "${TARGETS[@]}"; do
    mp="$(model_path "$model")"
    if [ -z "$mp" ]; then log "Unknown model: $model"; continue; fi
    for lang in "${ALL_LANGS[@]}"; do
        # Throttle: wait until a slot frees up.
        while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do wait -n; done
        gen_one "$model" "$lang" "$mp" &
    done
done

wait
log "=== COMPLETE: ${TARGETS[*]} ==="
