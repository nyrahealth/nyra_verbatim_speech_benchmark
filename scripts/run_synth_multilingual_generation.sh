#!/bin/bash
# Generate raw benchmark predictions on synthetic multilingual test corpora.
#
# Requires API keys in the environment (see below). GPU models use the local
# finetune_whisper conda env and CrisperWhisper from /home/ubuntu/code/research.
#
# Usage:
#   export DEEPGRAM_API_KEY=...
#   export GRADIUM_API_KEY=...
#   ...
#   ./scripts/run_synth_multilingual_generation.sh deepgram elevenlabs
#   ./scripts/run_synth_multilingual_generation.sh all
#   ./scripts/run_synth_multilingual_generation.sh pro-turbo-no-timing

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK="$(cd "$SCRIPT_DIR/.." && pwd)"
RESEARCH="/home/ubuntu/code/research"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/finetune_whisper/bin/python}"
LOG_DIR="$BENCHMARK/logs/synth_multilingual"
SPLITS=(test)
LANGS=(es nl fr it pl pt ru sv)

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$RESEARCH:${PYTHONPATH:-}"
export CRISPERWHISPER_BACKEND="${CRISPERWHISPER_BACKEND:-transformers}"

mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_DIR/runner.log"; }

is_complete() {
    local model="$1"
    local lang="$2"
    local out="$BENCHMARK/predictions/raw/$model/$lang.json"
    local expected
    expected=$(python3 - <<PY
import json
from pathlib import Path
lang = "$lang"
roots = {
    "es": "/home/ubuntu/data/spanish_speech/synth_verbatim_es",
    "nl": "/home/ubuntu/data/dutch_speech/synth_verbatim_nl",
    "fr": "/home/ubuntu/data/french_speech/synth_verbatim_fr",
    "it": "/home/ubuntu/data/italian_speech/synth_verbatim_it",
    "pl": "/home/ubuntu/data/polish_speech/synth_verbatim_pl",
    "pt": "/home/ubuntu/data/portuguese_speech/synth_verbatim_pt",
    "ru": "/home/ubuntu/data/russian_speech/synth_verbatim_ru",
    "sv": "/home/ubuntu/data/swedish_speech/synth_verbatim_sv",
}
labels = json.loads(Path(roots[lang], "labels.json").read_text())
print(sum(1 for row in labels if row.get("split") == "test"))
PY
)
    if [ ! -f "$out" ]; then
        return 1
    fi
    local count
    count=$(python3 -c "import json; print(sum(1 for r in json.load(open('$out')) if r.get('raw_verbatim_prediction')))")
    [ "$count" -ge "$expected" ]
}

run_api_model() {
    local model="$1"
    local script="$2"
    shift 2
    local extra_args=("$@")
    cd "$BENCHMARK"
    for lang in "${LANGS[@]}"; do
        if is_complete "$model" "$lang"; then
            log "SKIP $model/$lang (complete)"
            continue
        fi
        local out="$BENCHMARK/predictions/raw/$model/$lang.json"
        local logf="$LOG_DIR/${model}_${lang}.log"
        log "START $model/$lang"
        mkdir -p "$(dirname "$out")"
        "$PYTHON" "generation/$script" \
            --language "$lang" \
            --splits "${SPLITS[@]}" \
            --output_path "$out" \
            "${extra_args[@]}" \
            >> "$logf" 2>&1
        log "DONE $model/$lang"
    done
}

run_gpu_model() {
    local model="$1"
    local script="$2"
    shift 2
    local extra_args=("$@")
    cd "$BENCHMARK"
    for lang in "${LANGS[@]}"; do
        if is_complete "$model" "$lang"; then
            log "SKIP $model/$lang (complete)"
            continue
        fi
        local out="$BENCHMARK/predictions/raw/$model/$lang.json"
        local logf="$LOG_DIR/${model}_${lang}.log"
        log "START $model/$lang"
        mkdir -p "$(dirname "$out")"
        "$PYTHON" "generation/$script" \
            --language "$lang" \
            --splits "${SPLITS[@]}" \
            --output_path "$out" \
            "${extra_args[@]}" \
            >> "$logf" 2>&1
        log "DONE $model/$lang"
    done
}

run_crisperwhisper_model() {
    local model_id="$1"
    local model_path="$2"
    cd "$BENCHMARK"
    for lang in "${LANGS[@]}"; do
        if is_complete "$model_id" "$lang"; then
            log "SKIP $model_id/$lang (complete)"
            continue
        fi
        local out="$BENCHMARK/predictions/raw/$model_id/$lang.json"
        local logf="$LOG_DIR/${model_id}_${lang}.log"
        log "START $model_id/$lang"
        mkdir -p "$(dirname "$out")"
        "$PYTHON" generation/generate_crisperwhisper_2.py \
            --model_id "$model_id" \
            --model "$model_path" \
            --language "$lang" \
            --splits "${SPLITS[@]}" \
            --output_path "$out" \
            --device cuda \
            --compute_type float16 \
            --backend "${CRISPERWHISPER_BACKEND}" \
            --cache_dir /home/ubuntu/data2/crisperwhisper_cache \
            --crisperwhisper_python_path "$RESEARCH" \
            --ctranslate2_python_path "" \
            >> "$logf" 2>&1
        log "DONE $model_id/$lang"
    done
}

run_model() {
    local name="$1"
    case "$name" in
        deepgram)
            run_api_model deepgram generate_deepgram_nova3.py \
                --api_key "${DEEPGRAM_API_KEY:?DEEPGRAM_API_KEY required}"
            ;;
        gradium)
            run_api_model gradium generate_gradium_asr.py \
                --api_key "${GRADIUM_API_KEY:?GRADIUM_API_KEY required}"
            ;;
        inworld)
            run_api_model inworld generate_inworld_stt1.py \
                --api_key "${INWORLD_API_KEY:?INWORLD_API_KEY required}"
            ;;
        xai_grok)
            run_api_model xai_grok generate_xai_grok_stt.py \
                --api_key "${XAI_API_KEY:?XAI_API_KEY required}"
            ;;
        assemblyai_p1)
            run_api_model assemblyai_p1 generate_assemblyai_universal3.py \
                --api_key "${ASSEMBLYAI_API_KEY:?ASSEMBLYAI_API_KEY required}" \
                --prompt 1
            ;;
        assemblyai_p2)
            run_api_model assemblyai_p2 generate_assemblyai_universal3.py \
                --api_key "${ASSEMBLYAI_API_KEY:?ASSEMBLYAI_API_KEY required}" \
                --prompt 2
            ;;
        cartesia)
            run_api_model cartesia generate_cartesia_ink_whisper.py \
                --api_key "${CARTESIA_API_KEY:?CARTESIA_API_KEY required}"
            ;;
        elevenlabs)
            run_api_model elevenlabs generate_elevenlabs_scribe.py \
                --api_key "${ELEVENLABS_API_KEY:?ELEVENLABS_API_KEY required}"
            ;;
        mai_transcribe_15)
            run_api_model mai_transcribe_15 generate_mai_transcribe_15.py \
                --api_key "${MAI_TRANSCRIBE_15_KEY:?MAI_TRANSCRIBE_15_KEY required}" \
                --endpoint "${MAI_TRANSCRIBE_15_ENDPOINT:-https://eastus.api.cognitive.microsoft.com}"
            ;;
        fish_audio)
            if [ -z "${FISH_AUDIO_API_KEY:-}" ]; then
                log "SKIP fish_audio (FISH_AUDIO_API_KEY not set)"
                return 0
            fi
            run_api_model fish_audio generate_fish_audio_asr.py \
                --api_key "$FISH_AUDIO_API_KEY"
            ;;
        canary)
            run_gpu_model canary generate_canary_1b_v2.py --device cuda
            ;;
        cohere)
            run_gpu_model cohere generate_cohere_transcribe.py
            ;;
        crisperwhisper_1)
            run_gpu_model crisperwhisper_1 generate_crisperwhisper_1.py --device cuda
            ;;
        crisperwhisper_2)
            run_crisperwhisper_model crisperwhisper_2 \
                "/home/ubuntu/data/models/CrisperWhisper_2_0/verbatimize-longform-english-no-timing-large-v2"
            ;;
        whisper_large_v3)
            run_gpu_model whisper_large_v3 generate_whisper_large_v3.py --device cuda
            ;;
        pro-turbo-no-timing)
            run_crisperwhisper_model pro-turbo-no-timing \
                "/home/ubuntu/data2/pro_models/turbo_verbatimize_longform_pro_no_timing"
            ;;
        pro-small-no-timing)
            run_crisperwhisper_model pro-small-no-timing \
                "/home/ubuntu/data2/pro_models/small_verbatimize_longform_pro_no_timing"
            ;;
        pro-medium-no-timing)
            run_crisperwhisper_model pro-medium-no-timing \
                "/home/ubuntu/data2/pro_models/medium_verbatimize_longform_pro_no_timing"
            ;;
        pro-large-v2-no-timing)
            run_crisperwhisper_model pro-large-v2-no-timing \
                "/home/ubuntu/data2/pro_models/large-v2_verbatimize_longform_pro_no_timing"
            ;;
        release-turbo-no-timing)
            run_crisperwhisper_model release-turbo-no-timing \
                "/home/ubuntu/data2/release_models/turbo_verbatimize_longform_no_timing_release"
            ;;
        release-small-no-timing)
            run_crisperwhisper_model release-small-no-timing \
                "/home/ubuntu/data2/release_models/small_verbatimize_longform_no_timing_release"
            ;;
        release-medium-no-timing)
            run_crisperwhisper_model release-medium-no-timing \
                "/home/ubuntu/data2/release_models/medium_verbatimize_longform_no_timing_release"
            ;;
        release-large-v2-no-timing)
            run_crisperwhisper_model release-large-v2-no-timing \
                "/home/ubuntu/data2/release_models/large-v2_verbatimize_longform_no_timing_release"
            ;;
        release-turbo-with-timing)
            run_crisperwhisper_model release-turbo-with-timing \
                "/home/ubuntu/data2/release_models/turbo_verbatimize_longform_with_timing_release"
            ;;
        release-small-with-timing)
            run_crisperwhisper_model release-small-with-timing \
                "/home/ubuntu/data2/release_models/small_verbatimize_longform_with_timing_release"
            ;;
        release-medium-with-timing)
            run_crisperwhisper_model release-medium-with-timing \
                "/home/ubuntu/data2/release_models/medium_verbatimize_longform_with_timing_release"
            ;;
        release-large-v2-with-timing)
            run_crisperwhisper_model release-large-v2-with-timing \
                "/home/ubuntu/data2/release_models/large-v2_verbatimize_longform_with_timing_release"
            ;;
        *)
            log "Unknown model: $name"
            return 1
            ;;
    esac
}

ALL_MODELS=(
    deepgram gradium inworld xai_grok assemblyai_p1 assemblyai_p2
    cartesia elevenlabs mai_transcribe_15 fish_audio
    canary cohere crisperwhisper_1 crisperwhisper_2 whisper_large_v3
    pro-turbo-no-timing pro-small-no-timing pro-medium-no-timing pro-large-v2-no-timing
    release-turbo-no-timing release-small-no-timing release-medium-no-timing release-large-v2-no-timing
    release-turbo-with-timing release-small-with-timing release-medium-with-timing release-large-v2-with-timing
)

TARGETS=("$@")
if [ "${#TARGETS[@]}" -eq 0 ] || [ "${TARGETS[0]}" = "all" ]; then
    TARGETS=("${ALL_MODELS[@]}")
fi

log "=== SYNTH MULTILINGUAL GENERATION ==="
log "Targets: ${TARGETS[*]}"

for model in "${TARGETS[@]}"; do
    run_model "$model"
done

log "=== COMPLETE ==="
