#!/usr/bin/env bash
# =============================================================================
# run_api_models.sh
#
# Generate doctor responses for negative cases using 7 API-only models.
# Input : Negative_cases/Negative_sampling_segment/{DS}_negative_cases_sampled.json
# Output: Negative_cases/negative_generate/generated/{model}_{dataset}.jsonl
#
# Usage:
#   bash Negative_cases/run_api_models.sh
# =============================================================================

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SAMPLED_DIR="$SCRIPT_DIR/Negative_sampling_segment"
GENERATED_DIR="$SCRIPT_DIR/negative_generate/generated"
GEN_SCRIPT="$ROOT/model_generator/generate_response.py"

mkdir -p "$GENERATED_DIR"

# ── 7 API-only doctor models ──────────────────────────────────────────────────
API_MODELS=(
    "gpt-5"
    "gpt-4o-mini"
    "gpt-4"
    "claude-sonnet-4-5-20250929"
    "gemini-2.5-flash"
    "deepseek-chat"
    "deepseek-reasoner"
)

# Per-model concurrency — lower = fewer 429 rate-limit errors
get_concurrency() {
    case "$1" in
        claude*)   echo "3" ;;   # Claude has tight RPM limits
        gemini*)   echo "3" ;;   # Gemini also rate-limits aggressively
        deepseek*) echo "5" ;;
        *)         echo "8" ;;   # OpenAI models tolerate higher concurrency
    esac
}

# dataset name → sampled JSON filename stem (IMCS21 file maps to IMCS dataset name)
DATASETS=("ACI" "IMCS" "MedDG" "MediTOD")
get_file_stem() {
    case "$1" in
        ACI)     echo "ACI_negative_cases_sampled" ;;
        IMCS)    echo "IMCS21_negative_cases_sampled" ;;
        MedDG)   echo "MedDG_negative_cases_sampled" ;;
        MediTOD) echo "MediTOD_negative_cases_sampled" ;;
    esac
}

N_MODELS=${#API_MODELS[@]}
N_DATASETS=${#DATASETS[@]}
MODEL_IDX=0

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " Generating doctor responses (async, per-model concurrency)"
echo " Input : $SAMPLED_DIR"
echo " Output: $GENERATED_DIR"
echo " Models: $N_MODELS  |  Datasets: $N_DATASETS"
echo "═══════════════════════════════════════════════════════════════"

for MODEL in "${API_MODELS[@]}"; do
    MODEL_IDX=$((MODEL_IDX + 1))
    CONCURRENCY="$(get_concurrency "$MODEL")"
    echo ""
    echo "┌─────────────────────────────────────────────────────────────"
    echo "│ Model [$MODEL_IDX/$N_MODELS]: $MODEL  (concurrency=$CONCURRENCY)"
    echo "└─────────────────────────────────────────────────────────────"

    for DS in "${DATASETS[@]}"; do
        FILE_STEM="$(get_file_stem "$DS")"
        INPUT_FILE="$SAMPLED_DIR/${FILE_STEM}.json"

        if [ ! -f "$INPUT_FILE" ]; then
            echo "  [WARN] Not found: $INPUT_FILE — skipping $DS"
            continue
        fi

        # Skip if output already exists (resume-friendly)
        MODEL_SAFE="${MODEL//:/_}"
        MODEL_SAFE="${MODEL_SAFE////_}"
        OUT_FILE="$GENERATED_DIR/${MODEL_SAFE}_${DS}.jsonl"
        if [ -f "$OUT_FILE" ]; then
            echo "  ⏭  $DS — already exists, skipping"
            continue
        fi

        echo "  ▶  $DS ..."
        python3 "$GEN_SCRIPT" \
            --dataset     "$DS" \
            --input_file  "$INPUT_FILE" \
            --model       "$MODEL" \
            --output-dir  "$GENERATED_DIR" \
            --concurrency "$CONCURRENCY"
    done
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " ✅ All done!  JSONL files in: $GENERATED_DIR"
echo "═══════════════════════════════════════════════════════════════"
