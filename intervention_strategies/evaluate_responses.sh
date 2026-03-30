#!/bin/bash

# Evaluate intervention strategy outputs using LLM-as-Judge
#
# Usage:
#   bash intervention_strategies/evaluate_responses.sh                          # default: cot
#   MODE=instruction bash intervention_strategies/evaluate_responses.sh
#   MODE=self_eval JUDGE=gpt-4o bash intervention_strategies/evaluate_responses.sh
#   CONCURRENCY=32 bash intervention_strategies/evaluate_responses.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment if present
if [ -f "$ROOT_DIR/.venv/bin/activate" ]; then
    source "$ROOT_DIR/.venv/bin/activate"
fi

# ---- Configuration ----
MODE="${MODE:-cot}"                # cot | instruction | eval_patient | self_eval
JUDGE="${JUDGE:-gpt-4o}"           # Judge model
CONTEXT="${CONTEXT:-full_context}" # full_context | current_turn | min_turn
CONCURRENCY="${CONCURRENCY:-16}"   # Async concurrency limit
# Map mode to output directory name
case "$MODE" in
    eval_patient)  OUTPUT_SUBDIR="eval_patient" ;;
    *)             OUTPUT_SUBDIR="$MODE" ;;
esac

INPUT_DIR="$SCRIPT_DIR/$OUTPUT_SUBDIR"
OUTPUT_DIR="$SCRIPT_DIR/eval_results/$MODE"

mkdir -p "$OUTPUT_DIR"

echo "====================================="
echo "  Intervention Strategy Evaluation"
echo "====================================="
echo "  Mode       : $MODE"
echo "  Judge      : $JUDGE"
echo "  Context    : $CONTEXT"
echo "  Input Dir  : $INPUT_DIR"
echo "  Output Dir : $OUTPUT_DIR"
echo "  Concurrency: $CONCURRENCY"
echo "====================================="

python "$ROOT_DIR/evaluators/failure_rate_eval/batch_evaluate.py" \
    --judge_model_name "$JUDGE" \
    --judge_model_type openai \
    --context_mode "$CONTEXT" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --concurrency "$CONCURRENCY" \
    --skip_if_exists

echo ""
echo "Evaluation complete for mode: $MODE"
echo "Results saved to: $OUTPUT_DIR"
