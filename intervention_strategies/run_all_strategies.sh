#!/bin/bash

# Run all intervention strategies across all API models, then evaluate.
#
# Usage:
#   bash intervention_strategies/run_all_strategies.sh                    # all strategies, all models, sequential
#   PARALLEL=true bash intervention_strategies/run_all_strategies.sh      # all strategies, all models, parallel per strategy
#   MODEL=gpt-4o bash intervention_strategies/run_all_strategies.sh       # all strategies, single model
#   STRATEGIES="cot self_eval" bash intervention_strategies/run_all_strategies.sh  # specific strategies only
#   SKIP_EVAL=true bash intervention_strategies/run_all_strategies.sh     # skip evaluation step

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Configuration ----
ALL_STRATEGIES=("cot" "instruction" "eval_patient" "self_eval")
STRATEGIES=(${STRATEGIES:-${ALL_STRATEGIES[@]}})
PARALLEL="${PARALLEL:-true}"
SKIP_EVAL="${SKIP_EVAL:-false}"

echo "============================================"
echo "  Intervention Strategies - Full Pipeline"
echo "============================================"
echo "  Strategies : ${STRATEGIES[*]}"
echo "  Group      : ${GROUP:-api}"
echo "  Model      : ${MODEL:-all models in group}"
echo "  Parallel   : $PARALLEL"
echo "  Skip Eval  : $SKIP_EVAL"
echo "============================================"
echo ""

# ---- Step 1: Generate responses for each strategy ----
for strategy in "${STRATEGIES[@]}"; do
    echo ""
    echo "########################################"
    echo "  Strategy: $strategy"
    echo "########################################"
    echo ""

    MODE="$strategy" \
    MODEL="${MODEL:-}" \
    GROUP="${GROUP:-api}" \
    PARALLEL="$PARALLEL" \
    bash "$SCRIPT_DIR/generate_responses.sh"
done

# ---- Step 2: Evaluate responses ----
if [ "$SKIP_EVAL" = "true" ]; then
    echo ""
    echo "Skipping evaluation (SKIP_EVAL=true)"
else
    echo ""
    echo "########################################"
    echo "  Evaluating all strategies"
    echo "########################################"
    echo ""

    for strategy in "${STRATEGIES[@]}"; do
        echo ""
        echo "--- Evaluating: $strategy ---"
        MODE="$strategy" bash "$SCRIPT_DIR/evaluate_responses.sh"
    done
fi

echo ""
echo "============================================"
echo "  All done!"
echo "============================================"
echo ""
echo "Results:"
for strategy in "${STRATEGIES[@]}"; do
    echo "  $strategy:"
    echo "    Generated -> $SCRIPT_DIR/$strategy/"
    if [ "$SKIP_EVAL" != "true" ]; then
        echo "    Evaluated -> $SCRIPT_DIR/eval_results/$strategy/"
    fi
done
