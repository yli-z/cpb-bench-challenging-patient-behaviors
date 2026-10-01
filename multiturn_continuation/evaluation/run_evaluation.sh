#!/bin/bash
# Convenience script for running multi-turn continuation evaluation

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Change to project root
cd "$(dirname "$0")/../../" || exit

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Multi-turn Continuation Evaluator${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Default values
JUDGE_MODEL="gpt-4o"
JUDGE_TYPE="openai"
SAMPLE=""
ASYNC_FLAG=""
CONCURRENCY=""
OUTPUT_DIR="multiturn_continuation/evaluation/results/multiturn_continuation_results"
# INPUT_FILE="multiturn_continuation/output/api_models_generated.json"
# INPUT_FILE="multiturn_continuation/output/vllm_llama_70b_generated.json"
# INPUT_FILE="multiturn_continuation/output/vllm_qwen_generated.json"
INPUT_FILE="multiturn_continuation/output/vllm_llama_8b_generated.json"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_FILE="$2"
            shift 2
            ;;
        --judge)
            JUDGE_MODEL="$2"
            shift 2
            ;;
        --type)
            JUDGE_TYPE="$2"
            shift 2
            ;;
        --sample)
            SAMPLE="--sample $2"
            shift 2
            ;;
        --async)
            ASYNC_FLAG="--async"
            shift
            ;;
        --concurrency)
            CONCURRENCY="--concurrency $2"
            shift 2
            ;;
        --help)
            echo "Usage: bash multiturn_continuation/evaluation/run_evaluation.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --input FILE       Input JSON file (default: multiturn_continuation/output/api_models_generated.json)"
            echo "  --judge MODEL      Judge model name (default: gpt-4o)"
            echo "  --type TYPE        Judge model type: openai|claude|gemini (default: openai)"
            echo "  --sample N         Evaluate only N samples"
            echo "  --async            Use async parallel judge calls"
            echo "  --concurrency N    Max concurrent calls with --async (default: 5)"
            echo "  --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  bash multiturn_continuation/evaluation/run_evaluation.sh --sample 5"
            echo "  bash multiturn_continuation/evaluation/run_evaluation.sh --async --concurrency 10"
            echo "  bash multiturn_continuation/evaluation/run_evaluation.sh --judge claude-sonnet-4-5-20250929 --type claude --async"
            echo "  bash multiturn_continuation/evaluation/run_evaluation.sh --input multiturn_continuation/output/api_models_generated_partial.json"
            exit 0
            ;;
        *)
            echo -e "${YELLOW}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Display configuration
echo -e "${GREEN}Configuration:${NC}"
echo "  Input: $INPUT_FILE"
echo "  Judge Model: $JUDGE_MODEL ($JUDGE_TYPE)"
echo "  Output: $OUTPUT_DIR"
if [ -n "$ASYNC_FLAG" ]; then
    CONC_VAL=${CONCURRENCY:---concurrency 5}
    echo "  Mode: Async (concurrency: $(echo $CONC_VAL | awk '{print $2}'))"
else
    echo "  Mode: Sync"
fi
if [ -n "$SAMPLE" ]; then
    echo "  Sample: Yes ($(echo $SAMPLE | awk '{print $2}') cases)"
else
    echo "  Sample: No (full dataset)"
fi
echo ""

# Run evaluation
echo -e "${GREEN}Starting evaluation...${NC}"
echo ""

python3 multiturn_continuation/evaluation/evaluate_continuation.py \
    --input "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --judge_model_name "$JUDGE_MODEL" \
    --judge_model_type "$JUDGE_TYPE" \
    $SAMPLE $ASYNC_FLAG $CONCURRENCY

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Evaluation completed successfully!${NC}"
else
    echo ""
    echo -e "${YELLOW}❌ Evaluation failed. Check error messages above.${NC}"
    exit 1
fi
