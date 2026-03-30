#!/bin/bash

# ============================================================================
# Batch Evaluation Script with GPT-4o as Judge
# ============================================================================
# This script uses batch_evaluate.py to process all model output files
# in the model_generator/model_output directory
# ============================================================================

# Configuration
JUDGE_MODEL_NAME="gpt-4o"
JUDGE_MODEL_TYPE="openai"
CONTEXT_MODE="full_context"  
TEMPERATURE=0

# Paths - Get absolute paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAILURE_RATE_EVAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$FAILURE_RATE_EVAL_DIR/.." && pwd)"
MODEL_OUTPUT_DIR="$PROJECT_ROOT/model_generator/model_output"
OUTPUT_DIR="$FAILURE_RATE_EVAL_DIR/output"  # Use absolute path to ensure correct location

# Change to failure_rate_eval directory to ensure correct imports
cd "$FAILURE_RATE_EVAL_DIR" || exit 1

# ============================================================================
# Batch Evaluation Mode
# ============================================================================
# Process all .jsonl files in the model_output directory
# - Default: Skip files that already exist (--skip_if_exists)
# - Auto-retry is enabled by default to ensure data consistency
# - Excel is generated after all JSON files are processed
# ============================================================================

echo "======================================================================"
echo "Batch Evaluation with GPT-4o as Judge"
echo "======================================================================"
echo "Judge Model: $JUDGE_MODEL_NAME ($JUDGE_MODEL_TYPE)"
echo "Context Mode: $CONTEXT_MODE"
echo "Input Directory: $MODEL_OUTPUT_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "======================================================================"
echo ""

# Check if input directory exists
if [ ! -d "$MODEL_OUTPUT_DIR" ]; then
    echo "Error: Input directory does not exist: $MODEL_OUTPUT_DIR"
    exit 1
fi

# Count .jsonl files
JSONL_COUNT=$(find "$MODEL_OUTPUT_DIR" -name "*.jsonl" | wc -l | tr -d ' ')
if [ "$JSONL_COUNT" -eq 0 ]; then
    echo "Error: No .jsonl files found in directory: $MODEL_OUTPUT_DIR"
    exit 1
fi

echo "Found $JSONL_COUNT .jsonl file(s) to process"
echo ""

# Run batch evaluation
# Note: --auto_retry is enabled by default in batch_evaluate.py
python batch_evaluate.py \
    --judge_model_name "$JUDGE_MODEL_NAME" \
    --judge_model_type "$JUDGE_MODEL_TYPE" \
    --context_mode "$CONTEXT_MODE" \
    --input_dir "$MODEL_OUTPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --temperature $TEMPERATURE \
    --skip_if_exists \
    --max_retries 3 \
    --retry_delay 2.0 \
    --retry_max_attempts 10 \
    --retry_delay_long 5.0

echo ""
echo "======================================================================"
echo "✅ Batch evaluation completed!"
echo "======================================================================"
echo "Results saved to: $OUTPUT_DIR"
echo "Summary report: $OUTPUT_DIR/summary_report.json"
echo "Excel report: $OUTPUT_DIR/detailed_results.xlsx"
echo "======================================================================"

