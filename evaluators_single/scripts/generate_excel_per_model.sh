#!/bin/bash

# ============================================================================
# Generate an Excel file for each model, with each behavior_category as a sheet for manual inspection
# ============================================================================

# Configuration
INPUT_DIR="evaluators_single/output_single_turn"  # Input directory (contains JSON files)
OUTPUT_DIR="evaluators_single/output_single_turn"  # Output directory (default is the same as the input directory)

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT" || exit 1

echo "======================================================================"
echo "Generate an Excel file for each model, with each behavior_category as a sheet"
echo "======================================================================"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "======================================================================"
echo ""

# Run generation script
python -m evaluators_single.scripts.generate_excel_per_model \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "======================================================================"
echo "✅ Excel file generated!"
echo "======================================================================"
