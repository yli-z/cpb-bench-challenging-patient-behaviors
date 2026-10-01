#!/bin/bash

# ============================================================================
# Generate an Excel file with sheets by behavior_category
# Each behavior_category has a sheet, containing all model data
# ============================================================================

# Configuration
EXCEL_FILENAME="detailed_results_by_category.xlsx"  # Output Excel file name

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT" || exit 1

echo "======================================================================"
echo "Generate an Excel file with sheets by behavior_category"
echo "======================================================================"
echo "Output directory: evaluators_single/output_single_turn"
echo "Excel file: $EXCEL_FILENAME"
echo "======================================================================"
echo ""

# Run generation script
python -m evaluators_single.scripts.generate_excel_by_category \
    --output_dir "evaluators_single/output_single_turn" \
    --excel_filename "$EXCEL_FILENAME"

echo ""
echo "======================================================================"
echo "✅ Excel report generated!"
echo "======================================================================"

