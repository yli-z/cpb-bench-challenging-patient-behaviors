#!/bin/bash

# ============================================================================
# 按 behavior_category 分 sheet 生成 Excel 报告
# ============================================================================
# 每个 behavior_category 一个工作表，包含所有模型的数据
# ============================================================================

# Configuration
EXCEL_FILENAME="detailed_results_by_category.xlsx"  # 输出的 Excel 文件名

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT" || exit 1

echo "======================================================================"
echo "按 behavior_category 分 sheet 生成 Excel 报告"
echo "======================================================================"
echo "输出目录: evaluators/failure_rate_eval/output"
echo "Excel 文件: $EXCEL_FILENAME"
echo "======================================================================"
echo ""

# 运行生成脚本
python -m evaluators.failure_rate_eval.generate_excel_by_category \
    --output_dir "evaluators/failure_rate_eval/output" \
    --excel_filename "$EXCEL_FILENAME"

echo ""
echo "======================================================================"
echo "✅ Excel 报告生成完成！"
echo "======================================================================"

