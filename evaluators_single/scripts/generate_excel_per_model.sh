#!/bin/bash

# ============================================================================
# 为每个模型单独生成按 behavior_category 分 sheet 的 Excel 文件
# ============================================================================
# 每个模型生成一个 Excel 文件，每个 behavior_category 一个工作表
# ============================================================================

# Configuration
INPUT_DIR="evaluators/failure_rate_eval/output_llama_deepseek"  # 输入目录（包含 JSON 文件）
OUTPUT_DIR="evaluators/failure_rate_eval/output_llama_deepseek"  # 输出目录（默认与输入相同）

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT" || exit 1

echo "======================================================================"
echo "为每个模型单独生成按 behavior_category 分 sheet 的 Excel 文件"
echo "======================================================================"
echo "输入目录: $INPUT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "======================================================================"
echo ""

# 运行生成脚本
python -m evaluators.failure_rate_eval.generate_excel_per_model \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "======================================================================"
echo "✅ Excel 文件生成完成！"
echo "======================================================================"
