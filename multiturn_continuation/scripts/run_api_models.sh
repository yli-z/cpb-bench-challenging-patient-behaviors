#!/bin/bash
# 运行API模型的多轮对话生成
# 包含78个cases（排除vLLM模型）

set -e

echo "================================================================================"
echo "🚀 Multi-Turn Continuation - API Models Only"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  • Cases: 78 (API models only)"
echo "  • Excluded: Qwen_Qwen3-32B_* (vLLM models)"
echo "  • Patient Model: gpt-4o-mini"
echo "  • Max Turns: 10 (patient可回复5-10轮)"
echo ""
echo "Models included:"
echo "  • gpt-4: 22 cases"
echo "  • gemini-2.5-flash: 15 cases"
echo "  • gpt-4o-mini: 12 cases"
echo "  • claude-sonnet-4-5-20250929: 10 cases"
echo "  • deepseek-reasoner: 9 cases"
echo "  • deepseek-chat: 8 cases"
echo "  • gpt-5: 2 cases"
echo ""
echo "Estimated time: 2-3 hours (depending on API rate limits)"
echo "Estimated cost: Varies by model usage"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "❌ Cancelled"
    exit 1
fi

echo ""
echo "================================================================================"
echo "Starting generation..."
echo "================================================================================"
echo ""

cd "$(dirname "$0")/.."

python multiturn_continuation/scripts/run_continuation.py \
    --input multiturn_continuation/data_processing/output/failed_cases_api_only.json \
    --patient_model gpt-4o-mini \
    --max_turns 10 \
    --concurrency 32 \
    --async \
    --output multiturn_continuation/output/api_models_generated.json

echo ""
echo "================================================================================"
echo "✅ Generation Complete!"
echo "================================================================================"
echo ""
echo "Output file: multiturn_continuation/output/api_models_generated.json"
echo ""
echo "Next steps:"
echo "  1. Review the generated dialogues"
echo "  2. Run Module 2 (evaluation) when ready"
echo ""
