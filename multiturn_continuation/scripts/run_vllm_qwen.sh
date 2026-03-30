#!/bin/bash
# 运行 Qwen vLLM 模型的多轮对话生成
# 自动启动/停止 vLLM 服务器

# ================= 配置区域 =================
MODEL="Qwen/Qwen3-32B"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6,7}
PORT=8006
TENSOR_PARALLEL=${TENSOR_PARALLEL:-2}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}
VLLM_API_BASE="http://localhost:${PORT}/v1"
# ===========================================

# ── 环境 ──────────────────────────────────────────────
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen_env

export PYTHONPATH=$PROJECT
export VLLM_API_BASE="http://localhost:8000/v1"

PROJECT=$PROJECT
SCRIPT_DIR=$PROJECT/multiturn_continuation/scripts
DATA_DIR=$PROJECT/multiturn_continuation/data_processing/output
OUTPUT_DIR=$PROJECT/multiturn_continuation/output

mkdir -p $OUTPUT_DIR
mkdir -p $PROJECT/multiturn_continuation/logs

# ── 启动 vLLM server ──────────────────────────────────
echo "========================================"
echo "Starting vLLM server for Qwen3-32B..."
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "========================================"

vllm serve "$MODEL" \
    --port 8000 \
    --tensor-parallel-size $TENSOR_PARALLEL \
    --dtype float16 \
    --max-model-len 8192 &

VLLM_PID=$!
echo "vLLM server PID: $VLLM_PID"

# ── 等待 server 就绪 ──────────────────────────────────
echo "Waiting for vLLM server to be ready..."
for i in $(seq 1 60); do
    sleep 10
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ vLLM server is ready! (waited ${i}0 seconds)"
        break
    fi
    echo "   Still waiting... (${i}0s)"
    if [ $i -eq 60 ]; then
        echo "❌ vLLM server failed to start after 600s"
        kill $VLLM_PID
        exit 1
    fi
done

# ── 运行 continuation ─────────────────────────────────
echo ""
echo "Configuration:"
echo "  • Model: $MODEL"
echo "  • GPUs: $CUDA_VISIBLE_DEVICES (tensor-parallel-size: $TENSOR_PARALLEL)"
echo "  • Port: $PORT"
echo "  • Thinking Modes: thinking (9 cases) + no_thinking (5 cases) — auto-detected per case"
echo "  • Patient Model: gpt-4o-mini"
echo "  • Max Turns: 10"
echo "  • Concurrency: 5"
echo ""

# --- A. Start vLLM server ---
echo "[Server] Starting vLLM for $MODEL..."

if lsof -Pi :${PORT} -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "[Warning] Port ${PORT} already in use. Killing existing process..."
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 5
fi

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size $TENSOR_PARALLEL \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --trust-remote-code &

SERVER_PID=$!
echo "[Server] vLLM started with PID: $SERVER_PID"

# --- B. Wait for server to be ready ---
echo "[Server] Waiting for readiness (up to 10 min)..."
TIMEOUT=600
START_TIME=$(date +%s)
while true; do
    if curl -s localhost:${PORT}/v1/models > /dev/null 2>&1; then
        echo "[Server] Ready!"
        break
    fi
    ELAPSED=$(( $(date +%s) - START_TIME ))
    if [ $ELAPSED -gt $TIMEOUT ]; then
        echo "❌ [Error] Server timed out after ${TIMEOUT}s"
        kill $SERVER_PID 2>/dev/null || true
        pkill -f "vllm serve" 2>/dev/null || true
        exit 1
    fi
    sleep 10
done

# --- C. Run continuation ---
echo ""
echo "================================================================================"
echo "Processing: vllm_failed_cases_qwen.json (14 cases: 9 thinking + 5 no_thinking)"
echo "================================================================================"
echo ""

python multiturn_continuation/scripts/run_continuation.py \
    --input-list \
        $DATA_DIR/vllm_abnormal_value_qwen.json \
        $DATA_DIR/vllm_failed_cases_qwen.json \
    --patient_model gpt-4o-mini \
    --max_turns 10 \
    --output multiturn_continuation/output/vllm_qwen_generated.json \
    --vllm-api-base "$VLLM_API_BASE" \
    --async \
    --concurrency 5

# --- D. Cleanup ---
echo ""
echo "[Cleanup] Stopping vLLM (PID: $SERVER_PID)..."
kill $SERVER_PID 2>/dev/null || true
for i in {1..30}; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "[Cleanup] vLLM process terminated"
        break
    fi
    sleep 1
done
pkill -9 -f "vllm serve" 2>/dev/null || true
echo "[Cleanup] Waiting for GPU memory to be released..."
sleep 20

if command -v nvidia-smi &> /dev/null; then
    echo "[Cleanup] Current GPU status:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
fi

echo ""
echo "================================================================================"
echo "✅ Qwen Models Complete!"
echo "================================================================================"
echo ""
echo "Output file:"
echo "  • multiturn_continuation/output/vllm_qwen_generated.json"
echo ""
