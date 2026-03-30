#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=ruishanl_1185
#SBATCH --gres=gpu:a40:2          # Qwen3-32B fp16 ~64GB → 2×A40 (48GB×2=96GB)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --job-name=neg_qwen32b_llama8b
#SBATCH --output=logs_neg_qwen32b_llama8b_%j.out
#SBATCH --error=logs_neg_qwen32b_llama8b_%j.err

# ================= 配置区域 =================
QWEN32B_NAME="Qwen/Qwen3-32B"
LLAMA8B_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"

DATASETS=("ACI" "IMCS" "MedDG" "MediTOD")
SAMPLED_DIR="Negative_cases/Negative_sampling_segment"
OUTPUT_DIR="Negative_cases/negative_generate/generated"

# ================= 环境准备 =================
module purge
source $MINIConda3/etc/profile.d/conda.sh
conda activate qwen_env
cd $PROJECT

export CUDA_VISIBLE_DEVICES=4,5,6,7

mkdir -p model_generator/logs_multi
mkdir -p "$OUTPUT_DIR"

echo "=================================================="
echo "Job runs on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "=================================================="

# ========== 工具函数 ==========
wait_for_server() {
    local PORT=$1
    local TIMEOUT=900
    local START_TIME=$(date +%s)
    echo "[Server] Waiting for readiness on port $PORT (max 15 min)..."
    while true; do
        if curl -s localhost:${PORT}/v1/models > /dev/null; then
            echo "[Server] Ready! $(date)"
            return 0
        fi
        if [ $(($(date +%s) - START_TIME)) -gt $TIMEOUT ]; then
            echo "[ERROR] Server timed out after ${TIMEOUT}s."
            return 1
        fi
        sleep 10
    done
}

cleanup_server() {
    local PID=$1
    echo "[Cleanup] Stopping vLLM (PID: $PID)..."
    kill $PID 2>/dev/null
    sleep 5
    pkill -9 -f "vllm serve" 2>/dev/null
    sleep 15
    echo "[Cleanup] Done. GPU memory status:"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
    echo ""
}

# dataset name → sampled JSON filename
get_input_file() {
    case "$1" in
        ACI)     echo "ACI_negative_cases_sampled.json" ;;
        IMCS)    echo "IMCS21_negative_cases_sampled.json" ;;
        MedDG)   echo "MedDG_negative_cases_sampled.json" ;;
        MediTOD) echo "MediTOD_negative_cases_sampled.json" ;;
    esac
}

# ==========================================
# Stage 1: Qwen3-32B No Thinking
# ==========================================
echo "=========================================="
echo ">>> STAGE 1: Qwen/Qwen3-32B  [No Thinking]"
echo "=========================================="

if lsof -Pi :8007 -sTCP:LISTEN -t >/dev/null; then
    pkill -f "vllm serve"; sleep 5
fi

nohup vllm serve "$QWEN32B_NAME" \
    --host 0.0.0.0 \
    --port 8007 \
    --served-model-name "$QWEN32B_NAME" \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.92 \
    --max-model-len 8192 \
    --trust-remote-code \
    > "model_generator/logs_multi/vllm_qwen3_32b_no_thinking_${SLURM_JOB_ID}.log" 2>&1 &
SERVER_PID=$!
echo "[Server] vLLM started with PID: $SERVER_PID"

if ! wait_for_server 8007; then
    echo "[ERROR] Qwen3-32B server failed. Check:"
    echo "  model_generator/logs_multi/vllm_qwen3_32b_no_thinking_${SLURM_JOB_ID}.log"
    cleanup_server $SERVER_PID
    exit 1
fi

for DS in "${DATASETS[@]}"; do
    INPUT_FILE="$SAMPLED_DIR/$(get_input_file $DS)"
    OUTPUT_CHECK="${OUTPUT_DIR}/Qwen_Qwen3-32B_no_thinking_${DS}.jsonl"
    if [ -f "$OUTPUT_CHECK" ]; then
        echo "  ⏭  [No Thinking] $DS — already exists, skipping"
        continue
    fi
    echo "--- [No Thinking] Dataset: $DS | $(date) ---"
    if python model_generator/generate_response.py \
        --dataset    "$DS" \
        --input_file "$INPUT_FILE" \
        --model      "$QWEN32B_NAME" \
        --vllm-api-base http://localhost:8007/v1 \
        --output-dir "$OUTPUT_DIR" \
        --concurrency 16; then
        echo "✓ Done: $DS"
    else
        echo "✗ Error on: $DS, continuing..."
    fi
done

cleanup_server $SERVER_PID

# ==========================================
# Stage 2: Qwen3-32B Thinking
# (max-model-len enlarged to accommodate CoT)
# ==========================================
echo "=========================================="
echo ">>> STAGE 2: Qwen/Qwen3-32B  [Thinking]"
echo "=========================================="

nohup vllm serve "$QWEN32B_NAME" \
    --host 0.0.0.0 \
    --port 8008 \
    --served-model-name "$QWEN32B_NAME" \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.92 \
    --max-model-len 16384 \
    --trust-remote-code \
    > "model_generator/logs_multi/vllm_qwen3_32b_thinking_${SLURM_JOB_ID}.log" 2>&1 &
SERVER_PID=$!
echo "[Server] vLLM started with PID: $SERVER_PID"

if ! wait_for_server 8008; then
    echo "[ERROR] Qwen3-32B (thinking) server failed. Check:"
    echo "  model_generator/logs_multi/vllm_qwen3_32b_thinking_${SLURM_JOB_ID}.log"
    cleanup_server $SERVER_PID
    exit 1
fi

for DS in "${DATASETS[@]}"; do
    INPUT_FILE="$SAMPLED_DIR/$(get_input_file $DS)"
    OUTPUT_CHECK="${OUTPUT_DIR}/Qwen_Qwen3-32B_thinking_${DS}.jsonl"
    if [ -f "$OUTPUT_CHECK" ]; then
        echo "  ⏭  [Thinking] $DS — already exists, skipping"
        continue
    fi
    echo "--- [Thinking] Dataset: $DS | $(date) ---"
    if python model_generator/generate_response.py \
        --dataset    "$DS" \
        --input_file "$INPUT_FILE" \
        --model      "$QWEN32B_NAME" \
        --vllm-api-base http://localhost:8008/v1 \
        --output-dir "$OUTPUT_DIR" \
        --concurrency 8 \
        --enable-thinking; then
        echo "✓ Done: $DS"
    else
        echo "✗ Error on: $DS, continuing..."
    fi
done

cleanup_server $SERVER_PID

# ==========================================
# Stage 3: Llama-3.1-8B
# (TP=1 — 8B fits on a single A40)
# ==========================================
echo "=========================================="
echo ">>> STAGE 3: meta-llama/Meta-Llama-3.1-8B-Instruct"
echo "=========================================="

nohup vllm serve "$LLAMA8B_NAME" \
    --host 0.0.0.0 \
    --port 8009 \
    --served-model-name "$LLAMA8B_NAME" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    > "model_generator/logs_multi/vllm_llama8B_neg_${SLURM_JOB_ID}.log" 2>&1 &
SERVER_PID=$!
echo "[Server] vLLM started with PID: $SERVER_PID"

if ! wait_for_server 8009; then
    echo "[ERROR] Llama-3.1-8B server failed. Check:"
    echo "  model_generator/logs_multi/vllm_llama8B_neg_${SLURM_JOB_ID}.log"
    cleanup_server $SERVER_PID
    exit 1
fi

for DS in "${DATASETS[@]}"; do
    INPUT_FILE="$SAMPLED_DIR/$(get_input_file $DS)"
    OUTPUT_CHECK="${OUTPUT_DIR}/meta-llama_Meta-Llama-3.1-8B-Instruct_${DS}.jsonl"
    if [ -f "$OUTPUT_CHECK" ]; then
        echo "  ⏭  [Llama-8B] $DS — already exists, skipping"
        continue
    fi
    echo "--- [Llama-8B] Dataset: $DS | $(date) ---"
    if python model_generator/generate_response.py \
        --dataset    "$DS" \
        --input_file "$INPUT_FILE" \
        --model      "$LLAMA8B_NAME" \
        --vllm-api-base http://localhost:8009/v1 \
        --output-dir "$OUTPUT_DIR" \
        --concurrency 16; then
        echo "✓ Done: $DS"
    else
        echo "✗ Error on: $DS, continuing..."
    fi
done

cleanup_server $SERVER_PID

echo "=================================================="
echo "All done! End time: $(date)"
echo "=================================================="
