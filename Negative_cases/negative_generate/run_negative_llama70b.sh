#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=ruishanl_1185
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --constraint=a100-80gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --job-name=neg_llama70b
#SBATCH --output=logs_neg_llama70b_%j.out
#SBATCH --error=logs_neg_llama70b_%j.err

# ================= 配置区域 =================
MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"

DATASETS=("ACI" "IMCS" "MedDG" "MediTOD")
SAMPLED_DIR="Negative_cases/Negative_sampling_segment"
OUTPUT_DIR="Negative_cases/negative_generate/generated"

# ================= 环境准备 =================
module purge
source $MINIConda3/etc/profile.d/conda.sh
conda activate qwen_env
cd $PROJECT_ROOT

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
    local TIMEOUT=900
    local START_TIME=$(date +%s)
    local PORT=$1
    echo "[Server] Waiting for readiness on port $PORT (max 15 min, 70B loads slower)..."
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
    sleep 10
    echo "[Cleanup] Done."
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
# 启动 Llama-3.3-70B vLLM
# ==========================================
echo "=========================================="
echo ">>> meta-llama/Llama-3.3-70B-Instruct  [2×A100-80GB, TP=2]"
echo "=========================================="

if lsof -Pi :8007 -sTCP:LISTEN -t >/dev/null; then
    pkill -f "vllm serve"; sleep 5
fi

nohup vllm serve "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port 8007 \
    --served-model-name "$MODEL_NAME" \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    > "model_generator/logs_multi/vllm_llama70b_neg_${SLURM_JOB_ID}.log" 2>&1 &
SERVER_PID=$!
echo "[Server] vLLM started with PID: $SERVER_PID"

if ! wait_for_server 8007; then
    echo "[ERROR] Llama-3.3-70B server failed. Check:"
    echo "  model_generator/logs_multi/vllm_llama70b_neg_${SLURM_JOB_ID}.log"
    cleanup_server $SERVER_PID
    exit 1
fi

# ==========================================
# 遍历数据集
# ==========================================
for DS in "${DATASETS[@]}"; do
    INPUT_FILE="$SAMPLED_DIR/$(get_input_file $DS)"
    OUTPUT_CHECK="${OUTPUT_DIR}/meta-llama_Llama-3.3-70B-Instruct_${DS}.jsonl"

    if [ -f "$OUTPUT_CHECK" ]; then
        echo "  ⏭  $DS — already exists, skipping"
        continue
    fi

    echo "--- Dataset: $DS | $(date) ---"
    if python model_generator/generate_response.py \
        --dataset    "$DS" \
        --input_file "$INPUT_FILE" \
        --model      "$MODEL_NAME" \
        --vllm-api-base http://localhost:8007/v1 \
        --output-dir "$OUTPUT_DIR" \
        --concurrency 12; then
        echo "✓ Done: $DS at $(date)"
    else
        echo "✗ Error on: $DS, continuing..."
    fi
done

# ==========================================
# 清理
# ==========================================
cleanup_server $SERVER_PID

echo "=================================================="
echo "All done! End time: $(date)"
echo "=================================================="
