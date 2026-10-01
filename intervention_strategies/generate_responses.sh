#!/bin/bash

# Generate doctor responses using intervention strategies
#
# Usage:
#   bash intervention_strategies/generate_responses.sh                               # positive cases, all API models sequentially
#   NEGATIVE=true bash intervention_strategies/generate_responses.sh                 # negative cases, all API models sequentially
#   MODE=self_eval bash intervention_strategies/generate_responses.sh                # run all API models for self_eval
#   MODEL=gpt-4o MODE=cot bash intervention_strategies/generate_responses.sh         # run single model
#   PARALLEL=true MODE=cot bash intervention_strategies/generate_responses.sh        # run all API models in parallel
#   NEGATIVE=true GROUP=all bash intervention_strategies/generate_responses.sh       # negative cases, all model groups
#   GROUP=vllm_llama bash intervention_strategies/generate_responses.sh              # run vLLM Llama models
#   GROUP=vllm_qwen bash intervention_strategies/generate_responses.sh               # run vLLM Qwen models
#   GROUP=all bash intervention_strategies/generate_responses.sh                     # run API + vLLM models

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment if present
if [ -f "$ROOT_DIR/.venv/bin/activate" ]; then
    source "$ROOT_DIR/.venv/bin/activate"
fi

# ---- Configuration ----
MODE="${MODE:-cot}"              # cot | instruction | eval_patient | self_eval
PARALLEL="${PARALLEL:-false}"    # true | false
GROUP="${GROUP:-api}"            # api | vllm_llama | vllm_qwen | all
NEGATIVE="${NEGATIVE:-false}"    # true = load negative sampled cases; false = positive benchmark

# Build the --negative flag (empty string when not needed)
NEGATIVE_FLAG=""
NEGATIVE_PREFIX=""
if [ "$NEGATIVE" = "true" ]; then
    NEGATIVE_FLAG="--negative"
    NEGATIVE_PREFIX="neg_"   # prefix log files so they don't overwrite positive-case logs
fi

# vLLM server settings
VLLM_PORT="${VLLM_PORT:-8006}"
VLLM_API_BASE="http://localhost:${VLLM_PORT}/v1"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.9}"
VLLM_CUDA="${VLLM_CUDA:-4,5,6,7}"
VLLM_TP="${VLLM_TP:-4}"            # tensor-parallel-size
LLAMA_MAX_MODEL_LEN="${LLAMA_MAX_MODEL_LEN:-81920}"  # only for Llama models
VLLM_SERVER_PID=""

# ---- Model Groups ----
API_MODELS=(
    "gpt-4o"
    "gpt-4o-mini"
    "gpt-4"
    "gpt-5"
    "claude-sonnet-4-5-20250929"
    "gemini-2.5-flash"
    "deepseek-chat"
    "deepseek-reasoner"
)

VLLM_LLAMA_MODELS=(
    "meta-llama/Llama-3.3-70B-Instruct"
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
)

VLLM_QWEN_MODELS=(
    "Qwen/Qwen3-32B"           # no_thinking
    "Qwen/Qwen3-32B:thinking"  # thinking mode (special flag)
)

# ---- Helper function ----
run_model() {
    local model="$1"
    local extra_args=("${@:2}")

    echo ""
    echo "====================================="
    echo "  Running: $model${NEGATIVE_FLAG:+ [negative]}"
    echo "====================================="
    python "$SCRIPT_DIR/generate_responses.py" "$MODE" --model "$model" \
        ${NEGATIVE_FLAG} "${extra_args[@]}"
}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$SCRIPT_DIR/logs"

make_logfile() {
    local model="$1"
    echo "$SCRIPT_DIR/logs/${MODE}_${model//\//_}_${TIMESTAMP}.log"
}

run_model_logged() {
    local model="$1"
    shift
    local extra_args=("$@")
    local logfile="$SCRIPT_DIR/${NEGATIVE_PREFIX}${MODE}_${model//\//_}.log"

    echo ""
    echo "====================================="
    echo "  Running: $model${NEGATIVE_FLAG:+ [negative]} -> $logfile"
    echo "====================================="
    python "$SCRIPT_DIR/generate_responses.py" "$MODE" --model "$model" \
        ${NEGATIVE_FLAG} "${extra_args[@]}" \
        2>&1 | tee "$logfile"
}

run_model_bg() {
    local model="$1"
    shift
    local extra_args=("$@")
    local logfile
    logfile="$(make_logfile "$model")"

    echo "[Starting] $model${NEGATIVE_FLAG:+ [negative]} -> $logfile"
    python "$SCRIPT_DIR/generate_responses.py" "$MODE" --model "$model" \
        ${NEGATIVE_FLAG} "${extra_args[@]}" \
        > "$logfile" 2>&1 &
}

# ---- Run API models ----
run_api_models() {
    local models=("${API_MODELS[@]}")
    # If MODEL is set, override
    if [ -n "$MODEL" ]; then
        models=("$MODEL")
    fi

    echo "====================================="
    echo "  Intervention Strategy: $MODE"
    echo "  API Models: ${models[*]}"
    echo "  Parallel: $PARALLEL"
    echo "====================================="

    if [ "$PARALLEL" = "true" ] && [ ${#models[@]} -gt 1 ]; then
        PIDS=()
        for model in "${models[@]}"; do
            run_model_bg "$model" "$SCRIPT_DIR/${NEGATIVE_PREFIX}${MODE}_${model//\//_}.log"
            PIDS+=($!)
        done

        echo ""
        echo "Launched ${#PIDS[@]} parallel jobs. Waiting..."

        FAILED=0
        for i in "${!PIDS[@]}"; do
            if wait "${PIDS[$i]}"; then
                echo "[Done] ${models[$i]}"
            else
                echo "[FAILED] ${models[$i]} (see log)"
                FAILED=$((FAILED + 1))
            fi
        done

        if [ $FAILED -gt 0 ]; then
            echo "$FAILED API model(s) failed. Continuing with remaining groups..."
        fi
    else
        for model in "${models[@]}"; do
            run_model "$model"
        done
    fi
}

# ---- vLLM server lifecycle ----
start_vllm_server() {
    local model_path="$1"
    local extra_args=("${@:2}")

    # Kill existing server on this port
    if lsof -Pi :${VLLM_PORT} -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "[vLLM] Port ${VLLM_PORT} in use. Killing existing process..."
        kill $(lsof -Pi :${VLLM_PORT} -sTCP:LISTEN -t) 2>/dev/null || true
        sleep 5
    fi

    echo "[vLLM] Starting server for $model_path on port $VLLM_PORT..."
    CUDA_VISIBLE_DEVICES=$VLLM_CUDA vllm serve "$model_path" \
        --host 0.0.0.0 \
        --port $VLLM_PORT \
        --tensor-parallel-size $VLLM_TP \
        --gpu-memory-utilization $VLLM_GPU_MEM_UTIL \
        --trust-remote-code \
        "${extra_args[@]}" &

    VLLM_SERVER_PID=$!
    echo "[vLLM] Server PID: $VLLM_SERVER_PID"

    # Wait for server to be ready (up to 10 min)
    echo "[vLLM] Waiting for server to be ready..."
    for i in $(seq 1 60); do
        if curl -s localhost:${VLLM_PORT}/v1/models > /dev/null 2>&1; then
            echo "[vLLM] Server ready! (waited ${i}0s)"
            return 0
        fi
        sleep 10
    done

    echo "[vLLM] Server failed to start after 600s"
    kill $VLLM_SERVER_PID 2>/dev/null || true
    VLLM_SERVER_PID=""
    return 1
}

stop_vllm_server() {
    if [ -n "$VLLM_SERVER_PID" ]; then
        echo "[vLLM] Stopping server (PID: $VLLM_SERVER_PID)..."
        kill $VLLM_SERVER_PID 2>/dev/null || true
        for i in {1..30}; do
            if ! kill -0 $VLLM_SERVER_PID 2>/dev/null; then
                echo "[vLLM] Server terminated."
                break
            fi
            sleep 1
        done
        kill -9 $VLLM_SERVER_PID 2>/dev/null || true
        VLLM_SERVER_PID=""
        echo "[vLLM] Waiting for GPU memory release..."
        sleep 15
    fi
}

# ---- Run vLLM Llama models ----
run_vllm_llama_models() {
    echo "====================================="
    echo "  vLLM Llama Models"
    echo "  GPUs: $VLLM_CUDA | TP: $VLLM_TP | Port: $VLLM_PORT"
    echo "====================================="

    for model in "${VLLM_LLAMA_MODELS[@]}"; do
        start_vllm_server "$model" --max-model-len $LLAMA_MAX_MODEL_LEN
        run_model_logged "$model" --vllm-api-base "$VLLM_API_BASE"
        stop_vllm_server
    done
}

# ---- Run vLLM Qwen models ----
run_vllm_qwen_models() {
    echo "====================================="
    echo "  vLLM Qwen Models"
    echo "  GPUs: $VLLM_CUDA | TP: $VLLM_TP | Port: $VLLM_PORT"
    echo "====================================="

    # Qwen doesn't need max-model-len override
    start_vllm_server "Qwen/Qwen3-32B"

    # No-thinking mode
    run_model_logged "Qwen/Qwen3-32B" --vllm-api-base "$VLLM_API_BASE"

    # Thinking mode (use custom log name to avoid overwriting)
    local logfile="$SCRIPT_DIR/logs/${NEGATIVE_PREFIX}${MODE}_Qwen_Qwen3-32B_thinking_${TIMESTAMP}.log"
    echo ""
    echo "====================================="
    echo "  Running: Qwen/Qwen3-32B (thinking)${NEGATIVE_FLAG:+ [negative]} -> $logfile"
    echo "====================================="
    python "$SCRIPT_DIR/generate_responses.py" "$MODE" --model "Qwen/Qwen3-32B" \
        --vllm-api-base "$VLLM_API_BASE" --enable-thinking \
        ${NEGATIVE_FLAG} 2>&1 | tee "$logfile"

    stop_vllm_server
}

# ---- Main ----
case "$GROUP" in
    api)
        run_api_models
        ;;
    vllm_llama)
        run_vllm_llama_models
        ;;
    vllm_qwen)
        run_vllm_qwen_models
        ;;
    all)
        run_api_models
        echo ""
        echo "===== API models done. Starting vLLM models... ====="
        echo ""
        run_vllm_llama_models
        run_vllm_qwen_models
        ;;
    *)
        echo "Unknown GROUP: $GROUP"
        echo "Available: api, vllm_llama, vllm_qwen, all"
        exit 1
        ;;
esac

echo ""
echo "All done."
