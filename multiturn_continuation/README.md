# Multi-turn Continuation

Tests whether LLM doctors self-correct or persist in failures over multi-turn dialogue.

## Pipeline

```
1. Prepare data    ->  2. Run continuation  ->  3. Evaluate  ->  4. Format for annotation
```

### 1. Prepare Data

Extract failed cases and build conversation history:

```bash
python multiturn_continuation/scripts/prepare_data.py
```

Output: `data_processing/output/` (per-model JSON files)

### 2. Run Continuation

**API models** (GPT, Claude, Gemini, DeepSeek):
```bash
bash multiturn_continuation/run_api_models.sh
```

**vLLM models**:
```bash
bash multiturn_continuation/run_vllm_llama.sh          # Llama 70B
bash multiturn_continuation/run_vllm_llama.sh --8b     # Llama 8B
bash multiturn_continuation/run_vllm_qwen.sh           # Qwen 32B (thinking + no-thinking)
```

Options for `run_continuation.py`:
- `--patient_model`: Patient LLM (default: gpt-4o-mini)
- `--max_turns`: Max additional turns (default: 10)
- `--async --concurrency N`: Parallel execution
- `--vllm-api-base URL`: vLLM server endpoint

### 3. Evaluate

```bash
bash multiturn_continuation/evaluation/run_evaluation.sh
```

## Directory Structure

```
multiturn_continuation/
├── agents/              # Doctor and patient agent wrappers
├── continuation/        # Core dialogue engine (engine.py, async_engine.py)
├── data_processing/     # Data preparation and conversation building
├── evaluation/          # Continuation evaluation (judge-based)
├── scripts/             # Data prep and main runner (run_continuation.py)
└── output/              # Generated multi-turn results
```
