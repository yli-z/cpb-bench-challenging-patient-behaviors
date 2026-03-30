# Intervention Strategies

This module implements different intervention strategies to improve doctor LLM responses when facing challenging patient behaviors (information contradiction, factual inaccuracy, self-diagnosis, care resistance).

## Strategies

| Mode | Class | Description |
|------|-------|-------------|
| `cot` | `CotStrategy` | The model reasons step-by-step before producing a final response. |
| `instruction` | `InstructionStrategy` | The model receives behavior-aware instructions for all four challenging types. |
| `eval_patient` | `EvalPatientStrategy` | Two-step: analyze the patient's query for concerns, then generate an informed response. |
| `self_eval` | `SelfEvalStrategy` | Two-step: generate a response, then self-check and revise if needed. |

## Usage

### Generate Responses

```bash
# Single model
python intervention_strategies/generate_responses.py cot --model gpt-4o

# Via shell script (API models, runs in parallel)
MODE=cot PARALLEL=true bash intervention_strategies/generate_responses.sh

# vLLM models (auto-starts/stops vLLM server)
MODE=cot GROUP=vllm_llama bash intervention_strategies/generate_responses.sh
MODE=cot GROUP=vllm_qwen bash intervention_strategies/generate_responses.sh

# All models (API + vLLM Llama + vLLM Qwen)
MODE=cot GROUP=all bash intervention_strategies/generate_responses.sh
```

**Environment variables for `generate_responses.sh`:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MODE` | `cot` | Strategy: `cot`, `instruction`, `eval_patient`, `self_eval` |
| `GROUP` | `api` | Model group: `api`, `vllm_llama`, `vllm_qwen`, `all` |
| `MODEL` | *(all API models)* | Single model override (e.g., `gpt-4o`) |
| `PARALLEL` | `true` | Run API models in parallel |
| `VLLM_CUDA` | `4,5,6,7` | GPUs for vLLM server |
| `VLLM_TP` | `2` | Tensor parallel size for vLLM |
| `VLLM_PORT` | `8006` | vLLM server port |

### Evaluate Responses (LLM-as-Judge)

```bash
# Evaluate with default settings (judge=gpt-4o, context=full_context, concurrency=16)
MODE=cot bash intervention_strategies/evaluate_responses.sh

# Custom judge and concurrency
MODE=instruction JUDGE=gpt-4o CONCURRENCY=32 bash intervention_strategies/evaluate_responses.sh

# All strategies
bash intervention_strategies/run_all_strategies.sh
```

**Environment variables for `evaluate_responses.sh`:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MODE` | `cot` | Strategy to evaluate |
| `JUDGE` | `gpt-4o` | Judge model |
| `CONTEXT` | `full_context` | Context mode: `full_context`, `current_turn`, `min_turn` |
| `CONCURRENCY` | `16` | Async parallel evaluation calls |

### Run All Strategies

```bash
# Run generation + evaluation for all strategies and all model groups
bash intervention_strategies/run_all_strategies.sh

# Only vLLM models
GROUP=vllm_llama bash intervention_strategies/run_all_strategies.sh
GROUP=vllm_qwen bash intervention_strategies/run_all_strategies.sh
```

## Architecture

```
intervention_strategies/
├── strategies/                  # Strategy classes (one per file)
│   ├── __init__.py              # Registry: STRATEGY_REGISTRY, get_strategy()
│   ├── base.py                  # BaseStrategy (ABC) + StrategyResult dataclass
│   ├── cot.py                   # CotStrategy
│   ├── instruction.py           # InstructionStrategy
│   ├── eval_patient.py          # EvalPatientStrategy
│   └── self_eval.py             # SelfEvalStrategy
├── common.py                    # Data loading, result saving, path setup
├── strategy_executor.py         # StrategyExecutor: async orchestration with concurrency
├── generate_responses.py        # CLI entry point for generation
├── generate_responses.sh        # Shell script: parallel API + vLLM server lifecycle
├── evaluate_responses.sh        # Shell script: async parallel LLM-as-Judge evaluation
├── run_all_strategies.sh        # Shell script: run all strategies end-to-end
├── cot/                         # CoT output JSONL files
├── instruction/                 # Instruction output JSONL files
├── eval_patient/                # EvalPatient output JSONL files
├── self_eval/                   # SelfEval output JSONL files
└── eval_results/                # Evaluation results per strategy
    ├── cot/
    ├── instruction/
    ├── eval_patient/
    └── self_eval/
```


## Pipeline

```
Safety Benchmark Data (JSON)
    |
    v
StrategyExecutor + <Strategy> --> Generated Responses (JSONL per model per dataset)
    |
    v
evaluate_responses.sh --> Async LLM-as-Judge --> eval_results/<mode>/
```
