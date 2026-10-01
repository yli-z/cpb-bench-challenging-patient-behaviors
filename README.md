# Challenging Patient Behaviors Benchmark

**Python:** 3.10+

Large language models (LLMs) are increasingly used for medical consultation and health information support. We introduce **Challenging Patient Behaviors Benchmark**, a bilingual benchmark of 692 dialogues annotated with four challenging behaviors: *information contradiction*, *factual inaccuracy*, *self-diagnosis*, and *care resistance*.

## Project structure

```
CPB-Bench/
├── cpb-bench_data                # CPB-Benchmark: positive challenging patient behavior(692 cases) + negative challenging patient behavior(352 cases)
├── data_loader/                  # Data loading; bilingual benchmark of 692 cases under data_loader/output_benchmark
├── models/                       # LLM wrappers (OpenAI, Claude, Gemini, DeepSeek, vLLM)
├── prompts/                      # Prompt templates
├── preprocess/                   # Preprocessing and behavior detection
├── model_generator_single/       # Single-turn doctor response generation
├── evaluators_single/            # Failure-rate evaluation (LLM-as-a-judge); scripts in scripts/
├── false_negative/               # False Failure sampling (LLM-as-a-judge); False-negative evaluation
│── finalized_case_study/         # Case-study finalized data, model outputs, and evaluation results
    ├── syn_generator/            # Synthetic dialogue/case generation
├── multiturn_continuation/       # Multi-turn continuation and persistence evaluation
├── intervention_strategies/      # Mitigation strategies (CoT, instruction, eval_patient, self_eval)
└── negative_cases/               # Negative-case sampling, generation, and evaluation
```

## Quick start

**Environment:**

```bash
conda create -n cpb-bench python=3.10
conda activate cpb-bench
pip install -r requirements.txt
```

**API keys:**

```bash
export OPENAI_API_KEY="your_key"
export ANTHROPIC_API_KEY="your_key"
export GEMINI_API_KEY="your_key"
```

**Data:** Generated/processed datasets live in `data_loader/output_benchmark`.

**Usage:** See the `README.md` in each subdirectory:

- **Single-turn generation:** `model_generator_single/` (e.g. `generate_response.py`)
- **Multi-turn continuation:** `multiturn_continuation/` (`prepare_data.py`, `run_continuation.py`)
- **Intervention strategies:** `intervention_strategies/` (CoT, instruction, eval_patient, self_eval; `generate_responses.sh`, `evaluate_responses.sh`)
- **Failure-rate evaluation:** `evaluators_single/scripts/` (e.g. `batch_evaluate.py`, `evaluate.py`); case-study outputs in `evaluators_single`
- **Negative cases:** `Negative_cases/` (sampling, generation, evaluator)
- **Synthetic case generation:** `finalized_case_study/syn_generator/`


