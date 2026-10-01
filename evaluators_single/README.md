# Evaluators (Single-Benchmark)

Failure-rate evaluation for doctor model outputs: GPT-4o as LLM-as-a-judge over four challenging patient behaviors (Information Contradiction, Factual Inaccuracy, Self-diagnosis, Care Resistance).

## Layout

- **scripts/**
  - `evaluate.py` — Single-file evaluation entry point.
  - `batch_evaluate.py` — Batch evaluation over multiple JSONL files.
  - `failure_rate_evaluator.py` — Core `FailureRateEvaluator` class (LLM-as-Judge).
  - `judge_utils.py` — Prompt formatting and judge response parsing.
  - `utils.py` — I/O helpers, statistics, Excel/report generation.
  - `generate_excel_per_model.py` / `generate_excel_by_category.py` — Standalone Excel report generators.
  - `generate_summary_excel.py` — Summary Excel across all models.
  - `run_batch_eval_gpt4o.sh` — Example shell script for batch evaluation with GPT-4o.
- **output_single_turn/** — GPT-4o judge results for all 692 cases of each doctor model (per-model JSON; `evaluation_result=True` means the judge flagged a failure). Default output directory of the scripts.
- **human_verified/** — Final failure set: GPT-4o-flagged responses confirmed as failures by human review (`judge_model="human"`), same JSON format.

## Usage

Run from project root `PYTHONPATH`.

```bash
# Single file
python -m evaluators_single.scripts.evaluate \
  --input_file path/to/model_dataset.jsonl \
  --judge_model_name gpt-4o \
  --judge_model_type openai \
  --context_mode full_context \
  --output_dir evaluators_single/output_single_turn

# Batch over multiple JSONL files
python -m evaluators_single.scripts.batch_evaluate \
  --input_dir model_generator_single/model_output_single \
  --output_dir evaluators_single/output_single_turn \
  --judge_model_name gpt-4o \
  --judge_model_type openai \
  --context_mode full_context

# Generate Excel reports from existing JSON results
python -m evaluators_single.scripts.generate_excel_per_model \
  --input_dir evaluators_single/output_single_turn
```

Optional: use `.env` in project root for API keys; scripts load it when present.

## Input / Output

- **Input:** JSONL from single-turn generation (e.g. `model_generator_single/model_output_single/`).
- **Output:** Per-model detailed JSON results and optional Excel reports under `output_dir`.

## Paths

All paths in examples are relative to the repository root.
