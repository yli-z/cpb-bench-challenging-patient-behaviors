# Model Generator (Single-Turn)

Generate single-turn doctor responses for safety-benchmark cases. Reads cases from `data_loader/output_benckmark` (or fallback paths) and writes model outputs as JSONL.

## Contents

- **generate_response.py** — Main script: load cases, call doctor LLM per case, save JSONL.
- **batch_generate.py** — Batch over models/datasets.
- **model_output_single/** — Default output directory for JSONL files (e.g. `{model}_{dataset}.jsonl`).

## Usage

Run from project root. Set API keys (e.g. `OPENAI_API_KEY`) or configure vLLM base for local models.

```bash
# Single dataset and model
python model_generator_single/generate_response.py --dataset ACI --model gpt-4o-mini

# Multiple datasets
python model_generator_single/generate_response.py --dataset ACI MediTOD --model gpt-4o-mini
```

Batch:

```bash
python model_generator_single/batch_generate.py ...
```

## Input / output paths

- **Input:** `data_loader/output_benckmark/{DATASET}_safety_benchmark.json` or, if missing, `syn_generator/output/complementary_{DATASET}_generated_cases_validated_gpt4o-mini.json`.
- **Output:** Writes to `data_loader/output_benckmark` by default (see `generate_response.py`); batch may use `model_generator_single/model_output_single/`.

## Dependencies

Uses `models.model_utils`, `prompts.prompt_templates`. Supports OpenAI, Claude, DeepSeek, and vLLM (Qwen, Llama).
