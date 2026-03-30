# Negative Cases Pipeline

---

## Step 1 — Sampling

Sample one truncation point per negative case, mirroring the positive-case turn-position distribution. (per-dataset)

```bash
python Negative_cases/Negative_sampling_segment/sample_negative_cases.py
```

**Input:** `Negative_cases/Negative_original_case_data/{DS}_negative_cases.json`  
**Output:** `Negative_cases/Negative_sampling_segment/{DS}_negative_cases_sampled.json`

Optional — visualise the sampled distribution:
```bash
python Negative_cases/Negative_sampling_segment/plot_negative_distribution.py
```

---

## Step 2 — Generate

Run doctor-response generation for all models × datasets.

**API models (gpt-5 / gpt-4o-mini / gpt-4 / claude / gemini / deepseek):**
```bash
bash Negative_cases/run_api_models.sh
```

**Open-source models (Llama-70B):**
```bash
bash Negative_cases/run_negative_llama70b.sh
```

**Open-source models (Qwen-32B / Llama-8B):**
```bash
bash Negative_cases/run_negative_qwen32b_llama8b.sh
```

**Input:** `Negative_cases/Negative_sampling_segment/{DS}_negative_cases_sampled.json`  
**Output:** `Negative_cases/negative_generate/generated/{model}_{dataset}.jsonl`

---

## Step 3 — Evaluate

Run the overreaction evaluator (LLM-as-judge) over all generated JSONL files.

```bash
python Negative_cases/negative_generate/evaluator/batch_evaluate_negative.py \
    --judge_model_name gpt-4o \
    --judge_model_type openai
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `--input_dir` | `negative_generate/generated/` | Directory of JSONL files |
| `--output_dir` | `evaluator/output/` | Where results are written |
| `--n N` | all | Evaluate only the first N entries per file (quick test) |
| `--no-skip_if_exists` | — | Force re-evaluation of existing results |

**Output:** `evaluator/output/{model}_{dataset}_negative_detailed_results.json`

---

## Step 4 — Analyze

Rebuild the summary report and regenerate the Excel (one sheet per model).

```bash
python Negative_cases/negative_generate/evaluator/generate_negative_excel.py
```

**Output:**
- `evaluator/output/negative_summary_report.json` — failure rates + triggered-dimension counts per model × dataset
- `evaluator/output/negative_detailed_results.xlsx` — one sheet per model, all datasets combined
