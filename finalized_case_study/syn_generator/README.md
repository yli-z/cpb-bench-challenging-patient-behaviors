# syn-generator

Generate synthetic doctor/patient conversations from a seed case by swapping the abnormal
measurement with entries from `data/seed-syn/abnormal_values/abnormal_clinical_items.csv`.

## Usage
```
python syn_generator/generate.py \
  --llm-model gpt-4o-mini \
  --seed data/seed-syn/seed-real.json \
  --items data/seed-syn/abnormal_values/abnormal_clinical_items.csv \
  --output syn_generator/output/synthetic_seed.json
```

Optional limit:
```
python syn_generator/generate.py --llm-model gpt-4o-mini --limit 5
```

Optional concurrency:
```
python syn_generator/generate.py --llm-model gpt-4o-mini --concurrency 5
```

## Convenience script
```
syn_generator/run_generate.sh
```

Environment overrides:
- `LLM_MODEL` (default: `gpt-4o-mini`)
- `SEED_PATH` (default: `data/seed-syn/abnormal_values/seed-real.json`)
- `ITEMS_PATH` (default: `data/seed-syn/abnormal_values/abnormal_clinical_items.csv`)
- `OUTPUT_PATH` (default: `syn_generator/output/synthetic_seed.json`)
- `LIMIT` (optional)

## What it does
- Reuses the seed conversation structure and wording.
- Converts the seed `conversation_segment` to plain text (`Doctor:`/`Patient:` lines).
- Rewrites the conversation in one LLM call so it includes the abnormal item/value.
- Writes a JSON payload with `cases` mirroring the seed schema (without `complete_conversation`).
