#!/usr/bin/env bash
set -euo pipefail

LLM_MODEL="gpt-4o-mini"
# SEED_PATH="data/seed-syn/abnormal_values/seed-real.json"
# ITEMS_PATH="finalized_case_study/abnormal_values/seed/abnormal_clinical_items.csv"

ITEMS_PATH="finalized_case_study/finalized_information_contradiction/seed/contradiction_baby_finalized_ids_01.csv"
SEED_PATH="finalized_case_study/information_contradiction/seed/seed-real.json"
OUTPUT_PATH="syn_generator/output/${CATEGORY}_01_generated_cases_by_${LLM_MODEL}.json"

python -m pdb -c continue -m syn_generator.generate \
  --llm-model "$LLM_MODEL" \
  --seed "$SEED_PATH" \
  --items "$ITEMS_PATH" \
  --output "$OUTPUT_PATH"
