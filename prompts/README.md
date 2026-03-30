# Prompts

Central prompt templates for doctors, patients, evaluators, and negative-case flows. Used by `model_generator_single`, `multiturn_continuation`, `intervention_strategies`, `Negative_cases`, and evaluation scripts.

## Modules

| File                  | Purpose                                      |
|-----------------------|----------------------------------------------|
| prompt_templates.py   | Doctor assistant prompts (standard, CoT, instruction) |
| doctor_prompts.py     | Doctor-side dialogue prompts                 |
| patient_prompts.py    | Patient-side dialogue prompts                |
| evaluator_prompts.py  | Judge / failure-rate evaluation prompts      |
| negative_prompt.py    | Negative-case evaluation (e.g. overreaction) |
| synthetic_patient_prompts.py | Synthetic patient generation prompts  |

## Usage

Import from project root (scripts add repo root to `sys.path`):

```python
from prompts.prompt_templates import DOCTOR_ASSISTANT_PROMPT, COT_DOCTOR_ASSISTANT_PROMPT
from prompts.evaluator_prompts import ...
```

## Paths

All paths in callers are relative to the repository root. This package has no config files; templates are string constants or functions in the modules above.
