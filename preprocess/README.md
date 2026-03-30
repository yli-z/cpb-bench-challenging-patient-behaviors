# Preprocess

Preprocessing and behavior detection for medical dialogues. Produces annotated data used by the benchmark and downstream modules.

## Contents

- **behavior_detector.py** — LLM-based detector for six patient behavior categories (e.g. information contradiction, factual inaccuracy, self-diagnosis, care resistance). Used to label or filter dialogues.
- **DATASET_CONFIG.json** — Dataset-level settings for preprocessing.
- **prompts/** — Prompts used by the behavior detector.

## Behavior categories

Information Contradiction, Critical Information Withholding, Factual Inaccuracy, Self-diagnosis, Care Resistance, Emotional Pressure.

## Usage

Typically used as a library or by data-prep pipelines that produce inputs for `data_loader` or Excel annotations. Instantiate with model type and name:

```python
from preprocess.behavior_detector import BehaviorDetector

detector = BehaviorDetector(model_type="openai", model_name="gpt-4o")
# Use detector to label dialogues
```

## Paths

Run scripts from project root. Config paths are relative to `preprocess/` (e.g. `preprocess/DATASET_CONFIG.json`).
