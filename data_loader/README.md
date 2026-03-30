# Data Loader

Load and process annotated medical dialogue data for the safety benchmark. Outputs go under `data_loader/output_benchmark`.

## Contents

- **get_data.py** — Main script: load source dialogs, merge Excel annotations, produce benchmark JSON per dataset.
- **config.py** — Dataset config (paths, Excel dirs, format).
- **utils.py** — Helpers for loading dialogs and annotations.
- **instruction.sh** — Wrapper for instruction-style data prep.

## Usage

Run from project root:

```bash
# Single dataset (e.g. ACI, MediTOD, IMCS, MedDG)
python data_loader/get_data.py --dataset ACI

# All configured datasets
python data_loader/get_data.py --dataset all
```

## Output

Processed benchmark files are written under `data_loader/output_benchmark/` (e.g. `ACI_safety_benchmark.json`). Other modules (e.g. `model_generator_single`, `syn_generator`) read from `data_loader/output_benchmark` or equivalent paths.

## Paths

- Input: Source dialogs and Excel annotations as defined in `data_loader/config.py`.
- Output: `data_loader/output_benchmark/`.
