#!/usr/bin/env python3
"""
Sample false-negative (LLM judge = False) cases for human annotation.

Sampling dimensions:
  - 11 models × 4 behavior categories = 44 cells
  - 5 cases per cell → 220 cases per sample
  - 2 samples → 440 cases total

Uniqueness:
  - Within each sample (220): all dialog_ids are unique
  - Across both samples (440): best-effort; allow minimal reuse only when
    a category has too few unique dialog_ids (Factual Inaccuracy, Info Contradiction)

Output: False_negative/sample_output/sample1.xlsx  and  sample2.xlsx
"""

import json
import random
from pathlib import Path

import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────────
SEED = 42
N_PER_CELL = 5
CATEGORIES = [
    "Care Resistance",
    "Factual Inaccuracy",
    "Information Contradiction",
    "Self-diagnosis",
]
INPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR = Path(__file__).parent / "sample_output"
OUTPUT_COLUMNS = [
    "model", "dataset", "behavior_category", "dialog_id", "turn_index",
    "evaluation_result", "patient_behavior_text", "response",
    "conversation_segment", "human check1", "human check2", "human check3",
]

# ── Load data ─────────────────────────────────────────────────────────────────

def load_all_false_cases(input_dir: Path) -> dict:
    """
    Returns a nested dict:
        pools[(model_name, category)] = [case, case, ...]
    where each case dict contains all original fields.
    """
    pools: dict[tuple, list] = {}
    files = sorted(input_dir.glob("*_false_cases.json"))
    if not files:
        raise FileNotFoundError(f"No *_false_cases.json files found in {input_dir}")

    print(f"Loading {len(files)} model files …")
    for f in files:
        model_name = f.stem.replace("_false_cases", "")
        with open(f, encoding="utf-8") as fp:
            data = json.load(fp)

        for dataset_key, cases in data.items():
            for case in cases:
                bc = case.get("behavior_category", "")
                if bc not in CATEGORIES:
                    continue
                key = (model_name, bc)
                pools.setdefault(key, []).append(case)

    return pools


# ── Sampling ──────────────────────────────────────────────────────────────────

def sample_one_batch(
    pools: dict,
    globally_used: set,
    rng: random.Random,
) -> list[dict]:
    """
    Sample one batch of 220 cases (N_PER_CELL per cell).

    Priority for each cell:
      1. candidates not in used_in_this_batch AND not in globally_used
      2. candidates not in used_in_this_batch  (cross-sample reuse allowed)
      3. any candidates  (within-sample dedup best-effort when pool < 5)

    Returns list of selected case dicts.
    """
    # Collect all cells and shuffle order to avoid bias
    cells = list(pools.keys())
    rng.shuffle(cells)

    used_in_this_batch: set = set()
    selected_rows: list[dict] = []

    for cell in cells:
        model_name, bc = cell
        pool = pools[cell][:]          # shallow copy so we can shuffle safely
        rng.shuffle(pool)

        # Build candidate tiers
        tier1 = [c for c in pool
                 if c["dialog_id"] not in used_in_this_batch
                 and c["dialog_id"] not in globally_used]
        tier2 = [c for c in pool
                 if c["dialog_id"] not in used_in_this_batch]

        chosen: list = []
        if len(tier1) >= N_PER_CELL:
            chosen = tier1[:N_PER_CELL]
        elif len(tier2) >= N_PER_CELL:
            # Use as many tier-1 as possible, fill rest from tier-2
            chosen = tier1[:]
            remaining = [c for c in tier2 if c not in chosen]
            chosen += remaining[: N_PER_CELL - len(chosen)]
        else:
            # Absolute fallback: take whatever is available from tier2,
            # then allow any case (minimises within-sample repeats)
            chosen = tier2[:]
            if len(chosen) < N_PER_CELL:
                extra = [c for c in pool if c not in chosen]
                chosen += extra[: N_PER_CELL - len(chosen)]
            chosen = chosen[:N_PER_CELL]

        # Record selected dialog_ids
        for c in chosen:
            used_in_this_batch.add(c["dialog_id"])

        selected_rows.extend(chosen)

    return selected_rows


# ── Build DataFrame ───────────────────────────────────────────────────────────

def cases_to_dataframe(cases: list[dict]) -> pd.DataFrame:
    rows = []
    for c in cases:
        rows.append({
            "model":                 c.get("model", ""),
            "dataset":               c.get("dataset", ""),
            "behavior_category":     c.get("behavior_category", ""),
            "dialog_id":             c.get("dialog_id", ""),
            "turn_index":            c.get("turn_index", ""),
            "evaluation_result":     c.get("evaluation_result", ""),
            "patient_behavior_text": c.get("patient_behavior_text", ""),
            "response":              c.get("response", ""),
            "conversation_segment":  c.get("conversation_segment", ""),
            "human check1":          "",
            "human check2":          "",
            "human check3":          "",
        })
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    return df


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pools = load_all_false_cases(INPUT_DIR)

    # Report pool sizes
    models = sorted({k[0] for k in pools})
    print(f"\nModels found ({len(models)}): {models}")
    print(f"Categories used: {CATEGORIES}")
    print("\nPool sizes per (model, category):")
    for cat in CATEGORIES:
        sizes = {m: len(pools.get((m, cat), [])) for m in models}
        total_unique = len({c["dialog_id"] for m in models for c in pools.get((m, cat), [])})
        min_sz = min(sizes.values())
        print(f"  {cat:30s} | unique dialog_ids across all models: {total_unique:4d} "
              f"| per-model min: {min_sz}")

    rng = random.Random(SEED)

    globally_used: set = set()
    for batch_idx in range(1, 3):
        print(f"\n── Sampling batch {batch_idx} ──────────────────────────────────")
        cases = sample_one_batch(pools, globally_used, rng)

        # Update globally_used for next batch
        for c in cases:
            globally_used.add(c["dialog_id"])

        df = cases_to_dataframe(cases)

        # ── Diagnostics ──
        print(f"  Total rows: {len(df)}")
        unique_dialogs = df["dialog_id"].nunique()
        print(f"  Unique dialog_ids in this sample: {unique_dialogs}")
        if unique_dialogs < len(df):
            dup_count = len(df) - unique_dialogs
            print(f"  ⚠ Within-sample dialog_id repeats: {dup_count} "
                  f"(expected for categories with limited data)")
        print("  Per (model, category) counts:")
        counts = df.groupby(["model", "dataset"]).size() if False else \
                 df.assign(category=[c.get("behavior_category","") for c in cases]) \
                   .groupby(["model", "category"]).size()
        for (m, cat), n in counts.items():
            print(f"    {m:50s} | {cat:30s} | {n}")

        out_path = OUTPUT_DIR / f"sample{batch_idx}.xlsx"
        df.to_excel(out_path, index=False)
        print(f"  Saved → {out_path}")

    # Cross-batch overlap report
    print(f"\n── Cross-sample dialog_id overlap ──────────────────────────────")
    # Reload both files to compute overlap
    df1 = pd.read_excel(OUTPUT_DIR / "sample1.xlsx")
    df2 = pd.read_excel(OUTPUT_DIR / "sample2.xlsx")
    overlap = set(df1["dialog_id"]) & set(df2["dialog_id"])
    print(f"  sample1 unique dialog_ids: {df1['dialog_id'].nunique()}")
    print(f"  sample2 unique dialog_ids: {df2['dialog_id'].nunique()}")
    print(f"  Shared dialog_ids between samples: {len(overlap)}")
    if overlap:
        print(f"  (These are expected reuses from low-data categories)")

    print("\nDone.")


if __name__ == "__main__":
    main()
