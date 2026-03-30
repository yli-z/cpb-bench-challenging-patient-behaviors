"""
resample_negative_cases.py

Re-sample ONE truncation point per negative case drawn from the complete
per-dataset positive-case empirical distribution.

Rationale
---------
The previous approach forced 2 samples per case (lower-half / upper-half),
which artificially created a bimodal distribution.
This script draws a single bootstrap sample from the observed positive-case
turn-position percentages so that the 352 truncation points naturally mirror
each dataset's positive-case distribution.

Sampling logic (per case)
-------------------------
1.  Collect pct = turn_index / len(complete_conversation) for every positive
    case in the same dataset (excluding "Emotional Pressure").
2.  Draw one pct via random.choice() (bootstrap from empirical distribution).
3.  Snap to the nearest Patient turn in the negative case's conversation.

Input  : Negative_cases/Negative_original_case_data/{DS}_negative_cases.json
         data_loader/output/{DS}_safety_benchmark.json
Output : Negative_cases/Negative_sampling_segment/{DS}_negative_cases_sampled.json
         (overwrites the old 2-per-case files)

Usage
-----
    python Negative_cases/resample_negative_cases.py
"""

import json
import random
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

SEED = 42
random.seed(SEED)

ROOT         = Path(__file__).resolve().parent.parent   # CPB-Bench/
NEG_DATA_DIR = ROOT / "Negative_cases" / "Negative_original_case_data"
BENCH_DIR    = ROOT / "data_loader" / "output"
OUT_DIR      = ROOT / "Negative_cases" / "Negative_sampling_segment"
OUT_DIR.mkdir(exist_ok=True)

# (neg_file,  bench_file,                   out_file)
DATASET_CONFIG = {
    "ACI":     ("ACI_negative_cases.json",
                "ACI_safety_benchmark.json",
                "ACI_negative_cases_sampled.json"),
    "IMCS21":  ("IMCS21_negative_cases.json",
                "IMCS_safety_benchmark.json",
                "IMCS21_negative_cases_sampled.json"),
    "MedDG":   ("MedDG_negative_cases.json",
                "MedDG_safety_benchmark.json",
                "MedDG_negative_cases_sampled.json"),
    "MediTOD": ("MediTOD_negative_cases.json",
                "MediTOD_safety_benchmark.json",
                "MediTOD_negative_cases_sampled.json"),
}

# ── Helper functions ──────────────────────────────────────────────────────────

def build_pos_pcts(bench_path: Path) -> list:
    """
    Return list of turn-position percentages from positive cases,
    excluding 'Emotional Pressure'.
    pct = turn_index / len(complete_conversation), clipped to [0, 1].
    """
    with open(bench_path, encoding="utf-8") as f:
        bench = json.load(f)
    pcts = []
    for c in bench["cases"]:
        if c.get("behavior_category") == "Emotional Pressure":
            continue
        total = len(c.get("complete_conversation", []))
        if total > 0:
            pct = min(c["turn_index"] / total, 1.0)  # clip >1 edge cases
            pcts.append(pct)
    return pcts


def get_patient_turns(conversation: list) -> list:
    """
    Return [(turn_index, position_pct), ...] for every Patient turn.
    conversation items have shape: {"Doctor"|"Patient": text, "turn index": int}
    """
    total = len(conversation)
    if total == 0:
        return []
    result = []
    for turn in conversation:
        if "Patient" in turn:
            ti = turn["turn index"]
            result.append((ti, ti / total))
    return result


def snap_to_nearest_patient_turn(patient_turns: list, target_pct: float):
    """Return (turn_index, actual_pct) of the Patient turn closest to target_pct."""
    if not patient_turns:
        return None, None
    best_ti, best_pct = min(patient_turns, key=lambda x: abs(x[1] - target_pct))
    return best_ti, best_pct


def build_conversation_segment(conversation: list, turn_index: int) -> list:
    """Return turns from the start up to and including turn_index."""
    return [t for t in conversation if t["turn index"] <= turn_index]


# ── Main ──────────────────────────────────────────────────────────────────────

print("=" * 60)
print(" Resampling negative cases (1 sample per case)")
print(" Source: Negative_original_case_data/")
print(" Output: Negative_sampling_segment/")
print("=" * 60)

total_records = 0

for ds_label, (neg_file, bench_file, out_file) in DATASET_CONFIG.items():
    neg_path   = NEG_DATA_DIR / neg_file
    bench_path = BENCH_DIR    / bench_file
    out_path   = OUT_DIR      / out_file

    # ── Load positive-case distribution ──────────────────────────────────────
    pos_pcts = build_pos_pcts(bench_path)
    if not pos_pcts:
        print(f"[WARN] {ds_label}: no positive pcts found in {bench_file}")
        continue

    # ── Load negative cases ───────────────────────────────────────────────────
    with open(neg_path, encoding="utf-8") as f:
        neg_cases = json.load(f)

    print(f"\n{ds_label}")
    print(f"  Positive distribution : {len(pos_pcts)} samples, "
          f"range=[{min(pos_pcts):.3f}, {max(pos_pcts):.3f}], "
          f"mean={sum(pos_pcts)/len(pos_pcts):.3f}")
    print(f"  Negative cases        : {len(neg_cases)}")

    # ── Sample one truncation point per negative case ─────────────────────────
    records = []
    skipped = 0

    for case in neg_cases:
        conv          = case["conversation"]
        patient_turns = get_patient_turns(conv)

        if not patient_turns:
            print(f"  [WARN] No Patient turns in dialog_id={case['dialog_id']} — skipped")
            skipped += 1
            continue

        # Bootstrap sample from the empirical positive-case distribution
        target_pct = random.choice(pos_pcts)

        # Snap to the nearest actual Patient turn
        turn_index, actual_pct = snap_to_nearest_patient_turn(patient_turns, target_pct)

        conv_seg = build_conversation_segment(conv, turn_index)

        records.append({
            "dataset":              ds_label,
            "dialog_id":            case["dialog_id"],
            "conversation":         conv,
            "turn_index":           turn_index,
            "conversation_segment": conv_seg,
            "target_pct":           round(target_pct, 4),
            "actual_pct":           round(actual_pct, 4),
        })

    # ── Write output ──────────────────────────────────────────────────────────
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    actual_pcts = [r["actual_pct"] for r in records]
    print(f"  Sampled actual_pct    : range=[{min(actual_pcts):.3f}, {max(actual_pcts):.3f}], "
          f"mean={sum(actual_pcts)/len(actual_pcts):.3f}")
    print(f"  → {len(records)} records saved to {out_path.name}"
          + (f"  ({skipped} skipped)" if skipped else ""))

    total_records += len(records)

print(f"\n{'=' * 60}")
print(f"Done. Total records across all datasets: {total_records}")
print(f"Output directory: {OUT_DIR}")
print(f"{'=' * 60}")
