import os
import sys
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Setup path for imports
ROOT = Path(__file__).resolve().parent
PARENT_DIR = ROOT.parent
MODEL_GENERATOR_DIR = PARENT_DIR / "model_generator"
for p in [str(PARENT_DIR), str(MODEL_GENERATOR_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model_generator_single.generate_response import build_model, convert_conversation_to_string

load_dotenv(os.path.join(str(PARENT_DIR), ".env"))


EXCLUDE_BEHAVIORS = {"Emotional Pressure"}


def load_all_cases():
    """Load all cases from the benchmark datasets and return a DataFrame.

    Excludes behavior categories listed in EXCLUDE_BEHAVIORS.
    """
    input_files = [
        (os.path.join(str(PARENT_DIR), "data_loader", "output", "ACI_safety_benchmark.json"), "ACI"),
        (os.path.join(str(PARENT_DIR), "data_loader", "output", "IMCS_safety_benchmark.json"), "IMCS"),
        (os.path.join(str(PARENT_DIR), "data_loader", "output", "MedDG_safety_benchmark.json"), "MedDG"),
        (os.path.join(str(PARENT_DIR), "data_loader", "output", "MediTOD_safety_benchmark.json"), "MediTOD"),
    ]

    all_cases = []
    for file_path, dataset_name in input_files:
        with open(file_path, "r") as f:
            data = json.load(f)
        for case in data["cases"]:
            if case.get("behavior_category") in EXCLUDE_BEHAVIORS:
                continue
            case["dataset"] = dataset_name
            all_cases.append(case)

    df = pd.DataFrame(all_cases)
    if EXCLUDE_BEHAVIORS:
        print(f"Excluded behaviors: {EXCLUDE_BEHAVIORS} -> {len(df)} cases loaded")
    return df


def load_negative_cases():
    """Load sampled negative cases and return a DataFrame compatible with the strategy pipeline."""
    neg_dir = os.path.join(str(PARENT_DIR), "Negative_cases", "Negative_sampling_segment")
    input_files = [
        (os.path.join(neg_dir, "ACI_negative_cases_sampled.json"),     "ACI"),
        (os.path.join(neg_dir, "IMCS21_negative_cases_sampled.json"),   "IMCS"),
        (os.path.join(neg_dir, "MedDG_negative_cases_sampled.json"),    "MedDG"),
        (os.path.join(neg_dir, "MediTOD_negative_cases_sampled.json"),  "MediTOD"),
    ]

    all_cases = []
    for file_path, dataset_name in input_files:
        with open(file_path, "r") as f:
            data = json.load(f)
        for case in data:
            # Normalize dataset label (file stores "IMCS21", pipeline expects "IMCS")
            case["dataset"] = dataset_name
            # Synthesise a case_id so strategy_executor doesn't KeyError
            case["case_id"] = f"{dataset_name}_{case['dialog_id']}"
            # Negative cases have no behavior category; set a placeholder
            case.setdefault("behavior_category", "No behavior")
            case.setdefault("patient_behavior_text", "")
            # Keep conversation_segment as a list (strategy_executor expects a list
            # and calls convert_conversation_to_string itself)
            all_cases.append(case)

    return pd.DataFrame(all_cases)


def save_results(df, output_dir, model_name):
    """Save processed results as JSONL files grouped by dataset."""
    os.makedirs(output_dir, exist_ok=True)
    safe_model_name = model_name.replace("/", "_").replace(":", "_")
    for dataset, group in df.groupby('dataset'):
        file_path = os.path.join(output_dir, f"{safe_model_name}_{dataset}.jsonl")
        group.to_json(file_path, orient="records", lines=True, force_ascii=False)
        print(f"Saved {len(group)} cases to {file_path}")
