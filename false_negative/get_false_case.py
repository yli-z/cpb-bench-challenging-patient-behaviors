#!/usr/bin/env python3
"""
Extract all LLM judge "False" (not fail) cases from single-turn evaluation results.

Reads all *_detailed_results.json files in the output_single_turn directory,
filters cases where evaluation_result is False, and saves per-model JSON files.
"""

import json
from pathlib import Path


INPUT_DIR = Path(__file__).parent.parent / "evaluators/failure_rate_eval/output_single_turn"
OUTPUT_DIR = Path(__file__).parent / "output"


def extract_false_cases(input_file: Path) -> dict:
    """
    Load a detailed_results JSON file and return only cases where
    evaluation_result is False, preserving the original key structure.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    false_cases = {}
    for dataset_key, cases in data.items():
        filtered = [
            case for case in cases
            if case.get("evaluation_result") is False
            and case.get("behavior_category") != "Emotional Pressure"  # We initially considered this category, but later excluded them.
        ]
        if filtered:
            false_cases[dataset_key] = filtered

    return false_cases


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_files = sorted(INPUT_DIR.glob("*_detailed_results.json"))
    if not input_files:
        print(f"No *_detailed_results.json files found in {INPUT_DIR}")
        return

    print(f"Found {len(input_files)} model result file(s) in {INPUT_DIR}\n")

    for input_file in input_files:
        model_name = input_file.name.replace("_detailed_results.json", "")
        print(f"Processing: {input_file.name}")

        false_cases = extract_false_cases(input_file)

        total = sum(len(v) for v in false_cases.values())
        print(f"  False (not fail) cases: {total}")
        for dataset_key, cases in false_cases.items():
            print(f"    {dataset_key}: {len(cases)}")

        output_file = OUTPUT_DIR / f"{model_name}_false_cases.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(false_cases, f, ensure_ascii=False, indent=2)
        print(f"  Saved -> {output_file}\n")

    print("Done.")


if __name__ == "__main__":
    main()
