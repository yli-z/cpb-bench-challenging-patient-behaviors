#!/usr/bin/env python3
"""
Single-file evaluation script for overreaction (negative-case) detection.

Evaluates a single JSONL file of doctor responses for overreaction,
using NegativeEvaluator and GPT's structured JSON output format.

Location : Negative_cases/negative_generate/evaluator/evaluate_negative.py
Input    : JSONL files from Negative_cases/negative_generate/generated/
Output   : Negative_cases/negative_generate/evaluator/output/

Usage:
    python Negative_cases/negative_generate/evaluator/evaluate_negative.py \\
        --judge_model_name gpt-4o \\
        --judge_model_type openai \\
        --input_file Negative_cases/negative_generate/generated/gpt-4o_ACI.jsonl \\
        --output_dir Negative_cases/negative_generate/evaluator/output

    # Or as a module:
    python -m Negative_cases.negative_generate.evaluator.evaluate_negative \\
        --judge_model_name gpt-4o --judge_model_type openai \\
        --input_file ... --output_dir ...
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

# Module-level lock: protects the shared summary-report JSON from concurrent writes
# when multiple files are evaluated in parallel (e.g. via asyncio.to_thread).
_summary_lock = threading.Lock()

from tqdm import tqdm

# ---------------------------------------------------------------------------
# sys.path setup — must run before any project-level imports.
# Adds: (1) project root for models/prompts/evaluators
#        (2) this directory for sibling imports (negative_evaluator, etc.)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_THIS_DIR = Path(__file__).resolve().parent
for _p in [str(_PROJECT_ROOT), str(_THIS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Project-level imports (resolved via project root on sys.path)
# ---------------------------------------------------------------------------
from evaluators_single.scripts.evaluate import create_judge_model, load_model_outputs
from evaluators_single.scripts.utils import (
    load_summary_report,
    parse_model_dataset_from_filename,
    save_summary_report,
    update_summary_report,
)

# ---------------------------------------------------------------------------
# Local sibling import (resolved via _THIS_DIR on sys.path)
# ---------------------------------------------------------------------------
from negative_evaluator import NegativeEvaluator


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------


def _evaluate_single_entry(
    evaluator: "NegativeEvaluator",
    entry: Dict,
    model_name: str,
    dataset: str,
    judge_model_name: str,
    max_retries: int = 6,
) -> Dict:
    """Evaluate one entry with exponential back-off on 429 rate-limit errors."""
    dialog_id = entry.get("dialog_id", "")
    for attempt in range(1, max_retries + 1):
        try:
            result = evaluator.evaluate_overreaction(
                dialog_id=dialog_id,
                conversation_segment=entry.get("conversation_segment", ""),
                doctor_response=entry.get("response", ""),
            )
            result["model"] = model_name
            result["dataset"] = dataset
            result["turn_index"] = entry.get("turn_index", 0)
            result["response"] = entry.get("response", "")
            result["patient_behavior_text"] = entry.get("patient_behavior_text", "")
            result["behavior_category"] = entry.get("behavior_category", "Negative")
            result["judge_model"] = judge_model_name
            return result
        except Exception as exc:
            is_rate_limit = "429" in str(exc)
            if is_rate_limit and attempt < max_retries:
                wait = min(2 ** attempt, 60)   # 2 → 4 → 8 → 16 → 32 → 60 s
                print(f"\n  [429] {dialog_id} — retry {attempt}/{max_retries - 1} in {wait}s")
                time.sleep(wait)
            else:
                raise


def evaluate_all_responses_negative(
    data: List[Dict],
    judge_model,
    output_dir: str,
    model_name: str,
    dataset: str,
    skip_if_exists: bool = False,
    generate_excel: bool = True,
    concurrency: int = 20,
) -> Optional[Dict[str, Any]]:
    """
    Run the overreaction evaluator over every entry in *data* concurrently.

    Expected fields per entry:
      - ``dialog_id``            (str)
      - ``response``             (str) — the doctor's latest response
      - ``conversation_segment`` (str) — prior patient–doctor exchange

    Args:
        data: List of JSONL entries to evaluate.
        judge_model: Judge model instance.
        output_dir: Directory where results will be written.
        model_name: Model name (parsed from filename).
        dataset: Dataset name (parsed from filename).
        skip_if_exists: Skip saving if the output file already exists.
        generate_excel: Generate an Excel summary after saving (requires openpyxl).
        concurrency: Number of parallel worker threads for API calls (default 20).

    Returns:
        Dict with ``results`` key, or *None* if skipped.
    """
    # -----------------------------------------------------------------------
    # Early-exit if output already exists
    # -----------------------------------------------------------------------
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(output_dir) / f"{model_name}_{dataset}_negative_detailed_results.json"
    if skip_if_exists and output_file.exists():
        print(f"Skipped (already exists): {output_file}")
        return None

    evaluator = NegativeEvaluator(judge_model=judge_model, temperature=0)
    n = len(data)
    print(f"Evaluating {n} responses for overreaction  [concurrency={concurrency}]...")

    # Pre-allocate result slots to preserve input order.
    results: List[Optional[Dict]] = [None] * n

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_idx = {
            executor.submit(
                _evaluate_single_entry,
                evaluator,
                entry,
                model_name,
                dataset,
                judge_model.model_name,
            ): i
            for i, entry in enumerate(data)
        }

        with tqdm(total=n, desc="Overreaction eval") as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    print(f"\n  Error evaluating {data[idx].get('dialog_id')}: {exc}")
                pbar.update(1)

    # Drop any slots that errored (None).
    results_clean: List[Dict] = [r for r in results if r is not None]
    if len(results_clean) < n:
        print(f"  Warning: {n - len(results_clean)} entries failed and were skipped.")

    # -----------------------------------------------------------------------
    # Persist results
    # -----------------------------------------------------------------------
    with open(output_file, "w", encoding="utf-8") as fh:
        json.dump(results_clean, fh, indent=2, ensure_ascii=False)
    print(f"Saved detailed results → {output_file}")

    # Update summary report (lock prevents concurrent overwrites across parallel files)
    summary_file = Path(output_dir) / "negative_summary_report.json"
    with _summary_lock:
        existing = load_summary_report(str(summary_file))
        updated = update_summary_report(existing, model_name, results_clean, output_dir=output_dir)
        save_summary_report(updated, str(summary_file))
    print(f"Updated summary report → {summary_file}")

    # Optional Excel
    if generate_excel:
        try:
            from evaluators_single.scripts.utils import generate_excel_from_json_files
            generate_excel_from_json_files(output_dir)
        except Exception as exc:
            print(f"  Could not generate Excel: {exc}")

    # -----------------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------------
    total = len(results_clean)
    overreacted = sum(1 for r in results_clean if r.get("evaluation_result", False))
    print(f"\n{'='*50}")
    print("OVERREACTION EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"Model   : {model_name}")
    print(f"Dataset : {dataset}")
    print(f"Total   : {total}")
    if total:
        print(f"Overreacted (failures) : {overreacted}  ({overreacted / total * 100:.1f}%)")
    print(f"{'='*50}\n")

    return {"results": results_clean}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Single-file overreaction (negative-case) evaluation"
    )
    parser.add_argument(
        "--judge_model_name", type=str, required=True,
        help="Judge model name (e.g. 'gpt-4o', 'gpt-4')",
    )
    parser.add_argument(
        "--judge_model_type", type=str, required=True,
        choices=["openai", "claude", "gemini"],
        help="Judge model backend",
    )
    parser.add_argument(
        "--input_file", type=str, required=True,
        help="Input JSONL file path",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--temperature", type=float, default=0,
        help="Temperature for the judge model (default: 0)",
    )
    parser.add_argument(
        "--skip_if_exists", action="store_true", default=True,
        help="Skip if output file already exists (default: True)",
    )
    parser.add_argument(
        "--no-skip_if_exists", dest="skip_if_exists", action="store_false",
        help="Force overwrite existing results",
    )

    args = parser.parse_args()

    # Parse model / dataset from filename (e.g. gpt-4o_ACI.jsonl).
    try:
        model_name, dataset = parse_model_dataset_from_filename(args.input_file)
        print(f"Model: {model_name}  |  Dataset: {dataset}")
    except ValueError as exc:
        print(f"Error parsing filename: {exc}")
        return

    data = load_model_outputs(args.input_file)
    print(f"Loaded {len(data)} entries from {args.input_file}")

    judge = create_judge_model(
        judge_model_name=args.judge_model_name,
        judge_model_type=args.judge_model_type,
        temperature=args.temperature,
    )

    evaluate_all_responses_negative(
        data=data,
        judge_model=judge,
        output_dir=args.output_dir,
        model_name=model_name,
        dataset=dataset,
        skip_if_exists=args.skip_if_exists,
        generate_excel=False,
    )


if __name__ == "__main__":
    main()
