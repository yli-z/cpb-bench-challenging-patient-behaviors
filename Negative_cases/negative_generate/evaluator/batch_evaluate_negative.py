#!/usr/bin/env python3
"""
Batch evaluation script for overreaction (negative-case) detection.

Processes all JSONL files in a directory (or an explicit list) sequentially,
calling NegativeEvaluator for each one and writing per-model result files.

Location : Negative_cases/negative_generate/evaluator/batch_evaluate_negative.py
Input    : Negative_cases/negative_generate/generated/*.jsonl  (default)
Output   : Negative_cases/negative_generate/evaluator/output/

Usage:
    # Evaluate all files in the generated/ directory:
    python Negative_cases/negative_generate/evaluator/batch_evaluate_negative.py \\
        --judge_model_name gpt-4o \\
        --judge_model_type openai \\
        --input_dir  Negative_cases/negative_generate/generated \\
        --output_dir Negative_cases/negative_generate/evaluator/output

    # Or pass explicit files:
    python Negative_cases/negative_generate/evaluator/batch_evaluate_negative.py \\
        --judge_model_name gpt-4o \\
        --judge_model_type openai \\
        --input_files Negative_cases/negative_generate/generated/gpt-4o_ACI.jsonl \\
                      Negative_cases/negative_generate/generated/gpt-4o_MedDG.jsonl \\
        --output_dir  Negative_cases/negative_generate/evaluator/output
"""

import argparse
import asyncio
import sys
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# sys.path setup — must run before any project-level imports.
# Adds: (1) project root (CPB-Bench/) for models/prompts/evaluators
#        (2) this directory for sibling imports (evaluate_negative, etc.)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_THIS_DIR = Path(__file__).resolve().parent
for _p in [str(_PROJECT_ROOT), str(_THIS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Project-level imports
# ---------------------------------------------------------------------------
from evaluators_single.scripts.evaluate import create_judge_model, load_model_outputs
from evaluators_single.scripts.utils import (
    generate_excel_from_json_files,
    parse_model_dataset_from_filename,
)

# ---------------------------------------------------------------------------
# Local sibling import
# ---------------------------------------------------------------------------
from evaluate_negative import evaluate_all_responses_negative


# ---------------------------------------------------------------------------
# Per-file helper
# ---------------------------------------------------------------------------


def process_single_file_negative(
    input_file: str,
    judge_model,
    output_dir: str,
    skip_if_exists: bool = True,
    max_entries: int = None,
    concurrency: int = 20,
) -> bool:
    """
    Evaluate a single JSONL file for overreaction.

    Args:
        input_file: Path to the input JSONL file.
        judge_model: Shared judge model instance.
        output_dir: Directory where results are written.
        skip_if_exists: Skip the file if results already exist.
        max_entries: If set, only evaluate the first N entries (for quick testing).
        concurrency: Number of parallel worker threads for API calls.

    Returns:
        True if the file was processed; False if skipped or errored.
    """
    try:
        model_name, dataset = parse_model_dataset_from_filename(input_file)
    except ValueError as exc:
        print(f"  Skipping {input_file}: {exc}")
        return False

    print(f"\n{'='*60}")
    print(f"Processing : {input_file}")
    print(f"Model      : {model_name}")
    print(f"Dataset    : {dataset}")
    print(f"{'='*60}\n")

    data = load_model_outputs(input_file)
    if max_entries:
        data = data[:max_entries]
        print(f"Loaded {len(data)} entries (capped at --n {max_entries})")
    else:
        print(f"Loaded {len(data)} entries")

    result = evaluate_all_responses_negative(
        data=data,
        judge_model=judge_model,
        output_dir=output_dir,
        model_name=model_name,
        dataset=dataset,
        skip_if_exists=skip_if_exists,
        generate_excel=False,  # deferred until all files are done
        concurrency=concurrency,
    )
    return result is not None


# ---------------------------------------------------------------------------
# Async file-level concurrency
# ---------------------------------------------------------------------------


async def _process_file_async(
    semaphore: asyncio.Semaphore,
    input_file: str,
    judge_model,
    output_dir: str,
    skip_if_exists: bool,
    max_entries: int,
    concurrency: int,
    idx: int,
    total: int,
) -> tuple:
    """Run one file's evaluation in a thread, guarded by a semaphore."""
    async with semaphore:
        print(f"\n[{idx}/{total}] Starting: {input_file}")
        try:
            ok = await asyncio.to_thread(
                process_single_file_negative,
                input_file=input_file,
                judge_model=judge_model,
                output_dir=output_dir,
                skip_if_exists=skip_if_exists,
                max_entries=max_entries,
                concurrency=concurrency,
            )
            return ok, input_file, None
        except Exception as exc:
            traceback.print_exc()
            return False, input_file, exc


async def _run_all_async(args, input_files, judge_model):
    """Dispatch all files concurrently, capped at --file_concurrency."""
    semaphore = asyncio.Semaphore(args.file_concurrency)
    total = len(input_files)

    tasks = [
        _process_file_async(
            semaphore=semaphore,
            input_file=f,
            judge_model=judge_model,
            output_dir=args.output_dir,
            skip_if_exists=args.skip_if_exists,
            max_entries=args.n,
            concurrency=args.concurrency,
            idx=i,
            total=total,
        )
        for i, f in enumerate(input_files, 1)
    ]

    outcomes = await asyncio.gather(*tasks)

    processed = sum(1 for ok, _, err in outcomes if ok and err is None)
    skipped   = sum(1 for ok, _, err in outcomes if not ok and err is None)
    errors    = sum(1 for _,  _, err in outcomes if err is not None)
    return processed, skipped, errors


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Batch overreaction (negative-case) evaluation"
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
        "--input_files", type=str, nargs="+", default=None,
        help="One or more explicit JSONL file paths",
    )
    parser.add_argument(
        "--input_dir", type=str, default=None,
        help="Directory containing JSONL files (alternative to --input_files). "
             "Defaults to the sibling generated/ folder when neither flag is given.",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=str(_THIS_DIR / "output"),
        help="Output directory (default: <this_dir>/output)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0,
        help="Temperature for the judge model (default: 0)",
    )
    parser.add_argument(
        "--skip_if_exists", action="store_true", default=True,
        help="Skip files whose output already exists (default: True)",
    )
    parser.add_argument(
        "--no-skip_if_exists", dest="skip_if_exists", action="store_false",
        help="Force overwrite existing results",
    )
    parser.add_argument(
        "--n", type=int, default=None,
        help="Only evaluate the first N entries per file (for quick sample testing)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=20,
        help="Number of parallel API calls within each file (default: 20)",
    )
    parser.add_argument(
        "--file_concurrency", type=int, default=4,
        help="Number of JSONL files evaluated concurrently (default: 4)",
    )

    args = parser.parse_args()

    # Default input_dir: sibling generated/ folder
    if not args.input_files and not args.input_dir:
        args.input_dir = str(_THIS_DIR.parent / "generated")
        print(f"No --input_dir / --input_files given; defaulting to: {args.input_dir}")

    # Collect files
    input_files = list(args.input_files or [])
    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"Error: directory not found: {args.input_dir}")
            return
        found = sorted(input_dir.glob("*.jsonl"))
        if not found:
            print(f"Error: no .jsonl files in {args.input_dir}")
            return
        input_files.extend(str(f) for f in found)

    if not input_files:
        print("Error: no input files specified or found.")
        return

    print(f"Found {len(input_files)} file(s) to process")
    print(f"File concurrency : {args.file_concurrency}  "
          f"(each file uses up to {args.concurrency} parallel API calls)\n")

    # Build the judge model once — shared across all files.
    print(f"Creating judge model: {args.judge_model_name} ({args.judge_model_type})")
    judge_model = create_judge_model(
        judge_model_name=args.judge_model_name,
        judge_model_type=args.judge_model_type,
        temperature=args.temperature,
    )

    # Run all files concurrently
    processed, skipped, errors = asyncio.run(
        _run_all_async(args, input_files, judge_model)
    )

    # Generate a combined Excel report from all JSON result files.
    print(f"\n{'='*60}")
    print("Generating combined Excel report...")
    print(f"{'='*60}")
    try:
        generate_excel_from_json_files(args.output_dir)
    except Exception as exc:
        print(f"  Could not generate Excel: {exc}")

    print(f"\n{'='*60}")
    print("Batch overreaction evaluation complete!")
    print(f"  Processed : {processed}")
    print(f"  Skipped   : {skipped}")
    print(f"  Errors    : {errors}")
    print(f"  Results   : {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
