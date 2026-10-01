#!/usr/bin/env python3
"""
Batch evaluation script for multiple model output files.

This script processes multiple model output files sequentially,
allowing batch evaluation of multiple models and datasets.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"))

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evaluators_single.scripts.evaluate import create_judge_model, load_model_outputs, evaluate_all_responses
from evaluators_single.scripts.utils import parse_model_dataset_from_filename, generate_excel_from_json_files
from models.base_model import BaseLLM


def process_single_file(
    input_file: str,
    judge_model: BaseLLM,
    context_mode: str,
    output_dir: str,
    is_multi_turn: bool,
    multiturn_patient_strategy: str = None,
    skip_if_exists: bool = True,
    generate_excel: bool = False,
    concurrency: int = 16
):
    """
    Process a single input file.
    
    Args:
        input_file: Path to input JSONL file
        judge_model: Judge model instance
        context_mode: Context mode ("full_context" or "current_turn")
        output_dir: Output directory
        skip_if_exists: If True, skip if model results file already exists with this dataset
        generate_excel: If True, generate/update Excel report (usually False for batch mode)
    """
    # Parse model name and dataset from filename
    try:
        model_name, dataset = parse_model_dataset_from_filename(input_file)
        print(f"\n{'='*60}")
        print(f"Processing: {input_file}")
        print(f"Model: {model_name}, Dataset: {dataset}")
        print(f"{'='*60}\n")
    except ValueError as e:
        print(f"Error parsing filename {input_file}: {e}")
        print("Skipping this file...")
        return False
    
    # Load data
    data = load_model_outputs(input_file)
    print(f"Loaded {len(data)} entries")
    # Evaluate
    result = evaluate_all_responses(
        data=data,
        judge_model=judge_model,
        context_mode=context_mode,
        is_multi_turn=is_multi_turn,
        multiturn_patient_strategy=multiturn_patient_strategy,
        output_dir=output_dir,
        model_name=model_name,
        dataset=dataset,
        skip_if_exists=skip_if_exists,
        generate_excel=generate_excel,
        concurrency=concurrency
    )
    
    return result is not None  # Return True if processed, False if skipped


def main():
    """
    Main function for batch processing.
    """
    parser = argparse.ArgumentParser(
        description="Batch evaluation script for multiple model output files"
    )
    parser.add_argument(
        "--judge_model_name", type=str, required=True,
        help="Judge model name (e.g., 'gpt-4', 'claude-sonnet-4-5')"
    )
    parser.add_argument(
        "--judge_model_type", type=str, required=True,
        choices=["openai", "claude", "gemini"],
        help="Judge model type"
    )
    parser.add_argument(
        "--context_mode", type=str, required=True,
        choices=["full_context", "current_turn", "min_turn"],
        help="Context mode: 'full_context', 'current_turn', or 'min_turn'"
    )
    parser.add_argument(
        "--is_multi_turn", action="store_true",
         help="Enable multi-turn evaluation"
    )
    parser.add_argument(
        "--multiturn_patient_strategy", 
        type=str, default=None,
        choices=["direct", "persona", "self_eval", "retrieval", "baseline"],
        help="Patient strategy for multi-turn evaluation"
    )
    parser.add_argument(
        "--input_files", type=str, nargs='+', default=None,
        help="One or more input JSONL file paths (space-separated)"
    )
    parser.add_argument(
        "--input_dir", type=str, default=None,
        help="Directory containing input files (alternative to --input_files)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--temperature", type=float, default=0,
        help="Temperature parameter (default: 0)"
    )
    parser.add_argument(
        "--skip_if_exists", action="store_true", default=True,
        help="Skip processing if model results file already exists with this dataset (default: True)"
    )
    parser.add_argument(
        "--no-skip_if_exists", dest="skip_if_exists", action="store_false",
        help="Disable skip_if_exists (force overwrite)"
    )
    parser.add_argument(
        "--concurrency", type=int, default=16,
        help="Max concurrent async evaluation calls (default: 16)"
    )
    
    args = parser.parse_args()
    
    # Validate that at least one input source is provided
    if not args.input_files and not args.input_dir:
        parser.error("At least one of --input_files or --input_dir must be provided")
    
    # Collect input files
    input_files = []
    
    if args.input_files:
        input_files.extend(args.input_files)
    
    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"Error: Input directory does not exist: {args.input_dir}")
            return
        
        # Find all .jsonl files in the directory
        found_files = list(input_dir.glob("*.jsonl"))
        if not found_files:
            print(f"Error: No .jsonl files found in directory: {args.input_dir}")
            return
        input_files.extend([str(f) for f in found_files])
    
    if not input_files:
        print("Error: No input files specified or found")
        return
    
    print(f"Found {len(input_files)} input file(s) to process")
    
    # Create judge model (shared across all files)
    print(f"\nCreating judge model: {args.judge_model_name} ({args.judge_model_type})")
    judge_model = create_judge_model(
        judge_model_name=args.judge_model_name,
        judge_model_type=args.judge_model_type,
        temperature=args.temperature
    )
    
    # Process each file
    processed_count = 0
    skipped_count = 0

    for i, input_file in enumerate(input_files, 1):
        print(f"\n[{i}/{len(input_files)}] Processing: {input_file}")
        try:
            processed = process_single_file(
                input_file=input_file,
                judge_model=judge_model,
                context_mode=args.context_mode,
                output_dir=args.output_dir,
                is_multi_turn=args.is_multi_turn,
                multiturn_patient_strategy=args.multiturn_patient_strategy,
                skip_if_exists=args.skip_if_exists,
                generate_excel=False,  # Don't generate Excel during batch processing
                concurrency=args.concurrency
            )
            if processed:
                processed_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate Excel report from all JSON files
    print(f"\n{'='*60}")
    print("Generating Excel report from all JSON files...")
    print(f"{'='*60}")
    try:
        generate_excel_from_json_files(args.output_dir, context_mode=args.context_mode)
    except Exception as e:
        print(f"Error generating Excel report: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("Batch processing completed!")
    print(f"Processed: {processed_count}, Skipped: {skipped_count}")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

