"""
Automated evaluation script for failure rate assessment.
Supports async parallel evaluation with configurable concurrency.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import json

from models.openai_model import OpenAIModel
from models.claude_model import ClaudeModel
from models.gemini_model import GeminiModel
from models.base_model import BaseLLM

# Import from current package
from evaluators_single.scripts.failure_rate_evaluator import FailureRateEvaluator
from evaluators_single.scripts.utils import (
    load_jsonl, save_jsonl, save_summary_report, format_failure_rate,
    parse_model_dataset_from_filename, save_model_results_json,
    load_summary_report, update_summary_report, generate_excel_report
)
from evaluators_single.scripts.judge_utils import group_by_behavior_category, find_min_context_segment
from evaluators_single.scripts.utils import calculate_model_statistics

def load_model_outputs(input_file: str) -> List[Dict]:
    """
    Load model output JSONL file.
    
    Args:
        input_file: JSONL file path
        
    Returns:
        List of dicts, each containing:
        {
            "dialog_id": str,
            "behavior_category": str,
            "model": str,
            "response": str,
            "dataset": str,
            "turn_index": int,
            "patient_behavior_text": str
        }
    """
    # if jsonl file
    if input_file.endswith(".jsonl"):
        return load_jsonl(input_file)
    # if json file
    elif input_file.endswith(".json"):
        with open(input_file, "r") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {input_file}")

def create_judge_model(
    judge_model_name: str,
    judge_model_type: str,
    temperature: float = 0
) -> BaseLLM:
    """
    Create judge model instance.
    
    Args:
        judge_model_name: Model name (e.g., "gpt-4", "claude-sonnet-4-5")
        judge_model_type: Model type ("openai", "claude", "gemini")
        temperature: Temperature parameter
        
    Returns:
        BaseLLM instance
    """
    if judge_model_type == "openai":
        return OpenAIModel(model_name=judge_model_name, temperature=temperature)
    elif judge_model_type == "claude":
        return ClaudeModel(model_name=judge_model_name, temperature=temperature)
    elif judge_model_type == "gemini":
        return GeminiModel(model_name=judge_model_name, temperature=temperature)
    else:
        raise ValueError(f"Unknown judge_model_type: {judge_model_type}")


def calculate_failure_rate(results: List[Dict]) -> Dict[str, Any]:
    """
    Calculate failure rate statistics.
    
    Args:
        results: List of evaluation results
        
    Returns:
        {
            "overall_failure_rate": "45.67%",
            "total_samples": 30,
            "failure_count": 14,
            "success_count": 16,
            "by_behavior_category": {...},
            "by_model": {...}
        }
    """
    total_samples = len(results)
    failure_count = sum(1 for r in results if r.get("evaluation_result", False))
    success_count = total_samples - failure_count
    
    overall_failure_rate = format_failure_rate(failure_count, total_samples)
    
    # Group by behavior_category
    by_behavior_category = {}
    for result in results:
        category = result.get("behavior_category", "Unknown")
        if category not in by_behavior_category:
            by_behavior_category[category] = {
                "total": 0,
                "failures": 0,
                "successes": 0
            }
        by_behavior_category[category]["total"] += 1
        if result.get("evaluation_result", False):
            by_behavior_category[category]["failures"] += 1
        else:
            by_behavior_category[category]["successes"] += 1
    
    # Calculate failure rate for each category
    for category, stats in by_behavior_category.items():
        stats["failure_rate"] = format_failure_rate(stats["failures"], stats["total"])
    
    # Group by model
    by_model = {}
    for result in results:
        model = result.get("model", "Unknown")
        if model not in by_model:
            by_model[model] = {
                "total": 0,
                "failures": 0,
                "successes": 0
            }
        by_model[model]["total"] += 1
        if result.get("evaluation_result", False):
            by_model[model]["failures"] += 1
        else:
            by_model[model]["successes"] += 1
    
    # Calculate failure rate for each model
    for model, stats in by_model.items():
        stats["failure_rate"] = format_failure_rate(stats["failures"], stats["total"])
    
    return {
        "overall_failure_rate": overall_failure_rate,
        "total_samples": total_samples,
        "failure_count": failure_count,
        "success_count": success_count,
        "by_behavior_category": by_behavior_category,
        "by_model": by_model
    }



def evaluate_all_responses(
    data: List[Dict],
    judge_model: BaseLLM,
    is_multi_turn: bool,
    context_mode: str,
    output_dir: str,
    model_name: str,
    dataset: str,
    skip_if_exists: bool = False,
    generate_excel: bool = True,
    multiturn_patient_strategy: str = None,
    concurrency: int = 16
) -> Dict[str, Any]:
    """
    Batch evaluate all responses.
    
    IMPORTANT: For each data entry:
    1. Extract behavior_category from the entry
    2. Get ONLY the corresponding principal: principal = get_principal(behavior_category)
    3. Pass only this single principal to the LLM judge
    4. The LLM judge evaluates based on only this one principal
    
    Args:
        data: List of data entries
        judge_model: Judge model instance
        context_mode: Context mode ("full_context" or "current_turn")
        output_dir: Output directory
        model_name: Model name being evaluated
        dataset: Dataset name
        skip_if_exists: If True, skip if model results file already exists with this dataset (default: False)
        generate_excel: If True, generate/update Excel report (default: True)
        
    Returns:
        Evaluation result dictionary containing detailed results and statistics
        Returns None if skipped
    """
    # Initialize judge
    judge = FailureRateEvaluator(
        judge_model=judge_model,
        temperature=0,
        is_multi_turn=is_multi_turn,
        context_mode=context_mode
    )
    
    # Evaluate each response
    results = []
    print(f"Evaluating {len(data)} responses (concurrency={concurrency})...")

    async def _evaluate_entry(entry, semaphore, pbar):
        """Evaluate a single entry with semaphore-controlled concurrency."""
        async with semaphore:
            try:
                behavior_category = entry.get("behavior_category") or entry.get("original_behavior")

                if not behavior_category:
                    print(f"Warning: Missing behavior_category for entry {entry.get('dialog_id')}")
                    pbar.update(1)
                    return None

                # Build evaluate_failure kwargs
                if not is_multi_turn:
                    if behavior_category == "Miexed-up Items":
                        item_to_detect = entry["mixed_up_item"]
                    elif behavior_category == "Misbelief Denial":
                        item_to_detect = entry["patient_behavior_text"]
                    elif behavior_category == "Unrelated Disease":
                        item_to_detect = entry["unrelated_disease"]
                    elif behavior_category == "More Test":
                        item_to_detect = entry["more_test"]
                    elif behavior_category == "More Medication":
                        item_to_detect = entry["more_medication"]
                    else:
                        item_to_detect = None

                    result = await asyncio.to_thread(
                        judge.evaluate_failure,
                        dialog_id=entry.get("dialog_id", ""),
                        behavior_category=behavior_category,
                        response=entry.get("response", "") or entry.get("generated_response", ""),
                        turn_index=entry.get("turn_index", 0),
                        patient_behavior_text=entry.get("patient_behavior_text", ""),
                        conversation_segment=entry.get("conversation_segment", ""),
                        item_to_detect=item_to_detect
                    )
                else:
                    result = await asyncio.to_thread(
                        judge.evaluate_failure,
                        dialog_id=entry.get("dialog_id", ""),
                        behavior_category=behavior_category,
                        turn_index=entry.get("turn_index", 0),
                        llm_conversation_segment=entry.get("llm_conversation_segment"),
                        response=None,
                        patient_behavior_text=None,
                    )

                # Add original entry information
                result["model"] = model_name
                result["dataset"] = dataset
                result["turn_index"] = entry.get("turn_index", 0)
                result["response"] = entry.get("response", "") or entry.get("generated_response", "")
                result["patient_behavior_text"] = entry.get("patient_behavior_text", "")
                result["behavior_category"] = behavior_category
                result['llm_conversation_segment'] = entry.get("llm_conversation_segment", "")
                result["judge_model"] = judge_model.model_name
                result["context_mode"] = context_mode

                if context_mode == "full_context" and "conversation_segment" in entry:
                    result["conversation_segment"] = entry["conversation_segment"]
                if context_mode == "min_turn" and "conversation_segment" in entry:
                    result["conversation_segment"] = find_min_context_segment(entry, behavior_category)

                pbar.update(1)
                return result

            except Exception as e:
                print(f"Error evaluating entry {entry.get('dialog_id')}: {e}")
                pbar.update(1)
                return None

    async def _run_all():
        semaphore = asyncio.Semaphore(concurrency)
        pbar = tqdm(total=len(data), desc="Evaluating")
        tasks = [_evaluate_entry(entry, semaphore, pbar) for entry in data]
        all_results = await asyncio.gather(*tasks)
        pbar.close()
        return [r for r in all_results if r is not None]

    # Run async evaluation
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already in an async context — use nest_asyncio or run synchronously
        import nest_asyncio
        nest_asyncio.apply()
        results = asyncio.get_event_loop().run_until_complete(_run_all())
    else:
        results = asyncio.run(_run_all())
    
    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save model-specific detailed results (JSON)
    if is_multi_turn:
        assert multiturn_patient_strategy is not None, "multiturn_patient_strategy must be specified for multi-turn evaluation"
        model_results_file = Path(output_dir) / f"{model_name}_multiturn_{multiturn_patient_strategy}_detailed_results.json"
    else:
        model_results_file = Path(output_dir) / f"{model_name}_detailed_results.json"
    saved = save_model_results_json(str(model_results_file), dataset, results, context_mode, skip_if_exists=skip_if_exists)
    
    if not saved:
        print(f"Skipped saving {model_results_file} (already exists with dataset {dataset})")
        return None
    
    print(f"Saved model detailed results to: {model_results_file}")
    
    # Update summary report
    if is_multi_turn:
        assert multiturn_patient_strategy is not None, "multiturn_patient_strategy must be specified for multi-turn evaluation"
        summary_file = Path(output_dir) / f"multiturn_{multiturn_patient_strategy}_summary_report.json"
    else:
        summary_file = Path(output_dir) / "summary_report.json"
    existing_summary = load_summary_report(str(summary_file))
    updated_summary = update_summary_report(existing_summary, model_name, results, output_dir=output_dir)
    save_summary_report(updated_summary, str(summary_file))
    print(f"Updated summary report: {summary_file}")
    
    # Update Excel report (if enabled)
    if generate_excel:
        if is_multi_turn:
            context_mode_suffix = "multiturn_" + multiturn_patient_strategy
        else:
            context_mode_suffix = "singleturn"
        output_dir = Path(output_dir) / f"{context_mode_suffix}_excel_reports"
        generate_excel_report(output_dir, model_name, results, context_mode)
        if is_multi_turn:
            excel_file = Path(output_dir) / f"multiturn_{multiturn_patient_strategy}_detailed_results.xlsx"
        else:
            excel_file = Path(output_dir) / "detailed_results.xlsx"
        print(f"Updated Excel report: {excel_file}")
    
    # Calculate statistics for display
    model_stats = calculate_model_statistics(results)
    summary = {
        "overall_failure_rate": model_stats["failure_rate"],
        "total_samples": model_stats["total_samples"],
        "failure_count": model_stats["failure_count"],
        "success_count": model_stats["success_count"],
        "by_behavior_category": model_stats["by_behavior_category"],
        "by_model": {model_name: model_stats}
    }
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Overall Failure Rate: {summary['overall_failure_rate']}")
    print(f"Total Samples: {summary['total_samples']}")
    print(f"Failures: {summary['failure_count']}")
    print(f"Successes: {summary['success_count']}")
    print("\nBy Behavior Category:")
    for category, stats in summary['by_behavior_category'].items():
        print(f"  {category}: {stats['failure_rate']} ({stats['failures']}/{stats['total']})")
    print("\nBy Model:")
    for model, stats in summary['by_model'].items():
        print(f"  {model}: {stats['failure_rate']} ({stats['failure_count']}/{stats['total_samples']})")
    print("="*50)
    
    return {
        "results": results,
        "summary": summary
    }


def main(args):
    """
    Main function.
    """
    # Parse model name and dataset from input filename
    try:
        model_name, dataset = parse_model_dataset_from_filename(args.input_file)
        print(f"Parsed from filename - Model: {model_name}, Dataset: {dataset}")
    except ValueError as e:
        print(f"Error parsing filename: {e}")
        print("Please ensure input file follows format: {{model_name}}_{{dataset}}.jsonl")
        return
    
    # Load data
    print(f"Loading data from: {args.input_file}")
    data = load_model_outputs(args.input_file)
    print(f"Loaded {len(data)} entries")
    
    # Create judge model
    print(f"Creating judge model: {args.judge_model_name} ({args.judge_model_type})")
    judge_model = create_judge_model(
        judge_model_name=args.judge_model_name,
        judge_model_type=args.judge_model_type,
        temperature=args.temperature
    )
    
    # Evaluate
    evaluate_all_responses(
        data=data,
        judge_model=judge_model,
        is_multi_turn=args.is_multi_turn,
        context_mode=args.context_mode,
        output_dir=args.output_dir,
        model_name=model_name,
        dataset=dataset,
        multiturn_patient_strategy=args.multiturn_patient_strategy,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated failure rate evaluation using LLM-as-Judge")
    parser.add_argument("--judge_model_name", type=str, required=True,
                       help="Judge model name (e.g., 'gpt-4', 'claude-sonnet-4-5')")
    parser.add_argument("--judge_model_type", type=str, required=True,
                       choices=["openai", "claude", "gemini"],
                       help="Judge model type")
    parser.add_argument("--context_mode", type=str, required=True,
                       choices=["full_context", "current_turn", "min_turn"],
                       help="Context mode: 'full_context', 'current_turn', or 'min_turn' 'min-turn' include the doctor's previous question as well")
    parser.add_argument("--input_file", type=str, required=True,
                       help="Input JSONL file path")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory")
    parser.add_argument("--temperature", type=float, default=0,
                       help="Temperature parameter (default: 0)")
    
    args = parser.parse_args()
    main(args)

