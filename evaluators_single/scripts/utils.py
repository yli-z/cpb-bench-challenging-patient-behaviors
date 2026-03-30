"""
General utility functions for LLM-as-Judge evaluation.
"""

import json
from typing import List, Dict, Any, Tuple
from pathlib import Path
import pandas as pd


def load_jsonl(file_path: str) -> List[Dict]:
    """
    Load JSONL file.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of dictionaries, one per line
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], file_path: str):
    """
    Save data to JSONL file.
    
    Args:
        data: List of dictionaries to save
        file_path: Path to output JSONL file
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def save_results(results: List[Dict], output_path: str):
    """
    Save evaluation results to JSONL file.
    
    Args:
        results: List of evaluation result dictionaries
        output_path: Path to output file
    """
    save_jsonl(results, output_path)


def save_summary_report(summary: Dict, output_path: str):
    """
    Save summary report to JSON file.
    
    Args:
        summary: Summary dictionary
        output_path: Path to output JSON file
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def format_failure_rate(count: int, total: int) -> str:
    """
    Format failure rate as percentage string.
    
    Args:
        count: Number of failures
        total: Total number of samples
        
    Returns:
        String in "xx.xx%" format
    """
    if total == 0:
        return "0.00%"
    rate = (count / total) * 100
    return f"{rate:.2f}%"


def parse_model_dataset_from_filename(filename: str) -> Tuple[str, str]:
    """
    Parse model name and dataset name from filename.
    
    Expected format: {model_name}_{dataset}.jsonl
    Example: gpt-4_ACI.jsonl -> ("gpt-4", "ACI")
    
    Args:
        filename: Input filename (can be full path or just filename)
        
    Returns:
        Tuple of (model_name, dataset_name)
    """
    # Extract just the filename if it's a path
    filename = Path(filename).name
    
    # Remove .jsonl extension
    if filename.endswith('.jsonl'):
        filename = filename[:-6]
    elif filename.endswith('.json'):
        filename = filename[:-5]
    
    # Split by last underscore
    parts = filename.rsplit('_', 1)
    if "mixed_up_items" in filename:
        parts = filename.split("_mixed_up_items")
        model_name = parts[0]
        dataset_name = "mixed_up_items"
        return model_name, dataset_name

    if len(filename.split('_')) == 6 and parts[0].startswith("syn"): # if multiturn synthetic conversation file
        parts = filename.split('_')
        model_name, dataset_name = parts[-1], parts[1]
        return model_name, dataset_name
        
    if len(parts) != 2:
        raise ValueError(f"Cannot parse model and dataset from filename: {filename}. Expected format: {{model_name}}_{{dataset}}.jsonl")
    
    model_name, dataset_name = parts
    return model_name, dataset_name


def load_model_results_json(file_path: str) -> Dict[str, List[Dict]]:
    """
    Load model results JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary with dataset names as keys and lists of results as values
        Returns empty dict if file doesn't exist
    """
    if not Path(file_path).exists():
        return {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_model_results_json(file_path: str, dataset: str, new_results: List[Dict], context_mode: str, skip_if_exists: bool = False) -> bool:
    """
    Save model results JSON file with append/overwrite logic.
    
    Args:
        file_path: Path to JSON file
        dataset: Current dataset name (will overwrite if exists)
        new_results: New evaluation results for this dataset
        context_mode: Context mode ("full_context" or "current_turn")
        skip_if_exists: If True and file exists with this dataset, skip saving (return False)
        
    Returns:
        True if saved successfully, False if skipped
    """
    file_path_obj = Path(file_path)
    
    # Check if file exists and dataset already exists
    if skip_if_exists and file_path_obj.exists():
        existing_data = load_model_results_json(file_path)
        if dataset in existing_data:
            return False  # Skip saving
    
    # Load existing data if file exists
    existing_data = load_model_results_json(file_path)
    
    # Prepare results: remove judge_reasoning, add conversation_segment if full_context
    prepared_results = []
    for result in new_results:
        # Create a copy without judge_reasoning
        prepared_result = {k: v for k, v in result.items() if k != "judge_reasoning"}
        
        # Add conversation_segment if context_mode is full_context
        if context_mode == "full_context" and "conversation_segment" in result:
            prepared_result["conversation_segment"] = result["conversation_segment"]
        
        prepared_results.append(prepared_result)
    
    # Update or add dataset
    existing_data[dataset] = prepared_results
    
    # Save to file
    file_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)
    
    return True


def load_summary_report(file_path: str) -> Dict:
    """
    Load existing summary report JSON file.
    
    Args:
        file_path: Path to summary report JSON file
        
    Returns:
        Summary dictionary, or empty dict with default structure if file doesn't exist
    """
    if not Path(file_path).exists():
        return {
            "overall_failure_rate": "0.00%",
            "total_samples": 0,
            "failure_count": 0,
            "success_count": 0,
            "by_model": {},
            "by_behavior_category": {}
        }
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_model_statistics(results: List[Dict]) -> Dict[str, Any]:
    """
    Calculate statistics for a single model's results.
    
    Args:
        results: List of evaluation results for one model
        
    Returns:
        Dictionary with statistics including by_behavior_category and by_dataset
    """
    total_samples = len(results)
    failure_count = sum(1 for r in results if r.get("evaluation_result", False))
    success_count = total_samples - failure_count
    failure_rate = format_failure_rate(failure_count, total_samples)
    
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
    
    # Group by dataset
    by_dataset = {}
    for result in results:
        dataset = result.get("dataset", "Unknown")
        if dataset not in by_dataset:
            by_dataset[dataset] = {
                "total": 0,
                "failures": 0,
                "successes": 0
            }
        by_dataset[dataset]["total"] += 1
        if result.get("evaluation_result", False):
            by_dataset[dataset]["failures"] += 1
        else:
            by_dataset[dataset]["successes"] += 1
    
    # Calculate failure rate for each dataset
    for dataset, stats in by_dataset.items():
        stats["failure_rate"] = format_failure_rate(stats["failures"], stats["total"])
    
    return {
        "failure_rate": failure_rate,
        "total_samples": total_samples,
        "failure_count": failure_count,
        "success_count": success_count,
        "by_behavior_category": by_behavior_category,
        "by_dataset": by_dataset
    }


def update_summary_report(existing_summary: Dict, model_name: str, results: List[Dict], output_dir: str = None) -> Dict:
    """
    Update summary report with new model results.
    
    Args:
        existing_summary: Existing summary report dictionary
        model_name: Model name being evaluated
        results: New evaluation results
        output_dir: Output directory to load all datasets for this model (optional)
        
    Returns:
        Updated summary dictionary
    """
    # If output_dir is provided, load all datasets for this model from JSON file
    if output_dir:
        model_json_file = Path(output_dir) / f"{model_name}_detailed_results.json"
        if model_json_file.exists():
            # Load all datasets for this model
            all_model_data = load_model_results_json(str(model_json_file))
            # Flatten all datasets into a single results list
            all_results = []
            for dataset, dataset_results in all_model_data.items():
                all_results.extend(dataset_results)
            # Calculate statistics from all datasets
            model_stats = calculate_model_statistics(all_results)
        else:
            # Fallback to current results only
            model_stats = calculate_model_statistics(results)
    else:
        # Calculate statistics for current results only
        model_stats = calculate_model_statistics(results)
    
    # Update by_model
    existing_summary["by_model"][model_name] = model_stats
    
    # Recalculate overall statistics from all models
    total_samples = 0
    total_failures = 0
    total_successes = 0
    
    by_behavior_category_agg = {}
    
    # Aggregate from all models in by_model
    for model, model_stats in existing_summary["by_model"].items():
        total_samples += model_stats["total_samples"]
        total_failures += model_stats["failure_count"]
        total_successes += model_stats["success_count"]
        
        # Aggregate by behavior_category
        for category, cat_stats in model_stats.get("by_behavior_category", {}).items():
            if category not in by_behavior_category_agg:
                by_behavior_category_agg[category] = {
                    "total": 0,
                    "failures": 0,
                    "successes": 0
                }
            by_behavior_category_agg[category]["total"] += cat_stats["total"]
            by_behavior_category_agg[category]["failures"] += cat_stats["failures"]
            by_behavior_category_agg[category]["successes"] += cat_stats["successes"]
    
    # Calculate overall failure rate
    existing_summary["overall_failure_rate"] = format_failure_rate(total_failures, total_samples)
    existing_summary["total_samples"] = total_samples
    existing_summary["failure_count"] = total_failures
    existing_summary["success_count"] = total_successes
    
    # Calculate failure rate for each behavior category
    for category, stats in by_behavior_category_agg.items():
        stats["failure_rate"] = format_failure_rate(stats["failures"], stats["total"])
    
    existing_summary["by_behavior_category"] = by_behavior_category_agg
    
    return existing_summary


def generate_excel_report(output_dir: str, model_name: str, results: List[Dict], context_mode: str):
    """
    Generate or update Excel report with model results.
    
    Args:
        output_dir: Output directory path
        model_name: Model name (will be sheet name)
        results: Evaluation results for this model
        context_mode: Context mode ("full_context" or "current_turn")
    """

    # create output directory if not exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    excel_path = Path(output_dir) / "detailed_results.xlsx"
    
    # Prepare results: remove judge_reasoning, add conversation_segment if full_context
    prepared_results = []
    for result in results:
        # Create a copy without judge_reasoning
        prepared_result = {k: v for k, v in result.items() if k != "judge_reasoning"}
        
        # Add conversation_segment if context_mode is full_context
        if context_mode == "full_context" and "conversation_segment" in result:
            prepared_result["conversation_segment"] = result["conversation_segment"]
        
        prepared_results.append(prepared_result)
    
    # Convert to DataFrame
    df = pd.DataFrame(prepared_results)
    
    # Reorder columns for better readability
    column_order = [
        "dialog_id", "behavior_category", "evaluation_result", "judge_response_raw",
        "model", "dataset", "turn_index", "response", "patient_behavior_text",
        "judge_model", "context_mode"
    ]
    
    # Add conversation_segment if present
    if "conversation_segment" in df.columns:
        column_order.insert(column_order.index("response") + 1, "conversation_segment")
    
    # Reorder columns (only include columns that exist)
    existing_columns = [col for col in column_order if col in df.columns]
    df = df[existing_columns]
    
    # Load existing Excel or create new one
    if excel_path.exists():
        # Load existing workbook and update/append sheet
        # Use mode='a' to append to existing file, if_sheet_exists='replace' to overwrite existing sheet
        with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name=model_name, index=False)
    else:
        # Create new workbook
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=model_name, index=False)


def generate_excel_from_json_files(output_dir: str, context_mode: str = None):
    """
    Generate Excel report from all existing JSON result files.
    This function reads all {model}_detailed_results.json files and generates a unified Excel.
    
    Args:
        output_dir: Output directory path containing JSON result files
        context_mode: Context mode to determine if conversation_segment should be included.
                     If None, will try to infer from JSON files.
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"Error: Output directory does not exist: {output_dir}")
        return
    
    # Find all model result JSON files
    json_files = list(output_path.glob("*_detailed_results.json"))
    if not json_files:
        print(f"No model result JSON files found in {output_dir}")
        return
    
    excel_path = output_path / "detailed_results.xlsx"
    
    # Collect all model data
    all_model_data = {}
    for json_file in json_files:
        # Extract model name from filename (e.g., "gpt-4_detailed_results.json" -> "gpt-4")
        model_name = json_file.stem.replace("_detailed_results", "")
        
        # Load JSON data
        data = load_model_results_json(str(json_file))
        
        # Flatten data: convert {dataset: [results]} to single list of results
        all_results = []
        for dataset, results in data.items():
            all_results.extend(results)
        
        if all_results:
            all_model_data[model_name] = all_results
    
    if not all_model_data:
        print("No data found in JSON files")
        return
    
    # Determine context_mode from first result if not provided
    if context_mode is None and all_model_data:
        first_model = list(all_model_data.keys())[0]
        first_result = all_model_data[first_model][0]
        context_mode = first_result.get("context_mode", "current_turn")
    
    # Generate Excel with all models
    # Delete existing Excel file to start fresh
    if excel_path.exists():
        excel_path.unlink()
    
    # Write all models to Excel
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for model_name, results in all_model_data.items():
            # Prepare results: remove judge_reasoning, add conversation_segment if full_context
            prepared_results = []
            for result in results:
                # Create a copy without judge_reasoning
                prepared_result = {k: v for k, v in result.items() if k != "judge_reasoning"}
                
                # Add conversation_segment if context_mode is full_context
                if context_mode == "full_context" and "conversation_segment" in result:
                    prepared_result["conversation_segment"] = result["conversation_segment"]
                
                prepared_results.append(prepared_result)
            
            # Convert to DataFrame
            df = pd.DataFrame(prepared_results)
            
            # Reorder columns for better readability
            column_order = [
                "dialog_id", "behavior_category", "evaluation_result", "judge_response_raw",
                "model", "dataset", "turn_index", "response", "patient_behavior_text",
                "judge_model", "context_mode"
            ]
            
            # Add conversation_segment if present
            if "conversation_segment" in df.columns:
                column_order.insert(column_order.index("response") + 1, "conversation_segment")
            
            # Reorder columns (only include columns that exist)
            existing_columns = [col for col in column_order if col in df.columns]
            df = df[existing_columns]
            
            # Write to Excel sheet
            df.to_excel(writer, sheet_name=model_name, index=False)
    
    print(f"Generated Excel report with {len(all_model_data)} model(s): {excel_path}")

