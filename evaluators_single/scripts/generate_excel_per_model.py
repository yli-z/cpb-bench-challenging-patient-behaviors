#!/usr/bin/env python3
"""
Generate an Excel file for each model, with each behavior_category as a sheet
For manual inspection
"""

import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evaluators_single.scripts.utils import load_model_results_json


def load_single_model_results(json_file_path):
    """Load a single model's JSON result file"""
    json_path = Path(json_file_path)
    if not json_path.exists():
        print(f"❌ Error: JSON file does not exist: {json_file_path}")
        return None, None
    
    # Extract model name
    model_name = json_path.stem.replace("_detailed_results", "")
    print(f"  Loading model: {model_name}")
    
    # Load JSON data
    data = load_model_results_json(str(json_path))
    
    # Flatten data: convert {dataset: [results]} to a single result list
    all_results = []
    for dataset, results in data.items():
        for result in results:
            # Add model and dataset fields (if not exists)
            result["model"] = model_name
            result["dataset"] = dataset
            all_results.append(result)
    
    return model_name, all_results


def group_by_behavior_category(results):
    """Group results by behavior_category"""
    grouped = defaultdict(list)
    
    for result in results:
        behavior_category = result.get("behavior_category", "Unknown")
        grouped[behavior_category].append(result)
    
    return grouped


def prepare_row(result):
    """Prepare a row of data, in the specified column order"""
    row = {
        "model": result.get("model", ""),
        "dataset": result.get("dataset", ""),
        "dialog_id": result.get("dialog_id", ""),
        "turn_index": result.get("turn_index", 0),
        "evaluation_result": result.get("evaluation_result", False),
        "patient_behavior_text": result.get("patient_behavior_text", ""),
        "response": result.get("response", ""),
        "conversation_segment": result.get("conversation_segment", ""),
        "human check": ""  # Leave empty for manual inspection
    }
    return row


def generate_excel_for_model(json_file_path, output_dir):
    """Generate an Excel file for each model, with each behavior_category as a sheet"""
    json_path = Path(json_file_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Generate an Excel file for each model, with each behavior_category as a sheet")
    print("="*60)
    print(f"JSON file: {json_path}")
    print(f"Output directory: {output_dir}")
    print("="*60)
    print()
    
    # Load model data
    model_name, results = load_single_model_results(json_file_path)
    if not results:
        print("❌ Failed to load data")
        return None
    
    print(f"  Loaded {len(results)} results")
    
    # Group by behavior_category
    grouped_by_category = group_by_behavior_category(results)
    
    print(f"\nFound {len(grouped_by_category)} behavior_categories:")
    for category, cat_results in sorted(grouped_by_category.items()):
        print(f"  {category}: {len(cat_results)} results")
    
    # Define column order
    column_order = [
        "model",
        "dataset",
        "dialog_id",
        "turn_index",
        "evaluation_result",
        "patient_behavior_text",
        "response",
        "conversation_segment",
        "human check"
    ]
    
    # Generate Excel file name
    excel_filename = f"{model_name}_by_category.xlsx"
    excel_path = output_path / excel_filename
    
    # Generate Excel file
    print(f"\nGenerating Excel file: {excel_filename}")
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for category, cat_results in sorted(grouped_by_category.items()):
            # Prepare data
            rows = [prepare_row(result) for result in cat_results]
            df = pd.DataFrame(rows)
            
            # Ensure column order is correct
            df = df[column_order]
            
            # Sheet name (Excel limit is 31 characters)
            sheet_name = category[:31] if len(category) > 31 else category
            
            # Write to Excel
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            print(f"  ✅ {category}: {len(cat_results)} results → Sheet: {sheet_name}")
    
    print(f"\n{'='*60}")
    print(f"✅ Excel report generated: {excel_path}")
    print(f"{'='*60}")
    print(f"\nStatistics:")
    print(f"  - Model: {model_name}")
    print(f"  - Behavior Category count: {len(grouped_by_category)}")
    total_rows = sum(len(cat_results) for cat_results in grouped_by_category.values())
    print(f"  - Total rows: {total_rows}")
    print(f"  - Excel file size: {excel_path.stat().st_size / 1024 / 1024:.2f} MB")
    print()
    
    return excel_path


def generate_excel_for_all_models(input_dir, output_dir):
    """Generate Excel files for all models in the input directory"""
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ Error: Input directory does not exist: {input_dir}")
        return []
    
    # Find all JSON result files
    json_files = list(input_path.glob("*_detailed_results.json"))
    if not json_files:
        print(f"⚠️ Warning: No JSON result files found in {input_dir}")
        return []
    
    print("="*60)
    print("Generate Excel files for all models in the input directory")
    print("="*60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(json_files)} JSON result files")
    print("="*60)
    print()
    
    generated_files = []
    for json_file in sorted(json_files):
        try:
            excel_path = generate_excel_for_model(str(json_file), output_dir)
            if excel_path:
                generated_files.append(excel_path)
        except Exception as e:
            print(f"❌ Error processing {json_file.name}: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("="*60)
    print(f"✅ Batch generation completed! Generated {len(generated_files)} Excel files")
    print("="*60)
    for excel_file in generated_files:
        print(f"  - {excel_file.name}")
    print()
    
    return generated_files


def main():
    parser = argparse.ArgumentParser(
        description="Generate an Excel file for each model, with each behavior_category as a sheet"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory (contains JSON result files)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default is the same as the input directory)"
    )
    parser.add_argument(
        "--json_file",
        type=str,
        default=None,
        help="Specify a single JSON file (optional, if not specified, process all files)"
    )
    
    args = parser.parse_args()
    
    # Process paths
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        # Try to resolve from project root
        project_root = Path(__file__).parent.parent.parent
        potential_path = project_root / input_dir
        if potential_path.exists() or str(input_dir).startswith("evaluators_single/"):
            input_dir = potential_path.resolve()
        else:
            # Otherwise relative to current directory
            input_dir = input_dir.resolve()
    
    # Output directory default is the same as the input directory
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    if not output_dir.is_absolute():
        project_root = Path(__file__).parent.parent.parent
        potential_path = project_root / output_dir
        if potential_path.exists() or str(output_dir).startswith("evaluators_single/"):
            output_dir = potential_path.resolve()
        else:
            output_dir = output_dir.resolve()
    
    # Generate Excel
    try:
        if args.json_file:
            # Process single file
            json_file_path = Path(args.json_file)
            if not json_file_path.is_absolute():
                json_file_path = input_dir / json_file_path
            generate_excel_for_model(str(json_file_path), str(output_dir))
        else:
            # Process all files
            generate_excel_for_all_models(str(input_dir), str(output_dir))
    except Exception as e:
        print(f"\n❌ Error generating Excel: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
