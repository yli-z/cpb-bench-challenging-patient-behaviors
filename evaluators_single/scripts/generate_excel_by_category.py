#!/usr/bin/env python3
"""
按 behavior_category 分 sheet 生成 Excel 报告
每个 behavior_category 一个工作表，包含所有模型的数据
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


def load_all_model_results(output_dir):
    """Load all model results from the output directory"""
    output_path = Path(output_dir)
    json_files = list(output_path.glob("*_detailed_results.json"))
    
    all_model_data = {}
    for json_file in sorted(json_files):
        model_name = json_file.stem.replace("_detailed_results", "")
        data = load_model_results_json(str(json_file))
        all_results = []
        for dataset, results in data.items():
            for result in results:
                result["model"] = model_name
                result["dataset"] = dataset
                all_results.append(result)
        
        if all_results:
            all_model_data[model_name] = all_results
    
    return all_model_data


def group_by_behavior_category(all_model_data):
    """Group all results by behavior_category"""
    grouped = defaultdict(list)
    
    for model_name, results in all_model_data.items():
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


def generate_excel_by_category(output_dir, excel_filename="detailed_results_by_category.xlsx"):
    """Generate an Excel file with sheets by behavior_category"""
    output_path = Path(output_dir)
    excel_path = output_path / excel_filename
    
    print("="*60)
    print("Generate an Excel file with sheets by behavior_category")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Excel file: {excel_path}")
    print("="*60)
    print()
    
    # Load all model data
    all_model_data = load_all_model_results(output_dir)
    if not all_model_data:
        return
    
    print(f"\nLoaded {len(all_model_data)} model data")
    
    # Group by behavior_category
    grouped_by_category = group_by_behavior_category(all_model_data)
    
    print(f"\nFound {len(grouped_by_category)} behavior_categories:")
    for category, results in sorted(grouped_by_category.items()):
        print(f"  {category}: {len(results)} results")
    
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
    
    # Generate Excel
    print(f"\nGenerating Excel file...")
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for category, results in sorted(grouped_by_category.items()):
            # Prepare data
            rows = [prepare_row(result) for result in results]
            df = pd.DataFrame(rows)
            
            # Ensure column order is correct
            df = df[column_order]
            
            # Sheet name (Excel limit is 31 characters)
            sheet_name = category[:31] if len(category) > 31 else category
            
            # Write to Excel
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            print(f"  ✅ {category}: {len(results)} results → Sheet: {sheet_name}")
    
    print(f"\n{'='*60}")
    print(f"✅ Excel report generated: {excel_path}")
    print(f"{'='*60}")
    print(f"\nStatistics:")
    print(f"  - Model count: {len(all_model_data)}")
    print(f"  - Behavior Category count: {len(grouped_by_category)}")
    total_rows = sum(len(results) for results in grouped_by_category.values())
    print(f"  - Total rows: {total_rows}")
    print(f"  - Excel file size: {excel_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    return excel_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate an Excel file with sheets by behavior_category"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluators_single/output_single_turn",
        help="Output directory (contains JSON result files)"
    )
    parser.add_argument(
        "--excel_filename",
        type=str,
        default="detailed_results_by_category.xlsx",
        help="Output Excel file name"
    )
    
    args = parser.parse_args()
    
    # Process path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        # Try to resolve from project root
        project_root = Path(__file__).parent.parent.parent
        potential_path = project_root / output_dir
        if potential_path.exists() or str(output_dir).startswith("evaluators_single/"):
            output_dir = potential_path.resolve()
        else:
            # Otherwise relative to script directory
            output_dir = (Path(__file__).parent / output_dir).resolve()
    
    # Generate Excel
    try:
        generate_excel_by_category(str(output_dir), args.excel_filename)
    except Exception as e:
        print(f"\n❌ Error generating Excel: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

