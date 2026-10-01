"""
Main script to extract and process annotated medical dialog data for safety benchmarking.

Usage:
    python get_data.py --dataset ACI
    python get_data.py --dataset MediTOD
    python get_data.py --dataset all
"""

import json
import argparse
from typing import Dict, List
from collections import defaultdict
import os

import config
import utils


def process_dataset(dataset_name: str) -> Dict:
    """
    Process a single dataset and generate safety benchmark data.
    
    Args:
        dataset_name: Name of the dataset to process (e.g., "ACI", "MediTOD")
        
    Returns:
        Dictionary containing processed benchmark data
    """
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name} Dataset")
    print(f"{'='*60}\n")
    
    # Get dataset configuration
    if dataset_name not in config.DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(config.DATASETS.keys())}")
    
    dataset_config = config.DATASETS[dataset_name]
    format_type = dataset_config["format"]
    
    # Step 1: Load source dialogs
    print("Step 1: Loading source dialogs...")
    dialogs_dict = utils.load_source_dialogs(dataset_name, dataset_config)
    
    if not dialogs_dict:
        print(f"Error: No dialogs loaded for {dataset_name}")
        return None
    
    # Step 2: Load Excel annotations from directory
    print(f"\nStep 2: Loading Excel annotations from {dataset_config['excel_dir']}...")
    annotations_df = utils.load_excel_annotations(
        dataset_config["excel_dir"],
        config.BEHAVIOR_EXCEL_FILES,
        config.FILE_TO_CATEGORY
    )
    
    if annotations_df.empty:
        print(f"Error: No annotations loaded for {dataset_name}")
        return None
    
    # Step 3: Filter positive annotations (Human check = "Y")
    print(f"\nStep 3: Filtering positive annotations (Human check = 'Y')...")
    positive_df = utils.filter_positive_annotations(
        annotations_df,
        config.HUMAN_CHECK_COLUMN,
        config.POSITIVE_ANNOTATION
    )
    
    if positive_df.empty:
        print(f"Warning: No positive annotations found for {dataset_name}")
        return None
    
    # Step 4: Process each positive annotation
    print("\nStep 4: Processing annotations and extracting conversation segments...")
    cases = []
    behavior_count = defaultdict(int)
    skipped_count = 0
    
    for idx, row in positive_df.iterrows():
        try:
            dialog_id = str(row['dialog_id'])
            turn_index = int(row['turn_index'])
            behavior_category = row['behavior_category']  # Already mapped in load_excel_annotations
            
            # Find dialog in source data
            if dialog_id not in dialogs_dict:
                print(f"Warning: Dialog {dialog_id} not found in source data")
                skipped_count += 1
                continue
            
            dialog_data = dialogs_dict[dialog_id]
            
            # Extract conversation segment
            conversation_segment = utils.extract_conversation_segment(
                dialog_data, 
                turn_index, 
                dataset_config,
                format_type
            )
            
            if not conversation_segment:
                print(f"Warning: Empty conversation segment for {dialog_id} at turn {turn_index}")
                skipped_count += 1
                continue
            
            # Extract complete conversation
            complete_conversation = utils.extract_complete_conversation(
                dialog_data,
                dataset_config,
                format_type
            )
            
            # Extract patient behavior text
            patient_behavior_text = utils.get_patient_behavior_text(
                dialog_data,
                turn_index,
                format_type
            )
            
            # Build case entry (without metadata section to avoid disturbing LLM)
            case = {
                "case_id": utils.generate_case_id(dataset_name, len(cases) + 1),
                "dialog_id": dialog_id,
                "turn_index": turn_index,
                "behavior_category": behavior_category,
                "patient_behavior_text": patient_behavior_text,
                "conversation_segment": conversation_segment,
                "complete_conversation": complete_conversation
            }
            
            cases.append(case)
            behavior_count[behavior_category] += 1
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            skipped_count += 1
            continue
    
    # Step 5: Build output structure
    print(f"\nStep 5: Building output structure...")
    print(f"Total cases processed: {len(cases)}")
    print(f"Skipped cases: {skipped_count}")
    print(f"\nBehavior category distribution:")
    for category, count in sorted(behavior_count.items()):
        print(f"  - {category}: {count}")
    
    output_data = {
        "dataset_name": dataset_name,
        "dataset_source": dataset_config["dataset_source"],
        "total_cases": len(cases),
        "behavior_categories": dict(behavior_count),
        "cases": cases
    }
    
    return output_data


def save_output(data: Dict, dataset_name: str):
    """
    Save processed data to JSON file.
    
    Args:
        data: Processed benchmark data
        dataset_name: Name of the dataset
    """
    output_file = os.path.join(config.OUTPUT_DIR, f"{dataset_name}_safety_benchmark.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved output to: {output_file}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Process annotated medical dialog data for safety benchmarking"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(config.DATASETS.keys()) + ["all"],
        help="Dataset to process (or 'all' for all datasets)"
    )
    
    args = parser.parse_args()
    
    # Import pandas here (only needed when running)
    global pd
    import pandas as pd
    
    print("🚀 Medical Dialog Safety Benchmark Data Processing")
    print(f"Output directory: {config.OUTPUT_DIR}\n")
    
    # Process datasets
    if args.dataset == "all":
        datasets_to_process = list(config.DATASETS.keys())
    else:
        datasets_to_process = [args.dataset]
    
    for dataset_name in datasets_to_process:
        try:
            output_data = process_dataset(dataset_name)
            
            if output_data:
                save_output(output_data, dataset_name)
            else:
                print(f"\n⚠️  No data generated for {dataset_name}")
                
        except Exception as e:
            print(f"\n❌ Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("✅ Processing Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

