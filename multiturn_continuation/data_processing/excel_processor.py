"""
Excel Processor

Extracts failed cases from the annotated Excel file.
"""

import pandas as pd
from typing import List, Dict
import os


def extract_abnormal_value_cases_from_excel(
    excel_path: str,
    sheet_name: str = "true_failure_abnormal_values_co"
) -> List[Dict]:
    """
    Extract abnormal-value failed cases from a dedicated sheet.

    Notes:
    - Uses ALL rows in the target sheet (no human-finalized filtering).
    - Normalizes output fields to match existing failed-case pipeline schema.

    Args:
        excel_path: Path to Finalized_Failure_case_abnormal.xlsx
        sheet_name: Sheet name containing finalized abnormal cases

    Returns:
        List of normalized case dictionaries
    """
    print(f"📂 Loading abnormal-value Excel file: {excel_path}")
    print(f"   Sheet: {sheet_name}")

    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    print(f"   Total rows (all included): {len(df)}")

    all_cases = []
    skipped = 0

    for _, row in df.iterrows():
        try:
            case = {
                "model": str(row["model"]).strip(),
                # Keep a stable dataset tag for downstream grouping and case_id construction.
                "dataset": "abnormal_value",
                "dialog_id": str(row["dialog_id"]).strip(),
                # Fixed failed doctor turn for this abnormal-value setup.
                "turn_index": 25,
                "behavior_category": "Abnormal Clinical Values",
                "patient_behavior_text": str(row["patient_behavior_text"]).strip(),
                "doctor_failed_response": str(row["Doctor LLM response"]).strip(),
                "conversation_segment": str(row["conversation_segment"]).strip(),
            }
            all_cases.append(case)
        except Exception as e:
            skipped += 1
            print(f"   ⚠️  Warning: Skipping row due to error: {e}")
            continue

    print(f"\n✅ Total abnormal-value cases extracted: {len(all_cases)}")
    if skipped:
        print(f"   ⚠️  Skipped rows: {skipped}")

    # Quick model stats for traceability
    by_model = {}
    for case in all_cases:
        model = case["model"]
        by_model[model] = by_model.get(model, 0) + 1

    print(f"   By Model: {len(by_model)} different models")
    return all_cases


def extract_failed_cases_from_excel(excel_path: str) -> List[Dict]:
    """
    Extract all failed cases from the Excel file.
    
    Args:
        excel_path: Path to the Finalized_detailed_results_by_category.xlsx file
        
    Returns:
        List of failed case dictionaries with basic information
    """
    # Define the 4 category sheets
    sheets = [
        'Information Contradiction',
        'Factual Inaccuracy',
        'Self-diagnosis',
        'Care Resistance'
    ]
    
    all_cases = []
    
    print(f"📂 Loading Excel file: {excel_path}")
    
    for sheet_name in sheets:
        try:
            # Read the sheet
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            
            # Keep only rows where human check-finalized == 'Y' (confirmed failed cases)
            if 'human check-finalized' in df.columns:
                before_count = len(df)
                df = df[df['human check-finalized'].str.upper() == 'Y'].copy()
                after_count = len(df)
                print(f"   Sheet '{sheet_name}': {before_count} total → {after_count} confirmed failed cases")
            else:
                print(f"   ⚠️  Sheet '{sheet_name}': No 'human check-finalized' column, processing all {len(df)} rows")
            
            # Extract information for each case
            for _, row in df.iterrows():
                try:
                    case = {
                        'model': str(row['model']).strip(),
                        'dataset': str(row['dataset']).strip(),
                        'dialog_id': str(row['dialog_id']).strip(),
                        'turn_index': int(row['turn_index']),
                        'behavior_category': sheet_name,
                        'patient_behavior_text': str(row['patient_behavior_text']).strip(),
                        'doctor_failed_response': str(row['response']).strip(),
                        'conversation_segment': str(row['conversation_segment']).strip()
                    }
                    all_cases.append(case)
                    
                except Exception as e:
                    print(f"      ⚠️  Warning: Skipping row due to error: {e}")
                    continue
            
        except Exception as e:
            print(f"   ❌ Error reading sheet '{sheet_name}': {e}")
            continue
    
    print(f"\n✅ Total failed cases extracted: {len(all_cases)}")
    
    # Print statistics
    by_category = {}
    by_model = {}
    by_dataset = {}
    
    for case in all_cases:
        category = case['behavior_category']
        model = case['model']
        dataset = case['dataset']
        
        by_category[category] = by_category.get(category, 0) + 1
        by_model[model] = by_model.get(model, 0) + 1
        by_dataset[dataset] = by_dataset.get(dataset, 0) + 1
    
    print("\n📊 Statistics:")
    print(f"   By Category: {dict(sorted(by_category.items()))}")
    print(f"   By Dataset: {dict(sorted(by_dataset.items()))}")
    print(f"   By Model: {len(by_model)} different models")
    
    return all_cases


def extract_cases_by_human_check3(excel_path: str) -> List[Dict]:
    """
    Extract positive failed cases from a per-model by_category Excel file,
    using ``human check3 == 'Y'`` as the acceptance criterion.

    This mirrors :func:`extract_failed_cases_from_excel` but is tailored to
    the annotation workflow where the final reviewer column is *human check3*
    (rather than *human check-finalized*).  It also handles the Care Resistance
    sheet whose model column is named ``Unnamed: 0`` instead of ``model``.

    Args:
        excel_path: Path to ``<model>_by_category.xlsx``.

    Returns:
        List of case dictionaries with the same schema as
        :func:`extract_failed_cases_from_excel`.
    """
    sheets = [
        'Information Contradiction',
        'Factual Inaccuracy',
        'Self-diagnosis',
        'Care Resistance',
    ]

    all_cases: List[Dict] = []

    print(f"📂 Loading Excel file: {excel_path}")

    for sheet_name in sheets:
        try:
            df = pd.read_excel(excel_path, sheet_name=sheet_name)

            # Care Resistance sheet stores model name under 'Unnamed: 0'
            if 'model' not in df.columns and 'Unnamed: 0' in df.columns:
                df = df.rename(columns={'Unnamed: 0': 'model'})

            check_col = 'human check3'
            if check_col not in df.columns:
                print(f"   ⚠️  Sheet '{sheet_name}': No '{check_col}' column, skipping")
                continue

            before_count = len(df)
            # Accept exactly 'Y' (case-insensitive), treat everything else as N
            df = df[df[check_col].astype(str).str.strip().str.upper() == 'Y'].copy()
            after_count = len(df)
            print(f"   Sheet '{sheet_name}': {before_count} total → {after_count} confirmed (human check3=Y)")

            for _, row in df.iterrows():
                try:
                    case = {
                        'model': str(row['model']).strip(),
                        'dataset': str(row['dataset']).strip(),
                        'dialog_id': str(row['dialog_id']).strip(),
                        'turn_index': int(row['turn_index']),
                        'behavior_category': sheet_name,
                        'patient_behavior_text': str(row['patient_behavior_text']).strip(),
                        'doctor_failed_response': str(row['response']).strip(),
                        'conversation_segment': str(row['conversation_segment']).strip(),
                    }
                    all_cases.append(case)
                except Exception as e:
                    print(f"      ⚠️  Warning: Skipping row due to error: {e}")
                    continue

        except Exception as e:
            print(f"   ❌ Error reading sheet '{sheet_name}': {e}")
            continue

    print(f"\n✅ Total cases extracted (human check3=Y): {len(all_cases)}")

    # Statistics
    by_category: Dict[str, int] = {}
    by_model: Dict[str, int] = {}
    by_dataset: Dict[str, int] = {}

    for case in all_cases:
        cat = case['behavior_category']
        mdl = case['model']
        ds  = case['dataset']
        by_category[cat] = by_category.get(cat, 0) + 1
        by_model[mdl]    = by_model.get(mdl, 0) + 1
        by_dataset[ds]   = by_dataset.get(ds, 0) + 1

    print("\n📊 Statistics:")
    print(f"   By Category: {dict(sorted(by_category.items()))}")
    print(f"   By Dataset:  {dict(sorted(by_dataset.items()))}")
    print(f"   By Model:    {len(by_model)} different model(s) — {list(by_model.keys())}")

    return all_cases


if __name__ == "__main__":
    # Test the processor
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python excel_processor.py <path_to_excel>")
        sys.exit(1)
    
    excel_path = sys.argv[1]
    cases = extract_failed_cases_from_excel(excel_path)
    
    print(f"\n📝 Sample case:")
    if cases:
        sample = cases[0]
        for key, value in sample.items():
            if key == 'conversation_segment':
                print(f"   {key}: {value[:100]}...")
            elif key == 'doctor_failed_response':
                print(f"   {key}: {value[:100]}...")
            else:
                print(f"   {key}: {value}")
