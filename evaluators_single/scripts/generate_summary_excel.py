"""
Generate Excel report from summary_report.json
Extracts failure rates by model and behavior category.
"""

import json
import pandas as pd
from pathlib import Path


def generate_summary_excel(summary_json_path: str, output_excel_path: str = None):
    """
    Generate Excel file with failure rates by model and behavior category.
    
    Args:
        summary_json_path: Path to summary_report.json
        output_excel_path: Path to output Excel file (default: same dir as JSON with .xlsx extension)
    """
    # Load summary report
    with open(summary_json_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    # Extract all models and behavior categories
    models = list(summary['by_model'].keys())
    behavior_categories = list(summary['by_behavior_category'].keys())
    
    # Create a matrix: rows = models, columns = behavior categories
    data = []
    for model in models:
        row = {'Model': model}
        model_data = summary['by_model'][model]
        
        # Get failure rate for each behavior category
        for behavior in behavior_categories:
            if behavior in model_data.get('by_behavior_category', {}):
                failure_rate = model_data['by_behavior_category'][behavior]['failure_rate']
                # Remove % sign and convert to float for sorting/formatting
                row[behavior] = failure_rate
            else:
                row[behavior] = 'N/A'
        
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Set Model as index for better display
    df.set_index('Model', inplace=True)
    
    # Determine output path
    if output_excel_path is None:
        json_path = Path(summary_json_path)
        output_excel_path = json_path.parent / f"{json_path.stem}.xlsx"
    else:
        output_excel_path = Path(output_excel_path)
    
    # Ensure output directory exists
    output_excel_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to Excel
    with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
        sheet_name = 'Failure Rates'
        df.to_excel(writer, sheet_name=sheet_name, index=True)
        
        # Get the worksheet to format
        worksheet = writer.sheets[sheet_name]
        
        # Auto-adjust column widths
        for idx, col in enumerate(df.columns, start=2):  # Start from column B (index 2)
            max_length = max(
                df[col].astype(str).map(len).max(),
                len(col)
            )
            worksheet.column_dimensions[chr(64 + idx)].width = min(max_length + 2, 20)
        
        # Adjust Model column width
        max_model_length = max(df.index.astype(str).map(len).max(), len('Model'))
        worksheet.column_dimensions['A'].width = min(max_model_length + 2, 30)
    
    print(f"✅ Excel report generated: {output_excel_path}")
    print(f"\nSummary:")
    print(f"  - Models: {len(models)}")
    print(f"  - Behavior Categories: {len(behavior_categories)}")
    print(f"\nPreview:")
    print(df.to_string())
    
    return output_excel_path


if __name__ == "__main__":
    # Default paths
    script_dir = Path(__file__).parent
    summary_json_path = script_dir / "output" / "summary_report.json"
    output_excel_path = script_dir / "output" / "summary_report.xlsx"
    
    generate_summary_excel(str(summary_json_path), str(output_excel_path))

