"""
Data Preparation Script

Prepares failed cases data for multi-turn continuation testing.

Usage:
    python scripts/prepare_data.py \\
        --excel data/Finalized_detailed_results_by_category.xlsx \\
        --source_data data_loader/output_benchmark \\
        --output data_processing/output/failed_cases_multiturn.json
"""

import argparse
import json
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.excel_processor import extract_failed_cases_from_excel
from data_processing.conversation_builder import enrich_with_complete_conversations


def main():
    parser = argparse.ArgumentParser(description="Prepare failed cases data for multi-turn continuation")
    parser.add_argument(
        "--excel",
        type=str,
        required=True,
        help="Path to Finalized_detailed_results_by_category.xlsx"
    )
    parser.add_argument(
        "--source_data",
        type=str,
        required=True,
        help="Directory containing source data (*_safety_benchmark.json files)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data_processing/output/failed_cases_multiturn.json",
        help="Output JSON file path (default: data_processing/output/failed_cases_multiturn.json)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 MULTI-TURN CONTINUATION - DATA PREPARATION")
    print("=" * 80)
    
    # Step 1: Extract failed cases from Excel
    print("\n📋 Step 1: Extracting failed cases from Excel...")
    print(f"   Excel file: {args.excel}")
    
    if not os.path.exists(args.excel):
        print(f"❌ Error: Excel file not found: {args.excel}")
        sys.exit(1)
    
    failed_cases = extract_failed_cases_from_excel(args.excel)
    
    if not failed_cases:
        print("❌ Error: No failed cases extracted!")
        sys.exit(1)
    
    # Step 2: Enrich with complete conversations
    print("\n📋 Step 2: Enriching with complete conversations...")
    print(f"   Source data directory: {args.source_data}")
    
    if not os.path.exists(args.source_data):
        print(f"❌ Error: Source data directory not found: {args.source_data}")
        sys.exit(1)
    
    enriched_cases = enrich_with_complete_conversations(failed_cases, args.source_data)
    
    if not enriched_cases:
        print("❌ Error: No cases enriched successfully!")
        sys.exit(1)
    
    # Step 3: Prepare metadata
    print("\n📋 Step 3: Preparing metadata...")
    
    metadata = {
        'total_failed_cases': len(enriched_cases),
        'by_category': {},
        'by_model': {},
        'by_dataset': {}
    }

    for case in enriched_cases:
        category = case['behavior_category']
        model = case['model']
        dataset = case['dataset']

        metadata['by_category'][category] = metadata['by_category'].get(category, 0) + 1
        metadata['by_model'][model] = metadata['by_model'].get(model, 0) + 1
        metadata['by_dataset'][dataset] = metadata['by_dataset'].get(dataset, 0) + 1
    
    # Step 4: Save output
    print("\n📋 Step 4: Saving output...")
    
    output_data = {
        'metadata': metadata,
        'failed_cases': enriched_cases
    }
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"   Created directory: {output_dir}")
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Output saved to: {args.output}")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print(f"Total Cases: {metadata['total_failed_cases']}")
    print(f"\nBy Category:")
    for cat, count in sorted(metadata['by_category'].items()):
        print(f"   {cat}: {count}")
    print(f"\nBy Dataset:")
    for ds, count in sorted(metadata['by_dataset'].items()):
        print(f"   {ds}: {count}")

    print(f"\nBy Model: {len(metadata['by_model'])} different models")
    
    print("\n✅ Data preparation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
