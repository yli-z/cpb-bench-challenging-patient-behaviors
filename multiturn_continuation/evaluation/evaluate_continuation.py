#!/usr/bin/env python3
"""
Main evaluation script for multi-turn continuation assessment.

Evaluates whether doctors maintain or correct their initial failures
across multi-turn dialogue continuations.

Usage:
    python multiturn_continuation/evaluation/evaluate_continuation.py \
        --input multiturn_continuation/output/api_models_generated.json \
        --output_dir multiturn_continuation/output/evaluation_results \
        --judge_model_name gpt-4o \
        --judge_model_type openai
"""

import argparse
import asyncio
import json
import sys
import os
import random
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env
load_dotenv(project_root / '.env')

from models.openai_model import OpenAIModel
from models.claude_model import ClaudeModel
from models.gemini_model import GeminiModel
from models.base_model import BaseLLM
from multiturn_continuation.evaluation.continuation_evaluator import ContinuationEvaluator

class Logger(object):
    def __init__(self, output_dir: str):
        self.terminal = sys.stdout
        os.makedirs(output_dir, exist_ok=True)
        log_filename = f"eval_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.log_path = os.path.join(output_dir, log_filename)
        self.log = open(self.log_path, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

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


def load_continuation_data(input_file: str) -> Dict:
    """
    Load continuation output JSON or JSONL file.
    
    Args:
        input_file: Path to JSON or JSONL file
        
    Returns:
        Dictionary with 'results' key containing list of cases
    """
    if input_file.endswith('.jsonl'):
        results = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        return {'results': results}
    else:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data


def calculate_statistics(results: List[Dict]) -> Dict:
    """
    Calculate evaluation statistics.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Dictionary with statistics
    """
    total_cases = len(results)
    maintains_failure = sum(1 for r in results if r.get('maintains_failure', False))
    corrected_failure = sum(1 for r in results if r.get('corrected_failure', False))
    
    correction_rate = f"{(corrected_failure / total_cases * 100):.2f}%" if total_cases > 0 else "0.00%"
    maintenance_rate = f"{(maintains_failure / total_cases * 100):.2f}%" if total_cases > 0 else "0.00%"
    
    # By behavior category
    by_behavior_category = {}
    for result in results:
        category = result.get('behavior_category', 'Unknown')
        if category not in by_behavior_category:
            by_behavior_category[category] = {
                'total': 0,
                'maintains_failure': 0,
                'corrected_failure': 0
            }
        by_behavior_category[category]['total'] += 1
        if result.get('maintains_failure', False):
            by_behavior_category[category]['maintains_failure'] += 1
        if result.get('corrected_failure', False):
            by_behavior_category[category]['corrected_failure'] += 1
    
    # Calculate rates for each category
    for category, stats in by_behavior_category.items():
        total = stats['total']
        stats['correction_rate'] = f"{(stats['corrected_failure'] / total * 100):.2f}%" if total > 0 else "0.00%"
        stats['maintenance_rate'] = f"{(stats['maintains_failure'] / total * 100):.2f}%" if total > 0 else "0.00%"
    
    # By model
    by_model = {}
    for result in results:
        model = result.get('model', 'Unknown')
        if model not in by_model:
            by_model[model] = {
                'total': 0,
                'maintains_failure': 0,
                'corrected_failure': 0
            }
        by_model[model]['total'] += 1
        if result.get('maintains_failure', False):
            by_model[model]['maintains_failure'] += 1
        if result.get('corrected_failure', False):
            by_model[model]['corrected_failure'] += 1
    
    # Calculate rates for each model
    for model, stats in by_model.items():
        total = stats['total']
        stats['correction_rate'] = f"{(stats['corrected_failure'] / total * 100):.2f}%" if total > 0 else "0.00%"
        stats['maintenance_rate'] = f"{(stats['maintains_failure'] / total * 100):.2f}%" if total > 0 else "0.00%"
    
    return {
        'total_cases': total_cases,
        'maintains_failure': maintains_failure,
        'corrected_failure': corrected_failure,
        'correction_rate': correction_rate,
        'maintenance_rate': maintenance_rate,
        'by_behavior_category': by_behavior_category,
        'by_model': by_model
    }


def evaluate_all_cases(
    data: Dict,
    judge_model: BaseLLM,
    output_dir: str,
    sample: int = None
) -> Dict:
    """
    Evaluate all cases in the dataset.
    
    Args:
        data: Loaded continuation data
        judge_model: Judge model instance
        output_dir: Output directory for results
        sample: If specified, only evaluate first N cases
        
    Returns:
        Dictionary with results and statistics
    """
    cases = data.get('results', data.get('failed_cases', []))
    total_cases_count = len(cases)
    
    if sample:
        cases = random.sample(cases, min(sample, total_cases_count))
        print(f"Evaluating VERY RANDOM sample of {sample} cases (out of {total_cases_count} total)")
    else:
        print(f"Evaluating all {len(cases)} cases")
    
    # Initialize evaluator
    evaluator = ContinuationEvaluator(judge_model=judge_model)
    
    # Evaluate each case
    results = []
    print("\nEvaluating cases...")
    for case in tqdm(cases, desc="Evaluating"):
        try:
            if 'conversation_segment' in case:
                behavior_category = case.get('category', case.get('behavior_category', 'Unknown'))
                case_id = str(case.get('dialog_id', case.get('case_id', 'unknown')))
                
                convo_history = case.get('conversation_segment', '')
                failed_response = case.get('doctor_failed_response', case.get('response', ''))
                
                # If there's a separately generated continuation, use it. Otherwise, for testing, 
                # we might just simulate the continuation as empty or the same for dummy printing.
                continuation_content = case.get('continuation_response', case.get('multi_turn_response', []))
                
                if isinstance(continuation_content, list) and continuation_content:
                    # It's an array of turns
                    multi_turn_response = [{'source': 'llm_failed', 'role': 'Doctor', 'content': failed_response}] + \
                        [{'source': 'llm_generated', 'role': t.get('role', 'Unknown'), 'content': t.get('content', '')} for t in continuation_content]
                elif isinstance(continuation_content, str) and continuation_content:
                    multi_turn_response = [
                        {'source': 'llm_failed', 'role': 'Doctor', 'content': failed_response},
                        {'source': 'llm_generated', 'role': 'Doctor (Continuation)', 'content': continuation_content}
                    ]
                else:
                    # If there's no continuation generated yet, just to make sure the script runs and prints 
                    # correctly aligned fields for user verification:
                    multi_turn_response = [
                        {'source': 'llm_failed', 'role': 'Doctor', 'content': failed_response},
                        {'source': 'llm_generated', 'role': 'Doctor (Continuation)', 'content': failed_response}
                    ]
                
                result = evaluator.evaluate_continuation(
                    case_id=case_id,
                    behavior_category=behavior_category,
                    conversation_history=convo_history,
                    multi_turn_response=multi_turn_response,
                    model=case.get('model', 'unknown')
                )
            else:
                behavior_category = case.get('behavior_category', 'Unknown')
                case_id = case.get('case_id', 'unknown')
                
                convo_history_list = case.get('conversation_history', [])
                if isinstance(convo_history_list, list) and len(convo_history_list) > 0:
                    convo_history_str = "\n".join([f"{t.get('role', 'Unknown')}: {t.get('content', '')}" for t in convo_history_list if t.get('source') == 'original'])
                    multi_turn_response = [t for t in convo_history_list if t.get('source') in ['llm_failed', 'llm_generated']]
                else:
                    convo_history_str = ""
                    multi_turn_response = case.get('multi_turn_response', [])
                
                result = evaluator.evaluate_continuation(
                    case_id=case_id,
                    behavior_category=behavior_category,
                    conversation_history=convo_history_str,
                    multi_turn_response=multi_turn_response,
                    model=case.get('model', 'unknown')
                )
            results.append(result)
        except Exception as e:
            print(f"\n❌ Error evaluating case {case.get('case_id')}: {e}")
            # Add error result
            results.append({
                'case_id': case.get('case_id', 'unknown'),
                'behavior_category': case.get('behavior_category', 'Unknown'),
                'model': case.get('model', 'unknown'),
                'error': str(e)
            })
    
    # Calculate statistics
    statistics = calculate_statistics(results)
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Total Cases: {statistics['total_cases']}")
    print(f"Maintains Failure: {statistics['maintains_failure']} ({statistics['maintenance_rate']})")
    print(f"Corrected Failure: {statistics['corrected_failure']} ({statistics['correction_rate']})")
    print("\nBy Behavior Category:")
    for category, stats in statistics['by_behavior_category'].items():
        print(f"  {category}:")
        print(f"    Total: {stats['total']}")
        print(f"    Maintains: {stats['maintains_failure']} ({stats['maintenance_rate']})")
        print(f"    Corrected: {stats['corrected_failure']} ({stats['correction_rate']})")
    print("\nBy Model:")
    for model, stats in statistics['by_model'].items():
        print(f"  {model}:")
        print(f"    Total: {stats['total']}")
        print(f"    Maintains: {stats['maintains_failure']} ({stats['maintenance_rate']})")
        print(f"    Corrected: {stats['corrected_failure']} ({stats['correction_rate']})")
    print("="*80)
    
    return {
        'results': results,
        'statistics': statistics
    }


async def evaluate_all_cases_async(
    data: Dict,
    judge_model: BaseLLM,
    output_dir: str,
    concurrency: int = 5,
    sample: int = None
) -> Dict:
    """
    Evaluate all cases in parallel using async concurrency control.

    Args:
        data: Loaded continuation data
        judge_model: Judge model instance
        output_dir: Output directory for results
        concurrency: Max concurrent evaluation calls
        sample: If specified, only evaluate first N cases

    Returns:
        Dictionary with results and statistics
    """
    cases = data.get('results', data.get('failed_cases', []))
    total_cases_count = len(cases)

    if sample:
        cases = random.sample(cases, min(sample, total_cases_count))
        print(f"Evaluating VERY RANDOM sample of {sample} cases (out of {total_cases_count} total)")
    else:
        print(f"Evaluating all {len(cases)} cases")

    evaluator = ContinuationEvaluator(judge_model=judge_model)
    semaphore = asyncio.Semaphore(concurrency)

    async def evaluate_one(case: Dict) -> Dict:
        async with semaphore:
            try:
                if 'conversation_segment' in case:
                    behavior_category = case.get('category', case.get('behavior_category', 'Unknown'))
                    case_id = str(case.get('dialog_id', case.get('case_id', 'unknown')))
                    convo_history = case.get('conversation_segment', '')
                    failed_response = case.get('doctor_failed_response', case.get('response', ''))
                    continuation_content = case.get('continuation_response', case.get('multi_turn_response', []))

                    if isinstance(continuation_content, list) and continuation_content:
                        multi_turn_response = [{'source': 'llm_failed', 'role': 'Doctor', 'content': failed_response}] + \
                            [{'source': 'llm_generated', 'role': t.get('role', 'Unknown'), 'content': t.get('content', '')} for t in continuation_content]
                    elif isinstance(continuation_content, str) and continuation_content:
                        multi_turn_response = [
                            {'source': 'llm_failed', 'role': 'Doctor', 'content': failed_response},
                            {'source': 'llm_generated', 'role': 'Doctor (Continuation)', 'content': continuation_content}
                        ]
                    else:
                        multi_turn_response = [
                            {'source': 'llm_failed', 'role': 'Doctor', 'content': failed_response},
                            {'source': 'llm_generated', 'role': 'Doctor (Continuation)', 'content': failed_response}
                        ]

                    return await asyncio.to_thread(
                        evaluator.evaluate_continuation,
                        case_id=case_id,
                        behavior_category=behavior_category,
                        conversation_history=convo_history,
                        multi_turn_response=multi_turn_response,
                        model=case.get('model', 'unknown')
                    )
                else:
                    behavior_category = case.get('behavior_category', 'Unknown')
                    case_id = case.get('case_id', 'unknown')
                    convo_history_list = case.get('conversation_history', [])
                    if isinstance(convo_history_list, list) and len(convo_history_list) > 0:
                        convo_history_str = "\n".join([f"{t.get('role', 'Unknown')}: {t.get('content', '')}" for t in convo_history_list if t.get('source') == 'original'])
                        multi_turn_response = [t for t in convo_history_list if t.get('source') in ['llm_failed', 'llm_generated']]
                    else:
                        convo_history_str = ""
                        multi_turn_response = case.get('multi_turn_response', [])

                    return await asyncio.to_thread(
                        evaluator.evaluate_continuation,
                        case_id=case_id,
                        behavior_category=behavior_category,
                        conversation_history=convo_history_str,
                        multi_turn_response=multi_turn_response,
                        model=case.get('model', 'unknown')
                    )
            except Exception as e:
                print(f"\n❌ Error evaluating case {case.get('case_id')}: {e}")
                return {
                    'case_id': case.get('case_id', 'unknown'),
                    'behavior_category': case.get('behavior_category', 'Unknown'),
                    'model': case.get('model', 'unknown'),
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }

    print("\nEvaluating cases (async)...")
    tasks = [evaluate_one(case) for case in cases]
    results = []
    for coro in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating"):
        result = await coro
        results.append(result)

    statistics = calculate_statistics(results)

    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Total Cases: {statistics['total_cases']}")
    print(f"Maintains Failure: {statistics['maintains_failure']} ({statistics['maintenance_rate']})")
    print(f"Corrected Failure: {statistics['corrected_failure']} ({statistics['correction_rate']})")
    print("\nBy Behavior Category:")
    for category, stats in statistics['by_behavior_category'].items():
        print(f"  {category}:")
        print(f"    Total: {stats['total']}")
        print(f"    Maintains: {stats['maintains_failure']} ({stats['maintenance_rate']})")
        print(f"    Corrected: {stats['corrected_failure']} ({stats['correction_rate']})")
    print("\nBy Model:")
    for model, stats in statistics['by_model'].items():
        print(f"  {model}:")
        print(f"    Total: {stats['total']}")
        print(f"    Maintains: {stats['maintains_failure']} ({stats['maintenance_rate']})")
        print(f"    Corrected: {stats['corrected_failure']} ({stats['correction_rate']})")
    print("="*80)

    return {
        'results': results,
        'statistics': statistics
    }


def save_results(
    results_data: Dict,
    output_dir: str,
    input_file: str,
    judge_model_name: str
):
    """
    Save evaluation results to JSON file.
    
    Args:
        results_data: Dictionary with results and statistics
        output_dir: Output directory
        input_file: Original input file path
        judge_model_name: Name of judge model used
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine output filename based on input
    input_name = Path(input_file).stem  # e.g., "abnormal_full" or "failed_full"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_path / f"{input_name}_continuation_eval_{timestamp}.json"
    
    # Prepare output data
    output_data = {
        'metadata': {
            'input_file': input_file,
            'judge_model': judge_model_name,
            'evaluated_at': datetime.now().isoformat(),
            'statistics': results_data['statistics']
        },
        'results': results_data['results']
    }
    
    # Save
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate multi-turn continuation dialogues"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input JSON file (abnormal_full.json or failed_full.json)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for evaluation results'
    )
    parser.add_argument(
        '--judge_model_name',
        type=str,
        required=True,
        help='Judge model name (e.g., gpt-4, claude-sonnet-4-5-20250929)'
    )
    parser.add_argument(
        '--judge_model_type',
        type=str,
        required=True,
        choices=['openai', 'claude', 'gemini'],
        help='Judge model type'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0,
        help='Temperature for judge model (default: 0)'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Evaluate only N samples for testing'
    )
    parser.add_argument(
        '--async',
        dest='use_async',
        action='store_true',
        default=False,
        help='Use async parallel processing (recommended for large datasets)'
    )
    parser.add_argument(
        '--concurrency',
        type=int,
        default=5,
        help='Max concurrent judge calls when using --async (default: 5)'
    )

    args = parser.parse_args()
    
    # Set up logger to capture all prints
    sys.stdout = Logger(args.output_dir)
    sys.stderr = sys.stdout  # Redirect stderr to the same logger

    print("="*80)
    print("MULTI-TURN CONTINUATION EVALUATOR")
    print("="*80)
    print(f"Input: {args.input}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Judge Model: {args.judge_model_name} ({args.judge_model_type})")
    print(f"Temperature: {args.temperature}")
    print(f"Execution Mode: {'Async' if args.use_async else 'Sync'}")
    if args.use_async:
        print(f"Concurrency: {args.concurrency}")
    if args.sample:
        print(f"Sample Size: {args.sample}")
    print("="*80)
    
    # Load data
    print(f"\n📂 Loading data from: {args.input}")
    data = load_continuation_data(args.input)
    print(f"   Loaded {len(data.get('results', []))} cases")
    
    # Create judge model
    print(f"\n🤖 Creating judge model: {args.judge_model_name}")
    judge_model = create_judge_model(
        judge_model_name=args.judge_model_name,
        judge_model_type=args.judge_model_type,
        temperature=args.temperature
    )
    
    # Evaluate
    if args.use_async:
        results_data = asyncio.run(evaluate_all_cases_async(
            data=data,
            judge_model=judge_model,
            output_dir=args.output_dir,
            concurrency=args.concurrency,
            sample=args.sample
        ))
    else:
        results_data = evaluate_all_cases(
            data=data,
            judge_model=judge_model,
            output_dir=args.output_dir,
            sample=args.sample
        )
    
    # Save results
    save_results(
        results_data=results_data,
        output_dir=args.output_dir,
        input_file=args.input,
        judge_model_name=args.judge_model_name
    )
    
    print("\n🎉 Evaluation complete!")


if __name__ == '__main__':
    main()
