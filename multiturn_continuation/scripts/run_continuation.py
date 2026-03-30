"""
Run Multi-turn Continuation Script

Generates multi-turn dialogue continuations for failed cases.
This script ONLY handles dialogue generation, NOT evaluation.

Note: Doctor model uses per-case strategy (original failed model for each case).
      Patient model is shared across all cases.

Usage:
    # Synchronous (original)
    python scripts/run_continuation.py \
        --input data_processing/output/failed_cases_multiturn.json \
        --patient_model gpt-4o-mini \
        --max_turns 10 \
        --output output/multiturn_generated.json \
        --sample 10

    # Async with concurrency control
    python scripts/run_continuation.py \
        --input data_processing/output/failed_cases_multiturn.json \
        --patient_model gpt-4o-mini \
        --max_turns 10 \
        --output output/multiturn_generated.json \
        --concurrency 8 \
        --async

    # Multiple input files
    python scripts/run_continuation.py \
        --input-list data_processing/output/abnormal_value_failed_cases_api_only.json \
                     data_processing/output/failed_cases_api_only.json \
        --patient_model gpt-4o-mini \
        --max_turns 10 \
        --output output/combined.json \
        --concurrency 10 \
        --async
"""

import argparse
import asyncio
import json
import os
import sys
import signal
import logging
from typing import Dict, List, Optional
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from continuation.engine import MultiTurnContinuationEngine
from continuation.async_engine import run_cases_parallel


def get_api_model_name(model: str) -> str:
    """
    Strip dataset-specific suffixes to obtain the real API model identifier.
    """
    known_suffixes = ("_abnormal",)
    for suffix in known_suffixes:
        if model.endswith(suffix):
            return model[: -len(suffix)]
    return model


def map_to_vllm_model_name(model_name: str) -> tuple:
    """
    Map dataset model names to actual vLLM model names.
    
    Returns:
        (vllm_model_name, enable_thinking)
    
    Examples:
        "Qwen_Qwen3-32B_thinking" -> ("Qwen/Qwen3-32B-Instruct", True)
        "Qwen_Qwen3-32B_no_thinking" -> ("Qwen/Qwen3-32B-Instruct", False)
        "meta-llama_Llama-3.3-70B-Instruct" -> ("meta-llama/Llama-3.3-70B-Instruct", False)
    """
    model_lower = model_name.lower()
    
    # Qwen models
    if "qwen" in model_lower:
        enable_thinking = "thinking" in model_lower and "no_thinking" not in model_lower
        # Map to Qwen3-32B (deployed on vLLM server)
        return "Qwen/Qwen3-32B", enable_thinking
    
    # Llama models - extract the actual model path
    if "llama" in model_lower:
        # Handle formats like "meta-llama_Llama-3.3-70B-Instruct"
        if "meta-llama" in model_lower:
            # Replace underscore with slash for proper model path
            model_path = model_name.replace("meta-llama_", "meta-llama/")
            # Remove any suffixes
            model_path = get_api_model_name(model_path)
            return model_path, False
        return "meta-llama/Llama-3.3-70B-Instruct", False
    
    # API models - return as is
    return model_name, False


def is_vllm_model(model_name: str) -> bool:
    """Check if a model requires vLLM."""
    model_lower = model_name.lower()
    return "qwen" in model_lower or "llama" in model_lower


def load_failed_cases(input_path: str, sample: Optional[int] = None) -> List[Dict]:
    """Load prepared failed cases data."""
    print(f"📂 Loading failed cases from: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    cases = data.get('failed_cases', [])
    
    if sample:
        cases = cases[:sample]
        print(f"   📊 Using sample of {len(cases)} cases (from total {data['metadata']['total_failed_cases']})")
    else:
        print(f"   📊 Total cases: {len(cases)}")
    
    return cases


def create_doctor_agent(model_name: str, generation_config: Dict, vllm_api_base: str = None):
    """
    Create Doctor agent for a specific model.
    
    Args:
        model_name: Original model name from dataset
        generation_config: Generation configuration dict
        vllm_api_base: Base URL for vLLM API server (for vLLM models)
    """
    from multiturn_continuation.agents.doctor import DirectDoctor
    import prompts.doctor_prompts as doctor_prompts
    
    # Map model name for vLLM models
    actual_model_name, enable_thinking = map_to_vllm_model_name(model_name)
    
    # Add vLLM parameters to generation_config if needed
    config = generation_config.copy()
    if is_vllm_model(model_name):
        if not vllm_api_base:
            raise ValueError(
                f"vLLM model '{model_name}' requires --vllm-api-base parameter or VLLM_API_BASE environment variable"
            )
        config['vllm_api_base'] = vllm_api_base
        config['enable_thinking'] = enable_thinking
    
    doctor = DirectDoctor(
        model_name=actual_model_name,
        prompt_key='default',
        generation_config=config
    )
    
    return doctor


def create_patient_agent(model_name: str, generation_config: Dict):
    """Create Patient agent for generate mode."""
    print(f"\n🧑 Creating Patient agent (for generation mode)...")
    print(f"   Model: {model_name}")
    
    from multiturn_continuation.agents.patient import DirectPatient
    
    patient = DirectPatient(
        model_name=model_name,
        generation_config=generation_config
    )
    
    print(f"   ✅ Patient agent created")
    return patient


def save_results(results: List[Dict], output_path: str, config: Dict):
    """Save generation results to JSON."""
    print(f"\n📋 Saving results...")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Prepare output
    output_data = {
        'config': config,
        'metadata': {
            'total_cases': len(results),
            'generated_at': datetime.now().isoformat(),
        },
        'results': results
    }
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ Results saved to: {output_path}")


def setup_logging(output_dir: str, log_level: str = 'INFO') -> str:
    """
    Setup logging configuration.
    
    Returns:
        Path to the error log file
    """
    # Create logs directory
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    error_log_path = os.path.join(log_dir, f'errors_{timestamp}.json')
    
    return error_log_path


def save_error_log(failed_cases: List[Dict], error_log_path: str):
    """Save detailed error information to JSON file."""
    if not failed_cases:
        return
    
    error_data = {
        'timestamp': datetime.now().isoformat(),
        'total_failed': len(failed_cases),
        'failed_cases': failed_cases
    }
    
    with open(error_log_path, 'w', encoding='utf-8') as f:
        json.dump(error_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n📝 Error log saved to: {error_log_path}")


# Global state for graceful shutdown
_shutdown_requested = False
_partial_results = []
_partial_failed = []
_output_path = None
_error_log_path = None


def signal_handler(signum, frame):
    """Handle Ctrl+C and other interrupts gracefully."""
    global _shutdown_requested
    print(f"\n\n⚠️  Interrupt received (signal {signum}). Saving partial results...")
    _shutdown_requested = True
    
    # Save what we have so far
    if _partial_results or _partial_failed:
        try:
            if _partial_results and _output_path:
                # Save partial results
                partial_output = _output_path.replace('.json', '_partial.json')
                config = {
                    'note': 'Partial results from interrupted run',
                    'interrupted_at': datetime.now().isoformat()
                }
                save_results(_partial_results, partial_output, config)
                print(f"   ✅ Saved {len(_partial_results)} completed cases to: {partial_output}")
            
            if _partial_failed and _error_log_path:
                # Save error log
                save_error_log(_partial_failed, _error_log_path)
                print(f"   ✅ Saved {len(_partial_failed)} failed cases to error log")
        except Exception as e:
            print(f"   ❌ Error saving partial results: {e}")
    
    print("\n👋 Exiting gracefully...")
    sys.exit(1)


def print_summary(results: List[Dict], failed_cases: List[Dict] = None):
    """Print generation summary."""
    print(f"\n{'='*80}")
    print("📊 GENERATION SUMMARY")
    print(f"{'='*80}")
    
    total_cases = len(results)
    total_new_turns = sum(len(r['multi_turn_response']) for r in results)
    avg_new_turns = total_new_turns / total_cases if total_cases > 0 else 0
    
    print(f"\nSuccessful Cases: {total_cases}")
    print(f"Total New Turns Generated: {total_new_turns}")
    print(f"Average New Turns per Case: {avg_new_turns:.1f}")
    
    # Failed cases summary
    if failed_cases:
        print(f"\n⚠️  Failed Cases: {len(failed_cases)}")
        
        # Group by error type
        error_types = {}
        for fc in failed_cases:
            error_type = fc.get('error_type', 'Unknown')
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append(fc)
        
        print(f"\nBy Error Type:")
        for error_type, cases in sorted(error_types.items()):
            print(f"   {error_type}: {len(cases)} cases")
            # Show first 3 case IDs
            for case in cases[:3]:
                print(f"      - {case['case_id']}")
            if len(cases) > 3:
                print(f"      ... and {len(cases) - 3} more")
    
    # By category
    by_category = {}
    for result in results:
        category = result['behavior_category']
        if category not in by_category:
            by_category[category] = 0
        by_category[category] += 1
    
    print(f"\nBy Category:")
    for category, count in sorted(by_category.items()):
        print(f"   {category}: {count} cases")
    
    print(f"\n{'='*80}")
    if failed_cases:
        print(f"⚠️  Generation complete with {len(failed_cases)} errors (see error log)")
    else:
        print("✅ Generation complete!")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Generate multi-turn dialogue continuations")
    
    # Input arguments (mutually exclusive: --input or --input-list)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        type=str,
        help="Path to a single failed_cases JSON file"
    )
    input_group.add_argument(
        "--input-list",
        nargs='+',
        type=str,
        help="Paths to multiple failed_cases JSON files (will be merged)"
    )
    
    parser.add_argument(
        "--patient_model",
        type=str,
        default="gpt-4o-mini",
        help="Patient model for generation mode (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=10,
        help="Maximum additional turns to generate (default: 10)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/multiturn_generated.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Test on a sample of N cases (default: all)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress"
    )
    parser.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        help="Use async parallel processing (recommended for large datasets)"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max concurrent cases when using --async (default: 5)"
    )
    parser.add_argument(
        "--vllm-api-base",
        type=str,
        default=None,
        help="vLLM API base URL (e.g., http://localhost:8000/v1). Can also use VLLM_API_BASE env var."
    )
    
    parser.add_argument(
        "--doctor_model",
        type=str,
        default=None,
        help="Override doctor model for all cases (e.g. gpt-4o-mini). Useful when the original model's API is unavailable."
    )

    args = parser.parse_args()
    
    # Get vLLM API base from args or environment
    import os
    vllm_api_base = args.vllm_api_base or os.getenv("VLLM_API_BASE")
    
    # Setup logging
    output_dir = os.path.dirname(args.output)
    error_log_path = setup_logging(output_dir)
    
    # Register signal handlers for graceful shutdown
    global _output_path, _error_log_path
    _output_path = args.output
    _error_log_path = error_log_path
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Kill signal
    
    # Print header
    print("=" * 80)
    print("🚀 MULTI-TURN CONTINUATION - DIALOGUE GENERATION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"   Doctor Model: {'Override: ' + args.doctor_model if args.doctor_model else 'Per-case (using original failed model)'}")
    print(f"   Patient Model: {args.patient_model}")
    print(f"   Patient Strategy: Generate (LLM)")
    print(f"   Max Turns: {args.max_turns}")
    print(f"   Execution Mode: {'Async' if args.use_async else 'Sync'}")
    if args.use_async:
        print(f"   Concurrency: {args.concurrency}")
    if vllm_api_base:
        print(f"   vLLM API Base: {vllm_api_base}")
    print(f"   Verbose: {args.verbose}")
    print(f"   Error Log: {error_log_path}")
    
    # Load cases (support single file or multiple files)
    if args.input:
        cases = load_failed_cases(args.input, args.sample)
    else:
        # Load and merge multiple input files
        print(f"\n📂 Loading from {len(args.input_list)} input files...")
        all_cases = []
        for input_path in args.input_list:
            file_cases = load_failed_cases(input_path, sample=None)
            all_cases.extend(file_cases)
        
        if args.sample:
            all_cases = all_cases[:args.sample]
            print(f"   📊 Using sample of {len(all_cases)} cases (from merged total)")
        
        cases = all_cases
        print(f"   📊 Total merged cases: {len(cases)}")
    
    # Analyze unique models
    unique_models = set(case.get('model', 'unknown') for case in cases)
    print(f"\n📊 Unique doctor models in dataset: {len(unique_models)}")
    
    # Check if any vLLM models are present
    vllm_models = [m for m in unique_models if is_vllm_model(m)]
    if vllm_models and not vllm_api_base:
        print(f"\n⚠️  WARNING: Dataset contains vLLM models but no vLLM API base provided:")
        for model in sorted(vllm_models):
            count = sum(1 for c in cases if c.get('model') == model)
            print(f"   • {model}: {count} cases")
        print(f"\n   Please provide --vllm-api-base or set VLLM_API_BASE environment variable")
        print(f"   Example: export VLLM_API_BASE='http://your-gpu-server:8000/v1'")
        sys.exit(1)
    
    for model in sorted(unique_models):
        count = sum(1 for c in cases if c.get('model') == model)
        model_type = "vLLM" if is_vllm_model(model) else "API"
        print(f"   • {model}: {count} cases [{model_type}]")
    
    # Create patient agent (shared across all cases)
    patient_generation_config = {'max_tokens': 4096}
    print(f"\n🧑 Creating Patient agent (shared)...")
    print(f"   Model: {args.patient_model}")
    patient_agent = create_patient_agent(args.patient_model, patient_generation_config)
    print(f"   ✅ Patient agent created")
    
    # Pre-create all doctor agents for unique models
    print(f"\n🩺 Pre-creating Doctor agents...")
    doctor_agents_cache = {}

    # If a doctor model override is specified, use it for all cases
    if args.doctor_model:
        print(f"   ⚠️  Overriding all doctor models with: {args.doctor_model}")
        for case in cases:
            case['model'] = args.doctor_model
        unique_models = {args.doctor_model}

    for model in sorted(unique_models):
        api_model = get_api_model_name(model)
        if api_model not in doctor_agents_cache:
            # Map to actual vLLM model name if needed
            actual_model, enable_thinking = map_to_vllm_model_name(api_model)
            print(f"   Creating agent for: {api_model}", end="")
            
            if is_vllm_model(api_model):
                print(f" -> {actual_model} (thinking={enable_thinking})")
                doctor_generation_config = {'max_tokens': 4096}
            elif 'gpt' in api_model.lower() or 'claude' in api_model.lower() or 'gemini' in api_model.lower() or 'deepseek' in api_model.lower():
                print()
                doctor_generation_config = {'max_tokens': 4096}
            else:
                print()
                doctor_generation_config = {'max_tokens': 4096}
            
            doctor_agents_cache[api_model] = create_doctor_agent(
                api_model,
                doctor_generation_config,
                vllm_api_base=vllm_api_base
            )
    print(f"   ✅ Created {len(doctor_agents_cache)} doctor agents")
    
    # Run continuation with per-case doctor models
    print(f"\n{'='*80}")
    print("🔄 Generating dialogue continuations...")
    print(f"{'='*80}")
    
    # Declare global variables at the start for signal handler
    global _partial_results, _partial_failed
    
    if args.use_async:
        # Async parallel execution
        # For async mode, default to non-verbose (only show progress bar)
        # unless explicitly requested with --verbose
        async_verbose = args.verbose
        
        try:
            results, failed = asyncio.run(
                run_cases_parallel(
                    cases=cases,
                    doctor_agents_cache=doctor_agents_cache,
                    patient_agent=patient_agent,
                    max_turns=args.max_turns,
                    concurrency=args.concurrency,
                    verbose=async_verbose
                )
            )
        except KeyboardInterrupt:
            print("\n\n⚠️  Keyboard interrupt during async execution")
            print("   Note: Async mode doesn't support partial saves during execution")
            print("   Consider using sync mode for interruptible runs")
            sys.exit(1)
        
        # Update global state
        _partial_results = results
        _partial_failed = failed
        
        # Save error log if any failures
        if failed:
            save_error_log(failed, error_log_path)
            print(f"\n⚠️  {len(failed)} cases failed:")
            for failure in failed[:10]:  # Show first 10
                print(f"   • {failure['case_id']} ({failure['model']}): {failure['error_type']}")
            if len(failed) > 10:
                print(f"   ... and {len(failed) - 10} more")
        
        failed_cases = failed
    else:
        # Synchronous execution (original behavior)
        results = []
        failed_cases = []
        
        # Update global state for signal handler
        _partial_results = results
        _partial_failed = failed_cases
        
        for case in tqdm(cases, desc="Processing cases"):
            # Check if shutdown was requested
            if _shutdown_requested:
                print("\n⚠️  Shutdown requested, stopping processing...")
                break
            
            try:
                # Get the model name for this case
                case_model = case.get('model', 'unknown')
                api_model = get_api_model_name(case_model)
                doctor_agent = doctor_agents_cache[api_model]
                
                # Create engine for this case
                engine = MultiTurnContinuationEngine(
                    doctor_agent=doctor_agent,
                    patient_agent=patient_agent,
                    verbose=args.verbose
                )
                
                # Run continuation
                result = engine.run_continuation(
                    case=case,
                    max_turns=args.max_turns
                )
                results.append(result)
                
            except Exception as e:
                # Record error
                import traceback
                error_info = {
                    'case_id': case.get('case_id', 'unknown'),
                    'case_index': cases.index(case),
                    'model': case.get('model', 'unknown'),
                    'dataset': case.get('dataset', 'unknown'),
                    'behavior_category': case.get('behavior_category', 'unknown'),
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'traceback': traceback.format_exc(),
                    'timestamp': datetime.now().isoformat()
                }
                failed_cases.append(error_info)
                
                print(f"\n❌ Error processing case {case.get('case_id', 'unknown')}: {e}")
                if args.verbose:
                    traceback.print_exc()
                continue
        
        # Save error log if any failures
        if failed_cases:
            save_error_log(failed_cases, error_log_path)
    
    # Save results
    config = {
        'doctor_model_strategy': 'per-case (original failed model)',
        'patient_model': args.patient_model,
        'max_turns': args.max_turns,
        'execution_mode': 'async' if args.use_async else 'sync',
        'unique_doctor_models': sorted(list(unique_models))
    }
    
    if args.use_async:
        config['concurrency'] = args.concurrency
    save_results(results, args.output, config)
    
    # Print summary
    print_summary(results, failed_cases if 'failed_cases' in locals() else None)


if __name__ == "__main__":
    main()
