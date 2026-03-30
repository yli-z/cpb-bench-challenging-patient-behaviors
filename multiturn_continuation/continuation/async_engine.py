"""
Async wrapper for Multi-turn Continuation Engine

Enables parallel processing of multiple cases while maintaining
strict sequential dialogue order within each case.
"""

import asyncio
import sys
import os
import traceback
from datetime import datetime
from typing import Dict, List
from tqdm.asyncio import tqdm as async_tqdm

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from continuation.engine import MultiTurnContinuationEngine


class AsyncMultiTurnEngine:
    """
    Async wrapper for MultiTurnContinuationEngine.
    
    Enables parallel processing of multiple cases with concurrency control.
    Each case is processed sequentially (Doctor-Patient dialogue order preserved),
    but multiple cases run concurrently.
    """
    
    def __init__(
        self,
        doctor_agents_cache: Dict,
        patient_agent,
        semaphore: asyncio.Semaphore,
        verbose: bool = False
    ):
        """
        Args:
            doctor_agents_cache: Shared cache of doctor agents by model name
            patient_agent: Shared patient agent
            semaphore: Semaphore to control max concurrent cases
            verbose: Print detailed progress
        """
        self.doctor_agents_cache = doctor_agents_cache
        self.patient_agent = patient_agent
        self.semaphore = semaphore
        self.verbose = verbose
    
    async def run_case_async(
        self,
        case: Dict,
        max_turns: int = 10
    ) -> Dict:
        """
        Run continuation for a single case asynchronously.
        
        The actual dialogue generation is sequential (Doctor-Patient order preserved).
        This method just wraps it in async to enable parallel processing of multiple cases.
        
        Args:
            case: Failed case data
            max_turns: Maximum number of additional turns to generate
            
        Returns:
            Result dictionary with generated dialogue
        """
        async with self.semaphore:
            # Get the appropriate doctor agent for this case's model
            from scripts.run_continuation import get_api_model_name
            
            case_model = case.get('model', 'unknown')
            api_model = get_api_model_name(case_model)
            
            if api_model not in self.doctor_agents_cache:
                raise ValueError(
                    f"Doctor agent for model '{api_model}' not found in cache. "
                    f"Please pre-create all doctor agents before running async."
                )
            
            doctor_agent = self.doctor_agents_cache[api_model]
            
            # Create synchronous engine for this case
            engine = MultiTurnContinuationEngine(
                doctor_agent=doctor_agent,
                patient_agent=self.patient_agent,
                verbose=self.verbose
            )
            
            # Run the continuation in a thread pool to avoid blocking
            # (LLM API calls are I/O bound, so this is efficient)
            result = await asyncio.to_thread(
                engine.run_continuation,
                case=case,
                max_turns=max_turns
            )
            
            return result


async def run_cases_parallel(
    cases: List[Dict],
    doctor_agents_cache: Dict,
    patient_agent,
    max_turns: int = 10,
    concurrency: int = 5,
    verbose: bool = False
) -> tuple[List[Dict], List[Dict]]:
    """
    Run multiple cases in parallel with concurrency control.
    
    Args:
        cases: List of failed case data
        doctor_agents_cache: Shared cache of doctor agents
        patient_agent: Shared patient agent
        max_turns: Maximum number of additional turns per case
        concurrency: Maximum number of concurrent cases
        verbose: Print detailed progress
        
    Returns:
        Tuple of (successful_results, failed_cases_with_errors)
    """
    semaphore = asyncio.Semaphore(concurrency)
    
    async_engine = AsyncMultiTurnEngine(
        doctor_agents_cache=doctor_agents_cache,
        patient_agent=patient_agent,
        semaphore=semaphore,
        verbose=verbose
    )
    
    # Create tasks for all cases with index tracking
    tasks = []
    for i, case in enumerate(cases):
        task = async_engine.run_case_async(case, max_turns)
        tasks.append((i, case, task))  # Store (index, case, task) tuple
    
    # Run all tasks and collect results with progress bar
    results = []
    failed = []
    
    if verbose:
        # Verbose mode: no progress bar, detailed logs from engine
        # Extract just the tasks from tuples
        task_list = [task for _, _, task in tasks]
        outcomes = await asyncio.gather(*task_list, return_exceptions=True)
        
        # Process outcomes with original indices
        for idx, (i, case, _) in enumerate(tasks):
            outcome = outcomes[idx]
            if isinstance(outcome, Exception):
                error_info = {
                    'case_id': case.get('case_id', f'case_{i}'),
                    'case_index': i,
                    'model': case.get('model', 'unknown'),
                    'dataset': case.get('dataset', 'unknown'),
                    'behavior_category': case.get('behavior_category', 'unknown'),
                    'error': str(outcome),
                    'error_type': type(outcome).__name__,
                    'traceback': ''.join(traceback.format_exception(type(outcome), outcome, outcome.__traceback__)),
                    'timestamp': datetime.now().isoformat()
                }
                failed.append(error_info)
            else:
                results.append(outcome)
    else:
        # Non-verbose mode: real-time progress bar with as_completed
        from tqdm import tqdm
        
        # Create wrapped tasks that preserve case information
        async def wrapped_task(index, case, task):
            """Wrapper that preserves case info with the task result"""
            try:
                result = await task
                return ('success', index, case, result)
            except Exception as e:
                return ('error', index, case, e)
        
        # Wrap all tasks
        wrapped_tasks = [wrapped_task(i, case, task) for i, case, task in tasks]
        
        with tqdm(total=len(wrapped_tasks), desc="Processing cases") as pbar:
            # Use as_completed for real-time progress updates
            for completed in asyncio.as_completed(wrapped_tasks):
                status, index, case, outcome = await completed
                
                if status == 'success':
                    results.append(outcome)
                else:  # status == 'error'
                    error_info = {
                        'case_id': case.get('case_id', f'case_{index}'),
                        'case_index': index,
                        'model': case.get('model', 'unknown'),
                        'dataset': case.get('dataset', 'unknown'),
                        'behavior_category': case.get('behavior_category', 'unknown'),
                        'error': str(outcome),
                        'error_type': type(outcome).__name__,
                        'traceback': ''.join(traceback.format_exception(type(outcome), outcome, outcome.__traceback__)),
                        'timestamp': datetime.now().isoformat()
                    }
                    failed.append(error_info)
                
                # Update progress bar after each task completes (real-time!)
                pbar.update(1)
    
    return results, failed


if __name__ == "__main__":
    print("Async Multi-turn Continuation Engine module loaded successfully.")
