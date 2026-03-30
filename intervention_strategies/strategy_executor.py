import os
import asyncio

from common import load_all_cases, load_negative_cases, save_results, build_model, convert_conversation_to_string
from strategies.base import BaseStrategy


class StrategyExecutor:
    """Orchestrates data loading, case iteration, and result saving for any strategy."""

    def __init__(self, strategy: BaseStrategy, model_name: str,
                 vllm_api_base: str = None, enable_thinking: bool = False,
                 concurrency: int = 1, input_mode: str = "positive"):
        self.strategy = strategy
        # Append suffix for thinking mode to avoid overwriting non-thinking outputs
        if enable_thinking:
            self.model_name = model_name + "_thinking"
        else:
            self.model_name = model_name
        self.concurrency = concurrency
        self.input_mode = input_mode  # "positive" or "negative"
        self.llm = build_model(model_name, vllm_api_base=vllm_api_base, enable_thinking=enable_thinking)

    async def _process_single(self, idx, row, df, semaphore, counter, total_cases, max_retries=3):
        async with semaphore:
            case_id = row.get('case_id') or row.get('dialog_id', f'case_{idx}')
            counter['count'] += 1
            print(f"\n[{counter['count']}/{total_cases}] {self.strategy.name} | {self.model_name} | Case: {case_id}")

            conversation_segment = row['conversation_segment']
            formatted_history = convert_conversation_to_string(conversation_segment)

            for attempt in range(1, max_retries + 1):
                try:
                    case = row.to_dict()
                    result = await self.strategy.process_case(case, formatted_history, conversation_segment, self.llm)
                    return idx, conversation_segment, result
                except Exception as e:
                    if attempt < max_retries:
                        wait = 2 ** attempt
                        print(f"  [Retry {attempt}/{max_retries}] Case {case_id} failed: {e}. Retrying in {wait}s...")
                        await asyncio.sleep(wait)
                    else:
                        print(f"  [FAILED] Case {case_id} after {max_retries} attempts: {e}")
                        raise

    async def run(self):
        if self.input_mode == "negative":
            df = load_negative_cases()
            output_subdir = f"negative_{self.strategy.output_dir}"
        else:
            df = load_all_cases()
            output_subdir = self.strategy.output_dir
        total_cases = len(df)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, output_subdir)
        os.makedirs(output_dir, exist_ok=True)

        # Check if all output files already exist
        if skip_if_exists:
            safe_model_name = self.model_name.replace("/", "_").replace(":", "_")
            datasets = df['dataset'].unique()
            existing = [
                os.path.join(output_dir, f"{safe_model_name}_{ds}.jsonl")
                for ds in datasets
                if os.path.exists(os.path.join(output_dir, f"{safe_model_name}_{ds}.jsonl"))
            ]
            if len(existing) == len(datasets):
                print(f"Skipping {self.strategy.name} | {self.model_name} — all {len(datasets)} output files already exist.")
                return

        # Initialize model (deferred to avoid unnecessary API client creation when skipping)
        self.llm = build_model(self._raw_model_name, vllm_api_base=self._vllm_api_base, enable_thinking=self._enable_thinking)

        mode = "async" if self.concurrency > 1 else "sync"
        print(f"Strategy: {self.strategy.name} | Model: {self.model_name} | Cases: {total_cases} | Mode: {mode} (concurrency={self.concurrency})")

        semaphore = asyncio.Semaphore(self.concurrency)
        counter = {'count': 0}

        tasks = []
        for idx, row in df.iterrows():
            task = asyncio.create_task(self._process_single(idx, row, df, semaphore, counter, total_cases))
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # Apply results to dataframe
        for idx, conversation_segment, result in results:
            new_turn = {
                "Doctor": result.generated_response,
                "turn index": len(conversation_segment) + 1,
                "source": "llm_generated"
            }
            df.at[idx, 'conversation_segment'] = list(conversation_segment) + [new_turn]
            df.at[idx, 'generated_response'] = result.generated_response

            for key, value in result.extra_fields.items():
                if key not in df.columns:
                    df[key] = None
                    df[key] = df[key].astype(object)
                df.at[idx, key] = value

        save_results(df, output_dir, self.model_name)
