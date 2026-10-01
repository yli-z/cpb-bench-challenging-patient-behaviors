import asyncio
import argparse

import common  # noqa: F401 — must be imported first to set up sys.path
from strategies import STRATEGY_REGISTRY, get_strategy
from strategy_executor import StrategyExecutor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run intervention strategies for doctor model failures.")
    parser.add_argument("mode", choices=list(STRATEGY_REGISTRY.keys()), help="The strategy to use.")
    parser.add_argument("--model", required=True, help="Model name (e.g. gpt-4o, meta-llama/Llama-3.3-70B-Instruct).")
    parser.add_argument("--vllm-api-base", type=str, default=None,
                        help="vLLM API base URL (e.g. http://localhost:8006/v1). Required for vLLM models.")
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Enable thinking mode (for Qwen3 thinking variant).")
    parser.add_argument("--concurrency", type=int, default=32,
                        help="Max concurrent API calls (default: 32).")
    parser.add_argument("--no-skip", dest="skip_if_exists", action="store_false", default=True,
                        help="Force regeneration even if output files exist.")
    parser.add_argument("--negative", action="store_true",
                        help="Run on negative cases instead of positive benchmark cases.")
    args = parser.parse_args()

    strategy = get_strategy(args.mode)
    executor = StrategyExecutor(
        strategy, args.model,
        vllm_api_base=args.vllm_api_base,
        enable_thinking=args.enable_thinking,
        concurrency=args.concurrency,
        input_mode="negative" if args.negative else "positive",
    )
    asyncio.run(executor.run(skip_if_exists=args.skip_if_exists))
