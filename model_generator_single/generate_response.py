"""
Generate doctor responses from LLMs for safety benchmark cases.

Usage examples:
  # Using environment variables:
  export OPENAI_API_KEY="sk-..."
  python generate_response.py --dataset ACI --model gpt-4
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# # Add parent directory to path to allow imports
ROOT = Path(__file__).resolve().parent  # model_generate directory
PARENT_DIR = ROOT.parent  # Formal directory
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from prompts.prompt_templates import DOCTOR_ASSISTANT_PROMPT
from models.openai_model import OpenAIModel
from models.claude_model import ClaudeModel
from models.deepseek_model import DeepSeekModel
from models.remote_vllm_model import RemoteVLLMModel
# Benchmark data is stored under data_loader/output (sibling to model_generate)
OUTPUT_DIR = PARENT_DIR / "data_loader" / "output_benchmark"
CONFIG_PATH = ROOT / "config.json"


def load_config() -> Dict:
    """Load config file (optional, only for model names, not API keys)."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {}

def convert_conversation_to_string(conversation_segment: List[Dict]) -> str:
    lines = []
    for turn in conversation_segment:
        # Each turn dict has one speaker key (either "Doctor" or "Patient") and "turn index"
        # Extract the speaker and text, ignoring "turn index"
        for key, value in turn.items():
            if key != "turn index":
                # key is the speaker (Doctor/Patient), value is the text
                lines.append(f"{key}: {value}")
                break  # Only one speaker key per turn dict
    return "\n".join(lines)


def _get_last_patient_text(conversation_segment: list) -> str:
    """Return the text of the last Patient turn in a conversation_segment list."""
    for turn in reversed(conversation_segment):
        if "Patient" in turn:
            return turn["Patient"]
    return ""


def load_cases(datasets: List[str], file_path: Optional[str] = None) -> List[Dict]:
    cases: List[Dict] = []
    for ds in datasets:

        if file_path is None:            
            primary_path = OUTPUT_DIR / f"{ds}_safety_benchmark.json"
            fallback_path = PARENT_DIR / "syn_generator" / "output" / f"complementary_{ds}_generated_cases_validated_gpt4o-mini.json"
            if primary_path.exists():
                file_path = primary_path
            elif fallback_path.exists():
                file_path = fallback_path
            else:
                raise FileNotFoundError(
                    f"Dataset file not found: {primary_path} or {fallback_path}"
                )
        with open(file_path, "r") as f:
            data = json.load(f)

        # Support both dict-with-"cases" (benchmark format) and flat list (negative-sample format)
        raw_cases: List[Dict] = data.get("cases", []) if isinstance(data, dict) else data

        for c in raw_cases:
            c["_dataset"] = ds
            # Auto-fill fields required by the evaluator when loading negative-case sampled files
            if "behavior_category" not in c:
                c["behavior_category"] = "No behavior"
            if "patient_behavior_text" not in c and isinstance(c.get("conversation_segment"), list):
                c["patient_behavior_text"] = _get_last_patient_text(c["conversation_segment"])
            # Convert conversation_segment from list format to string format
            if "conversation_segment" in c and isinstance(c["conversation_segment"], list):
                c["conversation_segment"] = convert_conversation_to_string(c["conversation_segment"])

        cases.extend(raw_cases)
    return cases


def build_model(model_name: str, api_keys: Dict[str, str] = None, vllm_api_base: str = None, enable_thinking: bool = False):
    if api_keys is None:
        api_keys = {}
    
    name_lower = model_name.lower()
    if name_lower.startswith("gpt-"):
        # API key from command line arg, env var, or None (model will use env var)
        api_key = api_keys.get("openai") or os.getenv("OPENAI_API_KEY")
        return OpenAIModel(
            model_name=model_name,
            api_key=api_key,
            temperature=None,
            max_tokens=4096,
        )
    if name_lower.startswith("claude"):
        api_key = api_keys.get("claude") or os.getenv("ANTHROPIC_API_KEY")
        # Anthropic API requires lowercase model names (e.g., "claude-sonnet-4-5-20250929")
        claude_model_name = model_name.lower()
        return ClaudeModel(
            model_name=claude_model_name,
            api_key=api_key,
            temperature=None,
            max_tokens=4096,
        )
    if name_lower.startswith("gemini"):
        from models.gemini_model import GeminiModel
        api_key = api_keys.get("gemini") or os.getenv("GEMINI_API_KEY")
        # Gemini API requires lowercase model names (e.g., "gemini-2.5-flash")
        gemini_model_name = model_name.lower()
        return GeminiModel(
            model_name=gemini_model_name,
            api_key=api_key,
            temperature=None,
            max_tokens=4096,
        )
    # if name_lower.startswith("deepseek"):
    #     api_key = api_keys.get("deepseek") or os.getenv("DEEPSEEK_API_KEY")
    #     # DeepSeek models: deepseek-reasoner (Thinking) or deepseek-chat (No Thinking)
    #     return DeepSeekModel(
    #         model_name=model_name,
    #         api_key=api_key,
    #         temperature=None,  # Use default (not set in API call)
    #         max_tokens=4096,
    #     )

    if name_lower.startswith("deepseek"):
        # deepseek-ai/ 开头的是 HuggingFace 模型，用 vLLM 跑
        if name_lower.startswith("deepseek-ai/"):
            if not vllm_api_base and not os.getenv("VLLM_API_BASE"):
                raise ValueError(
                    f"vLLM API base URL required for model '{model_name}'. "
                    "Please provide via --vllm-api-base argument or VLLM_API_BASE environment variable."
                )
            return RemoteVLLMModel(
                model_name=model_name,
                vllm_api_base=vllm_api_base,
                max_tokens=4096,
            )
        # deepseek-chat / deepseek-reasoner 调 DeepSeek API
        api_key = api_keys.get("deepseek") or os.getenv("DEEPSEEK_API_KEY")
        return DeepSeekModel(
            model_name=model_name,
            api_key=api_key,
            temperature=None,
            max_tokens=4096,
        )
        
    if name_lower.startswith(("qwen", "gpt-oss")):
        # Use RemoteVLLM for remote GPU models (via vLLM service)
        if not vllm_api_base and not os.getenv("VLLM_API_BASE"):
            raise ValueError(
                f"vLLM API base URL required for model '{model_name}'. "
                "Please provide via --vllm-api-base argument or VLLM_API_BASE environment variable."
            )
        return RemoteVLLMModel(
            model_name=model_name,
            vllm_api_base=vllm_api_base,
            max_tokens=4096,
            temperature=None,
            enable_thinking=enable_thinking,
        )
    if name_lower.startswith("meta"):
        # Use RemoteVLLM for remote GPU models (via vLLM service)
        if not vllm_api_base and not os.getenv("VLLM_API_BASE"):
            raise ValueError(
                f"vLLM API base URL required for model '{model_name}'. "
                "Please provide via --vllm-api-base argument or VLLM_API_BASE environment variable."
            )
        return RemoteVLLMModel(
            model_name=model_name,
            vllm_api_base=vllm_api_base,
            max_tokens=4096,
        )
    if "gemma" in name_lower or name_lower.startswith("google"):
        if not vllm_api_base and not os.getenv("VLLM_API_BASE"):
            raise ValueError(
                f"vLLM API base URL required for model '{model_name}'. "
                "Please provide via --vllm-api-base argument or VLLM_API_BASE environment variable."
            )
        return RemoteVLLMModel(
            model_name=model_name,
            vllm_api_base=vllm_api_base,
            max_tokens=4096,
        )

    raise ValueError(f"Unsupported model name: {model_name}")


def save_results(results: List[Dict], model_name: str, datasets: List[str], append: bool = False, enable_thinking: bool = False, output_dir: Optional[str] = None):

    ds_tag = "all" if len(datasets) > 1 else datasets[0]
    if output_dir:
        MODEL_OUTPUT_DIR = Path(output_dir)
    else:
        MODEL_OUTPUT_DIR = ROOT / f"model_output_syn_{ds_tag}_v1"
    MODEL_OUTPUT_DIR.mkdir(exist_ok=True, parents=True) 
    # Add thinking mode suffix for Qwen3 models
    model_name_safe = model_name.replace(':', '_').replace('/', '_')
    if "qwen3" in model_name.lower() or "qwen" in model_name.lower():
        mode_suffix = "_thinking" if enable_thinking else "_no_thinking"
        model_name_safe = f"{model_name_safe}{mode_suffix}"
    out_path = MODEL_OUTPUT_DIR / f"{model_name_safe}_{ds_tag}.jsonl"
    
    mode = "a" if append else "w"
    existing_count = 0
    if append and out_path.exists():
        with open(out_path, "r") as f:
            existing_count = sum(1 for _ in f)
    
    with open(out_path, mode) as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    total = existing_count + len(results) if append else len(results)
    print(f"✅ Saved {len(results)} responses to {out_path} (total: {total})")


async def generate_doctor_response_async(
    llm,
    case: Dict,
    semaphore: asyncio.Semaphore,
) -> Dict:
    async with semaphore:
        return await asyncio.to_thread(
            llm.generate_doctor_response,
            case,
            DOCTOR_ASSISTANT_PROMPT,
            "",
        )

async def process_case(
    index: int,
    case: Dict,
    llm,
    semaphore: asyncio.Semaphore,
    is_gemini: bool,
    retry_limit: int,
    gemini_lock: asyncio.Lock,
    gemini_counter: Dict[str, int],
    test_mode: bool,
) -> Tuple[int, Optional[Dict]]:
    case_key = f"{case.get('dialog_id')}_{case.get('turn_index')}"
    for attempt in range(1, retry_limit + 1):
        try:
            if test_mode:
                print(f"\n{'='*60}")
                print(f"Case {index}: {case_key}")
                print(f"Dataset: {case.get('_dataset')}")
                print(f"Behavior: {case.get('behavior_category', 'N/A')}")
                preview = case.get("conversation_segment", "")
                print(f"Conversation segment preview: {preview[:100]}...")
                print(f"{'='*60}")

            resp = await generate_doctor_response_async(llm, case, semaphore)

            if test_mode:
                print(f"\nResponse: {resp.get('response', '')[:200]}...")
                print(f"{'='*60}\n")

            resp.update(case)  # Include original case data in response

            if is_gemini:
                async with gemini_lock:
                    gemini_counter["count"] += 1
                    if gemini_counter["count"] % 5 == 0:
                        await asyncio.sleep(60)

            return index, resp
        except Exception as e:
            is_rate_limit = "429" in str(e) or "quota" in str(e).lower() or "rate_limit" in str(e).lower()
            if attempt < retry_limit:
                # Exponential backoff: 30s, 60s, 120s, ... for rate limits; 5s flat otherwise
                wait_time = min(30 * (2 ** (attempt - 1)), 120) if is_rate_limit else 5
                print(f"  -> Error (retry {attempt}/{retry_limit}) for {case_key}: {str(e)[:80]}")
                await asyncio.sleep(wait_time)
                continue
            # Final attempt also failed
            print(f"  -> Error after {retry_limit} retries. Skipping case {case_key}")
            return index, None
    # Should never reach here, but guard against implicit None return
    return index, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LLM doctor responses for safety benchmark cases.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
  API keys can be provided via:
  1. Command line arguments (--openai-key, --claude-key, --gemini-key, --deepseek-key)
  2. Environment variables:
     - OPENAI_API_KEY
     - ANTHROPIC_API_KEY
     - GEMINI_API_KEY
     - DEEPSEEK_API_KEY
     - VLLM_API_BASE (for vLLM models, e.g., http://your-server:8000/v1)
  
  Examples:
    # Test mode with Qwen model
    python generate_response.py --dataset ACI --model Qwen/Qwen3-8B --test --vllm-api-base http://your-server:8000/v1
    
    # DeepSeek with Thinking mode
    python generate_response.py --dataset ACI --model deepseek-reasoner --deepseek-key YOUR_KEY
    
    # DeepSeek with No Thinking mode
    python generate_response.py --dataset ACI --model deepseek-chat --deepseek-key YOUR_KEY
    
    # Qwen3 with Thinking mode
    python generate_response.py --dataset ACI --model Qwen/Qwen3-32B-Instruct --enable-thinking --vllm-api-base http://your-server:8000/v1
    
    # Qwen3 with No Thinking mode (default)
    python generate_response.py --dataset ACI --model Qwen/Qwen3-32B-Instruct --vllm-api-base http://your-server:8000/v1
    
    # Full run
    python generate_response.py --dataset ACI --model Qwen/Qwen3-8B --vllm-api-base http://your-server:8000/v1
        """
    )
    parser.add_argument("--dataset", type=str, required=True, choices=[
        "ACI", "MediTOD","MedDG", "IMCS",  "all", "abnormal_values", 
        "self_diagnosis", "factual_inaccuracy"], help="Which dataset to use")
    parser.add_argument("--input_file", type=str, default=None, help="Optional input file path (JSON format) for cases")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., gpt-4, gpt-5, claude-3-opus-20240229, Qwen/Qwen3-8B)")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit of cases to process")
    parser.add_argument("--test", action="store_true", help="Test/debug mode: process only 5 cases with detailed logging")
    
    # API key arguments (optional, will fall back to environment variables)
    parser.add_argument("--openai-key", type=str, default=None, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--claude-key", type=str, default=None, help="Anthropic Claude API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--gemini-key", type=str, default=None, help="Google Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--deepseek-key", type=str, default=None, help="DeepSeek API key (or set DEEPSEEK_API_KEY env var)")
    parser.add_argument("--vllm-api-base", type=str, default=None, help="vLLM API base URL (e.g., http://your-server:8000/v1) or set VLLM_API_BASE env var")
    parser.add_argument("--enable-thinking", action="store_true", help="Enable thinking mode for Qwen3 models (default: False, non-thinking mode)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for generated JSONL files (default: model_output_syn_{dataset}_v1/)")
    
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Number of cases to run in parallel for a single model.",
    )

    return parser.parse_args()


async def run_async(args: argparse.Namespace) -> None:
    datasets = ["ACI", "MediTOD"] if args.dataset == "all" else [args.dataset]

    # Collect API keys from command line arguments
    api_keys = {}
    if args.openai_key:
        api_keys["openai"] = args.openai_key
    if args.claude_key:
        api_keys["claude"] = args.claude_key
    if args.gemini_key:
        api_keys["gemini"] = args.gemini_key
    if args.deepseek_key:
        api_keys["deepseek"] = args.deepseek_key
    
    llm = build_model(args.model, api_keys, vllm_api_base=args.vllm_api_base, enable_thinking=args.enable_thinking)

    cases = load_cases(datasets, file_path=args.input_file)
    
    # Test mode: limit to 5 cases unless --limit is explicitly set
    if args.test and args.limit is None:
        cases = cases[:5]
        print(f"🧪 TEST MODE: Processing only {len(cases)} cases for testing")
    elif args.limit:
        cases = cases[: args.limit]

    print(f"Processing {len(cases)} cases with model {args.model} ...")
    results: List[Optional[Dict]] = [None] * len(cases)
    is_gemini = args.model.lower().startswith("gemini")
    max_retries = 3

    semaphore = asyncio.Semaphore(args.concurrency)
    gemini_lock = asyncio.Lock()
    gemini_counter = {"count": 0}

    tasks = [
        asyncio.create_task(
            process_case(
                index,
                case,
                llm,
                semaphore,
                is_gemini,
                max_retries,
                gemini_lock,
                gemini_counter,
                args.test,
            )
        )
        for index, case in enumerate(cases, start=1)
    ]

    task_results = await asyncio.gather(*tasks)
    for completed, (index, resp) in enumerate(task_results, start=1):
        if resp is not None:
            results[index - 1] = resp
        if completed % 10 == 0 or completed == 1 or completed == len(cases):
            print(f"  -> completed {completed}/{len(cases)}")

    filtered_results = [r for r in results if r is not None]
    save_results(filtered_results, args.model, datasets, enable_thinking=args.enable_thinking, output_dir=args.output_dir)


def main() -> None:
    asyncio.run(run_async(parse_args()))


if __name__ == "__main__":
    main()
