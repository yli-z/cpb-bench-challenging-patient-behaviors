#!/usr/bin/env python3
"""
批量生成多个模型的医生回复。

用法:
    python batch_generate.py --dataset ACI
    python batch_generate.py --dataset IMCS --test
    python batch_generate.py --dataset all --limit 10
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict

# 默认模型列表
DEFAULT_MODELS = [
    "claude-sonnet-4-5-20250929",
    "deepseek-chat",
    "deepseek-reasoner",
    "gemini-2.5-flash",
    "gpt-4",
    "gpt-5",
    "gpt-4o-mini",
]

# 模型特定的配置
MODEL_CONFIGS = {
    "claude-sonnet-4-5-20250929": {
        "api_key_env": "ANTHROPIC_API_KEY",
        "enable_thinking": False,
    },
    "deepseek-chat": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "enable_thinking": False,
    },
    "deepseek-reasoner": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "enable_thinking": False,  # deepseek-reasoner 本身就是 thinking 模式
    },
    "gemini-2.5-flash": {
        "api_key_env": "GEMINI_API_KEY",
        "enable_thinking": False,
    },
    "gpt-4": {
        "api_key_env": "OPENAI_API_KEY",
        "enable_thinking": False,
    },
    "gpt-5": {
        "api_key_env": "OPENAI_API_KEY",
        "enable_thinking": False,
    },
    "gpt-4o-mini": {
        "api_key_env": "OPENAI_API_KEY",
        "enable_thinking": False,
    },
}


def check_api_keys(models: List[str]) -> Dict[str, bool]:
    """Check if the required API keys are set"""
    import os
    
    required_keys = set()
    for model in models:
        if model in MODEL_CONFIGS:
            key_env = MODEL_CONFIGS[model]["api_key_env"]
            required_keys.add(key_env)
    
    available = {}
    for key_env in required_keys:
        available[key_env] = os.getenv(key_env) is not None
    
    return available


def load_env_from_zshrc():
    """Try to load environment variables from .zshrc file"""
    import os
    import re
    
    zshrc_path = Path.home() / ".zshrc"
    if not zshrc_path.exists():
        return {}
    
    env_vars = {}
    try:
        with open(zshrc_path, "r") as f:
            for line in f:
                # Match export KEY="value" or export KEY='value' or export KEY=value
                match = re.match(r'^\s*export\s+(\w+)=["\']?([^"\']+)["\']?\s*$', line.strip())
                if match:
                    key, value = match.groups()
                    env_vars[key] = value
    except Exception as e:
        print(f"Warning: Could not read .zshrc: {e}")
    
    return env_vars


def run_model_generation(
    dataset: str,
    model: str,
    limit: int = None,
    test: bool = False,
    vllm_api_base: str = None,
) -> bool:
    """Run a single model generation task"""
    import os
    
    script_path = Path(__file__).parent / "generate_response.py"
    
    cmd = [
        sys.executable,
        str(script_path),
        "--dataset",
        dataset,
        "--model",
        model,
    ]
    
    if limit:
        cmd.extend(["--limit", str(limit)])
    
    if test:
        cmd.append("--test")
    
    if vllm_api_base:
        cmd.extend(["--vllm-api-base", vllm_api_base])
    
    # Prepare environment variable dictionary (to pass to subprocess)
    env = os.environ.copy()
    
    # Add API key parameter (if model configuration has)
    if model in MODEL_CONFIGS:
        config = MODEL_CONFIGS[model]
        key_env = config["api_key_env"]
        
        # First try to get from current environment variables
        api_key = os.getenv(key_env)
        
        # If not in current environment variables, try to load from .zshrc
        if not api_key:
            zshrc_env = load_env_from_zshrc()
            api_key = zshrc_env.get(key_env)
            if api_key:
                # Set to current process environment variables, so subprocess can inherit
                env[key_env] = api_key
                print(f"   Loaded {key_env} from .zshrc")
        
        # If found API key, pass through command line argument (higher priority)
        if api_key:
            if key_env == "OPENAI_API_KEY":
                cmd.extend(["--openai-key", api_key])
            elif key_env == "ANTHROPIC_API_KEY":
                cmd.extend(["--claude-key", api_key])
            elif key_env == "GEMINI_API_KEY":
                cmd.extend(["--gemini-key", api_key])
            elif key_env == "DEEPSEEK_API_KEY":
                cmd.extend(["--deepseek-key", api_key])
        else:
            print(f"   Warning: {key_env} not found in environment or .zshrc")
    
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real time
            env=env,  # Pass environment variables to subprocess
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\nError running model {model}: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\nInterrupted while running model {model}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch generate doctor responses for multiple models"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["ACI", "MediTOD", "IMCS", "MedDG", "all"],
        help="Dataset to process",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="List of models to run (default: all models)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of cases to process (for testing)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: each model only processes 5 cases",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip completed models (check if output file exists)",
    )
    parser.add_argument(
        "--vllm-api-base",
        type=str,
        default=None,
        help="vLLM API base URL (for Qwen models)",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=5,
        help="模型之间的延迟时间（秒，默认：5）",
    )
    
    args = parser.parse_args()
    
    # 确定要运行的模型列表
    models = args.models if args.models else DEFAULT_MODELS
    
    # 检查 API keys
    print("🔑 Checking API keys...")
    available_keys = check_api_keys(models)
    missing_keys = [k for k, v in available_keys.items() if not v]
    if missing_keys:
        print(f"⚠️  Warning: Missing API keys: {', '.join(missing_keys)}")
        print("   Some models may fail if API keys are not set.")
    else:
        print("✅ All required API keys are set.")
    
    # 确定数据集列表
    if args.dataset == "all":
        datasets = ["ACI", "MediTOD", "IMCS", "MedDG"]
    else:
        datasets = [args.dataset]
    
    # 检查输出目录
    output_dir = Path(__file__).parent / "model_output"
    output_dir.mkdir(exist_ok=True)
    
    # 统计信息
    total_models = len(models) * len(datasets)
    completed = 0
    failed = 0
    skipped = 0
    
    print(f"\n🚀 Starting batch generation")
    print(f"   Models: {len(models)}")
    print(f"   Datasets: {len(datasets)}")
    print(f"   Total tasks: {total_models}")
    if args.test:
        print(f"   Mode: TEST (5 cases per model)")
    if args.limit:
        print(f"   Limit: {args.limit} cases per model")
    print()
    
    # 处理每个数据集和模型
    for dataset in datasets:
        print(f"\n{'#'*80}")
        print(f"# Processing dataset: {dataset}")
        print(f"{'#'*80}\n")
        
        for model in models:
            # 检查是否跳过已完成的
            if args.skip_completed:
                output_file = output_dir / f"{model.replace(':', '_').replace('/', '_')}_{dataset}.jsonl"
                if output_file.exists():
                    print(f"⏭️  Skipping {model} on {dataset} (output file exists)")
                    skipped += 1
                    continue
            
            # 运行模型生成
            success = run_model_generation(
                dataset=dataset,
                model=model,
                limit=args.limit,
                test=args.test,
                vllm_api_base=args.vllm_api_base,
            )
            
            if success:
                completed += 1
                print(f"✅ Completed: {model} on {dataset}")
            else:
                failed += 1
                print(f"❌ Failed: {model} on {dataset}")
            
            # 模型之间的延迟（除了最后一个）
            if model != models[-1] or dataset != datasets[-1]:
                if args.delay > 0:
                    print(f"⏳ Waiting {args.delay} seconds before next model...")
                    time.sleep(args.delay)
    
    # 打印总结
    print(f"\n{'='*80}")
    print("📊 Batch Generation Summary")
    print(f"{'='*80}")
    print(f"✅ Completed: {completed}")
    print(f"❌ Failed: {failed}")
    if args.skip_completed:
        print(f"⏭️  Skipped: {skipped}")
    print(f"📁 Output directory: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import os
    main()

