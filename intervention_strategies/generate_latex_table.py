"""
Generate LaTeX table from intervention strategy evaluation results.

Reads eval_results/{cot,instruction,eval_patient,self_eval}/ and produces
a LaTeX table with failure rates per model, per strategy, per behavior category.

Usage:
    python intervention_strategies/generate_latex_table.py
    python intervention_strategies/generate_latex_table.py --output intervention_strategies/results_table.tex
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_RESULTS_DIR = SCRIPT_DIR / "eval_results"

# Strategy display names (order matters for table rows)
STRATEGIES = [
    ("cot", "CoT"),
    ("instruction", "Instruction"),
    ("eval_patient", "Assessment"),
    ("self_eval", "Self-Review"),
]

# Behavior categories (order matters for table columns)
BEHAVIOR_CATEGORIES = [
    "Information Contradiction",
    "Factual Inaccuracy",
    "Self-diagnosis",
    "Care Resistance",
]

# Short column headers for LaTeX
CATEGORY_SHORT = {
    "Information Contradiction": "IC",
    "Factual Inaccuracy": "FI",
    "Self-diagnosis": "SD",
    "Care Resistance": "CR",
}

# Model display names
MODEL_DISPLAY = {
    "gpt-4o": "GPT-4o",
    "gpt-4o-mini": "GPT-4o-mini",
    "gpt-4": "GPT-4",
    "gpt-5": "GPT-5",
    "claude-sonnet-4-5-20250929": "Claude Sonnet 4.5",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "deepseek-chat": "DeepSeek-V3",
    "deepseek-reasoner": "DeepSeek-R1",
    "meta-llama_Llama-3.3-70B-Instruct": "Llama-3.3-70B",
    "meta-llama_Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "Qwen_Qwen3-32B": "Qwen3-32B",
    "Qwen_Qwen3-32B_thinking": "Qwen3-32B (think)",
}

# Model order for table
MODEL_ORDER = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4",
    "gpt-5",
    "claude-sonnet-4-5-20250929",
    "gemini-2.5-flash",
    "deepseek-chat",
    "deepseek-reasoner",
    "meta-llama_Llama-3.3-70B-Instruct",
    "meta-llama_Meta-Llama-3.1-8B-Instruct",
    "Qwen_Qwen3-32B",
    "Qwen_Qwen3-32B_thinking",
]


def load_eval_results(strategy_dir: str) -> dict:
    """Load all model results from a strategy's eval_results directory.

    Returns: {model_name: [list of result entries across all datasets]}
    """
    results = defaultdict(list)
    strategy_path = Path(strategy_dir)
    if not strategy_path.exists():
        return results

    for json_file in strategy_path.glob("*_detailed_results.json"):
        # Extract model name from filename: {model}_detailed_results.json
        model_name = json_file.stem.replace("_detailed_results", "")
        with open(json_file) as f:
            data = json.load(f)

        # Data is {dataset: [entries]}
        if isinstance(data, dict):
            for dataset, entries in data.items():
                if isinstance(entries, list):
                    results[model_name].extend(entries)
        elif isinstance(data, list):
            results[model_name].extend(data)

    return dict(results)


def compute_failure_rates(entries: list) -> dict:
    """Compute failure rate per behavior category.

    Returns: {category: (failures, total, rate_pct)}
    """
    counts = defaultdict(lambda: {"failures": 0, "total": 0})
    for entry in entries:
        cat = entry.get("behavior_category", "Unknown")
        counts[cat]["total"] += 1
        if entry.get("evaluation_result", False):
            counts[cat]["failures"] += 1

    rates = {}
    for cat, c in counts.items():
        rate = (c["failures"] / c["total"] * 100) if c["total"] > 0 else 0
        rates[cat] = (c["failures"], c["total"], rate)
    return rates


def compute_overall(entries: list, categories: list = None) -> tuple:
    """Compute overall failure rate, optionally filtered to specific categories."""
    if categories:
        entries = [e for e in entries if e.get("behavior_category") in categories]
    total = len(entries)
    failures = sum(1 for e in entries if e.get("evaluation_result", False))
    rate = (failures / total * 100) if total > 0 else 0
    return failures, total, rate


def generate_latex(output_path: str = None):
    # Collect all data: {model: {strategy: {category: (fail, total, rate)}}}
    all_data = {}

    for strategy_key, strategy_display in STRATEGIES:
        strategy_dir = EVAL_RESULTS_DIR / strategy_key
        model_results = load_eval_results(str(strategy_dir))

        for model_name, entries in model_results.items():
            if model_name not in all_data:
                all_data[model_name] = {}
            rates = compute_failure_rates(entries)
            overall = compute_overall(entries, categories=BEHAVIOR_CATEGORIES)
            all_data[model_name][strategy_key] = {
                "by_category": rates,
                "overall": overall,
            }

    # Validate against summary_report.json
    for strategy_key, strategy_display in STRATEGIES:
        summary_path = EVAL_RESULTS_DIR / strategy_key / "summary_report.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        for model_name, model_stats in summary.get("by_model", {}).items():
            if model_name not in all_data or strategy_key not in all_data[model_name]:
                continue
            computed = all_data[model_name][strategy_key]
            # Check per-category counts
            for cat, stats in model_stats.get("by_behavior_category", {}).items():
                if cat not in computed["by_category"]:
                    continue
                fail_computed, total_computed, _ = computed["by_category"][cat]
                assert fail_computed == stats["failures"], (
                    f"Mismatch {strategy_key}/{model_name}/{cat}: "
                    f"computed {fail_computed} failures vs summary {stats['failures']}"
                )
                assert total_computed == stats["total"], (
                    f"Mismatch {strategy_key}/{model_name}/{cat}: "
                    f"computed {total_computed} total vs summary {stats['total']}"
                )
        print(f"  Validated {strategy_key} against summary_report.json")

    # Determine which models are present
    models_present = [m for m in MODEL_ORDER if m in all_data]
    # Add any models not in MODEL_ORDER
    for m in sorted(all_data.keys()):
        if m not in models_present:
            models_present.append(m)

    # Build LaTeX
    lines = []
    lines.append("% Auto-generated by intervention_strategies/generate_latex_table.py")
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{ll" + "r" * len(BEHAVIOR_CATEGORIES) + "r" + r"}")
    lines.append(r"\toprule")

    # Header
    cat_headers = " & ".join(CATEGORY_SHORT[c] for c in BEHAVIOR_CATEGORIES)
    lines.append(rf"\textbf{{Model}} & \textbf{{Strategy}} & {cat_headers} & \textbf{{Overall}} \\")
    lines.append(r"\midrule")

    for model in models_present:
        display_name = MODEL_DISPLAY.get(model, model)
        strategy_data = all_data[model]
        num_strategies = len([s for s, _ in STRATEGIES if s in strategy_data])
        if num_strategies == 0:
            continue

        first = True
        for strategy_key, strategy_display in STRATEGIES:
            if strategy_key not in strategy_data:
                continue

            data = strategy_data[strategy_key]
            by_cat = data["by_category"]
            overall = data["overall"]

            # Format each category cell as failure count
            cells = []
            for cat in BEHAVIOR_CATEGORIES:
                if cat in by_cat:
                    fail, total, rate = by_cat[cat]
                    cells.append(f"{fail}")
                else:
                    cells.append("--")

            overall_str = f"{overall[0]}"
            cell_str = " & ".join(cells)

            if first:
                lines.append(rf"      \multirow{{{num_strategies}}}{{*}}{{{display_name}}}")
                first = False

            lines.append(rf"      & {strategy_display} & {cell_str} & {overall_str} \\")

        lines.append(r"\midrule")

    # Remove last \midrule and replace with \bottomrule
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"

    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Failure rates (\%) by intervention strategy and behavior category.}")
    lines.append(r"\label{tab:intervention_results}")
    lines.append(r"\end{table}")

    output = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(output)
        print(f"LaTeX table saved to {output_path}")
    else:
        print(output)

    # Also print a plain-text summary
    print("\n--- Plain Text Summary ---")
    print(f"{'Model':<30} {'Strategy':<15} ", end="")
    for cat in BEHAVIOR_CATEGORIES:
        print(f"{CATEGORY_SHORT[cat]:>10}", end="")
    print(f"{'Overall':>10}")
    print("-" * 100)

    for model in models_present:
        display_name = MODEL_DISPLAY.get(model, model)
        strategy_data = all_data[model]
        for strategy_key, strategy_display in STRATEGIES:
            if strategy_key not in strategy_data:
                continue
            data = strategy_data[strategy_key]
            by_cat = data["by_category"]
            overall = data["overall"]
            print(f"{display_name:<30} {strategy_display:<15} ", end="")
            for cat in BEHAVIOR_CATEGORIES:
                if cat in by_cat:
                    fail, total, _ = by_cat[cat]
                    print(f"{fail:>8}", end="")
                else:
                    print(f"{'--':>8}", end="")
            print(f"{overall[0]:>8}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LaTeX table from evaluation results.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .tex file path (default: print to stdout)")
    args = parser.parse_args()
    generate_latex(args.output)
