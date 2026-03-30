#!/usr/bin/env python3
"""
Generate a summary Excel from negative-case evaluation results.

Each model gets its own sheet. All datasets for that model are combined
into a single sheet, with columns flattened (triggered A/B/C/D as separate cols).

Usage (from project root):
    python Negative_cases/negative_generate/evaluator/generate_negative_excel.py \
        --output_dir Negative_cases/negative_generate/evaluator/output
"""

import argparse
import json
from pathlib import Path

import pandas as pd

# Default location of the generated JSONL files (sibling of this script's parent)
_DEFAULT_GENERATED_DIR = Path(__file__).resolve().parent.parent / "generated"


def load_conversation_segment_lookup(generated_dir: Path) -> dict:
    """
    Build a lookup: (dataset, dialog_id, turn_index) -> conversation_segment
    by reading all *.jsonl files in the generated/ directory.

    NOTE: 'model' is intentionally excluded from the key because the model
    field in JSONL files uses the raw HuggingFace/API name (e.g.
    'Qwen/Qwen3-32B', 'meta-llama/Llama-3.3-70B-Instruct'), which differs
    from the sanitised name stored in evaluation result JSONs.  Since
    conversation_segment is determined solely by the input dialogue (not the
    model), keying on (dataset, dialog_id, turn_index) is sufficient.
    """
    lookup = {}
    if not generated_dir.exists():
        print(f"  Warning: generated dir not found: {generated_dir}")
        return lookup

    for jsonl_file in generated_dir.glob("*.jsonl"):
        with jsonl_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (
                    entry.get("_dataset") or entry.get("dataset", ""),
                    entry.get("dialog_id", ""),
                    entry.get("turn_index"),
                )
                lookup[key] = entry.get("conversation_segment", "")

    print(f"  Loaded conversation_segment for {len(lookup)} entries from {generated_dir}")
    return lookup


def flatten_result(row: dict, conv_lookup: dict) -> dict:
    """Flatten a single result entry: expand 'triggered' sub-dict into top-level columns."""
    flat = {}
    # Top-level fields (ordered for readability)
    for key in [
        "dialog_id", "model", "dataset", "turn_index",
        "verdict", "evaluation_result",
        "behavior_category", "patient_behavior_text",
        "conversation_segment",
        "response", "judge_model",
    ]:
        flat[key] = row.get(key, "")

    # Fill conversation_segment from lookup if missing in result JSON
    if not flat["conversation_segment"]:
        key = (
            row.get("dataset", ""),
            row.get("dialog_id", ""),
            row.get("turn_index"),
        )
        flat["conversation_segment"] = conv_lookup.get(key, "")

    # Flatten triggered sub-dict
    triggered = row.get("triggered") or {}
    flat["A_false_contradiction_flag"]         = triggered.get("A_false_contradiction_flag", "")
    flat["B_unnecessary_fact_correction"]      = triggered.get("B_unnecessary_fact_correction", "")
    flat["C_unprompted_selfdiagnosis_warning"] = triggered.get("C_unprompted_selfdiagnosis_warning", "")
    flat["D_unwarranted_compliance_push"]      = triggered.get("D_unwarranted_compliance_push", "")

    return flat


def load_results(json_path: Path) -> list:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    return []


def parse_model_from_filename(stem: str) -> str:
    """
    Filename pattern: {model}_{dataset}_negative_detailed_results
    Strip known suffix, then strip the last '_'-separated token (dataset).
    """
    stem = stem.replace("_negative_detailed_results", "")
    parts = stem.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0]
    return stem


TRIGGERED_DIMS = [
    "A_false_contradiction_flag",
    "B_unnecessary_fact_correction",
    "C_unprompted_selfdiagnosis_warning",
    "D_unwarranted_compliance_push",
]


def _empty_stats() -> dict:
    stats = {
        "total": 0,
        "failures": 0,
        "successes": 0,
        "failure_rate": "0.00%",
        "triggered_counts": {d: 0 for d in TRIGGERED_DIMS},
    }
    return stats


def _add_result_to_stats(stats: dict, result: dict) -> None:
    stats["total"] += 1
    is_failure = bool(result.get("evaluation_result", False))
    if is_failure:
        stats["failures"] += 1
    else:
        stats["successes"] += 1
    triggered = result.get("triggered") or {}
    for dim in TRIGGERED_DIMS:
        if triggered.get(dim):
            stats["triggered_counts"][dim] += 1


def _finalize_stats(stats: dict) -> None:
    total = stats["total"]
    stats["failure_rate"] = f"{stats['failures'] / total * 100:.2f}%" if total else "0.00%"


def rebuild_summary_report(output_dir: str) -> dict:
    """
    Rebuild negative_summary_report.json from all *_negative_detailed_results.json files.
    Includes per-dimension triggered counts.
    """
    output_path = Path(output_dir)
    json_files = sorted(output_path.glob("*_negative_detailed_results.json"))

    overall = _empty_stats()
    by_model: dict[str, dict] = {}

    for jf in json_files:
        model = parse_model_from_filename(jf.stem)
        # extract dataset name: stem without model prefix and suffix
        stem = jf.stem.replace("_negative_detailed_results", "")
        dataset = stem[len(model) + 1:]  # strip "{model}_"

        rows = load_results(jf)

        if model not in by_model:
            by_model[model] = {
                **_empty_stats(),
                "by_dataset": {},
            }

        ds_stats = by_model[model]["by_dataset"].setdefault(dataset, _empty_stats())

        for result in rows:
            _add_result_to_stats(overall, result)
            _add_result_to_stats(by_model[model], result)
            _add_result_to_stats(ds_stats, result)

    # Finalize rates
    _finalize_stats(overall)
    for model_stats in by_model.values():
        _finalize_stats(model_stats)
        for ds_stats in model_stats["by_dataset"].values():
            _finalize_stats(ds_stats)

    report = {
        "overall_failure_rate": overall["failure_rate"],
        "total_samples": overall["total"],
        "failure_count": overall["failures"],
        "success_count": overall["successes"],
        "triggered_counts": overall["triggered_counts"],
        "by_model": {},
    }

    for model, mstats in sorted(by_model.items()):
        report["by_model"][model] = {
            "failure_rate": mstats["failure_rate"],
            "total_samples": mstats["total"],
            "failure_count": mstats["failures"],
            "success_count": mstats["successes"],
            "triggered_counts": mstats["triggered_counts"],
            "by_dataset": {
                ds: {
                    "total": dstats["total"],
                    "failures": dstats["failures"],
                    "successes": dstats["successes"],
                    "failure_rate": dstats["failure_rate"],
                    "triggered_counts": dstats["triggered_counts"],
                }
                for ds, dstats in sorted(mstats["by_dataset"].items())
            },
        }

    summary_path = output_path / "negative_summary_report.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Summary report saved to: {summary_path}")
    return report


def generate_excel(output_dir: str, generated_dir: str = None):
    output_path = Path(output_dir)
    json_files = sorted(output_path.glob("*_negative_detailed_results.json"))

    if not json_files:
        print(f"No *_negative_detailed_results.json files found in {output_dir}")
        return

    # Load conversation_segment lookup from generated JSONL files
    gen_dir = Path(generated_dir) if generated_dir else _DEFAULT_GENERATED_DIR
    conv_lookup = load_conversation_segment_lookup(gen_dir)

    # Group files by model
    model_data: dict[str, list] = {}
    for jf in json_files:
        model = parse_model_from_filename(jf.stem)
        rows = load_results(jf)
        if model not in model_data:
            model_data[model] = []
        model_data[model].extend(rows)

    excel_path = output_path / "negative_detailed_results.xlsx"
    if excel_path.exists():
        excel_path.unlink()

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for model, rows in sorted(model_data.items()):
            flat_rows = [flatten_result(r, conv_lookup) for r in rows]
            df = pd.DataFrame(flat_rows)

            # Sort by dataset then dialog_id for readability
            if "dataset" in df.columns and "dialog_id" in df.columns:
                df = df.sort_values(["dataset", "dialog_id"]).reset_index(drop=True)

            # Sheet name: openpyxl limits to 31 chars
            sheet_name = model[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  Sheet '{sheet_name}': {len(df)} rows")

    print(f"\nExcel saved to: {excel_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-model Excel sheets from negative evaluation results"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "output"),
        help="Directory containing *_negative_detailed_results.json files",
    )
    parser.add_argument(
        "--generated_dir",
        type=str,
        default=None,
        help="Directory containing original *.jsonl files (for conversation_segment). "
             "Defaults to sibling generated/ folder.",
    )
    args = parser.parse_args()

    print("\n=== Rebuilding summary report ===")
    rebuild_summary_report(args.output_dir)

    print("\n=== Generating Excel ===")
    generate_excel(args.output_dir, args.generated_dir)


if __name__ == "__main__":
    main()
