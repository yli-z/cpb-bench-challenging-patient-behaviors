#!/usr/bin/env python3
"""
为每个模型单独生成 Excel 文件，每个 behavior_category 一个 sheet
用于人工标注
"""

import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evaluators_single.scripts.utils import load_model_results_json


def load_single_model_results(json_file_path):
    """加载单个模型的 JSON 结果文件"""
    json_path = Path(json_file_path)
    if not json_path.exists():
        print(f"❌ 错误: JSON 文件不存在: {json_file_path}")
        return None, None
    
    # 提取模型名称
    model_name = json_path.stem.replace("_detailed_results", "")
    print(f"  加载模型: {model_name}")
    
    # 加载 JSON 数据
    data = load_model_results_json(str(json_path))
    
    # 展平数据：将 {dataset: [results]} 转换为单个结果列表
    all_results = []
    for dataset, results in data.items():
        for result in results:
            # 添加 model 和 dataset 字段（如果不存在）
            result["model"] = model_name
            result["dataset"] = dataset
            all_results.append(result)
    
    return model_name, all_results


def group_by_behavior_category(results):
    """按 behavior_category 分组结果"""
    grouped = defaultdict(list)
    
    for result in results:
        behavior_category = result.get("behavior_category", "Unknown")
        grouped[behavior_category].append(result)
    
    return grouped


def prepare_row(result):
    """准备一行数据，按照指定的列顺序"""
    row = {
        "model": result.get("model", ""),
        "dataset": result.get("dataset", ""),
        "dialog_id": result.get("dialog_id", ""),
        "turn_index": result.get("turn_index", 0),
        "evaluation_result": result.get("evaluation_result", False),
        "patient_behavior_text": result.get("patient_behavior_text", ""),
        "response": result.get("response", ""),
        "conversation_segment": result.get("conversation_segment", ""),
        "human check": ""  # 留空，供人工检查
    }
    return row


def generate_excel_for_model(json_file_path, output_dir):
    """为单个模型生成按 behavior_category 分 sheet 的 Excel 文件"""
    json_path = Path(json_file_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("为单个模型生成按 behavior_category 分 sheet 的 Excel")
    print("="*60)
    print(f"JSON 文件: {json_path}")
    print(f"输出目录: {output_dir}")
    print("="*60)
    print()
    
    # 加载模型数据
    model_name, results = load_single_model_results(json_file_path)
    if not results:
        print("❌ 未能加载数据")
        return None
    
    print(f"  总共加载了 {len(results)} 条结果")
    
    # 按 behavior_category 分组
    grouped_by_category = group_by_behavior_category(results)
    
    print(f"\n找到 {len(grouped_by_category)} 个 behavior_category:")
    for category, cat_results in sorted(grouped_by_category.items()):
        print(f"  {category}: {len(cat_results)} 条")
    
    # 定义列顺序
    column_order = [
        "model",
        "dataset",
        "dialog_id",
        "turn_index",
        "evaluation_result",
        "patient_behavior_text",
        "response",
        "conversation_segment",
        "human check"
    ]
    
    # 生成 Excel 文件名
    excel_filename = f"{model_name}_by_category.xlsx"
    excel_path = output_path / excel_filename
    
    # 生成 Excel
    print(f"\n生成 Excel 文件: {excel_filename}")
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for category, cat_results in sorted(grouped_by_category.items()):
            # 准备数据
            rows = [prepare_row(result) for result in cat_results]
            df = pd.DataFrame(rows)
            
            # 确保列顺序正确
            df = df[column_order]
            
            # 工作表名称（Excel 限制为 31 个字符）
            sheet_name = category[:31] if len(category) > 31 else category
            
            # 写入 Excel
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            print(f"  ✅ {category}: {len(cat_results)} 条 → Sheet: {sheet_name}")
    
    print(f"\n{'='*60}")
    print(f"✅ Excel 报告已生成: {excel_path}")
    print(f"{'='*60}")
    print(f"\n统计信息:")
    print(f"  - 模型: {model_name}")
    print(f"  - Behavior Category 数量: {len(grouped_by_category)}")
    total_rows = sum(len(cat_results) for cat_results in grouped_by_category.values())
    print(f"  - 总行数: {total_rows}")
    print(f"  - Excel 文件大小: {excel_path.stat().st_size / 1024 / 1024:.2f} MB")
    print()
    
    return excel_path


def generate_excel_for_all_models(input_dir, output_dir):
    """为输入目录中的所有模型生成 Excel 文件"""
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ 错误: 输入目录不存在: {input_dir}")
        return []
    
    # 查找所有 JSON 结果文件
    json_files = list(input_path.glob("*_detailed_results.json"))
    if not json_files:
        print(f"⚠️  警告: 在 {input_dir} 中未找到任何 JSON 结果文件")
        return []
    
    print("="*60)
    print("批量为所有模型生成 Excel 文件")
    print("="*60)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"找到 {len(json_files)} 个 JSON 结果文件")
    print("="*60)
    print()
    
    generated_files = []
    for json_file in sorted(json_files):
        try:
            excel_path = generate_excel_for_model(str(json_file), output_dir)
            if excel_path:
                generated_files.append(excel_path)
        except Exception as e:
            print(f"❌ 处理 {json_file.name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("="*60)
    print(f"✅ 批量生成完成！共生成 {len(generated_files)} 个 Excel 文件")
    print("="*60)
    for excel_file in generated_files:
        print(f"  - {excel_file.name}")
    print()
    
    return generated_files


def main():
    parser = argparse.ArgumentParser(
        description="为每个模型单独生成按 behavior_category 分 sheet 的 Excel 文件"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="输入目录（包含 JSON 结果文件）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录（默认与输入目录相同）"
    )
    parser.add_argument(
        "--json_file",
        type=str,
        default=None,
        help="指定单个 JSON 文件（可选，如果不指定则处理所有文件）"
    )
    
    args = parser.parse_args()
    
    # 处理路径
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        # 尝试从项目根目录解析
        project_root = Path(__file__).parent.parent.parent
        potential_path = project_root / input_dir
        if potential_path.exists() or str(input_dir).startswith("evaluators/"):
            input_dir = potential_path.resolve()
        else:
            # 否则相对于当前目录
            input_dir = input_dir.resolve()
    
    # 输出目录默认与输入目录相同
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    if not output_dir.is_absolute():
        project_root = Path(__file__).parent.parent.parent
        potential_path = project_root / output_dir
        if potential_path.exists() or str(output_dir).startswith("evaluators/"):
            output_dir = potential_path.resolve()
        else:
            output_dir = output_dir.resolve()
    
    # 生成 Excel
    try:
        if args.json_file:
            # 处理单个文件
            json_file_path = Path(args.json_file)
            if not json_file_path.is_absolute():
                json_file_path = input_dir / json_file_path
            generate_excel_for_model(str(json_file_path), str(output_dir))
        else:
            # 处理所有文件
            generate_excel_for_all_models(str(input_dir), str(output_dir))
    except Exception as e:
        print(f"\n❌ 生成 Excel 时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
