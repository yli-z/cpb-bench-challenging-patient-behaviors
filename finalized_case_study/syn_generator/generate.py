import argparse
import asyncio
import csv
import json
import re
from pathlib import Path

from models.model_utils import create_model
from syn_generator.prompts import build_rewrite_prompt


MAX_RETRIES = 10

def load_seed(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_items(path: Path, case_type: str) -> list:
    items = []
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if case_type == "abnormal_values":
                label_key = "item"
                value_key = "value"
            elif case_type == "information_contradiction":
                label_key = "statement_1"
                value_key = "statement_2"
            else:
                raise ValueError(f"Unknown case type for loading items: {case_type}")
            for row in reader:
                label = row[label_key].strip()
                value = row[value_key].strip()
                if not label or not value:
                    continue
                items.append({
                    "label": label,
                    "value": value,
                    "raw": f"{label}: {value}",
                })

        return items
    
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            if ":" in text:
                label, value = text.split(":", 1)
                items.append({
                    "label": label.strip(),
                    "value": value.strip(),
                    "raw": text,
                })
            else:
                items.append({
                    "label": text,
                    "value": None,
                    "raw": text,
                })
    return items

def slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_")
    return cleaned.lower()

def detect_case_type(seed_path: Path) -> str:
    normalized = seed_path.as_posix().lower()
    if "information_contradiction" in normalized or "information-contradiction" in normalized or "information contradiction" in normalized:
        return "information_contradiction"
    if "abnormal_values" in normalized or "abnormal_value" in normalized:
        return "abnormal_values"
    raise ValueError(f"Could not detect case type from seed path: {seed_path}")

def generate_line(model, user_prompt: str) -> str:
    messages = [{"role": "user", "content": user_prompt}]

    if hasattr(model, "generate_text_response"):
        response = model.generate_text_response(messages)
    else:
        response = model.generate_response(messages)

    return response.strip()

async def generate_line_async(
    model,
    user_prompt: str,
    semaphore: asyncio.Semaphore,
) -> str:
    messages = [{"role": "user", "content": user_prompt}]

    async with semaphore:
        if hasattr(model, "async_generate_response"):
            response = await model.async_generate_response(messages)
            return (response or "").strip()
        return await asyncio.to_thread(generate_line, model, user_prompt)

def format_conversation(conversation: list) -> str:
    if type(conversation) is str:
        return conversation
    lines = []
    for turn in conversation:
        if "Doctor" in turn:
            lines.append(f"Doctor: {turn['Doctor']}")
        elif "Patient" in turn:
            lines.append(f"Patient: {turn['Patient']}")
    return "\n".join(lines)

def parse_conversation_text(text: str) -> list:
    parsed = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Doctor:"):
            parsed.append(("Doctor", stripped[len("Doctor:"):].strip()))
        elif stripped.startswith("Patient:"):
            parsed.append(("Patient", stripped[len("Patient:"):].strip()))
    return parsed

def apply_conversation_text(parsed_lines: list) -> list:
    updated = []
    for idx, (role, content) in enumerate(parsed_lines, start=1):
        if role == "Doctor":
            updated.append({"Doctor": content, "turn index": idx})
        elif role == "Patient":
            updated.append({"Patient": content, "turn index": idx})
        else:
            raise ValueError(f"Unexpected role in parsed conversation: {role}")
    return updated

# for quality check
def prompt_user_accept_case(
    index: int,
    label: str,
    value: str,
    updated_segment: list,
    patient_behavior_text: str,
) -> bool:
    print("\n" + "-" * 80)
    print(f"Case {index} | item: {label} / {value}")
    print("Patient behavior:", patient_behavior_text)
    print("Conversation:\n" + format_conversation(updated_segment))
    while True:
        choice = input("Accept this case? [Y/n] (r=regenerate, q=quit): ").strip().lower()
        if choice in ("", "y", "yes"):
            return True
        if choice in ("n", "no", "r", "regen", "regenerate"):
            return False
        if choice in ("q", "quit", "exit"):
            raise KeyboardInterrupt("User aborted during manual review.")
        print("Please enter y, n, r, or q.")

async def rewrite_conversation(
    model,
    conversation_segment: list,
    label: str,
    value: str,
    semaphore: asyncio.Semaphore,
    case_type: str,
) -> str:
    user_prompt = build_rewrite_prompt(
        format_conversation(conversation_segment),
        case_type,
        label,
        value,
    )
    response = await generate_line_async(model, user_prompt, semaphore)
    # breakpoint()
    return response.strip()

async def generate_case(
    seed: dict,
    item: dict,
    index: int,
    model,
    semaphore: asyncio.Semaphore,
    case_type: str,
    manual_review: bool = False,
) -> tuple[int, dict]:
    label = item["label"]
    value = item["value"]
    slug = slugify(label) # just for file naming
    new_case = None

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"Generating case {index} with item: {label} / {value} (attempt {attempt})")
        try:
            revised_text = await rewrite_conversation(
                model,
                seed["conversation_segment"],
                label,
                value,
                semaphore,
                case_type,
            )
            parsed_lines = parse_conversation_text(revised_text)
            updated_segment = apply_conversation_text(parsed_lines)
            patient_behavior_text = ""
            patient_turn_index = None
            for line_index, (role, content) in enumerate(parsed_lines, start=1):
                if role == "Patient":
                    patient_behavior_text = content
                    patient_turn_index = line_index
            if not patient_behavior_text or patient_turn_index is None:
                raise ValueError("Missing Patient line in revised conversation.")
                
            if case_type == "abnormal_values":
            #     # Verify that the abnormal value (value) is mentioned in patient behavior text
                if "Doctor" not in updated_segment[-2]:
                    raise ValueError("The second last turn in the conversation is not a Doctor turn.")
            #
                doctor_turn = updated_segment[-2]['Doctor']
            #     if value.lower() not in patient_behavior_text.lower():
            #         breakpoint()
            #         raise ValueError(
            #             f"Patient behavior text does not mention abnormal value ({value}): {patient_behavior_text}"
            #         )
                if label.lower() not in doctor_turn.lower():

                    if label.lower() == "blood ph minimum":
                        if "blood ph" in doctor_turn.lower():
                            if "low" in doctor_turn.lower() or "minimum" in doctor_turn.lower() or "lowest" in doctor_turn.lower():
                                break
                    raise ValueError(
                        f"Patient behavior text does not mention clinical item ({label}): {doctor_turn}"
                    )
            # Add later: the last turn should be patient turn
            if parsed_lines[-1][0] != "Patient":
                raise ValueError("The last turn in the conversation is not a Patient turn.")
            new_case = dict(seed)
            new_case["case_id"] = f"{seed['case_id']}_syn_{index:03d}_{slug}"
            new_case["dialog_id"] = f"{seed['dialog_id']}_syn_{index:03d}_{slug}"
            new_case["patient_behavior_text"] = patient_behavior_text
            new_case["turn_index"] = patient_turn_index
            new_case["conversation_segment"] = updated_segment
            new_case.pop("complete_conversation")

            if manual_review:
                if prompt_user_accept_case(
                    index,
                    label,
                    value,
                    updated_segment,
                    patient_behavior_text,
                ):
                    break
                print("Manual review rejected; regenerating.", flush=True)
                new_case = None
                continue
            break
        except ValueError:
            if attempt == MAX_RETRIES:
                raise
            continue

    if new_case is None:
        raise ValueError("Failed to generate an acceptable case.")
    return index, new_case

async def generate_case_with_slot(
    slot: int,
    seed: dict,
    item: dict,
    index: int,
    model,
    semaphore: asyncio.Semaphore,
    case_type: str,
    manual_review: bool = False,
) -> tuple[int, int, dict]:
    case_index, case = await generate_case(
        seed,
        item,
        index,
        model,
        semaphore,
        case_type,
        manual_review=manual_review,
    )
    return slot, case_index, case

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic conversations from a seed case.")
    parser.add_argument(
        "--llm-model",
        required=True,
        help="LLM model name used to rewrite the doctor/patient turns.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max tokens for LLM rewrite outputs.",
    )
    parser.add_argument(
        "--seed",
        default=None,
        help="Path to the seed JSON file.",
    )
    parser.add_argument(
        "--items",
        default=None,
        help="Path to the abnormal clinical items list.",
    )
    parser.add_argument(
        "--output",
        default="syn_generator/output/synthetic_seed.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of items to generate.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=25,
        help="Number of concurrent LLM rewrite requests.",
    )
    parser.add_argument(
        "--manual-review",
        action="store_true",
        help="Interactively review each generated case and reject to regenerate.",
    )
    return parser.parse_args()


async def run_async(args: argparse.Namespace) -> None:
    items_path = Path(args.items)
    output_path = Path(args.output)

    seed_path = Path(args.seed)
    case_type = detect_case_type(seed_path)
    seeds = [load_seed(seed_path)]
    items = load_items(items_path, case_type)

    model = create_model(
        model_name=args.llm_model,
        generation_config={"max_tokens": args.max_tokens},
    )

    if args.limit is not None:
        items = items[: args.limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    per_seed_count = len(items)
    total = per_seed_count * len(seeds)
    cases: list[dict] = []
    completed = 0
    if args.manual_review:
        if args.concurrency != 1:
            print("Manual review enabled: overriding --concurrency to 1.", flush=True)
        semaphore = asyncio.Semaphore(1)
        for seed in seeds:
            seed_items = items[:per_seed_count]
            for i, item in enumerate(seed_items, start=1):
                _, case = await generate_case(
                    seed,
                    item,
                    i,
                    model,
                    semaphore,
                    case_type,
                    manual_review=True,
                )
                cases.append(case)
                completed += 1
                print(f"[{completed}/{total}] generated", flush=True)
    else:
        cases = [None] * total
        semaphore = asyncio.Semaphore(args.concurrency)
        tasks = []
        slot_index = 0
        for seed in seeds:
            seed_items = items[:per_seed_count]
            for i, item in enumerate(seed_items, start=1):
                tasks.append(
                    asyncio.create_task(
                        generate_case_with_slot(
                            slot_index,
                            seed,
                            item,
                            i,
                            model,
                            semaphore,
                            case_type,
                            manual_review=False,
                        )
                    )
                )
                slot_index += 1
        for task in asyncio.as_completed(tasks):
            slot, _, case = await task
            cases[slot] = case
            completed += 1
            print(f"[{completed}/{total}] generated", flush=True)

    payload = {
        "source_seed": str(seed_path),
        "generated_model": args.llm_model,
        "items_file": str(items_path),
        "seed_category": case_type,
        "cases": cases,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(cases)} cases to {output_path}")


def main() -> None:
    asyncio.run(run_async(parse_args()))

if __name__ == "__main__":
    main()
