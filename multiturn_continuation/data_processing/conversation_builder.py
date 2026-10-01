"""
Conversation Builder

Builds conversation history and extracts remaining patient turns from source data.
"""

import json
from typing import List, Dict, Optional
import os
import copy


def load_source_data(source_data_dir: str) -> Dict[str, Dict]:
    """
    Load all dataset source files from data_loader/output_benchmark.
    
    Args:
        source_data_dir: Directory containing safety_benchmark.json files
        
    Returns:
        Dictionary mapping dataset name to {dialog_id: case_data}
    """
    datasets = {}
    dataset_names = ['ACI', 'MedDG', 'MediTOD', 'IMCS']
    
    print(f"\n📂 Loading source data from: {source_data_dir}")
    
    for dataset_name in dataset_names:
        json_path = os.path.join(source_data_dir, f"{dataset_name}_safety_benchmark.json")
        
        if not os.path.exists(json_path):
            print(f"   ⚠️  {dataset_name}_safety_benchmark.json not found, skipping")
            continue
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Build index: {dialog_id: case}
            dataset_index = {}
            for case in data.get('cases', []):
                dialog_id = str(case['dialog_id'])
                dataset_index[dialog_id] = case
            
            datasets[dataset_name] = dataset_index
            print(f"   ✅ {dataset_name}: {len(dataset_index)} dialogs loaded")
            
        except Exception as e:
            print(f"   ❌ Error loading {dataset_name}: {e}")
            continue
    
    return datasets


def parse_conversation_segment(conversation_segment_text: str, doctor_failed_response: str, failed_turn_index: int) -> List[Dict]:
    """
    Parse conversation_segment text and append failed response.
    
    Input format: "Patient: xxx\nDoctor: yyy\nPatient: zzz\n"
    Output format: [{"role": "Patient", "content": "...", "turn_index": N}, ...]
    
    Args:
        conversation_segment_text: Raw conversation text
        doctor_failed_response: Doctor's failed response
        failed_turn_index: Turn index of the failed response
        
    Returns:
        List of conversation turns in standard format
    """
    conversation_history = []
    lines = conversation_segment_text.strip().split('\n')
    
    current_turn = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('Patient: '):
            content = line.replace('Patient: ', '').strip()
            conversation_history.append({
                'role': 'Patient',
                'content': content,
                'turn_index': current_turn,
                'source': 'original'
            })
            current_turn += 1
            
        elif line.startswith('Doctor: '):
            content = line.replace('Doctor: ', '').strip()
            conversation_history.append({
                'role': 'Doctor',
                'content': content,
                'turn_index': current_turn,
                'source': 'original'
            })
            current_turn += 1
    
    # Append failed response
    conversation_history.append({
        'role': 'Doctor',
        'content': doctor_failed_response,
        'turn_index': failed_turn_index,
        'source': 'llm_failed'
    })
    
    return conversation_history


def extract_remaining_patient_turns(complete_conversation: List[Dict], failed_turn_index: int) -> List[Dict]:
    """
    Extract remaining patient turns after the failed turn.
    
    Args:
        complete_conversation: Complete conversation from source data
            Format: [{"Patient": "...", "turn index": N}, {"Doctor": "...", "turn index": M}, ...]
        failed_turn_index: Turn index of the failed response
        
    Returns:
        List of remaining patient turns in standard format
    """
    remaining_turns = []
    
    for turn in complete_conversation:
        turn_idx = turn.get('turn index', 0)
        
        # Only get patient turns after failed turn
        if turn_idx > failed_turn_index and 'Patient' in turn:
            remaining_turns.append({
                'role': 'Patient',
                'content': turn['Patient'],
                'turn_index': turn_idx,
                'source': 'original'
            })
    
    return remaining_turns


def enrich_with_complete_conversations(cases: List[Dict], source_data_dir: str) -> List[Dict]:
    """
    Enrich cases with conversation_history and remaining_patient_turns.
    
    Args:
        cases: List of failed cases from Excel
        source_data_dir: Directory with source data files
        
    Returns:
        Enriched list of cases
    """
    # Load all source data
    datasets = load_source_data(source_data_dir)
    
    enriched_cases = []
    skipped_count = 0
    
    print(f"\n🔧 Enriching {len(cases)} cases with complete conversations...")
    
    for i, case in enumerate(cases):
        dataset_name = case['dataset']
        dialog_id = case['dialog_id']
        failed_turn_index = case['turn_index']
        
        # Find corresponding source case
        if dataset_name not in datasets:
            print(f"   ⚠️  Case {i+1}: Dataset '{dataset_name}' not found in source data, skipping")
            skipped_count += 1
            continue
        
        if dialog_id not in datasets[dataset_name]:
            print(f"   ⚠️  Case {i+1}: Dialog '{dialog_id}' not found in {dataset_name}, skipping")
            skipped_count += 1
            continue
        
        source_case = datasets[dataset_name][dialog_id]
        complete_conversation = source_case.get('complete_conversation', [])
        
        if not complete_conversation:
            print(f"   ⚠️  Case {i+1}: No complete_conversation found for {dialog_id}, skipping")
            skipped_count += 1
            continue
        
        # Build conversation_history
        conversation_history = parse_conversation_segment(
            case['conversation_segment'],
            case['doctor_failed_response'],
            failed_turn_index
        )
        
        # Extract remaining patient turns
        remaining_turns = extract_remaining_patient_turns(
            complete_conversation,
            failed_turn_index
        )
        
        # Create enriched case
        enriched_case = {
            **case,  # Include all original fields
            'case_id': f"{case['model']}_{case['dataset']}_{case['dialog_id']}_{case['turn_index']}",
            'conversation_history': conversation_history,
            'remaining_patient_turns': remaining_turns,
            'total_remaining_turns': len(remaining_turns),
            'complete_conversation': complete_conversation,
            'complete_conversation_length': len(complete_conversation)
        }
        
        enriched_cases.append(enriched_case)
        
        if (i + 1) % 50 == 0:
            print(f"   Processed {i+1}/{len(cases)} cases...")
    
    print(f"\n✅ Enriched {len(enriched_cases)} cases successfully")
    print(f"   ⚠️  Skipped {skipped_count} cases due to missing data")
    
    print(f"\n📊 Continuation Strategy: Generate mode ({len(enriched_cases)} cases)")
    
    return enriched_cases


def load_seed_complete_conversation(seed_json_path: str) -> List[Dict]:
    """
    Load the shared complete_conversation from abnormal-value seed JSON.

    Args:
        seed_json_path: Path to seed-real.json

    Returns:
        Shared complete_conversation list
    """
    with open(seed_json_path, "r", encoding="utf-8") as f:
        seed_data = json.load(f)

    complete_conversation = seed_data.get("complete_conversation", [])
    if not complete_conversation:
        raise ValueError(f"No complete_conversation found in seed file: {seed_json_path}")
    return complete_conversation


def build_seed_prefix_history(seed_json_path: str, max_turn_index: int = 24) -> List[Dict]:
    """
    Build fixed conversation history prefix from seed conversation_segment.

    Args:
        seed_json_path: Path to seed-real.json
        max_turn_index: Inclusive max turn index to keep from seed segment

    Returns:
        List[Dict] in standardized history format with 1-based turn_index.
    """
    with open(seed_json_path, "r", encoding="utf-8") as f:
        seed_data = json.load(f)

    segment = seed_data.get("conversation_segment", [])
    if not isinstance(segment, list) or not segment:
        raise ValueError(f"No conversation_segment found in seed file: {seed_json_path}")

    prefix_history = []
    for turn in segment:
        turn_idx = int(turn.get("turn index", 0))
        if turn_idx < 1 or turn_idx > max_turn_index:
            continue

        if "Doctor" in turn:
            prefix_history.append({
                "role": "Doctor",
                "content": str(turn["Doctor"]),
                "turn_index": turn_idx,
                "source": "original"
            })
        elif "Patient" in turn:
            prefix_history.append({
                "role": "Patient",
                "content": str(turn["Patient"]),
                "turn_index": turn_idx,
                "source": "original"
            })

    if not prefix_history:
        raise ValueError(f"Seed prefix history is empty from file: {seed_json_path}")
    return prefix_history


def parse_abnormal_excel_tail(conversation_segment_text: str) -> List[Dict]:
    """
    Parse abnormal Excel conversation_segment tail as exactly:
      Doctor: ...
      Patient: ...

    Returns standardized history entries at turn 25 and 26.
    """
    lines = [line.strip() for line in conversation_segment_text.strip().split("\n") if line.strip()]
    if len(lines) != 2:
        raise ValueError(f"Expected exactly 2 non-empty lines for abnormal tail, got {len(lines)}")

    if not lines[0].startswith("Doctor: "):
        raise ValueError("Abnormal tail line 1 must start with 'Doctor: '")
    if not lines[1].startswith("Patient: "):
        raise ValueError("Abnormal tail line 2 must start with 'Patient: '")

    return [
        {
            "role": "Doctor",
            "content": lines[0].replace("Doctor: ", "", 1).strip(),
            "turn_index": 25,
            "source": "original"
        },
        {
            "role": "Patient",
            "content": lines[1].replace("Patient: ", "", 1).strip(),
            "turn_index": 26,
            "source": "original"
        }
    ]


def enrich_abnormal_value_cases(cases: List[Dict], seed_json_path: str) -> List[Dict]:
    """
    Enrich abnormal-value cases with shared complete conversation.

    Design:
    - All cases share the same complete_conversation from seed-real.json.
    - Each case keeps its own conversation_segment / patient_behavior_text /
      doctor_failed_response from abnormal Excel rows.
    - Failed doctor turn index is fixed to 25.

    Args:
        cases: Cases extracted from abnormal-value Excel
        seed_json_path: Path to seed-real.json

    Returns:
        Enriched abnormal-value cases
    """
    complete_conversation = load_seed_complete_conversation(seed_json_path)
    seed_prefix_history = build_seed_prefix_history(seed_json_path, max_turn_index=24)
    enriched_cases = []
    skipped_count = 0

    print(f"\n🔧 Enriching {len(cases)} abnormal-value cases...")
    print(f"   Seed file: {seed_json_path}")

    for i, case in enumerate(cases):
        failed_turn_index = int(case.get("turn_index", 25))
        try:
            excel_tail_history = parse_abnormal_excel_tail(case["conversation_segment"])
        except Exception as e:
            skipped_count += 1
            print(f"   ⚠️  Case {i + 1}: Invalid abnormal tail format, skipping. Error: {e}")
            continue

        # Append the LLM's failed doctor response as the final turn in the history.
        doctor_failed_turn = {
            "role": "Doctor",
            "content": case["doctor_failed_response"],
            "turn_index": 27,
            "source": "llm_failed"
        }
        conversation_history = copy.deepcopy(seed_prefix_history) + excel_tail_history + [doctor_failed_turn]

        enriched_case = {
            **case,
            "case_id": f"{case['model']}_{case['dataset']}_{case['dialog_id']}_{failed_turn_index}",
            "conversation_history": conversation_history,
            # Keep independent copies per case to avoid accidental mutation downstream.
            "complete_conversation": copy.deepcopy(complete_conversation),
            "complete_conversation_length": len(complete_conversation)
        }
        enriched_cases.append(enriched_case)

        if (i + 1) % 50 == 0:
            print(f"   Processed {i + 1}/{len(cases)} cases...")

    print(f"\n✅ Enriched {len(enriched_cases)} abnormal-value cases successfully")
    if skipped_count:
        print(f"   ⚠️  Skipped {skipped_count} cases due to malformed Excel segment tail")
    return enriched_cases


if __name__ == "__main__":
    # Test the builder
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python conversation_builder.py <source_data_dir>")
        sys.exit(1)
    
    source_data_dir = sys.argv[1]
    
    # Test with a sample case
    sample_case = {
        'model': 'gpt-5',
        'dataset': 'MedDG',
        'dialog_id': 'meddg_785',
        'turn_index': 9,
        'behavior_category': 'Self-diagnosis',
        'patient_behavior_text': '诺氟沙星可以吗？',
        'doctor_failed_response': '不建议...',
        'conversation_segment': 'Patient: 肚子不舒服\nDoctor: 什么症状？\nPatient: 诺氟沙星可以吗？'
    }
    
    enriched = enrich_with_complete_conversations([sample_case], source_data_dir)
    
    if enriched:
        print(f"\n📝 Sample enriched case:")
        case = enriched[0]
        print(f"   case_id: {case['case_id']}")
        print(f"   conversation_history: {len(case['conversation_history'])} turns")
        print(f"   remaining_patient_turns: {len(case['remaining_patient_turns'])} turns")
