"""
Utility functions for processing medical dialog data.
"""

import json
import pandas as pd
from typing import Dict, List, Any, Optional
import os


def load_source_dialogs(dataset_name: str, config: Dict) -> Dict:
    """
    Load source dialog data from JSON file.
    
    Args:
        dataset_name: Name of the dataset (e.g., "ACI", "MediTOD")
        config: Dataset configuration dictionary
        
    Returns:
        Dictionary mapping dialog_id to dialog data
    """
    source_path = config["source_data"]
    format_type = config["format"]
    
    print(f"Loading source data from: {source_path}")
    
    with open(source_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    dialogs_dict = {}
    
    if format_type == "ACI":
        # ACI format: {"train": [...], "dev": [...]}
        if isinstance(raw_data, dict):
            for split in ["train", "dev", "test"]:
                if split in raw_data:
                    for dialog in raw_data[split]:
                        dialog_id = dialog.get("id")
                        if dialog_id:
                            dialogs_dict[dialog_id] = dialog
        print(f"Loaded {len(dialogs_dict)} dialogs from ACI dataset")
        
    elif format_type == "MediTOD":
        # MediTOD format: {dialog_id: {utterances: [...]}, ...}
        if isinstance(raw_data, dict):
            for dialog_id, dialog_data in raw_data.items():
                dialogs_dict[dialog_id] = {
                    "dialog_id": dialog_id,
                    "utterances": dialog_data.get("utterances", [])
                }
        print(f"Loaded {len(dialogs_dict)} dialogs from MediTOD dataset")
    
    return dialogs_dict


def extract_conversation_segment(dialog_data: Dict, turn_index: int, 
                                 config: Dict, format_type: str) -> List[Dict]:
    """
    Extract conversation segment from start to turn_index (inclusive).
    
    Args:
        dialog_data: Full dialog data
        turn_index: Index of the turn where behavior occurs
        config: Dataset configuration
        format_type: Format type ("ACI" or "MediTOD")
        
    Returns:
        List of dictionaries, each with speaker key and "turn index"
        Format: [{"Doctor": "text", "turn index": 1}, {"Patient": "text", "turn index": 2}, ...]
    """
    speaker_mapping = config["speaker_mapping"]
    conversation_list = []
    
    if format_type == "ACI":
        # ACI format: transcript is a list of turns
        transcript = dialog_data.get("transcript", [])
        
        # Convert turn_id to 0-based index (ACI uses turn_id starting from 1)
        for turn in transcript:
            turn_id = turn.get("turn_id", 0)
            # Include turns from 1 to turn_index (inclusive)
            if turn_id <= turn_index:
                speaker = turn.get("speaker", "")
                text = turn.get("transcript", "").strip()
                
                # Map speaker to standard format
                normalized_speaker = speaker_mapping.get(speaker, speaker)
                
                # Create dictionary with speaker as key and turn index
                turn_dict = {
                    normalized_speaker: text,
                    "turn index": turn_id
                }
                conversation_list.append(turn_dict)
            
            if turn_id >= turn_index:
                break
    
    elif format_type == "MediTOD":
        # MediTOD format: utterances with uttr_id
        utterances = dialog_data.get("utterances", [])
        
        for utterance in utterances:
            uttr_id = utterance.get("uttr_id", 0)
            # Include utterances from 0 to turn_index (inclusive)
            if uttr_id <= turn_index:
                speaker = utterance.get("speaker", "")
                text = utterance.get("text", "").strip()
                
                # Map speaker to standard format
                normalized_speaker = speaker_mapping.get(speaker, speaker)
                
                # Create dictionary with speaker as key and turn index
                turn_dict = {
                    normalized_speaker: text,
                    "turn index": uttr_id
                }
                conversation_list.append(turn_dict)
            
            if uttr_id >= turn_index:
                break
    
    return conversation_list


def extract_complete_conversation(dialog_data: Dict, config: Dict, format_type: str) -> List[Dict]:
    """
    Extract complete conversation from the entire dialog.
    
    Args:
        dialog_data: Full dialog data
        config: Dataset configuration
        format_type: Format type ("ACI" or "MediTOD")
        
    Returns:
        List of dictionaries, each with speaker key and "turn index"
        Format: [{"Doctor": "text", "turn index": 1}, {"Patient": "text", "turn index": 2}, ...]
    """
    speaker_mapping = config["speaker_mapping"]
    conversation_list = []
    
    if format_type == "ACI":
        # ACI format: transcript is a list of turns
        transcript = dialog_data.get("transcript", [])
        
        for turn in transcript:
            turn_id = turn.get("turn_id", 0)
            speaker = turn.get("speaker", "")
            text = turn.get("transcript", "").strip()
            
            # Map speaker to standard format
            normalized_speaker = speaker_mapping.get(speaker, speaker)
            
            # Create dictionary with speaker as key and turn index
            turn_dict = {
                normalized_speaker: text,
                "turn index": turn_id
            }
            conversation_list.append(turn_dict)
    
    elif format_type == "MediTOD":
        # MediTOD format: utterances with uttr_id
        utterances = dialog_data.get("utterances", [])
        
        for utterance in utterances:
            uttr_id = utterance.get("uttr_id", 0)
            speaker = utterance.get("speaker", "")
            text = utterance.get("text", "").strip()
            
            # Map speaker to standard format
            normalized_speaker = speaker_mapping.get(speaker, speaker)
            
            # Create dictionary with speaker as key and turn index
            turn_dict = {
                normalized_speaker: text,
                "turn index": uttr_id
            }
            conversation_list.append(turn_dict)
    
    return conversation_list


def load_excel_annotations(excel_dir: str, excel_files: List[str], 
                          file_to_category: Dict[str, str]) -> pd.DataFrame:
    """
    Load and combine all Excel annotation files from a directory.
    
    Args:
        excel_dir: Directory containing Excel files
        excel_files: List of Excel file names (without .xlsx extension)
        file_to_category: Mapping from file names to category names
        
    Returns:
        Combined DataFrame with all annotations
    """
    all_annotations = []
    
    if not os.path.exists(excel_dir):
        print(f"Error: Excel directory not found: {excel_dir}")
        return pd.DataFrame()
    
    for file_name in excel_files:
        excel_path = os.path.join(excel_dir, f"{file_name}.xlsx")
        
        if not os.path.exists(excel_path):
            print(f"Warning: Excel file not found: {excel_path}")
            continue
        
        try:
            # Try to read Excel file (some have header=1, some have header=0)
            # Try header=1 first (after definition row)
            df = pd.read_excel(excel_path, header=1)
            
            if 'dialog_id' not in df.columns:
                # Try header=0 instead
                df = pd.read_excel(excel_path, header=0)
            
            # Remove rows without dialog_id
            if 'dialog_id' in df.columns:
                df = df.dropna(subset=['dialog_id'])
            else:
                print(f"Warning: 'dialog_id' column not found in {file_name}.xlsx")
                continue
            
            # Normalize "Human check" column name (handle with/without trailing space)
            if 'Human check ' in df.columns:
                df['Human check'] = df['Human check ']
            elif 'Human check' not in df.columns:
                print(f"Warning: 'Human check' column not found in {file_name}.xlsx")
                continue
            
            # Add behavior category
            category_name = file_to_category.get(file_name, file_name.replace("_", " "))
            df['behavior_category'] = category_name
            
            all_annotations.append(df)
            print(f"Loaded {len(df):3d} annotations from '{file_name}.xlsx' ({category_name})")
            
        except Exception as e:
            print(f"Error loading '{file_name}.xlsx': {e}")
    
    if all_annotations:
        combined_df = pd.concat(all_annotations, ignore_index=True)
        print(f"\nTotal annotations loaded: {len(combined_df)}")
        return combined_df
    else:
        return pd.DataFrame()


def filter_positive_annotations(df: pd.DataFrame, human_check_column: str, 
                                positive_value: str) -> pd.DataFrame:
    """
    Filter annotations to keep only positive human checks.
    
    Args:
        df: DataFrame with annotations
        human_check_column: Name of the human check column
        positive_value: Value indicating positive annotation
        
    Returns:
        Filtered DataFrame
    """
    if human_check_column not in df.columns:
        print(f"Warning: Column '{human_check_column}' not found in DataFrame")
        return df
    
    # Strip whitespace from Human check column values to handle " Y", "Y ", etc.
    df[human_check_column] = df[human_check_column].astype(str).str.strip()
    
    # Filter for positive annotations
    positive_df = df[df[human_check_column] == positive_value].copy()
    print(f"Filtered to {len(positive_df)} positive annotations (out of {len(df)})")
    
    return positive_df


def get_patient_behavior_text(dialog_data: Dict, turn_index: int, format_type: str) -> str:
    """
    Extract the patient's text at the specified turn index.
    
    Args:
        dialog_data: Full dialog data
        turn_index: Index of the turn
        format_type: Format type ("ACI" or "MediTOD")
        
    Returns:
        Patient's text at the turn
    """
    if format_type == "ACI":
        transcript = dialog_data.get("transcript", [])
        for turn in transcript:
            if turn.get("turn_id") == turn_index:
                return turn.get("transcript", "").strip()
    
    elif format_type == "MediTOD":
        utterances = dialog_data.get("utterances", [])
        for utterance in utterances:
            if utterance.get("uttr_id") == turn_index:
                return utterance.get("text", "").strip()
    
    return ""


def generate_case_id(dataset_name: str, index: int) -> str:
    """
    Generate a unique case ID.
    
    Args:
        dataset_name: Name of the dataset
        index: Sequential index
        
    Returns:
        Case ID string
    """
    return f"{dataset_name}_{index:03d}"

