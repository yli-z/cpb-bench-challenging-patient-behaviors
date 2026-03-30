"""
Configuration file for medical dialog safety benchmark data processing.
Add new datasets here following the same structure.
"""

import os

# Base paths - all relative to this file's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # data_preparation directory
LAB_MED_DIR = os.path.dirname(os.path.dirname(BASE_DIR))  # LAB-Med directory (go up 2 levels)
EXPERIMENT_DIR = os.path.join(LAB_MED_DIR, "Experiment")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Dataset configurations
DATASETS = {
    "ACI": {
        "excel_dir": os.path.join(EXPERIMENT_DIR, "Mediqa_oe_ACI-filterout"),
        "source_data": os.path.join(EXPERIMENT_DIR, "Data/mediqa-oe_ACI/data/orders_data_transcript.json"),
        "dataset_source": "mediqa-oe_ACI",
        "speaker_mapping": {
            "DOCTOR": "Doctor",
            "PATIENT": "Patient"
        },
        "dialog_id_field": "id",  # Field name for dialog ID in source data
        "transcript_field": "transcript",  # Field name for conversation in source data
        "format": "ACI"  # Data format type
    },
    "MediTOD": {
        "excel_dir": os.path.join(EXPERIMENT_DIR, "Meditod-filterout"),
        "source_data": os.path.join(EXPERIMENT_DIR, "Data/MediTOD/raw_data/dialogs.json"),
        "dataset_source": "MediTOD",
        "speaker_mapping": {
            "doctor": "Doctor",
            "patient": "Patient"
        },
        "dialog_id_field": "dialog_id",  # Dialog ID is the key in the JSON
        "transcript_field": "utterances",  # Field name for conversation in source data
        "format": "MediTOD"  # Data format type
    }
}

# Excel file names (excluding "Critical_Information_Withholding")
BEHAVIOR_EXCEL_FILES = [
    "Care_Resistance",
    "Self_diagnosis",
    "Factual_Inaccuracy",
    "Information_Contradiction"
]

# Map Excel file names to standard category names
FILE_TO_CATEGORY = {
    "Care_Resistance": "Care Resistance",
    "Self_diagnosis": "Self-diagnosis",
    "Factual_Inaccuracy": "Factual Inaccuracy",
    "Information_Contradiction": "Information Contradiction"
}

# Human annotation column name (use only the standardized column)
HUMAN_CHECK_COLUMN = "Human check"
POSITIVE_ANNOTATION = "Y"

# Output configuration
os.makedirs(OUTPUT_DIR, exist_ok=True)

