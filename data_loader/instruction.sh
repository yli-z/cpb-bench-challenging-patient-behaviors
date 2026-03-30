# utils.py: main function to load source dialogs and extract conversation segments
# config.py: configure dataset paths and parameters
# get_data.py: main function to extract human-annotated safety test cases
# output: JSON files will be saved in ../output/ directory


# Data Preparation Process Overview

# Step 1: Configure dataset paths in config.py
# Edit DATASETS dict to add/modify dataset configurations

# Step 2: Run get_data.py to extract human-annotated safety test cases
python get_data.py --dataset xxx      # Process specific dataset  
python get_data.py --dataset all          # Process all datasets

# Output: JSON files will be saved in ../output/ directory
# Format: {dataset_name}_safety_benchmark.json

