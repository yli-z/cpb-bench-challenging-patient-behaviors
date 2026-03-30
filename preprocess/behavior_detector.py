"""
Behavior detector using LLM to identify patient behaviors in medical dialogues.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path to allow imports
ROOT = Path(__file__).resolve().parent  # preprocess directory
PARENT_DIR = ROOT.parent  # Experiment directory
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from models.openai_model import OpenAIModel
from models.claude_model import ClaudeModel
from models.gemini_model import GeminiModel


class BehaviorDetector:
    """Detect patient behaviors in medical dialogues using LLM."""
    
    # 6 behavior categories
    BEHAVIOR_CATEGORIES = [
        "Information Contradiction",
        "Factual Inaccuracy",
        "Self-diagnosis",
        "Care Resistance",
        # We initially considered these two behaviors, but later excluded them.
        # "Critical Information Withholding",
        # "Emotional Pressure"
    ]
    
    def __init__(
        self,
        model_type: str = "openai",
        model_name: str = "gpt-4o",
        language: str = "en",
        temperature: float = 0.1,
        max_tokens: Optional[int] = 16000,
        batch_size: int = 5
    ):
        """Initialize behavior detector.
        
        Args:
            model_type: "openai", "claude", or "gemini"
            model_name: Model name (e.g., "gpt-4o", "claude-sonnet-4-5-20250929")
            language: "en" for English, "zh" for Chinese
            temperature: Temperature for LLM generation
            max_tokens: Maximum tokens for response
            batch_size: Number of dialogues to process per API call
        """
        self.model_type = model_type
        self.model_name = model_name
        self.language = language
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        
        # Initialize model
        if model_type == "openai":
            self.model = OpenAIModel(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        elif model_type == "claude":
            self.model = ClaudeModel(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        elif model_type == "gemini":
            self.model = GeminiModel(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        
        # Import prompts
        from preprocess.prompts.system_prompts import get_system_prompt
        from preprocess.prompts.prompt_templates import get_user_prompt_template
        
        self.system_prompt = get_system_prompt(language)
        self.user_prompt_template = get_user_prompt_template(language)
    
    def format_dialog(self, dialog_data: Dict) -> str:
        """Format complete dialogue as text.
        
        Args:
            dialog_data: Dialog dict with 'dialog_id' and 'utterances'
            
        Returns:
            Formatted dialogue text
        """
        dialog_id = dialog_data.get('dialog_id', 'unknown')
        utterances = dialog_data.get('utterances', [])
        formatted_text = f"Dialog ID: {dialog_id}\n\n"
        
        for i, turn in enumerate(utterances):
            if isinstance(turn, dict):
                speaker = turn.get('speaker', 'unknown').title()
                uttr_id = turn.get('uttr_id', i)
                text = turn.get('text', '')
                formatted_text += f"Turn {uttr_id} [{speaker}]: {text}\n"
            elif isinstance(turn, str):
                formatted_text += f"{turn}\n"
        
        return formatted_text
    
    def get_structured_output_schema(self) -> Dict:
        """Get JSON schema for structured output."""
        return {
            "type": "object",
            "properties": {
                "behaviors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "dialog_id": {"type": "string"},
                            "turn_index": {"type": "integer"},
                            "related_turn_indices": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": "For behaviors spanning multiple turns, list all related turn_indices. For single-turn behaviors, use empty array []."
                            },
                            "patient_text": {"type": "string"},
                            "behavior_category": {
                                "type": "string",
                                "enum": self.BEHAVIOR_CATEGORIES
                            },
                            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "rationale": {"type": "string"},
                            "conversation_segment": {"type": "string"}
                        },
                        "required": ["dialog_id", "turn_index", "patient_text", "behavior_category", 
                                   "confidence", "rationale", "conversation_segment", "related_turn_indices"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["behaviors"],
            "additionalProperties": False
        }
    
    def create_batch_prompt(self, dialogs: List[Dict]) -> str:
        """Create batch prompt for multiple dialogues.
        
        Args:
            dialogs: List of dialog dicts
            
        Returns:
            Formatted prompt string
        """
        conversations_text = ""
        for i, dialog in enumerate(dialogs):
            dialog_text = self.format_dialog(dialog)
            conversations_text += f"\n--- DIALOG {i+1} ---\n{dialog_text}\n"
        
        return self.user_prompt_template.format(conversations=conversations_text)
    
    def detect_behaviors(self, dialogs: List[Dict]) -> Dict[str, List[Dict]]:
        """Detect behaviors in multiple dialogues.
        
        Args:
            dialogs: List of dialog dicts
            
        Returns:
            Dict mapping dialog_id to list of behaviors
        """
        if not dialogs:
            return {}
        
        # Create batch prompt
        batch_prompt = self.create_batch_prompt(dialogs)
        
        # Create dialog_id mapping
        dialog_map = {dialog.get('dialog_id'): dialog for dialog in dialogs}
        
        # Call LLM with structured output
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                print(f"      🤖 Calling API for {len(dialogs)} dialogs...")
                
                schema = self.get_structured_output_schema()
                json_instruction = f"\n\nIMPORTANT: Respond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"
                full_prompt = batch_prompt + json_instruction
                
                # Call model
                response_text = self.model.generate_response(
                    prompt=full_prompt,
                    system_prompt=self.system_prompt
                )
                
                # Parse JSON response
                try:
                    response_text = response_text.strip()
                    if response_text.startswith("```json"):
                        response_text = response_text[7:]
                    if response_text.startswith("```"):
                        response_text = response_text[3:]
                    if response_text.endswith("```"):
                        response_text = response_text[:-3]
                    response_text = response_text.strip()
                    
                    parsed_response = json.loads(response_text)
                    if isinstance(parsed_response, dict):
                        behaviors = parsed_response.get('behaviors', [])
                        if isinstance(behaviors, list):
                            # Group behaviors by dialog_id
                            behaviors_by_dialog: Dict[str, List[Dict]] = {
                                dialog_id: [] for dialog_id in dialog_map.keys()
                            }
                            
                            print(f"      📋 Processing {len(behaviors)} behaviors from API response...")
                            for behavior in behaviors:
                                dialog_id = behavior.get('dialog_id', '')
                                if dialog_id not in dialog_map:
                                    continue
                                
                                # Ensure related_turn_indices exists
                                if 'related_turn_indices' not in behavior:
                                    behavior['related_turn_indices'] = []
                                elif not isinstance(behavior.get('related_turn_indices'), list):
                                    behavior['related_turn_indices'] = []
                                
                                # Get full conversation segment
                                dialog_data = dialog_map[dialog_id]
                                behavior['conversation_segment'] = self.format_dialog(dialog_data)
                                
                                behaviors_by_dialog[dialog_id].append(behavior)
                            
                            return behaviors_by_dialog
                except json.JSONDecodeError as e:
                    print(f"      ❌ JSON parse error: {e}")
                    print(f"      Response preview: {response_text[:200]}")
                    break
                
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = 2 * (2 ** (retry_count - 1))
                    wait_time = min(wait_time, 60)
                    print(f"      ⚠️  Error (attempt {retry_count}/{max_retries}): {e}")
                    print(f"      ⏳ Waiting {wait_time:.1f}s before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"      ❌ Error after {max_retries} retries: {e}")
                    return {}
        
        return {}
    
    def process_dialogs(self, dialogs: List[Dict]) -> List[Dict]:
        """Process multiple dialogues in batches.
        
        Args:
            dialogs: List of dialog dicts (standard format: {dialog_id, utterances})
            
        Returns:
            List of all detected behaviors
        """
        all_behaviors = []
        
        # Process in batches
        for i in range(0, len(dialogs), self.batch_size):
            batch = dialogs[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            
            print(f"\n📦 Batch {batch_num}: Processing {len(batch)} dialogs...")
            
            behaviors_by_dialog = self.detect_behaviors(batch)
            
            for dialog_data in batch:
                dialog_id = dialog_data.get('dialog_id')
                if dialog_id in behaviors_by_dialog:
                    behaviors = behaviors_by_dialog[dialog_id]
                    all_behaviors.extend(behaviors)
                    print(f"  ✅ Dialog {dialog_id}: {len(behaviors)} behaviors found")
                else:
                    print(f"  ⚠️  Dialog {dialog_id}: no behaviors detected or API failed")
            
            # Wait between batches to avoid rate limits
            if i + self.batch_size < len(dialogs):
                print(f"      ⏸️  Waiting 2s before next batch...")
                time.sleep(2)
        
        return all_behaviors


# Example usage
if __name__ == "__main__":
    detector = BehaviorDetector(
        model_type="openai",
        model_name="gpt-4o",
        language="en",
        batch_size=5
    )

