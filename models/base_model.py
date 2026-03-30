"""
Base LLM interface for doctor-assistant responses.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional


class BaseLLM(ABC):
    """Abstract base class for LLM backends."""

    def __init__(self, model_name: str, temperature: float = 0.7, max_tokens: Optional[int] = None):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response from the LLM."""
        raise NotImplementedError

    def generate_doctor_response(self, case: Dict, user_prompt_template: str, system_prompt: Optional[str] = None) -> Dict:
        """
        Render prompt from a case dict and get model response.
        Expected keys in case: dialog_id, behavior_category, conversation_segment
        """
        prompt = user_prompt_template.format(
            behavior_category=case.get("behavior_category", "Unknown"), 
            conversation_segment=case.get("conversation_segment", ""),
            behavior_instruction=case.get("behavior_instruction", "")
        )
        print("prompt===========================================")
        print(prompt)
        answer = self.generate_response(prompt, system_prompt=system_prompt)

        return {
            "dialog_id": case.get("dialog_id"),
            "behavior_category": case.get("behavior_category"),
            "model": self.model_name,
            "response": answer
        }

