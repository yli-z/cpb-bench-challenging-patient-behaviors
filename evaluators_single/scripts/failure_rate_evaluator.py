"""
LLM-as-Judge core evaluation class.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

from models.base_model import BaseLLM
from prompts.evaluator_prompts import get_principal, SYSTEM_PROMPT
from evaluators_single.scripts.judge_utils import format_single_turn_evaluation_prompt, format_multi_turn_evaluation_prompt, parse_judge_response, find_min_context_segment


class FailureRateEvaluator:
    """
    LLM-as-Judge evaluator for failure rate assessment.
    """
    
    def __init__(
        self,
        judge_model: BaseLLM,
        is_multi_turn: bool,
        temperature: float = 0,
        context_mode: str = "current_turn"
    ):
        """
        Initialize LLM judge.
        
        Args:
            judge_model: Judge model instance (OpenAIModel, ClaudeModel, GeminiModel)
            temperature: Temperature parameter, recommended 0 for consistency
            context_mode: Context mode ("full_context" or "current_turn")
        """
        self.judge_model = judge_model
        self.temperature = temperature
        self.is_multi_turn = is_multi_turn
        self.context_mode = context_mode
        
        if context_mode not in ["full_context", "current_turn", "min_turn"]:
            raise ValueError(f"Invalid context_mode: {context_mode}. Must be 'full_context' or 'current_turn' or 'min_turn'.")
    
    def evaluate_failure(
        self,
        dialog_id: str,
        behavior_category: str,
        response: str,
        turn_index: int,
        patient_behavior_text: str,
        conversation_segment: Optional[str] = None,
        llm_conversation_segment: Optional[str] = None,
        item_to_detect: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate failure for a single response.
        
        IMPORTANT: 
        - The principal passed to the LLM judge contains ONLY the single principal
          for the given behavior_category
        - The LLM judge sees only one principal, not all five principals
        
        Args:
            dialog_id: Dialog ID
            behavior_category: Behavior category (used to get the single principal)
            response: Doctor response to evaluate
            turn_index: Current turn index
            patient_behavior_text: Patient behavior text
            conversation_segment: Conversation segment from JSONL data (optional, used for full_context mode)
            
        Returns:
            {
                "dialog_id": str,
                "behavior_category": str,
                "evaluation_result": bool,  # True = failure, False = not failure
                "judge_reasoning": str,     # Evaluation reasoning (optional)
                "confidence": float         # Confidence (optional)
            }
        """
        # Get ONLY the single principal for this behavior_category
        principal = get_principal(behavior_category)
        
        if not principal:
            raise ValueError(f"No principal found for behavior_category: {behavior_category}")
        
        # Format evaluation prompt
        # For full_context mode, pass conversation_segment if available
        # For current_turn mode, conversation_segment will be None/empty and won't appear in prompt
        if self.is_multi_turn:
            assert llm_conversation_segment is not None, "llm_conversation_segment must be provided for multi-turn evaluation"  
            user_prompt = format_multi_turn_evaluation_prompt(
                behavior_category=behavior_category,
                principal=principal,  # Only single principal here
                llm_conversation_segment=llm_conversation_segment
            )
        else:
            if self.context_mode == "min_turn":
                # extract minimal context segment
                conversation_segment = find_min_context_segment({"conversation_segment": conversation_segment}, behavior_category)
            elif self.context_mode == "current_turn":
                conversation_segment = None  # no context for current_turn mode
            else:
                # full_context mode, use provided conversation_segment
                conversation_segment = conversation_segment
            if behavior_category in [
                "Mixed-up Items", "Misbelief Denial", 
                "Unrelated Disease", "More Test", "More Medication"]:
                user_prompt = format_single_turn_evaluation_prompt(
                    behavior_category=behavior_category,
                    response=response,
                    patient_behavior_text=patient_behavior_text,
                    principal=principal,  # Only single principal here
                    conversation_segment=conversation_segment,
                    item_to_detect=item_to_detect
                )
            else:
                user_prompt = format_single_turn_evaluation_prompt(
                    behavior_category=behavior_category,
                    response=response,
                    patient_behavior_text=patient_behavior_text,
                    principal=principal,  # Only single principal here
                    conversation_segment=conversation_segment,
                )
        # Call LLM judge
        judge_response = self.judge_model.generate_response(
            prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT
        )

        # Parse response
        parsed = parse_judge_response(judge_response)

        return {
            "dialog_id": dialog_id,
            "behavior_category": behavior_category,
            "evaluation_result": parsed["result"],  # True = failure, False = not failure
            "judge_reasoning": parsed["reasoning"],
            "judge_response_raw": judge_response
        }