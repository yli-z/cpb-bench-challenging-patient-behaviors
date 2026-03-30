"""
LLM-as-Judge evaluator for overreaction (negative-case) detection.

This evaluator checks whether a doctor OVERREACTED to a benign patient
turn — i.e., treated normal patient behaviour as a clinical red flag.
It uses the structured JSON output defined in prompts/negative_prompt.py.

Location : Negative_cases/negative_generate/evaluator/negative_evaluator.py
Project root is resolved as Path(__file__).parent x4 (evaluator →
negative_generate → Negative_cases → CPB-Bench).
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Ensure the project root (CPB-Bench/) is on sys.path so that
# `models` and `prompts` packages can be imported from anywhere.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.base_model import BaseLLM
from prompts.negative_prompt import OVERREACTION_JSON_SCHEMA, SYSTEM_PROMPT, USER_PROMPT


class NegativeEvaluator:
    """
    LLM-as-Judge evaluator for overreaction (negative-case) detection.

    Detects whether a doctor over-reacted to a benign patient turn.
    Uses GPT's official JSON output format (response_format) when the
    judge model is an OpenAI model; falls back to plain text + JSON
    parsing for other backends.
    """

    def __init__(
        self,
        judge_model: BaseLLM,
        temperature: float = 0,
        context_mode: str = "full_context",
    ):
        """
        Args:
            judge_model: Judge model instance (OpenAIModel, ClaudeModel, GeminiModel).
            temperature: Temperature parameter; 0 recommended for consistency.
            context_mode: Currently only "full_context" is supported for negative eval.
        """
        self.judge_model = judge_model
        self.temperature = temperature
        self.context_mode = context_mode

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_json(self, user_prompt: str) -> str:
        """
        Call the judge model and request a JSON response.

        For OpenAI models, uses the official ``response_format`` parameter
        (structured outputs).  For other backends, falls back to plain
        ``generate_response`` and parses the returned text as JSON.
        """
        # Lazy import to avoid circular dependency issues.
        try:
            from models.openai_model import OpenAIModel
            if isinstance(self.judge_model, OpenAIModel):
                return self.judge_model.generate_json_evaluation_response(
                    prompt=user_prompt,
                    system_prompt=SYSTEM_PROMPT,
                    schema=OVERREACTION_JSON_SCHEMA,
                )
        except ImportError:
            pass

        # Fallback for non-OpenAI models: generate_response returns text which
        # the system prompt already asks to be JSON.
        return self.judge_model.generate_response(
            prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
        )

    @staticmethod
    def _parse_response(raw: str) -> Dict[str, Any]:
        """
        Parse the JSON-formatted judge response into a structured dict.

        Tries ``json.loads`` first; falls back to regex extraction of the
        first ``{...}`` block when the model wraps the JSON in prose.
        """
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            m = re.search(r"\{[\s\S]*\}", raw)
            if m:
                try:
                    data = json.loads(m.group())
                except (json.JSONDecodeError, TypeError):
                    data = {}
            else:
                data = {}

        raw_verdict = data.get("verdict", "NOT OVERREACTED")
        if isinstance(raw_verdict, str):
            verdict = (
                "OVERREACTED"
                if "OVERREACTED" in raw_verdict.upper() and "NOT" not in raw_verdict.upper()
                else "NOT OVERREACTED"
            )
        else:
            verdict = "NOT OVERREACTED"

        # Current schema: verdict + triggered (4 boolean flags)
        triggered = data.get("triggered", {})

        return {
            "verdict": verdict,
            "evaluation_result": verdict == "OVERREACTED",  # True = overreacted
            "triggered": triggered,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_overreaction(
        self,
        dialog_id: str,
        conversation_segment: str,
        doctor_response: str,
    ) -> Dict[str, Any]:
        """
        Evaluate whether the doctor overreacted in the latest response.

        Args:
            dialog_id: Unique identifier for the conversation.
            conversation_segment: Prior patient–doctor exchange (context).
            doctor_response: The doctor's latest response to evaluate.

        Returns:
            dict with keys:
              - dialog_id (str)
              - verdict (str): "OVERREACTED" or "NOT OVERREACTED"
              - evaluation_result (bool): True = overreacted
              - triggered (dict): {
                    A_false_contradiction_flag: bool,
                    B_unnecessary_fact_correction: bool,
                    C_unprompted_selfdiagnosis_warning: bool,
                    D_unwarranted_compliance_push: bool
                }
              - judge_response_raw (str)
        """
        user_prompt = USER_PROMPT.format(
            conversation_segment=conversation_segment,
            doctor_response=doctor_response,
        )

        raw = self._call_json(user_prompt)
        parsed = self._parse_response(raw)
        parsed["dialog_id"] = dialog_id
        parsed["judge_response_raw"] = raw
        return parsed
