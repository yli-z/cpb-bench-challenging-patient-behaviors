import re
import asyncio

from prompts.prompt_templates import COT_DOCTOR_ASSISTANT_PROMPT
from .base import BaseStrategy, StrategyResult


class CotStrategy(BaseStrategy):
    """Chain-of-Thought: the model reasons step-by-step before producing a final response."""

    @property
    def name(self) -> str:
        return "Chain-of-Thought"

    @property
    def output_dir(self) -> str:
        return "cot"

    async def process_case(self, case, formatted_history, conversation_segment, llm):
        case['conversation_segment'] = formatted_history

        resp = await asyncio.to_thread(
            llm.generate_doctor_response, case, COT_DOCTOR_ASSISTANT_PROMPT, ""
        )
        generated_resp = resp.get('response', 'No response generated.')
        print(f"\n--- Generated Doctor Response ---\n{generated_resp}")

        final_response = generated_resp.strip()
        reasoning = ""

        parts = re.split(r"\*?\*?Final\s+Doctor'?s?\s+Response:\*?\*?\s*", generated_resp, flags=re.IGNORECASE)
        if len(parts) > 1:
            reasoning = parts[0].strip()
            final_response = parts[-1].strip()
        print(f"\n--- Extracted Reasoning ---\n{reasoning}")
        print(f"\n--- Extracted Final Response ---\n{final_response}")

        return StrategyResult(
            generated_response=final_response,
            extra_fields={'reasoning': reasoning},
        )
