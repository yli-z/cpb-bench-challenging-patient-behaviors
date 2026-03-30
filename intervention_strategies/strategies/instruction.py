import asyncio

from prompts.prompt_templates import Instruction_DOCTOR_ASSISTANT_PROMPT
from .base import BaseStrategy, StrategyResult


class InstructionStrategy(BaseStrategy):
    """Instruction: the model receives behavior-aware instructions for all four challenging types."""

    @property
    def name(self) -> str:
        return "Instruction"

    @property
    def output_dir(self) -> str:
        return "instruction"

    async def process_case(self, case, formatted_history, conversation_segment, llm):
        case['conversation_segment'] = formatted_history

        resp = await asyncio.to_thread(
            llm.generate_doctor_response, case, Instruction_DOCTOR_ASSISTANT_PROMPT, ""
        )
        generated_resp = resp.get('response', 'No response generated.')
        print(f"\n--- Generated Doctor Response ---\n{generated_resp}")
        return StrategyResult(generated_response=generated_resp.strip())
