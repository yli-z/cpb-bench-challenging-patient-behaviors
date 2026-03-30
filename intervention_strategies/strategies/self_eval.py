import re
import asyncio

from prompts.prompt_templates import SELF_EVAL_DOCTOR_ASSISTANT_PROMPT, SELF_EVAL_CHECK_PROMPT
from .base import BaseStrategy, StrategyResult


class SelfEvalStrategy(BaseStrategy):
    """Self-Evaluation: generate a response, then self-check and revise if needed."""

    @property
    def name(self) -> str:
        return "Self-Evaluation"

    @property
    def output_dir(self) -> str:
        return "self_eval"

    def _parse_check_result(self, check_result: str):
        """Parse the self-evaluation output into verdict, issues, and revised response."""
        verdict_match = re.search(r'Verdict:\s*(PASS|REVISE)', check_result, re.IGNORECASE)
        verdict = verdict_match.group(1).upper() if verdict_match else "PASS"

        issues = ""
        revised_response = ""
        if verdict == "REVISE":
            issues_match = re.search(
                r'Issues:\s*(.*?)(?=Revised Response:|$)', check_result, re.DOTALL | re.IGNORECASE
            )
            if issues_match:
                issues = issues_match.group(1).strip()

            revised_match = re.search(r'Revised Response:\s*(.*)', check_result, re.DOTALL)
            if revised_match:
                revised_response = revised_match.group(1).strip()

        return verdict, issues, revised_response

    async def process_case(self, case, formatted_history, conversation_segment, llm):
        case['conversation_segment'] = formatted_history

        # Step 1: Generate initial response
        print("Generating response...")
        resp = await asyncio.to_thread(
            llm.generate_doctor_response, case, SELF_EVAL_DOCTOR_ASSISTANT_PROMPT, ""
        )
        generated_resp = resp.get('response', 'No response generated.')
        print(f"\n--- Generated Doctor Response ---\n{generated_resp}")

        final_response = generated_resp.strip()
        original_response = final_response

        # Step 2: Self-check
        print("\n--- Self-Evaluation Check ---")
        check_prompt = SELF_EVAL_CHECK_PROMPT.format(
            conversation_segment=formatted_history,
            doctor_response=final_response
        )
        check_messages = [{"role": "user", "content": check_prompt}]
        check_result = await asyncio.to_thread(llm.generate_text_response, check_messages)
        print(check_result)

        verdict, issues, revised_response = self._parse_check_result(check_result)
        if verdict == "REVISE" and revised_response:
            final_response = revised_response
            print(f"\n--- Revised Response ---\n{final_response}")

        return StrategyResult(
            generated_response=final_response,
            extra_fields={
                'original_response': original_response,
                'self_eval_verdict': verdict,
                'self_eval_issues': issues,
                'self_eval_revised_response': revised_response,
                'self_eval_raw_output': check_result.strip(),
            },
        )
