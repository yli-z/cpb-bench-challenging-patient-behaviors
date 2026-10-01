import asyncio

from prompts.prompt_templates import DOCTOR_ASSISTANT_PROMPT
from .base import BaseStrategy, StrategyResult

PATIENT_EVAL_PROMPT = """You are a medical AI assistant. Before responding to the patient, carefully review the conversation and analyze the patient's most recent message.

Identify anything in the patient's statements that may require attention, such as inaccurate claims, inconsistencies with earlier statements, unreasonable requests, or any other issues a doctor should consider before replying.

Provide a brief analysis.

Conversation:
{conversation}"""

RESPONSE_PROMPT = """Based on the conversation history and the analysis of the patient's query, provide a response as the doctor.

Conversation:
{conversation_segment}

Analysis of Patient's Query:
{patient_evaluation}

Doctor:"""


class EvalPatientStrategy(BaseStrategy):
    """Evaluate Patient: two-step strategy that first analyzes the patient's query, then generates an informed response."""

    @property
    def name(self) -> str:
        return "Evaluate Patient"

    @property
    def output_dir(self) -> str:
        return "eval_patient"

    async def process_case(self, case, formatted_history, conversation_segment, llm):
        # Step 1: Evaluate the patient's query
        print("Step 1: Evaluating patient query...")
        eval_prompt = PATIENT_EVAL_PROMPT.format(conversation=formatted_history)
        eval_messages = [{"role": "user", "content": eval_prompt}]
        patient_evaluation = await asyncio.to_thread(llm.generate_text_response, eval_messages)
        print(f"\n--- Patient Evaluation ---\n{patient_evaluation}")

        # Step 2: Generate doctor response informed by the evaluation
        print("\nStep 2: Generating doctor response...")
        response_prompt = RESPONSE_PROMPT.format(
            conversation_segment=formatted_history,
            patient_evaluation=patient_evaluation
        )
        response_messages = [{"role": "user", "content": response_prompt}]
        generated_resp = await asyncio.to_thread(llm.generate_text_response, response_messages)
        print(f"\n--- Generated Doctor Response ---\n{generated_resp}")
        return StrategyResult(
            generated_response=generated_resp.strip(),
            extra_fields={
                'patient_evaluation': patient_evaluation.strip(),
            },
        )
