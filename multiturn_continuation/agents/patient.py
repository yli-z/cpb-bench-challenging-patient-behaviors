import prompts.patient_prompts as patient_prompts
from models.model_utils import load_model
from multiturn_continuation.agents.utils import format_prompt_to_messages, format_conversations_to_string
from abc import ABC, abstractmethod


class Patient(ABC):

    def __init__(self, model_name, generation_config):
        self.model_name = model_name
        self.model = load_model(
            model_name,
            gpu_memory_utilization=0.5,
            seed=0,
            num_samples=1,
            max_tokens=generation_config.get("max_tokens"),
            vllm_api_base=generation_config.get("vllm_api_base"),
            enable_thinking=generation_config.get("enable_thinking", False))
        self.generation_config = generation_config

    @abstractmethod
    def respond(self, truncated_conversation, complete_conversation):
        raise NotImplementedError

    def _clean_prefix(self, response: str) -> str:
        if response.lower().startswith("patient:"):
            response = response[len("patient:"):].strip()
        return response


class DirectPatient(Patient):

    def __init__(self, model_name, generation_config):
        super().__init__(model_name, generation_config)

    def respond(self, truncated_conversation, complete_conversation):
        trunc_conv_str = format_conversations_to_string(truncated_conversation, is_complete=False)
        comp_conv_str = format_conversations_to_string(complete_conversation, is_complete=True)
        system_prompt = patient_prompts.DIRECT_PATIENT_SYSTEM_PROMPT.format(
            complete_conversation=comp_conv_str
        )
        user_prompt = patient_prompts.DIRECT_PATIENT_USER_PROMPT.format(
            truncated_conversation=trunc_conv_str
        )
        messages = format_prompt_to_messages(user_prompt, system_prompt=system_prompt)
        response = self.model.generate_text_response(messages)
        return {"response": self._clean_prefix(response)}
