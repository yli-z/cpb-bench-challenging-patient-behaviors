import prompts.prompt_templates as prompt_templates
from models.model_utils import load_model
from multiturn_continuation.agents.utils import (
    format_prompt_to_messages,
    format_conversations_role_content_to_string
)
from abc import ABC, abstractmethod

# =========================================================================
# Base Class (Unchanged)
# =========================================================================
class Doctor(ABC):
    def __init__(self, model_name, prompt_key, generation_config):
        self.model_name = model_name
        self.prompt_key = prompt_key
        self.model = load_model(
            model_name,
            gpu_memory_utilization=0.5,
            seed=0,
            num_samples=1,
            max_tokens=generation_config.get("max_tokens", None),
            vllm_api_base=generation_config.get("vllm_api_base", None),
            enable_thinking=generation_config.get("enable_thinking", False)
        )
        self.generation_config = generation_config

    @abstractmethod
    def respond(self, current_conversation, complete_conversation):
        raise NotImplementedError

PROMPT_OPTIONS = {
    'default': prompt_templates.DOCTOR_ASSISTANT_PROMPT,
    'cot': prompt_templates.COT_DOCTOR_ASSISTANT_PROMPT,
    'instruction': prompt_templates.Instruction_DOCTOR_ASSISTANT_PROMPT,
}

class DirectDoctor(Doctor):
    """
    Standard Doctor Agent.
    prompt_key options: 'default', 'cot', 'instruction'
    """
    def __init__(self, model_name, prompt_key='default', generation_config=None):
        super().__init__(model_name, prompt_key, generation_config or {})
        if prompt_key not in PROMPT_OPTIONS:
            raise ValueError(f"Invalid prompt_key '{prompt_key}'. Choose from: {list(PROMPT_OPTIONS)}")
        self.prompt_template = PROMPT_OPTIONS[prompt_key]

    def respond(self, current_conversation, complete_conversation, behavior_instruction=None):
        conv_str = format_conversations_role_content_to_string(current_conversation)
        if self.prompt_key == 'instruction':
            prompt = self.prompt_template.format(
                behavior_instruction=behavior_instruction or '',
                conversation_segment=conv_str
            )
        else:
            prompt = self.prompt_template.format(conversation_segment=conv_str)
        messages = format_prompt_to_messages(prompt, system_prompt=None)

        # Use split_think if model supports it (e.g., Qwen3 thinking mode)
        from models.remote_vllm_model import RemoteVLLMModel
        thinking = None
        if isinstance(self.model, RemoteVLLMModel) and self.model.enable_thinking:
            response, thinking = self.model.generate_text_response(messages, split_think=True)
        else:
            response = self.model.generate_text_response(messages)

        if response.lower().startswith("doctor:"):
            response = response[len("doctor:"):].strip()
        return response, thinking
