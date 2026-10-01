"""
Anthropic Claude model wrapper.
"""

import os
from typing import Optional

import anthropic

from .base_model import BaseLLM


class ClaudeModel(BaseLLM):
    """Wrapper for Claude models."""

    def __init__(self, model_name: str = "claude-3-opus-20240229", api_key: Optional[str] = None, temperature: float = 0.7, max_tokens: Optional[int] = None):
        super().__init__(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        api_key = (api_key or os.getenv("ANTHROPIC_API_KEY", "")).strip()
        if not api_key:
            raise ValueError("Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter.")
        self.client = anthropic.Anthropic(api_key=api_key)

    def _supports_structured_outputs(self) -> bool:
        name_lower = self.model_name.lower()
        supported_models = ["claude-sonnet-4-5", "claude-opus-4-1", "claude-opus-4-5", "claude-haiku-4-5"]
        return any(model in name_lower for model in supported_models)

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        sys_prompt = system_prompt or ""
        
        kwargs = {
            "model": self.model_name,
            "max_tokens": self.max_tokens or 4096,
            "system": sys_prompt,
            "messages": [{"role": "user", "content": prompt}],
        }
        
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        
        message = self.client.messages.create(**kwargs)
        return message.content[0].text.strip()

    def generate_text_response(self, messages) -> str:
        sys_prompt = None
        user_prompt = None
        
        assert len(messages) >= 1, "Messages must contain at least one message."
        assert messages[-1]['role'] == 'user', "The last message must be from the user."
        user_prompt = messages[-1]['content']
        if len(messages) >=2 and messages[0]['role'] == 'system':
            sys_prompt = messages[0]['content']
        
            kwargs = {
                "model": self.model_name,
                "max_tokens": self.max_tokens or 4096,
                "system": sys_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            }
        else:
            kwargs = {
                "model": self.model_name,
                "max_tokens": self.max_tokens or 4096,
                "messages": [{"role": "user", "content": user_prompt}],
            }
        
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        
        message = self.client.messages.create(**kwargs)
        return message.content[0].text.strip()

