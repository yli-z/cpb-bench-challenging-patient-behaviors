"""
DeepSeek API model wrapper.
Supports Thinking mode (deepseek-reasoner) and No Thinking mode (deepseek-chat).
"""
import os
from typing import Optional
from openai import OpenAI

from .base_model import BaseLLM


class DeepSeekModel(BaseLLM):
    """Wrapper for DeepSeek models via API (deepseek-reasoner, deepseek-chat)."""

    def __init__(self, model_name: str = "deepseek-chat", api_key: Optional[str] = None, 
                 temperature: Optional[float] = None, max_tokens: Optional[int] = None):
        """
        Initialize DeepSeekModel.
        
        Args:
            model_name: Model name (deepseek-reasoner for Thinking, deepseek-chat for No Thinking)
            api_key: DeepSeek API key (or set DEEPSEEK_API_KEY env var)
            temperature: Temperature (None for default, not set in API call)
            max_tokens: Maximum tokens (default: 4096)
        """
        super().__init__(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        
        # DeepSeek API endpoint
        api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "DeepSeek API key not provided. "
                "Please provide via --deepseek-key argument or DEEPSEEK_API_KEY environment variable."
            )
        
        # Initialize OpenAI client with DeepSeek API endpoint
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
        
        # Determine mode based on model name
        self.is_thinking_mode = "reasoner" in model_name.lower()
        mode_str = "Thinking" if self.is_thinking_mode else "No Thinking"
        print(f">>>>>> DeepSeek Model: {model_name} ({mode_str} mode)")

    def generate_response(self, prompt, system_prompt=None) -> str:
        """Generate a response from the DeepSeek API."""
        
        if isinstance(prompt, str):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
        elif isinstance(prompt, list):
            messages = prompt
        else:
            raise ValueError("Prompt must be a string or a list of messages.")

        kwargs = {
            "model": self.model_name,
            "messages": messages,
        }
        
        # Don't set temperature (use default)
        # Set max_tokens
        kwargs["max_tokens"] = self.max_tokens or 4096

        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content.strip()
        return content

    def generate_text_response(self, messages) -> str:
        """
        Alias for generate_response() to maintain compatibility with DirectDoctor agent.
        DirectDoctor expects generate_text_response() method.
        """
        return self.generate_response(messages)

