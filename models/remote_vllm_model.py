"""
Remote vLLM model wrapper that connects to a remote vLLM service via OpenAI-compatible API.
Supports Qwen3 models with Thinking Mode and Non-Thinking Mode.

This class is designed for remote GPU servers and does not require local GPU.
It does not modify the existing VLLMModel class to avoid affecting other modules.
"""
import os
import re
from typing import Optional, Tuple
from openai import OpenAI
from models.base_model import BaseLLM


class RemoteVLLMModel(BaseLLM):
    """
    Remote vLLM model wrapper that connects to a remote vLLM service via OpenAI-compatible API.
    Supports Qwen3 models with Thinking Mode and Non-Thinking Mode.
    """
    
    def __init__(self, 
                 model_name: str,
                 vllm_api_base: Optional[str] = None,
                 max_tokens: Optional[int] = 4096,
                 temperature: Optional[float] = None,
                 enable_thinking: bool = False):
        """
        Initialize RemoteVLLMModel.
        
        Args:
            model_name: Model name (e.g., "Qwen/Qwen3-8B")
            vllm_api_base: Base URL for vLLM API server (e.g., "http://your-server:8000/v1")
                          If None, will use VLLM_API_BASE environment variable
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (None for default)
            enable_thinking: For Qwen3 models, enable thinking mode (default: False for non-thinking mode)
        """
        super().__init__(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        
        # Get API base URL from parameter, environment variable, or default
        self.vllm_api_base = vllm_api_base or os.getenv("VLLM_API_BASE")
        if not self.vllm_api_base:
            raise ValueError(
                "vLLM API base URL not provided. "
                "Please provide via --vllm-api-base argument or VLLM_API_BASE environment variable. "
                "Example: http://your-gpu-server:8000/v1"
            )
        
        # Ensure URL ends with /v1
        if not self.vllm_api_base.endswith("/v1"):
            if self.vllm_api_base.endswith("/"):
                self.vllm_api_base = self.vllm_api_base + "v1"
            else:
                self.vllm_api_base = self.vllm_api_base + "/v1"
        
        # Initialize OpenAI client for vLLM API
        self.client = OpenAI(
            api_key="EMPTY",  # vLLM doesn't require a real API key
            base_url=self.vllm_api_base,
        )
        
        # Check if this is a Qwen3 model
        self.is_qwen3 = "qwen3" in model_name.lower() or "qwen" in model_name.lower()
        
        # Store thinking mode setting (only used for Qwen3 models)
        self.enable_thinking = enable_thinking if self.is_qwen3 else False
        
        print(f">>>>>> Connected to vLLM service at {self.vllm_api_base}")
        print(f">>>>>> Model: {model_name}")
        if self.is_qwen3:
            mode_str = "Thinking Mode" if self.enable_thinking else "Non-Thinking Mode"
            print(f">>>>>> Qwen3 {mode_str}: enabled")

    def _split_thinking(self, content: str) -> Tuple[str, Optional[str]]:
        """
        Separate <think>...</think> from the actual response content.

        Returns:
            (response_content, thinking_content) where thinking_content is None
            if no thinking tags are present or model is in non-thinking mode.
        """
        # Empty thinking tags (non-thinking mode artifact)
        if content.startswith("<think>\n\n</think>\n\n"):
            return content.replace("<think>\n\n</think>\n\n", "", 1), None

        # Extract thinking content
        match = re.match(r'<think>(.*?)</think>\s*(.*)', content, re.DOTALL)
        if match:
            thinking = match.group(1).strip()
            response = match.group(2).strip()
            return response, thinking if thinking else None

        return content, None

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        # Build messages list
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Prepare request kwargs
        kwargs = {
            "model": self.model_name,
            "messages": messages,
        }
        
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        
        # For Qwen3 models, set thinking mode via extra_body
        if self.is_qwen3:
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking}
            }
        
        # Make API call
        response = self.client.chat.completions.create(**kwargs)
        raw_content = response.choices[0].message.content.strip()

        # Strip empty think tags (non-thinking mode artifact),
        # but keep raw content with <think> tags if thinking mode is enabled
        if not self.enable_thinking and raw_content.startswith("<think>\n\n</think>\n\n"):
            return raw_content.replace("<think>\n\n</think>\n\n", "", 1)
        return raw_content

    def generate_text_response(self, messages: list, split_think: bool = False):
        """
        Compatible interface for Doctor/Patient agents (takes full messages list).

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            split_think: If True, returns (content, thinking) tuple.
                        If False (default), returns content string only (legacy behavior).

        Returns:
            str if split_think=False, or (str, Optional[str]) tuple if split_think=True
        """
        # Prepare request kwargs
        kwargs = {
            "model": self.model_name,
            "messages": messages,
        }
        
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        
        # For Qwen3 models, set thinking mode via extra_body
        if self.is_qwen3:
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking}
            }
        
        # Make API call
        response = self.client.chat.completions.create(**kwargs)
        raw_content = response.choices[0].message.content.strip()

        if split_think:
            content, thinking = self._split_thinking(raw_content)
            return content, thinking
        # No split: strip empty think tags (non-thinking mode artifact),
        # but keep raw content with <think> tags if thinking mode is enabled
        if not self.enable_thinking and raw_content.startswith("<think>\n\n</think>\n\n"):
            return raw_content.replace("<think>\n\n</think>\n\n", "", 1)
        return raw_content

