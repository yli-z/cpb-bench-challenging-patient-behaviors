"""
Gemini model wrapper.
"""

import os
import json
from typing import Optional

import google.generativeai as genai

from .base_model import BaseLLM


class GeminiModel(BaseLLM):
    """Wrapper for Google Gemini models."""

    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: Optional[str] = None, temperature: float = 0.7, max_tokens: Optional[int] = None):
        super().__init__(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
        genai.configure(api_key=api_key)
        self.model_name = model_name

    def _safe_response_text(self, response) -> str:
        """
        Safely extract text from a Gemini response.
        Returns empty string when the response is blocked by safety filters
        instead of raising an exception.
        """
        try:
            return response.text.strip()
        except Exception:
            # response.text raises when Gemini blocks the content (safety filter).
            # Log the finish reason / safety ratings if available.
            finish_reason = None
            safety_ratings = None
            try:
                candidate = response.candidates[0]
                finish_reason = candidate.finish_reason
                safety_ratings = candidate.safety_ratings
            except Exception:
                pass
            print(f"  [Gemini] Blocked response — finish_reason={finish_reason}, safety_ratings={safety_ratings}")
            return ""

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        # Configure generation without JSON schema - natural output
        generation_config = genai.GenerationConfig()
        if self.temperature is not None:
            generation_config.temperature = self.temperature
        generation_config.max_output_tokens = self.max_tokens or 4096
        
        # Gemini uses system_instruction parameter for system prompts
        model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config,
            system_instruction=system_prompt if system_prompt else None
        )
        
        response = model.generate_content(prompt)
        return self._safe_response_text(response)

    def generate_text_response(self, messages) -> str:
        # for messages format input
        assert len(messages) >= 1, "Messages should contain at least one message."
        assert messages[-1]["role"] == "user", "First message should be from user."
        user_prompt = messages[-1]["content"]

        if len(messages) > 1 and messages[0]["role"] == "system":
            system_prompt = messages[0]["content"]
        else:
            system_prompt = None
        
        # Configure generation without JSON schema - natural output
        generation_config = genai.GenerationConfig()
        if self.temperature is not None:
            generation_config.temperature = self.temperature
        generation_config.max_output_tokens = self.max_tokens or 4096
        
        # Gemini uses system_instruction parameter for system prompts
        model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config,
            system_instruction=system_prompt if system_prompt else None
        )
        
        response = model.generate_content(user_prompt)
        return self._safe_response_text(response)
