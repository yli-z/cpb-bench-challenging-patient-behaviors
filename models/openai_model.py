"""
OpenAI (GPT-4 / GPT-5) model wrapper.
"""

import json
import os
import re
from typing import Optional
import asyncio

from openai import OpenAI, AsyncOpenAI

from .base_model import BaseLLM


class OpenAIModel(BaseLLM):
    """Wrapper for OpenAI models (GPT-4, GPT-5)."""

    def __init__(self, model_name: str = "gpt-4", api_key: Optional[str] = None, temperature: float = 0.7, max_tokens: Optional[int] = None):
        super().__init__(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.async_client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def _supports_structured_outputs(self) -> bool:
        name_lower = self.model_name.lower()
        return bool(re.match(r'gpt-(4o|4-turbo|3\.5-turbo)', name_lower))

    def generate_response(self, prompt, system_prompt=None) -> str:
        # only single turn response supported

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

        if self.temperature is not None:
            if self.model_name.startswith("gpt-5"):
                if self.temperature == 1 or self.temperature == 1.0:
                    kwargs["temperature"] = self.temperature
            else:
                kwargs["temperature"] = self.temperature
        
        # gpt-5 doesn't support max_tokens parameter
        if not self.model_name.startswith("gpt-5"):
            kwargs["max_tokens"] = self.max_tokens or 4096

        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content.strip()
        return content
        
    async def async_generate_response(self, messages) -> str:
        # only single turn response supported
        kwargs = {
            "model": self.model_name,
            "messages": messages,
        }

        if self.temperature is not None:
            if self.model_name.startswith("gpt-5"):
                if self.temperature == 1 or self.temperature == 1.0:
                    kwargs["temperature"] = self.temperature
            else:
                kwargs["temperature"] = self.temperature
        
        # gpt-5 doesn't support max_tokens parameter
        if not self.model_name.startswith("gpt-5") and self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        cur_sleep = 0.8
        max_retries = 5
        last_err = None

        for attempt in range(max_retries + 1):
            try:
                response = await self.async_client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                return (content or "").strip()
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    await asyncio.sleep(cur_sleep)
                    cur_sleep *= 2
                else:
                    raise

    def generate_text_response(self, messages) -> str:
        # only single turn response supported
        # for messages format input
        kwargs = {
            "model": self.model_name,
            "messages": messages,
        }
        if self.temperature is not None:
            if self.model_name.startswith("gpt-5"):
                if self.temperature == 1 or self.temperature == 1.0:
                    kwargs["temperature"] = self.temperature
            else:
                kwargs["temperature"] = self.temperature
        
        # gpt-5 doesn't support max_tokens parameter
        if not self.model_name.startswith("gpt-5") and self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()

    def generate_json_evaluation_response(
        self, prompt: str, system_prompt: Optional[str] = None, schema: Optional[dict] = None
    ) -> str:
        """
        Generate a JSON response using OpenAI's official JSON output format.

        Uses ``response_format={"type": "json_schema", ...}`` when a *schema* is
        provided (structured outputs) and falls back to
        ``response_format={"type": "json_object"}`` otherwise.

        Args:
            prompt: User prompt string.
            system_prompt: Optional system prompt.
            schema: Optional JSON schema dict (OVERREACTION_JSON_SCHEMA style).

        Returns:
            Raw JSON string returned by the model.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if schema is not None:
            response_format = {"type": "json_schema", "json_schema": schema}
        else:
            response_format = {"type": "json_object"}

        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "response_format": response_format,
        }

        if self.temperature is not None:
            if self.model_name.startswith("gpt-5"):
                if self.temperature == 1 or self.temperature == 1.0:
                    kwargs["temperature"] = self.temperature
            else:
                kwargs["temperature"] = self.temperature

        if not self.model_name.startswith("gpt-5"):
            kwargs["max_tokens"] = self.max_tokens or 4096

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()

    def generate_json_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        json_schema = {
            "type": "object",
            "properties": {
                "response": {"type": "string", "description": "The doctor's response to the patient"}
            },
            "required": ["response"]
        }

        kwargs = {
            "model": self.model_name,
            "messages": messages,
        }

        if self._supports_structured_outputs():
            kwargs["response_format"] = {"type": "json_schema", "json_schema": json_schema}
            user_content = prompt
        else:
            user_content = prompt + "\n\nPlease respond in JSON format with a 'response' field containing your answer."
        
        messages.append({"role": "user", "content": user_content})

        if self.temperature is not None:
            if self.model_name.startswith("gpt-5"):
                if self.temperature == 1 or self.temperature == 1.0:
                    kwargs["temperature"] = self.temperature
            else:
                kwargs["temperature"] = self.temperature
        
        # gpt-5 doesn't support max_tokens parameter
        if not self.model_name.startswith("gpt-5") and self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content.strip()
        
        if self._supports_structured_outputs():
            response_json = json.loads(content)
            return response_json.get("response", "")
        else:
            json_match = re.search(r'\{[^{}]*"response"[^{}]*\}', content, re.DOTALL)
            if json_match:
                response_json = json.loads(json_match.group())
                return response_json.get("response", "")
            return content

