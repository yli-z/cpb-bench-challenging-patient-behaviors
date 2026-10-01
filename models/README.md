# Models

Unified LLM abstraction layer used across the project. All backends implement a common interface for generation.

## Supported backends

| Provider   | Module              | Examples                    |
|-----------|----------------------|-----------------------------|
| OpenAI    | openai_model.py      | gpt-4, gpt-4o, gpt-4o-mini  |
| Anthropic | claude_model.py      | claude-3-5-sonnet, etc.     |
| Google    | gemini_model.py      | gemini-2.5-flash, etc.     |
| DeepSeek  | deepseek_model.py    | deepseek-chat, deepseek-r1  |
| vLLM (remote) | remote_vllm_model.py | Qwen, Llama via API base |

## Usage

Use the factory so the rest of the code stays backend-agnostic:

```python
from models.model_utils import load_model

model = load_model("gpt-4o-mini", temperature=0.7, max_tokens=1024)
response = model.generate(messages)
```

For vLLM-hosted models (Qwen, Llama), pass `vllm_api_base` and optionally `enable_thinking`.

## Files

- **base_model.py** — Base interface.
- **model_utils.py** — `load_model()`, `create_model()`; routes by name to the right backend.
- **openai_model.py**, **claude_model.py**, **gemini_model.py**, **deepseek_model.py** — API clients.
- **remote_vllm_model.py** — vLLM server client.

## Environment

Set the corresponding API keys (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`) or configure the vLLM base URL for local models.
