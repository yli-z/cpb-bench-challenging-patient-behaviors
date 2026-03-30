from models.openai_model import OpenAIModel
from models.claude_model import ClaudeModel
# from models.gemini_model import GeminiModel
# from models.deepseek_model import DeepSeekModel
from models.remote_vllm_model import RemoteVLLMModel


def load_model(model_name, gpu_memory_utilization=None, seed=None, num_samples=None,
               temperature=None, max_tokens=None, vllm_api_base=None, enable_thinking=False):
    model_lower = model_name.lower()

    kwargs = {}
    if temperature is not None:
        kwargs['temperature'] = temperature
    if max_tokens is not None:
        kwargs['max_tokens'] = max_tokens

    if "gpt" in model_lower:
        return OpenAIModel(model_name=model_name, **kwargs)
    elif "claude" in model_lower:
        return ClaudeModel(model_name=model_name, **kwargs)
    elif "gemini" in model_lower:
        from models.gemini_model import GeminiModel
        model = GeminiModel(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
    elif "deepseek" in model_lower:
        from models.deepseek_model import DeepSeekModel
        model = DeepSeekModel(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
    elif "qwen" in model_lower or "llama" in model_lower:
        if vllm_api_base is not None:
            kwargs['vllm_api_base'] = vllm_api_base
        return RemoteVLLMModel(model_name=model_name, enable_thinking=enable_thinking, **kwargs)
    else:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Supported models: gpt-*, claude-*, gemini-*, deepseek-*, qwen*, llama*"
        )


def create_model(model_name: str, generation_config: dict):
    config = generation_config or {}
    kwargs = {k: config[k] for k in ("gpu_memory_utilization", "seed", "num_samples",
                                      "temperature", "max_tokens", "vllm_api_base")
              if config.get(k) is not None}
    kwargs.setdefault("enable_thinking", config.get("enable_thinking", False))
    return load_model(model_name=model_name, **kwargs)
