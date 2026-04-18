"""Prompt execution via litellm (unified interface to OpenAI, Anthropic, etc.)."""

from typing import Optional

import litellm
from loguru import logger

from llm_bench.config.base import BaseConfig
from llm_bench.executors.mcp_server import MCPServerConfig
from llm_bench.models.answers import QueryResult

# Suppress litellm's own verbose logging
litellm.suppress_debug_info = True


def _to_litellm_model(model_name: str) -> str:
    """Convert 'provider:model' to litellm's 'provider/model' format."""
    return model_name.replace(":", "/", 1)


def execute_prompt(prompt: str, config: BaseConfig, mcp_config: Optional["MCPServerConfig"] = None) -> QueryResult:
    """Execute a single prompt via litellm. Synchronous and thread-safe."""
    model = _to_litellm_model(config.model_name)

    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "timeout": config.llm_timeout,
    }
    if config.reasoning_effort:
        kwargs["reasoning_effort"] = config.reasoning_effort

    logger.debug(f"[litellm] Calling {model} (prompt: {len(prompt)} chars)...")
    response = litellm.completion(**kwargs)

    text = response.choices[0].message.content or ""
    usage = None
    if response.usage:
        usage = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return QueryResult(text=text, usage=usage, model_name=model)
