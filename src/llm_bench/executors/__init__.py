"""Executors for AI prompts and MCP server integration."""

from llm_bench.executors.mcp_server import MCPServerConfig
from llm_bench.executors.prompt_executor import execute_prompt


__all__ = [
    "MCPServerConfig",
    "execute_prompt",
]
