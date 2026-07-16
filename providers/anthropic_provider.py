"""Placeholder for the future Claude adapter."""

from __future__ import annotations

from pydantic import BaseModel

from models import ModelTurn, ProviderError, ToolDefinition, ToolResult
from providers.base import BaseProvider


class AnthropicProvider(BaseProvider):
    provider_name = "anthropic"

    def __init__(self, *, model: str) -> None:
        self.model = model

    def start_agent(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        tools: list[ToolDefinition],
    ) -> ModelTurn:
        raise ProviderError("Anthropic provider is not implemented in the first POC.")

    def continue_agent(
        self,
        *,
        tool_results: list[ToolResult],
        force_tool: str | None = None,
    ) -> ModelTurn:
        raise ProviderError("Anthropic provider is not implemented in the first POC.")

    def extract_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[BaseModel],
        tool_name: str,
    ) -> BaseModel:
        raise ProviderError("Anthropic provider is not implemented in the first POC.")
