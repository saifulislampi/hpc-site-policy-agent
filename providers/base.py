"""Abstract provider interface used by the agent."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel

from models import ModelTurn, ToolDefinition, ToolResult


class BaseProvider(ABC):
    provider_name = "unknown"

    @abstractmethod
    def start_agent(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        tools: list[ToolDefinition],
    ) -> ModelTurn:
        """Start a new tool-calling agent session."""

    @abstractmethod
    def continue_agent(
        self,
        *,
        tool_results: list[ToolResult],
        force_tool: str | None = None,
    ) -> ModelTurn:
        """Continue the session, optionally forcing one named tool call."""

    @abstractmethod
    def extract_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[BaseModel],
        tool_name: str,
    ) -> BaseModel:
        """Extract one bounded field group using a provider-neutral schema."""

        raise NotImplementedError
