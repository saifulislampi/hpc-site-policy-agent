"""Abstract provider interface used by the agent."""

from __future__ import annotations

from abc import ABC, abstractmethod

from models import ModelTurn, ToolDefinition, ToolResult
from schemas import ExtractedPolicy


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
    def extract_report(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> ExtractedPolicy:
        """Run the non-agent structured extraction phase."""
