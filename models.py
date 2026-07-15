"""Provider-neutral runtime models for the agent loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ScoutError(RuntimeError):
    """Base exception for the project."""


class AgentError(ScoutError):
    """Raised when the agent loop cannot complete safely."""


class ProviderError(ScoutError):
    """Raised when a model provider returns an invalid response."""


class ToolExecutionError(ScoutError):
    """Raised when a tool cannot validate or execute a request."""


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(slots=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ToolResult:
    call_id: str
    name: str
    output: dict[str, Any]
    terminal: bool = False


@dataclass(slots=True)
class ModelTurn:
    text: str | None
    tool_calls: list[ToolCall]
    input_tokens: int | None = None
    output_tokens: int | None = None
    response_id: str | None = None
