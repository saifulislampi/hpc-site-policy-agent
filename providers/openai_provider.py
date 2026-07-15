"""OpenAI Responses API implementation."""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI, OpenAIError
from pydantic import ValidationError

from models import ModelTurn, ProviderError, ToolCall, ToolDefinition, ToolResult
from providers.base import BaseProvider
from providers.openai_schema import openai_compatible_schema
from schemas import ExtractedPolicy


class OpenAIProvider(BaseProvider):
    provider_name = "openai"

    def __init__(
        self,
        *,
        model: str,
        timeout: float = 90.0,
        max_retries: int = 0,
    ) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise ProviderError(
                "OPENAI_API_KEY is not set. Export it or place it in a local .env file."
            )

        self.model = model
        self.client = OpenAI(timeout=timeout, max_retries=max_retries)
        self._conversation: list[Any] = []
        self._instructions: str | None = None
        self._tools: list[dict] = []
        self.last_extraction_usage: dict[str, int | None] = {
            "input_tokens": None,
            "output_tokens": None,
        }

    def start_agent(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        tools: list[ToolDefinition],
    ) -> ModelTurn:
        self._instructions = system_prompt
        self._tools = [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": openai_compatible_schema(tool.parameters),
                "strict": True,
            }
            for tool in tools
        ]
        self._conversation = [{"role": "user", "content": user_prompt}]
        return self._request_agent_turn()

    def continue_agent(
        self,
        *,
        tool_results: list[ToolResult],
        force_tool: str | None = None,
    ) -> ModelTurn:
        if self._instructions is None:
            raise ProviderError("Agent session was not started.")

        for result in tool_results:
            self._conversation.append(
                {
                    "type": "function_call_output",
                    "call_id": result.call_id,
                    "output": json.dumps(result.output, ensure_ascii=False),
                }
            )

        return self._request_agent_turn(force_tool=force_tool)

    def _request_agent_turn(self, *, force_tool: str | None = None) -> ModelTurn:
        request: dict[str, Any] = {
            "model": self.model,
            "instructions": self._instructions,
            "input": self._conversation,
            "tools": self._tools,
            "parallel_tool_calls": False,
            "store": False,
        }
        if force_tool is not None:
            request["tool_choice"] = {"type": "function", "name": force_tool}

        try:
            response = self.client.responses.create(**request)
        except OpenAIError as exc:
            raise ProviderError(f"OpenAI discovery request failed: {exc}") from exc

        # Preserve the model's native output items for the next Responses API turn.
        self._conversation.extend(response.output)
        return self._normalize_turn(response)

    def _normalize_turn(self, response: Any) -> ModelTurn:
        calls: list[ToolCall] = []

        for item in response.output:
            if getattr(item, "type", None) != "function_call":
                continue

            try:
                arguments = json.loads(item.arguments)
            except json.JSONDecodeError as exc:
                raise ProviderError(
                    f"Model returned invalid JSON arguments for {item.name}: {exc}"
                ) from exc

            calls.append(
                ToolCall(
                    call_id=item.call_id,
                    name=item.name,
                    arguments=arguments,
                )
            )

        usage = getattr(response, "usage", None)
        return ModelTurn(
            text=response.output_text or None,
            tool_calls=calls,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            response_id=getattr(response, "id", None),
        )

    def extract_report(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> ExtractedPolicy:
        report_tool = {
            "type": "function",
            "name": "submit_policy_report",
            "description": (
                "Submit the complete grounded HPC Slurm and networking policy report."
            ),
            "parameters": openai_compatible_schema(ExtractedPolicy.model_json_schema()),
            "strict": True,
        }

        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=system_prompt,
                input=[{"role": "user", "content": user_prompt}],
                tools=[report_tool],
                tool_choice={"type": "function", "name": "submit_policy_report"},
                parallel_tool_calls=False,
                store=False,
            )
        except OpenAIError as exc:
            raise ProviderError(f"OpenAI extraction request failed: {exc}") from exc

        usage = getattr(response, "usage", None)
        self.last_extraction_usage = {
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
        }

        function_calls = [
            item
            for item in response.output
            if getattr(item, "type", None) == "function_call"
            and getattr(item, "name", None) == "submit_policy_report"
        ]
        if len(function_calls) != 1:
            raise ProviderError(
                "Expected exactly one submit_policy_report function call."
            )

        try:
            data = json.loads(function_calls[0].arguments)
            return ExtractedPolicy.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ProviderError(f"Invalid structured policy report: {exc}") from exc
