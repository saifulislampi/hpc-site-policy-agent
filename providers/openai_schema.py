"""Helpers for producing JSON Schemas accepted by OpenAI function tools."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


SUPPORTED_STRING_FORMATS = frozenset(
    {
        "date-time",
        "time",
        "date",
        "duration",
        "email",
        "hostname",
        "ipv4",
        "ipv6",
        "uuid",
    }
)


def openai_compatible_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove string formats unsupported by OpenAI Structured Outputs."""

    compatible = deepcopy(schema)
    _remove_unsupported_formats(compatible)
    return compatible


def _remove_unsupported_formats(value: Any) -> None:
    if isinstance(value, dict):
        schema_format = value.get("format")
        if (
            isinstance(schema_format, str)
            and schema_format not in SUPPORTED_STRING_FORMATS
        ):
            value.pop("format")
        for child in value.values():
            _remove_unsupported_formats(child)
    elif isinstance(value, list):
        for child in value:
            _remove_unsupported_formats(child)

