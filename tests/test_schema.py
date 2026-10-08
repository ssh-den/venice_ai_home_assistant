"""Tests for formatting Home Assistant LLM tools for the Venice AI API."""

from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv, llm, selector
from homeassistant.util.json import JsonObjectType
import voluptuous as vol

from custom_components.venice_ai.conversation import _format_tool


class _Tool(llm.Tool):
    def __init__(self, parameters: vol.Schema, description: str | None) -> None:
        self.name = "test_tool"
        self.description = description
        self.parameters = parameters

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> JsonObjectType:
        return {}


def _params(schema: dict[Any, Any]) -> dict[str, Any]:
    tool = _Tool(vol.Schema(schema), "desc")
    return _format_tool(tool, llm.selector_serializer)["function"]["parameters"]


def test_function_envelope() -> None:
    tool = _Tool(vol.Schema({vol.Required("name"): str}), "Does things")
    formatted = _format_tool(tool, llm.selector_serializer)
    assert formatted["type"] == "function"
    assert formatted["function"]["name"] == "test_tool"
    assert formatted["function"]["description"] == "Does things"


def test_description_omitted_when_missing() -> None:
    tool = _Tool(vol.Schema({}), None)
    assert "description" not in _format_tool(tool, None)["function"]


def test_basic_types() -> None:
    params = _params(
        {
            vol.Required("name"): str,
            vol.Optional("count"): int,
            vol.Optional("ratio"): float,
            vol.Optional("enabled"): bool,
        }
    )
    assert params["type"] == "object"
    assert params["required"] == ["name"]
    assert params["properties"]["name"] == {"type": "string"}
    assert params["properties"]["count"] == {"type": "integer"}
    assert params["properties"]["ratio"] == {"type": "number"}
    assert params["properties"]["enabled"] == {"type": "boolean"}


def test_select_selector_becomes_enum() -> None:
    params = _params(
        {vol.Optional("mode"): selector.SelectSelector({"options": ["a", "b"]})}
    )
    assert params["properties"]["mode"]["enum"] == ["a", "b"]


def test_list_of_strings() -> None:
    params = _params({vol.Optional("names"): [cv.string]})
    assert params["properties"]["names"] == {
        "type": "array",
        "items": {"type": "string"},
    }
