"""Tests for the tool schema conversion helpers in ``conversation.py``."""

from __future__ import annotations

from homeassistant.helpers import selector
import pytest

from custom_components.venice_ai.conversation import (
    _convert_schema_to_hashable,
    _format_venice_schema,
)


class TestFormatVeniceSchema:
    """Validate the per-key type mapping in ``_format_venice_schema``."""

    def test_string_mapping(self) -> None:
        schema = _format_venice_schema({"name": str})
        assert schema == {"name": {"type": "string"}}

    def test_int_mapping(self) -> None:
        schema = _format_venice_schema({"count": int})
        assert schema == {"count": {"type": "integer"}}

    def test_float_mapping(self) -> None:
        schema = _format_venice_schema({"score": float})
        assert schema == {"score": {"type": "number"}}

    def test_bool_mapping(self) -> None:
        schema = _format_venice_schema({"enabled": bool})
        assert schema == {"enabled": {"type": "boolean"}}

    def test_unknown_type_defaults_to_string(self) -> None:
        class Custom:
            pass

        schema = _format_venice_schema({"thing": Custom})
        assert schema == {"thing": {"type": "string"}}

    def test_multiple_keys(self) -> None:
        schema = _format_venice_schema({"a": str, "b": int})
        assert schema == {"a": {"type": "string"}, "b": {"type": "integer"}}

    def test_empty_schema(self) -> None:
        assert _format_venice_schema({}) == {}


class TestConvertSchemaToHashable:
    """Validate the hashable conversion used by ``voluptuous_openapi``."""

    def test_dict_stays_dict(self) -> None:
        assert _convert_schema_to_hashable({"a": str}) == {"a": str}

    def test_list_stays_list(self) -> None:
        assert _convert_schema_to_hashable([str, int]) == [str, int]

    def test_plain_type_passthrough(self) -> None:
        assert _convert_schema_to_hashable(str) is str
        assert _convert_schema_to_hashable(int) is int

    def test_nested_dict_and_list(self) -> None:
        assert _convert_schema_to_hashable({"items": [str]}) == {"items": [str]}

    def test_empty_dict(self) -> None:
        assert _convert_schema_to_hashable({}) == {}

    def test_selector_value_becomes_str(self) -> None:
        sel = selector.TextSelector()
        assert _convert_schema_to_hashable({"a": sel}) == {"a": str}


@pytest.mark.parametrize(
    ("input_value", "expected_type"),
    [
        (str, "string"),
        (int, "integer"),
        (float, "number"),
        (bool, "boolean"),
    ],
)
def test_schema_type_mapping_param(input_value: type, expected_type: str) -> None:
    """Parametrised regression test for the JSON-schema type mapping."""
    schema = _format_venice_schema({"value": input_value})
    assert schema["value"]["type"] == expected_type
