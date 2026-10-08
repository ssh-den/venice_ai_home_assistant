"""Tests for the Venice AI AI Task entity."""

from __future__ import annotations

from unittest.mock import MagicMock

from homeassistant.components import ai_task
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry
import voluptuous as vol

from custom_components.venice_ai.const import CONF_CHAT_MODEL, DOMAIN

from .conftest import SCHEMA_MODEL


def _entity_id(hass: HomeAssistant, entry: MockConfigEntry) -> str:
    entity_id = er.async_get(hass).async_get_entity_id(
        "ai_task", DOMAIN, f"{entry.entry_id}_task"
    )
    assert entity_id is not None
    return entity_id


async def test_generate_structured_data(
    hass: HomeAssistant, setup_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """A vol.Schema structure is sent as JSON schema and the reply is parsed."""
    mock_client.chat.create_non_streaming.return_value = {
        "choices": [{"message": {"content": '<think>hm</think>{"name": "Bob"}'}}]
    }
    result = await ai_task.async_generate_data(
        hass,
        task_name="test",
        entity_id=_entity_id(hass, setup_integration),
        instructions="Make up a name",
        structure=vol.Schema({vol.Required("name"): str}),
    )
    assert result.data == {"name": "Bob"}

    messages = mock_client.chat.create_non_streaming.call_args.kwargs["messages"]
    assert messages[-1] == {"role": "user", "content": "Make up a name"}
    assert '"name"' in messages[-2]["content"]
    assert messages[-2]["role"] == "system"
    assert (
        mock_client.chat.create_non_streaming.call_args.kwargs["response_format"]
        is None
    )


async def test_generate_structured_data_native_schema(
    hass: HomeAssistant, setup_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """Models with response schema support get a native response_format."""
    hass.config_entries.async_update_entry(
        setup_integration, options={CONF_CHAT_MODEL: SCHEMA_MODEL}
    )
    mock_client.chat.create_non_streaming.return_value = {
        "choices": [{"message": {"content": '{"name": "Bob"}'}}]
    }
    result = await ai_task.async_generate_data(
        hass,
        task_name="test",
        entity_id=_entity_id(hass, setup_integration),
        instructions="Make up a name",
        structure=vol.Schema({vol.Required("name"): str}),
    )
    assert result.data == {"name": "Bob"}

    kwargs = mock_client.chat.create_non_streaming.call_args.kwargs
    assert kwargs["model"] == SCHEMA_MODEL
    assert kwargs["response_format"]["type"] == "json_schema"
    schema = kwargs["response_format"]["json_schema"]["schema"]
    assert schema["required"] == ["name"]
    assert all("JSON schema" not in m["content"] for m in kwargs["messages"])


async def test_generate_structured_data_invalid(
    hass: HomeAssistant, setup_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """Unparseable structured output raises an error."""
    mock_client.chat.create_non_streaming.return_value = {
        "choices": [{"message": {"content": "nope"}}]
    }
    with pytest.raises(HomeAssistantError, match="structured response"):
        await ai_task.async_generate_data(
            hass,
            task_name="test",
            entity_id=_entity_id(hass, setup_integration),
            instructions="Make up a name",
            structure=vol.Schema({vol.Required("name"): str}),
        )


async def test_generate_data_empty_choices(
    hass: HomeAssistant, setup_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """An empty API response raises an error."""
    mock_client.chat.create_non_streaming.return_value = {"choices": []}
    with pytest.raises(HomeAssistantError, match="Invalid Venice AI response"):
        await ai_task.async_generate_data(
            hass,
            task_name="test",
            entity_id=_entity_id(hass, setup_integration),
            instructions="Hi",
        )
