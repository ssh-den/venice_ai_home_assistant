"""Tests for the Venice AI service actions."""

from __future__ import annotations

from unittest.mock import MagicMock

from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.setup import async_setup_component
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.venice_ai.client import VeniceAIError
from custom_components.venice_ai.const import DOMAIN


async def test_services_registered_without_loaded_entry(
    hass: HomeAssistant, ha_core: None
) -> None:
    """Services are available as soon as the integration is set up."""
    assert await async_setup_component(hass, DOMAIN, {})
    assert hass.services.has_service(DOMAIN, "generate_image")
    assert hass.services.has_service(DOMAIN, "ai_task")


async def test_generate_image(
    hass: HomeAssistant, setup_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """The image service returns the first image without base64 payload."""
    result = await hass.services.async_call(
        DOMAIN,
        "generate_image",
        {"config_entry": setup_integration.entry_id, "prompt": "a cat"},
        blocking=True,
        return_response=True,
    )
    assert result == {"url": "https://example.com/a.png"}
    kwargs = mock_client.images.generate.call_args.kwargs
    assert kwargs["prompt"] == "a cat"
    assert kwargs["size"] == "1024x1024"


async def test_generate_image_api_error(
    hass: HomeAssistant, setup_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """API errors surface as HomeAssistantError."""
    mock_client.images.generate.side_effect = VeniceAIError("boom")
    with pytest.raises(HomeAssistantError, match="boom"):
        await hass.services.async_call(
            DOMAIN,
            "generate_image",
            {"config_entry": setup_integration.entry_id, "prompt": "a cat"},
            blocking=True,
            return_response=True,
        )


async def test_generate_image_entry_not_loaded(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    """Calling a service for an unloaded entry raises a validation error."""
    await hass.config_entries.async_unload(setup_integration.entry_id)
    assert setup_integration.state is ConfigEntryState.NOT_LOADED
    with pytest.raises(ServiceValidationError) as exc:
        await hass.services.async_call(
            DOMAIN,
            "generate_image",
            {"config_entry": setup_integration.entry_id, "prompt": "a cat"},
            blocking=True,
            return_response=True,
        )
    assert exc.value.translation_key == "config_entry_not_loaded"


async def test_generate_image_wrong_domain(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    """An entry of another integration is rejected."""
    other = MockConfigEntry(domain="other")
    other.add_to_hass(hass)
    with pytest.raises(ServiceValidationError) as exc:
        await hass.services.async_call(
            DOMAIN,
            "generate_image",
            {"config_entry": other.entry_id, "prompt": "a cat"},
            blocking=True,
            return_response=True,
        )
    assert exc.value.translation_key == "invalid_config_entry"


async def test_ai_task_plain_text(
    hass: HomeAssistant, setup_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """The ai_task service returns the model's text answer."""
    mock_client.chat.create_non_streaming.return_value = {
        "choices": [{"message": {"content": "Hello there"}}]
    }
    result = await hass.services.async_call(
        DOMAIN,
        "ai_task",
        {"config_entry": setup_integration.entry_id, "task": "Say hi"},
        blocking=True,
        return_response=True,
    )
    assert result is not None
    assert result["data"] == "Hello there"
    messages = mock_client.chat.create_non_streaming.call_args.kwargs["messages"]
    assert [m["content"] for m in messages if m["role"] == "user"] == ["Say hi"]


async def test_ai_task_structured(
    hass: HomeAssistant, setup_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """A structure request is parsed into JSON, tolerating code fences."""
    mock_client.chat.create_non_streaming.return_value = {
        "choices": [{"message": {"content": '```json\n{"temp": 21}\n```'}}]
    }
    result = await hass.services.async_call(
        DOMAIN,
        "ai_task",
        {
            "config_entry": setup_integration.entry_id,
            "task": "Give me the temperature",
            "structure": '{"temp": "number"}',
        },
        blocking=True,
        return_response=True,
    )
    assert result is not None
    assert result["data"] == {"temp": 21}


async def test_ai_task_invalid_json(
    hass: HomeAssistant, setup_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """Non-JSON output for a structured task raises an error."""
    mock_client.chat.create_non_streaming.return_value = {
        "choices": [{"message": {"content": "not json"}}]
    }
    with pytest.raises(HomeAssistantError, match="invalid JSON"):
        await hass.services.async_call(
            DOMAIN,
            "ai_task",
            {
                "config_entry": setup_integration.entry_id,
                "task": "Give me the temperature",
                "structure": '{"temp": "number"}',
            },
            blocking=True,
            return_response=True,
        )
