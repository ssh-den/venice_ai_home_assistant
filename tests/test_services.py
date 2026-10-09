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
from custom_components.venice_ai.const import CONF_IMAGE_MODEL, DOMAIN


async def test_services_registered_without_loaded_entry(
    hass: HomeAssistant, ha_core: None
) -> None:
    """Services are available as soon as the integration is set up."""
    assert await async_setup_component(hass, DOMAIN, {})
    assert hass.services.has_service(DOMAIN, "generate_image")
    assert not hass.services.has_service(DOMAIN, "ai_task")


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


async def test_generate_image_uses_requested_model(
    hass: HomeAssistant, setup_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    await hass.services.async_call(
        DOMAIN,
        "generate_image",
        {
            "config_entry": setup_integration.entry_id,
            "prompt": "a cat",
            "model": "hidream",
        },
        blocking=True,
        return_response=True,
    )
    assert mock_client.images.generate.call_args.kwargs["model"] == "hidream"


async def test_generate_image_defaults_to_option(
    hass: HomeAssistant, setup_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    hass.config_entries.async_update_entry(
        setup_integration, options={CONF_IMAGE_MODEL: "venice-sd35"}
    )
    await hass.async_block_till_done()
    await hass.services.async_call(
        DOMAIN,
        "generate_image",
        {"config_entry": setup_integration.entry_id, "prompt": "a cat"},
        blocking=True,
        return_response=True,
    )
    assert mock_client.images.generate.call_args.kwargs["model"] == "venice-sd35"


async def test_generate_image_unknown_model(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    with pytest.raises(ServiceValidationError) as exc:
        await hass.services.async_call(
            DOMAIN,
            "generate_image",
            {
                "config_entry": setup_integration.entry_id,
                "prompt": "a cat",
                "model": "nope",
            },
            blocking=True,
            return_response=True,
        )
    assert exc.value.translation_key == "invalid_image_model"
