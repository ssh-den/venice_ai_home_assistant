"""Tests for the Venice AI diagnostic sensors."""

from __future__ import annotations

from unittest.mock import MagicMock

from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.venice_ai.const import DOMAIN


def _state(hass: HomeAssistant, entry: MockConfigEntry, key: str) -> str:
    entity_id = er.async_get(hass).async_get_entity_id(
        "sensor", DOMAIN, f"{entry.entry_id}_{key}"
    )
    assert entity_id is not None
    state = hass.states.get(entity_id)
    assert state is not None
    return state.state


async def test_sensors_have_translated_names(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    entity_id = er.async_get(hass).async_get_entity_id(
        "sensor", DOMAIN, f"{setup_integration.entry_id}_request_count"
    )
    assert entity_id == "sensor.venice_ai_api_requests"


async def test_sensors_update_on_metric_change(
    hass: HomeAssistant, setup_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    metrics = mock_client.metrics
    before = int(_state(hass, setup_integration, "request_count"))

    metrics.record_request()
    metrics.record_usage(
        {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    )
    metrics.record_error(RuntimeError("boom"))
    await hass.async_block_till_done()

    assert int(_state(hass, setup_integration, "request_count")) == before + 1
    assert _state(hass, setup_integration, "total_tokens") == "3"
    assert _state(hass, setup_integration, "error_count") == "1"
    assert _state(hass, setup_integration, "last_error") == "RuntimeError: boom"


async def test_listener_removed_on_unload(
    hass: HomeAssistant, setup_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    assert mock_client.metrics._listeners
    await hass.config_entries.async_unload(setup_integration.entry_id)
    assert not mock_client.metrics._listeners
