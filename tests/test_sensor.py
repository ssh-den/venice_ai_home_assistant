"""Tests for the usage sensors of the Venice AI services."""

from __future__ import annotations

from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr, entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.venice_ai.const import (
    DOMAIN,
    SUBENTRY_CONVERSATION,
    SUBENTRY_TTS,
)

from .conftest import SUBENTRY_IDS


def _state(hass: HomeAssistant, subentry_type: str, key: str) -> str:
    entity_id = er.async_get(hass).async_get_entity_id(
        "sensor", DOMAIN, f"{SUBENTRY_IDS[subentry_type]}_{key}"
    )
    assert entity_id is not None
    state = hass.states.get(entity_id)
    assert state is not None
    return state.state


async def test_sensors_belong_to_each_service(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    subentry_id = SUBENTRY_IDS[SUBENTRY_CONVERSATION]
    entity = er.async_get(hass).async_get(
        "sensor.venice_ai_conversation_api_requests"
    )
    assert entity is not None
    assert entity.config_subentry_id == subentry_id
    device = dr.async_get(hass).async_get_device(identifiers={(DOMAIN, subentry_id)})
    assert device is not None
    assert entity.device_id == device.id
    assert (
        dr.async_get(hass).async_get_device(
            identifiers={(DOMAIN, setup_integration.entry_id)}
        )
        is None
    )


async def test_sensors_count_their_service_only(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    metrics = setup_integration.runtime_data.metrics[
        SUBENTRY_IDS[SUBENTRY_CONVERSATION]
    ]
    metrics.record_request()
    metrics.record_usage(
        {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    )
    metrics.record_error(RuntimeError("boom"))
    await hass.async_block_till_done()

    assert _state(hass, SUBENTRY_CONVERSATION, "request_count") == "1"
    assert _state(hass, SUBENTRY_CONVERSATION, "total_tokens") == "3"
    assert _state(hass, SUBENTRY_CONVERSATION, "error_count") == "1"
    assert _state(hass, SUBENTRY_CONVERSATION, "last_error") == "RuntimeError: boom"
    assert _state(hass, SUBENTRY_TTS, "request_count") == "0"
    assert (
        er.async_get(hass).async_get_entity_id(
            "sensor", DOMAIN, f"{SUBENTRY_IDS[SUBENTRY_TTS]}_total_tokens"
        )
        is None
    )


async def test_listeners_removed_on_unload(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    all_metrics = setup_integration.runtime_data.metrics.values()
    assert all(m._listeners for m in all_metrics)
    await hass.config_entries.async_unload(setup_integration.entry_id)
    assert not any(m._listeners for m in all_metrics)
