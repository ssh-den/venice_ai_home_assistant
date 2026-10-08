"""Tests for Venice AI setup, unload and migration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.venice_ai.client import AuthenticationError, NetworkError
from custom_components.venice_ai.const import CONF_REQUEST_TIMEOUT, DOMAIN


async def test_setup_and_unload(
    hass: HomeAssistant, setup_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    assert setup_integration.state is ConfigEntryState.LOADED
    assert await hass.config_entries.async_unload(setup_integration.entry_id)
    assert setup_integration.state is ConfigEntryState.NOT_LOADED
    mock_client.close.assert_awaited_once()


async def test_setup_auth_failure(
    hass: HomeAssistant,
    ha_core: None,
    mock_config_entry: MockConfigEntry,
    mock_client: MagicMock,
) -> None:
    mock_client.validate_api_key.side_effect = AuthenticationError("bad key")
    assert not await hass.config_entries.async_setup(mock_config_entry.entry_id)
    assert mock_config_entry.state is ConfigEntryState.SETUP_ERROR
    assert any(mock_config_entry.async_get_active_flows(hass, {"reauth"}))


async def test_setup_not_ready(
    hass: HomeAssistant,
    ha_core: None,
    mock_config_entry: MockConfigEntry,
    mock_client: MagicMock,
) -> None:
    mock_client.validate_api_key.side_effect = NetworkError("down")
    assert not await hass.config_entries.async_setup(mock_config_entry.entry_id)
    assert mock_config_entry.state is ConfigEntryState.SETUP_RETRY


async def test_downgrade_is_rejected(
    hass: HomeAssistant, ha_core: None, mock_client: MagicMock
) -> None:
    entry = MockConfigEntry(domain=DOMAIN, data={CONF_API_KEY: "k"}, version=2)
    entry.add_to_hass(hass)
    assert not await hass.config_entries.async_setup(entry.entry_id)
    assert entry.state is ConfigEntryState.MIGRATION_ERROR


async def test_request_timeout_option_passed_to_client(
    hass: HomeAssistant, ha_core: None, mock_client: MagicMock
) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN, data={CONF_API_KEY: "k"}, options={CONF_REQUEST_TIMEOUT: 30}
    )
    entry.add_to_hass(hass)
    with patch(
        "custom_components.venice_ai.AsyncVeniceAIClient", return_value=mock_client
    ) as client_cls:
        assert await hass.config_entries.async_setup(entry.entry_id)
    assert client_cls.call_args.kwargs["timeout"] == 30.0


async def test_entity_unique_ids_are_stable(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    """Unique IDs must not change, or existing registry entries are orphaned."""
    entry_id = setup_integration.entry_id
    unique_ids = {
        entity.unique_id
        for entity in er.async_entries_for_config_entry(er.async_get(hass), entry_id)
    }
    assert {
        f"{entry_id}_conversation",
        f"{entry_id}_task",
        f"{entry_id}_tts",
        f"{entry_id}_stt",
        f"{entry_id}_request_count",
    } <= unique_ids
