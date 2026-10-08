"""Tests for Venice AI setup, unload and migration."""

from __future__ import annotations

from unittest.mock import MagicMock

from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.venice_ai.client import AuthenticationError, NetworkError
from custom_components.venice_ai.const import DOMAIN


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
    mock_client.models.list.side_effect = AuthenticationError("bad key")
    assert not await hass.config_entries.async_setup(mock_config_entry.entry_id)
    assert mock_config_entry.state is ConfigEntryState.SETUP_ERROR
    assert any(mock_config_entry.async_get_active_flows(hass, {"reauth"}))


async def test_setup_not_ready(
    hass: HomeAssistant,
    ha_core: None,
    mock_config_entry: MockConfigEntry,
    mock_client: MagicMock,
) -> None:
    mock_client.models.list.side_effect = NetworkError("down")
    assert not await hass.config_entries.async_setup(mock_config_entry.entry_id)
    assert mock_config_entry.state is ConfigEntryState.SETUP_RETRY


async def test_downgrade_is_rejected(
    hass: HomeAssistant, ha_core: None, mock_client: MagicMock
) -> None:
    entry = MockConfigEntry(domain=DOMAIN, data={CONF_API_KEY: "k"}, version=2)
    entry.add_to_hass(hass)
    assert not await hass.config_entries.async_setup(entry.entry_id)
    assert entry.state is ConfigEntryState.MIGRATION_ERROR
