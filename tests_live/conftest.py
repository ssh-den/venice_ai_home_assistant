"""Fixtures for tests against the real Venice AI API.

The API key is read from ``VENICE_API_KEY`` or from ``.env`` in the repository
root. Without it the tests are skipped.
"""

# pylint: disable=redefined-outer-name

from __future__ import annotations

from collections.abc import AsyncGenerator
import os
from pathlib import Path

from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry
import pytest_socket

from custom_components.venice_ai.const import DOMAIN

_ENV_FILE = Path(__file__).parents[1] / ".env"


def _api_key() -> str | None:
    if key := os.environ.get("VENICE_API_KEY"):
        return key
    if _ENV_FILE.is_file():
        for line in _ENV_FILE.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "VENICE_API_KEY" and value.strip():
                return value.strip().strip("'\"")
    return None


API_KEY = _api_key()


@pytest.fixture(autouse=True)
def allow_network(enable_custom_integrations: None) -> None:
    """Lift the socket restrictions of the Home Assistant test plugin."""
    pytest_socket._remove_restrictions()  # pylint: disable=protected-access


@pytest.fixture
async def live_entry(hass: HomeAssistant) -> AsyncGenerator[MockConfigEntry]:
    """Set up the integration with the real API key."""
    if not API_KEY:
        pytest.skip("VENICE_API_KEY is not set")
    assert await async_setup_component(hass, "homeassistant", {})
    entry = MockConfigEntry(
        domain=DOMAIN, title="Venice AI", data={CONF_API_KEY: API_KEY}, options={}
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    yield entry
    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
