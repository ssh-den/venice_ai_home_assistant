"""Fixtures for tests against the real Venice AI API.

The API key is read from ``VENICE_API_KEY`` or from ``.env`` in the repository
root. Without it the tests are skipped.
"""

# pylint: disable=redefined-outer-name

from __future__ import annotations

from collections.abc import AsyncGenerator
import os
from pathlib import Path

from homeassistant.config_entries import ConfigSubentryData
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry
import pytest_socket

from custom_components.venice_ai.const import (
    DEFAULT_AI_TASK_NAME,
    DEFAULT_CONVERSATION_NAME,
    DEFAULT_NAME,
    DEFAULT_STT_NAME,
    DEFAULT_TTS_NAME,
    DOMAIN,
    RECOMMENDED_AI_TASK_OPTIONS,
    RECOMMENDED_CONVERSATION_OPTIONS,
    RECOMMENDED_STT_OPTIONS,
    RECOMMENDED_TTS_OPTIONS,
    SUBENTRY_AI_TASK,
    SUBENTRY_CONVERSATION,
    SUBENTRY_STT,
    SUBENTRY_TTS,
)

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
        domain=DOMAIN,
        title=DEFAULT_NAME,
        data={CONF_API_KEY: API_KEY},
        version=2,
        minor_version=2,
        subentries_data=[
            ConfigSubentryData(
                subentry_type=subentry_type,
                title=title,
                data=data,
                unique_id=None,
            )
            for subentry_type, title, data in (
                (
                    SUBENTRY_CONVERSATION,
                    DEFAULT_CONVERSATION_NAME,
                    RECOMMENDED_CONVERSATION_OPTIONS,
                ),
                (SUBENTRY_AI_TASK, DEFAULT_AI_TASK_NAME, RECOMMENDED_AI_TASK_OPTIONS),
                (SUBENTRY_TTS, DEFAULT_TTS_NAME, RECOMMENDED_TTS_OPTIONS),
                (SUBENTRY_STT, DEFAULT_STT_NAME, RECOMMENDED_STT_OPTIONS),
            )
        ],
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    yield entry
    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()


async def update_subentry(
    hass: HomeAssistant, entry: MockConfigEntry, subentry_type: str, **data: object
) -> None:
    """Change the settings of the subentry of a type and wait for the reload."""
    subentry = next(
        s for s in entry.subentries.values() if s.subentry_type == subentry_type
    )
    hass.config_entries.async_update_subentry(
        entry, subentry, data={**subentry.data, **data}
    )
    await hass.async_block_till_done()
