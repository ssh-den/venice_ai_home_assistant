"""Tests for the Venice AI diagnostics."""

from __future__ import annotations

from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.venice_ai.const import CONF_PROMPT
from custom_components.venice_ai.diagnostics import (
    async_get_config_entry_diagnostics,
)


async def test_diagnostics_hide_secrets_and_prompt(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    hass.config_entries.async_update_entry(
        setup_integration, options={CONF_PROMPT: "I live at 1 Main Street"}
    )

    diagnostics = await async_get_config_entry_diagnostics(hass, setup_integration)

    assert diagnostics["data"]["api_key"] == "***-key"
    assert diagnostics["options"][CONF_PROMPT] == "**REDACTED**"
    coordinator = diagnostics["coordinator"]
    assert coordinator["text_models_count"] == 2
    assert coordinator["unlisted_speech_models"] == ["tts-wav-only"]
