"""Tests for the Venice AI config and options flows."""

# pyright: reportTypedDictNotRequiredAccess=false

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from homeassistant import config_entries
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.venice_ai.client import AuthenticationError, NetworkError
from custom_components.venice_ai.config_flow import VeniceAIOptionsFlow
from custom_components.venice_ai.const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_TTS_MODEL,
    CONF_TTS_VOICE,
    DOMAIN,
    RECOMMENDED_CHAT_MODEL,
)

COMPONENT_DIR = Path(__file__).parents[1] / "custom_components" / DOMAIN
STRINGS = json.loads((COMPONENT_DIR / "strings.json").read_text())


async def test_user_flow(
    hass: HomeAssistant, ha_core: None, mock_client: MagicMock
) -> None:
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] is FlowResultType.FORM

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "secret"}
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"] == {CONF_API_KEY: "secret"}


@pytest.mark.parametrize(
    ("error", "key"),
    [
        (AuthenticationError("bad"), "invalid_auth"),
        (NetworkError("down"), "cannot_connect"),
        (RuntimeError("boom"), "unknown"),
    ],
)
async def test_user_flow_errors(
    hass: HomeAssistant,
    ha_core: None,
    mock_client: MagicMock,
    error: Exception,
    key: str,
) -> None:
    mock_client.validate_api_key.side_effect = error
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "secret"}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": key}


async def test_options_flow(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    result = await hass.config_entries.options.async_init(setup_integration.entry_id)
    assert result["type"] is FlowResultType.FORM

    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {
            CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL,
            CONF_MAX_TOKENS: 256,
            CONF_LLM_HASS_API: ["assist"],
            "tts_model_voice": "tts-kokoro → af_heart",
        },
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert setup_integration.options[CONF_MAX_TOKENS] == 256
    assert setup_integration.options[CONF_TTS_MODEL] == "tts-kokoro"
    assert setup_integration.options[CONF_TTS_VOICE] == "af_heart"


def test_options_range_validation() -> None:
    errors = VeniceAIOptionsFlow()._validate_numeric_options({CONF_MAX_TOKENS: 0})
    assert errors == {CONF_MAX_TOKENS: "max_tokens_out_of_range"}
    assert errors[CONF_MAX_TOKENS] in STRINGS["options"]["error"]


async def test_options_fields_are_translated(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    result = await hass.config_entries.options.async_init(setup_integration.entry_id)
    schema = result["data_schema"]
    assert schema is not None
    keys = {str(key) for key in schema.schema}
    step = STRINGS["options"]["step"]["init"]
    assert keys <= set(step["data"])
    assert keys <= set(step["data_description"])


def test_english_translation_matches_strings() -> None:
    english = json.loads((COMPONENT_DIR / "translations" / "en.json").read_text())
    assert english == STRINGS
