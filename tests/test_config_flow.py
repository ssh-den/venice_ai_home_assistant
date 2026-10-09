"""Tests for the Venice AI config and options flows."""

# pyright: reportTypedDictNotRequiredAccess=false

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from homeassistant import config_entries
from homeassistant.config_entries import SubentryFlowContext
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.venice_ai.client import AuthenticationError, NetworkError
from custom_components.venice_ai.const import (
    CONF_CHAT_MODEL,
    CONF_IMAGE_MODEL,
    CONF_MAX_HISTORY_MESSAGES,
    CONF_MAX_TOKENS,
    CONF_PRIVATE_MODELS_ONLY,
    CONF_PROMPT,
    CONF_RECOMMENDED,
    CONF_REQUEST_TIMEOUT,
    CONF_STRUCTURE_PROMPT,
    CONF_STT_MODEL,
    CONF_THINKING_TAGS,
    CONF_TTS_MODEL,
    CONF_TTS_SPEED,
    CONF_TTS_VOICE,
    CONF_VENICE_SYSTEM_PROMPT,
    DOMAIN,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_STT_MODEL,
    RECOMMENDED_TTS_MODEL,
    SUBENTRY_AI_TASK,
    SUBENTRY_CONVERSATION,
    SUBENTRY_STT,
    SUBENTRY_TTS,
)

from .conftest import SCHEMA_MODEL, SUBENTRY_IDS, WAV_TTS_MODEL

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
    entry = result["result"]
    assert entry.version == 2
    assert sorted(s.subentry_type for s in entry.subentries.values()) == sorted(
        [SUBENTRY_AI_TASK, SUBENTRY_CONVERSATION, SUBENTRY_STT, SUBENTRY_TTS]
    )


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


def _options(result: Any, key: str) -> dict[str, str]:
    schema = result["data_schema"]
    assert schema is not None
    selector = next(v for k, v in schema.schema.items() if str(k) == key)
    return {o["value"]: o["label"] for o in selector.config["options"]}


def _fields(result: Any) -> set[str]:
    schema = result["data_schema"]
    assert schema is not None
    return {str(key) for key in schema.schema}


def _assert_translated(result: Any, section: dict[str, Any]) -> None:
    step = section["step"][result["step_id"]]
    assert _fields(result) <= set(step["data"])


async def _start(
    hass: HomeAssistant, entry: MockConfigEntry, subentry_type: str, new: bool = False
) -> Any:
    context = SubentryFlowContext(source=config_entries.SOURCE_USER)
    if not new:
        context = SubentryFlowContext(
            source=config_entries.SOURCE_RECONFIGURE,
            subentry_id=SUBENTRY_IDS[subentry_type],
        )
    return await hass.config_entries.subentries.async_init(
        (entry.entry_id, subentry_type), context=context
    )


async def test_options_flow(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    result = await hass.config_entries.options.async_init(setup_integration.entry_id)
    assert result["type"] is FlowResultType.FORM
    _assert_translated(result, STRINGS["options"])
    assert set(_options(result, CONF_IMAGE_MODEL)) == {
        "default",
        "venice-sd35",
        "hidream",
    }

    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {CONF_PRIVATE_MODELS_ONLY: True, CONF_REQUEST_TIMEOUT: 30},
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert setup_integration.options == {
        CONF_PRIVATE_MODELS_ONLY: True,
        CONF_REQUEST_TIMEOUT: 30,
    }


async def test_add_conversation_with_recommended_settings(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    result = await _start(hass, setup_integration, SUBENTRY_CONVERSATION, new=True)
    assert result["step_id"] == "init"
    _assert_translated(result, STRINGS["config_subentries"][SUBENTRY_CONVERSATION])
    labels = _options(result, CONF_CHAT_MODEL)
    assert labels[RECOMMENDED_CHAT_MODEL] == (
        f"{RECOMMENDED_CHAT_MODEL} (TEE, tools, $0.18/$0.37 per 1M)"
    )
    assert labels[SCHEMA_MODEL] == f"{SCHEMA_MODEL} (Anonymized)"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            "name": "Kitchen",
            CONF_PROMPT: "Be brief",
            CONF_CHAT_MODEL: SCHEMA_MODEL,
            CONF_RECOMMENDED: True,
        },
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "Kitchen"
    assert result["data"] == {
        CONF_PROMPT: "Be brief",
        CONF_CHAT_MODEL: SCHEMA_MODEL,
        CONF_RECOMMENDED: True,
    }


async def test_reconfigure_conversation_advanced(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    result = await _start(hass, setup_integration, SUBENTRY_CONVERSATION)
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL, CONF_RECOMMENDED: False},
    )
    assert result["step_id"] == "advanced"
    _assert_translated(result, STRINGS["config_subentries"][SUBENTRY_CONVERSATION])
    assert CONF_STRUCTURE_PROMPT not in _fields(result)

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_MAX_TOKENS: 256,
            CONF_THINKING_TAGS: "thought",
            CONF_VENICE_SYSTEM_PROMPT: True,
            CONF_MAX_HISTORY_MESSAGES: 10,
        },
    )
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    data = setup_integration.subentries[SUBENTRY_IDS[SUBENTRY_CONVERSATION]].data
    assert data[CONF_PROMPT] == ""
    assert data[CONF_MAX_TOKENS] == 256
    assert data[CONF_THINKING_TAGS] == "thought"
    assert data[CONF_VENICE_SYSTEM_PROMPT] is True
    assert CONF_LLM_HASS_API not in data


async def test_ai_task_advanced_has_structure_prompt(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    result = await _start(hass, setup_integration, SUBENTRY_AI_TASK)
    assert CONF_PROMPT not in _fields(result)
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL, CONF_RECOMMENDED: False},
    )
    _assert_translated(result, STRINGS["config_subentries"][SUBENTRY_AI_TASK])
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_STRUCTURE_PROMPT: "JSON please:"}
    )
    assert result["reason"] == "reconfigure_successful"
    data = setup_integration.subentries[SUBENTRY_IDS[SUBENTRY_AI_TASK]].data
    assert data[CONF_STRUCTURE_PROMPT] == "JSON please:"


async def test_tts_model_then_voice(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    result = await _start(hass, setup_integration, SUBENTRY_TTS)
    _assert_translated(result, STRINGS["config_subentries"][SUBENTRY_TTS])
    labels = _options(result, CONF_TTS_MODEL)
    assert labels[RECOMMENDED_TTS_MODEL] == "tts-kokoro (Private, 2 languages, $3.5 per 1M chars)"
    assert WAV_TTS_MODEL in labels
    assert "voice" not in str(labels)

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_TTS_MODEL: RECOMMENDED_TTS_MODEL}
    )
    assert result["step_id"] == "voice"
    _assert_translated(result, STRINGS["config_subentries"][SUBENTRY_TTS])
    assert _options(result, CONF_TTS_VOICE) == {
        "bm_daniel": "bm_daniel (en)",
        "af_heart": "af_heart (en)",
        "jf_alpha": "jf_alpha (ja)",
    }

    result = await _start(hass, setup_integration, SUBENTRY_TTS)
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_TTS_MODEL: WAV_TTS_MODEL}
    )
    assert list(_options(result, CONF_TTS_VOICE)) == ["tara"]
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_TTS_VOICE: "tara", CONF_TTS_SPEED: 1.5}
    )
    assert result["reason"] == "reconfigure_successful"
    data = setup_integration.subentries[SUBENTRY_IDS[SUBENTRY_TTS]].data
    assert data == {
        CONF_TTS_MODEL: WAV_TTS_MODEL,
        CONF_TTS_VOICE: "tara",
        CONF_TTS_SPEED: 1.5,
    }


async def test_stt_model(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    result = await _start(hass, setup_integration, SUBENTRY_STT, new=True)
    _assert_translated(result, STRINGS["config_subentries"][SUBENTRY_STT])
    assert _options(result, CONF_STT_MODEL) == {
        RECOMMENDED_STT_MODEL: f"Parakeet ASR ({RECOMMENDED_STT_MODEL}, Private, $0.006 per min)"
    }
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {"name": "Bedroom STT", CONF_STT_MODEL: RECOMMENDED_STT_MODEL},
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"] == {CONF_STT_MODEL: RECOMMENDED_STT_MODEL}


async def test_private_models_only_filters_selectors(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    hass.config_entries.async_update_entry(
        setup_integration, options={CONF_PRIVATE_MODELS_ONLY: True}
    )
    await hass.async_block_till_done()

    result = await _start(hass, setup_integration, SUBENTRY_CONVERSATION)
    assert set(_options(result, CONF_CHAT_MODEL)) == {RECOMMENDED_CHAT_MODEL}
    result = await _start(hass, setup_integration, SUBENTRY_TTS)
    assert set(_options(result, CONF_TTS_MODEL)) == {RECOMMENDED_TTS_MODEL}
    result = await _start(hass, setup_integration, SUBENTRY_STT)
    assert set(_options(result, CONF_STT_MODEL)) == {RECOMMENDED_STT_MODEL}


async def test_subentry_needs_loaded_entry(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    assert await hass.config_entries.async_unload(setup_integration.entry_id)
    for subentry_type in SUBENTRY_IDS:
        result = await _start(hass, setup_integration, subentry_type)
        assert result["type"] is FlowResultType.ABORT
        assert result["reason"] == "entry_not_loaded"


def test_strings_have_no_markup() -> None:
    """The frontend parses translations as ICU messages, so tags break them."""
    text = (COMPONENT_DIR / "strings.json").read_text()
    assert "<" not in text
    assert ">" not in text


def test_english_translation_matches_strings() -> None:
    english = json.loads((COMPONENT_DIR / "translations" / "en.json").read_text())
    assert english == STRINGS
