"""Tests for Venice AI setup, unload and migration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er, issue_registry as ir
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.venice_ai.client import AuthenticationError, NetworkError
from custom_components.venice_ai.const import (
    CONF_CHAT_MODEL,
    CONF_IMAGE_MODEL,
    CONF_MAX_TOKENS,
    CONF_PRIVATE_MODELS_ONLY,
    CONF_PROMPT,
    CONF_RECOMMENDED,
    CONF_REQUEST_TIMEOUT,
    CONF_STT_MODEL,
    CONF_TTS_VOICE,
    DOMAIN,
    SUBENTRY_AI_TASK,
    SUBENTRY_CONVERSATION,
    SUBENTRY_STT,
    SUBENTRY_TTS,
)

from .conftest import SCHEMA_MODEL, SUBENTRY_IDS, update_subentry


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
    entry = MockConfigEntry(domain=DOMAIN, data={CONF_API_KEY: "k"}, version=3)
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
    entities = er.async_entries_for_config_entry(er.async_get(hass), entry_id)
    subentry_of = {entity.unique_id: entity.config_subentry_id for entity in entities}
    for subentry_id in SUBENTRY_IDS.values():
        assert subentry_of[subentry_id] == subentry_id
    assert subentry_of[f"{entry_id}_request_count"] is None


async def test_migrate_options_to_subentries(
    hass: HomeAssistant, ha_core: None, mock_client: MagicMock
) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Venice AI",
        data={CONF_API_KEY: "k"},
        options={
            CONF_PROMPT: "Be brief",
            CONF_LLM_HASS_API: ["assist"],
            CONF_CHAT_MODEL: SCHEMA_MODEL,
            CONF_MAX_TOKENS: 256,
            CONF_TTS_VOICE: "af_heart",
            CONF_STT_MODEL: "openai/whisper-large-v3",
            CONF_REQUEST_TIMEOUT: 30,
        },
    )
    entry.add_to_hass(hass)
    registry = er.async_get(hass)
    old = {
        platform: registry.async_get_or_create(
            platform, DOMAIN, f"{entry.entry_id}_{suffix}", config_entry=entry
        ).entity_id
        for platform, suffix in (
            ("conversation", "conversation"),
            ("ai_task", "task"),
            ("tts", "tts"),
            ("stt", "stt"),
        )
    }

    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    assert entry.version == 2
    assert entry.options == {CONF_REQUEST_TIMEOUT: 30}
    subentries = {s.subentry_type: s for s in entry.subentries.values()}
    assert subentries[SUBENTRY_CONVERSATION].data == {
        CONF_PROMPT: "Be brief",
        CONF_LLM_HASS_API: ["assist"],
        CONF_CHAT_MODEL: SCHEMA_MODEL,
        CONF_MAX_TOKENS: 256,
        CONF_RECOMMENDED: False,
    }
    assert subentries[SUBENTRY_AI_TASK].data == {
        CONF_CHAT_MODEL: SCHEMA_MODEL,
        CONF_MAX_TOKENS: 256,
        CONF_RECOMMENDED: False,
    }
    assert subentries[SUBENTRY_TTS].data[CONF_TTS_VOICE] == "af_heart"
    assert subentries[SUBENTRY_STT].data == {CONF_STT_MODEL: "openai/whisper-large-v3"}

    for subentry_type, platform in (
        (SUBENTRY_CONVERSATION, "conversation"),
        (SUBENTRY_AI_TASK, "ai_task"),
        (SUBENTRY_TTS, "tts"),
        (SUBENTRY_STT, "stt"),
    ):
        moved = registry.async_get(old[platform])
        assert moved is not None
        subentry_id = subentries[subentry_type].subentry_id
        assert moved.unique_id == subentry_id
        assert moved.config_subentry_id == subentry_id
        assert hass.states.get(old[platform]) is not None


async def test_model_issues_follow_the_model_list(
    hass: HomeAssistant, setup_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    registry = ir.async_get(hass)
    entry_id = setup_integration.entry_id
    update_subentry(
        hass,
        setup_integration,
        SUBENTRY_CONVERSATION,
        **{CONF_CHAT_MODEL: "schema-model"},
    )
    hass.config_entries.async_update_entry(
        setup_integration,
        options={CONF_PRIVATE_MODELS_ONLY: True, CONF_IMAGE_MODEL: "gone"},
    )
    await hass.async_block_till_done()

    conversation = SUBENTRY_IDS[SUBENTRY_CONVERSATION]
    not_private = f"not_private_model_{entry_id}_{conversation}"
    image_gone = f"unavailable_model_{entry_id}_{CONF_IMAGE_MODEL}"
    assert registry.async_get_issue(DOMAIN, not_private)
    assert registry.async_get_issue(DOMAIN, image_gone)
    assert not registry.async_get_issue(
        DOMAIN, f"not_private_model_{entry_id}_{SUBENTRY_IDS[SUBENTRY_TTS]}"
    )

    coordinator = setup_integration.runtime_data.coordinator
    data = dict(coordinator.data)
    data["text_models"] = [{"id": "schema-model", "model_spec": {"privacy": "private"}}]
    data["image_models"] = [{"id": "gone"}]
    coordinator.async_set_updated_data(data)

    assert not registry.async_get_issue(DOMAIN, not_private)
    assert not registry.async_get_issue(DOMAIN, image_gone)

    assert await hass.config_entries.async_unload(entry_id)
    assert not [i for d, i in registry.issues if d == DOMAIN and entry_id in i]
