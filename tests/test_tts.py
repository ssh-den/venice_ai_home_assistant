"""Tests for the Venice AI TTS platform."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

from homeassistant.components.tts import ATTR_AUDIO_OUTPUT, ATTR_VOICE
from homeassistant.components.tts.const import DATA_COMPONENT
from homeassistant.components.tts.entity import TTSAudioRequest
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.venice_ai.const import (
    CONF_TTS_MODEL,
    CONF_TTS_SPEED,
    CONF_TTS_VOICE,
    RECOMMENDED_TTS_MODEL,
)
from custom_components.venice_ai.tts import VeniceAITTS


def _entity(hass: HomeAssistant) -> VeniceAITTS:
    entity = hass.data[DATA_COMPONENT].get_entity("tts.venice_ai_tts")
    assert isinstance(entity, VeniceAITTS)
    return entity


async def _message(*parts: str) -> AsyncGenerator[str]:
    for part in parts:
        yield part


async def test_get_tts_audio_uses_entry_options(
    hass: HomeAssistant, setup_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    hass.config_entries.async_update_entry(
        setup_integration,
        options={CONF_TTS_VOICE: "af_heart", CONF_TTS_SPEED: 1.5},
    )
    mock_client.speech.generate = AsyncMock(return_value=b"audio")

    result = await _entity(hass).async_get_tts_audio(
        "Hello", "en", {ATTR_AUDIO_OUTPUT: "wav"}
    )

    assert result == ("wav", b"audio")
    mock_client.speech.generate.assert_awaited_once_with(
        text="Hello",
        voice="af_heart",
        model=RECOMMENDED_TTS_MODEL,
        audio_output="wav",
        speed=1.5,
    )


async def test_get_tts_audio_empty(
    hass: HomeAssistant, setup_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    mock_client.speech.generate = AsyncMock(return_value=b"")
    with pytest.raises(HomeAssistantError, match="empty audio"):
        await _entity(hass).async_get_tts_audio("Hello", "en")


async def test_stream_tts_audio(
    hass: HomeAssistant, setup_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    async def _stream(**kwargs: object) -> AsyncGenerator[bytes]:
        assert kwargs["text"] == "Hello world"
        assert kwargs["voice"] == "bm_daniel"
        yield b"ab"
        yield b"cd"

    mock_client.speech.generate_streaming = _stream

    response = await _entity(hass).async_stream_tts_audio(
        TTSAudioRequest(
            language="en",
            options={ATTR_VOICE: "bm_daniel"},
            message_gen=_message("Hello ", "world"),
        )
    )

    assert response.extension == "mp3"
    assert b"".join([c async for c in response.data_gen]) == b"abcd"


async def test_stream_tts_audio_empty_message(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    with pytest.raises(HomeAssistantError, match="No TTS message"):
        await _entity(hass).async_stream_tts_audio(
            TTSAudioRequest(language="en", options={}, message_gen=_message())
        )


async def test_supported_voices_follow_active_model(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    entity = _entity(hass)
    voices = entity.async_get_supported_voices("en")
    assert voices is not None
    assert [v.voice_id for v in voices] == ["bm_daniel", "af_heart"]

    hass.config_entries.async_update_entry(
        setup_integration, options={CONF_TTS_MODEL: "unknown"}
    )
    voices = entity.async_get_supported_voices("en")
    assert voices is not None
    assert [v.voice_id for v in voices] == ["bm_daniel", "af_heart"]


async def test_default_options(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    options = _entity(hass).default_options
    assert options[ATTR_VOICE]
    assert options[CONF_TTS_MODEL] == RECOMMENDED_TTS_MODEL
