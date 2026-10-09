"""Tests for the Venice AI TTS platform."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

from homeassistant.components.tts import ATTR_VOICE
from homeassistant.components.tts.const import DATA_COMPONENT
from homeassistant.components.tts.entity import TTSAudioRequest
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.venice_ai.audio import pcm_to_wav, wav_parts
from custom_components.venice_ai.const import (
    CONF_TTS_MODEL,
    CONF_TTS_SPEED,
    CONF_TTS_VOICE,
    RECOMMENDED_TTS_MODEL,
)
from custom_components.venice_ai.tts import VeniceAITTS

from .conftest import WAV_TTS_MODEL


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

    result = await _entity(hass).async_get_tts_audio("Hello", "en-US")

    assert result == ("mp3", b"audio")
    mock_client.speech.generate.assert_awaited_once_with(
        text="Hello",
        voice="af_heart",
        model=RECOMMENDED_TTS_MODEL,
        audio_output="mp3",
        speed=1.5,
        language=None,
    )


async def test_get_tts_audio_splits_long_messages(
    hass: HomeAssistant, setup_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    mock_client.speech.generate = AsyncMock(side_effect=[b"one", b"two"])
    message = "A" * 3000 + ". " + "B" * 3000 + "."

    result = await _entity(hass).async_get_tts_audio(message, "en")

    assert result == ("mp3", b"onetwo")
    texts = [c.kwargs["text"] for c in mock_client.speech.generate.call_args_list]
    assert all(len(text) <= 4096 for text in texts)


async def test_wav_only_model_joins_audio(
    hass: HomeAssistant, setup_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    hass.config_entries.async_update_entry(
        setup_integration, options={CONF_TTS_MODEL: WAV_TTS_MODEL}
    )
    mock_client.speech.generate = AsyncMock(
        side_effect=[pcm_to_wav(b"\x01\x02"), pcm_to_wav(b"\x03\x04")]
    )

    extension, data = await _entity(hass).async_get_tts_audio(
        "First. " + "x" * 4095, "ru"
    )

    assert extension == "wav"
    assert data is not None
    assert wav_parts(data)[1] == b"\x01\x02\x03\x04"
    assert mock_client.speech.generate.call_args.kwargs["audio_output"] == "wav"


async def test_get_tts_audio_empty(
    hass: HomeAssistant, setup_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    mock_client.speech.generate = AsyncMock(return_value=b"")
    with pytest.raises(HomeAssistantError, match="empty audio"):
        await _entity(hass).async_get_tts_audio("Hello", "en")


async def test_stream_tts_audio_by_sentence(
    hass: HomeAssistant, setup_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    texts: list[str] = []

    async def _stream(**kwargs: object) -> AsyncGenerator[bytes]:
        texts.append(str(kwargs["text"]))
        assert kwargs["voice"] == "bm_daniel"
        assert kwargs["language"] is None
        yield f"<{len(texts)}>".encode()

    mock_client.speech.generate_streaming = _stream

    response = await _entity(hass).async_stream_tts_audio(
        TTSAudioRequest(
            language="en",
            options={ATTR_VOICE: "bm_daniel"},
            message_gen=_message("Hello world. ", "How are ", "you? Fine."),
        )
    )

    assert response.extension == "mp3"
    assert b"".join([c async for c in response.data_gen]) == b"<1><2>"
    assert texts == ["Hello world.", "How are you? Fine."]


async def test_stream_wav_is_one_stream(
    hass: HomeAssistant, setup_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    hass.config_entries.async_update_entry(
        setup_integration, options={CONF_TTS_MODEL: WAV_TTS_MODEL}
    )
    mock_client.speech.generate = AsyncMock(
        side_effect=[pcm_to_wav(b"\x01\x02"), pcm_to_wav(b"\x03\x04")]
    )

    response = await _entity(hass).async_stream_tts_audio(
        TTSAudioRequest(
            language="ru", options={}, message_gen=_message("Привет. ", "Пока.")
        )
    )
    data = b"".join([c async for c in response.data_gen])

    assert response.extension == "wav"
    assert data.count(b"RIFF") == 1
    assert data.endswith(b"\x01\x02\x03\x04")


async def test_stream_tts_audio_empty_message(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    response = await _entity(hass).async_stream_tts_audio(
        TTSAudioRequest(language="en", options={}, message_gen=_message(" "))
    )
    with pytest.raises(HomeAssistantError, match="No TTS message"):
        _ = [chunk async for chunk in response.data_gen]


async def test_languages_and_voices_follow_model(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    entity = _entity(hass)
    assert entity.supported_languages == ["en", "ja"]
    voices = entity.async_get_supported_voices("ja")
    assert voices is not None
    assert [v.voice_id for v in voices] == ["jf_alpha"]

    hass.config_entries.async_update_entry(
        setup_integration, options={CONF_TTS_MODEL: WAV_TTS_MODEL}
    )
    assert "ru" in entity.supported_languages
    voices = entity.async_get_supported_voices("ru")
    assert voices is not None
    assert [v.voice_id for v in voices] == ["tara"]

    hass.config_entries.async_update_entry(
        setup_integration, options={CONF_TTS_MODEL: "unknown"}
    )
    assert entity.async_get_supported_voices("en") is None
    assert "ru" in entity.supported_languages


async def test_default_options(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    entity = _entity(hass)
    assert entity.default_language == "en"
    options = entity.default_options
    assert options[ATTR_VOICE]
    assert options[CONF_TTS_MODEL] == RECOMMENDED_TTS_MODEL


@pytest.mark.parametrize(
    ("model", "language", "hint"),
    [
        ("tts-xai-v1", "ru", "ru"),
        ("tts-qwen3-1-7b", "ru-RU", "Russian"),
        ("tts-gemini-3-1-flash", "ru", None),
    ],
)
async def test_language_hint_follows_model(
    hass: HomeAssistant,
    setup_integration: MockConfigEntry,
    mock_client: MagicMock,
    model: str,
    language: str,
    hint: str | None,
) -> None:
    mock_client.speech.generate = AsyncMock(return_value=b"audio")
    await _entity(hass).async_get_tts_audio("Привет", language, {CONF_TTS_MODEL: model})
    assert mock_client.speech.generate.call_args.kwargs["language"] == hint
