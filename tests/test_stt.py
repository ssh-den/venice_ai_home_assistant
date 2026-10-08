"""Tests for the Venice AI STT platform."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

from homeassistant.components import stt
from homeassistant.components.stt.const import DATA_COMPONENT
from homeassistant.core import HomeAssistant
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.venice_ai.client import NetworkError
from custom_components.venice_ai.const import CONF_STT_MODEL
from custom_components.venice_ai.stt import VeniceAISTT, _pcm_to_wav


def _entity(hass: HomeAssistant) -> VeniceAISTT:
    entities = list(hass.data[DATA_COMPONENT].entities)
    assert len(entities) == 1
    entity = entities[0]
    assert isinstance(entity, VeniceAISTT)
    return entity


def _metadata(**overrides: object) -> stt.SpeechMetadata:
    values: dict[str, object] = {
        "language": "en",
        "format": stt.AudioFormats.WAV,
        "codec": stt.AudioCodecs.PCM,
        "bit_rate": stt.AudioBitRates.BITRATE_16,
        "sample_rate": stt.AudioSampleRates.SAMPLERATE_16000,
        "channel": stt.AudioChannels.CHANNEL_MONO,
    }
    values.update(overrides)
    return stt.SpeechMetadata(**values)  # type: ignore[arg-type]


async def _audio(*chunks: bytes) -> AsyncGenerator[bytes]:
    for chunk in chunks:
        yield chunk


def test_pcm_to_wav_header() -> None:
    wav = _pcm_to_wav(b"\x01\x02\x03\x04")
    assert wav[:4] == b"RIFF"
    assert wav[8:16] == b"WAVEfmt "
    assert len(wav) == 48
    assert wav.endswith(b"\x01\x02\x03\x04")


async def test_transcribe(
    hass: HomeAssistant, setup_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    hass.config_entries.async_update_entry(
        setup_integration, options={CONF_STT_MODEL: "custom-asr"}
    )
    mock_client.transcriptions.create = AsyncMock(return_value={"text": "hello"})

    result = await _entity(hass).async_process_audio_stream(
        _metadata(), _audio(b"\x00\x01", b"\x02\x03")
    )

    assert result.result is stt.SpeechResultState.SUCCESS
    assert result.text == "hello"
    kwargs = mock_client.transcriptions.create.call_args.kwargs
    assert kwargs["model"] == "custom-asr"
    assert kwargs["audio_data"] == _pcm_to_wav(b"\x00\x01\x02\x03")


async def test_unsupported_format(
    hass: HomeAssistant, setup_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    mock_client.transcriptions.create = AsyncMock()
    result = await _entity(hass).async_process_audio_stream(
        _metadata(format=stt.AudioFormats.OGG), _audio(b"\x00")
    )
    assert result.result is stt.SpeechResultState.ERROR
    mock_client.transcriptions.create.assert_not_awaited()


async def test_empty_stream(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    result = await _entity(hass).async_process_audio_stream(_metadata(), _audio())
    assert result.result is stt.SpeechResultState.ERROR


async def test_buffer_limit(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    with patch("custom_components.venice_ai.stt.MAX_STT_BUFFER_SIZE", 3):
        result = await _entity(hass).async_process_audio_stream(
            _metadata(), _audio(b"\x00\x01", b"\x02\x03")
        )
    assert result.result is stt.SpeechResultState.ERROR


@pytest.mark.parametrize("error", [NetworkError("down"), ValueError("boom")])
async def test_transcription_error(
    hass: HomeAssistant,
    setup_integration: MockConfigEntry,
    mock_client: MagicMock,
    error: Exception,
) -> None:
    mock_client.transcriptions.create = AsyncMock(side_effect=error)
    result = await _entity(hass).async_process_audio_stream(
        _metadata(), _audio(b"\x00\x01")
    )
    assert result.result is stt.SpeechResultState.ERROR
