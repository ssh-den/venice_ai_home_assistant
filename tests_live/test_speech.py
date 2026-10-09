"""Text-to-speech and speech-to-text with short phrases on cheap models."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
import shutil

from homeassistant.components import stt
from homeassistant.components.stt.const import DATA_COMPONENT as STT_COMPONENT
from homeassistant.components.tts import ATTR_VOICE
from homeassistant.components.tts.const import DATA_COMPONENT as TTS_COMPONENT
from homeassistant.components.tts.entity import TTSAudioRequest
from homeassistant.core import HomeAssistant
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.venice_ai.const import CONF_TTS_MODEL
from custom_components.venice_ai.models import parse_tts_models
from custom_components.venice_ai.stt import VeniceAISTT
from custom_components.venice_ai.tts import VeniceAITTS

RUSSIAN_TTS_MODEL = "tts-qwen3-0-6b"


def _tts(hass: HomeAssistant) -> VeniceAITTS:
    entity = hass.data[TTS_COMPONENT].get_entity("tts.venice_ai_tts")
    assert isinstance(entity, VeniceAITTS)
    return entity


def _stt(hass: HomeAssistant) -> VeniceAISTT:
    entity = next(iter(hass.data[STT_COMPONENT].entities))
    assert isinstance(entity, VeniceAISTT)
    return entity


def _is_mp3(data: bytes) -> bool:
    return data[:3] == b"ID3" or (data[0] == 0xFF and data[1] & 0xE0 == 0xE0)


async def _to_pcm(audio: bytes) -> bytes:
    """Decode audio into the 16 kHz mono PCM that voice pipelines record."""
    if not (ffmpeg := shutil.which("ffmpeg")):
        pytest.skip("ffmpeg is not installed")
    process = await asyncio.create_subprocess_exec(
        *(ffmpeg, "-loglevel", "error", "-i", "pipe:0"),
        *("-f", "s16le", "-ac", "1", "-ar", "16000", "pipe:1"),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
    )
    pcm, _ = await process.communicate(audio)
    assert process.returncode == 0
    return pcm


async def _chunks(*parts: bytes | str) -> AsyncGenerator:
    for part in parts:
        yield part


async def _round_trip(
    hass: HomeAssistant, text: str, language: str, options: dict | None = None
) -> str:
    extension, audio = await _tts(hass).async_get_tts_audio(
        text, language, options or {}
    )
    assert extension == "mp3"
    assert audio and _is_mp3(audio)
    metadata = stt.SpeechMetadata(
        language=language,
        format=stt.AudioFormats.WAV,
        codec=stt.AudioCodecs.PCM,
        bit_rate=stt.AudioBitRates.BITRATE_16,
        sample_rate=stt.AudioSampleRates.SAMPLERATE_16000,
        channel=stt.AudioChannels.CHANNEL_MONO,
    )
    result = await _stt(hass).async_process_audio_stream(
        metadata, _chunks(await _to_pcm(audio))
    )
    assert result.result is stt.SpeechResultState.SUCCESS
    return str(result.text).lower()


async def test_tts_streams_by_sentence(
    hass: HomeAssistant, live_entry: MockConfigEntry
) -> None:
    response = await _tts(hass).async_stream_tts_audio(
        TTSAudioRequest(language="en", options={}, message_gen=_chunks("Hi. ", "Bye."))
    )
    streamed = b"".join([chunk async for chunk in response.data_gen])
    assert response.extension == "mp3"
    assert _is_mp3(streamed)


async def test_english_round_trip(
    hass: HomeAssistant, live_entry: MockConfigEntry
) -> None:
    text = await _round_trip(hass, "Turn on the kitchen light.", "en")
    assert "kitchen" in text, text


async def test_russian_round_trip(
    hass: HomeAssistant, live_entry: MockConfigEntry
) -> None:
    tts_models = parse_tts_models(
        live_entry.runtime_data.coordinator.data["tts_models"]
    )
    model = tts_models[RUSSIAN_TTS_MODEL]
    assert "ru" in model.languages
    text = await _round_trip(
        hass,
        "Включи свет на кухне.",
        "ru",
        {CONF_TTS_MODEL: model.id, ATTR_VOICE: model.voices[0]},
    )
    assert "свет" in text, text
