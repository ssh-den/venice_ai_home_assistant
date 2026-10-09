"""Venice AI TTS platform."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterable
from dataclasses import dataclass
import logging
from typing import Any

from homeassistant.components.tts import (
    ATTR_VOICE,
    TextToSpeechEntity,
    TTSAudioResponse,
    TtsAudioType,
    Voice,
)
from homeassistant.components.tts.entity import TTSAudioRequest
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import VeniceAIConfigEntry
from .audio import WavStreamJoiner, async_stream_segments, join_wav, split_text
from .const import (
    CONF_TTS_MODEL,
    CONF_TTS_SPEED,
    CONF_TTS_VOICE,
    RECOMMENDED_TTS_MODEL,
    RECOMMENDED_TTS_SPEED,
    RECOMMENDED_TTS_VOICE,
)
from .entity import device_info
from .languages import MULTILINGUAL, tts_language_hint
from .models import TTSModel, get_tts_model

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: VeniceAIConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up Venice AI TTS platform."""
    async_add_entities([VeniceAITTS(entry)])


@dataclass(frozen=True, slots=True)
class _Request:
    """Settings for one synthesis."""

    model: str
    voice: str
    speed: float
    audio_format: str
    language: str | None


class VeniceAITTS(TextToSpeechEntity):
    """Venice AI TTS entity."""

    _attr_has_entity_name = True
    _attr_name = "TTS"

    def __init__(self, entry: VeniceAIConfigEntry) -> None:
        """Initialize TTS entity."""
        self.entry = entry
        self._client = entry.runtime_data.client
        self._attr_unique_id = f"{entry.entry_id}_tts"
        self._attr_device_info = device_info(entry)
        self._attr_supported_options = [ATTR_VOICE, CONF_TTS_MODEL, CONF_TTS_SPEED]

    def _model(self, model_id: str | None = None) -> TTSModel | None:
        return get_tts_model(
            self.entry,
            model_id or self.entry.options.get(CONF_TTS_MODEL, RECOMMENDED_TTS_MODEL),
        )

    @property
    def supported_languages(self) -> list[str]:
        """Return the languages of the configured model."""
        model = self._model()
        return model.languages if model else list(MULTILINGUAL)

    @property
    def default_language(self) -> str:
        """Return the default language."""
        languages = self.supported_languages
        return "en" if "en" in languages else languages[0]

    @property
    def default_options(self) -> dict[str, Any]:
        """Return default options mapped from config entry."""
        options = self.entry.options
        return {
            ATTR_VOICE: options.get(CONF_TTS_VOICE, RECOMMENDED_TTS_VOICE),
            CONF_TTS_MODEL: options.get(CONF_TTS_MODEL, RECOMMENDED_TTS_MODEL),
            CONF_TTS_SPEED: options.get(CONF_TTS_SPEED, RECOMMENDED_TTS_SPEED),
        }

    def async_get_supported_voices(self, language: str) -> list[Voice] | None:
        """Return the voices of the configured model for a language."""
        model = self._model()
        if model is None:
            return None
        return [Voice(voice, voice) for voice in model.voices_for(language)]

    def _request(self, language: str, options: dict[str, Any] | None) -> _Request:
        settings = {**self.default_options, **(options or {})}
        model_id = settings[CONF_TTS_MODEL]
        model = self._model(model_id)
        return _Request(
            model=model_id,
            voice=settings[ATTR_VOICE],
            speed=float(settings[CONF_TTS_SPEED]),
            audio_format=model.audio_format if model else "mp3",
            language=tts_language_hint(model_id, language),
        )

    async def _synthesize(self, request: _Request, text: str) -> bytes:
        return await self._client.speech.generate(
            text=text,
            voice=request.voice,
            model=request.model,
            audio_output=request.audio_format,
            speed=request.speed,
            language=request.language,
        )

    async def async_get_tts_audio(
        self, message: str, language: str, options: dict[str, Any] | None = None
    ) -> TtsAudioType:
        """Generate TTS audio, one request per segment of the message."""
        request = self._request(language, options)
        audio = [await self._synthesize(request, text) for text in split_text(message)]
        if not any(audio):
            raise HomeAssistantError("TTS generation returned empty audio")
        if request.audio_format == "wav":
            return "wav", join_wav(audio)
        return request.audio_format, b"".join(audio)

    async def async_stream_tts_audio(
        self, request: TTSAudioRequest
    ) -> TTSAudioResponse:
        """Synthesize streamed text sentence by sentence."""
        settings = self._request(request.language, request.options)
        return TTSAudioResponse(
            settings.audio_format, self._stream(settings, request.message_gen)
        )

    async def _stream(
        self, request: _Request, message: AsyncIterable[str]
    ) -> AsyncGenerator[bytes]:
        joiner = WavStreamJoiner() if request.audio_format == "wav" else None
        spoken = False
        async for text in async_stream_segments(message):
            spoken = True
            _LOGGER.debug("Synthesizing %d characters", len(text))
            if joiner is not None:
                yield joiner.add(await self._synthesize(request, text))
                continue
            async for chunk in self._client.speech.generate_streaming(
                text=text,
                voice=request.voice,
                model=request.model,
                audio_output=request.audio_format,
                speed=request.speed,
                language=request.language,
            ):
                yield chunk
        if not spoken:
            raise HomeAssistantError(f"No TTS message for {self.entity_id}")
