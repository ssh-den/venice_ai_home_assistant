"""Speech-to-Text provider for Venice AI."""

from __future__ import annotations

from collections.abc import AsyncIterable
import logging

from homeassistant.components import stt
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import VeniceAIConfigEntry
from .audio import pcm_to_wav
from .client import VeniceAIError
from .const import (
    CONF_STT_MODEL,
    MAX_STT_BUFFER_SIZE,
    RECOMMENDED_STT_MODEL,
    SUBENTRY_STT,
)
from .entity import VeniceAIEntity, subentries_of
from .languages import primary_language, stt_languages

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: VeniceAIConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up a speech-to-text entity for each STT subentry."""
    for subentry in subentries_of(entry, SUBENTRY_STT):
        async_add_entities(
            [VeniceAISTT(entry, subentry)], config_subentry_id=subentry.subentry_id
        )


class VeniceAISTT(stt.SpeechToTextEntity, VeniceAIEntity):
    """The Venice AI Speech-to-Text provider."""

    def __init__(self, entry: VeniceAIConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize Venice AI STT."""
        super().__init__(
            entry, subentry, subentry.data.get(CONF_STT_MODEL, RECOMMENDED_STT_MODEL)
        )

    @property
    def _model(self) -> str:
        return str(self.options.get(CONF_STT_MODEL, RECOMMENDED_STT_MODEL))

    @property
    def supported_languages(self) -> list[str]:
        """Return the languages of the configured model."""
        return stt_languages(self._model)

    @property
    def supported_formats(self) -> list[stt.AudioFormats]:
        """Return list of supported audio formats."""
        return [stt.AudioFormats.WAV]

    @property
    def supported_codecs(self) -> list[stt.AudioCodecs]:
        """Return list of supported audio codecs."""
        return [stt.AudioCodecs.PCM]

    @property
    def supported_bit_rates(self) -> list[stt.AudioBitRates]:
        """Return list of supported bit rates in bps."""
        return [stt.AudioBitRates.BITRATE_16]

    @property
    def supported_sample_rates(self) -> list[stt.AudioSampleRates]:
        """Return list of supported sample rates in Hz."""
        return [stt.AudioSampleRates.SAMPLERATE_16000]

    @property
    def supported_channels(self) -> list[stt.AudioChannels]:
        """Return list of supported channel counts."""
        return [stt.AudioChannels.CHANNEL_MONO]

    async def async_process_audio_stream(
        self,
        metadata: stt.SpeechMetadata,
        stream: AsyncIterable[bytes],
    ) -> stt.SpeechResult:
        """Buffer the audio stream and transcribe it in one request.

        Venice AI does not accept chunked uploads for transcriptions.
        """
        if not self.check_metadata(metadata):
            _LOGGER.error("Unsupported audio metadata: %s", metadata)
            return stt.SpeechResult(None, stt.SpeechResultState.ERROR)

        audio = bytearray()
        async for chunk in stream:
            audio.extend(chunk)
            if len(audio) > MAX_STT_BUFFER_SIZE:
                _LOGGER.error(
                    "Audio exceeds %d bytes; aborting transcription",
                    MAX_STT_BUFFER_SIZE,
                )
                return stt.SpeechResult(None, stt.SpeechResultState.ERROR)
        if not audio:
            _LOGGER.warning("Received empty audio stream for transcription")
            return stt.SpeechResult(None, stt.SpeechResultState.ERROR)

        try:
            text = await self.entry.runtime_data.client.transcriptions.create(
                audio_data=pcm_to_wav(bytes(audio)),
                model=self._model,
                language=primary_language(metadata.language),
            )
        except VeniceAIError as err:
            _LOGGER.error("Venice AI transcription error: %s", err)
            return stt.SpeechResult(None, stt.SpeechResultState.ERROR)

        _LOGGER.debug(
            "Transcribed %d bytes of audio into %d characters", len(audio), len(text)
        )
        return stt.SpeechResult(text, stt.SpeechResultState.SUCCESS)
