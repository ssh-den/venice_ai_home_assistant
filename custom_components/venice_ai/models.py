"""Capabilities of Venice AI models as reported by the /models endpoint."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .languages import tts_languages, tts_voices

if TYPE_CHECKING:
    from . import VeniceAIConfigEntry


def _strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(v for v in value if isinstance(v, str) and v)


def _price(pricing: Mapping[str, Any], key: str) -> float | None:
    value = pricing.get(key)
    if isinstance(value, Mapping):
        value = value.get("usd")
    return float(value) if isinstance(value, int | float) else None


def _spec(data: Mapping[str, Any]) -> Mapping[str, Any]:
    spec = data.get("model_spec")
    return spec if isinstance(spec, Mapping) else {}


def is_private(data: Mapping[str, Any]) -> bool:
    """Whether Venice keeps the prompts of a model away from third-party providers."""
    spec = _spec(data)
    privacy = spec.get("privacy")
    if isinstance(privacy, str) and privacy.lower() in ("private", "tee", "e2ee"):
        return True
    caps = spec.get("capabilities")
    return isinstance(caps, Mapping) and (
        caps.get("supportsTeeAttestation") is True or caps.get("supportsE2EE") is True
    )


def privacy_label(data: Mapping[str, Any]) -> str | None:
    """Return the privacy level Venice reports for a model, for display."""
    privacy = _spec(data).get("privacy")
    return privacy.capitalize() if isinstance(privacy, str) and privacy else None


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Parsed description of a single chat model."""

    id: str
    name: str
    privacy: str | None = None
    supports_function_calling: bool = False
    supports_response_schema: bool = False
    supports_tee: bool = False
    supports_e2ee: bool = False
    input_price: float | None = None
    output_price: float | None = None
    offline: bool = False

    @classmethod
    def from_api(cls, data: Mapping[str, Any]) -> ModelInfo:
        """Build from one entry of the /models response."""
        spec = _spec(data)
        caps = spec.get("capabilities")
        caps = caps if isinstance(caps, Mapping) else {}
        pricing = spec.get("pricing")
        pricing = pricing if isinstance(pricing, Mapping) else {}
        model_id = str(data.get("id", ""))
        privacy = spec.get("privacy")
        return cls(
            id=model_id,
            name=str(spec.get("name") or model_id),
            privacy=privacy if isinstance(privacy, str) else None,
            supports_function_calling=caps.get("supportsFunctionCalling") is True,
            supports_response_schema=caps.get("supportsResponseSchema") is True,
            supports_tee=caps.get("supportsTeeAttestation") is True,
            supports_e2ee=caps.get("supportsE2EE") is True,
            input_price=_price(pricing, "input"),
            output_price=_price(pricing, "output"),
            offline=spec.get("offline") is True,
        )

    @property
    def label(self) -> str:
        """Human readable label for model selectors."""
        tags: list[str] = []
        if self.supports_tee or self.supports_e2ee:
            tags.append("TEE")
        elif self.privacy:
            tags.append(self.privacy.capitalize())
        if self.supports_function_calling:
            tags.append("tools")
        if self.input_price is not None and self.output_price is not None:
            tags.append(f"${self.input_price:g}/${self.output_price:g} per 1M")
        if self.offline:
            tags.append("offline")
        return f"{self.id} ({', '.join(tags)})" if tags else self.id


@dataclass(frozen=True, slots=True)
class TTSModel:
    """Voices and audio formats of a text-to-speech model."""

    id: str
    voices: tuple[str, ...]
    formats: tuple[str, ...] = ()
    default_format: str | None = None
    privacy: str | None = None

    @classmethod
    def from_api(cls, data: Mapping[str, Any]) -> TTSModel:
        """Build from one entry of the /models?type=tts response."""
        spec = _spec(data)
        default_format = spec.get("default_format")
        return cls(
            id=str(data.get("id", "")),
            voices=_strings(spec.get("voices")) or _strings(data.get("voice_models")),
            formats=_strings(spec.get("supported_formats")),
            default_format=default_format if isinstance(default_format, str) else None,
            privacy=privacy_label(data),
        )

    @property
    def languages(self) -> list[str]:
        """Languages the model speaks with its voices."""
        return tts_languages(self.id, self.voices)

    @property
    def audio_format(self) -> str:
        """Format to request: mp3 or wav, which can be joined across requests."""
        for audio_format in ("mp3", "wav"):
            if not self.formats or audio_format in self.formats:
                return audio_format
        return self.default_format or self.formats[0]

    def voices_for(self, language: str) -> list[str]:
        """Return the voices suitable for a language."""
        return tts_voices(self.id, self.voices, language)


def parse_models(models: Iterable[Any]) -> dict[str, ModelInfo]:
    """Index chat model entries by model ID."""
    return {
        info.id: info
        for info in (ModelInfo.from_api(m) for m in models if isinstance(m, Mapping))
        if info.id
    }


def parse_tts_models(models: Iterable[Any]) -> dict[str, TTSModel]:
    """Index TTS model entries that offer voices by model ID."""
    return {
        info.id: info
        for info in (TTSModel.from_api(m) for m in models if isinstance(m, Mapping))
        if info.id and info.voices
    }


def find_model(models: Iterable[Any] | None, model_id: str) -> ModelInfo | None:
    """Return the capabilities of a model, or None when it is unknown."""
    return parse_models(models or []).get(model_id)


def get_chat_model_info(entry: VeniceAIConfigEntry, model_id: str) -> ModelInfo | None:
    """Return the capabilities of a chat model known to the coordinator."""
    data = entry.runtime_data.coordinator.data
    return find_model(data["text_models"] if data else None, model_id)


def get_tts_model(entry: VeniceAIConfigEntry, model_id: str) -> TTSModel | None:
    """Return a TTS model known to the coordinator."""
    data = entry.runtime_data.coordinator.data
    return parse_tts_models(data["tts_models"] if data else []).get(model_id)
