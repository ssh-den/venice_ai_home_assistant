"""Capabilities of Venice AI models as reported by the /models endpoint."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from . import VeniceAIConfigEntry


def _price(pricing: Mapping[str, Any], key: str) -> float | None:
    value = pricing.get(key)
    if isinstance(value, Mapping):
        value = value.get("usd")
    return float(value) if isinstance(value, int | float) else None


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Parsed description of a single model."""

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
        spec = data.get("model_spec")
        spec = spec if isinstance(spec, Mapping) else {}
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
        if self.supports_e2ee:
            tags.append("E2EE")
        elif self.supports_tee:
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


def parse_models(models: Iterable[Any]) -> dict[str, ModelInfo]:
    """Index /models entries by model ID."""
    return {
        info.id: info
        for info in (ModelInfo.from_api(m) for m in models if isinstance(m, Mapping))
        if info.id
    }


def find_model(models: Iterable[Any] | None, model_id: str) -> ModelInfo | None:
    """Return the capabilities of a model, or None when it is unknown."""
    return parse_models(models or []).get(model_id)


def model_voices(model: Mapping[str, Any]) -> list[str]:
    """Return the voices of a TTS model from /models metadata."""
    spec = model.get("model_spec")
    voices = spec.get("voices") if isinstance(spec, Mapping) else None
    if not isinstance(voices, list) or not voices:
        voices = model.get("voice_models")
    if not isinstance(voices, list):
        return []
    return [v for v in voices if isinstance(v, str) and v]


def get_chat_model_info(entry: VeniceAIConfigEntry, model_id: str) -> ModelInfo | None:
    """Return the capabilities of a chat model known to the coordinator."""
    data = entry.runtime_data.coordinator.data
    return find_model(data["text_models"] if data else None, model_id)
