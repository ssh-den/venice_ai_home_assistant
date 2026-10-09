"""Tests for the Venice AI data update coordinator."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import UpdateFailed
import pytest

from custom_components.venice_ai.client import (
    NetworkError,
    RateLimitError,
    VeniceAIError,
)
from custom_components.venice_ai.coordinator import VeniceAIDataUpdateCoordinator

from .conftest import MODELS_BY_TYPE


def _failing_list(
    failures: dict[str, VeniceAIError],
) -> Any:
    async def _list(model_type: str = "text") -> list[dict[str, Any]]:
        if model_type in failures:
            raise failures[model_type]
        return [dict(m) for m in MODELS_BY_TYPE.get(model_type, [])]

    return _list


async def test_collects_models_and_voices(
    hass: HomeAssistant, mock_client: MagicMock
) -> None:
    data = await VeniceAIDataUpdateCoordinator(hass, mock_client)._async_update_data()
    assert len(data["text_models"]) == len(MODELS_BY_TYPE["text"])
    assert [m["model_type"] for m in data["audio_models"]] == ["tts", "asr"]
    assert data["voices"] == ["bm_daniel", "af_heart"]
    assert len(data["image_models"]) == len(MODELS_BY_TYPE["image"])


async def test_partial_failure_is_tolerated(
    hass: HomeAssistant, mock_client: MagicMock
) -> None:
    mock_client.models.list.side_effect = _failing_list(
        {"asr": NetworkError("down"), "image": VeniceAIError("boom")}
    )
    data = await VeniceAIDataUpdateCoordinator(hass, mock_client)._async_update_data()
    assert data["text_models"]
    assert [m["model_type"] for m in data["audio_models"]] == ["tts"]
    assert data["image_models"] == []


async def test_rate_limit_fails_update(
    hass: HomeAssistant, mock_client: MagicMock
) -> None:
    mock_client.models.list.side_effect = _failing_list(
        {"text": RateLimitError("slow down")}
    )
    with pytest.raises(UpdateFailed, match="slow down"):
        await VeniceAIDataUpdateCoordinator(hass, mock_client)._async_update_data()


async def test_no_data_fails_update(
    hass: HomeAssistant, mock_client: MagicMock
) -> None:
    mock_client.models.list.side_effect = _failing_list(
        {t: NetworkError("down") for t in ("text", "tts", "asr", "image")}
    )
    with pytest.raises(UpdateFailed, match="All Venice AI data fetches failed"):
        await VeniceAIDataUpdateCoordinator(hass, mock_client)._async_update_data()
