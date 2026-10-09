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


async def test_collects_models_by_type(
    hass: HomeAssistant, mock_client: MagicMock
) -> None:
    data = await VeniceAIDataUpdateCoordinator(hass, mock_client)._async_update_data()
    assert dict(data) == {
        f"{model_type}_models": models for model_type, models in MODELS_BY_TYPE.items()
    }


async def test_partial_failure_is_tolerated(
    hass: HomeAssistant, mock_client: MagicMock
) -> None:
    mock_client.models.list.side_effect = _failing_list(
        {"asr": NetworkError("down"), "image": VeniceAIError("boom")}
    )
    data = await VeniceAIDataUpdateCoordinator(hass, mock_client)._async_update_data()
    assert data["text_models"]
    assert data["tts_models"]
    assert data["asr_models"] == []
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
