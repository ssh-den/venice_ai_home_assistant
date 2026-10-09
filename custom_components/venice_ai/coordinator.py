"""DataUpdateCoordinator for Venice AI."""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .client import (
    AsyncVeniceAIClient,
    AuthenticationError,
    RateLimitError,
    VeniceAIError,
)
from .const import UPDATE_INTERVAL
from .models import model_voices

_LOGGER = logging.getLogger(__name__)


class VeniceAICoordinatorData(TypedDict):
    """Models and voices fetched from Venice AI."""

    text_models: list[dict[str, Any]]
    audio_models: list[dict[str, Any]]
    image_models: list[dict[str, Any]]
    voices: list[str]


class VeniceAIDataUpdateCoordinator(DataUpdateCoordinator[VeniceAICoordinatorData]):
    """Coordinator to fetch and cache Venice AI metadata across platforms."""

    def __init__(
        self,
        hass: HomeAssistant,
        client: AsyncVeniceAIClient,
    ) -> None:
        """Initialize the coordinator."""
        self.client = client
        super().__init__(
            hass,
            _LOGGER,
            name="Venice AI",
            update_interval=UPDATE_INTERVAL,
        )

    async def _async_update_data(self) -> VeniceAICoordinatorData:
        """Fetch models and voices from Venice AI.

        Each category is fetched independently so a failure in one
        does not block the others.
        """
        data: VeniceAICoordinatorData = {
            "text_models": [],
            "audio_models": [],
            "image_models": [],
            "voices": [],
        }

        try:
            await self.client.validate_api_key()
        except AuthenticationError as err:
            raise ConfigEntryAuthFailed("Invalid API key") from err
        except VeniceAIError as err:
            raise UpdateFailed(f"Venice AI is unavailable: {err}") from err

        data["text_models"] = await self._async_fetch_models("text")
        tts_models = await self._async_fetch_models("tts")
        asr_models = await self._async_fetch_models("asr")
        data["image_models"] = await self._async_fetch_models("image")
        for model_type, models in (("tts", tts_models), ("asr", asr_models)):
            for model in models:
                model["model_type"] = model_type
            data["audio_models"].extend(models)
        for model in tts_models:
            for voice in model_voices(model):
                if voice not in data["voices"]:
                    data["voices"].append(voice)

        # Partial failures are tolerated so the other platforms keep working.
        if not data["text_models"] and not data["audio_models"] and not data["voices"]:
            raise UpdateFailed(
                "All Venice AI data fetches failed; coordinator has no data to return."
            )

        return data

    async def _async_fetch_models(self, model_type: str) -> list[dict[str, Any]]:
        """Fetch one model category, tolerating transient failures."""
        try:
            models = await self.client.models.list(model_type=model_type)
        except (AuthenticationError, RateLimitError) as err:
            raise UpdateFailed(f"Fetching {model_type} models failed: {err}") from err
        except VeniceAIError as err:
            _LOGGER.warning("Failed to fetch %s models: %s", model_type, err)
            return []
        _LOGGER.debug("Fetched %d %s models", len(models), model_type)
        return models
