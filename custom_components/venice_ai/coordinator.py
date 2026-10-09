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
from .languages import unlisted_models

_LOGGER = logging.getLogger(__name__)


class VeniceAICoordinatorData(TypedDict):
    """Models offered by Venice AI, by type."""

    text_models: list[dict[str, Any]]
    tts_models: list[dict[str, Any]]
    asr_models: list[dict[str, Any]]
    image_models: list[dict[str, Any]]


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
        """Fetch the model lists from Venice AI."""
        try:
            await self.client.validate_api_key()
        except AuthenticationError as err:
            raise ConfigEntryAuthFailed("Invalid API key") from err
        except VeniceAIError as err:
            raise UpdateFailed(f"Venice AI is unavailable: {err}") from err

        data: VeniceAICoordinatorData = {
            "text_models": await self._async_fetch_models("text"),
            "tts_models": await self._async_fetch_models("tts"),
            "asr_models": await self._async_fetch_models("asr"),
            "image_models": await self._async_fetch_models("image"),
        }
        # Partial failures are tolerated so the other platforms keep working.
        if not any(data.values()):
            raise UpdateFailed(
                "All Venice AI data fetches failed; coordinator has no data to return."
            )
        if unlisted := unlisted_models(
            (m.get("id", "") for m in data["tts_models"]),
            (m.get("id", "") for m in data["asr_models"]),
        ):
            _LOGGER.info(
                "Languages of these Venice speech models are unknown, so they are "
                "offered for every language: %s",
                ", ".join(unlisted),
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
