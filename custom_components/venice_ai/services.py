"""Service actions for the Venice AI integration."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
)
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.helpers import config_validation as cv, selector
import voluptuous as vol

from .client import VeniceAIError
from .const import CONF_IMAGE_MODEL, DOMAIN, RECOMMENDED_IMAGE_MODEL

if TYPE_CHECKING:
    from . import VeniceAIConfigEntry

SERVICE_GENERATE_IMAGE = "generate_image"

ATTR_CONFIG_ENTRY = "config_entry"

_CONFIG_ENTRY_SELECTOR = selector.ConfigEntrySelector({"integration": DOMAIN})

IMAGE_SIZES = (
    "auto",
    "256x256",
    "512x512",
    "1024x1024",
    "1536x1024",
    "1024x1536",
    "1792x1024",
    "1024x1792",
)
IMAGE_QUALITIES = ("auto", "low", "medium", "high", "standard", "hd")

GENERATE_IMAGE_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_CONFIG_ENTRY): _CONFIG_ENTRY_SELECTOR,
        vol.Required("prompt"): cv.string,
        vol.Optional("model"): cv.string,
        vol.Optional("size", default="1024x1024"): vol.In(IMAGE_SIZES),
        vol.Optional("quality", default="standard"): vol.In(IMAGE_QUALITIES),
        vol.Optional("style", default="vivid"): vol.In(("vivid", "natural")),
    }
)


def _async_get_loaded_entry(hass: HomeAssistant, entry_id: str) -> VeniceAIConfigEntry:
    """Return the loaded Venice AI config entry or raise a validation error."""
    entry = hass.config_entries.async_get_entry(entry_id)
    if entry is None or entry.domain != DOMAIN:
        raise ServiceValidationError(
            translation_domain=DOMAIN,
            translation_key="invalid_config_entry",
            translation_placeholders={"config_entry": entry_id},
        )
    if entry.state is not ConfigEntryState.LOADED:
        raise ServiceValidationError(
            translation_domain=DOMAIN,
            translation_key="config_entry_not_loaded",
            translation_placeholders={"config_entry": entry_id},
        )
    return cast("VeniceAIConfigEntry", entry)


async def _async_generate_image(call: ServiceCall) -> ServiceResponse:
    """Generate an image with Venice AI."""
    entry = _async_get_loaded_entry(call.hass, call.data[ATTR_CONFIG_ENTRY])
    client = entry.runtime_data.client
    model: str = call.data.get("model") or entry.options.get(
        CONF_IMAGE_MODEL, RECOMMENDED_IMAGE_MODEL
    )
    available = {
        m.get("id")
        for m in (entry.runtime_data.coordinator.data or {}).get("image_models", [])
    }
    if model != RECOMMENDED_IMAGE_MODEL and available and model not in available:
        raise ServiceValidationError(
            translation_domain=DOMAIN,
            translation_key="invalid_image_model",
            translation_placeholders={
                "model": model,
                "models": ", ".join(sorted(m for m in available if m)),
            },
        )

    try:
        response = await client.images.generate(
            model=model,
            prompt=call.data["prompt"],
            size=call.data["size"],
            quality=call.data["quality"],
            style=call.data["style"],
            response_format="url",
            n=1,
        )
    except VeniceAIError as err:
        raise HomeAssistantError(f"Error generating image: {err}") from err

    data = response.get("data") if isinstance(response, dict) else None
    if not data or not isinstance(data, list) or not isinstance(data[0], dict):
        raise HomeAssistantError("No image data returned from Venice AI")
    result = dict(data[0])
    result.pop("b64_json", None)
    return result


def async_setup_services(hass: HomeAssistant) -> None:
    """Register the Venice AI service actions."""
    hass.services.async_register(
        DOMAIN,
        SERVICE_GENERATE_IMAGE,
        _async_generate_image,
        schema=GENERATE_IMAGE_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
