"""Service actions for the Venice AI integration."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from homeassistant.components import ai_task
from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import Platform
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
)
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.helpers import (
    config_validation as cv,
    entity_registry as er,
    selector,
)
import voluptuous as vol

from .client import VeniceAIError
from .const import CONF_IMAGE_MODEL, DOMAIN, RECOMMENDED_IMAGE_MODEL
from .venice_api import extract_json

if TYPE_CHECKING:
    from . import VeniceAIConfigEntry

SERVICE_GENERATE_IMAGE = "generate_image"
SERVICE_AI_TASK = "ai_task"

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

AI_TASK_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_CONFIG_ENTRY): _CONFIG_ENTRY_SELECTOR,
        vol.Required("task"): cv.string,
        vol.Optional("structure"): cv.string,
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


async def _async_run_ai_task(call: ServiceCall) -> ServiceResponse:
    """Run a data generation task through the entry's AI Task entity."""
    hass = call.hass
    entry = _async_get_loaded_entry(hass, call.data[ATTR_CONFIG_ENTRY])

    entity_id = er.async_get(hass).async_get_entity_id(
        Platform.AI_TASK, DOMAIN, f"{entry.entry_id}_task"
    )
    if entity_id is None:
        raise ServiceValidationError(
            translation_domain=DOMAIN,
            translation_key="entity_not_found",
            translation_placeholders={"entry_id": entry.entry_id},
        )

    instructions: str = call.data["task"]
    structure: str | None = call.data.get("structure")
    if structure:
        instructions = (
            f"{instructions}\n\nRespond only with valid JSON matching this "
            f"structure, without any surrounding text:\n{structure}"
        )

    result = await ai_task.async_generate_data(
        hass,
        task_name=f"{DOMAIN}.{SERVICE_AI_TASK}",
        entity_id=entity_id,
        instructions=instructions,
    )

    data = result.data
    if structure and isinstance(data, str):
        try:
            data = extract_json(data)
        except ValueError as err:
            raise HomeAssistantError(
                f"Venice AI returned invalid JSON for the requested structure: {err}"
            ) from err

    return {"conversation_id": result.conversation_id, "data": data}


def async_setup_services(hass: HomeAssistant) -> None:
    """Register the Venice AI service actions."""
    hass.services.async_register(
        DOMAIN,
        SERVICE_GENERATE_IMAGE,
        _async_generate_image,
        schema=GENERATE_IMAGE_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_AI_TASK,
        _async_run_ai_task,
        schema=AI_TASK_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
