"""The Venice AI Conversation integration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import logging
from typing import Any, cast

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import config_validation as cv, issue_registry as ir
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.issue_registry import IssueSeverity
from homeassistant.helpers.typing import ConfigType

from .client import AsyncVeniceAIClient, AuthenticationError, RateLimitError
from .const import (
    CONF_CHAT_MODEL,
    CONF_IMAGE_MODEL,
    CONF_PRIVATE_MODELS_ONLY,
    CONF_REQUEST_TIMEOUT,
    CONF_STT_MODEL,
    CONF_TTS_MODEL,
    DOMAIN,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_IMAGE_MODEL,
    RECOMMENDED_PRIVATE_MODELS_ONLY,
    RECOMMENDED_REQUEST_TIMEOUT,
    RECOMMENDED_STT_MODEL,
    RECOMMENDED_TTS_MODEL,
)
from .coordinator import VeniceAIDataUpdateCoordinator
from .models import is_private
from .services import async_setup_services

_LOGGER = logging.getLogger(__name__)

PLATFORMS = [
    Platform.AI_TASK,
    Platform.CONVERSATION,
    Platform.SENSOR,
    Platform.STT,
    Platform.TTS,
]

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

_ISSUE_UNAVAIL = "unavailable_model_{entry_id}_{model_key}"
_ISSUE_NOT_PRIVATE = "not_private_model_{entry_id}_{model_key}"
_ISSUE_AUTH = "auth_failure_{entry_id}"
_ISSUE_API_DOWN = "api_unavailable_{entry_id}"
_ISSUE_RATE_LIMIT = "rate_limited_{entry_id}"


# Option, default and coordinator model list of every configurable model
_CONFIGURED_MODELS = (
    (CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL, "text_models"),
    (CONF_TTS_MODEL, RECOMMENDED_TTS_MODEL, "tts_models"),
    (CONF_STT_MODEL, RECOMMENDED_STT_MODEL, "asr_models"),
    (CONF_IMAGE_MODEL, RECOMMENDED_IMAGE_MODEL, "image_models"),
)


@dataclass
class VeniceAIRuntimeData:
    """Runtime data stored in the config entry."""

    client: AsyncVeniceAIClient
    coordinator: VeniceAIDataUpdateCoordinator


type VeniceAIConfigEntry = ConfigEntry[VeniceAIRuntimeData]


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up Venice AI Conversation."""
    async_setup_services(hass)
    return True


@callback
def _async_on_coordinator_update(
    hass: HomeAssistant, entry: ConfigEntry, coordinator: Any
) -> None:
    """Create or clear auth/API-availability repair issues based on coordinator state."""
    entry_id = entry.entry_id
    ir.async_delete_issue(hass, DOMAIN, _ISSUE_AUTH.format(entry_id=entry_id))
    ir.async_delete_issue(hass, DOMAIN, _ISSUE_API_DOWN.format(entry_id=entry_id))
    ir.async_delete_issue(hass, DOMAIN, _ISSUE_RATE_LIMIT.format(entry_id=entry_id))

    last_exc = coordinator.last_exception
    if last_exc is None:
        _LOGGER.debug(
            "Coordinator update succeeded for entry %s; connectivity issues cleared",
            entry_id,
        )
        return

    cause = getattr(last_exc, "__cause__", None) or last_exc

    if isinstance(cause, AuthenticationError):
        ir.async_create_issue(
            hass,
            DOMAIN,
            _ISSUE_AUTH.format(entry_id=entry_id),
            is_fixable=False,
            is_persistent=True,
            severity=IssueSeverity.ERROR,
            translation_key="auth_failure",
            translation_placeholders={"entry_title": entry.title},
        )
        _LOGGER.warning(
            "Coordinator auth failure for entry %s — repair issue created", entry_id
        )
        entry.async_start_reauth(hass)
    elif isinstance(cause, RateLimitError):
        ir.async_create_issue(
            hass,
            DOMAIN,
            _ISSUE_RATE_LIMIT.format(entry_id=entry_id),
            is_fixable=False,
            is_persistent=False,
            severity=IssueSeverity.WARNING,
            translation_key="rate_limited",
            translation_placeholders={"entry_title": entry.title},
        )
        _LOGGER.warning(
            "Coordinator rate limit for entry %s — repair issue created", entry_id
        )
    else:
        ir.async_create_issue(
            hass,
            DOMAIN,
            _ISSUE_API_DOWN.format(entry_id=entry_id),
            is_fixable=False,
            is_persistent=False,
            severity=IssueSeverity.WARNING,
            translation_key="api_unavailable",
            translation_placeholders={
                "entry_title": entry.title,
                "error": str(cause),
            },
        )
        _LOGGER.warning(
            "Coordinator API error for entry %s — repair issue created: %s",
            entry_id,
            cause,
        )


@callback
def _async_check_models(hass: HomeAssistant, entry: VeniceAIConfigEntry) -> None:
    """Report configured models that Venice no longer offers or runs without privacy."""
    options = entry.options
    data = cast(Mapping[str, list[Any]], entry.runtime_data.coordinator.data or {})
    private_only = options.get(
        CONF_PRIVATE_MODELS_ONLY, RECOMMENDED_PRIVATE_MODELS_ONLY
    )

    for model_key, default, model_type in _CONFIGURED_MODELS:
        model_id = options.get(model_key, default)
        models = {
            m.get("id"): m for m in data.get(model_type, []) if isinstance(m, dict)
        }
        checked = bool(models) and model_id != RECOMMENDED_IMAGE_MODEL
        model = models.get(model_id)
        placeholders = {
            "model": model_id,
            "model_type": model_key.replace("_model", "").upper(),
        }
        for issue, severity, found in (
            (_ISSUE_UNAVAIL, IssueSeverity.ERROR, checked and model is None),
            (
                _ISSUE_NOT_PRIVATE,
                IssueSeverity.WARNING,
                private_only and model is not None and not is_private(model),
            ),
        ):
            issue_id = issue.format(entry_id=entry.entry_id, model_key=model_key)
            if not found:
                ir.async_delete_issue(hass, DOMAIN, issue_id)
                continue
            ir.async_create_issue(
                hass,
                DOMAIN,
                issue_id,
                is_fixable=False,
                is_persistent=False,
                severity=severity,
                translation_key=issue.split("_{", 1)[0],
                translation_placeholders=placeholders,
            )
            _LOGGER.warning("Model %s (%s): %s", model_id, model_key, issue_id)


async def async_setup_repairs(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Set up repair issues for a config entry."""
    _async_check_models(hass, entry)

    coordinator = entry.runtime_data.coordinator

    @callback
    def _on_coordinator_update() -> None:
        _async_on_coordinator_update(hass, entry, coordinator)
        _async_check_models(hass, entry)

    entry.async_on_unload(coordinator.async_add_listener(_on_coordinator_update))


async def async_unload_repairs(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Unload repair issues for a config entry."""
    entry_id = entry.entry_id
    registry = ir.async_get(hass)
    issues = [
        *(
            issue.format(entry_id=entry_id, model_key=model_key)
            for issue in (_ISSUE_UNAVAIL, _ISSUE_NOT_PRIVATE)
            for model_key, _, _ in _CONFIGURED_MODELS
        ),
        _ISSUE_AUTH.format(entry_id=entry_id),
        _ISSUE_API_DOWN.format(entry_id=entry_id),
        _ISSUE_RATE_LIMIT.format(entry_id=entry_id),
    ]
    for issue_id in issues:
        if registry.async_get_issue(DOMAIN, issue_id):
            ir.async_delete_issue(hass, DOMAIN, issue_id)
            _LOGGER.debug("Deleted repair issue %s", issue_id)


async def async_setup_entry(hass: HomeAssistant, entry: VeniceAIConfigEntry) -> bool:
    """Set up Venice AI Conversation from a config entry."""
    client = AsyncVeniceAIClient(
        api_key=entry.data[CONF_API_KEY],
        http_client=get_async_client(hass),
        timeout=float(
            entry.options.get(CONF_REQUEST_TIMEOUT, RECOMMENDED_REQUEST_TIMEOUT)
        ),
    )

    coordinator = VeniceAIDataUpdateCoordinator(hass, client)
    await coordinator.async_config_entry_first_refresh()
    entry.runtime_data = VeniceAIRuntimeData(
        client=client,
        coordinator=coordinator,
    )

    _LOGGER.info("Forwarding entry setups to platforms: %s", PLATFORMS)
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    _LOGGER.info("Successfully forwarded entry setups")

    await async_setup_repairs(hass, entry)
    return True


async def async_migrate_entry(hass: HomeAssistant, entry: VeniceAIConfigEntry) -> bool:
    """Migrate a config entry to the current version."""
    if entry.version > 1:
        # Downgrading from a future version is not supported.
        _LOGGER.error(
            "Cannot downgrade Venice AI entry %s from version %s.%s",
            entry.entry_id,
            entry.version,
            entry.minor_version,
        )
        return False

    # Add version-specific steps here, each ending with
    # hass.config_entries.async_update_entry(entry, version=..., minor_version=...)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: VeniceAIConfigEntry) -> bool:
    """Unload Venice AI."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    await async_unload_repairs(hass, entry)

    client: AsyncVeniceAIClient = entry.runtime_data.client
    await client.close()

    return unload_ok
