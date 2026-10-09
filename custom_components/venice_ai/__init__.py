"""The Venice AI Conversation integration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import logging
from types import MappingProxyType
from typing import Any, cast

from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, Platform
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import (
    config_validation as cv,
    device_registry as dr,
    entity_registry as er,
    issue_registry as ir,
)
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.issue_registry import IssueSeverity
from homeassistant.helpers.typing import ConfigType

from .client import (
    AsyncVeniceAIClient,
    AuthenticationError,
    RateLimitError,
    VeniceAIMetrics,
)
from .const import (
    CONF_CHAT_MODEL,
    CONF_DISABLE_THINKING,
    CONF_IMAGE_MODEL,
    CONF_MAX_HISTORY_MESSAGES,
    CONF_MAX_TOKENS,
    CONF_MAX_TOOL_ITERATIONS,
    CONF_PRIVATE_MODELS_ONLY,
    CONF_PROMPT,
    CONF_RECOMMENDED,
    CONF_REQUEST_TIMEOUT,
    CONF_STREAM_RESPONSE,
    CONF_STRIP_THINKING_RESPONSE,
    CONF_STT_MODEL,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_TTS_MODEL,
    CONF_TTS_SPEED,
    CONF_TTS_VOICE,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_CONVERSATION_NAME,
    DEFAULT_STT_NAME,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TTS_NAME,
    DOMAIN,
    RECOMMENDED_AI_TASK_OPTIONS,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_IMAGE_MODEL,
    RECOMMENDED_PRIVATE_MODELS_ONLY,
    RECOMMENDED_REQUEST_TIMEOUT,
    RECOMMENDED_STT_OPTIONS,
    RECOMMENDED_TTS_OPTIONS,
    SUBENTRY_AI_TASK,
    SUBENTRY_CONVERSATION,
    SUBENTRY_MODELS,
    SUBENTRY_STT,
    SUBENTRY_TTS,
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

_ISSUE_UNAVAIL = "unavailable_model_{entry_id}_{target}"
_ISSUE_NOT_PRIVATE = "not_private_model_{entry_id}_{target}"
_ISSUE_AUTH = "auth_failure_{entry_id}"
_ISSUE_API_DOWN = "api_unavailable_{entry_id}"
_ISSUE_RATE_LIMIT = "rate_limited_{entry_id}"

# Coordinator model list of each subentry type
_MODEL_LISTS = {
    SUBENTRY_CONVERSATION: "text_models",
    SUBENTRY_AI_TASK: "text_models",
    SUBENTRY_TTS: "tts_models",
    SUBENTRY_STT: "asr_models",
}


@dataclass
class VeniceAIRuntimeData:
    """Runtime data stored in the config entry."""

    client: AsyncVeniceAIClient
    coordinator: VeniceAIDataUpdateCoordinator
    metrics: dict[str, VeniceAIMetrics]


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
    data = cast(Mapping[str, list[Any]], entry.runtime_data.coordinator.data or {})
    private_only = entry.options.get(
        CONF_PRIVATE_MODELS_ONLY, RECOMMENDED_PRIVATE_MODELS_ONLY
    )
    targets = [
        (
            subentry.subentry_id,
            subentry.title,
            subentry.data.get(key, default),
            _MODEL_LISTS[subentry.subentry_type],
        )
        for subentry in entry.subentries.values()
        for key, default in [SUBENTRY_MODELS[subentry.subentry_type]]
    ]
    image_model = entry.options.get(CONF_IMAGE_MODEL, RECOMMENDED_IMAGE_MODEL)
    if image_model != RECOMMENDED_IMAGE_MODEL:
        targets.append((CONF_IMAGE_MODEL, entry.title, image_model, "image_models"))

    for target, name, model_id, model_type in targets:
        models = {
            m.get("id"): m for m in data.get(model_type, []) if isinstance(m, dict)
        }
        model = models.get(model_id)
        for issue, severity, found in (
            (_ISSUE_UNAVAIL, IssueSeverity.ERROR, bool(models) and model is None),
            (
                _ISSUE_NOT_PRIVATE,
                IssueSeverity.WARNING,
                private_only and model is not None and not is_private(model),
            ),
        ):
            issue_id = issue.format(entry_id=entry.entry_id, target=target)
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
                translation_placeholders={"model": model_id, "name": name},
            )
            _LOGGER.warning("Model %s of %s: %s", model_id, name, issue_id)


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
    for domain, issue_id in list(registry.issues):
        if domain == DOMAIN and entry_id in issue_id:
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
        metrics={subentry_id: VeniceAIMetrics() for subentry_id in entry.subentries},
    )

    _LOGGER.info("Forwarding entry setups to platforms: %s", PLATFORMS)
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    _LOGGER.info("Successfully forwarded entry setups")

    await async_setup_repairs(hass, entry)
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))
    return True


async def _async_update_listener(
    hass: HomeAssistant, entry: VeniceAIConfigEntry
) -> None:
    """Reload the entry when its options or subentries change."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_migrate_entry(hass: HomeAssistant, entry: VeniceAIConfigEntry) -> bool:
    """Migrate a config entry to the current version."""
    if entry.version > 2:
        # Downgrading from a future version is not supported.
        _LOGGER.error(
            "Cannot downgrade Venice AI entry %s from version %s.%s",
            entry.entry_id,
            entry.version,
            entry.minor_version,
        )
        return False

    if entry.version == 1:
        _migrate_to_subentries(hass, entry)
    if entry.minor_version < 2:
        _remove_entry_device(hass, entry)
    return True


# Options of version 1 that move into each subentry, and the unique ID suffix of
# the entity that moves with them
_V1_SUBENTRIES = (
    (
        SUBENTRY_CONVERSATION,
        DEFAULT_CONVERSATION_NAME,
        Platform.CONVERSATION,
        "conversation",
        (
            CONF_PROMPT,
            CONF_LLM_HASS_API,
            CONF_CHAT_MODEL,
            CONF_MAX_TOKENS,
            CONF_TEMPERATURE,
            CONF_TOP_P,
            CONF_DISABLE_THINKING,
            CONF_STRIP_THINKING_RESPONSE,
            CONF_STREAM_RESPONSE,
            CONF_MAX_TOOL_ITERATIONS,
            CONF_MAX_HISTORY_MESSAGES,
        ),
    ),
    (
        SUBENTRY_AI_TASK,
        DEFAULT_AI_TASK_NAME,
        Platform.AI_TASK,
        "task",
        (
            CONF_CHAT_MODEL,
            CONF_MAX_TOKENS,
            CONF_TEMPERATURE,
            CONF_TOP_P,
            CONF_DISABLE_THINKING,
        ),
    ),
    (
        SUBENTRY_TTS,
        DEFAULT_TTS_NAME,
        Platform.TTS,
        "tts",
        (CONF_TTS_MODEL, CONF_TTS_VOICE, CONF_TTS_SPEED),
    ),
    (SUBENTRY_STT, DEFAULT_STT_NAME, Platform.STT, "stt", (CONF_STT_MODEL,)),
)
_V1_ENTRY_OPTIONS = (CONF_PRIVATE_MODELS_ONLY, CONF_REQUEST_TIMEOUT, CONF_IMAGE_MODEL)
_DEFAULTS = {
    SUBENTRY_CONVERSATION: {
        CONF_PROMPT: DEFAULT_SYSTEM_PROMPT,
        CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL,
    },
    SUBENTRY_AI_TASK: RECOMMENDED_AI_TASK_OPTIONS,
    SUBENTRY_TTS: RECOMMENDED_TTS_OPTIONS,
    SUBENTRY_STT: RECOMMENDED_STT_OPTIONS,
}


def _migrate_to_subentries(hass: HomeAssistant, entry: VeniceAIConfigEntry) -> None:
    """Move the options of each feature, and its entity, into a subentry."""
    options = entry.options
    entity_registry = er.async_get(hass)
    for subentry_type, title, platform, suffix, keys in _V1_SUBENTRIES:
        data = {**_DEFAULTS[subentry_type]}
        data.update((key, options[key]) for key in keys if key in options)
        if subentry_type in (SUBENTRY_CONVERSATION, SUBENTRY_AI_TASK):
            data[CONF_RECOMMENDED] = False
        subentry = ConfigSubentry(
            data=MappingProxyType(data),
            subentry_type=subentry_type,
            title=title,
            unique_id=None,
        )
        hass.config_entries.async_add_subentry(entry, subentry)
        entity_id = entity_registry.async_get_entity_id(
            platform, DOMAIN, f"{entry.entry_id}_{suffix}"
        )
        if entity_id is not None:
            entity_registry.async_update_entity(
                entity_id,
                config_subentry_id=subentry.subentry_id,
                device_id=None,
                new_unique_id=subentry.subentry_id,
            )
    hass.config_entries.async_update_entry(
        entry,
        options={key: options[key] for key in _V1_ENTRY_OPTIONS if key in options},
        version=2,
        minor_version=1,
    )


def _remove_entry_device(hass: HomeAssistant, entry: VeniceAIConfigEntry) -> None:
    """Remove the device of the entry with its usage sensors, now on each service."""
    device_registry = dr.async_get(hass)
    device = device_registry.async_get_device(identifiers={(DOMAIN, entry.entry_id)})
    if device is not None:
        device_registry.async_remove_device(device.id)
    hass.config_entries.async_update_entry(entry, minor_version=2)


async def async_unload_entry(hass: HomeAssistant, entry: VeniceAIConfigEntry) -> bool:
    """Unload Venice AI."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    await async_unload_repairs(hass, entry)

    client: AsyncVeniceAIClient = entry.runtime_data.client
    await client.close()

    return unload_ok
