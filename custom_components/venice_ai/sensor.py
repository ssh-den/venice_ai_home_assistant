"""Diagnostic sensors with the API usage of each Venice AI service."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from homeassistant.components.sensor import (
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .client import VeniceAIMetrics
from .const import SUBENTRY_AI_TASK, SUBENTRY_CONVERSATION
from .entity import service_device_info

if TYPE_CHECKING:
    from . import VeniceAIConfigEntry


@dataclass(frozen=True, kw_only=True)
class VeniceAISensorDescription(SensorEntityDescription):
    """Describes a Venice AI diagnostic sensor."""

    value_fn: Callable[[VeniceAIMetrics], int | str | None]
    tokens: bool = False


SENSORS: tuple[VeniceAISensorDescription, ...] = (
    VeniceAISensorDescription(
        key="request_count",
        translation_key="request_count",
        icon="mdi:api",
        state_class=SensorStateClass.TOTAL_INCREASING,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda m: m.request_count,
    ),
    VeniceAISensorDescription(
        key="error_count",
        translation_key="error_count",
        icon="mdi:alert-circle",
        state_class=SensorStateClass.TOTAL_INCREASING,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda m: m.error_count,
    ),
    VeniceAISensorDescription(
        key="total_tokens",
        translation_key="total_tokens",
        icon="mdi:counter",
        native_unit_of_measurement="tokens",
        state_class=SensorStateClass.TOTAL_INCREASING,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda m: m.total_tokens,
        tokens=True,
    ),
    VeniceAISensorDescription(
        key="prompt_tokens",
        translation_key="prompt_tokens",
        icon="mdi:counter",
        native_unit_of_measurement="tokens",
        state_class=SensorStateClass.TOTAL_INCREASING,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda m: m.prompt_tokens,
        tokens=True,
    ),
    VeniceAISensorDescription(
        key="completion_tokens",
        translation_key="completion_tokens",
        icon="mdi:counter",
        native_unit_of_measurement="tokens",
        state_class=SensorStateClass.TOTAL_INCREASING,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda m: m.completion_tokens,
        tokens=True,
    ),
    VeniceAISensorDescription(
        key="last_error",
        translation_key="last_error",
        icon="mdi:message-alert",
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda m: m.last_error,
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: VeniceAIConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the usage sensors of each service."""
    for subentry in entry.subentries.values():
        metrics = entry.runtime_data.metrics[subentry.subentry_id]
        async_add_entities(
            (
                VeniceAIUsageSensor(subentry, metrics, description)
                for description in SENSORS
                if not description.tokens
                or subentry.subentry_type in (SUBENTRY_CONVERSATION, SUBENTRY_AI_TASK)
            ),
            config_subentry_id=subentry.subentry_id,
        )


class VeniceAIUsageSensor(SensorEntity):
    """A diagnostic sensor reporting a single Venice AI usage metric."""

    _attr_has_entity_name = True
    _attr_should_poll = False
    entity_description: VeniceAISensorDescription

    def __init__(
        self,
        subentry: ConfigSubentry,
        metrics: VeniceAIMetrics,
        description: VeniceAISensorDescription,
    ) -> None:
        """Initialize the usage sensor."""
        self.entity_description = description
        self._metrics = metrics
        self._attr_unique_id = f"{subentry.subentry_id}_{description.key}"
        self._attr_device_info = service_device_info(subentry)

    async def async_added_to_hass(self) -> None:
        """Push state updates whenever the metrics change."""
        await super().async_added_to_hass()
        self.async_on_remove(self._metrics.add_listener(self.async_write_ha_state))

    @property
    def native_value(self) -> int | str | None:
        """Return the current metric value."""
        return self.entity_description.value_fn(self._metrics)
