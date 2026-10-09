"""Shared entity helpers for Venice AI."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any

from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.entity import Entity

from .const import DOMAIN, SUBENTRY_MODELS

if TYPE_CHECKING:
    from . import VeniceAIConfigEntry


def subentries_of(entry: ConfigEntry, subentry_type: str) -> Iterator[ConfigSubentry]:
    """Return the subentries of one type."""
    return (s for s in entry.subentries.values() if s.subentry_type == subentry_type)


def service_device_info(subentry: ConfigSubentry) -> dr.DeviceInfo:
    """Return the service device of a subentry."""
    key, default = SUBENTRY_MODELS[subentry.subentry_type]
    return dr.DeviceInfo(
        identifiers={(DOMAIN, subentry.subentry_id)},
        name=subentry.title,
        manufacturer="Venice AI",
        model=str(subentry.data.get(key, default)),
        entry_type=dr.DeviceEntryType.SERVICE,
    )


class VeniceAIEntity(Entity):
    """The entity of one subentry, on its service device."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: VeniceAIConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the entity."""
        self.entry = entry
        self.subentry = subentry
        self.client = entry.runtime_data.client.scoped(
            entry.runtime_data.metrics[subentry.subentry_id]
        )
        self._attr_unique_id = subentry.subentry_id
        self._attr_device_info = service_device_info(subentry)

    @property
    def options(self) -> Mapping[str, Any]:
        """Return the settings of the subentry."""
        return self.subentry.data
