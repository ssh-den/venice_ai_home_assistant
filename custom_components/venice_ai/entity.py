"""Shared entity helpers for Venice AI."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any

from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.entity import Entity

from .const import DOMAIN

if TYPE_CHECKING:
    from . import VeniceAIConfigEntry


def device_info(entry: ConfigEntry) -> dr.DeviceInfo:
    """Return the service device of the config entry, for its usage sensors."""
    return dr.DeviceInfo(
        identifiers={(DOMAIN, entry.entry_id)},
        name=entry.title,
        manufacturer="Venice AI",
        entry_type=dr.DeviceEntryType.SERVICE,
    )


def subentries_of(entry: ConfigEntry, subentry_type: str) -> Iterator[ConfigSubentry]:
    """Return the subentries of one type."""
    return (s for s in entry.subentries.values() if s.subentry_type == subentry_type)


class VeniceAIEntity(Entity):
    """An entity of one subentry, on a service device of its own."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(
        self, entry: VeniceAIConfigEntry, subentry: ConfigSubentry, model: str
    ) -> None:
        """Initialize the entity."""
        self.entry = entry
        self.subentry = subentry
        self._attr_unique_id = subentry.subentry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer="Venice AI",
            model=model,
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    @property
    def options(self) -> Mapping[str, Any]:
        """Return the settings of the subentry."""
        return self.subentry.data
