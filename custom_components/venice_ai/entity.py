"""Shared entity helpers for Venice AI."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers import device_registry as dr

from .const import DOMAIN


def device_info(entry: ConfigEntry) -> dr.DeviceInfo:
    """Return the service device every Venice AI entity belongs to."""
    return dr.DeviceInfo(
        identifiers={(DOMAIN, entry.entry_id)},
        name=entry.title,
        manufacturer="Venice AI",
        entry_type=dr.DeviceEntryType.SERVICE,
    )
