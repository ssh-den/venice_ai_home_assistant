"""Shared pytest fixtures and fakes for the Venice AI test suite."""

# pylint: disable=redefined-outer-name

from __future__ import annotations

from collections.abc import Generator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.venice_ai.client import VeniceAIMetrics
from custom_components.venice_ai.const import (
    DOMAIN,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_STT_MODEL,
    RECOMMENDED_TTS_MODEL,
)

# Text model with structured output but without function calling
SCHEMA_MODEL = "schema-model"

MODELS_BY_TYPE: dict[str, list[dict[str, Any]]] = {
    "text": [
        {
            "id": RECOMMENDED_CHAT_MODEL,
            "type": "text",
            "model_spec": {
                "privacy": "private",
                "capabilities": {
                    "supportsFunctionCalling": True,
                    "supportsTeeAttestation": True,
                    "supportsE2EE": True,
                },
                "pricing": {"input": {"usd": 0.18}, "output": {"usd": 0.37}},
            },
        },
        {
            "id": SCHEMA_MODEL,
            "type": "text",
            "model_spec": {
                "privacy": "anonymized",
                "capabilities": {"supportsResponseSchema": True},
            },
        },
    ],
    "tts": [
        {
            "id": RECOMMENDED_TTS_MODEL,
            "type": "tts",
            "model_spec": {"voices": ["bm_daniel", "af_heart"]},
        }
    ],
    "asr": [{"id": RECOMMENDED_STT_MODEL, "type": "asr"}],
    "image": [
        {"id": "venice-sd35", "type": "image"},
        {"id": "hidream", "type": "image"},
    ],
}


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations: None) -> None:
    """Enable loading the custom integration in every test."""


@pytest.fixture
async def ha_core(hass: HomeAssistant) -> None:
    """Set up the homeassistant core component required by conversation."""
    assert await async_setup_component(hass, "homeassistant", {})


@pytest.fixture
def mock_config_entry(hass: HomeAssistant) -> MockConfigEntry:
    """Return a Venice AI config entry added to hass."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Venice AI",
        data={CONF_API_KEY: "test-key"},
        options={},
    )
    entry.add_to_hass(hass)
    return entry


@pytest.fixture
def mock_client() -> Generator[MagicMock]:
    """Patch the Venice AI client used by the integration."""
    client = MagicMock()
    client.metrics = VeniceAIMetrics()
    client.close = AsyncMock()
    client.validate_api_key = AsyncMock()

    async def _list(model_type: str = "text") -> list[dict[str, Any]]:
        return [dict(m) for m in MODELS_BY_TYPE.get(model_type, [])]

    client.models.list = AsyncMock(side_effect=_list)
    client.chat.create_non_streaming = AsyncMock(
        return_value={"choices": [{"message": {"content": "ok"}}]}
    )
    client.images.generate = AsyncMock(
        return_value={"data": [{"url": "https://example.com/a.png", "b64_json": "x"}]}
    )
    client.__aenter__.return_value = client
    with (
        patch("custom_components.venice_ai.AsyncVeniceAIClient", return_value=client),
        patch(
            "custom_components.venice_ai.config_flow.AsyncVeniceAIClient",
            return_value=client,
        ),
    ):
        yield client


@pytest.fixture
async def setup_integration(
    hass: HomeAssistant,
    ha_core: None,
    mock_config_entry: MockConfigEntry,
    mock_client: MagicMock,
) -> MockConfigEntry:
    """Set up the integration with a mocked client."""
    assert await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    return mock_config_entry


class FakeChunk:
    """Minimal stand-in for ``ChatCompletionChunk`` used by the service layer."""

    def __init__(
        self,
        choices: list[dict[str, Any]],
        usage: dict[str, Any] | None = None,
    ) -> None:
        self.choices = choices
        self.usage = usage


class FakeStream:
    """Async-iterable stream of :class:`FakeChunk` objects."""

    def __init__(self, chunks: list[FakeChunk]) -> None:
        self._chunks = chunks
        self._iter = iter(chunks)

    def __aiter__(self) -> FakeStream:
        self._iter = iter(self._chunks)
        return self

    async def __anext__(self) -> FakeChunk:
        try:
            return next(self._iter)
        except StopIteration as err:  # pragma: no cover - trivial
            raise StopAsyncIteration from err


class FakeChatCompletions:
    """Fake of ``client.chat`` supporting streaming and non-streaming calls."""

    def __init__(
        self,
        *,
        chunks: list[FakeChunk] | None = None,
        non_streaming_response: dict[str, Any] | None = None,
    ) -> None:
        self._chunks = chunks or []
        self._non_streaming_response = non_streaming_response or {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}]
        }
        self.last_create_kwargs: dict[str, Any] | None = None
        self.last_non_streaming_kwargs: dict[str, Any] | None = None

    @asynccontextmanager
    async def create(self, **kwargs: Any):
        """Mimic the streaming async-context-manager API."""
        self.last_create_kwargs = kwargs
        yield FakeStream(self._chunks)

    async def create_non_streaming(self, **kwargs: Any) -> dict[str, Any]:
        """Mimic the non-streaming API."""
        self.last_non_streaming_kwargs = kwargs
        return self._non_streaming_response


class FakeClient:
    """Fake ``AsyncVeniceAIClient`` exposing only the ``chat`` namespace."""

    def __init__(self, chat: FakeChatCompletions) -> None:
        self.chat = chat


@pytest.fixture
def make_client():
    """Return a factory building a :class:`FakeClient`."""

    def _factory(
        *,
        chunks: list[FakeChunk] | None = None,
        non_streaming_response: dict[str, Any] | None = None,
    ) -> FakeClient:
        return FakeClient(
            FakeChatCompletions(
                chunks=chunks,
                non_streaming_response=non_streaming_response,
            )
        )

    return _factory


@pytest.fixture
def chunk():
    """Return a helper to build a single-choice :class:`FakeChunk` from a delta."""

    def _build(delta: dict[str, Any]) -> FakeChunk:
        return FakeChunk([{"delta": delta}])

    return _build
