"""Shared pytest fixtures and fakes for the Venice AI test suite."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations: None) -> None:
    """Enable loading the custom integration in every test."""


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
