"""Tests for the Venice AI HTTP client against a mocked transport."""

from __future__ import annotations

from collections.abc import Callable
import json
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from custom_components.venice_ai import client as client_module
from custom_components.venice_ai.client import (
    AsyncVeniceAIClient,
    AuthenticationError,
    NetworkError,
    VeniceAIMetrics,
)


def _make_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> AsyncVeniceAIClient:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return AsyncVeniceAIClient(api_key="key", http_client=http_client)


@pytest.fixture(autouse=True)
def no_retry_delay() -> Any:
    with patch.object(client_module, "RETRY_BASE_DELAY", 0):
        yield


def test_metrics_listener_called_and_removed() -> None:
    metrics = VeniceAIMetrics()
    calls: list[int] = []
    remove = metrics.add_listener(lambda: calls.append(metrics.request_count))
    metrics.record_request()
    metrics.record_usage({"total_tokens": 3})
    metrics.record_error(ValueError("x"))
    assert calls == [1, 1, 1]
    remove()
    metrics.record_request()
    assert calls == [1, 1, 1]
    remove()


async def test_non_streaming_records_request_and_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer key"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hi"}}],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "total_tokens": 5,
                },
            },
        )

    client = _make_client(handler)
    result = await client.chat.create_non_streaming(model="m", messages=[])
    assert result["choices"][0]["message"]["content"] == "hi"
    assert client.metrics.request_count == 1
    assert client.metrics.total_tokens == 5
    assert client.metrics.error_count == 0


async def test_http_error_recorded_once() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "nope"})

    client = _make_client(handler)
    with pytest.raises(AuthenticationError):
        await client.chat.create_non_streaming(model="m", messages=[])
    assert client.metrics.request_count == 1
    assert client.metrics.error_count == 1
    assert client.metrics.last_error is not None


async def test_network_error_after_retries_recorded() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = _make_client(handler)
    with pytest.raises(NetworkError):
        await client.images.generate(model="m", prompt="p")
    assert client.metrics.request_count == 1
    assert client.metrics.error_count == 1


async def test_retry_on_server_error_then_success() -> None:
    responses = iter([httpx.Response(503), httpx.Response(200, json={"data": []})])

    def handler(request: httpx.Request) -> httpx.Response:
        return next(responses)

    client = _make_client(handler)
    assert await client.models.list(model_type="retry-test") == []
    assert client.metrics.request_count == 1
    assert client.metrics.error_count == 0


async def test_streaming_records_request_and_usage() -> None:
    lines = [
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]},
        {
            "choices": [],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        },
    ]
    body = (
        "".join(f"data: {json.dumps(line)}\n\n" for line in lines) + "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, text=body)

    client = _make_client(handler)
    content = ""
    async with client.chat.create(model="m", messages=[]) as stream:
        async for chunk in stream:
            for choice in chunk.choices:
                content += choice["delta"].get("content") or ""
    assert content == "Hello"
    assert client.metrics.request_count == 1
    assert client.metrics.total_tokens == 6


async def test_streaming_http_error_recorded() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="down")

    client = _make_client(handler)
    with pytest.raises(client_module.ServiceUnavailableError):
        async with client.chat.create(model="m", messages=[]):
            pass
    assert client.metrics.request_count == 1
    assert client.metrics.error_count == 1


async def test_request_timeout_is_applied() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions["timeout"])
        return httpx.Response(200, json={"data": [{"url": "u"}]})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AsyncVeniceAIClient(api_key="k", http_client=http_client, timeout=42.0)
    await client.images.generate(model="m", prompt="p")
    assert seen[0]["read"] == 42.0
