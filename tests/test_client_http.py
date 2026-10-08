"""Tests for the Venice AI client against a mocked HTTP transport."""

from __future__ import annotations

from collections.abc import Callable, Iterator
import json
from typing import Any
from unittest.mock import patch

import httpx
import openai
import pytest

from custom_components.venice_ai.client import (
    AsyncVeniceAIClient,
    AuthenticationError,
    NetworkError,
    ServiceUnavailableError,
    VeniceAIMetrics,
)

Handler = Callable[[httpx.Request], httpx.Response]


def _make_client(handler: Handler, **kwargs: Any) -> AsyncVeniceAIClient:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return AsyncVeniceAIClient(api_key="key", http_client=http_client, **kwargs)


@pytest.fixture(autouse=True)
def no_retry_delay() -> Iterator[None]:
    with patch.object(
        openai._base_client.BaseClient, "_calculate_retry_timeout", return_value=0
    ):
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


async def test_non_streaming_chat() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer key"
        body = json.loads(request.content)
        assert body["venice_parameters"] == {"disable_thinking": True}
        assert body["stream"] is False
        assert "top_p" not in body
        return httpx.Response(
            200,
            json={
                "id": "1",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "hi"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "total_tokens": 5,
                },
            },
        )

    client = _make_client(handler)
    result = await client.chat.create_non_streaming(
        model="m",
        messages=[{"role": "user", "content": "hello"}],
        top_p=None,
        venice_parameters={"disable_thinking": True},
    )
    assert result["choices"][0]["message"]["content"] == "hi"
    assert client.metrics.request_count == 1
    assert client.metrics.total_tokens == 5
    assert client.metrics.error_count == 0


async def test_http_error_is_categorized_and_recorded() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "nope"})

    client = _make_client(handler)
    with pytest.raises(AuthenticationError):
        await client.chat.create_non_streaming(model="m", messages=[])
    assert client.metrics.request_count == 1
    assert client.metrics.error_count == 1


async def test_server_error_after_retries() -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(503, json={"error": {"message": "busy"}})

    client = _make_client(handler)
    with pytest.raises(ServiceUnavailableError, match="busy"):
        await client.images.generate(model="m", prompt="p")
    assert len(attempts) == 4
    assert client.metrics.error_count == 1


async def test_network_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = _make_client(handler)
    with pytest.raises(NetworkError):
        await client.images.generate(model="m", prompt="p")
    assert client.metrics.request_count == 1
    assert client.metrics.error_count == 1


async def test_validate_api_key_uses_authenticated_endpoint() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(401, json={"error": "Authentication failed"})

    client = _make_client(handler)
    with pytest.raises(AuthenticationError):
        await client.validate_api_key()
    assert paths == ["/api/v1/api_keys/rate_limits"]


async def test_models_list_passes_type_and_keeps_venice_fields() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["type"] == "image"
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {
                        "id": "flux",
                        "object": "model",
                        "created": 0,
                        "owned_by": "venice.ai",
                        "type": "image",
                        "model_spec": {"privacy": "private"},
                    }
                ],
            },
        )

    client = _make_client(handler)
    models = await client.models.list(model_type="image")
    assert models[0]["id"] == "flux"
    assert models[0]["model_spec"] == {"privacy": "private"}
    assert await client.models.list(model_type="image") is models


async def test_streaming_chat() -> None:
    lines: list[dict[str, Any]] = [
        {"choices": [{"index": 0, "delta": {"content": "Hel"}}]},
        {
            "choices": [
                {"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}
            ]
        },
        {
            "choices": [],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        },
    ]
    base = {"id": "1", "object": "chat.completion.chunk", "created": 0, "model": "m"}
    body = "".join(f"data: {json.dumps({**base, **line})}\n\n" for line in lines)
    body += "data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(
            200, text=body, headers={"content-type": "text/event-stream"}
        )

    client = _make_client(handler)
    content = ""
    async with client.chat.create(model="m", messages=[]) as stream:
        async for chunk in stream:
            for choice in chunk.choices:
                content += choice["delta"].get("content") or ""
    assert content == "Hello"
    assert client.metrics.request_count == 1
    assert client.metrics.total_tokens == 6


async def test_streaming_chat_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "bad model"}})

    client = _make_client(handler)
    with pytest.raises(Exception, match="bad model"):
        async with client.chat.create(model="m", messages=[]):
            pass
    assert client.metrics.error_count == 1


async def test_streaming_speech() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["streaming"] is True
        assert body["response_format"] == "ogg"
        return httpx.Response(200, content=b"audio-bytes")

    client = _make_client(handler)
    chunks = [
        chunk
        async for chunk in client.speech.generate_streaming(
            text="hi", audio_output="ogg"
        )
    ]
    assert b"".join(chunks) == b"audio-bytes"


async def test_transcription() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert b'name="timestamps"' in request.content
        return httpx.Response(200, json={"text": "hello"})

    client = _make_client(handler)
    result = await client.transcriptions.create(audio_data=b"RIFF", timestamps=True)
    assert result["text"] == "hello"


async def test_request_timeout_is_applied() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions["timeout"])
        return httpx.Response(200, json={"created": 0, "data": [{"url": "u"}]})

    client = _make_client(handler, timeout=42.0)
    await client.images.generate(model="m", prompt="p")
    assert seen[0]["read"] == 42.0


async def test_shared_http_client_is_not_closed() -> None:
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200))
    )
    async with AsyncVeniceAIClient(api_key="k", http_client=http_client):
        pass
    assert not http_client.is_closed
