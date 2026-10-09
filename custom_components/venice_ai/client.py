"""Venice AI API client built on the OpenAI Python SDK.

Venice exposes an OpenAI-compatible API, so transport, retries and SSE parsing
are delegated to ``openai.AsyncOpenAI``. This module keeps a small facade that
maps SDK errors to the integration's exception types, records usage metrics
and passes Venice-specific parameters through ``extra_body``.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import logging
import time
from typing import Any, cast

import openai

from .const import MAX_RETRIES, RECOMMENDED_REQUEST_TIMEOUT

_LOGGER = logging.getLogger(__name__)

VENICE_BASE_URL = "https://api.venice.ai/api/v1"

# Chat completion arguments the SDK accepts directly; anything else (for
# example ``venice_parameters``) is sent through ``extra_body``.
_CHAT_ARGS = frozenset(
    {
        "model",
        "messages",
        "max_tokens",
        "temperature",
        "top_p",
        "tools",
        "tool_choice",
        "stream_options",
        "response_format",
    }
)


def _sanitize_header_value(value: str | None) -> str:
    """Strip CR/LF from a header value before it goes on the wire.

    Only carriage returns and line feeds are removed. The value is not
    otherwise trimmed, since altering whitespace in an API key turns a valid
    key into a rejected one.
    """
    if not value:
        return ""
    return value.replace("\r", "").replace("\n", "")


@dataclass
class VeniceAIMetrics:
    """In-memory usage counters backing the diagnostic sensors.

    Counters are cumulative for the lifetime of the client, i.e. until the
    config entry is reloaded.
    """

    request_count: int = 0
    error_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    last_error: str | None = None
    _listeners: list[Callable[[], None]] = field(
        default_factory=list, repr=False, compare=False
    )

    def add_listener(self, listener: Callable[[], None]) -> Callable[[], None]:
        """Call ``listener`` after every change; return a function to remove it."""
        self._listeners.append(listener)

        def _remove() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return _remove

    def _notify(self) -> None:
        for listener in list(self._listeners):
            listener()

    def record_request(self) -> None:
        """Increment the total request counter."""
        self.request_count += 1
        self._notify()

    def record_error(self, error: BaseException) -> None:
        """Increment the error counter and remember the last error message."""
        self.error_count += 1
        self.last_error = f"{type(error).__name__}: {error}"
        self._notify()

    def record_usage(self, usage: dict[str, Any] | None) -> None:
        """Accumulate token usage from an API ``usage`` block, if present."""
        if not isinstance(usage, dict):
            return
        self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        self.total_tokens += int(usage.get("total_tokens", 0) or 0)
        self._notify()


class VeniceAIError(Exception):
    """Base exception for Venice AI errors."""


class AuthenticationError(VeniceAIError):
    """Authentication error (HTTP 401)."""


class RateLimitError(VeniceAIError):
    """Rate-limit error (HTTP 429)."""


class ServiceUnavailableError(VeniceAIError):
    """Venice AI is temporarily unavailable (HTTP 5xx)."""


class NetworkError(VeniceAIError):
    """The Venice AI API could not be reached (timeout, connection error)."""


def _categorize_http_error(
    status_code: int, error_detail: str, context: str = ""
) -> VeniceAIError:
    """Return the most specific VeniceAIError subtype for an HTTP status code."""
    suffix = f" ({context})" if context else ""
    msg = f"HTTP error {status_code}{suffix}: {error_detail}"
    if status_code == 401:
        return AuthenticationError(f"Invalid API key{suffix}")
    if status_code == 429:
        return RateLimitError(msg)
    if status_code >= 500:
        return ServiceUnavailableError(msg)
    return VeniceAIError(msg)


def _error_message(err: openai.APIStatusError) -> str:
    body = err.body
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict):
            return str(error.get("message") or error)
        return str(error)
    return err.message


def _convert_error(err: openai.OpenAIError, context: str) -> VeniceAIError:
    """Map an OpenAI SDK exception to the integration's exception types."""
    if isinstance(err, openai.APIStatusError):
        return _categorize_http_error(err.status_code, _error_message(err), context)
    if isinstance(err, openai.APIConnectionError):
        return NetworkError(f"Request error ({context}): {err}")
    return VeniceAIError(f"{context}: {err}")


class ChatCompletionChunk:
    """A streamed chat completion chunk as plain data."""

    def __init__(self, data: dict[str, Any]) -> None:
        """Initialize the chunk from its JSON payload."""
        self.choices: list[dict[str, Any]] = data.get("choices") or []
        self.usage: dict[str, Any] | None = data.get("usage")


def _split_chat_args(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a chat payload into SDK arguments and Venice ``extra_body``."""
    args: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for key, value in payload.items():
        if value is None or key == "stream":
            continue
        if key in _CHAT_ARGS:
            args[key] = value
        else:
            extra[key] = value
    return args, extra


class ChatCompletions:
    """Chat completions API."""

    def __init__(self, client: AsyncVeniceAIClient) -> None:
        """Initialize chat completions."""
        self.client = client
        self.completions = self

    @asynccontextmanager
    async def create(
        self, **kwargs: Any
    ) -> AsyncGenerator[AsyncIterator[ChatCompletionChunk]]:
        """Open a streaming chat completion."""
        args, extra = _split_chat_args(kwargs)
        metrics = self.client.metrics
        metrics.record_request()
        try:
            stream = await self.client.sdk.chat.completions.create(
                **args, stream=True, extra_body=extra or None
            )
        except openai.OpenAIError as err:
            converted = _convert_error(err, "streaming chat")
            metrics.record_error(converted)
            raise converted from err

        async def _chunks() -> AsyncIterator[ChatCompletionChunk]:
            try:
                async for raw in stream:
                    chunk = ChatCompletionChunk(raw.model_dump(exclude_none=True))
                    if chunk.usage:
                        metrics.record_usage(chunk.usage)
                    yield chunk
            except openai.OpenAIError as err:
                converted = _convert_error(err, "streaming chat")
                metrics.record_error(converted)
                raise converted from err

        try:
            yield _chunks()
        finally:
            await stream.close()

    async def create_non_streaming(
        self, payload: dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Create a non-streaming chat completion and return it as a dict."""
        args, extra = _split_chat_args({**(payload or {}), **kwargs})
        result = await self.client.call(
            "chat completion",
            self.client.sdk.chat.completions.create(
                **args, stream=False, extra_body=extra or None
            ),
        )
        data = result.model_dump(exclude_none=True)
        self.client.metrics.record_usage(data.get("usage"))
        return data


class Models:
    """Models API with a short-lived cache."""

    _CACHE_TTL_SECONDS = 3600

    def __init__(self, client: AsyncVeniceAIClient) -> None:
        """Initialize the models API."""
        self.client = client
        self._cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}

    async def list(self, model_type: str = "text") -> list[dict[str, Any]]:
        """List models of the given Venice type."""
        cached = self._cache.get(model_type)
        now = time.monotonic()
        if cached and now - cached[0] < self._CACHE_TTL_SECONDS:
            return cached[1]
        page = await self.client.call(
            "fetching models",
            self.client.sdk.models.list(extra_query={"type": model_type}),
        )
        models = [model.model_dump() for model in page.data]
        self._cache[model_type] = (now, models)
        return models


class Speech:
    """Text-to-speech API."""

    def __init__(self, client: AsyncVeniceAIClient) -> None:
        """Initialize the speech API."""
        self.client = client

    async def generate(
        self,
        text: str,
        voice: str,
        model: str,
        audio_output: str = "mp3",
        speed: float = 1.0,
        language: str | None = None,
    ) -> bytes:
        """Generate speech audio from text."""
        response = await self.client.call(
            "generating speech",
            self.client.sdk.audio.speech.create(
                input=text,
                model=model,
                voice=voice,
                response_format=cast(Any, audio_output),
                speed=speed,
                extra_body={"language": language} if language else None,
            ),
        )
        return response.content

    async def generate_streaming(
        self,
        text: str,
        voice: str,
        model: str,
        audio_output: str = "mp3",
        speed: float = 1.0,
        language: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Generate speech audio, yielding chunks as they arrive."""
        extra: dict[str, Any] = {"streaming": True}
        if language:
            extra["language"] = language
        metrics = self.client.metrics
        metrics.record_request()
        try:
            async with self.client.sdk.audio.speech.with_streaming_response.create(
                input=text,
                model=model,
                voice=voice,
                response_format=cast(Any, audio_output),
                speed=speed,
                extra_body=extra,
            ) as response:
                async for chunk in response.iter_bytes():
                    yield chunk
        except openai.OpenAIError as err:
            converted = _convert_error(err, "streaming speech")
            metrics.record_error(converted)
            raise converted from err


class Transcriptions:
    """Speech-to-text API."""

    def __init__(self, client: AsyncVeniceAIClient) -> None:
        """Initialize the transcriptions API."""
        self.client = client

    async def create(
        self, audio_data: bytes, model: str, language: str | None = None
    ) -> str:
        """Transcribe WAV audio and return the text."""
        result = await self.client.call(
            "creating transcription",
            self.client.sdk.audio.transcriptions.create(
                file=("audio.wav", audio_data, "audio/wav"),
                model=model,
                response_format="json",
                extra_body={"language": language} if language else None,
            ),
        )
        return str(result.text)


class Images:
    """Image generation API."""

    def __init__(self, client: AsyncVeniceAIClient) -> None:
        """Initialize the images API."""
        self.client = client

    async def generate(
        self,
        model: str,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "standard",
        style: str = "vivid",
        response_format: str = "url",
        n: int = 1,
    ) -> dict[str, Any]:
        """Generate an image."""
        result = await self.client.call(
            "generating image",
            self.client.sdk.images.generate(
                model=model,
                prompt=prompt,
                size=cast(Any, size),
                quality=cast(Any, quality),
                style=cast(Any, style),
                response_format=cast(Any, response_format),
                n=n,
            ),
        )
        return result.model_dump(exclude_none=True)


class AsyncVeniceAIClient:
    """Async client for the Venice AI API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = VENICE_BASE_URL,
        http_client: Any = None,
        timeout: float = RECOMMENDED_REQUEST_TIMEOUT,
    ) -> None:
        """Initialize the client.

        ``http_client`` should be Home Assistant's shared async HTTP client;
        it is never closed by this client.
        """
        self.timeout = timeout
        self.metrics = VeniceAIMetrics()
        self._owns_http_client = http_client is None
        self.sdk = openai.AsyncOpenAI(
            api_key=_sanitize_header_value(api_key),
            base_url=base_url,
            http_client=http_client,
            timeout=timeout,
            max_retries=MAX_RETRIES,
        )
        self.chat = ChatCompletions(self)
        self.models = Models(self)
        self.speech = Speech(self)
        self.transcriptions = Transcriptions(self)
        self.images = Images(self)

    async def call(self, context: str, request: Any) -> Any:
        """Await an SDK request, recording metrics and converting errors."""
        self.metrics.record_request()
        try:
            return await request
        except openai.OpenAIError as err:
            converted = _convert_error(err, context)
            self.metrics.record_error(converted)
            raise converted from err

    async def validate_api_key(self) -> None:
        """Raise AuthenticationError if the API key is not accepted.

        The models endpoint is public, so an authenticated endpoint is used.
        """
        await self.call(
            "validating API key",
            self.sdk.get("/api_keys/rate_limits", cast_to=object),
        )

    async def close(self) -> None:
        """Close the HTTP client if this client created it."""
        if self._owns_http_client:
            await self.sdk.close()

    async def __aenter__(self) -> AsyncVeniceAIClient:
        """Enter the async context."""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Close the client on exit."""
        await self.close()
