"""Chat completion helpers shared by the conversation and AI Task entities."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
import json
import logging
from typing import Any

from .client import AsyncVeniceAIClient

_LOGGER = logging.getLogger(__name__)

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


@dataclass
class ChatParameters:
    """Tunable parameters for a chat completion request.

    Centralises the per-request knobs so platform code can build one object
    from config options instead of threading many positional arguments through
    the service methods.
    """

    model: str
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    tools: list[dict[str, Any]] | None = None
    venice_parameters: dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None


@dataclass
class StreamOutcome:
    """Values only known once a streamed completion has finished."""

    finish_reason: str = "unknown"


class ThinkingFilter:
    """Split streamed text into answer and <think>...</think> reasoning.

    Tags may be split across chunks, so a possible tag prefix is held back
    until the next chunk arrives.
    """

    def __init__(self) -> None:
        """Initialize the filter outside a thinking block."""
        self._buffer = ""
        self._thinking = False

    def feed(self, text: str) -> tuple[str, str]:
        """Return the (answer, thinking) text that can be emitted so far."""
        self._buffer += text
        answer: list[str] = []
        thinking: list[str] = []
        while self._buffer:
            tag = _THINK_CLOSE if self._thinking else _THINK_OPEN
            out = thinking if self._thinking else answer
            index = self._buffer.lower().find(tag)
            if index != -1:
                out.append(self._buffer[:index])
                self._buffer = self._buffer[index + len(tag) :]
                self._thinking = not self._thinking
                continue
            keep = _partial_tag_length(self._buffer, tag)
            out.append(self._buffer[: len(self._buffer) - keep])
            self._buffer = self._buffer[len(self._buffer) - keep :]
            break
        return "".join(answer), "".join(thinking)

    def flush(self) -> tuple[str, str]:
        """Return whatever text is still held back."""
        rest, self._buffer = self._buffer, ""
        return ("", rest) if self._thinking else (rest, "")


def _partial_tag_length(text: str, tag: str) -> int:
    """Length of the longest suffix of text that is a prefix of tag."""
    lowered = text.lower()
    for size in range(min(len(tag) - 1, len(text)), 0, -1):
        if tag.startswith(lowered[-size:]):
            return size
    return 0


class VeniceConversationService:
    """Run chat completions for Home Assistant entities."""

    def __init__(self, client: AsyncVeniceAIClient) -> None:
        """Initialize the service with a Venice AI client."""
        self._client = client

    async def chat(
        self,
        messages: list[dict[str, Any]],
        params: ChatParameters,
    ) -> dict[str, Any]:
        """Perform a non-streaming chat completion and return the raw response."""
        return await self._client.chat.create_non_streaming(
            model=params.model,
            messages=messages,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            tools=params.tools or None,
            venice_parameters=params.venice_parameters,
            response_format=params.response_format,
            stream=False,
        )

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        params: ChatParameters,
        outcome: StreamOutcome,
    ) -> AsyncGenerator[dict[str, Any]]:
        """Stream a chat completion as deltas.

        Yields ``{"content": ...}`` and ``{"reasoning": ...}`` as they arrive,
        then one ``{"tool_calls": [...]}`` with the reassembled tool calls.
        """
        tool_calls: dict[int, dict[str, Any]] = {}
        async with self._client.chat.create(
            model=params.model,
            messages=messages,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            tools=params.tools or None,
            venice_parameters=params.venice_parameters,
            response_format=params.response_format,
            stream_options={"include_usage": True},
        ) as stream:
            async for chunk in stream:
                for choice in chunk.choices:
                    if finish_reason := choice.get("finish_reason"):
                        outcome.finish_reason = finish_reason
                    delta = choice.get("delta") or {}
                    if reasoning := delta.get("reasoning_content"):
                        yield {"reasoning": reasoning}
                    if content := delta.get("content"):
                        yield {"content": content}
                    for fragment in delta.get("tool_calls") or []:
                        _merge_tool_call_fragment(tool_calls, fragment)
        if tool_calls:
            yield {"tool_calls": [tool_calls[i] for i in sorted(tool_calls)]}


def _merge_tool_call_fragment(
    accumulator: dict[int, dict[str, Any]],
    fragment: dict[str, Any],
) -> None:
    """Merge a streamed tool-call delta fragment into the accumulator.

    Streaming APIs send tool calls in pieces: the first fragment for a given
    ``index`` typically carries the ``id`` and function ``name`` while later
    fragments append to the function ``arguments`` string. This reassembles
    them into a single OpenAI-style tool-call dict.
    """
    if not isinstance(fragment, dict):
        return
    index = fragment.get("index", 0)
    existing = accumulator.setdefault(
        index,
        {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
    )

    if fragment.get("id"):
        existing["id"] = fragment["id"]
    if fragment.get("type"):
        existing["type"] = fragment["type"]

    func = fragment.get("function", {})
    if isinstance(func, dict):
        if func.get("name"):
            existing["function"]["name"] += func["name"]
        if func.get("arguments"):
            existing["function"]["arguments"] += func["arguments"]


def extract_json(text: str) -> Any:
    """Parse a JSON payload, tolerating surrounding markdown code fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
        cleaned = cleaned.rsplit("```", 1)[0]
    return json.loads(cleaned)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output.

    Handles both the XML-style tags used by some reasoning models and the
    literal ' thinking' / ' end of thinking' markers emitted by Venice AI.
    """
    if not text:
        return text
    # XML-style <think>...</think>
    while True:
        start = text.lower().find("<think>")
        if start == -1:
            break
        end = text.lower().find("</think>", start)
        if end == -1:
            # unmatched open tag - strip to end to be safe
            text = text[:start].strip()
            break
        text = text[:start] + text[end + 8 :]
    # Venice-style ' thinking' ... ' end of thinking'
    if " thinking" in text:
        parts = text.split(" end of thinking")
        if len(parts) > 1:
            text = parts[-1].strip()
    return text.strip()
