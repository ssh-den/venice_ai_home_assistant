"""Chat completion helpers shared by the conversation and AI Task entities."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass
import json
import logging
from typing import Any

from homeassistant.components import conversation

from .client import AsyncVeniceAIClient, VeniceAIError
from .const import (
    CONF_DISABLE_THINKING,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
    CONF_THINKING_TAGS,
    CONF_TOP_P,
    CONF_VENICE_SYSTEM_PROMPT,
    RECOMMENDED_DISABLE_THINKING,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_THINKING_TAGS,
    RECOMMENDED_TOP_P,
    RECOMMENDED_VENICE_SYSTEM_PROMPT,
)

_LOGGER = logging.getLogger(__name__)


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


def chat_parameters(
    options: Mapping[str, Any],
    model: str,
    *,
    tools: list[dict[str, Any]] | None = None,
    response_format: dict[str, Any] | None = None,
) -> ChatParameters:
    """Build request parameters from config entry options."""
    return ChatParameters(
        model=model,
        max_tokens=options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
        temperature=options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
        top_p=options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
        tools=tools,
        venice_parameters={
            "include_venice_system_prompt": bool(
                options.get(CONF_VENICE_SYSTEM_PROMPT, RECOMMENDED_VENICE_SYSTEM_PROMPT)
            ),
            **(
                {"disable_thinking": True}
                if options.get(CONF_DISABLE_THINKING, RECOMMENDED_DISABLE_THINKING)
                else {}
            ),
        },
        response_format=response_format,
    )


def thinking_tags(options: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the configured names of the tags that wrap reasoning."""
    value = str(options.get(CONF_THINKING_TAGS, RECOMMENDED_THINKING_TAGS))
    names = (name.strip(" <>/").lower() for name in value.split(","))
    return tuple(dict.fromkeys(name for name in names if name))


def _convert_content(
    content: conversation.Content, tags: tuple[str, ...]
) -> dict[str, Any] | None:
    """Convert one chat log entry to a Venice AI chat message."""
    if isinstance(content, conversation.SystemContent):
        return {"role": "system", "content": content.content}
    if isinstance(content, conversation.UserContent):
        return {"role": "user", "content": content.content}
    if isinstance(content, conversation.ToolResultContent):
        return {
            "role": "tool",
            "tool_call_id": content.tool_call_id,
            "content": json.dumps(content.tool_result),
        }
    if isinstance(content, conversation.AssistantContent):
        text = strip_thinking(content.content or "", tags)
        message: dict[str, Any] = {"role": "assistant", "content": text}
        if content.tool_calls:
            message["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.tool_name,
                        "arguments": json.dumps(tool_call.tool_args),
                    },
                }
                for tool_call in content.tool_calls
            ]
        return message
    _LOGGER.warning("Unsupported chat log content: %s", type(content).__name__)
    return None


def chat_log_to_messages(
    chat_log: conversation.ChatLog, tags: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Convert the chat log into chat messages, removing reasoning in ``tags``."""
    messages = []
    for content in chat_log.content:
        if (message := _convert_content(content, tags)) is not None:
            messages.append(message)
    return messages


@dataclass
class StreamOutcome:
    """Values only known once a streamed completion has finished."""

    finish_reason: str = "unknown"


class ThinkingFilter:
    """Split streamed text into answer and reasoning wrapped in the given tags.

    Tags may be split across chunks, so a possible tag prefix is held back
    until the next chunk arrives.
    """

    def __init__(self, tags: tuple[str, ...]) -> None:
        """Initialize the filter outside a thinking block."""
        self._closing = {f"<{tag}>": f"</{tag}>" for tag in tags}
        self._close: str | None = None
        self._buffer = ""

    def _next_tag(self, text: str) -> tuple[int, str] | None:
        if self._close is not None:
            index = text.find(self._close)
            return (index, self._close) if index != -1 else None
        found = [(i, tag) for tag in self._closing if (i := text.find(tag)) != -1]
        return min(found) if found else None

    def _held_back(self, text: str) -> int:
        tags = [self._close] if self._close is not None else list(self._closing)
        return max((_partial_tag_length(text, tag) for tag in tags), default=0)

    def feed(self, text: str) -> tuple[str, str]:
        """Return the (answer, thinking) text that can be emitted so far."""
        self._buffer += text
        answer: list[str] = []
        thinking: list[str] = []
        while self._buffer:
            out = answer if self._close is None else thinking
            found = self._next_tag(self._buffer.lower())
            if found is not None:
                index, tag = found
                out.append(self._buffer[:index])
                self._buffer = self._buffer[index + len(tag) :]
                self._close = self._closing[tag] if self._close is None else None
                continue
            keep = self._held_back(self._buffer)
            out.append(self._buffer[: len(self._buffer) - keep])
            self._buffer = self._buffer[len(self._buffer) - keep :]
            break
        return "".join(answer), "".join(thinking)

    def flush(self) -> tuple[str, str]:
        """Return whatever text is still held back."""
        rest, self._buffer = self._buffer, ""
        return (rest, "") if self._close is None else ("", rest)


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

    async def chat_deltas(
        self,
        messages: list[dict[str, Any]],
        params: ChatParameters,
        outcome: StreamOutcome,
        stream: bool,
    ) -> AsyncGenerator[dict[str, Any]]:
        """Yield the deltas of a completion, streamed or in one response."""
        if stream:
            async for delta in self.chat_stream(messages, params, outcome):
                yield delta
            return
        response = await self.chat(messages, params)
        choices = response.get("choices") if isinstance(response, dict) else None
        if not choices or not isinstance(choices[0], dict):
            raise VeniceAIError("Received invalid response from Venice AI")
        outcome.finish_reason = choices[0].get("finish_reason") or "unknown"
        message = choices[0].get("message") or {}
        if reasoning := message.get("reasoning_content"):
            yield {"reasoning": reasoning}
        if content := message.get("content"):
            yield {"content": content}
        if tool_calls := message.get("tool_calls"):
            yield {"tool_calls": tool_calls}

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


def strip_thinking(text: str, tags: tuple[str, ...]) -> str:
    """Remove reasoning wrapped in ``tags`` from model output.

    Also drops a leading reasoning section that ends with the literal
    ' end of thinking' marker emitted by some Venice AI models.
    """
    if not text or not tags:
        return text
    thinking_filter = ThinkingFilter(tags)
    text = thinking_filter.feed(text)[0] + thinking_filter.flush()[0]
    if text.lstrip().lower().startswith("thinking"):
        _head, marker, tail = text.partition(" end of thinking")
        if marker:
            text = tail
    return text.strip()
