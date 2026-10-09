"""Tests for the chat completion helpers."""

from __future__ import annotations

from typing import Any

import pytest

from custom_components.venice_ai import venice_api
from custom_components.venice_ai.const import (
    CONF_DISABLE_THINKING,
    CONF_THINKING_TAGS,
    CONF_VENICE_SYSTEM_PROMPT,
)
from custom_components.venice_ai.venice_api import (
    ChatParameters,
    StreamOutcome,
    ThinkingFilter,
    VeniceConversationService,
)

from .conftest import FakeChunk

THINK = ("think",)


def _feed_all(parts: list[str], tags: tuple[str, ...] = THINK) -> tuple[str, str]:
    thinking_filter = ThinkingFilter(tags)
    results = [thinking_filter.feed(part) for part in parts]
    results.append(thinking_filter.flush())
    return "".join(a for a, _ in results), "".join(t for _, t in results)


@pytest.mark.parametrize(
    ("parts", "expected"),
    [
        (["Hello"], ("Hello", "")),
        (["<think>hm</think>Hi"], ("Hi", "hm")),
        (["<th", "ink>a", "b</thi", "nk>", "ok"], ("ok", "ab")),
        (["<THINK>x</THINK>y"], ("y", "x")),
        (["a <", "b"], ("a <b", "")),
        (["<think>never closed"], ("", "never closed")),
        (["x<thi"], ("x<thi", "")),
    ],
)
def test_thinking_filter(parts: list[str], expected: tuple[str, str]) -> None:
    assert _feed_all(parts) == expected


def test_thinking_filter_with_custom_tags() -> None:
    tags = ("thought", "reasoning")
    assert _feed_all(["<thought>a</thought>b<reason", "ing>c</reasoning>d"], tags) == (
        "bd",
        "ac",
    )
    assert _feed_all(["<think>kept</think>"], tags) == ("<think>kept</think>", "")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("think", ("think",)),
        ("<think>, Thought , ,think", ("think", "thought")),
        ("", ()),
    ],
)
def test_thinking_tags(value: str, expected: tuple[str, ...]) -> None:
    assert venice_api.thinking_tags({CONF_THINKING_TAGS: value}) == expected


def test_chat_parameters_venice_system_prompt() -> None:
    assert venice_api.chat_parameters({}, "m").venice_parameters == {
        "include_venice_system_prompt": False,
        "disable_thinking": True,
    }
    params = venice_api.chat_parameters(
        {CONF_VENICE_SYSTEM_PROMPT: True, CONF_DISABLE_THINKING: False}, "m"
    )
    assert params.venice_parameters == {"include_venice_system_prompt": True}


def test_thinking_filter_holds_back_partial_tag() -> None:
    thinking_filter = ThinkingFilter(THINK)
    assert thinking_filter.feed("Hi <thi") == ("Hi ", "")
    assert thinking_filter.feed("nk>secret") == ("", "secret")


class TestToolCallFragmentMerge:
    """Tests for incremental tool-call reassembly during streaming."""

    def test_fragments_merge_by_index(self) -> None:
        acc: dict[int, dict] = {}
        venice_api._merge_tool_call_fragment(
            acc,
            {
                "index": 0,
                "id": "call_1",
                "function": {"name": "get_", "arguments": '{"a"'},
            },
        )
        venice_api._merge_tool_call_fragment(
            acc,
            {"index": 0, "function": {"name": "weather", "arguments": ":1}"}},
        )
        assert acc[0]["id"] == "call_1"
        assert acc[0]["function"]["name"] == "get_weather"
        assert acc[0]["function"]["arguments"] == '{"a":1}'

    def test_non_dict_fragment_ignored(self) -> None:
        acc: dict[int, dict] = {}
        venice_api._merge_tool_call_fragment(acc, "nonsense")  # type: ignore[arg-type]
        assert not acc

    def test_multiple_indices_kept_separate(self) -> None:
        acc: dict[int, dict] = {}
        venice_api._merge_tool_call_fragment(
            acc, {"index": 0, "function": {"name": "a"}}
        )
        venice_api._merge_tool_call_fragment(
            acc, {"index": 1, "function": {"name": "b"}}
        )
        assert acc[0]["function"]["name"] == "a"
        assert acc[1]["function"]["name"] == "b"


class TestVeniceConversationService:
    """End-to-end-ish tests of the service against a mocked client."""

    async def test_chat_forwards_parameters(self, make_client) -> None:
        client = make_client(
            non_streaming_response={"choices": [{"message": {"content": "answer"}}]}
        )
        service = VeniceConversationService(client)
        params = ChatParameters(
            model="venice-llm", max_tokens=128, temperature=0.5, top_p=0.9
        )

        resp = await service.chat([{"role": "user", "content": "hi"}], params)

        assert resp["choices"][0]["message"]["content"] == "answer"
        sent = client.chat.last_non_streaming_kwargs
        assert sent["model"] == "venice-llm"
        assert sent["max_tokens"] == 128
        assert sent["temperature"] == 0.5
        assert sent["top_p"] == 0.9
        assert sent["stream"] is False

    async def test_chat_stream_yields_deltas(self, make_client) -> None:
        chunks = [
            FakeChunk([{"delta": {"reasoning_content": "hm"}}]),
            FakeChunk([{"delta": {"content": "Hello"}}]),
            FakeChunk([{"delta": {"content": " world"}, "finish_reason": "stop"}]),
            FakeChunk([], usage={"total_tokens": 3}),
        ]
        service = VeniceConversationService(make_client(chunks=chunks))
        outcome = StreamOutcome()

        deltas = [
            delta
            async for delta in service.chat_stream(
                [{"role": "user", "content": "hi"}],
                ChatParameters(model="venice-llm"),
                outcome,
            )
        ]

        assert deltas == [
            {"reasoning": "hm"},
            {"content": "Hello"},
            {"content": " world"},
        ]
        assert outcome.finish_reason == "stop"

    async def test_chat_stream_reassembles_tool_calls(self, make_client) -> None:
        chunks = [
            FakeChunk(
                [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {"name": "get_", "arguments": '{"x"'},
                                }
                            ]
                        }
                    }
                ]
            ),
            FakeChunk(
                [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"name": "time", "arguments": ":1}"},
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            ),
        ]
        client = make_client(chunks=chunks)
        service = VeniceConversationService(client)
        outcome = StreamOutcome()

        deltas: list[dict[str, Any]] = [
            delta
            async for delta in service.chat_stream(
                [{"role": "user", "content": "time?"}],
                ChatParameters(model="m", tools=[{"type": "function"}]),
                outcome,
            )
        ]

        assert deltas == [
            {
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_time", "arguments": '{"x":1}'},
                    }
                ]
            }
        ]
        assert outcome.finish_reason == "tool_calls"
        sent = client.chat.last_create_kwargs
        assert sent["stream_options"] == {"include_usage": True}
        assert sent["tools"] == [{"type": "function"}]


async def test_chat_deltas_without_streaming(make_client: Any) -> None:
    client = make_client(
        non_streaming_response={
            "choices": [
                {
                    "message": {
                        "reasoning_content": "hm",
                        "content": "Hi",
                        "tool_calls": [{"id": "1"}],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )
    outcome = StreamOutcome()
    deltas = [
        d
        async for d in VeniceConversationService(client).chat_deltas(
            [], ChatParameters(model="m"), outcome, stream=False
        )
    ]
    assert deltas == [
        {"reasoning": "hm"},
        {"content": "Hi"},
        {"tool_calls": [{"id": "1"}]},
    ]
    assert outcome.finish_reason == "tool_calls"


async def test_chat_deltas_invalid_response(make_client: Any) -> None:
    client = make_client(non_streaming_response={"choices": []})
    with pytest.raises(venice_api.VeniceAIError, match="invalid response"):
        _ = [
            d
            async for d in VeniceConversationService(client).chat_deltas(
                [], ChatParameters(model="m"), StreamOutcome(), stream=False
            )
        ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<think>x</think>Answer", "Answer"),
        ("<think>never closed", ""),
        ("Thinking about it end of thinking Answer", "Answer"),
        ("I like the end of thinking part", "I like the end of thinking part"),
    ],
)
def test_strip_thinking(text: str, expected: str) -> None:
    assert venice_api.strip_thinking(text, THINK) == expected


def test_strip_thinking_without_tags() -> None:
    assert venice_api.strip_thinking(" <think>x</think> ", ()) == " <think>x</think> "
