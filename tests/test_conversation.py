"""Tests for the Venice AI conversation entity."""

# pylint: disable=redefined-outer-name

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from homeassistant.components import conversation
from homeassistant.components.conversation.const import DATA_COMPONENT
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import intent, llm
from homeassistant.util.json import JsonObjectType
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry
import voluptuous as vol

from custom_components.venice_ai.client import RateLimitError
from custom_components.venice_ai.const import (
    CONF_CHAT_MODEL,
    CONF_STREAM_RESPONSE,
    CONF_STRIP_THINKING_RESPONSE,
    MAX_API_MESSAGES,
)
from custom_components.venice_ai.conversation import _final_text, _trim_api_messages

from .conftest import SCHEMA_MODEL, FakeChunk, FakeStream

TEST_API_ID = "venice_test_api"


class _EchoTool(llm.Tool):
    name = "echo"
    description = "Echo the given text"
    parameters = vol.Schema({vol.Required("text"): str})

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> JsonObjectType:
        self.calls.append(tool_input.tool_args)
        return {"echo": tool_input.tool_args["text"]}


class _TestAPI(llm.API):
    def __init__(self, hass: HomeAssistant, tool: _EchoTool) -> None:
        super().__init__(hass=hass, id=TEST_API_ID, name="Test API")
        self._tool = tool

    async def async_get_api_instance(
        self, llm_context: llm.LLMContext
    ) -> llm.APIInstance:
        return llm.APIInstance(
            api=self,
            api_prompt="Use the echo tool when asked.",
            llm_context=llm_context,
            tools=[self._tool],
        )


def _reply(content: str | None = None, **message: Any) -> dict[str, Any]:
    return {
        "choices": [
            {"message": {"content": content, **message}, "finish_reason": "stop"}
        ]
    }


def _tool_reply(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    return _reply(
        "",
        tool_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    )


@pytest.fixture
def options() -> dict[str, Any]:
    return {CONF_STREAM_RESPONSE: False}


@pytest.fixture
def mock_config_entry(hass: HomeAssistant, options: dict[str, Any]) -> MockConfigEntry:
    entry = MockConfigEntry(
        domain="venice_ai", title="Venice AI", data={"api_key": "k"}, options=options
    )
    entry.add_to_hass(hass)
    return entry


@pytest.fixture
def echo_tool(hass: HomeAssistant) -> _EchoTool:
    tool = _EchoTool()
    llm.async_register_api(hass, _TestAPI(hass, tool))
    return tool


async def _converse(
    hass: HomeAssistant, text: str, conversation_id: str | None = None
) -> conversation.ConversationResult:
    return await conversation.async_converse(
        hass,
        text,
        conversation_id,
        Context(),
        agent_id="conversation.venice_ai",
    )


def _sent_messages(mock_client: MagicMock, call: int = -1) -> list[dict[str, Any]]:
    calls = mock_client.chat.create_non_streaming.call_args_list
    return calls[call].kwargs["messages"]


async def test_plain_answer(
    hass: HomeAssistant, setup_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    mock_client.chat.create_non_streaming.return_value = _reply("Hello!")
    result = await _converse(hass, "Hi there")

    assert result.response.response_type is intent.IntentResponseType.ACTION_DONE
    assert result.response.speech["plain"]["speech"] == "Hello!"

    messages = _sent_messages(mock_client)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[1]["content"] == "Hi there"


async def test_history_is_kept_by_chat_log(
    hass: HomeAssistant, setup_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    mock_client.chat.create_non_streaming.return_value = _reply("First")
    result = await _converse(hass, "One")
    mock_client.chat.create_non_streaming.return_value = _reply("Second")
    await _converse(hass, "Two", result.conversation_id)

    messages = _sent_messages(mock_client)
    assert [(m["role"], m["content"]) for m in messages[1:]] == [
        ("user", "One"),
        ("assistant", "First"),
        ("user", "Two"),
    ]


@pytest.mark.parametrize(
    "options", [{CONF_STREAM_RESPONSE: False, CONF_LLM_HASS_API: [TEST_API_ID]}]
)
async def test_tool_call_round_trip(
    hass: HomeAssistant,
    echo_tool: _EchoTool,
    setup_integration: MockConfigEntry,
    mock_client: MagicMock,
) -> None:
    mock_client.chat.create_non_streaming.side_effect = [
        _tool_reply("call_1", "echo", '{"text": "ping"}'),
        _reply("Echoed ping"),
    ]
    result = await _converse(hass, "Echo ping")

    assert echo_tool.calls == [{"text": "ping"}]
    assert result.response.speech["plain"]["speech"] == "Echoed ping"

    first = mock_client.chat.create_non_streaming.call_args_list[0].kwargs
    assert first["tools"][0]["function"]["name"] == "echo"
    messages = _sent_messages(mock_client)
    assert messages[-2]["role"] == "assistant"
    assert messages[-2]["tool_calls"][0]["function"] == {
        "name": "echo",
        "arguments": '{"text": "ping"}',
    }
    assert messages[-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": '{"echo": "ping"}',
    }


@pytest.mark.parametrize(
    "options", [{CONF_STREAM_RESPONSE: False, CONF_LLM_HASS_API: [TEST_API_ID]}]
)
async def test_invalid_tool_arguments_are_reported(
    hass: HomeAssistant,
    echo_tool: _EchoTool,
    setup_integration: MockConfigEntry,
    mock_client: MagicMock,
) -> None:
    mock_client.chat.create_non_streaming.side_effect = [
        _tool_reply("call_1", "echo", '["not", "an", "object"]'),
        _reply("Sorry"),
    ]
    result = await _converse(hass, "Echo")

    assert echo_tool.calls == []
    assert result.response.speech["plain"]["speech"] == "Sorry"
    tool_message = _sent_messages(mock_client)[-1]
    assert tool_message["role"] == "tool"
    assert "JSON object" in tool_message["content"]


async def test_rate_limit_returns_error(
    hass: HomeAssistant, setup_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    mock_client.chat.create_non_streaming.side_effect = RateLimitError("slow down")
    result = await _converse(hass, "Hi")
    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert "rate limit" in result.response.speech["plain"]["speech"]


@pytest.mark.parametrize("options", [{}])
async def test_streaming_answer(
    hass: HomeAssistant, setup_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    chunks = [
        FakeChunk([{"delta": {"content": "<think>hmm</think>"}}]),
        FakeChunk([{"delta": {"content": "Streamed"}, "finish_reason": "stop"}]),
    ]

    @asynccontextmanager
    async def _create(**kwargs: Any):
        yield FakeStream(chunks)

    mock_client.chat.create = MagicMock(side_effect=_create)
    mock_client.chat.create_non_streaming = AsyncMock()
    result = await _converse(hass, "Hi")

    assert result.response.speech["plain"]["speech"] == "Streamed"
    mock_client.chat.create_non_streaming.assert_not_awaited()


def test_trim_keeps_system_and_drops_orphan_tool_results() -> None:
    system = [{"role": "system", "content": "s"}]
    rest: list[dict[str, Any]] = [
        {"role": "user", "content": str(i)} for i in range(MAX_API_MESSAGES)
    ]
    rest.insert(1, {"role": "tool", "content": "orphan"})
    trimmed = _trim_api_messages(system + rest)
    assert trimmed[0] == system[0]
    assert trimmed[1]["role"] != "tool"
    assert len(trimmed) <= MAX_API_MESSAGES + 1


def test_trim_noop_for_short_conversations() -> None:
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    assert _trim_api_messages(messages) == messages


@pytest.mark.parametrize(
    ("raw", "finish_reason", "strip", "expected"),
    [
        ("Hello", "stop", False, "Hello"),
        ("<think>x</think>Hello", "stop", True, "Hello"),
        ("<think>only</think>", "stop", True, "<think>only</think>"),
        ("", "length", False, "cut off"),
        ("", "stop", False, "didn't receive"),
    ],
)
def test_final_text(raw: str, finish_reason: str, strip: bool, expected: str) -> None:
    assert expected in _final_text(raw, finish_reason, strip)


@pytest.mark.parametrize(
    "options",
    [
        {
            CONF_STREAM_RESPONSE: False,
            CONF_LLM_HASS_API: [TEST_API_ID],
            CONF_CHAT_MODEL: SCHEMA_MODEL,
        }
    ],
)
async def test_tools_not_sent_to_model_without_function_calling(
    hass: HomeAssistant,
    echo_tool: _EchoTool,
    setup_integration: MockConfigEntry,
    mock_client: MagicMock,
) -> None:
    mock_client.chat.create_non_streaming.return_value = _reply("No tools")
    result = await _converse(hass, "Echo ping")

    assert result.response.speech["plain"]["speech"] == "No tools"
    assert mock_client.chat.create_non_streaming.call_args.kwargs["tools"] is None


@pytest.mark.parametrize(
    "options",
    [
        {
            CONF_STREAM_RESPONSE: False,
            CONF_LLM_HASS_API: [TEST_API_ID],
            CONF_CHAT_MODEL: "unlisted-model",
        }
    ],
)
async def test_tools_sent_to_unknown_model(
    hass: HomeAssistant,
    echo_tool: _EchoTool,
    setup_integration: MockConfigEntry,
    mock_client: MagicMock,
) -> None:
    mock_client.chat.create_non_streaming.return_value = _reply("Hi")
    await _converse(hass, "Hi")

    tools = mock_client.chat.create_non_streaming.call_args.kwargs["tools"]
    assert tools[0]["function"]["name"] == "echo"


def _stream_from(*batches: list[FakeChunk]) -> MagicMock:
    responses = list(batches)

    @asynccontextmanager
    async def _create(**kwargs: Any):
        yield FakeStream(responses.pop(0))

    return MagicMock(side_effect=_create)


def _tool_chunk(call_id: str, arguments: str) -> FakeChunk:
    return FakeChunk(
        [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": call_id,
                            "type": "function",
                            "function": {"name": "echo", "arguments": arguments},
                        }
                    ]
                },
                "finish_reason": "tool_calls",
            }
        ]
    )


@pytest.mark.parametrize(
    "options", [{CONF_STREAM_RESPONSE: True, CONF_LLM_HASS_API: [TEST_API_ID]}]
)
async def test_streaming_tool_call_round_trip(
    hass: HomeAssistant,
    echo_tool: _EchoTool,
    setup_integration: MockConfigEntry,
    mock_client: MagicMock,
) -> None:
    mock_client.chat.create = _stream_from(
        [_tool_chunk("call_1", '{"text": "ping"}')],
        [FakeChunk([{"delta": {"content": "Echoed"}, "finish_reason": "stop"}])],
    )
    result = await _converse(hass, "Echo ping")

    assert echo_tool.calls == [{"text": "ping"}]
    assert result.response.speech["plain"]["speech"] == "Echoed"
    second = mock_client.chat.create.call_args_list[1].kwargs["messages"]
    assert second[-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": '{"echo": "ping"}',
    }


@pytest.mark.parametrize(
    "options", [{CONF_STREAM_RESPONSE: True, CONF_LLM_HASS_API: [TEST_API_ID]}]
)
async def test_streaming_invalid_tool_arguments(
    hass: HomeAssistant,
    echo_tool: _EchoTool,
    setup_integration: MockConfigEntry,
    mock_client: MagicMock,
) -> None:
    mock_client.chat.create = _stream_from(
        [_tool_chunk("call_1", "[1]")],
        [FakeChunk([{"delta": {"content": "Sorry"}, "finish_reason": "stop"}])],
    )
    result = await _converse(hass, "Echo")

    assert echo_tool.calls == []
    assert result.response.speech["plain"]["speech"] == "Sorry"
    second = mock_client.chat.create.call_args_list[1].kwargs["messages"]
    assert second[-1]["role"] == "tool"
    assert "JSON object" in second[-1]["content"]


@pytest.mark.parametrize(
    ("options", "chunks", "expected"),
    [
        (
            {CONF_STREAM_RESPONSE: True},
            [FakeChunk([{"delta": {}, "finish_reason": "length"}])],
            "cut off",
        ),
        (
            {CONF_STREAM_RESPONSE: True},
            [FakeChunk([{"delta": {}, "finish_reason": "stop"}])],
            "didn't receive",
        ),
        (
            {CONF_STREAM_RESPONSE: True, CONF_STRIP_THINKING_RESPONSE: True},
            [FakeChunk([{"delta": {"content": "<think>only this</think>"}}])],
            "only this",
        ),
    ],
)
async def test_streaming_fallbacks(
    hass: HomeAssistant,
    setup_integration: MockConfigEntry,
    mock_client: MagicMock,
    chunks: list[FakeChunk],
    expected: str,
) -> None:
    mock_client.chat.create = _stream_from(chunks)
    result = await _converse(hass, "Hi")
    assert expected in result.response.speech["plain"]["speech"]


@pytest.mark.parametrize("options", [{CONF_STREAM_RESPONSE: True}])
async def test_streaming_reasoning_is_not_spoken(
    hass: HomeAssistant, setup_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    mock_client.chat.create = _stream_from(
        [
            FakeChunk([{"delta": {"reasoning_content": "thinking..."}}]),
            FakeChunk([{"delta": {"content": "Answer"}, "finish_reason": "stop"}]),
        ]
    )
    result = await _converse(hass, "Hi")
    assert result.response.speech["plain"]["speech"] == "Answer"


@pytest.mark.parametrize("options", [{CONF_STREAM_RESPONSE: True}])
async def test_supports_streaming_follows_option(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    entity = hass.data[DATA_COMPONENT].get_entity("conversation.venice_ai")
    assert entity is not None
    assert entity.supports_streaming
    hass.config_entries.async_update_entry(
        setup_integration, options={CONF_STREAM_RESPONSE: False}
    )
    assert not entity.supports_streaming
