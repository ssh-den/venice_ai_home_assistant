"""Conversation support for Venice AI."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
import json
import logging
import time
from typing import Any, Literal

from homeassistant.components import conversation
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, intent, llm
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from voluptuous_openapi import convert

from . import VeniceAIConfigEntry
from .client import RateLimitError, VeniceAIError
from .const import (
    CONF_CHAT_MODEL,
    CONF_DISABLE_THINKING,
    CONF_MAX_TOKENS,
    CONF_MAX_TOOL_ITERATIONS,
    CONF_PROMPT,
    CONF_STREAM_RESPONSE,
    CONF_STRIP_THINKING_RESPONSE,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DEFAULT_SYSTEM_PROMPT,
    DOMAIN,
    MAX_API_MESSAGES,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_DISABLE_THINKING,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_MAX_TOOL_ITERATIONS,
    RECOMMENDED_STREAM_RESPONSE,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
)
from .models import get_chat_model_info
from .venice_api import (
    ChatParameters,
    StreamOutcome,
    ThinkingFilter,
    VeniceConversationService,
    strip_thinking,
)

_LOGGER = logging.getLogger(__name__)

_TRUNCATED_MESSAGE = (
    "I'm sorry, my response was cut off because it exceeded the maximum token "
    "limit. Please try a shorter question or increase the Max Tokens setting "
    "in the Venice AI integration options."
)
_EMPTY_MESSAGE = "I didn't receive a response from the model. Please try again."


async def async_setup_entry(
    hass: HomeAssistant,
    entry: VeniceAIConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the Venice AI conversation entity."""
    async_add_entities([VeniceAIConversationEntity(entry)])


def _format_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> dict[str, Any]:
    """Format a Home Assistant LLM tool as a Venice AI function tool."""
    function: dict[str, Any] = {
        "name": tool.name,
        "parameters": convert(tool.parameters, custom_serializer=custom_serializer),
    }
    if tool.description:
        function["description"] = tool.description
    return {"type": "function", "function": function}


def _convert_content(
    content: conversation.Content, strip_thinking_output: bool
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
        text = content.content or ""
        if strip_thinking_output:
            text = strip_thinking(text)
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


def _convert_chat_log(
    chat_log: conversation.ChatLog, strip_thinking_output: bool
) -> list[dict[str, Any]]:
    """Convert the Home Assistant chat log into Venice AI chat messages."""
    messages = []
    for content in chat_log.content:
        if (message := _convert_content(content, strip_thinking_output)) is not None:
            messages.append(message)
    return messages


def _trim_api_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Limit the API payload to the leading system messages plus the most
    recent MAX_API_MESSAGES conversation messages.

    The tail never starts with a tool result, since the API rejects tool
    messages without their preceding assistant tool call.
    """
    system = []
    for message in messages:
        if message.get("role") != "system":
            break
        system.append(message)
    rest = messages[len(system) :]
    if len(rest) <= MAX_API_MESSAGES:
        return messages

    tail = rest[-MAX_API_MESSAGES:]
    while tail and tail[0].get("role") == "tool":
        tail.pop(0)
    _LOGGER.debug(
        "Trimmed API messages from %d to %d", len(messages), len(system) + len(tail)
    )
    return system + tail


def _parse_tool_calls(
    raw_tool_calls: list[dict[str, Any]],
) -> tuple[list[llm.ToolInput], list[conversation.ToolResultContent]]:
    """Split API tool calls into executable inputs and rejected calls.

    Rejected calls are marked external so Home Assistant does not run them,
    and get an error result so the model can recover.
    """
    tool_inputs: list[llm.ToolInput] = []
    rejected: list[conversation.ToolResultContent] = []
    for raw in raw_tool_calls:
        function = raw.get("function") or {}
        call_id = raw.get("id")
        name = function.get("name")
        if not call_id or not name or raw.get("type", "function") != "function":
            _LOGGER.warning("Skipping malformed tool call: %s", raw)
            continue
        try:
            args = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = None
        if isinstance(args, dict):
            tool_inputs.append(
                llm.ToolInput(id=call_id, tool_name=name, tool_args=args)
            )
            continue
        _LOGGER.warning("Tool %s called with invalid arguments: %s", name, function)
        tool_inputs.append(
            llm.ToolInput(id=call_id, tool_name=name, tool_args={}, external=True)
        )
        rejected.append(
            conversation.ToolResultContent(
                agent_id=DOMAIN,
                tool_call_id=call_id,
                tool_name=name,
                tool_result={"error": "Tool arguments must be a JSON object"},
            )
        )
    return tool_inputs, rejected


def _final_text(raw_text: str, finish_reason: str, strip_thinking_output: bool) -> str:
    """Return the text to show for a final answer, with useful fallbacks."""
    text = strip_thinking(raw_text) if strip_thinking_output else raw_text
    if text.strip():
        return text
    if finish_reason == "length":
        return _TRUNCATED_MESSAGE
    if strip_thinking_output and raw_text.strip():
        _LOGGER.warning(
            "Stripping thinking removed the whole answer; returning raw output"
        )
        return raw_text
    return _EMPTY_MESSAGE


def _filtered_deltas(
    answer: str, thinking: str
) -> list[conversation.AssistantContentDeltaDict]:
    deltas: list[conversation.AssistantContentDeltaDict] = []
    if thinking:
        deltas.append({"thinking_content": thinking})
    if answer:
        deltas.append({"content": answer})
    return deltas


def _add_stream_fallback(
    chat_log: conversation.ChatLog, agent_id: str, finish_reason: str
) -> None:
    """Add a fallback answer when a streamed reply had no visible text."""
    last = chat_log.content[-1]
    thinking = ""
    if isinstance(last, conversation.AssistantContent):
        if (last.content or "").strip():
            return
        thinking = last.thinking_content or ""
    if thinking.strip() and finish_reason != "length":
        _LOGGER.warning("Model replied with reasoning only; returning it as answer")
    chat_log.async_add_assistant_content_without_tools(
        conversation.AssistantContent(
            agent_id=agent_id, content=_final_text(thinking, finish_reason, False)
        )
    )


class VeniceAIConversationEntity(conversation.ConversationEntity):
    """Venice AI conversation entity."""

    def __init__(self, entry: VeniceAIConfigEntry) -> None:
        """Initialize the entity."""
        self.entry = entry
        self._service = VeniceConversationService(entry.runtime_data.client)
        self._attr_unique_id = f"{entry.entry_id}_conversation"
        self._attr_name = entry.title
        if entry.options.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="Venice AI",
            model="Conversation",
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    @property
    def supports_streaming(self) -> bool:
        """Return whether answers are streamed to Home Assistant."""
        return bool(
            self.entry.options.get(CONF_STREAM_RESPONSE, RECOMMENDED_STREAM_RESPONSE)
        )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Process a conversation turn."""
        options = self.entry.options

        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                options.get(CONF_LLM_HASS_API),
                options.get(CONF_PROMPT, DEFAULT_SYSTEM_PROMPT),
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        try:
            await self._async_run_turn(user_input, chat_log)
        except RateLimitError as err:
            _LOGGER.warning("Rate limit hit during conversation: %s", err)
            return self._error_result(
                user_input,
                chat_log,
                "Venice AI rate limit exceeded. Please wait a moment and try again.",
            )
        except (VeniceAIError, HomeAssistantError) as err:
            _LOGGER.error("Error during conversation processing: %s", err)
            return self._error_result(user_input, chat_log, f"Venice AI error: {err}")

        return conversation.async_get_result_from_chat_log(user_input, chat_log)

    async def _async_run_turn(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> None:
        """Call the model until it answers without tool calls."""
        options = self.entry.options
        strip = bool(options.get(CONF_STRIP_THINKING_RESPONSE, False))
        stream = self.supports_streaming
        max_iterations = int(
            options.get(CONF_MAX_TOOL_ITERATIONS, RECOMMENDED_MAX_TOOL_ITERATIONS)
        )

        model = options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
        info = get_chat_model_info(self.entry, model)
        tools: list[dict[str, Any]] | None = None
        if chat_log.llm_api and info is not None and not info.supports_function_calling:
            _LOGGER.debug("Model %s does not support tools, not sending them", model)
        elif chat_log.llm_api:
            tools = [
                _format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ] or None

        params = ChatParameters(
            model=model,
            max_tokens=options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
            temperature=options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
            top_p=options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
            tools=tools,
            venice_parameters=(
                {"disable_thinking": True}
                if options.get(CONF_DISABLE_THINKING, RECOMMENDED_DISABLE_THINKING)
                else None
            ),
        )

        for iteration in range(max_iterations):
            messages = _trim_api_messages(_convert_chat_log(chat_log, strip))
            started = time.monotonic()
            if stream:
                outcome = StreamOutcome()
                async for _content in chat_log.async_add_delta_content_stream(
                    user_input.agent_id,
                    self._async_delta_stream(
                        chat_log, messages, params, outcome, strip
                    ),
                ):
                    pass
                _LOGGER.debug(
                    "Venice AI stream %d took %.3fs (messages=%d, tools=%d)",
                    iteration + 1,
                    time.monotonic() - started,
                    len(messages),
                    len(tools or []),
                )
                last = chat_log.content[-1]
                if isinstance(last, conversation.ToolResultContent):
                    continue
                _add_stream_fallback(
                    chat_log, user_input.agent_id, outcome.finish_reason
                )
                return

            response = await self._service.chat(messages, params)
            _LOGGER.debug(
                "Venice AI call %d took %.3fs (messages=%d, tools=%d)",
                iteration + 1,
                time.monotonic() - started,
                len(messages),
                len(tools or []),
            )

            choices = response.get("choices") if isinstance(response, dict) else None
            if not choices or not isinstance(choices[0], dict):
                raise HomeAssistantError("Received invalid response from Venice AI")
            choice = choices[0]
            message = choice.get("message") or {}
            finish_reason = choice.get("finish_reason") or "unknown"
            raw_text = message.get("content") or ""
            if finish_reason == "length":
                _LOGGER.warning(
                    "Venice AI response truncated by max_tokens (%s)", params.max_tokens
                )

            raw_tool_calls = message.get("tool_calls") or []
            if not raw_tool_calls:
                chat_log.async_add_assistant_content_without_tools(
                    conversation.AssistantContent(
                        agent_id=user_input.agent_id,
                        content=_final_text(raw_text, finish_reason, strip),
                    )
                )
                return

            tool_inputs, rejected = _parse_tool_calls(raw_tool_calls)
            assistant = conversation.AssistantContent(
                agent_id=user_input.agent_id,
                content=strip_thinking(raw_text) if strip else raw_text,
                tool_calls=tool_inputs or None,
            )
            if chat_log.llm_api is None:
                chat_log.async_add_assistant_content_without_tools(assistant)
                return
            async for _tool_result in chat_log.async_add_assistant_content(assistant):
                pass
            for result in rejected:
                async for _tool_result in chat_log.async_add_assistant_content(result):
                    pass

        _LOGGER.warning("Reached max tool iterations (%d)", max_iterations)
        last = chat_log.content[-1]
        if not isinstance(last, conversation.AssistantContent):
            chat_log.async_add_assistant_content_without_tools(
                conversation.AssistantContent(
                    agent_id=user_input.agent_id, content=_EMPTY_MESSAGE
                )
            )

    async def _async_delta_stream(
        self,
        chat_log: conversation.ChatLog,
        messages: list[dict[str, Any]],
        params: ChatParameters,
        outcome: StreamOutcome,
        strip: bool,
    ) -> AsyncGenerator[
        conversation.AssistantContentDeltaDict | conversation.ToolResultContentDeltaDict
    ]:
        """Translate Venice AI stream deltas into chat log deltas."""
        yield {"role": "assistant"}
        thinking_filter = ThinkingFilter() if strip else None
        raw_tool_calls: list[dict[str, Any]] = []
        async for delta in self._service.chat_stream(messages, params, outcome):
            if reasoning := delta.get("reasoning"):
                yield {"thinking_content": reasoning}
            if content := delta.get("content"):
                if thinking_filter is None:
                    yield {"content": content}
                else:
                    for item in _filtered_deltas(*thinking_filter.feed(content)):
                        yield item
            raw_tool_calls.extend(delta.get("tool_calls") or [])
        if thinking_filter is not None:
            for item in _filtered_deltas(*thinking_filter.flush()):
                yield item

        if not raw_tool_calls:
            return
        if chat_log.llm_api is None:
            _LOGGER.warning("Ignoring tool calls without an LLM API configured")
            return
        tool_inputs, rejected = _parse_tool_calls(raw_tool_calls)
        if tool_inputs:
            yield {"tool_calls": tool_inputs}
        for result in rejected:
            yield {
                "role": "tool_result",
                "tool_call_id": result.tool_call_id,
                "tool_name": result.tool_name,
                "tool_result": result.tool_result,
            }

    def _error_result(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
        message: str,
    ) -> conversation.ConversationResult:
        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, message)
        return conversation.ConversationResult(
            conversation_id=chat_log.conversation_id,
            response=intent_response,
            continue_conversation=chat_log.continue_conversation,
        )
