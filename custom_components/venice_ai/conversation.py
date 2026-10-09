"""Conversation support for Venice AI."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
import json
import logging
import time
from typing import Any, Literal

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import intent, llm
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from voluptuous_openapi import convert

from . import VeniceAIConfigEntry
from .client import RateLimitError, VeniceAIError
from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_HISTORY_MESSAGES,
    CONF_MAX_TOOL_ITERATIONS,
    CONF_PROMPT,
    CONF_STREAM_RESPONSE,
    CONF_STRIP_THINKING_RESPONSE,
    DEFAULT_SYSTEM_PROMPT,
    DOMAIN,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_HISTORY_MESSAGES,
    RECOMMENDED_MAX_TOOL_ITERATIONS,
    RECOMMENDED_STREAM_RESPONSE,
    RECOMMENDED_STRIP_THINKING_RESPONSE,
    SUBENTRY_CONVERSATION,
)
from .entity import VeniceAIEntity, subentries_of
from .models import get_chat_model_info
from .venice_api import (
    ChatParameters,
    StreamOutcome,
    ThinkingFilter,
    VeniceConversationService,
    chat_log_to_messages,
    chat_parameters,
    thinking_tags,
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
    """Set up a conversation entity for each conversation subentry."""
    for subentry in subentries_of(entry, SUBENTRY_CONVERSATION):
        async_add_entities(
            [VeniceAIConversationEntity(entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


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


def _trim_api_messages(
    messages: list[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    """Keep the leading system messages and the last ``limit`` messages.

    The current turn, from the last user message on, is always kept. The tail
    never starts with a tool result, since the API rejects tool messages
    without their preceding assistant tool call.
    """
    system = []
    for message in messages:
        if message.get("role") != "system":
            break
        system.append(message)
    rest = messages[len(system) :]
    last_user = max(
        (i for i, message in enumerate(rest) if message.get("role") == "user"),
        default=0,
    )
    start = min(max(len(rest) - limit, 0), last_user)
    while start < last_user and rest[start].get("role") == "tool":
        start += 1
    if not start:
        return messages
    _LOGGER.debug(
        "Trimmed API messages from %d to %d",
        len(messages),
        len(messages) - start,
    )
    return system + rest[start:]


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
            _LOGGER.warning("Skipping malformed call of tool %s", name)
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
        _LOGGER.warning("Tool %s called with arguments that are not an object", name)
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


def _fallback_text(thinking: str, finish_reason: str) -> str:
    """Return the text to show when the model gave no visible answer."""
    if finish_reason == "length":
        return _TRUNCATED_MESSAGE
    if thinking.strip():
        _LOGGER.warning("Model replied with reasoning only; returning it as answer")
        return thinking
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


def _add_fallback(
    chat_log: conversation.ChatLog, agent_id: str, finish_reason: str
) -> None:
    """Add a fallback answer when the reply had no visible text."""
    last = chat_log.content[-1]
    thinking = ""
    if isinstance(last, conversation.AssistantContent):
        if (last.content or "").strip():
            return
        thinking = last.thinking_content or ""
    chat_log.async_add_assistant_content_without_tools(
        conversation.AssistantContent(
            agent_id=agent_id, content=_fallback_text(thinking, finish_reason)
        )
    )


class VeniceAIConversationEntity(conversation.ConversationEntity, VeniceAIEntity):
    """Venice AI conversation entity."""

    def __init__(self, entry: VeniceAIConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the entity."""
        super().__init__(
            entry, subentry, subentry.data.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
        )
        self._service = VeniceConversationService(entry.runtime_data.client)
        if subentry.data.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

    @property
    def supports_streaming(self) -> bool:
        """Return whether answers are streamed to Home Assistant."""
        return bool(self.options.get(CONF_STREAM_RESPONSE, RECOMMENDED_STREAM_RESPONSE))

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
        options = self.options

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
        options = self.options
        strip = bool(
            options.get(
                CONF_STRIP_THINKING_RESPONSE, RECOMMENDED_STRIP_THINKING_RESPONSE
            )
        )
        tags = thinking_tags(options) if strip else ()
        max_iterations = int(
            options.get(CONF_MAX_TOOL_ITERATIONS, RECOMMENDED_MAX_TOOL_ITERATIONS)
        )
        max_history = int(
            options.get(CONF_MAX_HISTORY_MESSAGES, RECOMMENDED_MAX_HISTORY_MESSAGES)
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
        params = chat_parameters(options, model, tools=tools)

        for iteration in range(max_iterations):
            messages = _trim_api_messages(
                chat_log_to_messages(chat_log, tags), max_history
            )
            outcome = StreamOutcome()
            started = time.monotonic()
            async for _content in chat_log.async_add_delta_content_stream(
                user_input.agent_id,
                self._async_deltas(chat_log, messages, params, outcome, tags),
            ):
                pass
            _LOGGER.debug(
                "Venice AI call %d took %.3fs (messages=%d, tools=%d)",
                iteration + 1,
                time.monotonic() - started,
                len(messages),
                len(tools or []),
            )
            if outcome.finish_reason == "length":
                _LOGGER.warning(
                    "Venice AI response truncated by max_tokens (%s)", params.max_tokens
                )
            if isinstance(chat_log.content[-1], conversation.ToolResultContent):
                continue
            _add_fallback(chat_log, user_input.agent_id, outcome.finish_reason)
            return

        _LOGGER.warning("Reached max tool iterations (%d)", max_iterations)
        if not isinstance(chat_log.content[-1], conversation.AssistantContent):
            chat_log.async_add_assistant_content_without_tools(
                conversation.AssistantContent(
                    agent_id=user_input.agent_id, content=_EMPTY_MESSAGE
                )
            )

    async def _async_deltas(
        self,
        chat_log: conversation.ChatLog,
        messages: list[dict[str, Any]],
        params: ChatParameters,
        outcome: StreamOutcome,
        tags: tuple[str, ...],
    ) -> AsyncGenerator[
        conversation.AssistantContentDeltaDict | conversation.ToolResultContentDeltaDict
    ]:
        """Translate Venice AI deltas into chat log deltas."""
        yield {"role": "assistant"}
        thinking_filter = ThinkingFilter(tags) if tags else None
        raw_tool_calls: list[dict[str, Any]] = []
        async for delta in self._service.chat_deltas(
            messages, params, outcome, self.supports_streaming
        ):
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
