"""AI Task integration for Venice AI."""

from __future__ import annotations

import json
import logging
from typing import Any

from homeassistant.components import ai_task, conversation
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, llm
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from voluptuous_openapi import convert

from . import VeniceAIConfigEntry
from .client import VeniceAIError
from .const import (
    CONF_CHAT_MODEL,
    CONF_DISABLE_THINKING,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
    DOMAIN,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_DISABLE_THINKING,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
)
from .models import get_chat_model_info
from .venice_api import (
    ChatParameters,
    VeniceConversationService,
    extract_json,
    strip_thinking,
)

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: VeniceAIConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up AI Task entities."""
    async_add_entities([VeniceAITaskEntity(entry)])


def _chat_log_to_messages(chat_log: conversation.ChatLog) -> list[dict[str, Any]]:
    """Convert the AI Task chat log into Venice chat messages."""
    messages: list[dict[str, Any]] = []
    for content in chat_log.content:
        if isinstance(content, conversation.SystemContent):
            messages.append({"role": "system", "content": content.content})
        elif isinstance(content, conversation.UserContent):
            messages.append({"role": "user", "content": content.content})
        elif isinstance(content, conversation.AssistantContent):
            messages.append({"role": "assistant", "content": content.content or ""})
    return messages


class VeniceAITaskEntity(ai_task.AITaskEntity):
    """Venice AI AI Task entity."""

    _attr_has_entity_name = True
    _attr_name = "AI Task"
    _attr_supported_features = ai_task.AITaskEntityFeature.GENERATE_DATA

    def __init__(self, entry: VeniceAIConfigEntry) -> None:
        """Initialize the entity."""
        self.entry = entry
        self._attr_unique_id = f"{entry.entry_id}_task"
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="Venice AI",
            model="AI Task",
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        self._service = VeniceConversationService(entry.runtime_data.client)

    async def _async_generate_data(
        self,
        task: ai_task.GenDataTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenDataTaskResult:
        """Handle a generate data task."""
        messages = _chat_log_to_messages(chat_log)
        if not messages or messages[-1]["role"] != "user":
            raise HomeAssistantError("No user message found in chat log")

        options = self.entry.options
        model = options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
        info = get_chat_model_info(self.entry, model)
        response_format: dict[str, Any] | None = None
        schema = (
            convert(
                task.structure,
                custom_serializer=(
                    chat_log.llm_api.custom_serializer
                    if chat_log.llm_api
                    else llm.selector_serializer
                ),
            )
            if task.structure
            else None
        )
        if schema and info is not None and info.supports_response_schema:
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": schema},
            }
        elif schema:
            messages.insert(
                len(messages) - 1,
                {
                    "role": "system",
                    "content": (
                        "Respond only with a JSON object matching this JSON "
                        "schema, without any surrounding text:\n"
                        f"{json.dumps(schema)}"
                    ),
                },
            )

        venice_params = (
            {"disable_thinking": True}
            if options.get(CONF_DISABLE_THINKING, RECOMMENDED_DISABLE_THINKING)
            else None
        )
        params = ChatParameters(
            model=model,
            max_tokens=options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
            temperature=options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
            venice_parameters=venice_params,
            response_format=response_format,
        )

        try:
            response = await self._service.chat(messages, params)
        except VeniceAIError as err:
            raise HomeAssistantError(f"Error generating data: {err}") from err

        choices = response.get("choices") if isinstance(response, dict) else None
        if not choices:
            raise HomeAssistantError("Invalid Venice AI response")
        text = strip_thinking(choices[0].get("message", {}).get("content") or "")

        chat_log.async_add_assistant_content_without_tools(
            conversation.AssistantContent(agent_id=self.entity_id, content=text)
        )

        if not task.structure:
            return ai_task.GenDataTaskResult(
                conversation_id=chat_log.conversation_id, data=text
            )

        try:
            data = extract_json(text)
        except json.JSONDecodeError as err:
            _LOGGER.error("Failed to parse JSON response: %s. Response: %s", err, text)
            raise HomeAssistantError("Error parsing structured response") from err

        return ai_task.GenDataTaskResult(
            conversation_id=chat_log.conversation_id, data=data
        )
