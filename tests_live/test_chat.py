"""Conversation and AI Task against the cheapest Venice chat model with tools."""

from __future__ import annotations

from homeassistant.components import ai_task, conversation
from homeassistant.components.homeassistant.exposed_entities import (
    async_expose_entity,
)
from homeassistant.const import CONF_LLM_HASS_API, STATE_ON
from homeassistant.core import Context, HomeAssistant
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry
import voluptuous as vol

from custom_components.venice_ai.const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_STREAM_RESPONSE,
)
from custom_components.venice_ai.models import parse_models

AGENT_ID = "conversation.venice_ai"


def _use_cheapest_model(
    hass: HomeAssistant, entry: MockConfigEntry, **options: object
) -> str:
    models = parse_models(entry.runtime_data.coordinator.data["text_models"])
    model = min(
        (
            m
            for m in models.values()
            if m.supports_function_calling
            and not m.offline
            and m.input_price is not None
            and m.output_price is not None
        ),
        key=lambda m: (m.input_price or 0) + (m.output_price or 0),
    )
    hass.config_entries.async_update_entry(
        entry,
        options={CONF_CHAT_MODEL: model.id, CONF_MAX_TOKENS: 128, **options},
    )
    return model.id


async def _speech(hass: HomeAssistant, text: str) -> str:
    result = await conversation.async_converse(
        hass, text, None, Context(), agent_id=AGENT_ID
    )
    return str(result.response.speech["plain"]["speech"])


async def test_answer_streamed_and_not(
    hass: HomeAssistant, live_entry: MockConfigEntry
) -> None:
    prompt = "Reply with the single word pong."
    model = _use_cheapest_model(hass, live_entry, **{CONF_STREAM_RESPONSE: True})
    assert "pong" in (await _speech(hass, prompt)).lower(), model

    _use_cheapest_model(hass, live_entry, **{CONF_STREAM_RESPONSE: False})
    assert "pong" in (await _speech(hass, prompt)).lower(), model


async def test_tool_call_controls_an_entity(
    hass: HomeAssistant, live_entry: MockConfigEntry
) -> None:
    assert await async_setup_component(hass, "intent", {})
    assert await async_setup_component(
        hass, "input_boolean", {"input_boolean": {"test_switch": {"name": "Test switch"}}}
    )
    async_expose_entity(hass, conversation.DOMAIN, "input_boolean.test_switch", True)
    model = _use_cheapest_model(hass, live_entry, **{CONF_LLM_HASS_API: ["assist"]})

    await _speech(hass, "Turn on the Test switch.")

    state = hass.states.get("input_boolean.test_switch")
    assert state is not None
    assert state.state == STATE_ON, model


async def test_ai_task_structure(
    hass: HomeAssistant, live_entry: MockConfigEntry
) -> None:
    model = _use_cheapest_model(hass, live_entry)
    result = await ai_task.async_generate_data(
        hass,
        task_name="sum",
        entity_id="ai_task.venice_ai_ai_task",
        instructions="What is 2 + 3? Return the number as answer.",
        structure=vol.Schema({vol.Required("answer"): int}),
    )
    assert result.data == {"answer": 5}, model
