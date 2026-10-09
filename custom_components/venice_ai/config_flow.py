"""Config flow for the Venice AI integration."""

from __future__ import annotations

from collections.abc import Mapping
import logging
from typing import TYPE_CHECKING, Any, cast

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigEntryState,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    OptionsFlow,
    SubentryFlowResult,
)
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_NAME
from homeassistant.core import callback
from homeassistant.helpers import config_validation as cv, llm
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.selector import (
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
    TextSelector,
    TextSelectorConfig,
)
from homeassistant.helpers.typing import VolDictType
import voluptuous as vol

from .client import AsyncVeniceAIClient, AuthenticationError, VeniceAIError
from .const import (
    CONF_CHAT_MODEL,
    CONF_DISABLE_THINKING,
    CONF_IMAGE_MODEL,
    CONF_MAX_HISTORY_MESSAGES,
    CONF_MAX_TOKENS,
    CONF_MAX_TOOL_ITERATIONS,
    CONF_PRIVATE_MODELS_ONLY,
    CONF_PROMPT,
    CONF_RECOMMENDED,
    CONF_REQUEST_TIMEOUT,
    CONF_STREAM_RESPONSE,
    CONF_STRIP_THINKING_RESPONSE,
    CONF_STRUCTURE_PROMPT,
    CONF_STT_MODEL,
    CONF_TEMPERATURE,
    CONF_THINKING_TAGS,
    CONF_TOP_P,
    CONF_TTS_MODEL,
    CONF_TTS_SPEED,
    CONF_TTS_VOICE,
    CONF_VENICE_SYSTEM_PROMPT,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_CONVERSATION_NAME,
    DEFAULT_NAME,
    DEFAULT_STRUCTURE_PROMPT,
    DEFAULT_STT_NAME,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TTS_NAME,
    DOMAIN,
    RECOMMENDED_AI_TASK_OPTIONS,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_CONVERSATION_OPTIONS,
    RECOMMENDED_DISABLE_THINKING,
    RECOMMENDED_IMAGE_MODEL,
    RECOMMENDED_MAX_HISTORY_MESSAGES,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_MAX_TOOL_ITERATIONS,
    RECOMMENDED_PRIVATE_MODELS_ONLY,
    RECOMMENDED_REQUEST_TIMEOUT,
    RECOMMENDED_STREAM_RESPONSE,
    RECOMMENDED_STRIP_THINKING_RESPONSE,
    RECOMMENDED_STT_MODEL,
    RECOMMENDED_STT_OPTIONS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_THINKING_TAGS,
    RECOMMENDED_TOP_P,
    RECOMMENDED_TTS_MODEL,
    RECOMMENDED_TTS_OPTIONS,
    RECOMMENDED_TTS_SPEED,
    RECOMMENDED_TTS_VOICE,
    RECOMMENDED_VENICE_SYSTEM_PROMPT,
    SUBENTRY_AI_TASK,
    SUBENTRY_CONVERSATION,
    SUBENTRY_STT,
    SUBENTRY_TTS,
)
from .languages import voice_language
from .models import (
    TTSModel,
    is_private,
    parse_models,
    parse_tts_models,
    privacy_label,
    speech_price_label,
)

if TYPE_CHECKING:
    from . import VeniceAIConfigEntry

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema({vol.Required(CONF_API_KEY): cv.string})

# Coordinator model list of each /models type
_COORDINATOR_KEYS = {
    "text": "text_models",
    "tts": "tts_models",
    "asr": "asr_models",
    "image": "image_models",
}

# Advanced chat settings, with the values used while "recommended" is on
_CHAT_ADVANCED: dict[str, Any] = {
    CONF_MAX_TOKENS: RECOMMENDED_MAX_TOKENS,
    CONF_TEMPERATURE: RECOMMENDED_TEMPERATURE,
    CONF_TOP_P: RECOMMENDED_TOP_P,
    CONF_DISABLE_THINKING: RECOMMENDED_DISABLE_THINKING,
    CONF_THINKING_TAGS: RECOMMENDED_THINKING_TAGS,
    CONF_VENICE_SYSTEM_PROMPT: RECOMMENDED_VENICE_SYSTEM_PROMPT,
}
_CONVERSATION_ADVANCED: dict[str, Any] = {
    **_CHAT_ADVANCED,
    CONF_STRIP_THINKING_RESPONSE: RECOMMENDED_STRIP_THINKING_RESPONSE,
    CONF_STREAM_RESPONSE: RECOMMENDED_STREAM_RESPONSE,
    CONF_MAX_TOOL_ITERATIONS: RECOMMENDED_MAX_TOOL_ITERATIONS,
    CONF_MAX_HISTORY_MESSAGES: RECOMMENDED_MAX_HISTORY_MESSAGES,
}
_AI_TASK_ADVANCED: dict[str, Any] = {
    **_CHAT_ADVANCED,
    CONF_STRUCTURE_PROMPT: DEFAULT_STRUCTURE_PROMPT,
}


def _number(
    low: float, high: float, step: float, mode: NumberSelectorMode
) -> NumberSelector:
    return NumberSelector(NumberSelectorConfig(min=low, max=high, step=step, mode=mode))


def _select(
    options: list[SelectOptionDict], *, custom_value: bool = False
) -> SelectSelector:
    return SelectSelector(
        SelectSelectorConfig(
            options=options, mode=SelectSelectorMode.DROPDOWN, custom_value=custom_value
        )
    )


def _model_label(model: Mapping[str, Any]) -> str:
    spec = model.get("model_spec")
    name = spec.get("name") if isinstance(spec, Mapping) else None
    details = ", ".join(
        part
        for part in (str(model["id"]), privacy_label(model), speech_price_label(model))
        if part
    )
    return f"{name} ({details})" if name else details


def _tts_model_label(model: TTSModel) -> str:
    count = len(model.languages)
    details = [
        model.privacy,
        f"{count} language{'s' if count != 1 else ''}",
        model.price,
    ]
    return f"{model.id} ({', '.join(d for d in details if d)})"


def _voice_label(voice: str) -> str:
    language = voice_language(voice)
    return f"{voice} ({language})" if language else voice


def _private_only(entry: ConfigEntry) -> bool:
    return bool(
        entry.options.get(CONF_PRIVATE_MODELS_ONLY, RECOMMENDED_PRIVATE_MODELS_ONLY)
    )


class VeniceAIConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Venice AI."""

    VERSION = 2
    MINOR_VERSION = 2

    async def _async_validate(self, api_key: str) -> str | None:
        """Return the error key for an API key, or None if it works."""
        try:
            await AsyncVeniceAIClient(
                api_key=api_key, http_client=get_async_client(self.hass)
            ).validate_api_key()
        except AuthenticationError:
            _LOGGER.warning("Venice AI authentication failed")
            return "invalid_auth"
        except VeniceAIError as err:
            _LOGGER.error("Cannot connect to Venice AI: %s", err)
            return "cannot_connect"
        except Exception:
            _LOGGER.exception("Unexpected exception while validating the API key")
            return "unknown"
        return None

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Ask for the API key and create the entry with one of each service."""
        errors: dict[str, str] = {}
        if user_input is not None:
            if error := await self._async_validate(user_input[CONF_API_KEY]):
                errors["base"] = error
            else:
                return self.async_create_entry(
                    title=DEFAULT_NAME,
                    data=user_input,
                    subentries=[
                        {
                            "subentry_type": subentry_type,
                            "data": data,
                            "title": title,
                            "unique_id": None,
                        }
                        for subentry_type, title, data in (
                            (
                                SUBENTRY_CONVERSATION,
                                DEFAULT_CONVERSATION_NAME,
                                RECOMMENDED_CONVERSATION_OPTIONS,
                            ),
                            (
                                SUBENTRY_AI_TASK,
                                DEFAULT_AI_TASK_NAME,
                                RECOMMENDED_AI_TASK_OPTIONS,
                            ),
                            (SUBENTRY_TTS, DEFAULT_TTS_NAME, RECOMMENDED_TTS_OPTIONS),
                            (SUBENTRY_STT, DEFAULT_STT_NAME, RECOMMENDED_STT_OPTIONS),
                        )
                    ],
                )

        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
        )

    async def async_step_reauth(
        self, entry_data: Mapping[str, Any]
    ) -> ConfigFlowResult:
        """Handle re-authentication when the API key becomes invalid."""
        return await self.async_step_reauth_confirm()

    async def async_step_reauth_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Ask for a new API key."""
        errors: dict[str, str] = {}
        reauth_entry = self._get_reauth_entry()
        if user_input is not None:
            if error := await self._async_validate(user_input[CONF_API_KEY]):
                errors["base"] = error
            else:
                return self.async_update_reload_and_abort(
                    reauth_entry, data_updates=user_input
                )

        return self.async_show_form(
            step_id="reauth_confirm",
            data_schema=STEP_USER_DATA_SCHEMA,
            errors=errors,
            description_placeholders={"name": reauth_entry.title},
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> VeniceAIOptionsFlow:
        """Get the options flow for this handler."""
        return VeniceAIOptionsFlow()

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return the services that can be added to the entry."""
        return {
            SUBENTRY_CONVERSATION: ChatSubentryFlow,
            SUBENTRY_AI_TASK: ChatSubentryFlow,
            SUBENTRY_TTS: TTSSubentryFlow,
            SUBENTRY_STT: STTSubentryFlow,
        }


class VeniceAIOptionsFlow(OptionsFlow):
    """Settings shared by all services of the entry."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the shared settings."""
        if user_input is not None:
            return self.async_create_entry(data=user_input)

        entry = self.config_entry
        runtime_data = getattr(entry, "runtime_data", None)
        data = runtime_data.coordinator.data if runtime_data else None
        image_models = [
            m
            for m in (data or {}).get("image_models", [])
            if m.get("id") and (not _private_only(entry) or is_private(m))
        ]
        schema = vol.Schema(
            {
                vol.Optional(CONF_PRIVATE_MODELS_ONLY): BooleanSelector(),
                vol.Optional(CONF_IMAGE_MODEL): _select(
                    [
                        SelectOptionDict(
                            label="Venice default", value=RECOMMENDED_IMAGE_MODEL
                        ),
                        *(
                            SelectOptionDict(label=_model_label(m), value=m["id"])
                            for m in image_models
                        ),
                    ],
                    custom_value=True,
                ),
                vol.Optional(CONF_REQUEST_TIMEOUT): _number(
                    10, 300, 5, NumberSelectorMode.SLIDER
                ),
            }
        )
        suggested = {
            CONF_PRIVATE_MODELS_ONLY: RECOMMENDED_PRIVATE_MODELS_ONLY,
            CONF_IMAGE_MODEL: RECOMMENDED_IMAGE_MODEL,
            CONF_REQUEST_TIMEOUT: RECOMMENDED_REQUEST_TIMEOUT,
            **entry.options,
        }
        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(schema, suggested),
        )


class _SubentryFlow(ConfigSubentryFlow):
    """Common steps of the subentry flows."""

    options: dict[str, Any]
    recommended: Mapping[str, Any] = {}

    @property
    def _is_new(self) -> bool:
        return self.source == "user"

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Add a subentry."""
        self.options = dict(self.recommended)
        return await self.async_step_init()

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Reconfigure a subentry."""
        self.options = dict(self._get_reconfigure_subentry().data)
        return await self.async_step_init()

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Show the first step."""
        raise NotImplementedError

    def _name_field(self, default: str) -> VolDictType:
        return {vol.Required(CONF_NAME, default=default): str} if self._is_new else {}

    async def _async_models(self, model_type: str) -> list[dict[str, Any]]:
        """List the models of a type, live, falling back to the last refresh."""
        entry = cast("VeniceAIConfigEntry", self._get_entry())
        try:
            models = await entry.runtime_data.client.models.list(model_type=model_type)
        except VeniceAIError as err:
            _LOGGER.warning("Failed to fetch %s models: %s", model_type, err)
            data = cast(
                Mapping[str, list[dict[str, Any]]],
                entry.runtime_data.coordinator.data or {},
            )
            models = list(data.get(_COORDINATOR_KEYS[model_type], []))
        return [
            m
            for m in models
            if isinstance(m, dict)
            and m.get("id")
            and (not _private_only(entry) or is_private(m))
        ]

    def _form(self, step_id: str, schema: VolDictType) -> SubentryFlowResult:
        return self.async_show_form(
            step_id=step_id,
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(schema), self.options
            ),
        )

    def _async_finish(self) -> SubentryFlowResult:
        if self._is_new:
            return self.async_create_entry(
                title=self.options.pop(CONF_NAME), data=self.options
            )
        return self.async_update_and_abort(
            self._get_entry(), self._get_reconfigure_subentry(), data=self.options
        )

    def _entry_loaded(self) -> bool:
        return self._get_entry().state is ConfigEntryState.LOADED


class ChatSubentryFlow(_SubentryFlow):
    """Conversation agent or AI Task."""

    @property
    def _advanced(self) -> dict[str, Any]:
        if self._subentry_type == SUBENTRY_AI_TASK:
            return _AI_TASK_ADVANCED
        return _CONVERSATION_ADVANCED

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Add a conversation agent or AI Task."""
        self.recommended = (
            RECOMMENDED_AI_TASK_OPTIONS
            if self._subentry_type == SUBENTRY_AI_TASK
            else RECOMMENDED_CONVERSATION_OPTIONS
        )
        return await super().async_step_user(user_input)

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Choose the model, and for conversations the prompt and APIs."""
        if not self._entry_loaded():
            return self.async_abort(reason="entry_not_loaded")

        if user_input is not None:
            if not user_input.get(CONF_LLM_HASS_API):
                user_input.pop(CONF_LLM_HASS_API, None)
                self.options.pop(CONF_LLM_HASS_API, None)
            if self._subentry_type == SUBENTRY_CONVERSATION:
                user_input.setdefault(CONF_PROMPT, "")
            self.options.update(user_input)
            if user_input[CONF_RECOMMENDED]:
                for key in self._advanced:
                    self.options.pop(key, None)
                return self._async_finish()
            return await self.async_step_advanced()

        conversation = self._subentry_type == SUBENTRY_CONVERSATION
        default_name = (
            DEFAULT_CONVERSATION_NAME if conversation else DEFAULT_AI_TASK_NAME
        )
        schema = self._name_field(default_name)
        if conversation:
            apis = {api.id: api.name for api in llm.async_get_apis(self.hass)}
            if suggested := self.options.get(CONF_LLM_HASS_API):
                if isinstance(suggested, str):
                    suggested = [suggested]
                self.options[CONF_LLM_HASS_API] = [a for a in suggested if a in apis]
            self.options.setdefault(CONF_PROMPT, DEFAULT_SYSTEM_PROMPT)
            schema[vol.Optional(CONF_PROMPT)] = TemplateSelector()
            schema[vol.Optional(CONF_LLM_HASS_API)] = SelectSelector(
                SelectSelectorConfig(
                    options=[
                        SelectOptionDict(label=name, value=api_id)
                        for api_id, name in apis.items()
                    ],
                    multiple=True,
                )
            )
        models = parse_models(await self._async_models("text"))
        self.options.setdefault(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
        schema[vol.Required(CONF_CHAT_MODEL)] = _select(
            [SelectOptionDict(label=m.label, value=m.id) for m in models.values()]
            or [
                SelectOptionDict(
                    label=RECOMMENDED_CHAT_MODEL, value=RECOMMENDED_CHAT_MODEL
                )
            ]
        )
        schema[
            vol.Required(
                CONF_RECOMMENDED, default=self.options.get(CONF_RECOMMENDED, True)
            )
        ] = bool
        return self._form("init", schema)

    async def async_step_advanced(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Tune sampling, reasoning and conversation handling."""
        if user_input is not None:
            for key in self._advanced:
                self.options.pop(key, None)
            self.options.update(user_input)
            return self._async_finish()

        for key, value in self._advanced.items():
            self.options.setdefault(key, value)
        schema: VolDictType = {
            vol.Optional(CONF_MAX_TOKENS): _number(1, 32768, 1, NumberSelectorMode.BOX),
            vol.Optional(CONF_TEMPERATURE): _number(
                0, 2, 0.05, NumberSelectorMode.SLIDER
            ),
            vol.Optional(CONF_TOP_P): _number(0, 1, 0.05, NumberSelectorMode.SLIDER),
            vol.Optional(CONF_DISABLE_THINKING): BooleanSelector(),
            vol.Optional(CONF_THINKING_TAGS): TextSelector(),
            vol.Optional(CONF_VENICE_SYSTEM_PROMPT): BooleanSelector(),
        }
        if self._subentry_type == SUBENTRY_AI_TASK:
            schema[vol.Optional(CONF_STRUCTURE_PROMPT)] = TextSelector(
                TextSelectorConfig(multiline=True)
            )
        else:
            schema.update(
                {
                    vol.Optional(CONF_STRIP_THINKING_RESPONSE): BooleanSelector(),
                    vol.Optional(CONF_STREAM_RESPONSE): BooleanSelector(),
                    vol.Optional(CONF_MAX_TOOL_ITERATIONS): _number(
                        1, 20, 1, NumberSelectorMode.SLIDER
                    ),
                    vol.Optional(CONF_MAX_HISTORY_MESSAGES): _number(
                        1, 100, 1, NumberSelectorMode.SLIDER
                    ),
                }
            )
        return self._form("advanced", schema)


class TTSSubentryFlow(_SubentryFlow):
    """Text-to-speech: the model first, then one of its voices."""

    recommended = RECOMMENDED_TTS_OPTIONS
    _models: dict[str, TTSModel]

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Choose the model."""
        if not self._entry_loaded():
            return self.async_abort(reason="entry_not_loaded")

        if user_input is not None:
            self.options.update(user_input)
            return await self.async_step_voice()

        self._models = parse_tts_models(await self._async_models("tts")) or {
            RECOMMENDED_TTS_MODEL: TTSModel(
                RECOMMENDED_TTS_MODEL, (RECOMMENDED_TTS_VOICE,)
            )
        }
        schema = self._name_field(DEFAULT_TTS_NAME)
        schema[vol.Required(CONF_TTS_MODEL)] = _select(
            [
                SelectOptionDict(label=_tts_model_label(model), value=model.id)
                for model in sorted(self._models.values(), key=lambda m: m.id)
            ]
        )
        return self._form("init", schema)

    async def async_step_voice(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Choose the voice and speed."""
        if user_input is not None:
            self.options.update(user_input)
            return self._async_finish()

        model_id = self.options[CONF_TTS_MODEL]
        model = self._models.get(model_id)
        voices = model.voices if model else (self.options.get(CONF_TTS_VOICE),)
        if self.options.get(CONF_TTS_VOICE) not in voices:
            self.options[CONF_TTS_VOICE] = voices[0]
        self.options.setdefault(CONF_TTS_SPEED, RECOMMENDED_TTS_SPEED)
        schema: VolDictType = {
            vol.Required(CONF_TTS_VOICE): _select(
                [
                    SelectOptionDict(label=_voice_label(voice), value=voice)
                    for voice in voices
                    if voice
                ]
            ),
            vol.Optional(CONF_TTS_SPEED): _number(
                0.25, 4, 0.05, NumberSelectorMode.SLIDER
            ),
        }
        return self.async_show_form(
            step_id="voice",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(schema), self.options
            ),
            description_placeholders={"model": model_id},
        )


class STTSubentryFlow(_SubentryFlow):
    """Speech-to-text."""

    recommended = RECOMMENDED_STT_OPTIONS

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Choose the model."""
        if not self._entry_loaded():
            return self.async_abort(reason="entry_not_loaded")

        if user_input is not None:
            self.options.update(user_input)
            return self._async_finish()

        schema = self._name_field(DEFAULT_STT_NAME)
        schema[vol.Required(CONF_STT_MODEL)] = _select(
            [
                SelectOptionDict(label=_model_label(m), value=m["id"])
                for m in await self._async_models("asr")
            ]
            or [
                SelectOptionDict(
                    label=RECOMMENDED_STT_MODEL, value=RECOMMENDED_STT_MODEL
                )
            ]
        )
        return self._form("init", schema)
