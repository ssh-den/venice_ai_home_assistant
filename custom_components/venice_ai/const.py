"""Constants for the Venice AI Conversation integration."""

from datetime import timedelta
from typing import Any

from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.helpers import llm

DOMAIN = "venice_ai"
DEFAULT_NAME = "Venice AI"

# Coordinator refresh interval — must be a timedelta for DataUpdateCoordinator
UPDATE_INTERVAL = timedelta(hours=12)

# Subentry types, one entity each
SUBENTRY_CONVERSATION = "conversation"
SUBENTRY_AI_TASK = "ai_task_data"
SUBENTRY_TTS = "tts"
SUBENTRY_STT = "stt"
DEFAULT_CONVERSATION_NAME = "Venice AI Conversation"
DEFAULT_AI_TASK_NAME = "Venice AI Task"
DEFAULT_TTS_NAME = "Venice AI TTS"
DEFAULT_STT_NAME = "Venice AI STT"

# Use the recommended values of the advanced chat settings
CONF_RECOMMENDED = "recommended"

CONF_PROMPT = "prompt"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful smart home assistant. Be concise and friendly."
)
CONF_CHAT_MODEL = "chat_model"
# Cheapest TEE-backed model with function calling
RECOMMENDED_CHAT_MODEL = "e2ee-deepseek-v4-flash"
CONF_MAX_TOKENS = "max_tokens"
RECOMMENDED_MAX_TOKENS = 512
CONF_TOP_P = "top_p"
RECOMMENDED_TOP_P = 1.0
CONF_TEMPERATURE = "temperature"
RECOMMENDED_TEMPERATURE = 1.0

# Venice adds its own system prompt to every request unless this is off
CONF_VENICE_SYSTEM_PROMPT = "include_venice_system_prompt"
RECOMMENDED_VENICE_SYSTEM_PROMPT = False

# Venice AI reasoning model options
CONF_STRIP_THINKING_RESPONSE = "strip_thinking_response"
RECOMMENDED_STRIP_THINKING_RESPONSE = True
CONF_DISABLE_THINKING = "disable_thinking"
# Disable thinking by default for automations to reduce latency, token usage, and cost,
# as complex reasoning is rarely needed for standard Home Assistant actions.
RECOMMENDED_DISABLE_THINKING = True
# Names of the tags that wrap reasoning in answers, separated by commas
CONF_THINKING_TAGS = "thinking_tags"
RECOMMENDED_THINKING_TAGS = "think"

# Stream conversation answers to Home Assistant as they are generated
CONF_STREAM_RESPONSE = "stream_response"
RECOMMENDED_STREAM_RESPONSE = True

# Conversation tool iteration limit
CONF_MAX_TOOL_ITERATIONS = "max_tool_iterations"
RECOMMENDED_MAX_TOOL_ITERATIONS = 5
# Earlier conversation messages sent with a request; the current turn is always sent
CONF_MAX_HISTORY_MESSAGES = "max_history_messages"
RECOMMENDED_MAX_HISTORY_MESSAGES = 50

# Instruction sent with the JSON schema to models without structured output
CONF_STRUCTURE_PROMPT = "structure_prompt"
DEFAULT_STRUCTURE_PROMPT = (
    "Respond only with a JSON object matching this JSON schema, "
    "without any surrounding text:"
)

# Venice AI TTS options
CONF_TTS_MODEL = "tts_model"
RECOMMENDED_TTS_MODEL = "tts-kokoro"
CONF_TTS_VOICE = "tts_voice"
RECOMMENDED_TTS_VOICE = "bm_daniel"
CONF_TTS_SPEED = "tts_speed"
RECOMMENDED_TTS_SPEED = 1.0

# Venice AI STT options
CONF_STT_MODEL = "stt_model"
RECOMMENDED_STT_MODEL = "nvidia/parakeet-tdt-0.6b-v3"

# Venice AI image options; "default" lets Venice pick its default model
CONF_IMAGE_MODEL = "image_model"
RECOMMENDED_IMAGE_MODEL = "default"

# Offer only models that Venice runs privately or in a TEE
CONF_PRIVATE_MODELS_ONLY = "private_models_only"
RECOMMENDED_PRIVATE_MODELS_ONLY = False

# Timeout in seconds for every Venice AI request; for streamed responses it
# bounds the wait between chunks.
CONF_REQUEST_TIMEOUT = "request_timeout"
RECOMMENDED_REQUEST_TIMEOUT = 120.0

RECOMMENDED_CONVERSATION_OPTIONS: dict[str, Any] = {
    CONF_RECOMMENDED: True,
    CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
    CONF_PROMPT: DEFAULT_SYSTEM_PROMPT,
    CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL,
}
RECOMMENDED_AI_TASK_OPTIONS: dict[str, Any] = {
    CONF_RECOMMENDED: True,
    CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL,
}
RECOMMENDED_TTS_OPTIONS: dict[str, Any] = {
    CONF_TTS_MODEL: RECOMMENDED_TTS_MODEL,
    CONF_TTS_VOICE: RECOMMENDED_TTS_VOICE,
    CONF_TTS_SPEED: RECOMMENDED_TTS_SPEED,
}
RECOMMENDED_STT_OPTIONS: dict[str, Any] = {CONF_STT_MODEL: RECOMMENDED_STT_MODEL}

# Model option and its default of each subentry type
SUBENTRY_MODELS: dict[str, tuple[str, str]] = {
    SUBENTRY_CONVERSATION: (CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
    SUBENTRY_AI_TASK: (CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
    SUBENTRY_TTS: (CONF_TTS_MODEL, RECOMMENDED_TTS_MODEL),
    SUBENTRY_STT: (CONF_STT_MODEL, RECOMMENDED_STT_MODEL),
}

# Maximum audio buffer size for STT to prevent memory spikes (10 MB).
# Venice AI does not support chunked/streaming STT uploads; the entire audio
# payload must be buffered before submission. Recordings exceeding this limit
# are rejected early with an ERROR result rather than causing an OOM spike.
MAX_STT_BUFFER_SIZE = 10 * 1024 * 1024

# Retries for transient API failures, handled by the OpenAI SDK
MAX_RETRIES = 3
