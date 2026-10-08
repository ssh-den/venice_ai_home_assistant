"""Constants for the Venice AI Conversation integration."""

from datetime import timedelta

DOMAIN = "venice_ai"

# Coordinator refresh interval — must be a timedelta for DataUpdateCoordinator
UPDATE_INTERVAL = timedelta(hours=12)

CONF_PROMPT = "prompt"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful smart home assistant. Be concise and friendly."
)
CONF_CHAT_MODEL = "chat_model"
# Cheapest TEE-backed model with function calling; without E2EE headers the
# request still runs inside the attested enclave.
RECOMMENDED_CHAT_MODEL = "e2ee-deepseek-v4-flash"
CONF_MAX_TOKENS = "max_tokens"
RECOMMENDED_MAX_TOKENS = 512
CONF_TOP_P = "top_p"
RECOMMENDED_TOP_P = 1.0
CONF_TEMPERATURE = "temperature"
RECOMMENDED_TEMPERATURE = 1.0

# Venice AI reasoning model options
CONF_STRIP_THINKING_RESPONSE = "strip_thinking_response"
RECOMMENDED_STRIP_THINKING_RESPONSE = True
CONF_DISABLE_THINKING = "disable_thinking"
# Disable thinking by default for automations to reduce latency, token usage, and cost,
# as complex reasoning is rarely needed for standard Home Assistant actions.
RECOMMENDED_DISABLE_THINKING = True

# MED-3: Opt-in streaming for conversation responses. When enabled, the
# conversation entity consumes the Venice AI streaming chat API via the
# VeniceConversationService and accumulates deltas (including tool calls).
CONF_STREAM_RESPONSE = "stream_response"
RECOMMENDED_STREAM_RESPONSE = True

# Venice AI TTS options
CONF_TTS_MODEL = "tts_model"
RECOMMENDED_TTS_MODEL = "tts-kokoro"
CONF_TTS_VOICE = "tts_voice"
RECOMMENDED_TTS_VOICE = "bm_daniel"
CONF_TTS_RESPONSE_FORMAT = "tts_response_format"
RECOMMENDED_TTS_RESPONSE_FORMAT = "mp3"
CONF_TTS_SPEED = "tts_speed"
RECOMMENDED_TTS_SPEED = 1.0

# Venice AI image options; "default" lets Venice pick its default model
CONF_IMAGE_MODEL = "image_model"
RECOMMENDED_IMAGE_MODEL = "default"

# Venice AI STT options
CONF_STT_MODEL = "stt_model"
RECOMMENDED_STT_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
CONF_STT_RESPONSE_FORMAT = "stt_response_format"
RECOMMENDED_STT_RESPONSE_FORMAT = "json"
CONF_STT_TIMESTAMPS = "stt_timestamps"
RECOMMENDED_STT_TIMESTAMPS = False

# Conversation tool iteration limit
CONF_MAX_TOOL_ITERATIONS = "max_tool_iterations"
RECOMMENDED_MAX_TOOL_ITERATIONS = 5
# Maximum number of conversation messages sent to the API per request
MAX_API_MESSAGES = 50


# Maximum audio buffer size for STT to prevent memory spikes (10 MB).
# Venice AI does not support chunked/streaming STT uploads; the entire audio
# payload must be buffered before submission. Recordings exceeding this limit
# are rejected early with an ERROR result rather than causing an OOM spike.
MAX_STT_BUFFER_SIZE = 10 * 1024 * 1024

# Timeout in seconds for every Venice AI request; for streamed responses it
# bounds the wait between chunks.
CONF_REQUEST_TIMEOUT = "request_timeout"
RECOMMENDED_REQUEST_TIMEOUT = 120.0

# Retries for transient API failures, handled by the OpenAI SDK
MAX_RETRIES = 3
