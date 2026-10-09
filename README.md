# Venice AI

> [!IMPORTANT]
> This is a fork of [grasponcrypto/venice_ai](https://github.com/grasponcrypto/venice_ai).
> Version 2.0 differs noticeably from upstream: the client is built on the OpenAI
> Python SDK, the conversation agent uses the Home Assistant chat log with native
> streaming, the default model changed and the minimum Home Assistant version is
> 2025.9. See the [changelog](CHANGELOG.md) for the full list.

Home Assistant integration for [Venice AI](https://venice.ai): a conversation agent,
AI Task entity, text-to-speech, speech-to-text and image generation.

## Features

- **Conversation agent** that can control Home Assistant through the Assist API,
  with answers streamed to Assist as they are generated.
- **AI Task entity** for `ai_task.generate_data`, including structured output.
  Models that support JSON schemas get the schema natively; other models are
  instructed through the prompt.
- **Text-to-speech** that starts speaking after the first sentence. Voices,
  audio formats and languages come from the selected Venice model.
- **Speech-to-text** for Assist pipelines; the pipeline language is sent to Venice.
- **Image generation** through the `venice_ai.generate_image` action.
- **Diagnostic sensors** for request, error and token counters, updated after
  every request.
- **Repair issues** when the API key is rejected, Venice is unreachable or rate
  limited, or a configured model is no longer offered.
- **Model capabilities from Venice**: the model selector shows privacy (E2EE, TEE,
  private or anonymized), tool support and price per million tokens. Tools are
  not sent to models without function calling.

## Requirements

- Home Assistant 2025.9 or newer.
- A Venice AI API key.

## Installation

### HACS

1. Open HACS, then the three-dot menu → **Custom repositories**.
2. Add `https://github.com/ssh-den/venice_ai_home_assistant` with category
   **Integration**.
3. Search for **Venice AI**, download it and restart Home Assistant.

### Manual

Copy `custom_components/venice_ai` into the `custom_components` directory of your
Home Assistant configuration and restart Home Assistant.

## Configuration

1. Go to **Settings → Devices & services → Add integration** and pick **Venice AI**.
2. Enter your API key. It is checked against Venice before the entry is created.
   If the key is later revoked, Home Assistant asks for a new one.
3. Open **Configure** on the integration to adjust the options.

| Option | Default | Description |
| --- | --- | --- |
| AI model | `e2ee-deepseek-v4-flash` | Chat model for the conversation agent and AI Task. |
| System prompt | built-in | Instructions sent with every conversation. |
| Control Home Assistant | off, the form suggests Assist | LLM APIs the agent may use. |
| Temperature / Top P / Max tokens | 1.0 / 1.0 / 512 | Sampling and answer length. |
| Max tool iterations | 5 | Model calls allowed per turn while using tools. |
| Disable thinking | on | Ask reasoning models to skip reasoning. |
| Strip thinking response | on | Keep `<think>` blocks out of the spoken answer. |
| Stream responses | on | Stream answers to Assist while they are generated. |
| Request timeout | 120 s | Timeout for every Venice AI request. |
| TTS voice / speed | `tts-kokoro → bm_daniel`, 1.0 | Text-to-speech model, voice and speed. |
| STT model | `nvidia/parakeet-tdt-0.6b-v3` | Speech-to-text model. |
| Image model | Venice default | Model used by `venice_ai.generate_image`. |

Changing options reloads the entry automatically. The model lists are fetched
live when the options open, and refreshed in the background every 12 hours.

### Choosing a model

The default `e2ee-deepseek-v4-flash` is an inexpensive model that runs in a
trusted execution environment and supports function calling. Pick a model tagged
`tools` if the agent should control your home. Reasoning output from the model is
kept as thinking content and is not spoken; `<think>` blocks in the answer
are removed while **Strip thinking response** is on.

End-to-end encrypted (E2EE) requests are not implemented yet: E2EE-capable models
are currently used like regular private TEE models.

### Voice assistant

Speech runs in the Venice cloud; nothing is downloaded to Home Assistant.

Venice does not report which languages a speech model supports, so the
integration keeps them in one table,
[`languages.py`](custom_components/venice_ai/languages.py), compiled from each
vendor's documentation. A voice pipeline only offers engines that support its
language, and the language is passed to Venice in the form each model expects.

| Text-to-speech | Languages |
| --- | --- |
| `tts-kokoro` | English, Spanish, French, Hindi, Italian, Japanese, Portuguese, Chinese; voices follow the pipeline language |
| `tts-qwen3-0-6b`, `tts-qwen3-1-7b` | 10 languages, including Russian |
| `tts-xai-v1` | 16 languages, including Russian |
| `tts-inworld-1-5-max` | 15 languages, including Russian |
| `tts-elevenlabs-turbo-v2-5` | 32 languages, including Russian and Ukrainian |
| `tts-minimax-speech-02-hd` | 37 languages, including Russian and Ukrainian |
| `tts-gemini-3-1-flash` | 78 languages |
| `tts-gradium-v1` | English, French, German, Spanish, Portuguese |
| `tts-orpheus`, `tts-chatterbox-hd` | English |

| Speech-to-text | Languages |
| --- | --- |
| `nvidia/parakeet-tdt-0.6b-v3` | 25 European languages, including Russian and Ukrainian |
| `openai/whisper-large-v3`, `fal-ai/wizper` | About 100 languages |
| `elevenlabs/scribe-v2` | About 100 languages |
| `stt-xai-v1` | 38 languages |

Models that Venice adds later are offered for every language until they are
added to the table; the log and the diagnostics list them.

Audio is requested as MP3, or WAV for models without MP3; Home Assistant
converts it for each speaker. Long answers are split at sentence boundaries,
so the first sentence plays while the rest is still being generated, and the
Venice limit of 4096 characters per request is never hit.

## Actions

### `venice_ai.generate_image`

| Field | Required | Description |
| --- | --- | --- |
| `config_entry` | yes | Venice AI entry to use. |
| `prompt` | yes | Image description. |
| `model` | no | Image model, defaults to the configured one. Must be offered by Venice. |
| `size` | no | `auto`, `256x256` … `1792x1024`, default `1024x1024`. |
| `quality` | no | `auto`, `low`, `medium`, `high`, `standard` or `hd`. |
| `style` | no | `vivid` or `natural`. |

Returns the image `url`, plus `revised_prompt` when Venice provides one.

### `venice_ai.ai_task`

| Field | Required | Description |
| --- | --- | --- |
| `config_entry` | yes | Venice AI entry to use. |
| `task` | yes | Instructions for the model. |
| `structure` | no | Description of the JSON structure to return. |

Returns `conversation_id` and `data`; `data` is parsed JSON when `structure` is
given. Prefer the built-in `ai_task.generate_data` action for new automations.

## Entities

- `conversation.venice_ai` — conversation agent for Assist pipelines.
- `ai_task.venice_ai_ai_task` — AI Task entity.
- `tts.venice_ai_tts` — text-to-speech.
- `stt.venice_ai` — speech-to-text.
- Diagnostic sensors: request count, error count, total, prompt and completion
  tokens, last error.

## Troubleshooting

| Symptom | Likely cause | Resolution |
| --- | --- | --- |
| Reauthentication requested | Invalid or revoked API key | Enter a new key in the reauthentication dialog. |
| `Rate limit exceeded` | Venice is throttling the account | Wait, or lower the request rate. |
| Entry stays in *setup retry* | Venice unreachable | Allow outbound HTTPS to `api.venice.ai`. |
| Agent does not control devices | Model without function calling, or no LLM API selected | Pick a model tagged `tools` and enable **Control Home Assistant**. |
| Answer cut off | Max tokens too low | Increase **Max tokens**. |
| Slow reasoning model | Model emits reasoning before answering | Enable **Disable thinking**. |

**Download diagnostics** on the integration page gives a redacted snapshot of
the options, coordinator state and model counts. Enable debug logging to
investigate further:

```yaml
logger:
  logs:
    custom_components.venice_ai: debug
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and checks.

## License

MIT, see [LICENSE](LICENSE).
