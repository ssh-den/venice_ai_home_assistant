# Venice AI

> [!IMPORTANT]
> This is a fork of [grasponcrypto/venice_ai](https://github.com/grasponcrypto/venice_ai).
> Version 2.0 differs noticeably from upstream: the client is built on the OpenAI
> Python SDK, the conversation agent uses the Home Assistant chat log with native
> streaming, the default model changed, the `todo` platform was removed and the
> minimum Home Assistant version is 2025.9. See the [changelog](CHANGELOG.md) for
> the full list.

Home Assistant integration for [Venice AI](https://venice.ai): a conversation agent,
AI Task entity, text-to-speech, speech-to-text and image generation.

## Features

- **Conversation agent** that can control Home Assistant through the Assist API,
  with answers streamed to Assist as they are generated.
- **AI Task entity** for `ai_task.generate_data`, including structured output.
  Models that support JSON schemas get it natively; other models are instructed
  through the prompt.
- **Text-to-speech** with streaming audio and per-model voice selection.
- **Speech-to-text** for Assist pipelines.
- **Image generation** through the `venice_ai.generate_image` action.
- **Diagnostic sensors** for request, error and token counters.
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
3. Open **Configure** on the integration to adjust the options.

| Option | Default | Description |
| --- | --- | --- |
| AI model | `e2ee-deepseek-v4-flash` | Chat model for the conversation agent and AI Task. |
| System prompt | built-in | Instructions sent with every conversation. |
| Control Home Assistant | off | LLM APIs (for example Assist) the agent may use. |
| Temperature / Top P / Max tokens | 1.0 / 1.0 / 512 | Sampling and answer length. |
| Max tool iterations | 5 | Model calls allowed per turn while using tools. |
| Disable thinking | on | Ask reasoning models to skip reasoning. |
| Strip thinking response | on | Keep `<think>` blocks out of the spoken answer. |
| Stream responses | on | Stream answers to Assist while they are generated. |
| Request timeout | 120 s | Timeout for every Venice AI request. |
| TTS voice / format / speed | `tts-kokoro → bm_daniel`, mp3, 1.0 | Text-to-speech settings. |
| STT model / format / timestamps | `nvidia/parakeet-tdt-0.6b-v3`, json, off | Speech-to-text settings. |
| Image model | Venice default | Model used by `venice_ai.generate_image`. |

Changing options reloads the entry automatically.

### Choosing a model

The default `e2ee-deepseek-v4-flash` is an inexpensive model that runs in a
trusted execution environment and supports function calling. Pick a model tagged
`tools` if the agent should control your home. Reasoning output from the model is
kept as thinking content and is not spoken.

End-to-end encrypted (E2EE) requests are not implemented yet: E2EE-capable models
are currently used like regular private TEE models.

## Actions

### `venice_ai.generate_image`

| Field | Required | Description |
| --- | --- | --- |
| `config_entry` | yes | Venice AI entry to use. |
| `prompt` | yes | Image description. |
| `model` | no | Image model, defaults to the configured one. |
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

Returns `conversation_id` and `data`. Prefer the built-in `ai_task.generate_data`
action for new automations.

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

Enable debug logging to investigate problems:

```yaml
logger:
  logs:
    custom_components.venice_ai: debug
```

## Development

```bash
uv sync
```

```bash
.venv/bin/pytest && .venv/bin/ruff check . && .venv/bin/black --check . && .venv/bin/mypy && .venv/bin/pyright && .venv/bin/pylint custom_components tests
```

## License

MIT, see [LICENSE](LICENSE).
