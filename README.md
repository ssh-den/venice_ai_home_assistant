# Venice AI

> [!IMPORTANT]
> This is a fork of [grasponcrypto/venice_ai](https://github.com/grasponcrypto/venice_ai).
> It differs noticeably from upstream: the client is built on the OpenAI Python
> SDK, the conversation agent uses the Home Assistant chat log with native
> streaming, every feature is a separate service of the entry, the default model
> changed and the minimum Home Assistant version is 2025.9. See the [changelog](https://github.com/ssh-den/venice_ai_home_assistant/blob/main/CHANGELOG.md) for the full list.

Home Assistant integration for [Venice AI](https://venice.ai): a conversation agent,
AI Task entity, text-to-speech, speech-to-text and image generation.

## Features

- **Conversation agent** that can control Home Assistant through the Assist API,
  with answers streamed to Assist as they are generated.
- **AI Task entity** for `ai_task.generate_data`, including structured output.
  Models that support JSON schemas get the schema natively; other models get it
  with configurable instructions.
- **Text-to-speech** that starts speaking after the first sentence. Voices,
  audio formats and languages come from the selected Venice model.
- **Speech-to-text** for Assist pipelines; the pipeline language is sent to Venice.
- **Image generation** through the `venice_ai.generate_image` action.
- **Diagnostic sensors** for the requests, errors and tokens of each service,
  updated after every request.
- **Repair issues** when the API key is rejected, Venice is unreachable or rate
  limited, or a configured model is no longer offered.
- **Services like the built-in integrations**: the conversation agent, AI Task,
  text-to-speech and speech-to-text are separate services of one entry, each
  with its own model and device. Add more, for example a second agent with
  another model or prompt.
- **Model capabilities from Venice**: the model selectors show privacy (TEE,
  private or anonymized), tool support and price per million tokens. Tools are
  not sent to models without function calling.
- **No hidden prompts**: the instructions sent to the model are the ones in the
  settings. The Venice system prompt is off unless you turn it on.

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
2. Enter your API key. It is checked against Venice before the entry is created,
   together with a conversation agent, an AI Task, text-to-speech and
   speech-to-text. If the key is later revoked, Home Assistant asks for a new one.
3. Use **Reconfigure** on a service to change its model and settings, or **Add**
   to create another one. **Configure** on the entry holds the settings shared
   by all services: private models only, the image model and the request timeout.

[docs/configuration.md](https://github.com/ssh-den/venice_ai_home_assistant/blob/main/docs/configuration.md) lists every setting and its
default.

### Voice assistant

Speech runs in the Venice cloud. See [docs/voice.md](https://github.com/ssh-den/venice_ai_home_assistant/blob/main/docs/voice.md) for the
languages of each speech model and how audio is produced.

### Privacy

Conversations, voice commands and the states of exposed entities are processed
by Venice AI. See [docs/privacy.md](https://github.com/ssh-den/venice_ai_home_assistant/blob/main/docs/privacy.md) for what is sent,
the privacy levels of Venice models and links to the Venice privacy policy and
terms of service.

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

## Entities

Each service is a device named after the service, for example
`conversation.venice_ai_conversation`, `ai_task.venice_ai_task`,
`tts.venice_ai_tts` and `stt.venice_ai_stt`. Its diagnostic sensors count the
requests and errors of that service and show its last error; conversation agents
and AI Tasks also count prompt, completion and total tokens. Image generation
and the model list refresh are not counted.

## Troubleshooting

| Symptom | Likely cause | Resolution |
| --- | --- | --- |
| Reauthentication requested | Invalid or revoked API key | Enter a new key in the reauthentication dialog. |
| `Rate limit exceeded` | Venice is throttling the account | Wait, or lower the request rate. |
| Entry stays in *setup retry* | Venice unreachable | Allow outbound HTTPS to `api.venice.ai`. |
| Agent does not control devices | Model without function calling, or no LLM API selected | Reconfigure the conversation agent: pick a model tagged `tools` and enable **Control Home Assistant**. |
| Answer cut off | Max tokens too low | Increase **Max tokens** in the advanced settings. |
| Slow reasoning model | Model emits reasoning before answering | Enable **Disable thinking** in the advanced settings. |
| Reasoning is spoken | The model wraps reasoning in another tag | Add its name to **Thinking tags**. |
| Venice AI TTS or STT missing from a voice pipeline | The selected speech model does not support the pipeline language | Pick a model that does, see [docs/voice.md](https://github.com/ssh-den/venice_ai_home_assistant/blob/main/docs/voice.md). |

**Download diagnostics** on the integration page gives a redacted snapshot of
the settings, coordinator state, model counts and speech models missing from
the language table. Enable debug logging to
investigate further:

```yaml
logger:
  logs:
    custom_components.venice_ai: debug
```

## Development

See [CONTRIBUTING.md](https://github.com/ssh-den/venice_ai_home_assistant/blob/main/CONTRIBUTING.md) for setup and checks.

## License

MIT, see [LICENSE](https://github.com/ssh-den/venice_ai_home_assistant/blob/main/LICENSE).
